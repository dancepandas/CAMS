#!/usr/bin/env bash
# ============================================================================
# CAMS 定稿流水线：一条命令复现"综合指标最好的方案"（S.20 非对称罚版）。
#
# 定稿依据（experiments/step18_compare/README.md §9 与 step18_val_compare/）：
# 两年评估（2023 正常年 / 2024 海伦超极端年，权重都没见过）里，非对称罚版
# 达标线最全——三小站 24h 偏差唯一 3/3 全进 ±10%、中位 NSE 两年 0.806/0.929、
# 胜持续性 15/15，洪峰偏差明显好于 S.14 基线。底模 = S.18 修正版配方
#（教师强迫训练），定稿 = 底模 + 整条 24 步轨迹 BPTT 非对称偏差罚微调。
#
# 执行顺序（全部串行，任一步失败即停，不污染后续）：
#   0. 单元测试（不碰大文件，约 10 秒）
#   1. 数据合同检查（降雨/汇水区/面雨量/站序/切分边界，不符即拒训）
#   2. 训练底模（教师强迫，epochs 200 + 早停耐心 12）→ runs/site_model_final_base
#   3. BPTT 非对称罚微调（--init-from 底模，epochs 30 + 耐心 8）→ runs/site_model_final
#   4. 两版 × 两年 dense 推理（2023 val / 2024 test，逐小时起报滚动 24h）
#   5. 对比数字 + 图件 → experiments/final_eval/
#
# 安全约定（与全项目一致）：
#   - 训练目标严格 <2023-01-01；2023 只早停；2024 权重冻结后才评估
#   - runs/site_model_final* 已存在即拒绝覆盖（想重跑先手动改名旧目录）
#   - 大产物走 staging→校验→原子替换；本脚本不下载降雨、不重拉流量，
#     数据链（_step18_chain.sh）需已跑通
#
# 用法:  bash scripts/pipeline_final.sh            # 全流程
#        FINAL_SKIP_TRAIN=1 bash scripts/pipeline_final.sh   # 跳过 2/3 训练
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="C:/Users/DELL/.conda/envs/HydroModel/python.exe"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/pipeline_rain1990.yaml"
BASE_DIR="runs/site_model_final_base"
FINAL_DIR="runs/site_model_final"
OUT_EVAL="experiments/final_eval"
mkdir -p logs "$OUT_EVAL"

note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { note "错误：$*" >&2; exit 1; }

# ---- 0. 单元测试 -----------------------------------------------------------
note "阶段 0：单元测试"
"$PY" -m unittest discover -s tests -q || fail "单元测试未通过，中止"

# ---- 1. 数据合同检查 -------------------------------------------------------
note "阶段 1：数据合同检查（降雨/汇水区/面雨量/站序/切分边界）"
"$PY" -u - <<'PYEOF' || fail "数据合同检查未通过，中止"
import sys
from pathlib import Path
import yaml
root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from data_contract import validate_data_contract
cfg = yaml.safe_load(open(root / "configs" / "pipeline_rain1990.yaml", encoding="utf-8"))
validate_data_contract(cfg, root)
print("数据合同 OK")
PYEOF

# ---- 2/3. 训练 --------------------------------------------------------------
BASE_RECIPE=(
  --config "$CONFIG"
  --set model.output_mode=level
  --set model.arch=moe --set model.transform=gstd
  --set model.horizon=1 --set model.rollout=24 --set model.stride=3
  --set model.loss=quantile --set 'model.quantiles=[0.1,0.5,0.9]'
  --set 'model.split_dates=["2023-01-01","2024-01-01"]'
  --set model.train_fit=true --set model.masked_pool=false
  --set model.delta_cap=0.0
  --set model.seed=42
)

if [[ "${FINAL_SKIP_TRAIN:-0}" != "1" ]]; then
    [[ -e "$BASE_DIR" ]] && fail "$BASE_DIR 已存在，拒绝覆盖（先手动改名旧目录）"
    [[ -e "$FINAL_DIR" ]] && fail "$FINAL_DIR 已存在，拒绝覆盖（先手动改名旧目录）"

    note "阶段 2：训练底模（教师强迫）→ $BASE_DIR"
    "$PY" -u scripts/train.py "${BASE_RECIPE[@]}" \
      --set "paths.out_dir=$BASE_DIR" \
      --set model.epochs=200 --set model.patience=12 \
      > logs/train_final_base.log 2>&1 || fail "底模训练失败（见 logs/train_final_base.log）"

    note "阶段 3：BPTT 非对称罚微调 → $FINAL_DIR"
    # 关键开关（S.20 三级实验的结论全部凝结在这里）：
    #   unroll=24 + bptt=true  —— 整条 24 步闭环轨迹不断梯度，梯度检查点保显存
    #   bias_pen_mode=asym, under=1.0 / over=0.3 —— under 侧物理上拆掉"往下压"
    #     捷径（压低预测立刻受罚），over 侧轻轻拉住小站正向漂移
    #   adv_weight=1.0 —— 逐站"对持续性基准的 NSE 提升量"，直接优化部署成绩
    "$PY" -u scripts/train.py "${BASE_RECIPE[@]}" \
      --set "paths.out_dir=$FINAL_DIR" \
      --set model.unroll=24 --set model.bptt=true \
      --set model.bias_pen=1.0 --set model.bias_pen_mode=asym \
      --set model.bias_pen_over=0.3 --set model.adv_weight=1.0 \
      --set model.batch_size=128 --set model.lr=0.001 \
      --set model.epochs=30 --set model.patience=8 \
      --init-from "$BASE_DIR/best.pt" \
      > logs/train_final_bptt.log 2>&1 || fail "BPTT 微调失败（见 logs/train_final_bptt.log）"
else
    note "阶段 2/3：FINAL_SKIP_TRAIN=1，跳过训练（使用已有 $FINAL_DIR）"
    [[ -e "$FINAL_DIR/best.pt" ]] || fail "跳过训练但 $FINAL_DIR/best.pt 不存在"
fi

# ---- 4. dense 推理（两版 × 两年） -------------------------------------------
dense() {  # $1=模型目录 $2=split
    note "阶段 4：dense 推理 $1 --split $2"
    "$PY" -u scripts/rollout_dense.py --config "$CONFIG" \
      --run-dir "$1" --split "$2" --roll 24 --stride 1 \
      > "logs/dense_$(basename "$1")_$2.log" 2>&1 \
      || fail "dense 推理失败：$1 $2"
}
for d in "$BASE_DIR" "$FINAL_DIR"; do
    dense "$d" val
    dense "$d" test
done

# ---- 5. 对比数字 + 图件 -----------------------------------------------------
note "阶段 5：对比与出图 → $OUT_EVAL"
PAIRS="S.18 底模=$BASE_DIR,S.20 非对称罚定稿=$FINAL_DIR"
for split in val test; do
    PYTHONIOENCODING=utf-8 "$PY" -u scripts/compare_step18.py \
      --a "$BASE_DIR/predictions_dense_${split}.npz" --label-a "S.18 底模" \
      --b "$FINAL_DIR/predictions_dense_${split}.npz" --label-b "S.20 非对称罚定稿" \
      > "$OUT_EVAL/final_vs_base_${split}.txt" 2>&1 || fail "对比失败 $split"
    PYTHONIOENCODING=utf-8 "$PY" -u scripts/plot_unroll.py --split "$split" \
      --models "$PAIRS" --out "$OUT_EVAL" || fail "出图失败 $split"
done

note "流水线完成。产物：$FINAL_DIR（定稿权重）、$OUT_EVAL（两年对比数字+图件）"
