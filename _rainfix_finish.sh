#!/usr/bin/env bash
# 修正版模型的无人值守收尾脚本。
#
# 前提：_step18_chain.sh 已在后台等待 AORC，并会完成两套面雨量重建与公共时段
# 核验。本脚本只等待它写出成功标志，随后严格串行执行：S.14 训练 → S.18 训练 →
# 两套 2024 年逐小时起报的 24h 滚动推理 → 按物理起报时刻公平比较。
#
# 不用“日志多久没更新”判断失败：下载器和数据链自身会失败即退出，而等待期间
# 日志本来可能长时间静默。错误地用静默时间作判断会提前杀死一个仍健康的流程。
set -euo pipefail
cd "$(dirname "$0")"

PY="C:/Users/DELL/.conda/envs/HydroModel/python.exe"
export PYTHONIOENCODING=utf-8
CHAIN_LOG="logs/step18_chain_rainfix.log"
MASTER_LOG="logs/rainfix_finish.log"
mkdir -p logs experiments/step18_compare

note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

{
    note "等待降雨数据链完成"
    while ! grep -q '降雨数据链全部结束，可以启动训练' "$CHAIN_LOG" 2>/dev/null; do
        sleep 60
    done

    note "数据链通过，开始 S.14 → S.18 串行训练"
    bash _rainfix_train.sh

    note "两套训练完成，开始 S.14 的 2024 密集滚动预报"
    "$PY" -u scripts/rollout_dense.py \
      --config configs/pipeline.yaml \
      --run-dir runs/site_model_step14_rainfix \
      --split test --roll 24 --stride 1 \
      --output runs/site_model_step14_rainfix/predictions_dense_test.npz \
      2>&1 | tee logs/rollout_step14_rainfix.log

    note "S.14 推理完成，开始 S.18 的 2024 密集滚动预报"
    "$PY" -u scripts/rollout_dense.py \
      --config configs/pipeline_rain1990.yaml \
      --run-dir runs/site_model_step18_rainfix \
      --split test --roll 24 --stride 1 \
      --output runs/site_model_step18_rainfix/predictions_dense_test.npz \
      2>&1 | tee logs/rollout_step18_rainfix.log

    note "两套推理完成，生成公平对比结果"
    "$PY" -u scripts/compare_step18.py \
      --a runs/site_model_step14_rainfix/predictions_dense_test.npz \
      --b runs/site_model_step18_rainfix/predictions_dense_test.npz \
      --label-a 'S.14 修正版' --label-b 'S.18 修正版' \
      2>&1 | tee experiments/step18_compare/rainfix_results.txt

    note "全部完成"
} 2>&1 | tee "$MASTER_LOG"
