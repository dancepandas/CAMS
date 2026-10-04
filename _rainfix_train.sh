#!/usr/bin/env bash
# 公平串行重训：同一张 GPU 先跑短数据 S.14，再跑延长数据 S.18。
set -euo pipefail
cd "$(dirname "$0")"

PY="C:/Users/DELL/.conda/envs/HydroModel/python.exe"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

S14_DIR="runs/site_model_step14_rainfix"
S18_DIR="runs/site_model_step18_rainfix"
for d in "$S14_DIR" "$S18_DIR"; do
    if [[ -e "$d" ]]; then
        echo "拒绝覆盖已有目录：$d" >&2
        exit 1
    fi
done
mkdir -p logs

train_one() {
    local cfg="$1" out="$2" log="$3"
    "$PY" -u scripts/train.py \
      --config "$cfg" \
      --set "paths.out_dir=$out" \
      --set model.output_mode=level \
      --set model.arch=moe --set model.transform=gstd \
      --set model.horizon=1 --set model.rollout=24 --set model.stride=3 \
      --set model.epochs=200 --set model.patience=12 \
      --set model.loss=quantile --set 'model.quantiles=[0.1,0.5,0.9]' \
      --set 'model.split_dates=["2023-01-01","2024-01-01"]' \
      --set model.train_fit=true --set model.masked_pool=false \
      --set model.delta_cap=0.0 \
      > "$log" 2>&1
}

train_one configs/pipeline.yaml "$S14_DIR" logs/train_step14_rainfix.log
train_one configs/pipeline_rain1990.yaml "$S18_DIR" logs/train_step18_rainfix.log
