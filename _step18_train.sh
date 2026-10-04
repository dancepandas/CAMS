#!/usr/bin/env bash
# S.18 训练：S.14 的配方原样照搬，只把训练数据从 2015-06 起换成 1990-01 起。
#
# 与 S.14 逐项对应的开关见 experiments/step18_extended_data.md。
# 唯一对不齐的一项是轮数上限：S.14 当时覆盖过 model.epochs，但运行目录没留配置档，
# 配置文件里一直是 40 而日志跑到 101 轮早停，实际值已查不到。这里取 200 + 早停
# 耐心 12，让早停决定（与 S.14 的行为一致）。
set -euo pipefail
cd "$(dirname "$0")"

PY="C:/Users/DELL/.conda/envs/HydroModel/python.exe"
export PYTHONIOENCODING=utf-8

"$PY" -u scripts/train.py \
  --config configs/pipeline_rain1990.yaml \
  --set paths.out_dir=runs/site_model_step18 \
  --set model.output_mode=level \
  --set model.arch=moe --set model.transform=gstd \
  --set model.horizon=1 --set model.rollout=24 --set model.stride=3 \
  --set model.epochs=200 --set model.patience=12 \
  --set model.loss=quantile --set 'model.quantiles=[0.1,0.5,0.9]' \
  --set 'model.split_dates=["2023-01-01","2024-01-01"]' \
  --set model.train_fit=true --set model.masked_pool=false \
  --set model.delta_cap=0.0 \
  > logs/train_h12_step18.log 2>&1
