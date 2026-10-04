#!/bin/bash
set -e
cd /d/chengs/CAMS
PY="C:/Users/DELL/.conda/envs/HydroModel/python.exe"
export PYTHONIOENCODING=utf-8
for m in site_model_step18_rainfix site_model_step18_unroll site_model_step20_bptt_up site_model_step20_bptt_asym; do
  echo "===== $m ====="
  "$PY" -X utf8 scripts/rollout_dense.py --config configs/pipeline_rain1990.yaml \
    --run-dir runs/$m --split val --roll 24 --stride 1 || exit 1
done
echo ALL_DONE
