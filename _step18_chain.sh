#!/usr/bin/env bash
# S.18 降雨数据链：等待严格版 AORC 年度件完成，再拼接降雨并重算面雨量。
# USGS 流量不在这里重拉；已有流量文件保持不动。
set -euo pipefail
cd "$(dirname "$0")"

PY="C:/Users/DELL/.conda/envs/HydroModel/python.exe"
export PYTHONIOENCODING=utf-8
DL_LOG="logs/fetch_aorc_1990_2015.log"
AORC_DIR="data/aorc_regrid_float32"
CONFIG="configs/pipeline_rain1990.yaml"

note() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { note "错误：$*" >&2; exit 1; }

note "AORC 前检查 MRMS、地形和汇水区空间合同"
"$PY" -u - <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from area_rain import _site_ids, load_catchments, validate_rain
from catchments import load_terrain
from spatial_grid import grid_signature, mrms_coordinates, mrms_window

with open(root / "configs" / "pipeline.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
paths, basin, period = config["paths"], config["basin"], config["period"]
rain_path = root / paths["rain_nc"]
if rain_path.resolve() != (root / "data" / "rain_mrms_1km.nc").resolve():
    raise ValueError("基准配置 rain_nc 与流水线指定的 MRMS 文件不是同一个路径")
rows, cols = mrms_window(float(basin["lat_min"]), float(basin["lat_max"]),
                         float(basin["lon_min"]), float(basin["lon_max"]))
expected_lat, expected_lon = mrms_coordinates(rows, cols)
expected_sig = grid_signature(expected_lat, expected_lon, "EPSG:4326")
with xr.open_dataset(rain_path) as dataset:
    rain = dataset["rain"]
    if rain.dims != ("time", "lat", "lon") or rain.dtype != np.float32:
        raise ValueError("MRMS rain 必须是 float32 的 (time, lat, lon)")
    if not np.array_equal(dataset["lat"].values, expected_lat):
        raise ValueError("MRMS 纬度不是配置对应的真实格点中心")
    if not np.array_equal(dataset["lon"].values, expected_lon):
        raise ValueError("MRMS 经度不是配置对应的真实格点中心")
    if dataset.attrs.get("crs") != "EPSG:4326":
        raise ValueError("MRMS 必须显式声明 EPSG:4326")
    if dataset.attrs.get("grid_signature") != expected_sig:
        raise ValueError("MRMS 网格指纹缺失或错误")
    times = pd.DatetimeIndex(dataset["time"].values)
    if times[0] != pd.Timestamp(period["start"]) or times[-1] != pd.Timestamp(period["end"]):
        raise ValueError(f"MRMS 时间范围 {times[0]} ~ {times[-1]} 与配置不一致")

load_terrain(root / paths["terrain"])
import pandas as pd
sites = pd.read_csv(root / paths["sites_csv"], encoding="utf-8", dtype={"site_id": str})
expected_ids = _site_ids(sites["site_id"])
_mask, _ids, _names, _areas, lat, lon, crs = load_catchments(
    root / paths["catchments"], expected_ids)
with xr.open_dataset(rain_path) as dataset:
    validate_rain(dataset, lat, lon, crs, period["start"], period["end"])
print("MRMS、terrain、catchments 空间合同全部通过")
PYEOF

note "等待 AORC 下载结束"
while ! grep -q "全部完成" "$DL_LOG" 2>/dev/null; do
    m=$(stat -c %Y "$DL_LOG" 2>/dev/null || printf '0')
    now=$(date +%s)
    if [ $((now - m)) -gt 1800 ]; then
        fail "下载日志已 30 分钟未更新：$DL_LOG"
    fi
    sleep 60
done

note "验证 1990--2015 共 26 个年度文件"
"$PY" -u - "$AORC_DIR" <<'PYEOF'
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from fetch_rain_aorc import target_grid, validate_annual_file

aorc_dir = root / sys.argv[1]
lat, lon = target_grid(root / "data" / "rain_mrms_1km.nc")
for year in range(1990, 2016):
    validate_annual_file(aorc_dir / f"aorc_{year}.npz", year, lat, lon)
print("26 个年度文件全部通过校验")
PYEOF

note "=== 1/3 重算 S.14 面雨量（原 MRMS + 原配置）==="
"$PY" -u scripts/area_rain.py --config configs/pipeline.yaml
note "S.14 面雨量重算结束"

note "=== 2/3 拼接 AORC + MRMS 降雨 ==="
"$PY" -u scripts/build_rain_extended.py \
    --config "$CONFIG" \
    --aorc-dir "$AORC_DIR" \
    --mrms-nc data/rain_mrms_1km.nc \
    --output-nc data/rain_1990_2024.nc
note "拼接结束"

note "=== 3/3 重算 S.18 面雨量（扩展降雨 + 扩展配置）==="
"$PY" -u scripts/area_rain.py --config "$CONFIG"
note "S.18 面雨量重算结束"

note "=== 产物与公共 MRMS 时段自检 ==="
"$PY" -u - <<'PYEOF'
import numpy as np
import pandas as pd
import xarray as xr

with xr.open_dataset("data/rain_1990_2024.nc") as ds:
    print("降雨网格", ds["rain"].shape, ds["rain"].dtype,
          str(ds.time.values[0])[:16], "~", str(ds.time.values[-1])[:16])

with np.load("data/area_rain.npz", allow_pickle=True) as base, \
        np.load("data/area_rain_1990_2024.npz", allow_pickle=True) as extended:
    base_rain = base["area_rain"]
    extended_rain = extended["area_rain"]
    base_times = pd.DatetimeIndex([str(value) for value in base["times"]])
    extended_times = pd.DatetimeIndex([str(value) for value in extended["times"]])
    base_sites = np.asarray([str(value) for value in base["site_ids"]])
    extended_sites = np.asarray([str(value) for value in extended["site_ids"]])

if not np.array_equal(base_sites, extended_sites):
    raise ValueError("S.14 与 S.18 面雨量的 site_ids 顺序不一致")
if extended_times[0] != pd.Timestamp("1990-01-01 00:00"):
    raise ValueError(f"S.18 面雨量起点错误：{extended_times[0]}")
if len(extended_times) > 1 and not np.all(np.diff(extended_times.values).astype(
        "timedelta64[s]").astype(np.int64) == 3600):
    raise ValueError("S.18 面雨量时间轴不是逐小时连续")

positions = extended_times.get_indexer(base_times)
if np.any(positions < 0):
    first = int(np.flatnonzero(positions < 0)[0])
    raise ValueError(f"S.18 缺少 S.14 时间点：{base_times[first]}")
overlap = extended_rain[:, positions]
if overlap.shape != base_rain.shape:
    raise ValueError(f"公共时段形状不一致：{overlap.shape} 对 {base_rain.shape}")
finite = np.isfinite(overlap) & np.isfinite(base_rain)
if not np.array_equal(np.isfinite(overlap), np.isfinite(base_rain)):
    raise ValueError("公共 MRMS 时段的缺测位置不一致")
max_error = float(np.max(np.abs(overlap[finite] - base_rain[finite]))) if finite.any() else 0.0
absolute_tolerance = 2e-5
if max_error > absolute_tolerance:
    raise ValueError(f"公共 MRMS 时段面雨量最大偏差 {max_error:.9g}，"
                     f"超过容差 {absolute_tolerance:g}")
print("S.18 面雨量", extended_rain.shape, extended_rain.dtype,
      str(extended_times[0])[:16], "~", str(extended_times[-1])[:16])
print(f"公共 MRMS 时段 {len(base_times)} 小时逐点对齐，最大偏差 {max_error:.9g}，"
      f"容差 {absolute_tolerance:g}")
PYEOF
note "降雨数据链全部结束，可以启动训练"
