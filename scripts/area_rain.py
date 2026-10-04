#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按汇水区覆盖比例计算各断面的逐小时面雨量。"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from data_contract import RAIN_VALUES_FORMAT, RainValueHasher
from spatial_grid import (atomic_savez, grid_signature,
                          validate_regular_coordinate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATCHMENT_FIELDS = {"mask1km", "site_ids", "names", "area_obs", "lat", "lon",
                    "crs", "mask_dims"}


def _site_ids(values, expected=None):
    ids = [str(x) for x in values]
    if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("站号必须非空且不得重复")
    if expected is not None and ids != [str(x) for x in expected]:
        raise ValueError("汇水区站序与站点配置不一致；请重新运行 catchments.py")
    return ids


def _hourly_times(values, expected_start=None, expected_end=None):
    arr = np.asarray(values).astype("datetime64[ns]")
    if arr.ndim != 1 or arr.size == 0 or np.isnat(arr).any():
        raise ValueError("time 必须是一维、非空且有效")
    if arr.size > 1 and not np.all(np.diff(arr) == np.timedelta64(1, "h")):
        raise ValueError("time 必须无重复、无缺口地逐小时递增")
    if expected_start is not None and arr[0] != np.datetime64(expected_start, "ns"):
        raise ValueError(f"time 起点 {arr[0]} 与配置 {expected_start} 不一致")
    if expected_end is not None and arr[-1] != np.datetime64(expected_end, "ns"):
        raise ValueError(f"time 终点 {arr[-1]} 与配置 {expected_end} 不一致")
    return arr


def _scalar_text(value, field):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{field} 必须是单个值")
    return str(arr.reshape(()).item())


def load_catchments(path, expected_site_ids):
    """读取并校验汇水区产物契约。"""
    with np.load(path, allow_pickle=False) as d:
        missing = CATCHMENT_FIELDS - set(d.files)
        if missing:
            raise ValueError(f"catchments 缺少 {sorted(missing)}；请重新运行 catchments.py")
        mask = d["mask1km"].astype(np.float32)
        ids = _site_ids(d["site_ids"], expected_site_ids)
        names = np.asarray(d["names"]).astype(str)
        areas = np.asarray(d["area_obs"], dtype=np.float32)
        lat, _ = validate_regular_coordinate(d["lat"], "catchments lat")
        lon, _ = validate_regular_coordinate(d["lon"], "catchments lon")
        crs = _scalar_text(d["crs"], "catchments crs")
        dims = tuple(np.asarray(d["mask_dims"]).astype(str).tolist())
    if dims != ("site", "lat", "lon"):
        raise ValueError(f"catchments mask_dims 应为 site/lat/lon，实际为 {dims}")
    if mask.shape != (len(ids), len(lat), len(lon)):
        raise ValueError("mask1km 形状与站点或坐标长度不一致")
    if len(names) != len(ids) or len(areas) != len(ids):
        raise ValueError("catchments 站点元数据长度不一致")
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("mask1km 必须是 0 到 1 的有限覆盖比例")
    if np.any(mask.reshape(len(ids), -1).sum(axis=1) <= 0):
        raise ValueError("至少一个站点的汇水区掩膜为空")
    return mask, ids, names, areas, lat, lon, crs


def validate_rain(ds, lat, lon, crs, expected_start=None, expected_end=None):
    """要求降水坐标、顺序、时间与汇水区产物逐项一致。"""
    if "rain" not in ds or tuple(ds["rain"].dims) != ("time", "lat", "lon"):
        raise ValueError("rain 必须按 (time, lat, lon) 存储")
    rain_lat, _ = validate_regular_coordinate(ds["lat"].values, "rain lat")
    rain_lon, _ = validate_regular_coordinate(ds["lon"].values, "rain lon")
    if not np.array_equal(rain_lat, lat) or not np.array_equal(rain_lon, lon):
        raise ValueError("rain 与 catchments 的经纬度不完全一致；请重新运行 catchments.py")
    rain_crs_value = ds.attrs.get("crs")
    if rain_crs_value is None:
        raise ValueError("rain 必须显式声明 crs")
    rain_crs = str(rain_crs_value)
    if rain_crs != crs:
        raise ValueError(f"rain 坐标系 {rain_crs} 与 catchments 坐标系 {crs} 不一致")
    times = _hourly_times(ds["time"].values, expected_start, expected_end)
    sig = grid_signature(lat, lon, crs)
    if ds.attrs.get("grid_signature") not in (None, sig):
        raise ValueError("rain 文件的网格指纹与实际坐标不一致")
    return times, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]

    sites = pd.read_csv(os.path.join(ROOT, p["sites_csv"]), encoding="utf-8",
                        dtype={"site_id": str})
    expected_ids = _site_ids(sites["site_id"])
    mask, ids, names, areas, lat, lon, crs = load_catchments(
        os.path.join(ROOT, p["catchments"]), expected_ids)

    rain_path = os.path.join(ROOT, p["rain_nc"])
    chunk_hours = int(cfg.get("area_rain_chunk_hours", 168))
    if chunk_hours < 1:
        raise ValueError("area_rain_chunk_hours 必须大于 0")
    with xr.open_dataset(rain_path) as ds:
        times, signature = validate_rain(ds, lat, lon, crs)
        source = ds["rain"]
        T = len(times)
        area_rain = np.full((len(ids), T), np.nan, dtype=np.float32)
        # W 的形状是（格点, 站点），每个元素表示目标格点被该站汇水区覆盖的比例。
        # 对每个小时：num=Σ(有效降雨×覆盖比例)，wt=Σ(有效格点覆盖比例)。NaN
        # 同时从分子和分母剔除，所以按剩余有效面积重新归一，绝不把缺测当成零雨。
        # 矩阵乘法先得到（小时, 站点），写入产物时再转成（站点, 小时）。
        W = mask.reshape(len(ids), -1).T
        hasher = RainValueHasher(source.shape)
        for lo in range(0, T, chunk_hours):
            hi = min(T, lo + chunk_hours)
            rain = source.isel(time=slice(lo, hi)).values.astype(np.float32)
            hasher.update(rain)
            finite = np.isfinite(rain)
            F = finite.reshape(hi - lo, -1).astype(np.float32)
            V = np.where(finite, rain, 0.0).reshape(hi - lo, -1)
            wt = F @ W
            num = V @ W
            ok = wt > 0
            area_rain[:, lo:hi] = np.where(
                ok, num / np.where(ok, wt, 1.0), np.nan).T
            print(f"  面雨量 {hi:7d}/{T} 小时", flush=True)
        source_rain_digest = hasher.hexdigest()

    print(f"降水 {(T, len(lat), len(lon))}  {times[0]} ~ {times[-1]}  "
          f"断面 {len(ids)} 个")

    print(f"面雨量 {area_rain.shape}  {np.nanmin(area_rain):.2f}~{np.nanmax(area_rain):.2f} 毫米/小时")
    out = os.path.join(ROOT, p["area_rain"])
    atomic_savez(out, area_rain=area_rain.astype(np.float32),
                 times=np.array([str(pd.Timestamp(t)) for t in times]),
                 site_ids=np.array(ids), names=names,
                 areas=areas.astype(np.float32), lat=lat, lon=lon,
                 crs=np.array(crs), grid_signature=np.array(signature),
                 source_rain_values_format=np.array(RAIN_VALUES_FORMAT),
                 source_rain_values_sha256=np.array(source_rain_digest))
    # 重开检查：原子发布成功后也要确认最终文件可读且字段齐全。
    with np.load(out, allow_pickle=False) as check:
        required = {"area_rain", "times", "site_ids", "names", "areas", "lat",
                    "lon", "crs", "grid_signature", "source_rain_values_format",
                    "source_rain_values_sha256"}
        if required - set(check.files) or check["area_rain"].shape != area_rain.shape:
            raise RuntimeError("面雨量文件发布后校验失败")
    print(f"已写 {out}  {os.path.getsize(out) / 1e6:.1f} MB  网格 {signature}")


if __name__ == "__main__":
    main()
