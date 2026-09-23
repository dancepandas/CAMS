#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算各断面的面雨量序列。

按汇水区覆盖比例对降水网格做面积加权平均，得到每个断面逐小时的面雨量。

用法: python3 scripts/pipeline/area_rain.py [--config configs/pipeline.yaml]
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]

    d = np.load(os.path.join(ROOT, p["catchments"]), allow_pickle=True)
    mask = d["mask1km"].astype(np.float32)
    ids = [str(x) for x in d["site_ids"]]
    names = [str(x) for x in d["names"]]
    areas = d["area_obs"]

    ds = xr.open_dataset(os.path.join(ROOT, p["rain_nc"]))
    rain = ds["rain"].values.astype(np.float32)
    times = pd.DatetimeIndex(ds.time.values)
    T = len(times)
    print(f"降水 {rain.shape}  {times[0]} ~ {times[-1]}  断面 {len(ids)} 个")

    # 时间轴长（八万余小时）时，逐断面展开浮点临时数组会吃掉上百 GB，
    # 改成一次性矩阵乘：把格点摊平，用掩膜权重直接投影到各断面。
    finite = np.isfinite(rain)
    F = finite.reshape(T, -1).astype(np.float32)
    V = np.where(finite, rain, 0.0).reshape(T, -1)
    del finite
    W = mask.reshape(len(ids), -1).T            # (格点, 断面)
    wt = F @ W                                  # (T, 断面) 有效权重
    num = V @ W                                 # (T, 断面) 加权和
    del F, V, rain
    ok = wt > 0
    area_rain = np.where(ok, num / np.where(ok, wt, 1.0), np.nan).T  # (断面, T)

    print(f"面雨量 {area_rain.shape}  {np.nanmin(area_rain):.2f}~{np.nanmax(area_rain):.2f} 毫米/小时")
    out = os.path.join(ROOT, p["area_rain"])
    np.savez_compressed(out, area_rain=area_rain.astype(np.float32),
                        times=np.array([str(t) for t in times]),
                        site_ids=np.array(ids), names=np.array(names),
                        areas=areas.astype(np.float32))
    print(f"已写 {out}  {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
