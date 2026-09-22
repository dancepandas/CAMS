#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""划出各断面的上游汇水区，并聚合到降水网格。

在 30 米流向场上从各断面反向追溯汇水范围，再把二值掩膜按面积加权重采样
到降水网格，得到每个断面的覆盖比例场，供计算面雨量使用。

注意须逐个断面单独追溯：若把嵌套的多个出口一次性交给划分函数，每个格子
只会归给最上游的那个出口，下游断面的汇水区会被上游切走。

用法: python3 scripts/pipeline/catchments.py [--config configs/pipeline.yaml]
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyflwdir
import yaml
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b, p = cfg["basin"], cfg["paths"]

    lon0, lat1 = float(b["lon_min"]), float(b["lat_max"])
    G, step_deg = int(b["grid"]), float(b["step"])

    sites = pd.read_csv(os.path.join(ROOT, p["sites_csv"]), encoding="utf-8",
                        dtype={"site_id": str})
    t = np.load(os.path.join(ROOT, p["terrain"]))
    flwdir, uparea = t["flwdir"], t["uparea"]
    nrow, ncol = uparea.shape
    tf30 = Affine(1 / SEC, 0, lon0, 0, -1 / SEC, lat1)
    flw = pyflwdir.from_array(flwdir, ftype="d8", transform=tf30, latlon=True)
    print(f"30 米网格 {nrow}×{ncol}  断面 {len(sites)} 个")

    outlets, meta = [], []
    for _, row in sites.iterrows():
        c = int(np.clip(round((row["lon"] - lon0) * SEC - 0.5), 0, ncol - 1))
        r = int(np.clip(round((lat1 - row["lat"]) * SEC - 0.5), 0, nrow - 1))
        win = uparea[max(0, r - 10):r + 11, max(0, c - 10):c + 11]
        wr, wc = np.unravel_index(np.nanargmax(win), win.shape)
        rr, cc = max(0, r - 10) + wr, max(0, c - 10) + wc
        outlets.append(rr * ncol + cc)
        meta.append((str(row["site_id"]), str(row["name"]), row["area_km2"]))

    cell_area = (111.32 / SEC) * (111.32 / SEC * np.cos(np.radians((float(b["lat_min"]) + lat1) / 2)))
    tf1 = from_origin(lon0 - step_deg / 2, lat1 + step_deg / 2, step_deg, step_deg)

    mask1km = np.zeros((len(meta), G, G), dtype=np.float32)
    print(f"\n{'站号':11s} {'名称':10s} {'掩膜 km²':>10s} {'实测 km²':>9s} {'比值':>6s}")
    for i, (sid, name, obs) in enumerate(meta):
        bmask = flw.basins(idxs=np.array([outlets[i]]))
        m30 = (bmask > 0).astype(np.float32)
        calc = float(m30.sum()) * cell_area
        dst = np.zeros((G, G), dtype=np.float32)
        reproject(m30, dst, src_transform=tf30, src_crs="EPSG:4326",
                  dst_transform=tf1, dst_crs="EPSG:4326",
                  resampling=Resampling.average)
        mask1km[i] = dst
        print(f"{sid:11s} {name:10s} {calc:10.1f} {obs:9.1f} {calc / obs:6.2f}")

    out = os.path.join(ROOT, p["catchments"])
    np.savez_compressed(out, mask1km=mask1km,
                        site_ids=np.array([m[0] for m in meta]),
                        names=np.array([m[1] for m in meta]),
                        area_obs=np.array([m[2] for m in meta], dtype=np.float32))
    print(f"\n已写 {out}  {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
