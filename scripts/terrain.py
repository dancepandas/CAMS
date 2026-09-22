#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 30 米高程提取水文地形因子。

在原始 30 米网格上做填洼与 D8 流向计算，得到流向、汇流累积面积、坡度与
河网级数，再裁剪到与降水网格相同的范围。汇流累积面积直接回答“某格子上游
汇集了多大范围的水”，是判断水往哪里跑的依据。

用法: python3 scripts/pipeline/terrain.py [--config configs/pipeline.yaml]
"""
import argparse
import os

import numpy as np
import pyflwdir
import rasterio
import yaml
from rasterio.merge import merge
from rasterio.windows import from_bounds, transform as window_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = 3600.0
NODATA = -32768.0


def tiles_for(lat_min, lat_max, lon_min, lon_max):
    import math
    out = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            out.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b, p = cfg["basin"], cfg["paths"]
    lat0, lat1 = float(b["lat_min"]), float(b["lat_max"])
    lon0, lon1 = float(b["lon_min"]), float(b["lon_max"])

    dem_dir = os.path.join(ROOT, p["dem30"])
    tiles = tiles_for(lat0, lat1, lon0, lon1)
    paths = []
    for t in tiles:
        hgt = os.path.join(dem_dir, f"{t}.hgt")
        if not os.path.exists(hgt):
            raise FileNotFoundError(f"缺少高程图幅 {hgt}，请先运行 fetch_dem.py")
        paths.append(hgt)
    print(f"图幅 {len(paths)} 块: {' '.join(tiles)}")

    srcs = [rasterio.open(x) for x in paths]
    mosaic, tf = merge(srcs)
    for s in srcs:
        s.close()
    full = mosaic[0].astype(np.float32)
    full[full == NODATA] = np.nan
    print(f"拼接后 {full.shape}  高程 {np.nanmin(full):.0f}~{np.nanmax(full):.0f} 米")

    win = from_bounds(lon0, lat0, lon1, lat1, transform=tf)
    r0, c0 = int(round(win.row_off)), int(round(win.col_off))
    r1 = int(round(win.row_off + win.height))
    c1 = int(round(win.col_off + win.width))
    dem = full[r0:r1, c0:c1]
    tf_crop = window_transform(win, tf)
    print(f"裁剪后 {dem.shape}  经度 {lon0}~{lon1}  纬度 {lat0}~{lat1}")

    dem_f = np.where(np.isfinite(dem), dem, NODATA)
    print("填洼与流向计算中…")
    flw = pyflwdir.from_dem(data=dem_f, nodata=NODATA, transform=tf_crop,
                            latlon=True, outlets="edge", max_depth=-1)
    flwdir = flw.to_array(ftype="d8").astype(np.uint8)
    uparea = flw.upstream_area(unit="km2").astype(np.float32)
    order = flw.stream_order().astype(np.uint8)

    mean_lat = np.radians((lat0 + lat1) / 2)
    dy, dx = 30.87, 30.87 * np.cos(mean_lat)
    filled = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    gy, gx = np.gradient(filled, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    slope[~np.isfinite(dem)] = np.nan

    print(f"完成  汇流累积 0~{np.nanmax(uparea):.1f} 平方公里  "
          f"坡度 0~{np.nanmax(slope):.1f} 度  河网级数最高 {int(order.max())} 级")
    out = os.path.join(ROOT, p["terrain"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, elevtn=dem.astype(np.float32), flwdir=flwdir,
                        uparea=uparea, slope=slope, order=order,
                        bounds=np.array([lon0, lat0, lon1, lat1]),
                        shape=np.array(dem.shape))
    print(f"已写 {out}  {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
