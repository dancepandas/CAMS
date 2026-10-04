#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 30 米高程提取水文地形因子。

在原始 30 米网格上做填洼与 D8 流向计算，得到流向、汇流累积面积、坡度与
河网级数，再裁剪到与降水网格相同的范围。汇流累积面积直接回答“某格子上游
汇集了多大范围的水”，是判断水往哪里跑的依据。

用法: python3 scripts/terrain.py [--config configs/pipeline.yaml]
"""
import argparse
import os

import numpy as np
import pyflwdir
import rasterio
import xarray as xr
import yaml
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds, transform as window_transform

from spatial_grid import atomic_savez, coordinate_edges, validate_regular_coordinate

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
    p = cfg["paths"]
    rain_path = os.path.join(ROOT, p["rain_nc"])
    with xr.open_dataset(rain_path) as rain_ds:
        if "rain" not in rain_ds or tuple(rain_ds["rain"].dims) != ("time", "lat", "lon"):
            raise ValueError("rain 必须按 (time, lat, lon) 存储")
        rain_lat, _ = validate_regular_coordinate(rain_ds["lat"].values, "rain lat")
        rain_lon, _ = validate_regular_coordinate(rain_ds["lon"].values, "rain lon")
        crs_value = rain_ds.attrs.get("crs")
        if crs_value is None:
            raise ValueError("rain 必须显式声明 crs")
        rain_crs = CRS.from_user_input(crs_value)
    if not rain_crs.is_geographic:
        raise ValueError(f"rain 网格应为经纬度坐标，实际为 {rain_crs}")
    lon0, lon1 = coordinate_edges(rain_lon, "rain lon")
    lat0, lat1 = coordinate_edges(rain_lat, "rain lat")

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
    try:
        mosaic, tf = merge(srcs)
        source_crs = srcs[0].crs
        if source_crs is None:
            raise ValueError("高程图幅缺少坐标系")
        if any(s.crs != source_crs for s in srcs):
            raise ValueError("高程图幅坐标系不一致")
    finally:
        for s in srcs:
            s.close()
    if source_crs is None or not source_crs.is_geographic:
        raise ValueError(f"高程图幅应为经纬度坐标，实际为 {source_crs}")
    full = mosaic[0].astype(np.float32)
    full[full == NODATA] = np.nan
    print(f"拼接后 {full.shape}  高程 {np.nanmin(full):.0f}~{np.nanmax(full):.0f} 米")

    requested = from_bounds(lon0, lat0, lon1, lat1, transform=tf)
    r0 = max(0, int(np.floor(requested.row_off)))
    c0 = max(0, int(np.floor(requested.col_off)))
    r1 = min(full.shape[0], int(np.ceil(requested.row_off + requested.height)))
    c1 = min(full.shape[1], int(np.ceil(requested.col_off + requested.width)))
    win = Window(c0, r0, c1 - c0, r1 - r0)
    dem = full[r0:r1, c0:c1]
    tf_crop = window_transform(win, tf)
    west, north = tf_crop * (0, 0)
    east, south = tf_crop * (dem.shape[1], dem.shape[0])
    print(f"裁剪后 {dem.shape}  经度 {west:.6f}~{east:.6f}  纬度 {south:.6f}~{north:.6f}")

    dem_f = np.where(np.isfinite(dem), dem, NODATA)
    print("填洼与流向计算中…")
    flw = pyflwdir.from_dem(data=dem_f, nodata=NODATA, transform=tf_crop,
                            latlon=True, outlets="edge", max_depth=-1)
    flwdir = flw.to_array(ftype="d8").astype(np.uint8)
    uparea = flw.upstream_area(unit="km2").astype(np.float32)
    order = flw.stream_order().astype(np.uint8)

    # SRTM 1 弧秒在纬向约为 30.87m；经向距离随纬度按 cos(lat) 缩短。
    # 对本流域小范围足够用于坡度特征，但不是测量级地形距离。缺测高程只为
    # np.gradient 临时填入最小值，算完立即把原缺测位置的坡度恢复成 NaN。
    mean_lat = np.radians(float(np.mean(rain_lat)))
    dy, dx = 30.87, 30.87 * np.cos(mean_lat)
    filled = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    gy, gx = np.gradient(filled, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    slope[~np.isfinite(dem)] = np.nan

    print(f"完成  汇流累积 0~{np.nanmax(uparea):.1f} 平方公里  "
          f"坡度 0~{np.nanmax(slope):.1f} 度  河网级数最高 {int(order.max())} 级")
    out = os.path.join(ROOT, p["terrain"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    crs = source_crs.to_string()
    atomic_savez(out, elevtn=dem.astype(np.float32), flwdir=flwdir,
                 uparea=uparea, slope=slope, order=order,
                 bounds=np.array([west, south, east, north]),
                 shape=np.array(dem.shape),
                 transform=np.array(tuple(tf_crop)[:6], dtype=np.float64),
                 crs=np.array(crs))
    print(f"已写 {out}  {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
