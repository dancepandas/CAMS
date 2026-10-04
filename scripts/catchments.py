#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""划出各断面的上游汇水区，并重投影到降水文件的真实网格。"""
import argparse
import os

import numpy as np
import pandas as pd
import pyflwdir
import xarray as xr
import yaml
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject

from spatial_grid import (atomic_savez, coordinate_edges, grid_signature,
                          validate_regular_coordinate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _site_ids(values):
    ids = [str(x) for x in values]
    if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("站号必须非空且不得重复")
    return ids


def _scalar_text(value, field):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"terrain 的 {field} 必须是单个值")
    return str(arr.reshape(()).item())


def load_terrain(path):
    """读取带真实仿射变换与坐标系的地形产物。"""
    archive = np.load(path, allow_pickle=False)
    missing = {"flwdir", "uparea", "transform", "crs"} - set(archive.files)
    if missing:
        archive.close()
        raise ValueError(f"terrain 缺少 {sorted(missing)}；请重新运行 terrain.py")
    flwdir = archive["flwdir"]
    uparea = archive["uparea"]
    coeff = np.asarray(archive["transform"], dtype=np.float64)
    crs = CRS.from_user_input(_scalar_text(archive["crs"], "crs"))
    archive.close()
    if coeff.shape != (6,):
        raise ValueError("terrain transform 必须包含 6 个仿射系数")
    if flwdir.shape != uparea.shape:
        raise ValueError("terrain 的 flwdir 与 uparea 形状不一致")
    return flwdir, uparea, Affine(*coeff), crs


def rain_grid(path):
    """读取并严格校验降水文件的二维经纬度网格。"""
    with xr.open_dataset(path) as ds:
        if "rain" not in ds or tuple(ds["rain"].dims) != ("time", "lat", "lon"):
            raise ValueError("rain 必须按 (time, lat, lon) 存储")
        lat, _ = validate_regular_coordinate(ds["lat"].values, "rain lat")
        lon, _ = validate_regular_coordinate(ds["lon"].values, "rain lon")
        if ds.sizes["lat"] != len(lat) or ds.sizes["lon"] != len(lon):
            raise ValueError("rain 维数与坐标长度不一致")
        crs_value = ds.attrs.get("crs")
        if crs_value is None:
            raise ValueError("rain 必须显式声明 crs")
        crs = CRS.from_user_input(crs_value)
    west, east = coordinate_edges(lon, "rain lon")
    south, north = coordinate_edges(lat, "rain lat")
    # rasterio 数组仍是北到南；保存前再翻回纬度递增。
    transform = from_bounds(west, south, east, north, len(lon), len(lat))
    return lat, lon, crs, transform


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
    terrain_path = os.path.join(ROOT, p["terrain"])
    flwdir, uparea, terrain_tf, terrain_crs = load_terrain(terrain_path)
    nrow, ncol = uparea.shape
    flw = pyflwdir.from_array(flwdir, ftype="d8", transform=terrain_tf,
                              latlon=terrain_crs.is_geographic)

    lat, lon, rain_crs, rain_tf = rain_grid(os.path.join(ROOT, p["rain_nc"]))
    ny, nx = len(lat), len(lon)
    print(f"地形网格 {nrow}×{ncol}  降水网格 {ny}×{nx}  断面 {len(sites)} 个")

    # 站点表为 WGS84；若 terrain 使用别的坐标系，先投影站点坐标。
    from rasterio.warp import transform as transform_points
    site_lon, site_lat = transform_points("EPSG:4326", rain_crs,
                                          sites["lon"].astype(float).tolist(),
                                          sites["lat"].astype(float).tolist())
    terrain_x, terrain_y = transform_points(rain_crs, terrain_crs,
                                             site_lon, site_lat)
    inv = ~terrain_tf
    outlets, meta = [], []
    for pos, (_, row) in enumerate(sites.iterrows()):
        # 站点经纬度可能因测量误差、DEM 河道栅格化而落在河岸格。先定位到 DEM
        # 像元，再在 ±10 个 30m 像元（约 600m 见方）内选上游汇流面积最大的格，
        # 把出口吸附到附近主河道。靠 DEM 边缘时窗口自动截断。
        col_f, row_f = inv * (terrain_x[pos], terrain_y[pos])
        c = int(np.clip(np.floor(col_f), 0, ncol - 1))
        r = int(np.clip(np.floor(row_f), 0, nrow - 1))
        ra, rb = max(0, r - 10), min(nrow, r + 11)
        ca, cb = max(0, c - 10), min(ncol, c + 11)
        win = uparea[ra:rb, ca:cb]
        if not np.isfinite(win).any():
            raise ValueError(f"站点 {row['site_id']} 周围找不到有效汇流面积")
        wr, wc = np.unravel_index(np.nanargmax(win), win.shape)
        rr, cc = ra + wr, ca + wc
        outlets.append(rr * ncol + cc)
        meta.append((str(row["site_id"]), str(row["name"]), float(row["area_km2"])))

    if terrain_crs.is_geographic:
        center_lat = float(np.mean(sites["lat"].astype(float)))
        cell_area = (abs(terrain_tf.e) * 111.32) * (
            abs(terrain_tf.a) * 111.32 * np.cos(np.radians(center_lat)))
    else:
        cell_area = abs(terrain_tf.a * terrain_tf.e - terrain_tf.b * terrain_tf.d) / 1e6

    # rasterio 的目标行序是北到南，保存契约则与 rain 一样保持 lat 递增。
    mask_north_to_south = np.zeros((len(meta), ny, nx), dtype=np.float32)
    print(f"\n{'站号':11s} {'名称':10s} {'掩膜 km²':>10s} {'实测 km²':>9s} {'比值':>6s}")
    for i, (sid, name, obs) in enumerate(meta):
        # 每个出口必须单独调用 basins。一次传入全部嵌套断面会得到互斥分区，导致
        # 下游流域丢掉上游面积；逐站划分才能保留真实的上下游嵌套关系。
        bmask = flw.basins(idxs=np.array([outlets[i]]))
        m_terrain = (bmask > 0).astype(np.float32)
        calc = float(m_terrain.sum()) * cell_area
        dst = np.zeros((ny, nx), dtype=np.float32)
        # average 不是简单的 0/1 最近邻：它给出每个 1km 降雨格被 30m 汇水区
        # 覆盖的比例，后续 area_rain.py 用这个比例做面积加权。
        reproject(m_terrain, dst, src_transform=terrain_tf, src_crs=terrain_crs,
                  dst_transform=rain_tf, dst_crs=rain_crs,
                  resampling=Resampling.average)
        mask_north_to_south[i] = dst
        print(f"{sid:11s} {name:10s} {calc:10.1f} {obs:9.1f} {calc / obs:6.2f}")

    mask1km = mask_north_to_south[:, ::-1, :]
    if mask1km.shape != (len(expected_ids), ny, nx):
        raise RuntimeError("汇水区掩膜形状不正确")
    out = os.path.join(ROOT, p["catchments"])
    atomic_savez(out, mask1km=mask1km,
                 site_ids=np.array([m[0] for m in meta]),
                 names=np.array([m[1] for m in meta]),
                 area_obs=np.array([m[2] for m in meta], dtype=np.float32),
                 lat=lat, lon=lon, crs=np.array(rain_crs.to_string()),
                 mask_dims=np.array(["site", "lat", "lon"]))
    print(f"\n已写 {out}  {os.path.getsize(out) / 1e6:.1f} MB  "
          f"网格 {grid_signature(lat, lon, rain_crs.to_string())}")


if __name__ == "__main__":
    main()
