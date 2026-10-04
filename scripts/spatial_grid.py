#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""空间网格的共同定义、校验和安全写盘工具。

整个项目的“网格真源”只此一处：MRMS 全球网格的行/列索引与经纬度换算、
项目 128×128 裁剪网格的坐标生成、跨文件比对用的网格指纹（SHA256）、以及
原子写盘。任何脚本需要坐标时都必须从这里取，不允许各自硬编码。

关键事实（改任何一处前先读这里）：

- MRMS 全球网格为 3500 行 × 7000 列，行从北纬 54.995° 向南每行减 0.01°，
  列从西经 -129.995° 向东每列加 0.01°；存的是格点中心，不是角点。
- 项目降雨数组按“纬度递增、第 0 行在最南”存储，与 xarray 的常规一致；
  从 MRMS 原始行序（北到南）读入时必须翻转（见 fetch_rain.decode）。
- 网格指纹把 CRS + 维度名 + 形状 + 坐标实值一起做 SHA256，任何半格偏移、
  方向翻转或形状变化都会得到不同指纹，用于下游入口的强制校验。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# MRMS 全球网格真源：爱荷华州立镜像上所有产品共用这一套行列定义。
# 坐标是格点中心；第 0 行在 54.995°N，第 0 列在 129.995°W。
MRMS_NORTH = 54.995
MRMS_WEST = -129.995
MRMS_STEP = 0.01
MRMS_NROW = 3500
MRMS_NCOL = 7000


def mrms_window(lat_min: float, lat_max: float, lon_min: float,
                lon_max: float) -> tuple[slice, slice]:
    """返回覆盖给定 MRMS 格点中心范围的全局行列窗口。

    用 round 而不是 floor：配置里写的 lat/lon 本来就是格点中心坐标，直接
    反解回整数索引；取整误差若用 floor 会在边界上差出一整行/列。
    """
    # 纬度方向：MRMS 行序是北→南，行号 = (最北 - 目标纬度) / 步长
    r0 = int(round((MRMS_NORTH - lat_max) / MRMS_STEP))
    r1 = int(round((MRMS_NORTH - lat_min) / MRMS_STEP)) + 1
    # 经度方向：列序是西→东，列号 = (目标经度 - 最西) / 步长
    c0 = int(round((lon_min - MRMS_WEST) / MRMS_STEP))
    c1 = int(round((lon_max - MRMS_WEST) / MRMS_STEP)) + 1
    if not (0 <= r0 < r1 <= MRMS_NROW and 0 <= c0 < c1 <= MRMS_NCOL):
        raise ValueError(f"范围超出 MRMS 美国本土网格：行 {r0}:{r1} 列 {c0}:{c1}")
    return slice(r0, r1), slice(c0, c1)


def mrms_coordinates(rows: slice, cols: slice) -> tuple[np.ndarray, np.ndarray]:
    """由 MRMS 全局索引生成坐标；纬度顺序与南到北的数值数组一致。"""
    if rows.step not in (None, 1) or cols.step not in (None, 1):
        raise ValueError("MRMS 裁剪窗口只支持连续索引")
    if None in (rows.start, rows.stop, cols.start, cols.stop):
        raise ValueError("MRMS 裁剪窗口必须有明确起止索引")
    # 先按 MRMS 原生的北→南行序生成，再翻转成项目统一的南→北（纬度递增）
    north_to_south = MRMS_NORTH - np.arange(rows.start, rows.stop) * MRMS_STEP
    lon = MRMS_WEST + np.arange(cols.start, cols.stop) * MRMS_STEP
    return north_to_south[::-1].astype(np.float64), lon.astype(np.float64)


def validate_regular_coordinate(values: Iterable[float], name: str,
                                increasing: bool = True) -> tuple[np.ndarray, float]:
    """校验一维、有限、严格单调且等间隔的格点中心坐标。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} 必须是一维且至少包含两个格点")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含非有限值")
    delta = np.diff(arr)
    if increasing and not np.all(delta > 0):
        raise ValueError(f"{name} 必须严格递增")
    if not increasing and not np.all(delta < 0):
        raise ValueError(f"{name} 必须严格递减")
    step = float(np.median(delta))
    atol = max(abs(step) * 1e-8, 1e-12)
    if not np.allclose(delta, step, rtol=0.0, atol=atol):
        raise ValueError(f"{name} 不是等间隔坐标")
    return arr, step


def coordinate_edges(values: Iterable[float], name: str) -> tuple[float, float]:
    """返回递增格点中心轴的外边界。"""
    arr, step = validate_regular_coordinate(values, name)
    return float(arr[0] - step / 2), float(arr[-1] + step / 2)


def grid_signature(lat: Iterable[float], lon: Iterable[float],
                   crs: str = "EPSG:4326",
                   dims: Sequence[str] = ("lat", "lon")) -> str:
    """生成可跨文件比较的网格指纹。"""
    lat_arr, _ = validate_regular_coordinate(lat, "lat")
    lon_arr, _ = validate_regular_coordinate(lon, "lon")
    meta = json.dumps({"dims": list(dims), "crs": str(crs),
                       "shape": [int(lat_arr.size), int(lon_arr.size)]},
                      ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(meta.encode("utf-8"))
    # 坐标实值按 little-endian float64 的字节序入哈希，
    # 保证同一网格在任何机器上得到同一个指纹
    digest.update(np.asarray(lat_arr, dtype="<f8").tobytes())
    digest.update(np.asarray(lon_arr, dtype="<f8").tobytes())
    return "sha256:" + digest.hexdigest()


def atomic_savez(path: os.PathLike | str, **arrays) -> None:
    """在目标同目录写临时文件，落盘后再一次性替换目标。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                    dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
