#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 1990--2015 年 AORC 逐小时降水并重采样到项目网格。

每个年度文件都必须覆盖该自然年的每一个整点。文件先写到同目录的 staging
文件，重开检查无误后再原子替换为 ``aorc_<年>.npz``；已有文件也只有通过
同一套检查才会跳过。任何网络、HTTP、解压或数据块错误都会让程序失败退出。
"""
from __future__ import annotations

import argparse
import json
import os
import time as _time
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import requests
import xarray as xr
from numcodecs import Zstd
from scipy.sparse import csr_matrix

from spatial_grid import grid_signature

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com"
VARIABLE = "APCP_surface"
SCALE = 0.1
FILL = -32767
CHUNK_T, CHUNK_Y, CHUNK_X = 144, 128, 256
TARGET_STEP = 0.01
REQUIRED_FIRST_YEAR = 1990
REQUIRED_LAST_YEAR = 2015
EXPECTED_TARGET_GRID_SIGNATURE = (
    "sha256:3be9b888f1b799e982fe36a81217e135271a5685380139f376913e4a6c5b28e2")
MIN_VALID_FRACTION = 0.50
VALIDATION_CHUNK_HOURS = 168


class AorcDownloadError(RuntimeError):
    """AORC 请求在重试后仍未成功。"""


class AorcReader:
    """用并发 HTTP 请求直接读取公开 AORC Zarr 数据块。"""

    def __init__(self, threads: int = 32, attempts: int = 3,
                 timeout: float = 90.0) -> None:
        if threads < 1 or attempts < 1:
            raise ValueError("threads 和 attempts 必须大于 0")
        self.threads = threads
        self.attempts = attempts
        self.timeout = timeout
        self._zst = Zstd()
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})
        # 重试只在这里做，避免 requests 在底层静默重试后又返回不清楚的结果。
        adapter = requests.adapters.HTTPAdapter(pool_connections=threads,
                                                pool_maxsize=threads,
                                                max_retries=0)
        self.session.mount("https://", adapter)
        self.pool = ThreadPoolExecutor(threads)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.session.close()

    def __enter__(self) -> "AorcReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get(self, url: str) -> bytes:
        last: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return response.content
                # 404 是确定性错误；不要把“文件不存在”伪装成合法缺测。
                if response.status_code == 404:
                    raise AorcDownloadError(f"AORC 资源不存在（404）：{url}")
                last = AorcDownloadError(
                    f"AORC 请求返回 HTTP {response.status_code}：{url}")
            except AorcDownloadError:
                raise
            except requests.RequestException as exc:
                last = exc
            if attempt < self.attempts:
                _time.sleep(1.5 * attempt)
        raise AorcDownloadError(
            f"AORC 请求重试 {self.attempts} 次仍失败：{url}") from last

    def chunks(self, year: int,
               jobs: Sequence[Tuple[int, int, int]]) -> list[bytes]:
        """并发下载一批块；任一块失败时整批抛错，不返回残缺结果。"""
        futures: list[Future[bytes]] = []
        for t_chunk, y_chunk, x_chunk in jobs:
            url = (f"{BUCKET}/{year}.zarr/{VARIABLE}/"
                   f"{t_chunk}.{y_chunk}.{x_chunk}")
            futures.append(self.pool.submit(self._get, url))
        try:
            # 单次请求最多会经历 attempts 次网络等待，因此线程等待上限不能短于
            # ``timeout × attempts``。这里保留 15 分钟的整批保险上限：正常失败仍由
            # _get 在约 5 分钟内报出具体 URL；只有底层网络库没有遵守自身超时时，
            # 外层才会兜底终止，避免无人值守任务永久挂住。
            retry_backoff = 1.5 * sum(range(1, self.attempts))
            request_budget = self.timeout * self.attempts + retry_backoff + 30.0
            future_timeout = max(request_budget, 15.0 * 60.0)
            parts = [future.result(timeout=future_timeout) for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        if len(parts) != len(jobs):
            raise AorcDownloadError(
                f"{year} 年请求 {len(jobs)} 块，只返回 {len(parts)} 块")
        return parts

    def coord(self, year: int, name: str) -> np.ndarray:
        """按照 Zarr 元数据读取并拼接一维坐标数组。"""
        meta = json.loads(self._get(
            f"{BUCKET}/{year}.zarr/{name}/.zarray"))
        n = int(meta["shape"][0])
        step = int(meta["chunks"][0])
        codec = (meta.get("compressor") or {}).get("id")
        if codec not in (None, "zstd", "zlib"):
            raise ValueError(f"{name} 使用了不支持的压缩方式 {codec}")
        dtype = np.dtype(meta["dtype"])

        parts: list[np.ndarray] = []
        for chunk_id in range((n + step - 1) // step):
            raw = self._get(f"{BUCKET}/{year}.zarr/{name}/{chunk_id}")
            if codec == "zstd":
                raw = self._zst.decode(raw)
            elif codec == "zlib":
                raw = zlib.decompress(raw)
            if len(raw) % dtype.itemsize:
                raise ValueError(
                    f"{year} 年坐标 {name} 第 {chunk_id} 块字节数非法：{len(raw)}")
            parts.append(np.frombuffer(raw, dtype=dtype))
        arr = np.concatenate(parts)[:n]
        if arr.size != n:
            raise ValueError(f"{name} 解出 {arr.size} 个值，与声明的 {n} 不符")
        return arr


def assemble(parts: Sequence[bytes], y_chunks: Sequence[int],
             x_chunks: Sequence[int]) -> np.ndarray:
    """拼接一个时间块；不允许缺块，也不把下载失败补成 FILL。"""
    expected_parts = len(y_chunks) * len(x_chunks)
    if len(parts) != expected_parts:
        raise ValueError(f"应有 {expected_parts} 个空间块，实际只有 {len(parts)} 个")
    expected_values = CHUNK_T * CHUNK_Y * CHUNK_X
    expected_bytes = expected_values * np.dtype("<i2").itemsize
    zst = Zstd()
    rows: list[np.ndarray] = []
    for row_id in range(len(y_chunks)):
        row: list[np.ndarray] = []
        for col_id in range(len(x_chunks)):
            part_id = row_id * len(x_chunks) + col_id
            blob = parts[part_id]
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise TypeError(f"第 {part_id} 个 AORC 块不是字节串")
            try:
                raw = zst.decode(blob)
            except Exception as exc:
                raise ValueError(f"第 {part_id} 个 AORC 块无法解压") from exc
            if len(raw) != expected_bytes:
                raise ValueError(
                    f"第 {part_id} 个 AORC 块解压后为 {len(raw)} 字节，"
                    f"应为 {expected_bytes} 字节")
            row.append(np.frombuffer(raw, dtype="<i2").reshape(
                CHUNK_T, CHUNK_Y, CHUNK_X))
        rows.append(np.concatenate(row, axis=2))
    return np.concatenate(rows, axis=1)


def _overlap_weights(src: np.ndarray, half_src: float,
                     dst: np.ndarray, half_dst: float,
                     cos_lat: np.ndarray | None = None) -> csr_matrix:
    """构造一条坐标轴上“源格 → 目标格”的边界重叠权重。

    每个目标格点的边界由 ``center ± half_dst`` 给出，只访问真正重叠的源格。
    纬向调用会额外乘 ``cos(latitude)``，近似补偿经纬网格随纬度变化的真实面积；
    经向调用只用重叠长度。每个目标行最后归一到 1。
    """
    rows, cols, vals = [], [], []
    for dst_id, center in enumerate(dst):
        lo, hi = center - half_dst, center + half_dst
        first = int(np.searchsorted(src + half_src, lo, side="right"))
        last = int(np.searchsorted(src - half_src, hi, side="left"))
        for src_id in range(max(first, 0), min(last + 1, len(src))):
            overlap = min(hi, src[src_id] + half_src) - max(
                lo, src[src_id] - half_src)
            if overlap <= 0:
                continue
            weight = overlap * (cos_lat[src_id] if cos_lat is not None else 1.0)
            rows.append(dst_id)
            cols.append(src_id)
            vals.append(weight)
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(dst), len(src)))
    total = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(total <= 0):
        missing = np.flatnonzero(total <= 0)
        raise ValueError(f"{len(missing)} 个目标格点未被源网格覆盖")
    return csr_matrix(matrix.multiply(1.0 / total[:, None]))


def assert_strictly_increasing(coord: np.ndarray, name: str,
                               expected_step: float | None = None) -> None:
    """拒绝多维、倒序、重复或步长不符合契约的空间坐标。"""
    values = np.asarray(coord)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} 必须是一维且至少含两个坐标值")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 必须是有限数值坐标")
    steps = np.diff(values.astype(np.float64))
    if np.any(steps <= 0):
        raise ValueError(f"{name} 必须严格递增且不能重复")
    if expected_step is not None and not np.allclose(
            steps, expected_step, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} 步长必须为 {expected_step} 度")


def build_regridder(aorc_lat: np.ndarray, aorc_lon: np.ndarray,
                    target_lat: np.ndarray, target_lon: np.ndarray
                    ) -> Tuple[csr_matrix, csr_matrix]:
    assert_strictly_increasing(aorc_lat, "AORC latitude")
    assert_strictly_increasing(aorc_lon, "AORC longitude")
    assert_strictly_increasing(target_lat, "目标 latitude", TARGET_STEP)
    assert_strictly_increasing(target_lon, "目标 longitude", TARGET_STEP)
    cos_lat = np.cos(np.deg2rad(aorc_lat))
    wy = _overlap_weights(aorc_lat, 1 / 240.0, target_lat, 0.005, cos_lat)
    wx = _overlap_weights(aorc_lon, 1 / 240.0, target_lon, 0.005)
    return wy, wx


def _apply_weights(slab: np.ndarray, wy_dense: np.ndarray,
                   wx_dense: np.ndarray) -> np.ndarray:
    """把可分离权重用于 ``(time, lat, lon)`` 数组，并保持同样的轴顺序。

    先沿经度、再沿纬度收缩，等价于二维面积权重的张量积，却不需要建立一个
    巨大的二维重采样矩阵。第二次 tensordot 会把纬度放到最前，最后再挪回中间。
    """
    out = np.tensordot(slab, wx_dense, axes=([2], [1]))
    out = np.tensordot(wy_dense, out, axes=([1], [1]))
    return np.moveaxis(out, 0, 1)


def apply_regrid(slab: np.ndarray, wy: csr_matrix,
                 wx: csr_matrix) -> np.ndarray:
    """面积重采样；合法缺测只从有效源格点重新归一，绝不按 0 处理。

    分子对“有效数值（缺测处临时填 0）”加权，分母对“是否有效”用完全相同
    的权重加权，最后两者相除。这样 NaN 同时从分子和分母移除：只按剩余的
    有效源面积取平均，而不会把资料缺口解释成零降雨。
    """
    values = np.asarray(slab, dtype=np.float32)
    valid = np.isfinite(values)
    wy_dense = np.asarray(wy.todense(), dtype=np.float64)
    wx_dense = np.asarray(wx.todense(), dtype=np.float64)
    numerator = _apply_weights(np.where(valid, values, 0.0), wy_dense, wx_dense)
    valid_weight = _apply_weights(valid.astype(np.float32), wy_dense, wx_dense)
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(numerator, valid_weight, out=result, where=valid_weight > 0)
    return result


def expected_year_times(year: int) -> np.ndarray:
    start = np.datetime64(f"{year:04d}-01-01T00:00:00", "s")
    end = np.datetime64(f"{year + 1:04d}-01-01T00:00:00", "s")
    return np.arange(start, end, np.timedelta64(1, "h")).astype(
        "datetime64[s]").astype(np.int64)


def _validate_precipitation(rain: np.ndarray, label: str,
                            min_valid_fraction: float = MIN_VALID_FRACTION) -> None:
    """逐块拒绝整时次缺测、负降雨和有效格点比例过低的产物。"""
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("最低有效格点比例必须在 (0, 1] 内")
    total = int(np.prod(rain.shape, dtype=np.int64))
    valid_total = 0
    min_fraction = 1.0
    for lo in range(0, rain.shape[0], VALIDATION_CHUNK_HOURS):
        hi = min(lo + VALIDATION_CHUNK_HOURS, rain.shape[0])
        block = np.asarray(rain[lo:hi])
        finite = np.isfinite(block)
        if np.isinf(block).any():
            raise ValueError(f"{label} 含有正负无穷")
        if np.any(block[finite] < 0):
            raise ValueError(f"{label} 含有负降雨")
        valid_per_hour = finite.reshape(len(block), -1).sum(axis=1)
        fractions = valid_per_hour / finite[0].size
        empty = np.flatnonzero(valid_per_hour == 0)
        if len(empty):
            raise ValueError(f"{label} 第 {lo + int(empty[0])} 小时全部缺测")
        low = np.flatnonzero(fractions < min_valid_fraction)
        if len(low):
            index = int(low[0])
            raise ValueError(
                f"{label} 第 {lo + index} 小时有效格点比例 "
                f"{fractions[index]:.4f} 低于 {min_valid_fraction:.4f}")
        valid_total += int(valid_per_hour.sum())
        min_fraction = min(min_fraction, float(fractions.min()))
    if valid_total == 0:
        raise ValueError(f"{label} 全年全部缺测")
    overall = valid_total / total
    if overall < min_valid_fraction:
        raise ValueError(
            f"{label} 全年有效格点比例 {overall:.4f} 低于 {min_valid_fraction:.4f}")


def validate_annual_file(path: Path, year: int,
                         expected_lat: np.ndarray | None = None,
                         expected_lon: np.ndarray | None = None) -> None:
    """重开并验证一个完整自然年的 float32 年度文件。"""
    expected_time = expected_year_times(year)
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"apcp", "time", "lat", "lon"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"缺少字段 {sorted(missing)}")
            rain = data["apcp"]
            times = data["time"]
            lat = data["lat"]
            lon = data["lon"]
            if rain.dtype != np.float32:
                raise ValueError(f"apcp 必须是 float32，实际为 {rain.dtype}")
            if rain.shape != (len(expected_time), len(lat), len(lon)):
                raise ValueError(
                    f"apcp 形状 {rain.shape} 与完整年度/坐标不符")
            times_int = times.astype(np.int64)
            if not np.array_equal(times_int, expected_time):
                raise ValueError(f"time 没有完整覆盖 {year} 年每一个整点")
            if not np.issubdtype(lat.dtype, np.number) or not np.issubdtype(lon.dtype, np.number):
                raise ValueError("lat/lon 必须是数值坐标")
            assert_strictly_increasing(lat, "年度 lat", TARGET_STEP)
            assert_strictly_increasing(lon, "年度 lon", TARGET_STEP)
            if expected_lat is not None and not np.array_equal(lat, expected_lat):
                raise ValueError("lat 与目标网格不一致")
            if expected_lon is not None and not np.array_equal(lon, expected_lon):
                raise ValueError("lon 与目标网格不一致")
            _validate_precipitation(rain, "apcp")
    except Exception as exc:
        raise ValueError(f"年度文件校验失败：{path}：{exc}") from exc


def target_grid(grid_nc: Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """读取目标网格，并拒绝未完成半格修正或数值方向错误的 MRMS。"""
    if grid_nc is None:
        grid_nc = ROOT / "data" / "rain_mrms_1km.nc"
    with xr.open_dataset(grid_nc) as dataset:
        crs_value = dataset.attrs.get("crs")
        if crs_value is None:
            raise ValueError("目标 MRMS 文件必须显式声明 crs")
        crs = str(crs_value)
        lat = dataset["lat"].values.astype(np.float64)
        lon = dataset["lon"].values.astype(np.float64)
        signature = grid_signature(lat, lon, crs)
        stored = dataset.attrs.get("grid_signature")
        if stored is None or str(stored) != signature:
            raise ValueError("目标 MRMS 文件缺少正确的网格指纹")
        if crs != "EPSG:4326":
            raise ValueError(f"目标 MRMS 坐标系必须是 EPSG:4326，实际为 {crs}")
        if signature != EXPECTED_TARGET_GRID_SIGNATURE:
            raise ValueError(
                "目标 MRMS 网格不是已修正的 128×128 流域网格；"
                f"实际为 {signature}")
        return lat, lon


def _chunk_ids(lo: int, hi: int, chunk_size: int) -> list[int]:
    return list(range(lo // chunk_size, (hi - 1) // chunk_size + 1))


def fetch_year(year: int, out_dir: Path, reader: AorcReader,
               force: bool = False, grid_nc: Path | None = None) -> bool:
    target_lat, target_lon = target_grid(grid_nc)
    assert_strictly_increasing(target_lat, "目标 latitude", TARGET_STEP)
    assert_strictly_increasing(target_lon, "目标 longitude", TARGET_STEP)
    output = out_dir / f"aorc_{year}.npz"
    if output.exists() and not force:
        validate_annual_file(output, year, target_lat, target_lon)
        print(f"{year} 已存在且校验通过，跳过")
        return True

    source_lat = reader.coord(year, "latitude")
    source_lon = reader.coord(year, "longitude")
    source_time = reader.coord(year, "time")
    assert_strictly_increasing(source_lat, "AORC latitude")
    assert_strictly_increasing(source_lon, "AORC longitude")
    expected_time = expected_year_times(year)
    source_time_int = source_time.astype(np.int64)
    if not np.array_equal(source_time_int, expected_time):
        raise ValueError(f"AORC 源 time 没有完整覆盖 {year} 年每一个整点")

    # 目标范围向四周多取少量源格，再换成覆盖它的完整 Zarr 块编号。
    # 多取的边缘可避免浮点边界误差造成目标格没有完整源覆盖。
    y0 = int(np.searchsorted(source_lat, target_lat.min() - 0.02)) - 1
    y1 = int(np.searchsorted(source_lat, target_lat.max() + 0.02)) + 1
    x0 = int(np.searchsorted(source_lon, target_lon.min() - 0.02)) - 1
    x1 = int(np.searchsorted(source_lon, target_lon.max() + 0.02)) + 1
    if y0 < 0 or x0 < 0 or y1 > len(source_lat) or x1 > len(source_lon):
        raise ValueError("目标网格超出 AORC 源网格覆盖范围")
    y_chunks = _chunk_ids(y0, y1, CHUNK_Y)
    x_chunks = _chunk_ids(x0, x1, CHUNK_X)
    origin_y, origin_x = y_chunks[0] * CHUNK_Y, x_chunks[0] * CHUNK_X
    wy, wx = build_regridder(source_lat[y0:y1], source_lon[x0:x1],
                             target_lat, target_lon)

    n_time = len(source_time)
    n_time_chunks = (n_time + CHUNK_T - 1) // CHUNK_T
    n_spatial = len(y_chunks) * len(x_chunks)
    per_batch = max(1, reader.threads // n_spatial)
    output_array = np.empty(
        (n_time, len(target_lat), len(target_lon)), dtype=np.float32)
    started = _time.time()
    for batch_start in range(0, n_time_chunks, per_batch):
        batch = list(range(batch_start, min(batch_start + per_batch,
                                           n_time_chunks)))
        # 任务顺序固定为“时间块 → y 块 → x 块”。reader.chunks 保持输入顺序，
        # 所以每个时间块连续的 n_spatial 个返回块可直接交给 assemble；assemble
        # 再按 y 为外层、x 为内层拼回整块。per_batch 保证一批请求数不超过线程数。
        jobs = [(chunk, cy, cx) for chunk in batch
                for cy in y_chunks for cx in x_chunks]
        parts = reader.chunks(year, jobs)
        for batch_id, chunk in enumerate(batch):
            block = assemble(parts[batch_id * n_spatial:(batch_id + 1) * n_spatial],
                             y_chunks, x_chunks).astype(np.float32)
            block[block == FILL] = np.nan
            # origin 是首个完整 Zarr 块的全局左上索引；减掉它，把源网格的
            # 全局 y0:y1/x0:x1 换成刚拼好 block 内的局部切片，去掉块对齐多下的边。
            source_subset = block[:, y0 - origin_y:y1 - origin_y,
                                  x0 - origin_x:x1 - origin_x] * SCALE
            regridded = apply_regrid(source_subset, wy, wx)
            lo = chunk * CHUNK_T
            hi = min((chunk + 1) * CHUNK_T, n_time)
            output_array[lo:hi] = regridded[:hi - lo]
        done = min((batch[-1] + 1) * CHUNK_T, n_time)
        elapsed = _time.time() - started
        remaining = elapsed / max(done, 1) * (n_time - done) / 60
        print(f"  {year} {done:5d}/{n_time} 小时  已用 {elapsed/60:5.1f} 分  "
              f"预计还需 {remaining:5.1f} 分", flush=True)

    staging = output.with_name(output.name + ".staging")
    try:
        with open(staging, "wb") as handle:
            np.savez_compressed(handle, apcp=output_array, time=source_time,
                                lat=target_lat, lon=target_lon)
            handle.flush()
            os.fsync(handle.fileno())
        validate_annual_file(staging, year, target_lat, target_lon)
        os.replace(staging, output)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    print(f"{year} 完成 -> {output}  ({output.stat().st_size/1e6:.0f} MB)")
    return True


def verify(reader: AorcReader, grid_nc: Path | None = None) -> None:
    """保留人工对拍入口；使用与正式下载相同的缺测归一和严格下载规则。"""
    target_lat, target_lon = target_grid(grid_nc)
    source_lat = reader.coord(2024, "latitude")
    source_lon = reader.coord(2024, "longitude")
    source_time = reader.coord(2024, "time")
    assert_strictly_increasing(target_lat, "目标 latitude", TARGET_STEP)
    assert_strictly_increasing(target_lon, "目标 longitude", TARGET_STEP)
    assert_strictly_increasing(source_lat, "AORC latitude")
    assert_strictly_increasing(source_lon, "AORC longitude")
    y0 = int(np.searchsorted(source_lat, target_lat.min() - 0.02)) - 1
    y1 = int(np.searchsorted(source_lat, target_lat.max() + 0.02)) + 1
    x0 = int(np.searchsorted(source_lon, target_lon.min() - 0.02)) - 1
    x1 = int(np.searchsorted(source_lon, target_lon.max() + 0.02)) + 1
    y_chunks = _chunk_ids(y0, y1, CHUNK_Y)
    x_chunks = _chunk_ids(x0, x1, CHUNK_X)
    origin_y, origin_x = y_chunks[0] * CHUNK_Y, x_chunks[0] * CHUNK_X
    wy, wx = build_regridder(source_lat[y0:y1], source_lon[x0:x1],
                             target_lat, target_lon)

    start = np.datetime64("2024-09-01T00", "s").astype(np.int64)
    stop = np.datetime64("2024-10-16T00", "s").astype(np.int64)
    t0 = int(np.searchsorted(source_time, start))
    t1 = int(np.searchsorted(source_time, stop))
    chunks = list(range(t0 // CHUNK_T, (t1 - 1) // CHUNK_T + 1))
    n_spatial = len(y_chunks) * len(x_chunks)
    parts = reader.chunks(2024, [(chunk, cy, cx) for chunk in chunks
                                 for cy in y_chunks for cx in x_chunks])
    rows = []
    for chunk_id, _chunk in enumerate(chunks):
        block = assemble(parts[chunk_id * n_spatial:(chunk_id + 1) * n_spatial],
                         y_chunks, x_chunks).astype(np.float32)
        block[block == FILL] = np.nan
        subset = block[:, y0 - origin_y:y1 - origin_y,
                       x0 - origin_x:x1 - origin_x] * SCALE
        rows.append(apply_regrid(subset, wy, wx))
    offset = t0 - chunks[0] * CHUNK_T
    regridded = np.concatenate(rows)[offset:offset + (t1 - t0)]
    times = source_time[t0:t1]

    mrms_path = grid_nc or ROOT / "data" / "rain_mrms_1km.nc"
    with xr.open_dataset(mrms_path) as dataset:
        subset = dataset["rain"].sel(time=slice("2024-09-01", "2024-10-15 23:00"))
        mrms = subset.values.astype(np.float32)
        mrms_time = subset["time"].values.astype("datetime64[s]").astype(np.int64)
    if not np.array_equal(times, mrms_time):
        raise ValueError("对拍窗口的 AORC 与 MRMS 时间轴不一致")
    a = np.nanmean(regridded, axis=(1, 2))
    b = np.nanmean(mrms, axis=(1, 2))
    valid = np.isfinite(a) & np.isfinite(b)
    print(f"\n对拍窗口 {len(times)} 小时，时间轴全部对齐")
    print(f"  面雨量合计  AORC {np.nansum(a):8.1f} mm   "
          f"MRMS {np.nansum(b):8.1f} mm   比值 {np.nansum(a)/np.nansum(b):.3f}")
    print(f"  逐小时相关 {np.corrcoef(a[valid], b[valid])[0, 1]:.4f}")


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并重采样 AORC 逐小时降水")
    parser.add_argument("--start-year", type=int, default=REQUIRED_FIRST_YEAR)
    parser.add_argument("--end-year", type=int, default=REQUIRED_LAST_YEAR)
    parser.add_argument("--out-dir", default="data/aorc_regrid_float32")
    parser.add_argument("--grid-nc", default="data/rain_mrms_1km.nc")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year 不能晚于 --end-year")

    output_dir = _resolve(args.out_dir)
    grid_nc = _resolve(args.grid_nc)
    output_dir.mkdir(parents=True, exist_ok=True)
    with AorcReader(args.threads) as reader:
        if args.verify:
            verify(reader, grid_nc)
            return
        for year in range(args.start_year, args.end_year + 1):
            print(f"=== {year} ===", flush=True)
            fetch_year(year, output_dir, reader, force=args.force,
                       grid_nc=grid_nc)
    print("全部完成")


if __name__ == "__main__":
    main()
