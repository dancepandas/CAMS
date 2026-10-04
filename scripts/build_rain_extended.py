#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把完整的 1990--2015 AORC 年度文件与 MRMS 尾段拼成 NetCDF。

大文件按时间分块写入 staging 文件，不在内存里建立约 20 GB 的总数组。写完后
重开逐块验证，确认时间、网格、float32 类型和 MRMS 尾段逐点一致，再用
``os.replace`` 原子发布。脚本只读配置，绝不改写配置文件。
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from spatial_grid import grid_signature

ROOT = Path(__file__).resolve().parents[1]
FIRST_YEAR = 1990
LAST_YEAR = 2015
START = pd.Timestamp("1990-01-01 00:00")
SPLIT = pd.Timestamp("2015-06-01 00:00")
DEFAULT_CHUNK_HOURS = 168
TIME_UNITS = "hours since 1970-01-01 00:00:00"
TIME_CALENDAR = "proleptic_gregorian"


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def expected_year_seconds(year: int) -> np.ndarray:
    start = np.datetime64(f"{year:04d}-01-01T00:00:00", "s")
    end = np.datetime64(f"{year + 1:04d}-01-01T00:00:00", "s")
    return np.arange(start, end, np.timedelta64(1, "h")).astype(
        "datetime64[s]").astype(np.int64)


def expected_aorc_files(aorc_dir: Path) -> list[Path]:
    """1990--2015 每年必须恰好有一个命名正确的年度文件。"""
    expected = [aorc_dir / f"aorc_{year}.npz"
                for year in range(FIRST_YEAR, LAST_YEAR + 1)]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{aorc_dir} 缺少 1990--2015 完整年份文件：{', '.join(missing)}")
    return expected


def assert_strictly_increasing(coord: np.ndarray, name: str) -> None:
    values = np.asarray(coord)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} 必须是一维且至少含两个坐标值")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 必须是有限数值坐标")
    if np.any(np.diff(values.astype(np.float64)) <= 0):
        raise ValueError(f"{name} 必须严格递增且不能重复")


def inspect_aorc_files(aorc_dir: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """验证全部年度文件，不把年度降水一起装入内存。"""
    files = expected_aorc_files(aorc_dir)
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    for year, path in zip(range(FIRST_YEAR, LAST_YEAR + 1), files):
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
                expected_time = expected_year_seconds(year)
                expected_shape = (len(expected_time), len(lat), len(lon))
                if rain.dtype != np.float32:
                    raise ValueError(f"apcp 必须是 float32，实际为 {rain.dtype}")
                if rain.shape != expected_shape:
                    raise ValueError(f"apcp 形状 {rain.shape}，应为 {expected_shape}")
                # AORC 源 time 坐标本来就是 float64 的 Unix 秒。年度下载器会保留
                # 该原始 dtype，因此这里不能只接受整数数组；先要求全部是可精确表示
                # 的有限整秒，再转成 int64 与完整自然年逐点比较。
                if (not np.issubdtype(times.dtype, np.number)
                        or not np.all(np.isfinite(times))):
                    raise ValueError("time 必须是有限数值的 Unix 秒")
                times_int = times.astype(np.int64)
                if (not np.array_equal(times.astype(np.float64),
                                       times_int.astype(np.float64))
                        or not np.array_equal(times_int, expected_time)):
                    raise ValueError(f"time 没有完整覆盖 {year} 年每一个整点")
                if np.isinf(rain).any():
                    raise ValueError("apcp 含有正负无穷")
                assert_strictly_increasing(lat, "年度 lat")
                assert_strictly_increasing(lon, "年度 lon")
                if reference_lat is None:
                    reference_lat = np.array(lat, copy=True)
                    reference_lon = np.array(lon, copy=True)
                elif not (np.array_equal(reference_lat, lat)
                          and np.array_equal(reference_lon, lon)):
                    raise ValueError("网格坐标与其它年份不一致")
            print(f"  校验 {path.name}：{len(expected_time):5d} 小时，float32")
        except Exception as exc:
            raise ValueError(f"AORC 年度文件校验失败：{path}：{exc}") from exc
    assert reference_lat is not None and reference_lon is not None
    return files, reference_lat, reference_lon


def _source_time_seconds(dataset: xr.Dataset) -> np.ndarray:
    values = dataset["time"].values.astype("datetime64[s]").astype(np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError("MRMS time 必须是一维非空坐标")
    return values


def _validate_hourly(times: np.ndarray, label: str) -> None:
    if len(times) > 1:
        bad = np.flatnonzero(np.diff(times) != 3600)
        if len(bad):
            raise ValueError(
                f"{label} 时间轴有 {len(bad)} 处不是 1 小时间隔，"
                f"首处索引为 {int(bad[0])}")


def validate_mrms(dataset: xr.Dataset, lat: np.ndarray,
                  lon: np.ndarray) -> tuple[np.ndarray, int]:
    if "rain" not in dataset:
        raise ValueError("MRMS 文件缺少 rain 变量")
    rain = dataset["rain"]
    if rain.dims != ("time", "lat", "lon"):
        raise ValueError(f"MRMS rain 维度应为 (time, lat, lon)，实际为 {rain.dims}")
    if rain.dtype != np.float32:
        raise ValueError(f"MRMS rain 必须是 float32，实际为 {rain.dtype}")
    assert_strictly_increasing(dataset["lat"].values, "MRMS lat")
    assert_strictly_increasing(dataset["lon"].values, "MRMS lon")
    crs_value = dataset.attrs.get("crs")
    if crs_value is None:
        raise ValueError("MRMS 必须显式声明 CRS")
    signature = grid_signature(dataset["lat"].values, dataset["lon"].values,
                               str(crs_value))
    if dataset.attrs.get("grid_signature") != signature:
        raise ValueError("MRMS 网格指纹缺失或与坐标不一致")
    if not (np.array_equal(dataset["lat"].values, lat)
            and np.array_equal(dataset["lon"].values, lon)):
        raise ValueError("AORC 与 MRMS 的网格坐标不一致")
    times = _source_time_seconds(dataset)
    _validate_hourly(times, "MRMS")
    split_seconds = int(SPLIT.to_datetime64().astype("datetime64[s]").astype(np.int64))
    start = int(np.searchsorted(times, split_seconds))
    if start >= len(times) or int(times[start]) != split_seconds:
        raise ValueError(f"MRMS 时间轴中没有拼接点 {SPLIT}")
    return times, start


def _hours_since_epoch(seconds: np.ndarray) -> np.ndarray:
    if np.any(seconds % 3600 != 0):
        raise ValueError("时间轴含有非整点时间")
    return seconds // 3600


def _iter_ranges(start: int, stop: int,
                 chunk_hours: int) -> Iterator[tuple[int, int]]:
    for lo in range(start, stop, chunk_hours):
        yield lo, min(lo + chunk_hours, stop)


def write_staging(staging: Path, aorc_files: Sequence[Path],
                  mrms: xr.Dataset, mrms_start: int,
                  lat: np.ndarray, lon: np.ndarray,
                  chunk_hours: int) -> int:
    """分块写出待校验的扩展降雨文件，并返回总小时数。

    时间规则是严格的半开区间：AORC 提供
    ``[1990-01-01 00:00, 2015-06-01 00:00)``，MRMS 从
    ``2015-06-01 00:00`` 开始提供到末尾，拼接点只出现一次。``destination``
    始终指向输出中的下一空位，并分别在 AORC 段尾和文件末尾核对计数。

    NPZ 压缩格式无法真正按块映射，因此每个 AORC 年度数组会一次解压到内存；
    写入约 20 GB 的 NetCDF 时仍按 ``chunk_hours`` 切块，避免建立完整总数组。
    """
    aorc_stop_seconds = int(SPLIT.to_datetime64().astype(
        "datetime64[s]").astype(np.int64))
    aorc_hours = int((SPLIT - START) / pd.Timedelta(hours=1))
    mrms_hours = int(mrms.sizes["time"] - mrms_start)
    total_hours = aorc_hours + mrms_hours

    staging.unlink(missing_ok=True)
    with netCDF4.Dataset(staging, "w", format="NETCDF4") as output:
        output.createDimension("time", total_hours)
        output.createDimension("lat", len(lat))
        output.createDimension("lon", len(lon))
        time_var = output.createVariable("time", "i8", ("time",))
        time_var.units = TIME_UNITS
        time_var.calendar = TIME_CALENDAR
        lat_var = output.createVariable("lat", lat.dtype, ("lat",))
        lon_var = output.createVariable("lon", lon.dtype, ("lon",))
        rain_var = output.createVariable(
            "rain", "f4", ("time", "lat", "lon"),
            chunksizes=(min(chunk_hours, total_hours), len(lat), len(lon)),
            zlib=False, fill_value=np.float32(np.nan))
        rain_var.units = "mm/h"
        lat_var[:] = lat
        lon_var[:] = lon
        crs_value = mrms.attrs.get("crs")
        if crs_value is None:
            raise ValueError("MRMS 必须显式声明 CRS")
        crs = str(crs_value)
        output.units = "mm/h"
        output.crs = crs
        output.grid_signature = grid_signature(lat, lon, crs)
        output.source = (f"{START:%Y-%m}~{SPLIT:%Y-%m} 取 AORC v1.1；"
                         f"{SPLIT:%Y-%m} 起取原 MRMS 1 小时 QPE")
        output.splice = str(SPLIT)
        output.start = str(START)

        destination = 0
        for year, path in zip(range(FIRST_YEAR, LAST_YEAR + 1), aorc_files):
            with np.load(path, allow_pickle=False) as data:
                source_time = data["time"]
                source_rain = data["apcp"]  # 每年只解压一次；NPZ 不能真正按块 mmap。
                stop = (len(source_time) if year < LAST_YEAR else
                        int(np.searchsorted(source_time, aorc_stop_seconds)))
                for lo, hi in _iter_ranges(0, stop, chunk_hours):
                    block = np.asarray(source_rain[lo:hi], dtype=np.float32)
                    count = hi - lo
                    rain_var[destination:destination + count] = block
                    time_var[destination:destination + count] = _hours_since_epoch(
                        source_time[lo:hi])
                    destination += count
            print(f"  已写 AORC {year}，累计 {destination} 小时", flush=True)
        if destination != aorc_hours:
            raise ValueError(f"AORC 实写 {destination} 小时，应为 {aorc_hours} 小时")

        mrms_times = _source_time_seconds(mrms)
        source_rain = mrms["rain"]
        for lo, hi in _iter_ranges(mrms_start, int(mrms.sizes["time"]), chunk_hours):
            count = hi - lo
            rain_var[destination:destination + count] = np.asarray(
                source_rain.isel(time=slice(lo, hi)).values, dtype=np.float32)
            time_var[destination:destination + count] = _hours_since_epoch(
                mrms_times[lo:hi])
            destination += count
            print(f"  已写 MRMS {hi - mrms_start}/{mrms_hours} 小时", flush=True)
        if destination != total_hours:
            raise ValueError(f"总计实写 {destination} 小时，应为 {total_hours} 小时")
        output.sync()
    return total_hours


def _array_equal_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left, right, equal_nan=True)


def validate_staging(staging: Path, mrms: xr.Dataset, mrms_start: int,
                     lat: np.ndarray, lon: np.ndarray,
                     total_hours: int, chunk_hours: int) -> None:
    """重开产物，并逐块确认 MRMS 尾段每个值和每个缺测点都没变。"""
    with xr.open_dataset(staging, decode_times=True) as output:
        rain = output["rain"]
        if rain.dtype != np.float32:
            raise ValueError(f"产物 rain 必须是 float32，实际为 {rain.dtype}")
        if rain.shape != (total_hours, len(lat), len(lon)):
            raise ValueError(f"产物 rain 形状错误：{rain.shape}")
        if not (np.array_equal(output["lat"].values, lat)
                and np.array_equal(output["lon"].values, lon)):
            raise ValueError("产物网格坐标重开后不一致")
        crs_value = mrms.attrs.get("crs")
        if crs_value is None:
            raise ValueError("MRMS 必须显式声明 CRS")
        expected_crs = str(crs_value)
        if str(output.attrs.get("crs", "")) != expected_crs:
            raise ValueError("产物 CRS 与 MRMS 不一致")
        if output.attrs.get("grid_signature") != grid_signature(lat, lon, expected_crs):
            raise ValueError("产物网格指纹与坐标不一致")
        output_time = output["time"].values.astype("datetime64[s]").astype(np.int64)
        expected_start = int(START.to_datetime64().astype(
            "datetime64[s]").astype(np.int64))
        if int(output_time[0]) != expected_start:
            raise ValueError("产物时间轴起点不是 1990-01-01 00:00")
        _validate_hourly(output_time, "产物")

        mrms_hours = int(mrms.sizes["time"] - mrms_start)
        aorc_hours = total_hours - mrms_hours
        source_rain = mrms["rain"]
        for lo, hi in _iter_ranges(mrms_start, int(mrms.sizes["time"]), chunk_hours):
            out_lo = aorc_hours + lo - mrms_start
            out_hi = out_lo + hi - lo
            tail = np.asarray(rain.isel(time=slice(out_lo, out_hi)).values,
                              dtype=np.float32)
            original = np.asarray(source_rain.isel(time=slice(lo, hi)).values,
                                  dtype=np.float32)
            if not _array_equal_with_nan(tail, original):
                raise ValueError(
                    f"MRMS 尾段逐点不一致，源时间索引 [{lo}, {hi})")
        if not np.array_equal(output_time[aorc_hours:],
                              _source_time_seconds(mrms)[mrms_start:]):
            raise ValueError("MRMS 尾段时间轴与源文件不一致")


def backup_existing_output(output_nc: Path) -> Path | None:
    """用同卷硬链接保留旧正式件；不允许为 20 GB 文件静默退化成复制。"""
    if not output_nc.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = output_nc.with_name(f"{output_nc.name}.bak.{stamp}")
    try:
        os.link(output_nc, backup)
    except OSError as exc:
        raise RuntimeError(
            f"无法为已有产物创建同卷硬链接备份：{output_nc} -> {backup}。"
            "为避免复制约 20 GB 文件，已停止发布；请确认文件系统支持硬链接且有权限。"
        ) from exc
    print(f"已保留旧正式件硬链接备份：{backup}")
    return backup


def build(aorc_dir: Path, mrms_nc: Path, output_nc: Path,
          chunk_hours: int = DEFAULT_CHUNK_HOURS) -> None:
    if chunk_hours < 1:
        raise ValueError("chunk-hours 必须大于 0")
    files, lat, lon = inspect_aorc_files(aorc_dir)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    staging = output_nc.with_name(output_nc.name + ".staging")
    try:
        with xr.open_dataset(mrms_nc) as mrms:
            _mrms_times, mrms_start = validate_mrms(mrms, lat, lon)
            total_hours = write_staging(staging, files, mrms, mrms_start,
                                        lat, lon, chunk_hours)
        # 关闭并重新打开源和产物，确保校验不是依赖写入时缓存。
        with xr.open_dataset(mrms_nc) as mrms:
            _mrms_times, mrms_start = validate_mrms(mrms, lat, lon)
            validate_staging(staging, mrms, mrms_start, lat, lon,
                             total_hours, chunk_hours)
        backup_existing_output(output_nc)
        os.replace(staging, output_nc)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    print(f"\n已原子发布 {output_nc}  {output_nc.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="拼接 AORC 与 MRMS 降雨")
    parser.add_argument("--config", default="configs/pipeline_rain1990.yaml")
    parser.add_argument("--aorc-dir", default="data/aorc_regrid_float32")
    parser.add_argument("--mrms-nc", default=None,
                        help="原 MRMS 文件；默认读取基准配置 configs/pipeline.yaml")
    parser.add_argument("--output-nc", default=None,
                        help="输出文件；默认读取 --config 的 paths.rain_nc")
    parser.add_argument("--chunk-hours", type=int, default=DEFAULT_CHUNK_HOURS)
    args = parser.parse_args()

    config_path = _resolve(args.config)
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.mrms_nc is None:
        with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle)
        mrms_nc = _resolve(base_config["paths"]["rain_nc"])
    else:
        mrms_nc = _resolve(args.mrms_nc)
    output_nc = _resolve(args.output_nc or config["paths"]["rain_nc"])
    build(_resolve(args.aorc_dir), mrms_nc, output_nc, args.chunk_hours)


if __name__ == "__main__":
    main()
