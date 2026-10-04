#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练和推理入口共用的数据门禁与训练清单工具。"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from spatial_grid import grid_signature as spatial_grid_signature
except ImportError:  # 允许以 scripts.data_contract 包形式导入
    from .spatial_grid import grid_signature as spatial_grid_signature

EXPECTED_CRS = "EPSG:4326"
MANIFEST_FORMAT = "cams-training-manifest-v2"
RAIN_VALUES_FORMAT = "cams-rain-values-v1"
FLOW_VALUES_FORMAT = "cams-flow-aligned-v1"
RAIN_HASH_CHUNK_HOURS = 168


@dataclass(frozen=True)
class DataContract:
    ids: tuple[str, ...]
    names: tuple[str, ...]
    areas: np.ndarray
    times: pd.DatetimeIndex
    area_rain: np.ndarray
    flow: np.ndarray
    mask1km: np.ndarray
    rain_lat: np.ndarray
    rain_lon: np.ndarray
    crs: str
    signature: str
    component_sha256: Mapping[str, str]


def _path(root: os.PathLike[str] | str, value: os.PathLike[str] | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(v) for v in np.asarray(values).tolist())


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_array_digest(format_name: str, values: np.ndarray,
                            logical_dtype: str) -> str:
    """哈希数值和缺测位置；结果不受 NaN 二进制写法影响。"""
    array = np.asarray(values)
    if np.isinf(array).any():
        raise ValueError(f"{format_name} 含有正负无穷")
    dtype = np.dtype(logical_dtype).newbyteorder("<")
    finite = np.isfinite(array)
    value_digest = hashlib.sha256()
    value_digest.update(np.where(finite, array, 0).astype(
        dtype, copy=False).tobytes(order="C"))
    mask_digest = hashlib.sha256(finite.astype(np.uint8).tobytes(order="C"))
    document = {
        "format": format_name,
        "shape": list(array.shape),
        "dtype": dtype.str,
        "finite_sha256": mask_digest.hexdigest(),
        "values_sha256": value_digest.hexdigest(),
    }
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class RainValueHasher:
    """按时间块生成与块大小和 NaN 位模式无关的 rain 数值摘要。

    有限值统一转成 little-endian float32；缺测位置在数值流中填 0，同时另行
    哈希“是否有限”的 0/1 掩膜。最终摘要再绑定格式名、完整形状和逻辑类型。
    因而数值 0 与缺测仍能区分，不同 NaN 二进制写法也不会造成误差。调用者
    必须按时间顺序连续更新，并恰好覆盖构造时声明的所有小时。
    """

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.count = 0
        self.values = hashlib.sha256()
        self.finite = hashlib.sha256()

    def update(self, block: np.ndarray) -> None:
        values = np.asarray(block)
        if values.ndim != 3 or tuple(values.shape[1:]) != self.shape[1:]:
            raise ValueError("rain 摘要块的空间形状不一致")
        if np.isinf(values).any():
            raise ValueError("rain 含有正负无穷")
        finite = np.isfinite(values)
        canonical = np.where(finite, values, 0).astype("<f4", copy=False)
        self.values.update(canonical.tobytes(order="C"))
        self.finite.update(finite.astype(np.uint8).tobytes(order="C"))
        self.count += len(values)

    def hexdigest(self) -> str:
        if self.count != self.shape[0]:
            raise ValueError(f"rain 摘要只读到 {self.count}/{self.shape[0]} 个小时")
        document = {
            "format": RAIN_VALUES_FORMAT,
            "shape": list(self.shape),
            "dtype": "<f4",
            "finite_sha256": self.finite.hexdigest(),
            "values_sha256": self.values.hexdigest(),
        }
        return hashlib.sha256(json.dumps(
            document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rain_values_sha256(rain: Any, chunk_hours: int = RAIN_HASH_CHUNK_HOURS) -> str:
    if chunk_hours < 1:
        raise ValueError("rain 摘要块大小必须大于 0")
    hasher = RainValueHasher(rain.shape)
    for start in range(0, int(rain.shape[0]), int(chunk_hours)):
        stop = min(int(rain.shape[0]), start + int(chunk_hours))
        block = rain.isel(time=slice(start, stop)).values if hasattr(
            rain, "isel") else rain[start:stop]
        hasher.update(block)
    return hasher.hexdigest()


def _load_aligned_flow(paths: Mapping[str, Any], root: os.PathLike[str] | str,
                       ids: Sequence[str], times: pd.DatetimeIndex,
                       *, hash_files: bool = True
                       ) -> tuple[np.ndarray, Dict[str, str]]:
    flow = np.full((len(ids), len(times)), np.nan, dtype=np.float64)
    hashes: Dict[str, str] = {}
    directory = _path(root, paths["sites"])
    for i, sid in enumerate(ids):
        path = directory / f"{sid}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少断面流量文件：{path}")
        hashes[f"flow_csv/{sid}"] = (_sha256_file(path) if hash_files
                                      else "not-computed")
        table = pd.read_csv(path, index_col=0, parse_dates=True)
        if "flow_m3s" not in table.columns:
            raise ValueError(f"流量文件缺少 flow_m3s：{path}")
        index = pd.DatetimeIndex(table.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        if index.hasnans or not index.is_unique:
            raise ValueError(f"流量文件时间有空值或重复：{path}")
        series = pd.Series(pd.to_numeric(table["flow_m3s"], errors="raise").values,
                           index=index)
        flow[i] = series.reindex(times).to_numpy(dtype=np.float64)
    hashes["flow_aligned"] = _canonical_array_digest(
        FLOW_VALUES_FORMAT, flow, "<f8")
    return flow, hashes


def _normalise_crs(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.ravel()[0]
    text = str(value).strip().upper().replace(" ", "")
    aliases = {
        "EPSG:4326": EXPECTED_CRS,
        "EPSG4326": EXPECTED_CRS,
        "WGS84": EXPECTED_CRS,
        "WGS_84": EXPECTED_CRS,
        "OGC:CRS84": EXPECTED_CRS,
    }
    return aliases.get(text, text)


def _npz_crs(archive: Any) -> str | None:
    for key in ("crs", "spatial_ref", "crs_wkt"):
        if key in archive.files:
            return _normalise_crs(archive[key])
    return None


def _rain_crs(ds: Any, rain: Any) -> str | None:
    for attrs in (rain.attrs, ds.attrs):
        for key in ("crs", "spatial_ref", "crs_wkt"):
            if key in attrs:
                return _normalise_crs(attrs[key])
    grid_mapping = rain.attrs.get("grid_mapping")
    if grid_mapping and grid_mapping in ds:
        ref = ds[grid_mapping]
        for key in ("spatial_ref", "crs_wkt", "crs"):
            if key in ref.attrs:
                return _normalise_crs(ref.attrs[key])
    for key in ("spatial_ref", "crs"):
        if key in ds:
            ref = ds[key]
            for attr in ("spatial_ref", "crs_wkt", "crs"):
                if attr in ref.attrs:
                    return _normalise_crs(ref.attrs[attr])
            if ref.ndim == 0:
                return _normalise_crs(ref.values)
    return None


def _require_hourly_unique(times: pd.DatetimeIndex, label: str) -> None:
    if times.tz is not None:
        raise ValueError(f"{label} 时间轴必须是不带时区的 UTC 小时")
    if times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError(f"{label} 时间轴必须严格递增、无重复、无空值")
    if len(times) > 1:
        seconds = np.diff(times.values).astype("timedelta64[s]").astype(np.int64)
        if not np.all(seconds == 3600):
            bad = int(np.flatnonzero(seconds != 3600)[0])
            raise ValueError(f"{label} 时间轴不是逐小时连续：第 {bad} 到 {bad + 1} 点")


def validate_data_contract(cfg: Mapping[str, Any], root: os.PathLike[str] | str,
                           *, hash_files: bool = True) -> DataContract:
    """核对所有训练输入的空间、时间、站序和来源；不一致立即报错。

    门禁同时绑定站点表、汇水区、格点降雨、面雨量和逐站流量 CSV：核对站序、
    完整逐小时时间轴、网格、CRS、面雨量记录的源降雨数值摘要，以及流量对齐
    后的摘要。``catchments.npz`` 必须显式保存 ``lat/lon/crs``，不能从配置猜，
    否则半格偏移或纬度上下颠倒会被掩盖。

    ``hash_files=False`` 只供快速测试，可跳过部分文件字节哈希；正式训练和推理
    必须保留默认值 ``True``，让训练清单能发现输入文件被悄悄替换。
    """
    import xarray as xr

    paths = cfg["paths"]
    files = {
        "sites_csv": _path(root, paths["sites_csv"]),
        "catchments": _path(root, paths["catchments"]),
        "area_rain": _path(root, paths["area_rain"]),
        "rain_nc": _path(root, paths["rain_nc"]),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("数据门禁找不到文件：" + "，".join(missing))

    sites = pd.read_csv(files["sites_csv"], encoding="utf-8", dtype={"site_id": str})
    required = {"site_id", "name", "area_km2"}
    absent = required.difference(sites.columns)
    if absent:
        raise ValueError(f"站点表缺字段：{sorted(absent)}")
    ids = tuple(sites["site_id"].astype(str))
    names = tuple(sites["name"].astype(str))
    areas = sites["area_km2"].to_numpy(dtype=np.float64)
    if len(set(ids)) != len(ids):
        raise ValueError("站点表 site_id 有重复")

    with np.load(files["catchments"], allow_pickle=True) as catch:
        need = {"mask1km", "site_ids", "names", "area_obs", "lat", "lon",
                "crs", "mask_dims"}
        absent = need.difference(catch.files)
        if absent:
            raise ValueError(
                "catchment 缺少空间契约字段 " + str(sorted(absent))
                + "；必须显式写入 lat/lon/crs，不能靠配置猜")
        mask = np.asarray(catch["mask1km"], dtype=np.float32)
        catch_ids = _strings(catch["site_ids"])
        catch_names = _strings(catch["names"])
        catch_lat = np.asarray(catch["lat"], dtype=np.float64)
        catch_lon = np.asarray(catch["lon"], dtype=np.float64)
        catch_crs = _npz_crs(catch)
        catch_dims = _strings(catch["mask_dims"])
        if "area_obs" in catch.files and not np.allclose(
                np.asarray(catch["area_obs"], dtype=np.float64), areas,
                rtol=1e-6, atol=1e-4):
            raise ValueError("catchment 面积与站点表不一致")

    area_rain_path = files["area_rain"]
    with np.load(area_rain_path, allow_pickle=True) as archive:
        need = {"area_rain", "times", "site_ids", "names", "areas", "lat",
                "lon", "crs", "grid_signature", "source_rain_values_format",
                "source_rain_values_sha256"}
        absent = need.difference(archive.files)
        if absent:
            raise ValueError(f"area_rain 缺字段：{sorted(absent)}")
        area_rain = np.asarray(archive["area_rain"], dtype=np.float64)
        area_ids = _strings(archive["site_ids"])
        area_names = _strings(archive["names"])
        area_times = pd.DatetimeIndex([str(v) for v in archive["times"]])
        area_lat = np.asarray(archive["lat"], dtype=np.float64)
        area_lon = np.asarray(archive["lon"], dtype=np.float64)
        area_crs = _npz_crs(archive)
        area_grid_signature = str(np.asarray(archive["grid_signature"]).item())
        source_rain_format = str(np.asarray(
            archive["source_rain_values_format"]).item())
        source_rain_digest = str(np.asarray(
            archive["source_rain_values_sha256"]).item())
        if "areas" in archive.files and not np.allclose(
                np.asarray(archive["areas"], dtype=np.float64), areas,
                rtol=1e-6, atol=1e-4):
            raise ValueError("area_rain 面积与站点表不一致")

    if ids != catch_ids or ids != area_ids:
        raise ValueError("站点表、catchment、area_rain 的站号或顺序不一致")
    if names != catch_names or names != area_names:
        raise ValueError("站点表、catchment、area_rain 的站名或顺序不一致")
    if catch_dims != ("site", "lat", "lon"):
        raise ValueError(f"catchment mask_dims 必须是 ('site','lat','lon')，实际 {catch_dims}")
    if mask.ndim != 3 or mask.shape[0] != len(ids):
        raise ValueError(f"catchment shape {mask.shape} 与 {len(ids)} 个站不一致")
    if area_rain.ndim != 2 or area_rain.shape[0] != len(ids):
        raise ValueError(f"area_rain shape {area_rain.shape} 与 {len(ids)} 个站不一致")
    _require_hourly_unique(area_times, "area_rain")

    with xr.open_dataset(files["rain_nc"]) as ds:
        if "rain" not in ds:
            raise ValueError("rain_nc 缺 rain 变量")
        rain = ds["rain"]
        if tuple(rain.dims) != ("time", "lat", "lon"):
            raise ValueError(f"rain 维度必须是 (time, lat, lon)，实际为 {rain.dims}")
        rain_times = pd.DatetimeIndex(ds["time"].values)
        rain_lat = np.asarray(ds["lat"].values, dtype=np.float64)
        rain_lon = np.asarray(ds["lon"].values, dtype=np.float64)
        rain_crs = _rain_crs(ds, rain)
        rain_shape = tuple(int(v) for v in rain.shape)
        rain_dtype = str(rain.dtype)
        rain_values_digest = rain_values_sha256(rain)

    _require_hourly_unique(rain_times, "rain")
    if not area_times.equals(rain_times):
        if len(area_times) != len(rain_times):
            detail = f"长度 {len(area_times)} != {len(rain_times)}"
        else:
            bad = int(np.flatnonzero(area_times.values != rain_times.values)[0])
            detail = f"第 {bad} 点 {area_times[bad]} != {rain_times[bad]}"
        raise ValueError(f"area_rain 与 rain 时间没有逐点一致：{detail}")
    if area_rain.shape[1] != rain_shape[0]:
        raise ValueError("area_rain 时间长度与 rain 不一致")
    if mask.shape[1:] != rain_shape[1:]:
        raise ValueError(f"catchment 网格 {mask.shape[1:]} 与 rain 网格 {rain_shape[1:]} 不一致")
    if catch_lat.shape != rain_lat.shape or not np.allclose(catch_lat, rain_lat,
                                                            rtol=0.0, atol=1e-10):
        raise ValueError("catchment 纬度坐标与 rain 不逐点一致（可能上下颠倒或偏半格）")
    if area_lat.shape != rain_lat.shape or not np.allclose(area_lat, rain_lat,
                                                           rtol=0.0, atol=1e-10):
        raise ValueError("area_rain 纬度坐标与 rain 不逐点一致")
    if catch_lon.shape != rain_lon.shape or not np.allclose(catch_lon, rain_lon,
                                                            rtol=0.0, atol=1e-10):
        raise ValueError("catchment 经度坐标与 rain 不逐点一致（可能左右颠倒或偏半格）")
    if area_lon.shape != rain_lon.shape or not np.allclose(area_lon, rain_lon,
                                                           rtol=0.0, atol=1e-10):
        raise ValueError("area_rain 经度坐标与 rain 不逐点一致")
    if catch_crs is None or rain_crs is None or area_crs is None:
        raise ValueError("catchment、area_rain、rain 都必须显式声明 CRS")
    if catch_crs != rain_crs or area_crs != rain_crs or catch_crs != EXPECTED_CRS:
        raise ValueError(
            f"CRS 不一致或不是 {EXPECTED_CRS}：catchment={catch_crs}, "
            f"area_rain={area_crs}, rain={rain_crs}")
    expected_grid_signature = spatial_grid_signature(rain_lat, rain_lon, rain_crs)
    if area_grid_signature != expected_grid_signature:
        raise ValueError("area_rain 的 grid_signature 与当前 rain 网格不一致")
    if source_rain_format != RAIN_VALUES_FORMAT:
        raise ValueError(
            f"area_rain 的源降雨摘要格式应为 {RAIN_VALUES_FORMAT}，"
            f"实际为 {source_rain_format}")
    if source_rain_digest != rain_values_digest:
        raise ValueError("area_rain 不是由当前 rain 数值生成；请重新运行 area_rain.py")

    flow, flow_hashes = _load_aligned_flow(
        paths, root, ids, rain_times, hash_files=hash_files)
    hashes = ({key: _sha256_file(path) for key, path in files.items()
               if key != "rain_nc"}
              if hash_files else {key: "not-computed" for key in files
                                  if key != "rain_nc"})
    if not hash_files:
        flow_hashes["flow_aligned"] = "not-computed"
    rain_meta_doc = {
        "shape": rain_shape,
        "dtype": rain_dtype,
        "time_ns": hashlib.sha256(
            np.asarray(rain_times.values, dtype="datetime64[ns]").view("<i8").tobytes()
        ).hexdigest(),
        "grid_signature": spatial_grid_signature(rain_lat, rain_lon, rain_crs),
    }
    hashes["rain_nc_metadata"] = hashlib.sha256(json.dumps(
        rain_meta_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    hashes["rain_values"] = rain_values_digest
    hashes.update(flow_hashes)
    signature_doc = {
        "format": "cams-data-contract-v1",
        "ids": ids,
        "shape": {"rain": rain_shape, "catchment": mask.shape,
                  "area_rain": area_rain.shape},
        "time": {"start": str(rain_times[0]), "end": str(rain_times[-1]),
                 "count": len(rain_times)},
        "crs": catch_crs,
        "sha256": hashes,
    }
    signature = hashlib.sha256(json.dumps(
        signature_doc, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return DataContract(ids, names, areas, rain_times, area_rain, flow, mask,
                        rain_lat, rain_lon, catch_crs, signature, hashes)


def target_time_range(samples: Sequence[Sequence[int]], times: pd.DatetimeIndex,
                      lookback: int, horizon: int) -> Dict[str, Any]:
    arr = np.asarray(samples, dtype=np.int64).reshape(-1, 2)
    if not len(arr):
        raise ValueError("切分样本为空，无法建立训练清单")
    starts = arr[:, 1] + int(lookback)
    ends = starts + int(horizon) - 1
    if starts.min() < 0 or ends.max() >= len(times):
        raise ValueError("切分目标越出时间轴")
    return {
        "samples": int(len(arr)),
        "target_start": str(times[int(starts.min())]),
        "target_end": str(times[int(ends.max())]),
        "target_start_index": int(starts.min()),
        "target_end_index": int(ends.max()),
    }


def assert_calendar_splits(samples: Mapping[str, Sequence[Sequence[int]]],
                           times: pd.DatetimeIndex, lookback: int,
                           horizon: int) -> Dict[str, Dict[str, Any]]:
    """硬断言：训练目标 <2023，验证目标在 2023，测试目标 >=2024。"""
    info = {name: target_time_range(samples[name], times, lookback, horizon)
            for name in ("train", "val", "test")}
    t2023, t2024 = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")
    tr_end = pd.Timestamp(info["train"]["target_end"])
    va_start = pd.Timestamp(info["val"]["target_start"])
    va_end = pd.Timestamp(info["val"]["target_end"])
    te_start = pd.Timestamp(info["test"]["target_start"])
    if not tr_end < t2023:
        raise AssertionError(f"训练目标越过 2023：{tr_end}")
    if not (va_start >= t2023 and va_end < t2024):
        raise AssertionError(f"验证目标不全在 2023：{va_start} ~ {va_end}")
    if not te_start >= t2024:
        raise AssertionError(f"测试目标早于 2024：{te_start}")
    return info


def write_training_manifest(path: os.PathLike[str] | str, *,
                            contract: DataContract,
                            split_info: Mapping[str, Mapping[str, Any]],
                            normalization_end_exclusive: pd.Timestamp,
                            config: Mapping[str, Any],
                            extra: Mapping[str, Any] | None = None) -> None:
    doc: Dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "data_signature": contract.signature,
        "data_component_sha256": dict(contract.component_sha256),
        "splits": {key: dict(value) for key, value in split_info.items()},
        "normalization": {
            "fit_data": "flow and area_rain from the training prefix only",
            "end_exclusive": str(pd.Timestamp(normalization_end_exclusive)),
        },
        "effective_config": json.loads(json.dumps(config, ensure_ascii=False, default=str)),
    }
    if extra:
        doc.update(json.loads(json.dumps(extra, ensure_ascii=False, default=str)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)
