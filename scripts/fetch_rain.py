#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载并裁剪 MRMS 逐小时降水，累积成 NetCDF。

产品为 MultiSensor_QPE_01H_Pass2（雷达与雨量计融合，约 1 公里），取自爱荷华
州立大学的存档镜像。按配置中的经纬度范围裁剪后存成格点场。

**产品分段**。该镜像上 MultiSensor_QPE_01H_Pass2 只回溯到 2020-11，更早的
时段只有 GaugeCorr_QPE_01H（同样经过雨量计订正，网格、缺测码、扫描方式逐项
一致，仅产品名与 parameterNumber 不同）。故按 `fetch.rain_products` 分段取：
2020-11 之前用 GaugeCorr_QPE_01H，之后用 MultiSensor_QPE_01H_Pass2。每个时刻
实际用的产品记在输出的 `product` 变量里，便于论文里声明产品切换点。

支持断点续传：按墙钟时间间隔把已完成的部分写一次检查点，中断后重跑会跳过
已下载的时刻，只补缺口。已有文件坐标可用 ``--migrate-coordinates`` 安全迁移；
必须显式给 ``--backup``，程序会临时写盘、重开检查后再原子替换。

用法: python3 scripts/fetch_rain.py [--config configs/pipeline.yaml]
                                     [--start 2015-06-01] [--end 2024-12-31]
       python3 scripts/fetch_rain.py --migrate-coordinates data/rain.nc \
                                     --backup data/rain.before-grid-fix.nc
"""
import argparse
import gzip
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

import eccodes
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from spatial_grid import (MRMS_NCOL, MRMS_NORTH, MRMS_NROW, MRMS_STEP,
                          MRMS_WEST, grid_signature, mrms_coordinates,
                          mrms_window, validate_regular_coordinate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = "https://mtarchive.geol.iastate.edu/{dt:%Y/%m/%d}/mrms/ncep/{prod}/"
FN = "{prod}_00.00_{dt:%Y%m%d-%H0000}.grib2.gz"

# 保留旧名，避免外部脚本依赖这些常量；坐标计算统一由 spatial_grid 负责。
ROW0, COL0, RES, NROW, NCOL = MRMS_NORTH, MRMS_WEST, MRMS_STEP, MRMS_NROW, MRMS_NCOL

# eccodes 非线程安全，所有解码调用须经此锁串行化
_ECC_LOCK = Lock()


def make_opener(proxy=None):
    """带代理的 urlopen 包装；proxy 为空则直连（仍尊重 HTTP(S)_PROXY 环境变量）。"""
    if proxy:
        h = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(h)
    return urllib.request.build_opener()


def product_for(segs, dt):
    """按分段表返回该时刻应使用的产品名。最后一段无 until，兜底生效。"""
    for s in segs:
        u = s.get("until")
        if u is None or dt < pd.Timestamp(u):
            return str(s["name"])
    return str(segs[-1]["name"])


def slices_for(lat_min, lat_max, lon_min, lon_max):
    return mrms_window(lat_min, lat_max, lon_min, lon_max)


def _validate_rain_dataset(ds, expected_shape=None, expected_lat=None,
                           expected_lon=None, expected_start=None,
                           expected_end=None):
    """写盘或迁移后重开校验坐标、维数、时段与网格指纹。"""
    if "rain" not in ds or tuple(ds["rain"].dims) != ("time", "lat", "lon"):
        raise ValueError("rain 必须按 (time, lat, lon) 存储")
    if ds["rain"].dtype != np.float32:
        raise ValueError(f"rain 必须是 float32，实际为 {ds['rain'].dtype}")
    lat, _ = validate_regular_coordinate(ds["lat"].values, "lat")
    lon, _ = validate_regular_coordinate(ds["lon"].values, "lon")
    if expected_lat is not None and not np.array_equal(lat, expected_lat):
        raise ValueError("rain 纬度与预期网格不一致")
    if expected_lon is not None and not np.array_equal(lon, expected_lon):
        raise ValueError("rain 经度与预期网格不一致")
    times = np.asarray(ds["time"].values).astype("datetime64[ns]")
    if times.ndim != 1 or times.size == 0 or np.isnat(times).any():
        raise ValueError("time 必须是一维、非空且有效")
    if times.size > 1 and not np.all(np.diff(times) == np.timedelta64(1, "h")):
        raise ValueError("time 必须无重复、无缺口地逐小时递增")
    if expected_start is not None and times[0] != np.datetime64(expected_start, "ns"):
        raise ValueError("rain 起点与预期不一致")
    if expected_end is not None and times[-1] != np.datetime64(expected_end, "ns"):
        raise ValueError("rain 终点与预期不一致")
    shape = tuple(int(x) for x in ds["rain"].shape)
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(f"rain 形状 {shape} 与预期 {tuple(expected_shape)} 不一致")
    crs_value = ds.attrs.get("crs")
    if crs_value is None:
        raise ValueError("rain 必须显式声明 crs")
    expected_sig = grid_signature(lat, lon, str(crs_value))
    stored_sig = ds.attrs.get("grid_signature")
    if stored_sig is None or str(stored_sig) != expected_sig:
        raise ValueError("NetCDF 网格指纹缺失或与坐标不一致")
    return expected_sig


def _atomic_write_netcdf(ds, target):
    """同目录临时写盘，重开验证后原子发布。"""
    if int(ds.sizes.get("time", 0)) == 0:
        raise ValueError("rain 数据集不得为空")
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(target) + ".",
                               suffix=".tmp.nc", dir=parent)
    os.close(fd)
    try:
        ds.to_netcdf(tmp)
        with xr.open_dataset(tmp) as check:
            _validate_rain_dataset(check, ds["rain"].shape)
            check["rain"].isel(time=[0, -1]).load()
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _compare_migrated_netcdf(source_path, migrated_path, chunk_hours=168):
    """逐块证明坐标迁移没有改变 time/product/rain 的任何值。"""
    if chunk_hours <= 0:
        raise ValueError("chunk_hours 必须为正整数")
    with xr.open_dataset(source_path) as source, xr.open_dataset(migrated_path) as migrated:
        for name in ("rain", "time"):
            if name not in source or name not in migrated:
                raise ValueError(f"迁移前后都必须含 {name}")
        old_rain, new_rain = source["rain"], migrated["rain"]
        if old_rain.dims != new_rain.dims or old_rain.shape != new_rain.shape:
            raise ValueError("迁移后 rain 维数或形状改变")
        if old_rain.dtype != new_rain.dtype:
            raise ValueError("迁移后 rain dtype 改变")

        old_time, new_time = source["time"], migrated["time"]
        if (old_time.dims != new_time.dims or old_time.shape != new_time.shape
                or old_time.dtype != new_time.dtype
                or not np.array_equal(old_time.values, new_time.values)):
            raise ValueError("迁移后 time 改变")

        old_has_product = "product" in source.variables
        if old_has_product != ("product" in migrated.variables):
            raise ValueError("迁移后 product 字段存在性改变")
        if old_has_product:
            old_product, new_product = source["product"], migrated["product"]
            if (old_product.dims != new_product.dims
                    or old_product.shape != new_product.shape
                    or old_product.dtype != new_product.dtype
                    or not np.array_equal(old_product.values, new_product.values)):
                raise ValueError("迁移后 product 改变")

        total = old_rain.sizes["time"]
        for start in range(0, total, int(chunk_hours)):
            stop = min(total, start + int(chunk_hours))
            old = old_rain.isel(time=slice(start, stop)).values
            new = new_rain.isel(time=slice(start, stop)).values
            if not np.array_equal(np.isnan(old), np.isnan(new)):
                raise ValueError(f"迁移后 rain 缺测位置改变：time[{start}:{stop}]")
            if not np.array_equal(old, new, equal_nan=True):
                raise ValueError(f"迁移后 rain 数值改变：time[{start}:{stop}]")


def migrate_netcdf_coordinates(path, lat, lon, backup=None):
    """安全修正现有 NetCDF 坐标，不改变数值数组的南到北顺序。

    xarray 保持惰性读取并由 NetCDF 后端分块复制，不会把整个降水场装入内存。
    默认拒绝覆盖已有文件；调用者须显式给出备份路径。
    """
    target = os.path.abspath(path)
    if not os.path.isfile(target):
        raise FileNotFoundError(target)
    if not backup:
        raise ValueError("迁移现有文件必须显式提供 --backup 路径")
    backup = os.path.abspath(backup)
    if os.path.exists(backup):
        raise FileExistsError(f"备份已存在：{backup}")
    if os.path.dirname(backup) != os.path.dirname(target):
        raise ValueError("备份必须与原文件同目录，才能原子改名")

    lat, _ = validate_regular_coordinate(lat, "lat")
    lon, _ = validate_regular_coordinate(lon, "lon")
    tmp = None
    try:
        with xr.open_dataset(target) as source:
            if source.sizes.get("lat") != len(lat) or source.sizes.get("lon") != len(lon):
                raise ValueError("新坐标长度与现有 rain 网格不一致")
            expected_shape = tuple(int(x) for x in source["rain"].shape)
            ds = source.assign_coords(lat=lat, lon=lon)
            ds.attrs.update(crs="EPSG:4326",
                            grid_signature=grid_signature(lat, lon))
            parent = os.path.dirname(target)
            fd, tmp = tempfile.mkstemp(prefix=os.path.basename(target) + ".migrate.",
                                       suffix=".tmp.nc", dir=parent)
            os.close(fd)
            ds.to_netcdf(tmp)
        # Windows 上原文件句柄关闭后才能原子改名。
        with xr.open_dataset(tmp) as check:
            _validate_rain_dataset(check, expected_shape)
        _compare_migrated_netcdf(target, tmp)
        os.replace(target, backup)
        try:
            os.replace(tmp, target)
            tmp = None
        except BaseException:
            os.replace(backup, target)
            raise
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def decode(data, rs, cs):
    """从 grib2 字节流解出裁剪后的数组。

    eccodes 的 GRIB 定义解析用的是非可重入的 flex 扫描器，多线程同时解码会
    直接段错误（实测 fatal flex scanner internal error 后进程崩溃）。故把全部
    eccodes 调用串行化；解码本身只占单文件耗时的小头，网络才是瓶颈。
    """
    with _ECC_LOCK:
        msg = eccodes.codes_new_from_message(data)
        try:
            flat = eccodes.codes_get_values(msg)
        finally:
            eccodes.codes_release(msg)
    # scanMode=0：按（行 3500 北→南，列 7000 西→东）行主序展开
    arr = np.asarray(flat, dtype=np.float32).reshape(NROW, NCOL)
    vals = arr[rs, cs][::-1, :].copy()      # 翻成自南向北
    vals[vals < 0] = np.nan                 # 负值为无覆盖填充码
    return vals


def fetch_crop(opener, dt, prods, rs, cs, attempts=3):
    """按 prods 顺序尝试各产品，返回首个取到的数据。

    镜像上两个产品的切换并不是整月对齐的（GaugeCorr 在 2020-10 月中就断了），
    与其预先扫描边界，不如取不到就换另一个产品，自愈。
    """
    last = None
    for prod in prods:
        url = BASE.format(dt=dt, prod=prod) + FN.format(dt=dt, prod=prod)
        for k in range(attempts):
            try:
                r = opener.open(url, timeout=180)
                raw = r.read()
                if getattr(r, "status", 200) != 200:
                    raise RuntimeError(f"http {r.status}")
                return dt, prod, decode(gzip.decompress(raw), rs, cs)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (403, 404):     # 该产品这天没有，换下一个
                    break
            except Exception as e:           # 网络抖动/限流，退避重试
                last = e
            if k < attempts - 1:
                time.sleep(1.5 * (k + 1))
    raise last


_G = {}


def _init_worker(proxy):
    """每个工作进程各自建一个 opener。"""
    _G["opener"] = make_opener(proxy)


def fetch_one(job):
    """工作进程的任务：下载 + 解码一个时刻，返回裁剪后的小数组。

    解码一个 MRMS 文件要取出 2450 万个 float64（约 263 毫秒，实测），占了单
    文件耗时的绝大部分。放线程池里会被 eccodes 的定义锁串行化，八万余个文件
    要跑六个小时；改用进程池，每个进程各自持有一份 eccodes 状态，解码就随核
    数并行，整体退回网络瓶颈。
    """
    i, dt, prods, rs, cs = job
    try:
        _, prod, vals = fetch_crop(_G["opener"], dt, prods, rs, cs)
        return i, prod, vals, None
    except Exception as e:
        # 异常对象里挂着 urllib 的文件句柄，没法跨进程 pickle，
        # 所以在这里就地转成字符串返回。
        return i, None, None, f"{type(e).__name__} {str(e)[:80]}"


def _checkpoint_contract(times, rs, cs, segs, lat, lon):
    """把所有会影响检查点含义的参数固定下来，防止同形旧数组被误用。"""
    return {
        "format": "cams-mrms-checkpoint-v1",
        "start": str(pd.Timestamp(times[0])),
        "end": str(pd.Timestamp(times[-1])),
        "hours": int(len(times)),
        "row_start": int(rs.start),
        "row_stop": int(rs.stop),
        "col_start": int(cs.start),
        "col_stop": int(cs.stop),
        "products": [{"name": str(s["name"]),
                      "until": None if s.get("until") is None else str(s["until"])}
                     for s in segs],
        "grid_signature": grid_signature(lat, lon),
    }


def _atomic_write_json(path, document):
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".",
                               suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_checkpoint(ckpt, prod_ckpt, meta_ckpt, expected, shape, defaults):
    paths = (ckpt, prod_ckpt, meta_ckpt)
    present = [os.path.exists(path) for path in paths]
    if not any(present):
        return np.full(shape, np.nan, dtype=np.float32), defaults.copy()
    if not all(present):
        raise ValueError("MRMS 检查点文件不完整；请使用 --reset 后重试")
    try:
        with open(meta_ckpt, encoding="utf-8") as handle:
            actual = json.load(handle)
    except Exception as exc:
        raise ValueError("MRMS 检查点元数据无法读取；请使用 --reset 后重试") from exc
    if actual != expected:
        raise ValueError("MRMS 检查点参数与本次下载不一致；请使用 --reset 后重试")
    arr = np.load(ckpt, allow_pickle=False)
    used = np.load(prod_ckpt, allow_pickle=False).astype(str)
    if arr.shape != shape or arr.dtype != np.float32:
        raise ValueError("MRMS 检查点降雨数组不符合本次合同；请使用 --reset 后重试")
    if used.shape != (shape[0],):
        raise ValueError("MRMS 检查点产品数组长度错误；请使用 --reset 后重试")
    return arr, used


def _atomic_save_array(path, values):
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".",
                               suffix=".tmp.npy", dir=parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _raise_retry_failures(retry_errors):
    if retry_errors:
        preview = "；".join(f"{dt}: {err}" for dt, err in retry_errors[:5])
        raise RuntimeError(
            f"MRMS 最终仍有 {len(retry_errors)} 个时刻下载失败，正式文件未发布。"
            f"检查点已保留；前几项：{preview}")


def _validate_complete_hours(arr, times):
    ok = np.isfinite(arr).any(axis=(1, 2))
    missing = np.flatnonzero(~ok)
    if len(missing):
        preview = ", ".join(str(times[i]) for i in missing[:5])
        raise RuntimeError(
            f"MRMS 有 {len(missing)} 个整小时全部缺测，正式文件未发布；"
            f"前几项：{preview}")
    return ok


def _load_existing_for_extension(out_nc, times, ny, nx, lats, lons):
    """如果正式文件已存在且是本次下载的前缀，则读入旧数据避免重复下载。"""
    if not os.path.exists(out_nc):
        return None
    try:
        with xr.open_dataset(out_nc) as ds:
            old_times = pd.DatetimeIndex(ds["time"].values)
            old_rain = ds["rain"].values
            old_lat = ds["lat"].values
            old_lon = ds["lon"].values
    except Exception:
        return None
    if not (np.array_equal(old_lat, lats) and np.array_equal(old_lon, lons)):
        return None
    # 检查旧文件是否是本次请求的前缀
    if len(old_times) > len(times):
        return None
    if not np.array_equal(old_times.values, times.values[:len(old_times)]):
        return None
    # 旧文件必须是完整的（无全缺测小时）
    old_ok = np.isfinite(old_rain).any(axis=(1, 2))
    if not old_ok.all():
        return None
    return old_rain, old_times


def _publish(arr, used, times, out_nc, lats, lons, segs):
    """验证、构造 Dataset 并原子发布正式 NetCDF。"""
    ok = _validate_complete_hours(arr, times)
    print(f"\n最终 成功 {int(ok.sum())}/{len(times)} 小时  缺 0 小时")
    all_names = [str(s["name"]) for s in segs]
    labels = [n for n in all_names if bool((used == n).any())]
    for n in labels:
        print(f"  {n:28s} 实取 {int((used == n).sum()):6d} 小时")
    pcodes = np.array([labels.index(x) if x in labels else -1 for x in used],
                      dtype=np.int8)
    signature = grid_signature(lats, lons)
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), arr, {"units": "mm/h",
                                                "source": "MRMS 1 小时 QPE"})},
        coords={"time": times, "lat": lats, "lon": lons,
                "product": ("time", pcodes, {"labels": "|".join(labels)})},
        attrs={"crs": "EPSG:4326", "grid_signature": signature,
               "grid_origin": "MRMS global indices"})
    _atomic_write_netcdf(ds, out_nc)
    print(f"已写 {out_nc}  {os.path.getsize(out_nc) / 1e9:.2f} GB")


def _run_download(arr, used, todo, times, cands, rs, cs, threads,
                  proxy, ckpt, prod_ckpt, meta_ckpt, ckpt_min,
                  out_nc, lats, lons, segs):
    """下载缺失小时、重试、验证并发布。"""
    opener = make_opener(proxy)
    checkpoint_doc = _checkpoint_contract(times, rs, cs, segs, lats, lons)
    last_ck = time.time()

    def checkpoint():
        nonlocal last_ck
        _atomic_save_array(ckpt, arr)
        _atomic_save_array(prod_ckpt, used)
        _atomic_write_json(meta_ckpt, checkpoint_doc)
        last_ck = time.time()

    failed, n_ok, t_start = [], 0, time.time()
    jobs = [(i, times[i], cands[i], rs, cs) for i in todo]
    with ProcessPoolExecutor(max_workers=threads, initializer=_init_worker,
                             initargs=(proxy,)) as pool:
        futs = {pool.submit(fetch_one, j): j[0] for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            i, prod, vals, err = fut.result()
            if err is None:
                arr[i] = vals
                used[i] = prod
                n_ok += 1
            else:
                failed.append(times[i])
                if len(failed) <= 5:
                    print(f"  {times[i]} 失败: {err}")
            if time.time() - last_ck > ckpt_min * 60:
                checkpoint()
                done = int(np.isfinite(arr).any(axis=(1, 2)).sum())
                el = time.time() - t_start
                rate = n / el
                eta = (len(todo) - n) / rate / 3600 if rate > 0 else float("nan")
                print(f"  进度 {n}/{len(todo)}  已成功 {done}/{len(times)}  "
                      f"{rate:.1f} 个/秒  已跑 {el / 3600:.2f} 小时  "
                      f"预计还需 {eta:.2f} 小时  已存检查点")

    checkpoint()
    if failed:
        print(f"\n重试 {len(failed)} 个失败时刻…")
        retry_errors = []
        for dt in failed:
            i = times.get_loc(dt)
            try:
                _, prod, vals = fetch_crop(opener, dt, cands[i], rs, cs)
                arr[i] = vals
                used[i] = prod
            except Exception as exc:
                retry_errors.append(
                    (dt, f"{type(exc).__name__}: {str(exc)[:160]}"))
        checkpoint()
        _raise_retry_failures(retry_errors)

    _publish(arr, used, times, out_nc, lats, lons, segs)
    for f_ck in (ckpt, prod_ckpt, meta_ckpt):
        if os.path.exists(f_ck):
            os.unlink(f_ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--threads", type=int, default=None, help="并发进程数")
    ap.add_argument("--reset", action="store_true", help="丢弃检查点重新下载")
    ap.add_argument("--migrate-coordinates", metavar="NETCDF",
                    help="只迁移指定现有 NetCDF 的坐标，不下载")
    ap.add_argument("--backup", help="迁移时必须显式指定、且不得已存在的备份路径")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b, p, f_ = cfg["basin"], cfg["paths"], cfg["fetch"]

    rs, cs = slices_for(float(b["lat_min"]), float(b["lat_max"]),
                        float(b["lon_min"]), float(b["lon_max"]))
    if args.migrate_coordinates:
        lat, lon = mrms_coordinates(rs, cs)
        src = args.migrate_coordinates
        if not os.path.isabs(src):
            src = os.path.join(ROOT, src)
        backup = args.backup
        if backup and not os.path.isabs(backup):
            backup = os.path.join(ROOT, backup)
        migrate_netcdf_coordinates(src, lat, lon, backup)
        print(f"已迁移 {src}；旧文件保存在 {backup}")
        return

    start = args.start or str(cfg["period"]["start"])
    end = args.end or str(cfg["period"]["end"])
    threads = args.threads or int(f_["rain_threads"])
    ckpt_min = float(f_.get("rain_checkpoint_minutes", 10))
    proxy = f_.get("proxy") or None
    segs = f_.get("rain_products") or [{"name": "MultiSensor_QPE_01H_Pass2"}]
    opener = make_opener(proxy)

    ny, nx = rs.stop - rs.start, cs.stop - cs.start
    print(f"裁剪窗口 行 {rs.start}:{rs.stop} 列 {cs.start}:{cs.stop}  →  网格 {ny}×{nx}")
    print(f"代理 {proxy or '（直连）'}   并发进程 {threads}   检查点间隔 {ckpt_min:g} 分钟")

    out_nc = os.path.join(ROOT, p["rain_nc"])
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ckpt = out_nc.replace(".nc", "_partial.npy")
    prod_ckpt = ckpt.replace(".npy", "_prod.npy")
    meta_ckpt = ckpt.replace(".npy", "_meta.json")
    times = pd.date_range(start, end, freq="h", tz="UTC").tz_localize(None)
    if len(times) == 0:
        raise ValueError("下载时间范围为空")
    print(f"共 {len(times)} 小时：{times[0]} ~ {times[-1]}")

    lats, lons = mrms_coordinates(rs, cs)

    # 尝试从已有正式文件扩展（仅当旧文件是本次下载的完整前缀时）
    existing = None
    if not args.reset:
        existing = _load_existing_for_extension(out_nc, times, ny, nx, lats, lons)
        if existing is not None:
            old_rain, old_times = existing
            print(f"从已有文件扩展：保留 {len(old_times)} 小时，"
                  f"新下载 {len(times) - len(old_times)} 小时")
            # 构造完整数组
            arr = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
            arr[:len(old_times)] = old_rain
            used = np.array([product_for(segs, t) for t in times],
                          dtype=f"<U{max(map(len, [str(s['name']) for s in segs]))}")
            # 标记已完成的小时
            done = np.isfinite(arr).any(axis=(1, 2))
            todo = np.where(~done)[0]
            prods = [product_for(segs, t) for t in times]
            cands = [[p] + [q for q in [str(s["name"]) for s in segs] if q != p]
                     for p in prods]
            _run_download(arr, used, todo, times, cands, rs, cs, threads,
                         proxy, ckpt, prod_ckpt, meta_ckpt, ckpt_min,
                         out_nc, lats, lons, segs)
            return

    prods = [product_for(segs, t) for t in times]
    all_names = [str(s["name"]) for s in segs]
    # 首选按分段表定，取不到再依次试其它产品
    cands = [[p] + [q for q in all_names if q != p] for p in prods]
    used = np.array(prods, dtype=f"<U{max(map(len, all_names))}")
    counts = Counter(prods)
    for n, c in counts.items():
        print(f"  {n:28s} 预计 {c:6d} 小时")

    checkpoint_doc = _checkpoint_contract(times, rs, cs, segs, lats, lons)
    if args.reset:
        for path in (ckpt, prod_ckpt, meta_ckpt):
            if os.path.exists(path):
                os.unlink(path)
    arr, used = _load_checkpoint(
        ckpt, prod_ckpt, meta_ckpt, checkpoint_doc,
        (len(times), ny, nx), used)

    done = np.isfinite(arr).any(axis=(1, 2))
    todo = np.where(~done)[0]
    print(f"待下载 {len(todo)} 小时\n")
    _run_download(arr, used, todo, times, cands, rs, cs, threads,
                  proxy, ckpt, prod_ckpt, meta_ckpt, ckpt_min,
                  out_nc, lats, lons, segs)


if __name__ == "__main__":
    main()
