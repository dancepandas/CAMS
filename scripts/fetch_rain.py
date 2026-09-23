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
已下载的时刻，只补缺口。

用法: python3 scripts/fetch_rain.py [--config configs/pipeline.yaml]
                                     [--start 2015-06-01] [--end 2024-12-31]
"""
import argparse
import gzip
import os
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = "https://mtarchive.geol.iastate.edu/{dt:%Y/%m/%d}/mrms/ncep/{prod}/"
FN = "{prod}_00.00_{dt:%Y%m%d-%H0000}.grib2.gz"

# MRMS 美国本土网格：3500 行 × 7000 列，0.01 度，左上角在北纬 54.995、西经 129.995
ROW0, COL0, RES, NROW, NCOL = 54.995, -129.995, 0.01, 3500, 7000

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
    r0 = int(round((ROW0 - lat_max) / RES))
    r1 = int(round((ROW0 - lat_min) / RES)) + 1
    c0 = int(round((lon_min - COL0) / RES))
    c1 = int(round((lon_max - COL0) / RES)) + 1
    if not (0 <= r0 < r1 <= NROW and 0 <= c0 < c1 <= NCOL):
        raise ValueError(f"范围超出 MRMS 美国本土网格：行 {r0}:{r1} 列 {c0}:{c1}")
    return slice(r0, r1), slice(c0, c1)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--threads", type=int, default=None, help="并发进程数")
    ap.add_argument("--reset", action="store_true", help="丢弃检查点重新下载")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b, p, f_ = cfg["basin"], cfg["paths"], cfg["fetch"]

    start = args.start or str(cfg["period"]["start"])
    end = args.end or str(cfg["period"]["end"])
    threads = args.threads or int(f_["rain_threads"])
    ckpt_min = float(f_.get("rain_checkpoint_minutes", 10))
    proxy = f_.get("proxy") or None
    segs = f_.get("rain_products") or [{"name": "MultiSensor_QPE_01H_Pass2"}]
    opener = make_opener(proxy)

    rs, cs = slices_for(float(b["lat_min"]), float(b["lat_max"]),
                        float(b["lon_min"]), float(b["lon_max"]))
    ny, nx = rs.stop - rs.start, cs.stop - cs.start
    print(f"裁剪窗口 行 {rs.start}:{rs.stop} 列 {cs.start}:{cs.stop}  →  网格 {ny}×{nx}")
    print(f"代理 {proxy or '（直连）'}   并发进程 {threads}   检查点间隔 {ckpt_min:g} 分钟")

    out_nc = os.path.join(ROOT, p["rain_nc"])
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ckpt = out_nc.replace(".nc", "_partial.npy")
    times = pd.date_range(start, end, freq="h", tz="UTC").tz_localize(None)
    print(f"共 {len(times)} 小时：{times[0]} ~ {times[-1]}")

    prods = [product_for(segs, t) for t in times]
    all_names = [str(s["name"]) for s in segs]
    # 首选按分段表定，取不到再依次试其它产品
    cands = [[p] + [q for q in all_names if q != p] for p in prods]
    used = np.array(prods, dtype=object)
    counts = Counter(prods)
    for n, c in counts.items():
        print(f"  {n:28s} 预计 {c:6d} 小时")

    if args.reset and os.path.exists(ckpt):
        os.unlink(ckpt)
    prod_ckpt = ckpt.replace(".npy", "_prod.npy")
    if os.path.exists(ckpt):
        arr = np.load(ckpt)
        if arr.shape != (len(times), ny, nx):
            print("检查点维数不符，重新开始")
            arr = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
        elif os.path.exists(prod_ckpt):
            used[:] = np.load(prod_ckpt, allow_pickle=True)
    else:
        arr = np.full((len(times), ny, nx), np.nan, dtype=np.float32)

    done = np.isfinite(arr).any(axis=(1, 2))
    todo = np.where(~done)[0]
    print(f"待下载 {len(todo)} 小时\n")

    last_ck = time.time()

    def checkpoint():
        nonlocal last_ck
        np.save(ckpt, arr)
        np.save(prod_ckpt, used, allow_pickle=True)
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
                      f"{rate:.1f} 个/秒  已跑 {el / 3600:.2f} 小时  预计还需 {eta:.2f} 小时  已存检查点")

    checkpoint()
    # 失败的时刻再试一轮
    if failed:
        print(f"\n重试 {len(failed)} 个失败时刻…")
        for dt in failed:
            i = times.get_loc(dt)
            try:
                _, prod, vals = fetch_crop(opener, dt, cands[i], rs, cs)
                arr[i] = vals
                used[i] = prod
            except Exception:
                pass
        checkpoint()

    ok = np.isfinite(arr).any(axis=(1, 2))
    print(f"\n最终 成功 {int(ok.sum())}/{len(times)} 小时  缺 {int((~ok).sum())} 小时")
    lats = float(b["lat_max"]) - (np.arange(ny) + 0.5) * RES
    lons = float(b["lon_min"]) + (np.arange(nx) + 0.5) * float(b["step"])
    labels = [n for n in all_names if bool((used == n).any())]
    for n in labels:
        print(f"  {n:28s} 实取 {int((used == n).sum()):6d} 小时")
    pcodes = np.array([labels.index(x) if x in labels else -1 for x in used],
                      dtype=np.int8)
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), arr, {"units": "mm/h",
                                                "source": "MRMS 1 小时 QPE"})},
        coords={"time": times, "lat": lats[::-1] if lats[0] > lats[-1] else lats,
                "lon": lons, "product": ("time", pcodes,
                                         {"labels": "|".join(labels)})})
    ds = ds.sortby("lat")
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)
    print(f"已写 {out_nc}  {os.path.getsize(out_nc) / 1e9:.2f} GB")
    for f_ck in (ckpt, prod_ckpt):
        if os.path.exists(f_ck):
            os.unlink(f_ck)


if __name__ == "__main__":
    main()
