#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载并裁剪 MRMS 逐小时降水，累积成 NetCDF。

产品为 MultiSensor_QPE_01H_Pass2（雷达与雨量计融合，约 1 公里），取自爱荷华
州立大学的存档镜像。按配置中的经纬度范围裁剪后存成格点场。

支持断点续传：每隔若干小时把已完成的部分写一次检查点，中断后重跑会跳过
已下载的时刻，只补缺口。

用法: python3 scripts/pipeline/fetch_rain.py [--config configs/pipeline.yaml]
                                              [--start 2014-05-01] [--end 2024-12-31]
"""
import argparse
import gzip
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import eccodes
import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = "https://mtarchive.geol.iastate.edu/{dt:%Y/%m/%d}/mrms/ncep/MultiSensor_QPE_01H_Pass2/"
FN = "MultiSensor_QPE_01H_Pass2_00.00_{dt:%Y%m%d-%H0000}.grib2.gz"

# MRMS 美国本土网格：3500 行 × 7000 列，0.01 度，左上角在北纬 54.995、西经 129.995
ROW0, COL0, RES, NROW, NCOL = 54.995, -129.995, 0.01, 3500, 7000


def slices_for(lat_min, lat_max, lon_min, lon_max):
    r0 = int(round((ROW0 - lat_max) / RES))
    r1 = int(round((ROW0 - lat_min) / RES)) + 1
    c0 = int(round((lon_min - COL0) / RES))
    c1 = int(round((lon_max - COL0) / RES)) + 1
    if not (0 <= r0 < r1 <= NROW and 0 <= c0 < c1 <= NCOL):
        raise ValueError(f"范围超出 MRMS 美国本土网格：行 {r0}:{r1} 列 {c0}:{c1}")
    return slice(r0, r1), slice(c0, c1)


def fetch_crop(dt, rs, cs):
    url = BASE.format(dt=dt) + FN.format(dt=dt)
    raw = urllib.request.urlopen(url, timeout=120).read()
    data = gzip.decompress(raw)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        with open(path, "rb") as f:
            msg = eccodes.codes_grib_new_from_file(f)
            flat = eccodes.codes_get_values(msg)
            eccodes.codes_release(msg)
        # scanMode=0：按（行 3500 北→南，列 7000 西→东）行主序展开
        arr = np.asarray(flat, dtype=np.float32).reshape(NROW, NCOL)
        vals = arr[rs, cs][::-1, :].copy()      # 翻成自南向北
        vals[vals < 0] = np.nan                 # 负值为无覆盖填充码
        return dt, vals
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="丢弃检查点重新下载")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b, p = cfg["basin"], cfg["paths"]

    start = args.start or str(cfg["period"]["start"])
    end = args.end or str(cfg["period"]["end"])
    threads = args.threads or int(cfg["fetch"]["rain_threads"])
    ckpt_every = int(cfg["fetch"]["rain_checkpoint_hours"])

    rs, cs = slices_for(float(b["lat_min"]), float(b["lat_max"]),
                        float(b["lon_min"]), float(b["lon_max"]))
    ny, nx = rs.stop - rs.start, cs.stop - cs.start
    print(f"裁剪窗口 行 {rs.start}:{rs.stop} 列 {cs.start}:{cs.stop}  →  网格 {ny}×{nx}")

    out_nc = os.path.join(ROOT, p["rain_nc"])
    ckpt = out_nc.replace(".nc", "_partial.npy")
    times = pd.date_range(start, end, freq="h", tz="UTC").tz_localize(None)
    print(f"共 {len(times)} 小时：{times[0]} ~ {times[-1]}")

    if args.reset and os.path.exists(ckpt):
        os.unlink(ckpt)
    if os.path.exists(ckpt):
        arr = np.load(ckpt)
        if arr.shape[0] != len(times):
            print("检查点维数不符，重新开始")
            arr = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
    else:
        arr = np.full((len(times), ny, nx), np.nan, dtype=np.float32)

    done = np.isfinite(arr).any(axis=(1, 2))
    todo = np.where(~done)[0]
    print(f"待下载 {len(todo)} 小时，并发 {threads}\n")

    failed = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = {pool.submit(fetch_crop, times[i], rs, cs): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                _, vals = fut.result()
                arr[i] = vals
            except Exception as e:
                failed.append(times[i])
                if len(failed) <= 5:
                    print(f"  {times[i]} 失败: {type(e).__name__} {str(e)[:90]}")
            if n % ckpt_every == 0:
                np.save(ckpt, arr)
                ok = int(np.isfinite(arr).any(axis=(1, 2)).sum())
                print(f"  进度 {n}/{len(todo)}  已成功 {ok}/{len(times)}  已存检查点")

    np.save(ckpt, arr)
    # 失败的时刻再试一轮
    if failed:
        print(f"\n重试 {len(failed)} 个失败时刻…")
        for dt in failed:
            try:
                i = times.get_loc(dt)
                _, vals = fetch_crop(dt, rs, cs)
                arr[i] = vals
            except Exception:
                pass

    ok = np.isfinite(arr).any(axis=(1, 2))
    lats = float(b["lat_max"]) - (np.arange(ny) + 0.5) * RES
    lons = float(b["lon_min"]) + (np.arange(nx) + 0.5) * float(b["step"])
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), arr, {"units": "mm/h",
                                                "source": "MRMS MultiSensor_QPE_01H_Pass2"})},
        coords={"time": times, "lat": lats[::-1] if lats[0] > lats[-1] else lats,
                "lon": lons})
    ds = ds.sortby("lat")
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)
    print(f"\n成功 {int(ok.sum())}/{len(times)} 小时  缺 {int((~ok).sum())} 小时")
    print(f"已写 {out_nc}  {os.path.getsize(out_nc) / 1e9:.2f} GB")
    if os.path.exists(ckpt):
        os.unlink(ckpt)


if __name__ == "__main__":
    main()
