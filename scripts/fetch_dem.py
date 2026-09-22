#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 30 米 SRTM 高程图幅。

按配置中的经纬度范围自动推算需要哪些图幅，避免手工填错导致缺块。
数据取自 AWS 公开数据集 elevation-tiles-prod，1 弧秒（约 30 米）。

用法: python3 scripts/pipeline/fetch_dem.py [--config configs/pipeline.yaml]
"""
import argparse
import gzip
import math
import os
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{band}/{tile}.hgt.gz"


def tiles_for(lat_min, lat_max, lon_min, lon_max):
    """返回覆盖该范围的所有 SRTM 图幅名。"""
    out = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            out.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    b = cfg["basin"]
    out_dir = os.path.join(ROOT, cfg["paths"]["dem30"])
    os.makedirs(out_dir, exist_ok=True)

    tiles = tiles_for(b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"])
    print(f"范围 纬度 {b['lat_min']}~{b['lat_max']}  经度 {b['lon_min']}~{b['lon_max']}")
    print(f"需要图幅 {len(tiles)} 块: {' '.join(tiles)}\n")

    for t in tiles:
        hgt = os.path.join(out_dir, f"{t}.hgt")
        if os.path.exists(hgt) and os.path.getsize(hgt) == 3601 * 3601 * 2:
            print(f"{t}  已存在，跳过")
            continue
        url = BASE.format(band=t[:3], tile=t)
        try:
            raw = urllib.request.urlopen(url, timeout=300).read()
            data = gzip.decompress(raw)
            if len(data) != 3601 * 3601 * 2:
                raise ValueError(f"解压后大小 {len(data)} 不符，应为 {3601 * 3601 * 2}")
            with open(hgt, "wb") as f:
                f.write(data)
            print(f"{t}  完成  {len(data) / 1e6:.1f} MB")
        except Exception as e:
            print(f"{t}  失败: {type(e).__name__} {str(e)[:120]}")


if __name__ == "__main__":
    main()
