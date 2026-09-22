#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按顺序执行断面流量预报的完整管道。

各步骤相互独立，产物落盘，可以单独重跑；本脚本只是把它们串起来并打印
每步耗时。

用法:
  python3 scripts/pipeline/run_all.py                    # 跑全部
  python3 scripts/pipeline/run_all.py --from catchments  # 从某一步开始
  python3 scripts/pipeline/run_all.py --only train       # 只跑一步
  python3 scripts/pipeline/run_all.py --skip fetch_rain  # 跳过某一步
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "scripts")

STEPS = [
    ("fetch_flow",  "下载各断面逐小时流量（USGS）"),
    ("fetch_rain",  "下载并裁剪 MRMS 逐小时降水"),
    ("fetch_dem",   "下载 30 米 SRTM 高程图幅"),
    ("terrain",     "提取流向、汇流累积、坡度、河网级数"),
    ("catchments",  "划定各断面上游汇水区"),
    ("area_rain",   "计算各断面面雨量"),
    ("train",       "训练并评估断面预报模型"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--only", default=None, help="只跑指定步骤")
    ap.add_argument("--from", dest="start", default=None, help="从指定步骤开始")
    ap.add_argument("--skip", action="append", default=[], help="跳过指定步骤")
    args = ap.parse_args()

    names = [s[0] for s in STEPS]
    for v in [args.only, args.start, *args.skip]:
        if v and v not in names:
            sys.exit(f"未知步骤 {v}，可选：{', '.join(names)}")

    picked = []
    for name, desc in STEPS:
        if args.only and name != args.only:
            continue
        if args.start and names.index(name) < names.index(args.start):
            continue
        if name in args.skip:
            continue
        picked.append((name, desc))

    print(f"待执行 {len(picked)} 步：" + " → ".join(n for n, _ in picked) + "\n")
    total = time.time()
    for name, desc in picked:
        print(f"{'=' * 62}\n{name}  {desc}\n{'=' * 62}")
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, f"{name}.py"),
                            "--config", args.config], cwd=ROOT)
        dt = time.time() - t0
        print(f"--- {name} 用时 {dt / 60:.1f} 分钟  退出码 {r.returncode}\n")
        if r.returncode != 0:
            sys.exit(f"步骤 {name} 失败，中止")
    print(f"全部完成，总用时 {(time.time() - total) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
