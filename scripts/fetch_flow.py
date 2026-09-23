#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载各断面的逐小时流量。

取自美国地质调查局全国水情信息系统（参数代码 00060 瞬时流量）。
原始为 15 分钟间隔，取整点瞬时值，单位由立方英尺每秒换算为立方米每秒。

用法: python3 scripts/pipeline/fetch_flow.py [--config configs/pipeline.yaml] [--start 2014-01-01]
"""
import argparse
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFS_TO_CMS = 0.028316846592


def fetch_chunked(nwis, sid, start, end, chunk_years):
    """按年分段取瞬时流量再拼接。

    一次要十年会超时（实测 9.5 年单请求读到超时，五年以内正常），故分段取。
    """
    parts = []
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    while a <= b:
        z = min(a + pd.DateOffset(years=chunk_years) - pd.Timedelta(seconds=1), b)
        # USGS 只吃日期（带时间分量的 startDT 会返回 400）
        df, _ = nwis.get_iv(sites=sid, start=a.strftime("%Y-%m-%d"),
                            end=z.strftime("%Y-%m-%d"), parameterCd="00060")
        if len(df):
            cols = [c for c in df.columns if "00060" in c and not c.endswith("_cd")]
            if not cols:
                raise RuntimeError(f"未找到 00060 列，实际 {list(df.columns)[:6]}")
            parts.append(df[cols[0]].astype(float))
        a = z + pd.Timedelta(seconds=1)
    if not parts:
        return pd.Series(dtype=float)
    q = pd.concat(parts)
    return q[~q.index.duplicated(keep="last")].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--start", default=None, help="覆盖配置中的起始日期")
    ap.add_argument("--end", default=None, help="覆盖配置中的结束日期")
    ap.add_argument("--chunk-years", type=int, default=3,
                    help="单次请求覆盖的年数（太大 USGS 会超时）")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proxy = (cfg["fetch"].get("proxy") or "").strip()
    if proxy:                       # dataretrieval 走 requests，靠环境变量识别代理
        os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy
    from dataretrieval import nwis

    sites = pd.read_csv(os.path.join(ROOT, cfg["paths"]["sites_csv"]),
                        encoding="utf-8", dtype={"site_id": str})
    start = args.start or str(cfg["period"]["start"])
    end = args.end or str(cfg["period"]["end"])
    out_dir = os.path.join(ROOT, cfg["paths"]["sites"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"时段 {start} ~ {end}  断面 {len(sites)} 个\n")

    ok = 0
    for _, row in sites.iterrows():
        sid, name = str(row["site_id"]), str(row["name"])
        try:
            q = fetch_chunked(nwis, sid, start, end, args.chunk_years) * CFS_TO_CMS
            if q.empty:
                raise RuntimeError("接口未返回任何数据")
            hourly = q[q.index.minute == 0]
            if hourly.empty:
                print(f"{sid:11s} {name:10s} 无整点数据")
                continue
            hourly.rename("flow_m3s").to_csv(os.path.join(out_dir, f"{sid}.csv"))
            print(f"{sid:11s} {name:10s} {len(hourly):6d} 条  "
                  f"{hourly.index.min():%Y-%m-%d} ~ {hourly.index.max():%Y-%m-%d}  "
                  f"{hourly.min():8.2f} ~ {hourly.max():9.2f} 立方米每秒")
            ok += 1
        except Exception as e:
            print(f"{sid} 失败: {type(e).__name__} {str(e)[:150]}")
    print(f"\n完成 {ok}/{len(sites)} → {cfg['paths']['sites']}")


if __name__ == "__main__":
    main()
