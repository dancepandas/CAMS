#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 S.18 相对 S.14 退步的三个小站（米尔斯河/比特里溪/弗莱彻）。

S.18 比 S.14 多的只是 1990~2015 这段历史。这段历史对某站是"养分"还是
"噪声"，取决于三件事：

  1. 流量记录：该站 1990~2015 有没有实测流量？缺测多少？
  2. 面雨量质量：AORC 段该站汇水区里有效降雨格点比例如何？缺测严重时
     "有效值加权和 ÷ 有效权重和"会在极少格点上求平均，雨量本身不可信。
  3. 训练样本：两段历史各能给该站凑出多少条有效训练样本。

另取两个进步的小站（艾维河、罗斯曼）做对照，看"退步"是不是这三站特有的。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SITES = ["03446000", "03450000", "03447687",          # 退步三站
         "03453000", "03439000"]                       # 进步对照
NAMES = {"03446000": "米尔斯河", "03450000": "比特里溪", "03447687": "弗莱彻",
         "03453000": "艾维河", "03439000": "罗斯曼"}

LOOKBACK = 72
HORIZON = 1


def flow_series(sid):
    q = pd.read_csv(os.path.join(ROOT, "data", "sites", f"{sid}.csv"),
                    index_col=0, parse_dates=True)["flow_m3s"]
    if getattr(q.index, "tz", None) is not None:
        q.index = q.index.tz_localize(None)
    return q


def valid_samples(q, times):
    """在统一时间轴上数有效训练样本：起报窗口回看 72h 和目标时刻流量都非缺测。

    训练目标必须早于 2023-01-01（两套模型共同的边界）。
    """
    qq = q.reindex(times)
    n = len(times)
    good = np.isfinite(qq.values)
    cnt = 0
    for t in range(LOOKBACK - 1, n - HORIZON):
        if good[t - LOOKBACK + 1: t + 1].all() and good[t + HORIZON]:
            cnt += 1
    return cnt


def rain_gap_stats(area_rain, times, i, lo, hi):
    """时段 [lo, hi) 内该站面雨量缺测情况。"""
    seg = area_rain[i, lo:hi]
    miss = ~np.isfinite(seg)
    # 最长连续缺测（小时）
    longest = run = 0
    for v in miss:
        run = run + 1 if v else 0
        longest = max(longest, run)
    return miss.mean(), longest


def main():
    a14 = np.load(os.path.join(ROOT, "data", "area_rain.npz"), allow_pickle=True)
    a18 = np.load(os.path.join(ROOT, "data", "area_rain_1990_2024.npz"),
                  allow_pickle=True)
    ids14 = [str(x) for x in a14["site_ids"]]
    ids18 = [str(x) for x in a18["site_ids"]]
    t14 = pd.DatetimeIndex([str(x) for x in a14["times"]])
    t18 = pd.DatetimeIndex([str(x) for x in a18["times"]])

    # 两段历史的物理边界
    split = pd.Timestamp("2015-06-01")
    train_end = pd.Timestamp("2023-01-01")
    aorc_lo, aorc_hi = 0, int((split - t18[0]).total_seconds() // 3600)
    mrms14_lo = 0
    mrms14_hi = int((train_end - t14[0]).total_seconds() // 3600)

    print(f"{'站':<14}{'段':<16}{'小时数':>8}{'流量缺测':>9}"
          f"{'面雨缺测':>9}{'最长连缺h':>9}{'段内最大流量':>11}")
    for sid in SITES:
        q = flow_series(sid)
        i18, i14 = ids18.index(sid), ids14.index(sid)
        # AORC 段（仅 S.18 有）
        seg_t = t18[aorc_lo:aorc_hi]
        qq = q.reindex(seg_t).values
        miss_r, longest_r = rain_gap_stats(a18["area_rain"], t18, i18,
                                           aorc_lo, aorc_hi)
        print(f"{NAMES[sid]:<6}{sid}  AORC 1990~2015  {len(seg_t):>8}"
              f"{np.mean(~np.isfinite(qq)):>9.1%}{miss_r:>9.1%}{longest_r:>9}"
              f"{np.nanmax(qq) if np.isfinite(qq).any() else float('nan'):>11.1f}")
        # MRMS 段（S.14 的全部、S.18 的尾段，两者的面雨量数值一致）
        seg_t = t14[mrms14_lo:mrms14_hi]
        qq = q.reindex(seg_t).values
        miss_r, longest_r = rain_gap_stats(a14["area_rain"], t14, i14,
                                           mrms14_lo, mrms14_hi)
        print(f"{'':<14}MRMS 2015~2022  {len(seg_t):>8}"
              f"{np.mean(~np.isfinite(qq)):>9.1%}{miss_r:>9.1%}{longest_r:>9}"
              f"{np.nanmax(qq) if np.isfinite(qq).any() else float('nan'):>11.1f}")

    # 有效训练样本量：AORC 段只 S.18 多得，MRMS 段两边一样，只数一遍。
    print("\n有效训练样本（回看72h+目标1h 全有效）：")
    print(f"{'站':<16}{'AORC段(多给S.18)':>18}{'MRMS段(两边共有)':>18}")
    for sid in SITES:
        q = flow_series(sid)
        n_a = valid_samples(q, t18[:aorc_hi])
        n_m = valid_samples(q, t14[:mrms14_hi])
        print(f"{NAMES[sid]:<6}{sid:<9}{n_a:>14,}{n_m:>18,}")


if __name__ == "__main__":
    main()
