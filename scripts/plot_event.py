#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""洪水事件全过程对比图：每个站取测试段最大洪水事件，窗口覆盖涨水-峰-退水全程。

每站一张图，行=模型。每行画：
  黑实线  实测流量（全程）
  彩实线  提前 24h 预报（每个目标时刻取 24h 前起报的那一次预报，连续曲线）
  彩浅线  提前 12h 预报
  灰虚线  持续性预报（提前 24h，即锚点流量平推）
  绿色副轴 面雨量柱状
行标题附：事件窗内 24h 预报 NSE。

用法: python scripts/plot_event.py [--out experiments/forecast_event]
       [--models "S.4 单步+分位,S.10 MoE双专家"] [--pre 144] [--post 96]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

MODELS = [  # 与 plot_compare.py 同步
    ("S.4 单步+分位", "runs/site_model_step4", "#6a3d9a"),
    ("S.5 DLinear", "runs/site_model_dlinear", "#984ea3"),
    ("S.7 掩膜池化", "runs/site_model_dlmcnn_mp", "#a65628"),
    ("S.10 MoE双专家", "runs/site_model_moe", "#1b9e77"),
    ("S.11 MoE+掩膜池", "runs/site_model_moe_mp", "#e7298a"),
]
LOOKBACK = 72
LEADS = [1, 3, 6, 12, 24]  # 画连续提前量曲线的步数（小时）
# 提前量配色：近=红暖色，远=蓝冷色（RdYlBu_r 取色），一眼区分
LEAD_COL = {1: "#a50026", 3: "#f46d43", 6: "#fdae61", 12: "#74add1", 24: "#4575b4"}


def load_run(run_dir):
    p = np.load(os.path.join(ROOT, run_dir, "predictions.npz"), allow_pickle=True)
    from train import make_inv
    tf = str(p["transform"]) if "transform" in p.files else "log1p"
    lams = p["lams"] if "lams" in p.files else np.zeros(1)
    return {"obs": p["obs"], "sim": p["sim"], "site": p["site"], "t0": p["t0"],
            "inv": make_inv(tf, p["q_mean"], p["q_std"], lams)}


def lead_series(d, i, lead, q_raw, times):
    """提前 lead 小时的连续预报序列：目标时刻 j 的预报 = 起报 t0=j-72-lead+1 样本的第 lead-1 步。

    返回 (目标时刻索引数组, 预报值数组, 持续性数组, 该窗实测数组)。
    """
    k = lead - 1  # 0-based 步
    mk = d["site"] == i
    t0 = d["t0"][mk]
    tgt = t0 + LOOKBACK + k
    sim = d["inv"](d["sim"][mk][:, k], i)
    pers = d["inv"](d["obs"][mk][:, 0], i)  # 锚点流量 = 持续性预报
    return tgt, sim, pers


def win_nse(obs, fc):
    m = np.isfinite(obs) & np.isfinite(fc)
    if m.sum() < 4:
        return np.nan
    o, s = obs[m], fc[m]
    v = np.nansum((o - np.nanmean(o)) ** 2)
    return 1 - np.nansum((o - s) ** 2) / v if v > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/forecast_event")
    ap.add_argument("--models", default=None,
                    help="逗号分隔的行标签，默认全部")
    ap.add_argument("--pre", type=int, default=144, help="峰前小时数（默认 6 天）")
    ap.add_argument("--post", type=int, default=96, help="峰后小时数（默认 4 天）")
    args = ap.parse_args()

    models = MODELS
    if args.models:
        want = set(args.models.split(","))
        models = [m for m in MODELS if m[0] in want]

    runs = []
    for lab, rd, col in models:
        if os.path.isfile(os.path.join(ROOT, rd, "predictions.npz")):
            runs.append((lab, load_run(rd), col))
        else:
            print(f"跳过（尚无预测文件）：{lab} {rd}")

    a = np.load(os.path.join(ROOT, "data", "area_rain.npz"), allow_pickle=True)
    area_rain = a["area_rain"].astype(np.float64)
    rain_times = pd.DatetimeIndex([str(x) for x in a["times"]])

    meta = np.load(os.path.join(ROOT, "runs/site_model_gstd", "predictions.npz"),
                   allow_pickle=True)
    times = pd.DatetimeIndex([str(x) for x in meta["times"]])
    ids, names, areas = [str(x) for x in meta["ids"]], list(meta["names"]), meta["areas"]

    flows = {}
    for sid in ids:
        q = pd.read_csv(os.path.join(ROOT, "data", "sites", f"{sid}.csv"),
                        index_col=0, parse_dates=True)["flow_m3s"]
        if getattr(q.index, "tz", None) is not None:
            q.index = q.index.tz_localize(None)
        flows[sid] = q.reindex(times).values

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(ids)):
        q_raw = flows[ids[i]]
        # 事件 = 测试段（2024）最大峰；窗口 = 峰前 pre h ~ 峰后 post h
        t_test = np.where(times >= "2024-01-01")[0]
        pk = t_test[np.nanargmax(q_raw[t_test])]
        lo, hi = max(0, pk - args.pre), min(len(times), pk + args.post)
        tt = times[lo:hi]
        obs_win = q_raw[lo:hi]

        fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs) + 0.8),
                                 sharex=True)
        fig.subplots_adjust(left=0.07, right=0.92, top=0.90, bottom=0.10, hspace=0.35)
        if len(runs) == 1:
            axes = [axes]

        n_lt_max = 0
        for r, (lab, d, col) in enumerate(runs):
            ax = axes[r]
            ax.plot(tt, obs_win, color="black", lw=1.8, label="实测")
            # 每条起报的完整 24h 逐小时预报轨迹（细线，显示模型真实输出的
            # 时间粒度； stitched 折线是 12h 抽样连线的画图假象）
            mk = d["site"] == i
            t0s = d["t0"][mk]
            sel = (t0s + LOOKBACK >= lo - 24) & (t0s + LOOKBACK < hi)
            for a, row in zip(t0s[sel], d["sim"][mk][sel]):
                tt_a = times[a + LOOKBACK: a + LOOKBACK + len(row)]
                ax.plot(tt_a, d["inv"](row, i), color=col, lw=0.7, alpha=0.30)
            nse24 = np.nan
            n_lt = 0
            for lead in LEADS:
                tgt, sim, pers = lead_series(d, i, lead, q_raw, times)
                m = (tgt >= lo) & (tgt < hi)
                if not m.any():
                    continue
                tj, sj = times[tgt[m]], sim[m]
                ax.plot(tj, sj, color=LEAD_COL[lead], lw=1.6, marker="o", ms=2.5,
                        label=f"提前{lead}h")
                if lead == max(LEADS):
                    nse24 = win_nse(q_raw[tgt[m]], sj)
                    n_lt = int(np.isfinite(q_raw[tgt[m]]).sum())
                    n_lt_max = max(n_lt_max, n_lt)
                    ax.plot(tj, pers[m], color="#999999", lw=1.0, ls="--",
                            label=f"持续性({lead}h)")
            ax.axvline(times[pk], color="#bbbbbb", lw=0.8, ls=":")
            lab_nse = (f"24h窗NSE {nse24:.2f}" if np.isfinite(nse24)
                       else f"24h预报点{n_lt}个")
            ax.set_ylabel(f"{lab}\n{lab_nse}", fontsize=9)
            ax.grid(alpha=0.25)
            axr = ax.twinx()
            axr.fill_between(tt, area_rain[i, lo:hi], color="#4dac26",
                             alpha=0.3, step="mid")
            axr.set_ylim(0, max(8, np.nanmax(area_rain[i, lo:hi]) * 4))
            axr.set_yticks([])
            if r == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=6)
                ax.set_title(f"事件峰 {times[pk]:%Y-%m-%d %H时}  "
                             f"实测峰 {q_raw[pk]:.0f} m³/s", fontsize=10)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.text(0.995, 0.5, "绿=面雨量 mm/h", rotation=90, va="center",
                 fontsize=8, color="#4dac26")
        warn = ""
        if n_lt_max < 14:
            warn = f"  ⚠ 事件窗内仅 {n_lt_max}/20 个预报点（本站记录缺测多，稀疏段无预报）"
        fig.suptitle(f"{ids[i]} {names[i]}  {areas[i]:.0f} km²  "
                     f"洪水事件全过程（涨水-峰-退水）× {len(runs)} 模型{warn}",
                     fontsize=12)
        out = os.path.join(out_dir, f"{ids[i]}_{names[i]}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"已写 {out}  峰 {times[pk]:%Y-%m-%d %H时} {q_raw[pk]:.0f} m³/s")


if __name__ == "__main__":
    main()
