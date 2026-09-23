#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""画测试段各断面的预报过程 vs 实测过程 + 降雨。

对每个断面取测试段若干代表性窗口（含其最大洪峰窗口），画三条曲线：
  - 实测流量（黑）
  - 模型预报（蓝，从窗口起点起报预见期小时）
  - 持续性预报（灰，平推窗口末刻）
  - 面雨量柱状（顶部，绿）

用法: python scripts/plot_forecast.py [--run runs/site_model] [--out experiments/forecast_check]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load(run_dir):
    p = np.load(os.path.join(ROOT, run_dir, "predictions.npz"), allow_pickle=True)
    d = {
        "obs": p["obs"], "sim": p["sim"], "site": p["site"], "t0": p["t0"],
        "times": pd.DatetimeIndex([str(x) for x in p["times"]]),
        "q_mean": p["q_mean"], "q_std": p["q_std"],
        "ids": [str(x) for x in p["ids"]], "names": list(p["names"]),
        "areas": p["areas"],
    }
    # 目标变换的逆：log1p 用 expm1，yeo-johnson 走 sklearn，std 直接线性还原
    from train import make_inv
    tf = str(p["transform"]) if "transform" in p.files else "log1p"
    lams = p["lams"] if "lams" in p.files else np.zeros(1)
    d["inv"] = make_inv(tf, d["q_mean"], d["q_std"], lams)
    a = np.load(os.path.join(ROOT, "data", "area_rain.npz"), allow_pickle=True)
    d["area_rain"] = a["area_rain"].astype(np.float64)
    d["rain_times"] = pd.DatetimeIndex([str(x) for x in a["times"]])
    return d


def site_windows(d, i, horizon, n_peaks=2, n_base=2, lookback=72):
    """挑该断面的代表性测试窗口：最大洪峰若干 + 基流若干。"""
    mk = np.where(d["site"] == i)[0]
    t0s = d["t0"][mk]
    obs_pk = d["obs"][mk].max(axis=1)          # 标准化空间的峰值
    order = np.argsort(obs_pk)[::-1]
    picks = []
    seen = set()
    for k in order[:n_peaks * 4]:
        if len(picks) >= n_peaks:
            break
        t0 = int(t0s[k])
        # 避免窗口重叠
        if any(abs(t0 - s) < 2 * horizon for s in seen):
            continue
        picks.append(t0)
        seen.add(t0)
    # 基流窗口：取测试段中部的低峰
    mid = len(order) // 2
    for k in order[mid:]:
        if len(picks) >= n_peaks + n_base:
            break
        t0 = int(t0s[k])
        if any(abs(t0 - s) < 2 * horizon for s in seen):
            continue
        picks.append(t0)
        seen.add(t0)
    return sorted(picks)


def plot_site(ax, d, i, t0, horizon, show_legend=False):
    times = d["times"]
    qm, qs = d["q_mean"][i], d["q_std"][i]
    lo = t0 - 72
    hi = t0 + horizon
    tt = times[lo:hi]
    # 实测流量过程（标准化反变换）
    q_obs = np.expm1(d["flow_n"][i, lo:hi] * qs + qm) if "flow_n" in d else None
    # 预报
    mk = np.where((d["site"] == i) & (d["t0"] == t0))[0]
    if len(mk) == 0:
        return
    k = mk[0]
    anchor = d["obs"][k, 0]
    obs_h = np.expm1((d["obs"][k] ) * qs + qm)       # 未来 horizon 实测（相对末刻）
    sim_h = np.expm1((d["sim"][k]) * qs + qm)
    per_h = np.expm1(np.repeat(d["obs"][k, :1], horizon) * qs + qm)
    anchor_q = np.expm1(anchor * qs + qm)
    t_anchor = times[t0 + 71]
    t_future = times[t0 + 72:t0 + 72 + horizon]

    # 历史段
    hist_t = times[lo:t0 + 72]
    hist_q = np.expm1(d["obs_hist"][i, lo:t0 + 72] * qs + qm) if "obs_hist" in d else None

    ax.plot(hist_t, hist_q, color="black", lw=1.2, label="实测(历史)")
    ax.plot([t_anchor, *t_future], [anchor_q, *obs_h], color="black", lw=1.6,
            label="实测(未来)")
    ax.plot([t_anchor, *t_future], [anchor_q, *sim_h], color="#2166ac", lw=1.4,
            label="模型预报")
    ax.plot([t_anchor, *t_future], [anchor_q, *per_h], color="#999999", lw=1.0,
            ls="--", label="持续性")
    ax.set_ylabel("流量 m³/s")
    ax.grid(alpha=0.3)
    ax.set_title(f"{d['names'][i]}  {d['areas'][i]:.0f} km²  起点 {t_anchor:%m-%d %H时}",
                 fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/site_model")
    ap.add_argument("--out", default="experiments/forecast_check")
    ap.add_argument("--sites", default=None, help="逗号分隔的站号，默认全部")
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    d = load(args.run)
    horizon = d["obs"].shape[1]
    # 反标准化历史流量需要 flow_n；从 area_rain.npz 无法得，改从预测里拼
    # 直接用 obs/sim 重建历史：obs 是 (n_sample, horizon) 相对末刻增量，
    # 历史段从原始流量 CSV 取更稳妥
    import glob
    flow_raw = {}
    for sid in d["ids"]:
        f = os.path.join(ROOT, "data", "sites", f"{sid}.csv")
        q = pd.read_csv(f, index_col=0, parse_dates=True)["flow_m3s"]
        q.index = q.index.tz_localize(None)
        flow_raw[sid] = q.reindex(d["times"])

    d["flow_raw"] = flow_raw
    d["obs_hist"] = None  # 用 flow_raw 代替

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    idx = list(range(len(d["ids"])))
    if args.sites:
        want = set(args.sites.split(","))
        idx = [i for i in idx if d["ids"][i] in want]

    for i in idx:
        wins = site_windows(d, i, horizon)
        fig, axes = plt.subplots(len(wins), 1, figsize=(11, 3.2 * len(wins)),
                                 sharex=False)
        if len(wins) == 1:
            axes = [axes]
        for ax, t0 in zip(axes, wins):
            # 手绘历史
            times = d["times"]
            qm, qs = d["q_mean"][i], d["q_std"][i]
            lo = t0 - 72
            hi = t0 + horizon
            tt = times[lo:hi]
            q_hist = d["flow_raw"][d["ids"][i]].values[lo:t0 + 72]
            mk = np.where((d["site"] == i) & (d["t0"] == t0))[0]
            if len(mk) == 0:
                continue
            k = mk[0]
            anchor_q = d["flow_raw"][d["ids"][i]].values[t0 + 71]
            obs_h = d["inv"](d["obs"][k], i)
            sim_h = d["inv"](d["sim"][k], i)
            per_h = d["inv"](np.repeat(d["obs"][k, :1], horizon), i)
            t_anchor = times[t0 + 71]
            t_future = times[t0 + 72:t0 + 72 + horizon]
            t_hist = times[lo:t0 + 72]

            ax.plot(t_hist, q_hist, color="black", lw=1.2, label="实测(历史)")
            ax.plot([t_anchor, *t_future], [anchor_q, *obs_h], color="black",
                    lw=1.8, label="实测(未来)")
            ax.plot([t_anchor, *t_future], [anchor_q, *sim_h], color="#2166ac",
                    lw=1.5, label="模型预报")
            ax.plot([t_anchor, *t_future], [anchor_q, *per_h], color="#999999",
                    lw=1.0, ls="--", label="持续性")

            # 面雨量（顶部子轴）
            axr = ax.twinx()
            rain = d["area_rain"][i]
            rmask = (d["rain_times"] >= tt[0]) & (d["rain_times"] <= tt[-1])
            axr.bar(d["rain_times"][rmask], rain[rmask], width=0.04,
                    color="#4dac26", alpha=0.35, label="面雨量")
            axr.set_ylabel("mm/h", color="#4dac26")
            axr.tick_params(axis="y", labelcolor="#4dac26")
            axr.set_ylim(0, max(10, np.nanmax(rain[rmask]) * 3))

            ax.set_ylabel("流量 m³/s")
            ax.grid(alpha=0.3)
            ax.set_title(
                f"{d['names'][i]}  {d['areas'][i]:.0f} km²  "
                f"起点 {t_anchor:%Y-%m-%d %H时}  预见期{horizon}h",
                fontsize=10)
        axes[0].legend(loc="upper left", fontsize=8, ncol=4)
        fig.tight_layout()
        out = os.path.join(out_dir, f"{d['ids'][i]}_{d['names'][i]}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"已写 {out}")


if __name__ == "__main__":
    main()
