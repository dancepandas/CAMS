#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多模型预报对比图：每断面一张图，行=模型，列=代表窗口（最大洪峰 + 基流）。

每个子图画：前 72h 实测流量（前期流量，黑细线）、预见期 12h 实测（黑粗线）、
模型预报（彩色）、持续性预报（灰虚线）、面雨量柱状（绿色副轴）。

用法: python scripts/plot_compare.py [--out experiments/forecast_compare]
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

MODELS = [  # (行标签, runs 目录, 颜色)
    ("C.1 log1p",   "runs/site_model_log1p",    "#7570b3"),
    ("C.2 std",     "runs/site_model_std",      "#e7298a"),
    ("C.3 yj全局",  "runs/site_model_yjg",      "#66a61e"),
    ("C.4 gstd",    "runs/site_model_gstd",     "#e6ab02"),
    ("C.8 gstd+rattn", "runs/site_model_gstd_rattn", "#1b9e77"),
    ("C.9 gstd+E1", "runs/site_model_gstd_nse", "#d95f02"),
]
LOOKBACK, HORIZON = 72, 12


def load_run(run_dir):
    p = np.load(os.path.join(ROOT, run_dir, "predictions.npz"), allow_pickle=True)
    from train import make_inv
    tf = str(p["transform"]) if "transform" in p.files else "log1p"
    lams = p["lams"] if "lams" in p.files else np.zeros(1)
    return {
        "obs": p["obs"], "sim": p["sim"], "site": p["site"], "t0": p["t0"],
        "inv": make_inv(tf, p["q_mean"], p["q_std"], lams),
    }


def site_full_nse(d, i):
    """该站整个测试段的 NSE（全部样本 × 预见期，逆变换到 m³/s）。"""
    mk = d["site"] == i
    o = d["inv"](d["obs"][mk], i).ravel()
    s = d["inv"](d["sim"][mk], i).ravel()
    v = np.nansum((o - np.nanmean(o)) ** 2)
    return 1 - np.nansum((o - s) ** 2) / v if v > 1e-12 else np.nan


def pick_windows(q_raw, t0s):
    """三个代表性窗口：最大洪峰、典型洪水（与洪峰窗错开 ≥40 天）、基流。"""
    with np.errstate(all="ignore"):
        peaks = np.array([np.nanmax(q_raw[t + LOOKBACK:t + LOOKBACK + HORIZON])
                          if np.isfinite(q_raw[t + LOOKBACK:t + LOOKBACK + HORIZON]).any()
                          else -np.inf for t in t0s])
    order = np.argsort(peaks)[::-1]
    flood_t0 = int(t0s[order[0]])
    # 典型洪水：距洪峰窗 ≥40 天（约 960h）内的最大峰
    typ_t0 = None
    for k in order[1:]:
        t = int(t0s[k])
        if abs(t - flood_t0) >= 960:
            typ_t0 = t
            break
    if typ_t0 is None:
        typ_t0 = int(t0s[order[min(3, len(order) - 1)]])
    # 基流：未来 12h 峰值最低
    base_t0 = int(t0s[np.argmin(peaks)])
    return [(flood_t0, "最大洪峰"), (typ_t0, "典型洪水"), (base_t0, "基流")]


def nse12(o, s):  # 保留：整段/大样本用，单窗勿用（分母≈0 病态）
    d = o - s
    v = float(np.nansum((o - np.nanmean(o)) ** 2))
    if not np.isfinite(v) or v < 1e-12:
        return np.nan
    return 1 - float(np.nansum(d * d)) / v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/forecast_compare")
    ap.add_argument("--sites", default=None, help="逗号分隔站号，默认全部")
    ap.add_argument("--models", default="all",
                    help="all=全部模型；best=仅最佳模型 C.9；或逗号分隔的行标签过滤")
    args = ap.parse_args()

    models = MODELS
    if args.models == "best":
        models = [m for m in MODELS if m[0].startswith("C.9")]
    elif args.models != "all":
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
        if flows is not None and getattr(q.index, "tz", None) is not None:
            q.index = q.index.tz_localize(None)
        flows[sid] = q.reindex(times).values

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    idx = list(range(len(ids)))
    if args.sites:
        want = set(args.sites.split(","))
        idx = [i for i in idx if ids[i] in want]

    for i in idx:
        ref = runs[0][1]  # 用首个可用 run 的样本取窗口（各 run 样本一致）
        t0s = ref["t0"][ref["site"] == i]
        q_raw = flows[ids[i]]
        wins = pick_windows(q_raw, t0s)
        # 每模型在该站整段测试的 NSE（权威指标，放行标签）
        full_nse = [site_full_nse(d, i) for _, d, _ in runs]

        fig, axes = plt.subplots(len(runs), len(wins),
                                 figsize=(5.2 * len(wins),
                                          3.8 if len(runs) == 1
                                          else 2.35 * len(runs) + 0.6))
        fig.subplots_adjust(left=0.09, right=0.98,
                            top=0.80 if len(runs) == 1 else 0.93,
                            bottom=0.07, hspace=0.55, wspace=0.14)
        if len(wins) == 1:
            axes = axes[:, None]
        axes = np.asarray(axes)
        if axes.ndim == 1:  # 单模型时 subplots 返回一维
            axes = axes[None, :]

        for r, (lab, d, col) in enumerate(runs):
            for cwt, (t0, wlab) in enumerate(wins):
                ax = axes[r, cwt]
                lo, hi = t0 - LOOKBACK, t0 + LOOKBACK + HORIZON
                tt = times[lo:hi]
                q_hist = q_raw[lo:t0 + LOOKBACK]
                t_anchor = times[t0 + LOOKBACK - 1]
                t_future = times[t0 + LOOKBACK:hi]
                anchor_q = q_raw[t0 + LOOKBACK - 1]

                mk = np.where((d["site"] == i) & (d["t0"] == t0))[0]
                obs_h = d["inv"](d["obs"][mk[0]], i)
                sim_h = d["inv"](d["sim"][mk[0]], i)
                per_h = d["inv"](np.repeat(d["obs"][mk[0], :1], HORIZON), i)
                # 单窗 12 点近常数时 NSE 分母≈0 会病态爆炸，改用平均绝对误差标注
                mae = float(np.nanmean(np.abs(obs_h - sim_h)))

                ax.plot(times[lo:t0 + LOOKBACK], q_hist, color="black", lw=0.9)
                ax.plot([t_anchor, *t_future], [anchor_q, *obs_h], color="black",
                        lw=1.8, label="实测")
                ax.plot([t_anchor, *t_future], [anchor_q, *sim_h], color=col,
                        lw=1.5, label=lab)
                ax.plot([t_anchor, *t_future], [anchor_q, *per_h], color="#999999",
                        lw=0.9, ls="--", label="持续性")
                ax.axvline(t_anchor, color="#bbbbbb", lw=0.6)
                ax.text(0.02, 0.04, f"MAE {mae:.2f} m³/s",
                        transform=ax.transAxes, fontsize=8, color=col)

                axr = ax.twinx()
                axr.fill_between(tt, area_rain[i, lo:hi], color="#4dac26",
                                 alpha=0.3, step="mid")
                axr.set_ylim(0, max(8, np.nanmax(area_rain[i, lo:hi]) * 4))
                axr.set_yticks([])
                if r == 0:
                    pk = np.nanmax(obs_h)
                    ax.set_title(f"{wlab}窗口  "
                                 f"起报 {t_anchor:%Y-%m-%d %H时}  峰 {pk:.1f} m³/s",
                                 fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.grid(alpha=0.25)
                if cwt == 0:
                    ax.set_ylabel(f"{lab}\n本站NSE {full_nse[r]:.2f}",
                                  fontsize=9)
        axes[0, 0].legend(loc="upper left", fontsize=7, ncol=3)
        fig.text(0.995, 0.5, "绿=面雨量 mm/h", rotation=90, va="center",
                 fontsize=8, color="#4dac26")
        who = (f"最佳模型 {runs[0][0]}" if len(runs) == 1
               else f"{len(runs)} 模型")
        fig.suptitle(f"{ids[i]} {names[i]}  {areas[i]:.0f} km²  "
                     f"测试段代表窗口 × {who}", fontsize=12)
        out = os.path.join(out_dir, f"{ids[i]}_{names[i]}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"已写 {out}")


if __name__ == "__main__":
    main()
