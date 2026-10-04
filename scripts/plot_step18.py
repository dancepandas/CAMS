#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S.14 修正版 / S.18 修正版 公平对比图。

背景：坐标修正（汇水区南北颠倒 + MRMS 经纬度各偏半格）和 AORC 失败静默补零
修好之后，用同一套修正数据公平重训了两套模型——

  - S.14 修正版：降雨只有 2015-06~2024（MRMS 一段）
  - S.18 修正版：降雨 1990-01~2024（1990~2015 用 AORC，2015-06 起用 MRMS）

本脚本只读两套 dense 推理存档，按"站号 + 物理起报时刻"对齐后画图。
整套图想回答三个问题：

  1. 精度随提前量怎么变？（汇总图 a）
  2. 洪峰低估（Helene 的核心病灶）到底好到什么程度？（汇总图 d、e）
  3. 2024 最大洪水（Helene）过程线上，两套预报和实测差在哪？（事件图）

产出（默认 experiments/step18_compare/）：

  summary.png        六联总览：分提前量 NSE、峰现时间偏差分布、逐站峰现偏差、
                     高流量低估、逐站 NSE 增减、涨/退水段时间偏差
  coverage.png       训练经历覆盖：各模型训练段见过的最大流量 vs 2024 洪峰
  event_<站>.png     各大站最大洪水的 24h 预报连续曲线，两个模型叠在同一张图上

用法: python scripts/plot_step18.py [--out experiments/step18_compare]
       [--events 03455000,03454500]   # 只画指定站；默认画四个大站
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
from step13_rl import PEAK_TIME_GATE_K, site_activity_scale  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

S14 = "runs/site_model_step14_rainfix/predictions_dense_test.npz"
S18 = "runs/site_model_step18_rainfix/predictions_dense_test.npz"

MODELS = [("S.14 修正版(2015起)", S14, "#1f78b4"),
          ("S.18 修正版(1990起)", S18, "#d73027")]

LOOKBACK = 72          # 回看小时数，与训练一致
GATE_MIN = 0.25        # 与 rollout_dense.peak_timing_metrics 同一条门槛
BIG_SITES = ["03455000", "03454500", "03453500", "03451500"]
EV_LEADS = [6, 12, 24]     # 事件过程图画哪几个提前量（各占一行）
# 两套模型的训练段起点不同：S.14 只能见 2015-06 之后，S.18 能回溯到 1990。
TRAIN_START = {MODELS[0][0]: "2015-06-01", MODELS[1][0]: "1990-01-01"}


def load(path):
    p = np.load(os.path.join(ROOT, path), allow_pickle=True)
    return {"sim": p["sim"], "obs": p["obs"], "site": p["site"],
            "t0": p["t0"], "names": [str(x) for x in p["names"]],
            "ids": [str(x) for x in p["ids"]], "times": p["times"],
            "q_mean": p["q_mean"].astype(float), "q_std": p["q_std"].astype(float)}


def phys(d, site_idx, a):
    """归一化值还原成物理流量。gstd 就是 (x-mu)/sigma，直接乘回来。"""
    return a * d["q_std"][site_idx, None] + d["q_mean"][site_idx, None]


def read_flow(ids, times):
    """从站点 CSV 读实测，对齐到模型的时间轴。"""
    tt = pd.DatetimeIndex([str(x) for x in times])
    out = np.full((len(ids), len(tt)), np.nan)
    for i, sid in enumerate(ids):
        q = pd.read_csv(os.path.join(ROOT, "data", "sites", f"{sid}.csv"),
                        index_col=0, parse_dates=True)["flow_m3s"]
        if getattr(q.index, "tz", None) is not None:
            q.index = q.index.tz_localize(None)
        out[i] = q.reindex(tt).values
    return out


def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 4:
        return np.nan
    o, s = o[m], s[m]
    v = np.sum((o - o.mean()) ** 2)
    return 1 - np.sum((o - s) ** 2) / v if v > 1e-12 else np.nan


def peak_bias(obs, sim, scale):
    """峰现时间偏差，只保留实测确有涨水的场次，和训练奖励同一套门槛。

    返回 (峰现偏差, 涨水段重心偏差, 退水段重心偏差, 预测重心时刻, 实测重心时刻)，
    后两项用于画校准曲线。重心 = 窗口内高出最低水位那部分的时间重心。
    """
    good = np.isfinite(obs)
    keep = good.sum(1) >= 4
    empty = (np.array([]),) * 5
    if not keep.any():
        return empty
    o, s, g = obs[keep], sim[keep], good[keep]
    oo = np.where(g, o, np.nan)
    gate = np.clip((np.nanmax(oo, 1) - np.nanmin(oo, 1))
                   / (PEAK_TIME_GATE_K * max(scale, 1e-6)), 0.0, 1.0)
    t = np.arange(o.shape[1], dtype=float)[None, :]
    base = np.nanmin(oo, axis=1)[:, None]
    wo = np.where(g, np.maximum(o - base, 0.0), 0.0)
    ws = np.where(g, np.maximum(s - base, 0.0), 0.0)

    def cen(w, seg=None):
        ww = w if seg is None else w * seg
        return (ww * t).sum(1) / np.where(ww.sum(1) > 1e-6, ww.sum(1), np.nan)

    to, ts = cen(wo), cen(ws)
    sel = np.where(gate >= GATE_MIN)[0]
    half = np.clip(np.rint(np.nan_to_num(to[sel], nan=12)).astype(int), 1, 21)
    rise = (t <= half[:, None]).astype(float)
    fall = (t >= half[:, None]).astype(float)
    out = (ts[sel] - to[sel],
           cen(ws[sel], rise) - cen(wo[sel], rise),
           cen(ws[sel], fall) - cen(wo[sel], fall),
           ts[sel], to[sel])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/step18_compare")
    ap.add_argument("--events", default=None, help="逗号分隔的站号，默认四个大站")
    args = ap.parse_args()

    data = {lab: load(p) for lab, p, _ in MODELS}
    ref = data[MODELS[0][0]]
    ids, names = ref["ids"], ref["names"]
    times = pd.DatetimeIndex([str(x) for x in ref["times"]])
    flow = read_flow(ids, ref["times"])

    # 每站常态波动尺度：全局归一化下固定阈值对大小站不公平。
    flow_n = np.stack([(flow[i] - ref["q_mean"][i]) / ref["q_std"][i]
                       for i in range(len(ids))])
    scale = site_activity_scale(flow_n, times)

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 汇总指标（按"站号 + 物理起报时刻"对齐，不用 t0 下标） ----
    # 两套模型的时间轴长度不同（S.18 多了 1990~2015），同一物理时刻的 t0
    # 整数下标完全不一样，直接比下标会全军覆没。物理起报时刻 =
    # times[t0 + LOOKBACK - 1]，与 compare_step18.py 同一套对齐口径。
    keys = {}
    for lab, d in data.items():
        tt = pd.DatetimeIndex([str(x) for x in d["times"]])
        issue = tt[d["t0"] + LOOKBACK - 1]
        keys[lab] = {(int(s), str(t)): i for i, (s, t) in
                     enumerate(zip(d["site"], issue))}
    common = sorted(set.intersection(*(set(v) for v in keys.values())))
    print(f"共同起报 {len(common)} 条")
    pick = {lab: np.array([keys[lab][c] for c in common]) for lab in data}

    site = ref["site"][pick[MODELS[0][0]]].astype(int)
    obs = phys(ref, site, ref["obs"][pick[MODELS[0][0]]])
    sims = {lab: phys(d, site, d["sim"][pick[lab]]) for lab, d in data.items()}

    leads = list(range(1, 25))
    curves, bias, bias_site = {}, {}, {}
    for lab in data:
        curves[lab] = [np.nanmedian([nse(obs[site == i, L - 1], sims[lab][site == i, L - 1])
                                     for i in range(len(ids))]) for L in leads]
        per = [peak_bias(obs[site == i], sims[lab][site == i], scale[i]) for i in range(len(ids))]
        bias[lab] = tuple(np.concatenate([p[k] for p in per]) for k in range(5))
        # 逐站先取中位再跨站取中位：站点等权，不让样本多的站主导。
        bias_site[lab] = np.nanmedian([np.nanmedian(np.abs(p[0])) for p in per])

    # ---- 总览六联图 ----
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.91, bottom=0.08,
                        wspace=0.24, hspace=0.30)

    for lab, _, col in MODELS:
        ax[0, 0].plot(leads, curves[lab], color=col, lw=1.8, marker="o", ms=3, label=lab)
    ax[0, 0].set_xlabel("预报提前量（小时）")
    ax[0, 0].set_ylabel("中位 NSE")
    ax[0, 0].set_title("(a) 精度随提前量衰减：S.18 在 2~21h 全面更高", fontsize=11)
    ax[0, 0].set_xticks([1, 3, 6, 12, 18, 24])
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].grid(alpha=0.3)

    for lab, _, col in MODELS:
        v = bias[lab][0]
        ax[0, 1].hist(v, bins=np.arange(-12, 12.5, 1), density=True, histtype="step",
                      lw=2.0, color=col,
                      label=f"{lab}  按站中位 {bias_site[lab]:.2f}h")
    ax[0, 1].axvline(0, color="black", lw=0.9, ls="--")
    ax[0, 1].set_xlabel("预测洪峰出现时刻 − 实测（小时）")
    ax[0, 1].set_ylabel("占比")
    ax[0, 1].set_title("(b) 洪峰出现时刻偏差：S.18 略有变宽", fontsize=11)
    ax[0, 1].legend(fontsize=9)
    ax[0, 1].grid(alpha=0.3)

    # 逐站峰现偏差：看改进到底来自哪些站，还是全局一起变好。
    per_site = {lab: [np.nanmedian(np.abs(peak_bias(obs[site == i],
                                                    sims[lab][site == i], scale[i])[0]))
                      for i in range(len(ids))] for lab in data}
    o2 = np.argsort(per_site[MODELS[0][0]])
    w = 0.32
    for j, (lab, _, col) in enumerate(MODELS):
        ax[0, 2].bar(np.arange(len(ids)) + (j - 0.5) * w,
                     [per_site[lab][i] for i in o2], w, color=col, label=lab)
    ax[0, 2].set_xticks(np.arange(len(ids)))
    ax[0, 2].set_xticklabels([names[i] for i in o2], rotation=60, ha="right", fontsize=7)
    ax[0, 2].set_ylabel("峰现时刻偏差（小时，按站中位）")
    v0, v1 = np.array(per_site[MODELS[0][0]]), np.array(per_site[MODELS[1][0]])
    n_better = int(np.sum(v1 < v0))
    ax[0, 2].set_title(f"(c) 逐站峰现偏差：{n_better}/{len(ids)} 站变好", fontsize=10)
    ax[0, 2].legend(fontsize=9)
    ax[0, 2].grid(alpha=0.3, axis="y")

    w = 0.32
    show = BIG_SITES + [None]      # 四个大站 + 全部合计
    for j, (lab, _, col) in enumerate(MODELS):
        vals = []
        for sid in show:
            idx = list(range(len(ids))) if sid is None else [ids.index(sid)]
            num = den = 0.0
            for i in idx:
                o = obs[site == i].ravel()
                v = np.isfinite(o)
                hi = v & (o >= np.nanquantile(o[v], .95))
                num += np.nansum(sims[lab][site == i].ravel()[hi])
                den += np.nansum(o[hi])
            vals.append(100 * (num / den - 1))
        ax[1, 0].bar(np.arange(len(show)) + (j - 0.5) * w, vals, w, color=col, label=lab)
    ax[1, 0].set_xticks(np.arange(len(show)))
    ax[1, 0].set_xticklabels([names[ids.index(s)] for s in BIG_SITES] + ["全部15站"],
                             rotation=20, ha="right", fontsize=9)
    ax[1, 0].axhline(0, color="black", lw=0.9)
    ax[1, 0].set_ylabel("高流量平均偏差（%）")
    ax[1, 0].set_title("(d) 实测最高 5% 流量的低估：S.18 大幅缓解洪峰低估", fontsize=11)
    ax[1, 0].legend(fontsize=9)
    ax[1, 0].grid(alpha=0.3, axis="y")

    order = np.argsort([nse(obs[site == i], sims[MODELS[0][0]][site == i])
                        for i in range(len(ids))])
    base_lab = MODELS[0][0]
    lab1, _, col1 = MODELS[1]
    d = [nse(obs[site == i], sims[lab1][site == i])
         - nse(obs[site == i], sims[base_lab][site == i]) for i in range(len(ids))]
    cols = ["#d73027" if d[i] < 0 else "#1b9e77" for i in order]
    ax[1, 1].barh(np.arange(len(ids)), [d[i] for i in order], color=cols,
                  label=f"{lab1} − {base_lab}", alpha=0.85)
    ax[1, 1].set_yticks(np.arange(len(ids)))
    ax[1, 1].set_yticklabels([names[i] for i in order], fontsize=8)
    ax[1, 1].axvline(0, color="black", lw=0.9)
    ax[1, 1].set_xlabel("整段 24h 中位 NSE 之差")
    n_up = int(np.sum(np.array(d) > 0))
    ax[1, 1].set_title(f"(e) 逐站精度增减：{n_up}/{len(ids)} 站变好（绿=红多）", fontsize=11)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].grid(alpha=0.3, axis="x")

    w = 0.32
    for j, (lab, _, col) in enumerate(MODELS):
        vals = [np.nanmedian(np.abs(bias[lab][1])), np.nanmedian(np.abs(bias[lab][2]))]
        ax[1, 2].bar([0, 1] + np.arange(2) + (j - 0.5) * w,
                     vals, w, color=col, label=lab)
    ax[1, 2].set_xticks([0, 1])
    ax[1, 2].set_xticklabels(["涨水段", "退水段"])
    ax[1, 2].set_ylabel("重心时刻偏差（小时）")
    ax[1, 2].set_title("(f) 时间项：涨水段 S.18 更准，退水段略差", fontsize=11)
    ax[1, 2].legend(fontsize=9)
    ax[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle("S.14 修正版 vs S.18 修正版公平对比（2024 测试段，同一批 "
                 f"{len(common):,} 条起报）", fontsize=13)
    p = os.path.join(out_dir, "summary_rainfix.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"已写 {p}")
    for lab in data:
        print(f"  {lab}: 按站中位峰现偏差 {bias_site[lab]:.2f}h  "
              f"涨水段 {np.nanmedian(np.abs(bias[lab][1])):.2f}h  "
              f"退水段 {np.nanmedian(np.abs(bias[lab][2])):.2f}h")

    # ---- 训练经历覆盖图：2024 那场洪水，在两个模型的训练段里各见过多大 ----
    yr = pd.DatetimeIndex(times).year
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10, wspace=0.22)

    # 左图：S.18 视角——1990~2022 历史里见过的大洪水相当于 2024 的几成。
    # 这是 S.18 合理的根本：1990 年代也有大事件，S.14 的训练段里完全没有。
    train18 = (times >= TRAIN_START[MODELS[1][0]]) & (times < "2023-01-01")
    ratio = np.array([np.nanmax(flow[i][yr == 2024]) / np.nanmax(flow[i][train18])
                      for i in range(len(ids))])
    o = np.argsort(ratio)
    cols = ["#d73027" if ratio[i] >= 2 else "#4575b4" for i in o]
    ax[0].barh(np.arange(len(ids)), [ratio[i] for i in o], color=cols)
    ax[0].axvline(1.0, color="black", lw=1.0, ls="--")
    ax[0].set_yticks(np.arange(len(ids)))
    ax[0].set_yticklabels([names[i] for i in o], fontsize=9)
    ax[0].set_xlabel("2024 年最大流量 ÷ S.18 训练段（1990~2022）最大流量")
    ax[0].set_title("(a) 加上 1990~2015 后，训练段见过的大洪水多接近 2024", fontsize=11)
    ax[0].grid(alpha=0.3, axis="x")
    for j, i in enumerate(o):
        ax[0].text(ratio[i] + 0.05, j, f"{ratio[i]:.1f}×", va="center", fontsize=8)

    # 右图：训练段流量取值范围 vs 2024 洪峰（S.18 训练段）。
    big = [ids.index(s) for s in BIG_SITES]
    for row, k in enumerate(big):
        t = flow[k][train18]
        t = t[np.isfinite(t)]
        ax[1].plot([t.min(), t.max()], [row, row], color="#bbbbbb", lw=1.2, zorder=1)
        q5, q95 = np.percentile(t, [5, 95])
        ax[1].plot([q5, q95], [row, row], color="#4575b4", lw=7, solid_capstyle="butt",
                   alpha=0.75, zorder=2)
        ax[1].scatter([np.median(t)], [row], color="#1f78b4", s=22, zorder=3)
        pk24 = np.nanmax(flow[k][yr == 2024])
        ax[1].scatter([pk24], [row], color="#d73027", marker="*", s=260,
                      edgecolor="white", lw=0.8, zorder=4)
        ax[1].annotate(f"{pk24:.0f}", (pk24, row), textcoords="offset points",
                       xytext=(0, 11), ha="center", fontsize=8, color="#d73027")
    ax[1].set_yticks(range(len(big)))
    ax[1].set_yticklabels([names[k] for k in big], fontsize=10)
    ax[1].set_ylim(-0.7, len(big) - 0.3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("流量 m³/s（对数轴）")
    ax[1].set_title("(b) S.18 训练段流量范围（蓝条=5~95%；★=2024 洪峰）", fontsize=11)
    ax[1].grid(alpha=0.3, which="both", axis="x")

    fig.suptitle("训练经历覆盖：1990~2015 历史让模型「见过」更接近 2024 的大洪水", fontsize=13)
    p = os.path.join(out_dir, "coverage_rainfix.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"已写 {p}")

    # ---- 事件过程线：每个提前量一行，行内两个模型同色比较 ----
    tgt_sites = (args.events.split(",") if args.events else BIG_SITES)
    for sid in tgt_sites:
        if sid not in ids:
            print(f"跳过未知站号 {sid}")
            continue
        i = ids.index(sid)
        q = flow[i]
        t_test = np.where(times >= "2024-01-01")[0]
        pk = t_test[np.nanargmax(q[t_test])]
        lo, hi = max(0, pk - 168), min(len(times), pk + 120)
        tt = times[lo:hi]

        # 各模型各提前量的连续预报：目标时刻 = 起报 + 回看 + (提前量-1)。
        # 注意下标要落在各自模型的时间轴上，再换算成物理日期画图——
        # 两套时间轴长度不同，借另一套的时间轴会把日期整个挪错。
        series = {}
        for lab, _, _ in MODELS:
            d = data[lab]
            ttm = pd.DatetimeIndex([str(x) for x in d["times"]])
            mk = d["site"] == i
            t0 = d["t0"][mk]
            series[lab] = {L: (ttm[t0 + LOOKBACK + L - 1],
                               phys(d, i, d["sim"][mk][:, L - 1])) for L in EV_LEADS}

        fig, axes = plt.subplots(len(EV_LEADS), 1, figsize=(12, 3.0 * len(EV_LEADS) + 1.0),
                                 sharex=True)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.88,
                            bottom=0.10, hspace=0.18)
        obs_pk_t = times[pk]
        for r, L in enumerate(EV_LEADS):
            ax = axes[r]
            ax.plot(tt, q[lo:hi], color="black", lw=1.9, label="实测", zorder=5)
            ax.axvline(obs_pk_t, color="#bbbbbb", lw=1.0, ls=":")
            peak_note = []
            for lab, _, col in MODELS:
                tgt, s = series[lab][L]
                m = (tgt >= tt[0]) & (tgt <= tt[-1])
                ax.plot(tgt[m], s[m], color=col, lw=1.5, label=lab)
                # 峰附近预报最高点落在哪一刻，以及报到了实测峰的几成
                w = m & (tgt >= obs_pk_t) & (tgt < obs_pk_t + pd.Timedelta(hours=72)) \
                    & (tgt >= obs_pk_t - pd.Timedelta(hours=72)) & np.isfinite(s)
                if w.any():
                    j = np.flatnonzero(w)[np.nanargmax(s[w])]
                    peak_note.append((tgt[j], float(s[w].max()), col, lab))
            for t_j, v_j, col, _ in peak_note:
                ax.scatter([t_j], [v_j], color=col, s=42, marker="v",
                           edgecolor="white", lw=0.6, zorder=6)
            ax.set_ylabel(f"提前{L}h\n流量 m³/s", fontsize=9)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.legend(loc="upper left", fontsize=8, ncol=3)
            note = "  ".join(f"{lab.split('(')[0]} {t:%m-%d %H时} {v:.0f}"
                             for t, v, _, lab in sorted(peak_note))
            ax.text(0.995, 0.06, f"预报峰时刻/峰值：{note}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#444444")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.suptitle(f"{sid} {names[i]}  测试段最大洪水  "
                     f"实测峰 {times[pk]:%m-%d %H时} {q[pk]:.0f} m³/s  "
                     f"（▼ = 各模型预报峰）", fontsize=12)
        p = os.path.join(out_dir, f"event_rainfix_{sid}_{names[i]}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"已写 {p}")


if __name__ == "__main__":
    main()
