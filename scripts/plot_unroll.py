#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S.20 闭环训练四版对比图：S.18 修正版（老冠军）/ S.18 闭环版 / S.20 单边罚 / S.20 非对称罚。

背景：S.18 修正版（长历史降雨）治好了一步预报和洪峰低估，但 level 模式滚动
24 步时三个小站出现正向漂移（米尔斯河 24h 偏差 +55%、比特里溪 +135%、
弗莱彻 +26%）——训练只看真实历史，滚动时输入换成自己的预测，小误差复利放大。
S.18 闭环版在训练时把模型自己的预测喂回历史窗（断梯度），损失打在滚动每一步，
专门治这个毛病。

本脚本只读三套 dense 推理存档，按"站号 + 物理起报时刻"对齐后画图（对齐口径
与 compare_step18.py 完全一致：S.14 的时间轴从 2015 起，S.18/闭环版从 1990
起，t0 下标不可互比，必须用物理起报时刻 times[t0+LOOKBACK-1]）。

整套图想回答三个问题：

  1. 闭环训练把 24h 深处的漂移治好了吗？精度随提前量怎么变？（汇总图 a）
  2. 治漂移的代价（洪峰重新低估、短提前量略降）有多大？（汇总图 d、e、f）
  3. 三个小站的事件过程线上，漂移是不是真消失了？（事件图，含三小站）

产出（默认 experiments/step18_compare/）：

  summary_s20.png   六联总览：分提前量 NSE、峰现时间偏差分布、逐站峰现偏差、
                       高流量低估、逐站 NSE 增减（闭环减S.18修正版）、涨/退水段偏差
  event_s20_<站>.png  大站+三小站最大洪水的 24h 预报连续曲线，三个模型叠加

不重复画 coverage 图：闭环版与 S.18 修正版用同一份训练数据（1990 起），
训练经历覆盖完全一样，见 coverage_rainfix.png。

用法: python scripts/plot_unroll.py [--out experiments/step18_compare]
       [--events 03455000,03446000]   # 默认四大站 + 三小站共 7 张
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

# 各模型存档目录；文件名由 --split 决定（test=2024 超极端年 / val=2023 正常年）。
# 出 2023 版的动机：2023 同样是权重没见过的年份，但洪水量级正常，正好和
# 2024（海伦飓风超极端洪峰）配成"正常 vs 极端"一对故事，避免单放 2024
# 让精度数字显得整体偏差大。
DIRS = {"S.18 修正版(1990起)": "runs/site_model_step18_rainfix",
        "S.18 闭环版": "runs/site_model_step18_unroll",
        "S.20 单边罚": "runs/site_model_step20_bptt_up",
        "S.20 非对称罚": "runs/site_model_step20_bptt_asym"}
COLORS = {"S.18 修正版(1990起)": "#d73027", "S.18 闭环版": "#f4a582",
          "S.20 单边罚": "#1b9e77", "S.20 非对称罚": "#6a3d9a"}
# 四版同口径读档：S.18 修正版（旧冠军）→ 闭环版（第一次治漂移）→
# S.20 单边罚（拆捷径，破纪录）→ S.20 非对称罚（达标线最全）。S.14 与
# 对称 BPTT 的对照数字在实验报告里，图里塞五根柱就太花了。
# BASE/NEW 在 main() 里按 models 列表现取；模块级只留名字，含义：
# BASE = 主要参照系（S.18 修正版，教师强迫训练的老冠军）
# NEW  = 关键被评对象（S.20 非对称罚，达标线最全的新冠军候选）

LOOKBACK = 72            # 回看小时数，与训练一致
GATE_MIN = 0.25          # 与 rollout_dense.peak_timing_metrics 同一条门槛
BIG_SITES = ["03455000", "03454500", "03453500", "03451500"]
SMALL_SITES = ["03446000", "03450000", "03447687"]   # 漂移三小站，事件图必须看
EV_LEADS = [6, 12, 24]   # 事件过程图画哪几个提前量（各占一行）


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
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="test=2024 超极端年（默认）；val=2023 正常年")
    ap.add_argument("--models", default=None,
                    help='覆盖默认四版，格式 "名称=runs/目录,名称2=runs/目录2"，'
                         "用于定稿流水线画任意模型组合（如底模 vs 定稿）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--events", default=None,
                    help="逗号分隔的站号；默认四大站 + 三小站")
    args = ap.parse_args()

    suffix = "s20" if args.split == "test" else f"s20_{args.split}"
    if args.models:
        # 定稿流水线用法：颜色按固定调色板循环分配，不再绑定写死的四版
        palette = ["#d73027", "#1b9e77", "#7570b3", "#e6ab02", "#666666"]
        pairs = [kv.split("=", 1) for kv in args.models.split(",")]
        models = [(lab.strip(), f"{d.strip()}/predictions_dense_{args.split}.npz",
                   palette[j % len(palette)])
                  for j, (lab, d) in enumerate(pairs)]
    else:
        models = [(lab, f"{d}/predictions_dense_{args.split}.npz", COLORS[lab])
                  for lab, d in DIRS.items()]
    year = 2024 if args.split == "test" else 2023
    # 不同年份的图分文件夹放，避免混在一起：2024 版进 step18_compare，
    # 2023 版单独进 step18_val_compare（用户要求各年份图件独立成册）
    out_arg = args.out or ("experiments/step18_compare" if args.split == "test"
                           else "experiments/step18_val_compare")
    data = {lab: load(p) for lab, p, _ in models}
    BASE, NEW = models[0][0], models[-1][0]   # 参照系=首个模型；被评=最后一个
    ref = data[models[0][0]]
    ids, names = ref["ids"], ref["names"]
    times = pd.DatetimeIndex([str(x) for x in ref["times"]])
    flow = read_flow(ids, ref["times"])

    # 每站常态波动尺度：全局归一化下固定阈值对大小站不公平。
    flow_n = np.stack([(flow[i] - ref["q_mean"][i]) / ref["q_std"][i]
                       for i in range(len(ids))])
    scale = site_activity_scale(flow_n, times)

    out_dir = os.path.join(ROOT, out_arg)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 汇总指标（按"站号 + 物理起报时刻"对齐，不用 t0 下标） ----
    # 各模型时间轴长度不同（S.14 从 2015 起，另两套从 1990 起），同一物理
    # 时刻的 t0 整数下标完全不同，直接比下标会全军覆没。物理起报时刻 =
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

    site = ref["site"][pick[models[0][0]]].astype(int)
    obs = phys(ref, site, ref["obs"][pick[models[0][0]]])
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

    for lab, _, col in models:
        ax[0, 0].plot(leads, curves[lab], color=col, lw=1.8, marker="o", ms=3, label=lab)
    ax[0, 0].set_xlabel("预报提前量（小时）")
    ax[0, 0].set_ylabel("中位 NSE")
    ax[0, 0].set_title("(a) 精度随提前量（中位 NSE）", fontsize=11)
    ax[0, 0].set_xticks([1, 3, 6, 12, 18, 24])
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].grid(alpha=0.3)

    for lab, _, col in models:
        v = bias[lab][0]
        ax[0, 1].hist(v, bins=np.arange(-12, 12.5, 1), density=True, histtype="step",
                      lw=2.0, color=col,
                      label=f"{lab}  按站中位 {bias_site[lab]:.2f}h")
    ax[0, 1].axvline(0, color="black", lw=0.9, ls="--")
    ax[0, 1].set_xlabel("预测洪峰出现时刻 − 实测（小时）")
    ax[0, 1].set_ylabel("占比")
    ax[0, 1].set_title("(b) 峰现时刻偏差分布（小时）", fontsize=11)
    ax[0, 1].legend(fontsize=9)
    ax[0, 1].grid(alpha=0.3)

    # 逐站峰现偏差：看闭环版是不是全局变好，而不只是几个站。
    per_site = {lab: [np.nanmedian(np.abs(peak_bias(obs[site == i],
                                                    sims[lab][site == i], scale[i])[0]))
                      for i in range(len(ids))] for lab in data}
    o2 = np.argsort(per_site[BASE])
    w = 0.2    # 四个模型并排，柱宽收窄
    for j, (lab, _, col) in enumerate(models):
        ax[0, 2].bar(np.arange(len(ids)) + (j - 1.5) * w,
                     [per_site[lab][i] for i in o2], w, color=col, label=lab)
    ax[0, 2].set_xticks(np.arange(len(ids)))
    ax[0, 2].set_xticklabels([names[i] for i in o2], rotation=60, ha="right", fontsize=7)
    ax[0, 2].set_ylabel("峰现时刻偏差（小时，按站中位）")
    v0, v1 = np.array(per_site[BASE]), np.array(per_site[NEW])
    n_better = int(np.sum(v1 < v0))
    ax[0, 2].set_title(f"(c) 逐站峰现偏差：非对称罚 {n_better}/{len(ids)} 站优于 S.18 修正版",
                       fontsize=10)
    ax[0, 2].legend(fontsize=9)
    ax[0, 2].grid(alpha=0.3, axis="y")

    w = 0.2
    show = BIG_SITES + [None]      # 四个大站 + 全部合计
    for j, (lab, _, col) in enumerate(models):
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
        ax[1, 0].bar(np.arange(len(show)) + (j - 1.5) * w, vals, w, color=col, label=lab)
    ax[1, 0].set_xticks(np.arange(len(show)))
    ax[1, 0].set_xticklabels([names[ids.index(s)] for s in BIG_SITES] + ["全部15站"],
                             rotation=20, ha="right", fontsize=9)
    ax[1, 0].axhline(0, color="black", lw=0.9)
    ax[1, 0].set_ylabel("高流量平均偏差（%）")
    ax[1, 0].set_title("(d) 高流量（各站最高 5%）平均偏差", fontsize=11)
    ax[1, 0].legend(fontsize=9)
    ax[1, 0].grid(alpha=0.3, axis="y")

    # 逐站 24h 整体 NSE 增减：闭环版相对 S.18 修正版。漂移三站应是巨幅正数。
    order = np.argsort([nse(obs[site == i], sims[NEW][site == i])
                        - nse(obs[site == i], sims[BASE][site == i])
                        for i in range(len(ids))])
    d = [nse(obs[site == i], sims[NEW][site == i])
         - nse(obs[site == i], sims[BASE][site == i]) for i in range(len(ids))]
    cols = ["#d73027" if d[i] < 0 else "#1b9e77" for i in order]
    ax[1, 1].barh(np.arange(len(ids)), [d[i] for i in order], color=cols,
                  label=f"{NEW} − {BASE}", alpha=0.85)
    ax[1, 1].set_yticks(np.arange(len(ids)))
    ax[1, 1].set_yticklabels([names[i] for i in order], fontsize=8)
    ax[1, 1].axvline(0, color="black", lw=0.9)
    ax[1, 1].set_xlabel("整段 24h 中位 NSE 之差")
    n_up = int(np.sum(np.array(d) > 0))
    ax[1, 1].set_title(f"(e) 逐站整段 24h NSE 增减：{n_up}/{len(ids)} 站变好",
                       fontsize=11)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].grid(alpha=0.3, axis="x")

    w = 0.2
    for j, (lab, _, col) in enumerate(models):
        vals = [np.nanmedian(np.abs(bias[lab][1])), np.nanmedian(np.abs(bias[lab][2]))]
        ax[1, 2].bar(np.arange(2) + (j - 1.5) * w,
                     vals, w, color=col, label=lab)
    ax[1, 2].set_xticks([0, 1])
    ax[1, 2].set_xticklabels(["涨水段", "退水段"])
    ax[1, 2].set_ylabel("重心时刻偏差（小时）")
    ax[1, 2].set_title("(f) 涨水段 / 退水段重心时刻偏差", fontsize=11)
    ax[1, 2].legend(fontsize=9)
    ax[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle(f"S.20 闭环训练四版对比（{year} 评估段，同一批 "
                 f"{len(common):,} 条起报）", fontsize=13)
    p = os.path.join(out_dir, f"summary_{suffix}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"已写 {p}")
    for lab in data:
        print(f"  {lab}: 按站中位峰现偏差 {bias_site[lab]:.2f}h  "
              f"涨水段 {np.nanmedian(np.abs(bias[lab][1])):.2f}h  "
              f"退水段 {np.nanmedian(np.abs(bias[lab][2])):.2f}h")

    # ---- 事件过程线：大站看洪峰代价，小站看漂移是否消失 ----
    # 每个提前量一行，行内三个模型同色比较。各模型用各自时间轴（ttm）换算
    # 目标时刻的物理日期——两套时间轴长度不同，借另一套的时间轴会把日期整个挪错。
    tgt_sites = (args.events.split(",") if args.events
                 else BIG_SITES + SMALL_SITES)
    for sid in tgt_sites:
        if sid not in ids:
            print(f"跳过未知站号 {sid}")
            continue
        i = ids.index(sid)
        q = flow[i]
        t_win = np.where((times >= f"{year}-01-01")
                         & (times < f"{year + 1}-01-01"))[0]
        # USGS 实测逐小时资料本身有缺测段，dense 推理只保留观测齐全的起报点，
        # 有效目标时刻在年内是稀疏且不均匀的（2023 年 3 月甚至整月没有）。
        # 若按实测最大峰取窗，很容易落在无预报覆盖的空洞里（事件图画成空白）。
        # 所以先在"有任一模型、任一提前量预报覆盖"的时刻里找峰，保证画得出线。
        have = np.zeros(len(times), dtype=bool)
        for lab, _, _ in models:
            d = data[lab]
            ttm = pd.DatetimeIndex([str(x) for x in d["times"]])
            mk = d["site"] == i
            t0 = d["t0"][mk]
            for L in EV_LEADS:
                have[np.searchsorted(times.values,
                                     ttm[t0 + LOOKBACK + L - 1].values)] = True
        q_masked = np.where(have, q, np.nan)
        t_cov = t_win[np.isfinite(q_masked[t_win])]
        pk = t_cov[np.nanargmax(q_masked[t_cov])]
        lo, hi = max(0, pk - 168), min(len(times), pk + 120)
        tt = times[lo:hi]

        # 各模型各提前量的连续预报：目标时刻 = 起报 + 回看 + (提前量-1)
        series = {}
        for lab, _, _ in models:
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
            for lab, _, col in models:
                tgt, s = series[lab][L]
                m = (tgt >= tt[0]) & (tgt <= tt[-1])
                tm, sm = tgt[m], np.asarray(s[m], dtype=float)
                # 实测缺测段内没有有效起报点，相邻两点的间隔可达数天。
                # 若照常连线，matplotlib 会把断档两端的真实预报点机械地连成
                # 一根跨越数天的假直线（看似"预测出一条直线"，其实那两个点
                # 之间一个预报都没有）。间隔 >3h 处插 NaN 断开。
                if len(tm) > 1:
                    gap = (tm.to_series().diff().dt.total_seconds().to_numpy()
                           > 3 * 3600)
                    sm[gap] = np.nan
                ax.plot(tm, sm, color=col, lw=1.5, label=lab)
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
                ax.legend(loc="upper left", fontsize=8, ncol=5)
            note = "  ".join(f"{lab.split('(')[0]} {t:%m-%d %H时} {v:.0f}"
                             for t, v, _, lab in sorted(peak_note))
            ax.text(0.995, 0.06, f"预报峰时刻/峰值：{note}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#444444")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.suptitle(f"{sid} {names[i]}  {year} 年最大洪水  "
                     f"实测峰 {times[pk]:%m-%d %H时} {q[pk]:.0f} m³/s  "
                     f"（▼ = 各模型预报峰）", fontsize=12)
        p = os.path.join(out_dir, f"event_{suffix}_{sid}_{names[i]}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"已写 {p}")


if __name__ == "__main__":
    main()
