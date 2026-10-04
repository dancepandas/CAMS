#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S.14 与 S.18（延长训练段）在 2024 年测试段的逐项对比。

两个模型的密集滚动结果都来自 scripts/rollout_dense.py，口径一致：
2024 年逐小时起报，每次连续滚动 24 小时。

两个必须注意的对齐问题：

1. **时间轴起点不同。** S.14 的时间轴从 2015-06-01 起（84,025 小时），S.18 从
   1990-01-01 起（306,793 小时）。两边 npz 里的 ``t0`` 是各自轴上的**下标**，
   同一个整数对应的物理时刻完全不同。所以这里按「站号 + 起报的物理时刻」对齐，
   不能按 (站, t0) 对齐。

2. **峰现偏差距的门槛。** 判一场 24 小时窗是不是真有涨水，用
   (窗内极差) / (K × 该站训练段流量标准差) 与 0.25 比较（K=3.0，与训练奖励
   同一套）。分母若各取各的训练段，S.14（2015 起）和 S.18（1990 起）筛出的
   窗口不是同一批，相比不公平。所以**固定用同一个窗口**（--scale-window，默认
   取 S.14 的训练段）算标准差，两边共用一道门槛。该比值里归一化的均值与标准差
   会自动约掉，标准差直接用物理流量（m³/s）算即可。

指标一律调用 train.py 里的 nse / kge 和 plot_step17.py 里的 peak_bias，
不另写一套，免得口径走样。

用法:
  python scripts/compare_step18.py \
      --a runs/site_model_step14/predictions_dense_R24.npz \
      --b runs/site_model_step18/predictions_dense_test_R24.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plot_step17 import peak_bias          # noqa: E402
from train import kge, make_inv, nse       # noqa: E402

LOOKBACK = 72
BIG_SITES = ["03455000", "03454500", "03453500", "03451500"]
LEADS = [1, 3, 6, 12, 24]


def load(path: str) -> dict:
    p = np.load(os.path.join(ROOT, path), allow_pickle=True)
    return {"sim": p["sim"], "obs": p["obs"], "site": p["site"], "t0": p["t0"],
            "names": [str(x) for x in p["names"]], "ids": [str(x) for x in p["ids"]],
            "times": pd.DatetimeIndex([str(x) for x in p["times"]]),
            "q_mean": p["q_mean"].astype(float), "q_std": p["q_std"].astype(float),
            "transform": str(p["transform"]), "lams": p["lams"]}


def read_flow(ids: list[str]) -> pd.DataFrame:
    """从站点 CSV 读实测流量（物理单位），按时刻对齐成一张表。"""
    cols = {}
    for sid in ids:
        q = pd.read_csv(os.path.join(ROOT, "data", "sites", f"{sid}.csv"),
                        index_col=0, parse_dates=True)["flow_m3s"]
        if getattr(q.index, "tz", None) is not None:
            q.index = q.index.tz_localize(None)
        cols[sid] = q[~q.index.duplicated(keep="first")].sort_index()
    return pd.DataFrame(cols)


def phys(d: dict, site_idx: np.ndarray, a: np.ndarray) -> np.ndarray:
    """标准化值还原成物理流量。直接调 train.py 的 make_inv，不另写一遍。

    gstd 的逆变换除了 x*σ+μ，还会把负流量截到 0（make_inv 里那句 np.maximum），
    漏掉这一步 KGE 会差到小数点后第四位，NSE / MAE 却几乎看不出来。
    """
    inv = make_inv(d["transform"], d["q_mean"], d["q_std"], d["lams"])
    out = np.empty(a.shape, dtype=np.float64)
    for i in range(len(d["ids"])):
        m = site_idx == i
        if m.any():
            out[m] = inv(a[m], i)
    return out


def safe_nse(o: np.ndarray, s: np.ndarray) -> float:
    good = np.isfinite(o) & np.isfinite(s)
    return nse(o[good], s[good]) if good.sum() >= 4 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/site_model_step14/predictions_dense_R24.npz")
    ap.add_argument("--b", default="runs/site_model_step18/predictions_dense_test_R24.npz")
    ap.add_argument("--label-a", default="S.14 基准")
    ap.add_argument("--label-b", default="S.18 长数据")
    ap.add_argument("--scale-window", default="2015-06-01,2023-01-01",
                    help="算峰现门槛用的固定窗口（含起、不含止），两个模型共用")
    args = ap.parse_args()
    la, lb = args.label_a, args.label_b

    A, B = load(args.a), load(args.b)
    if A["ids"] != B["ids"]:
        raise ValueError("两个结果的站点顺序不一致，不能直接比")
    ids, names = A["ids"], A["names"]
    n_site = len(ids)

    # 按「站号 + 起报的物理时刻」对齐：两边的 t0 是各自时间轴上的下标，不可互比。
    key_a = {(int(s), A["times"][int(t)]): i
             for i, (s, t) in enumerate(zip(A["site"], A["t0"]))}
    key_b = {(int(s), B["times"][int(t)]): i
             for i, (s, t) in enumerate(zip(B["site"], B["t0"]))}
    common = sorted(set(key_a) & set(key_b))
    print(f"共有起报 {len(common)} 条（{la} {len(key_a)}，{lb} {len(key_b)}）")
    if len(common) < 100:
        raise ValueError("共有起报太少，两个结果的测试段口径可能不同")

    ia = np.array([key_a[c] for c in common])
    ib = np.array([key_b[c] for c in common])
    site = A["site"][ia].astype(int)
    obs = phys(A, site, A["obs"][ia])
    sims = {la: phys(A, site, A["sim"][ia]),
            lb: phys(B, site, B["sim"][ib])}

    # 持续性预报：起报前那一小时的实测流量，一直平推 24 小时
    # （与 rollout_dense.robust_metrics 里的 anchor 同一个时刻）。
    flow = read_flow(ids)
    anchor_t = A["times"][A["t0"][ia] + LOOKBACK - 1]
    anchor_q = flow.reindex(anchor_t)[pd.Index(ids)].values      # (n, 站)

    # ---------- 逐站指标 ----------
    stats = {}
    print(f"\n{'站号':<11s} {'名称':<8s} " + " ".join(
        f"{l:>10s}" for l in (la, lb)) + f" {'NSE变化':>9s}")
    print("-" * 72)
    per = {}
    for lab, sim in ((la, sims[la]), (lb, sims[lb])):
        rows = {}
        for i in range(n_site):
            take = site == i
            if not take.any():
                continue
            oo, ss = obs[take], sim[take]
            pp = np.repeat(anchor_q[take, i][:, None], obs.shape[1], axis=1)
            good = np.isfinite(oo) & np.isfinite(ss)
            if good.sum() < 4:
                continue
            o, s, p = oo[good], ss[good], pp[good]
            rows[i] = {
                "nse": nse(o, s), "pers": nse(o, p),
                "kge": kge(o, s)[0],
                "mae": float(np.mean(np.abs(o - s))),
                "rmse": float(np.sqrt(np.mean((o - s) ** 2))),
                "bias": float(np.mean(s - o)),
                "lead": {L: safe_nse(oo[:, L - 1], ss[:, L - 1]) for L in LEADS},
            }
        per[lab] = rows

    for i in range(n_site):
        va = per[la][i]["nse"] if i in per[la] else np.nan
        vb = per[lb][i]["nse"] if i in per[lb] else np.nan
        print(f"{ids[i]:<11s} {names[i]:<8s} {va:>10.4f} {vb:>10.4f} {vb - va:>+9.4f}")

    print(f"\n{'指标':<26s} {la:>12s} {lb:>12s} {'变化':>10s}")
    print("-" * 64)
    for key, name, fmt in [("nse", "24h 整体中位 NSE", "{:>12.4f}"),
                           ("kge", "中位 KGE", "{:>12.4f}"),
                           ("mae", "中位 MAE（m³/s）", "{:>12.2f}"),
                           ("rmse", "中位 RMSE（m³/s）", "{:>12.2f}"),
                           ("bias", "中位偏差（m³/s）", "{:>12.2f}")]:
        va = np.nanmedian([r[key] for r in per[la].values()])
        vb = np.nanmedian([r[key] for r in per[lb].values()])
        print(f"{name:<26s} {fmt.format(va)} {fmt.format(vb)} {vb - va:>+10.4f}")
    for L in LEADS:
        va = np.nanmedian([r["lead"][L] for r in per[la].values()])
        vb = np.nanmedian([r["lead"][L] for r in per[lb].values()])
        print(f"{'+' + str(L) + 'h 中位 NSE':<26s} {va:>12.4f} {vb:>12.4f} {vb - va:>+10.4f}")
    wa = int(np.sum([r["nse"] > r["pers"] for r in per[la].values()]))
    wb = int(np.sum([r["nse"] > r["pers"] for r in per[lb].values()]))
    print(f"{'胜过持续性预报的站数':<26s} {wa:>9d}/{n_site} {wb:>9d}/{n_site} "
          f"{wb - wa:>+10d}")

    # ---------- 洪峰低估 ----------
    # 逐站先各自取「实测最高 5%」的分位，把 sim 和 obs 在这批点上的总量相除。
    # 这是 plot_step17.py 里 (d) 子图的算法，四个大站的数字与实验记录一致。
    print("\n洪峰低估：实测最高 5% 流量上的平均偏差（%，负=报低）")
    print(f"{'断面':<14s} {la:>12s} {lb:>12s} {'变化(百分点)':>14s}")
    print("-" * 58)
    hf_site = {la: {}, lb: {}}
    for lab, sim in ((la, sims[la]), (lb, sims[lb])):
        for i in range(n_site):
            o = obs[site == i].ravel()
            v = np.isfinite(o)
            if not v.any():
                continue
            hi = v & (o >= np.nanquantile(o[v], 0.95))
            den = float(np.nansum(o[hi]))
            hf_site[lab][i] = (100 * (float(np.nansum(sim[site == i].ravel()[hi])) / den - 1)
                               if den else np.nan)
    for sid in BIG_SITES:
        i = ids.index(sid)
        va, vb = hf_site[la][i], hf_site[lb][i]
        print(f"{names[i]:<14s} {va:>11.2f}% {vb:>11.2f}% {vb - va:>+13.2f}")
    for lab in (la, lb):
        num = den = 0.0
        for i in range(n_site):
            o = obs[site == i].ravel()
            v = np.isfinite(o)
            hi = v & (o >= np.nanquantile(o[v], 0.95))
            num += float(np.nansum(sims[lab][site == i].ravel()[hi]))
            den += float(np.nansum(o[hi]))
        hf_site[lab]["all"] = 100 * (num / den - 1)
    print(f"{'全部15站(合并)':<14s} {hf_site[la]['all']:>11.2f}% "
          f"{hf_site[lb]['all']:>11.2f}% {hf_site[lb]['all'] - hf_site[la]['all']:>+13.2f}")
    ma = np.nanmedian(list(hf_site[la][i] for i in range(n_site)))
    mb = np.nanmedian(list(hf_site[lb][i] for i in range(n_site)))
    print(f"{'15站(逐站中位)':<14s} {ma:>11.2f}% {mb:>11.2f}% {mb - ma:>+13.2f}")

    # ---------- 峰现偏差 ----------
    lo_s, hi_s = args.scale_window.split(",")
    seg = flow.loc[(flow.index >= pd.Timestamp(lo_s)) & (flow.index < pd.Timestamp(hi_s))]
    scale = seg.std().reindex(ids).values
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    print(f"\n峰现时刻偏差（小时，逐站取场次中位；门槛固定用 {lo_s}~{hi_s} 的标准差）")
    print(f"{'断面':<14s} {la:>12s} {lb:>12s} {'变化':>10s}")
    print("-" * 54)
    pk = {la: [], lb: []}
    for lab, sim in ((la, sims[la]), (lb, sims[lb])):
        for i in range(n_site):
            pk[lab].append(np.nanmedian(np.abs(
                peak_bias(obs[site == i], sim[site == i], scale[i])[0])))
    for i in np.argsort(pk[la]):
        print(f"{names[i]:<14s} {pk[la][i]:>12.2f} {pk[lb][i]:>12.2f} "
              f"{pk[lb][i] - pk[la][i]:>+10.2f}")
    ma, mb = float(np.nanmedian(pk[la])), float(np.nanmedian(pk[lb]))
    print("-" * 54)
    print(f"{'15 站中位':<14s} {ma:>12.2f} {mb:>12.2f} {mb - ma:>+10.2f}")
    print("\n门槛共用同一个窗口，两边筛出的场次完全一致，可直接比。")


if __name__ == "__main__":
    main()
