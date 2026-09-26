#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐小时起报的密集滚动推理（当前仅支持最优模型 S.10 MoE，gstd 变换）。

与存档评估（runs/*/predictions.npz，起报点每 24h 一个）的区别：
1. 起报原点逐小时铺满测试段 → 各提前量曲线逐小时连续，无 24h 抽样折线问题；
2. 滚动语义：原点 t 出 R 步，滚到第 k 步时输入窗 = 实测 t-72+k..t + 预报
   t+1..t+k-1（滑窗前进，与"分块重锚"数学等价）；
3. 计分口径修正：目标小时缺测仅跳过该小时（存档协议是整窗丢弃，大洪水窗口
   被系统性剔除——审计发现 A）；历史窗 72h 仍要求完整（模型输入需要）；
4. 持续性基准用锚点流量（t0+71 实测），不用 obs[:,0]（那是 t0+72，偷看 1h）。

产物：runs/<run>/predictions_dense.npz（obs 含 NaN=缺测未计分；full=目标窗
12 小时全干净的样本标记，便于与存档口径分解对比）。

用法: python scripts/rollout_dense.py [--run moe] [--roll 12]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train import (MoENet, SiteDataset, apply_overrides, build_statics,
                   load_config, load_inputs, make_inv, make_loader, nse)

LEADS = [1, 3, 6, 12]          # 用户指定的时间尺度
STRIDE = 1                     # 起报原点逐小时
PARAM_EXPECT = {"moe": 162.7, "step12": 162.7}  # 存档日志参数量（千），构造自检


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="moe",
                    choices=["moe", "step12"], help="run 目录名（runs/site_model_<run>）")
    ap.add_argument("--roll", type=int, default=max(LEADS))
    args = ap.parse_args()
    R = args.roll
    run_dir = os.path.join(ROOT, "runs", f"site_model_{args.run}")

    # S.10 的训练配置（来自存档日志头核对）：gstd / quantile×3 / 单步+滚动
    cfg = apply_overrides(load_config(os.path.join(ROOT, "configs/pipeline.yaml")),
                          ["model.arch=moe", "model.horizon=1",
                           "model.transform=gstd", "model.loss=quantile",
                           'model.split_dates=["2023-01-01","2024-01-01"]'])
    m = cfg["model"]
    lookback = int(m["lookback"])
    use_spatial = bool(m["use_spatial"])
    use_future = bool(m["use_future_rain"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids, names, areas, times, area_rain, flow = load_inputs(cfg)
    mask1km = np.load(os.path.join(ROOT, cfg["paths"]["catchments"]),
                      allow_pickle=True)["mask1km"].astype(np.float32)
    statics = build_statics(cfg, mask1km)
    n_site, T = flow.shape
    print(f"断面 {n_site} 个  时间轴 {times[0]} ~ {times[-1]}  共 {T} 小时")

    cs = None
    if use_spatial:
        import xarray as xr
        rain_grid = xr.open_dataset(os.path.join(ROOT, cfg["paths"]["rain_nc"]))["rain"].values
        cs = np.nan_to_num(rain_grid.astype(np.float32), nan=0.0).cumsum(axis=0)
        del rain_grid

    n_win = T - lookback - R + 1
    sd = m.get("split_dates")
    b1 = min(max(int(times.searchsorted(pd.Timestamp(sd[0]))), 1), n_win - 1)
    b2 = min(max(int(times.searchsorted(pd.Timestamp(sd[1]))), b1 + 1), n_win)
    bounds = {"train": (0, b1), "val": (b1, b2), "test": (b2, n_win)}
    n_tr = bounds["train"][1]
    print(f"按日期划分  train 至 {times[b1]}  |  val 至 {times[b2]}  |  test 至 {times[-1]}")

    # gstd：全局标准化参数与训练时同式（训练段 pooled）
    flow_pos = np.maximum(flow, 0).astype(np.float64)
    gm = float(np.nanmean(flow_pos[:, :n_tr]))
    gs = float(np.nanstd(flow_pos[:, :n_tr])) or 1.0
    q_mean = np.full(n_site, gm)
    q_std = np.full(n_site, gs)
    flow_n = (flow_pos - gm) / gs
    r_mean = float(np.nanmean(area_rain[:, :n_tr]))
    r_std = float(np.nanstd(area_rain[:, :n_tr])) or 1.0
    rain_n = (area_rain - r_mean) / r_std
    print(f"全局均值 {gm:.2f}  全局标准差 {gs:.2f} m³/s")

    # 密集样本：原点逐小时；历史窗(72h流量+72h+R降雨)须完整，目标窗允许缺测
    lo, hi = bounds["test"]
    ss = []
    for i in range(n_site):
        for t0 in range(lo, hi, STRIDE):
            h = flow_n[i, t0:t0 + lookback]
            r = rain_n[i, t0:t0 + lookback + R]
            if np.isfinite(h).all() and np.isfinite(r).all():
                ss.append((i, t0))
    print(f"密集测试样本（逐小时起报）{len(ss)} 个"
          f"（存档协议 {4819} 个）")
    by_t0 = {}
    for i, t0 in ss:
        by_t0.setdefault(int(t0), []).append(int(i))

    # 空数据集仅用于把 cs/statics 挪到 GPU（与 train.py 评估路径同法）
    ds0 = SiteDataset(cs, statics, flow_n, rain_n,
                      np.zeros((0, 2), dtype=np.int64),
                      use_spatial, use_future, 1, lookback)
    make_loader(ds0, 1, False, device)
    cs_t, st_t = ds0.cs, ds0.statics

    model = MoENet(use_spatial, bool(m.get("use_site", False)),
                   int(m["hidden"]), int(m["spatial_dim"]), 1, n_quant=3,
                   masked_pool=bool(m.get("masked_pool", False)),
                   delta_cap=float(m.get("delta_cap", 0.0)),
                   areas=areas).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e3
    exp = PARAM_EXPECT[args.run]
    print(f"参数量 {n_par:.1f} 千（存档 {exp} 千）")
    if abs(n_par - exp) > 1.0:
        sys.exit("参数量与存档不符，模型构造参数有误，终止")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"),
                                     map_location=device))
    model.eval()

    obs_l, sim_l, site_l, t0_l = [], [], [], []
    flow_t = torch.from_numpy(np.nan_to_num(flow_n, nan=0.0)).float().to(device)
    rain_t = torch.from_numpy(np.nan_to_num(rain_n, nan=0.0)).float().to(device)
    with torch.no_grad():
        for t0 in sorted(by_t0):
            idx = torch.tensor(by_t0[t0], device=device)
            B = len(idx)
            cur = flow_t[idx, t0:t0 + lookback].clone()
            sims = []
            for k in range(R):
                t = t0 + lookback - 1 + k
                mk_ = st_t[idx][:, 0]
                rnow = cs_t[t] - (cs_t[t - 1] if t > 0 else 0.0)
                lo2 = max(0, t - 71)
                base = cs_t[lo2 - 1] if lo2 > 0 else 0.0
                rain72 = cs_t[t] - base
                x = torch.stack([rnow * mk_, rain72 * mk_], 1)
                fut = ((cs_t[t + 1] - cs_t[t]) * mk_).unsqueeze(1)
                if not use_future:
                    fut = torch.zeros_like(fut)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                fut = torch.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)
                hist = cur.unsqueeze(-1)
                nxt = cur[:, -1] + model(x, hist, fut, idx)[:, 0, model.mid]
                cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], 1)
                sims.append(nxt)
            sim_l.append(torch.stack(sims, 1).cpu().numpy())
            ii = idx.cpu().numpy()
            obs_l.append(flow_n[ii, t0 + lookback:t0 + lookback + R])
            site_l.append(ii)
            t0_l.append(np.full(B, t0, dtype=np.int64))

    obs_n = np.concatenate(obs_l).astype(np.float32)      # 目标窗缺测 = NaN
    sim_n = np.concatenate(sim_l).astype(np.float32)
    st = np.concatenate(site_l)
    t0_arr = np.concatenate(t0_l)
    full = np.isfinite(obs_n).all(axis=1)                  # 旧口径子集（目标窗全干净）
    inv = make_inv("gstd", q_mean, q_std, np.zeros(1))
    np.savez_compressed(os.path.join(run_dir, "predictions_dense.npz"),
                        obs=obs_n, sim=sim_n, site=st, t0=t0_arr,
                        times=np.array([str(t) for t in times]),
                        ids=np.array(ids), names=np.array(names),
                        areas=areas, q_mean=q_mean, q_std=q_std,
                        transform="gstd", full=full, R=R)
    print(f"已写 {run_dir}/predictions_dense.npz  "
          f"样本 {len(st)}（其中目标窗全干净 {int(full.sum())}）")

    # ---------- 指标（缺测小时逐点跳过；持续性 = 锚点流量 t0+71 实测）----------
    def lead_nse(i, L):
        mk = st == i
        oo, ss_ = inv(obs_n[mk, L - 1], i), inv(sim_n[mk, L - 1], i)
        fin = np.isfinite(oo) & np.isfinite(ss_)
        return nse(oo[fin], ss_[fin]) if fin.sum() >= 4 else np.nan

    print(f"\n{'站号':11s} {'名称':10s} " +
          "  ".join(f"NSE@{L}h" for L in LEADS) + "  （全起点·缺测逐点跳过）")
    rows = []
    for i in range(n_site):
        r = [lead_nse(i, L) for L in LEADS]
        rows.append(r)
        print(f"{ids[i]:11s} {names[i]:10s} " +
              "  ".join(f"{v:7.3f}" for v in r))
    med = np.nanmedian(np.asarray(rows), axis=0)
    print("中位        " + "  ".join(f"{v:7.3f}" for v in med))

    # 旧口径分解：仅目标窗全干净样本
    rows_full = []
    for i in range(n_site):
        mk = (st == i) & full
        r = []
        for L in LEADS:
            oo, ss_ = inv(obs_n[mk, L - 1], i), inv(sim_n[mk, L - 1], i)
            fin = np.isfinite(oo) & np.isfinite(ss_)
            r.append(nse(oo[fin], ss_[fin]) if fin.sum() >= 4 else np.nan)
        rows_full.append(r)
    med_full = np.nanmedian(np.asarray(rows_full), axis=0)
    print("\n同上网格但仅目标窗全干净样本（≈旧剔除口径，供分解）：")
    print("中位        " + "  ".join(f"{v:7.3f}" for v in med_full))


if __name__ == "__main__":
    main()
