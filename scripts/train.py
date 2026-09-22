#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""断面流量预报：训练与评估。

一个共享权重的模型同时服务流域内多个断面，预报未来若干小时的流量过程。
输入分三路：历史流量与历史面雨量（时序）、预见期内的面雨量（时序）、
汇水区内的多尺度累积降雨图与静态场（空间）。目标为逐站标准化后的对数流量，
模型预测相对窗口末刻的增量。

用法: python3 scripts/pipeline/train.py [--config configs/pipeline.yaml]
                                        [--set model.epochs=8]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, items):
    for kv in items:
        k, v = kv.split("=", 1)
        node, keys = cfg, k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        node[keys[-1]] = v
    return cfg


def load_inputs(cfg):
    """读站点表、面雨量、各断面流量，按共同时间轴对齐。"""
    p = cfg["paths"]
    sites = pd.read_csv(os.path.join(ROOT, p["sites_csv"]), encoding="utf-8",
                        dtype={"site_id": str})
    ids = [str(s) for s in sites["site_id"]]
    names = list(sites["name"])
    areas = sites["area_km2"].to_numpy(dtype=np.float64)

    a = np.load(os.path.join(ROOT, p["area_rain"]), allow_pickle=True)
    times = pd.DatetimeIndex([str(t) for t in a["times"]])
    area_rain = a["area_rain"].astype(np.float64)
    if area_rain.shape[0] != len(ids):
        raise ValueError(f"面雨量断面数 {area_rain.shape[0]} 与站点表 {len(ids)} 不符")

    flow = np.full_like(area_rain, np.nan)
    for i, sid in enumerate(ids):
        f = os.path.join(ROOT, p["sites"], f"{sid}.csv")
        q = pd.read_csv(f, index_col=0, parse_dates=True)["flow_m3s"]
        q.index = q.index.tz_localize(None)
        flow[i] = q.reindex(times).values
    return ids, names, areas, times, area_rain, flow


def build_statics(cfg, mask1km):
    """每个断面的静态场：汇水区掩膜、对数汇流累积、坡度、高程。"""
    p = cfg["paths"]
    G = int(cfg["basin"]["grid"])
    t = np.load(os.path.join(ROOT, p["terrain"]))
    up, slope, elev = t["uparea"], t["slope"], t["elevtn"]
    step = min(up.shape[0] // G, up.shape[1] // G)
    nr, nc = G * step, G * step
    agg = lambda a: a[:nr, :nc].reshape(G, step, G, step).mean(axis=(1, 3))

    static = np.stack([np.log1p(agg(up)) / 5.0, agg(slope) / 45.0,
                       (agg(elev) - 900.0) / 500.0])[None]
    static = static.repeat(len(mask1km), axis=0)
    statics = np.concatenate([mask1km[:, None], static], axis=1)
    return statics.astype(np.float32)


class SiteDataset(Dataset):
    """按需组装空间输入，避免把全部样本预先展开成巨大数组。"""

    def __init__(self, cs, statics, flow_n, rain_n, samples, cum_hours,
                 use_spatial, use_future, horizon, lookback):
        self.cs, self.statics = cs, statics
        self.flow_n, self.rain_n, self.samples = flow_n, rain_n, samples
        self.cum_hours = cum_hours
        self.use_spatial, self.use_future = use_spatial, use_future
        self.horizon, self.lookback = horizon, lookback

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        i, t0 = self.samples[k]
        t = t0 + self.lookback - 1
        hist = self.flow_n[i, t0:t0 + self.lookback]
        tgt = self.flow_n[i, t0 + self.lookback:t0 + self.lookback + self.horizon]
        delta = tgt - hist[-1]

        if self.use_spatial:
            chans = []
            for h in self.cum_hours:
                lo = max(0, t - h + 1)
                base = self.cs[lo - 1] if lo > 0 else 0.0
                chans.append((self.cs[t] - base) / (t - lo + 1))
            x = np.concatenate([np.stack(chans), self.statics[i]], axis=0)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = np.zeros((1, 1, 1), dtype=np.float32)

        hrain = self.rain_n[i, t0:t0 + self.lookback]
        frain = self.rain_n[i, t0 + self.lookback:t0 + self.lookback + self.horizon]
        if not self.use_future:
            frain = np.zeros_like(frain)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(np.stack([hist, hrain], axis=-1), dtype=torch.float32),
                torch.tensor(frain, dtype=torch.float32),
                i, torch.tensor(delta, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self, n_spatial_ch, n_cum, use_spatial, use_site, hidden,
                 spatial_dim, horizon):
        super().__init__()
        self.use_spatial, self.use_site = use_spatial, use_site
        self.n_cum = n_cum          # 掩膜通道在空间输入中的下标
        if use_spatial:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_spatial_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.proj = nn.Linear(32, spatial_dim)
        self.lstm = nn.LSTM(2, hidden, batch_first=True)     # 历史流量与历史面雨量
        self.lstm_f = nn.LSTM(1, 32, batch_first=True)       # 预见期内的面雨量
        fused = (hidden + (spatial_dim if use_spatial else 0)
                 + (3 if use_site else 0) + 32)
        self.head = nn.Sequential(nn.Linear(fused, 96), nn.ReLU(),
                                  nn.Linear(96, horizon))

    def forward(self, x, hist, frain, site):
        _, (h, _) = self.lstm(hist)
        feats = [h[-1]]
        _, (hf, _) = self.lstm_f(frain.unsqueeze(-1))
        feats.append(hf[-1])
        if self.use_spatial:
            f = self.cnn(x)
            # 掩膜排在累积降雨之后，用它在汇水区内做加权池化
            m = torch.nn.functional.adaptive_avg_pool2d(
                x[:, self.n_cum:self.n_cum + 1], f.shape[-2:])
            w = m.clamp(min=0)
            pooled = (f * w).sum(dim=(2, 3)) / w.sum(dim=(2, 3)).clamp(min=1e-6)
            feats.append(self.proj(pooled))
        if self.use_site:
            feats.append(site)
        return self.head(torch.cat(feats, dim=-1))


def nse(obs, sim):
    return float(1 - ((obs - sim) ** 2).sum()
                 / max(((obs - obs.mean()) ** 2).sum(), 1e-9))


def kge(obs, sim):
    r = float(np.corrcoef(obs, sim)[0, 1])
    a = float(sim.std() / max(obs.std(), 1e-9))
    b = float(sim.mean() / max(obs.mean(), 1e-9))
    return 1 - float(np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)), r, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="key=value")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(os.path.join(ROOT, args.config)), args.set)

    m = cfg["model"]
    lookback, horizon = int(m["lookback"]), int(m["horizon"])
    stride, split = int(m["stride"]), list(m["split"])
    cum_hours = [int(h) for h in m["cum_hours"]]
    use_spatial, use_future, use_site = (bool(m["use_spatial"]),
                                         bool(m["use_future_rain"]),
                                         bool(m.get("use_site", False)))
    torch.manual_seed(int(m["seed"]))
    np.random.seed(int(m["seed"]))

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
        print(f"累积降雨数组 {cs.shape}  {cs.nbytes / 1e9:.2f} GB")

    n_win = T - lookback - horizon + 1
    n_tr = int(n_win * split[0])
    n_va = int(n_win * split[1])
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va),
              "test": (n_tr + n_va, n_win)}

    # 逐站标准化：对数流量减本站训练期均值、除本站训练期标准差。
    # 这样各断面目标分布一致，共享模型只需学一套映射。
    log_flow = np.log1p(np.maximum(flow, 0))
    q_mean = np.nanmean(log_flow[:, :n_tr], axis=1)
    q_std = np.nanstd(log_flow[:, :n_tr], axis=1)
    q_std[q_std < 1e-6] = 1.0
    flow_n = (log_flow - q_mean[:, None]) / q_std[:, None]

    r_mean = float(np.nanmean(area_rain[:, :n_tr]))
    r_std = float(np.nanstd(area_rain[:, :n_tr])) or 1.0
    rain_n = (area_rain - r_mean) / r_std

    samples = {}
    for sp, (lo, hi) in bounds.items():
        step = stride if sp == "train" else horizon
        ss = []
        for i in range(n_site):
            for t0 in range(lo, hi, step):
                h = flow_n[i, t0:t0 + lookback]
                y = flow_n[i, t0 + lookback:t0 + lookback + horizon]
                r = rain_n[i, t0:t0 + lookback + horizon]
                if (np.isfinite(h).all() and np.isfinite(y).all()
                        and np.isfinite(r).all()):
                    ss.append((i, t0))
        samples[sp] = ss
        print(f"{sp:5s} {len(ss)} 个样本")

    ds = {k: SiteDataset(cs, statics, flow_n, rain_n, v, cum_hours,
                         use_spatial, use_future, horizon, lookback)
          for k, v in samples.items()}
    ldr = {k: DataLoader(v, batch_size=int(m["batch_size"]),
                         shuffle=(k == "train")) for k, v in ds.items()}

    model = Net(len(cum_hours) + 4, len(cum_hours), use_spatial, use_site,
                int(m["hidden"]), int(m["spatial_dim"]), horizon)
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e3:.1f} 千  "
          f"空间分支 {'开' if use_spatial else '关'}  "
          f"预见期降雨 {'有' if use_future else '无'}")

    out_dir = os.path.join(ROOT, cfg["paths"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(m["epochs"]))
    lossf = nn.MSELoss()

    def run(loader, train):
        model.train(train)
        tot, n = 0.0, 0
        for x, hist, frain, idx, y in loader:
            pred = model(x, hist, frain, None)
            loss = lossf(pred, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        return tot / max(n, 1)

    best, best_state, wait = np.inf, None, 0
    for ep in range(1, int(m["epochs"]) + 1):
        tr = run(ldr["train"], True)
        sched.step()
        va = run(ldr["val"], False)
        if va < best:
            best, wait = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  第 {ep:3d} 轮  训练 {tr:.4f}  验证 {va:.4f}")
        if wait >= int(m["patience"]):
            print(f"  第 {ep} 轮早停")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    # ---------- 评估 ----------
    model.eval()
    obs_l, sim_l, site_l = [], [], []
    with torch.no_grad():
        for x, hist, frain, idx, y in ldr["test"]:
            delta = model(x, hist, frain, None)
            anchor = hist[:, -1, 0:1]
            obs_l.append((y + anchor).numpy())
            sim_l.append((delta + anchor).numpy())
            site_l.append(idx.numpy())
    obs_n, sim_n = np.concatenate(obs_l), np.concatenate(sim_l)
    st = np.concatenate(site_l)
    inv = lambda z, i: np.expm1(z * q_std[i] + q_mean[i])

    print(f"\n{'站号':11s} {'名称':10s} {'面积':>6s} {'NSE':>7s} {'持续NSE':>8s} | "
          f"{'KGE':>7s} {'持续KGE':>8s} | {'r':>5s} {'变率比':>7s} {'偏置比':>7s} | "
          f"{'峰误差':>8s}")
    rows, bo, bs = [], [], []
    for i in range(n_site):
        mk = st == i
        if mk.sum() == 0:
            continue
        obs = inv(obs_n[mk], i).ravel()
        sim = inv(sim_n[mk], i).ravel()
        per = inv(np.repeat(obs_n[mk][:, :1], horizon, axis=1), i).ravel()
        a, b = nse(obs, sim), nse(obs, per)
        ka, r, al, be = kge(obs, sim)
        kp = kge(obs, per)[0]
        op = inv(obs_n[mk], i).max(axis=1)
        sp = inv(sim_n[mk], i).max(axis=1)
        big = op >= np.quantile(op, 0.9)
        if big.sum():
            bo.append(inv(obs_n[mk], i)[big].ravel())
            bs.append(inv(sim_n[mk], i)[big].ravel())
        pkb = np.median((sp[big] - op[big]) / np.maximum(op[big], 1e-6)) if big.sum() else np.nan
        rows.append((a, b, ka, kp))
        print(f"{ids[i]:11s} {names[i]:10s} {areas[i]:6.0f} {a:7.3f} {b:8.3f} | "
              f"{ka:7.3f} {kp:8.3f} | {r:5.2f} {al:7.2f} {be:7.2f} | {pkb:7.1%}")

    arr = np.array(rows)
    summary = {
        "nse_median": float(np.median(arr[:, 0])),
        "nse_persist_median": float(np.median(arr[:, 1])),
        "nse_win": int((arr[:, 0] > arr[:, 1]).sum()),
        "kge_median": float(np.median(arr[:, 2])),
        "kge_persist_median": float(np.median(arr[:, 3])),
        "kge_win": int((arr[:, 2] > arr[:, 3]).sum()),
        "n_sites": len(arr),
    }
    if bo:
        summary["nse_peak"] = nse(np.concatenate(bo), np.concatenate(bs))
    print(f"\n中位 NSE {summary['nse_median']:.3f}  持续性 {summary['nse_persist_median']:.3f}  "
          f"胜出 {summary['nse_win']}/{summary['n_sites']}")
    print(f"中位 KGE {summary['kge_median']:.3f}  持续性 {summary['kge_persist_median']:.3f}  "
          f"胜出 {summary['kge_win']}/{summary['n_sites']}")
    if "nse_peak" in summary:
        print(f"大洪水时段 NSE {summary['nse_peak']:.3f}")

    np.savez_compressed(
        os.path.join(out_dir, "predictions.npz"),
        obs=obs_n, sim=sim_n, site=st, t0=np.array([t0 for _, t0 in samples["test"]]),
        times=np.array([str(t) for t in times]), q_mean=q_mean, q_std=q_std,
        ids=np.array(ids), names=np.array(names), areas=areas,
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {cfg['paths']['out_dir']}")


if __name__ == "__main__":
    main()
