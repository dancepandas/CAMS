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
                 use_spatial, use_future, horizon, lookback, weights=None):
        self.cs, self.statics = cs, statics
        self.flow_n, self.rain_n = flow_n, rain_n
        self.samples = np.asarray(samples, dtype=np.int64)  # (n, 2) 便于整批索引
        self.cum_hours = cum_hours
        self.use_spatial, self.use_future = use_spatial, use_future
        self.horizon, self.lookback = horizon, lookback
        # 样本级损失权重（如逐站方差归一化）；默认 1 不改变行为
        self.weights = (np.ones(len(self.samples), dtype=np.float32) if weights is None
                        else np.asarray(weights, dtype=np.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        i, t0 = (int(v) for v in self.samples[k])
        t = t0 + self.lookback - 1
        hist = self.flow_n[i, t0:t0 + self.lookback]
        tgt = self.flow_n[i, t0 + self.lookback:t0 + self.lookback + self.horizon]
        delta = tgt - hist[-1]

        if self.use_spatial:
            # cs 在 GPU 常驻时按通道手工堆叠，避免把整张累积雨图转成 numpy；
            # 仍走 numpy 路径时先一次性取回 GPU 切片，减少反复传输。
            if torch.is_tensor(self.cs):
                chans = []
                for h in self.cum_hours:
                    lo = max(0, t - h + 1)
                    base = self.cs[lo - 1] if lo > 0 else 0.0
                    chans.append(((self.cs[t] - base) / (t - lo + 1)).unsqueeze(0))
                sp = torch.cat(chans, dim=0)                       # (n_cum, G, G)
                st = self.statics[i]
                x = torch.cat([sp.to(st.device), st], dim=0)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                return (x,
                        torch.tensor(np.stack([hist, self.rain_n[i, t0:t0 + self.lookback]], axis=-1),
                                     dtype=torch.float32),
                        torch.tensor(self.rain_n[i, t0 + self.lookback:t0 + self.lookback + self.horizon],
                                     dtype=torch.float32),
                        i,
                        torch.tensor(delta, dtype=torch.float32),
                        self.weights[k])
            cs_t = self.cs[t]
            chans = []
            for h in self.cum_hours:
                lo = max(0, t - h + 1)
                base = self.cs[lo - 1] if lo > 0 else 0.0
                chans.append((cs_t - base) / (t - lo + 1))
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
                i, torch.tensor(delta, dtype=torch.float32),
                self.weights[k])


def make_loader(ds, batch_size, shuffle, device, num_workers=0):
    """组装 DataLoader，并按需把常驻张量挪到 GPU（cuda:0）。

    数据量上来后，空间输入累计降雨 cs 与静态场 statics 占用数 GB 内存；
    放到 GPU 上既省内存又能让 __getitem__ 只回传小切片。
    """
    if device.type == "cuda":
        ds.cs = torch.from_numpy(ds.cs).to(device)          # 常驻 GPU
        ds.statics = torch.from_numpy(ds.statics).to(device)
    # 空间输入已常驻 GPU，其余小切片就地创建，无需 pin_memory
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)


class Net(nn.Module):
    def __init__(self, n_spatial_ch, n_cum, use_spatial, use_site, hidden,
                 spatial_dim, horizon, att_heads=0, att_mode="cat",
                 att_excl=False, n_site=0, tok_dim=8, use_feat=False,
                 areas=None):
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
        # 物理锚定的静态特征：log(面积/全域中位面积)，每时刻随行输入，
        # 同一函数作用于所有站（全局变换，无站点拟合参数），可迁移到新站。
        self.use_feat = use_feat
        in_dim = 2
        if use_feat:
            med = float(np.median(areas))
            self.register_buffer(
                "lg_area", torch.log(torch.as_tensor(areas, dtype=torch.float32) / med))
            in_dim += 1
        self.site_emb = None
        self.tok_dim = 0
        if n_site:      # 特殊 token：每站一个可学习向量，拼在历史序列最前面，
            self.site_emb = nn.Embedding(n_site, tok_dim)   # 编码历史时即带站点身份
            self.tok_dim = tok_dim
            in_dim += tok_dim
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)  # 历史流量与历史面雨量
        self.lstm_f = nn.LSTM(1, 32, batch_first=True)       # 预见期内的面雨量
        self.att, self.att_mode, self.att_excl = None, att_mode, att_excl
        if att_heads:           # 历史支路自注意力：末态作查询，检索相似历史片段
            self.att = nn.MultiheadAttention(hidden, att_heads, batch_first=True)
        hist_dim = hidden * (2 if (self.att is not None and att_mode == "cat") else 1)
        fused = (hist_dim + (spatial_dim if use_spatial else 0)
                 + (3 if use_site else 0) + 32)
        self.head = nn.Sequential(nn.Linear(fused, 96), nn.ReLU(),
                                  nn.Linear(96, horizon))

    def forward(self, x, hist, frain, site):
        B, T, _ = hist.shape
        consts = []             # 随行输入的静态量（每个时刻相同）
        if self.use_feat:
            consts.append(self.lg_area[site].reshape(B, 1, 1).expand(-1, T, -1))
        tok_first = None
        if self.site_emb is not None:
            consts.append(torch.zeros(B, T, self.tok_dim, device=hist.device,
                                      dtype=hist.dtype))
            tok_first = self.site_emb(site).unsqueeze(1)    # token 只出现在第 0 行
        if consts:
            body = torch.cat([hist] + consts, dim=-1)
            firsts = [tok_first if (tok_first is not None and j == len(consts) - 1)
                      else c[:, :1] for j, c in enumerate(consts)]
            row0 = torch.cat([torch.zeros(B, 1, 2, device=hist.device,
                                          dtype=hist.dtype)] + firsts, dim=-1)
            out, (h, _) = self.lstm(torch.cat([row0, body], dim=1))
        else:
            out, (h, _) = self.lstm(hist)
        if self.att is not None:
            amask = None
            if self.att_excl:   # 末态自身不参与检索（排除当前时刻）
                amask = torch.zeros(1, out.shape[1], device=out.device)
                amask[0, -1] = float("-inf")
            ctx, _ = self.att(out[:, -1:, :], out, out, attn_mask=amask)
            ctx = ctx.squeeze(1)
            if self.att_mode == "res":
                feats = [h[-1] + ctx]      # 残差：注意力输出是叠加在末态上的修正
            else:
                feats = [torch.cat([h[-1], ctx], dim=-1)]
        else:
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
        if self.use_site and site is not None and site.dim() > 1:
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


def inv_yeojohnson(y, lam):
    """Yeo-Johnson 逆变换（scipy 只提供正向）。"""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    pos = y >= 0
    if abs(lam) < 1e-8:
        out[pos] = np.expm1(y[pos])
    else:
        out[pos] = np.power(y[pos] * lam + 1.0, 1.0 / lam) - 1.0
    if abs(lam - 2.0) < 1e-8:
        out[~pos] = 1.0 - np.exp(-y[~pos])
    else:
        out[~pos] = 1.0 - np.power(1.0 - y[~pos] * (2.0 - lam), 1.0 / (2.0 - lam))
    return np.maximum(out, 0.0)


def make_inv(transform, q_mean, q_std, lams):
    """按目标变换构造逆变换，把标准化空间换回 m³/s。

    全部委托成熟库（log1p 用 numpy 的 expm1，yeo-johnson 用 sklearn 的
    PowerTransformer.inverse_transform）。负 lambda 时 YJ 正支有上界
    -1/lambda，模型输出越界处没有实数逆，先截断再逆变换。
    """
    if transform == "yeo-johnson":
        from sklearn.preprocessing import PowerTransformer
        lam = float(np.asarray(lams).ravel()[0])        # 全局 lambda
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.lambdas_ = np.array([lam])
        pt.n_features_in_ = 1
        bound = (-1.0 / lam - 1e-6) if lam < 0 else None

        def inv(z, i):
            z = np.asarray(z, dtype=np.float64)
            y = z * q_std[i] + q_mean[i]
            if bound is not None:
                y = np.minimum(y, bound)
            x = pt.inverse_transform(y.reshape(-1, 1)).reshape(z.shape)
            return np.maximum(x, 0.0)
        return inv
    if transform in ("std", "gstd"):
        return lambda z, i: np.maximum(
            np.asarray(z, dtype=np.float64) * q_std[i] + q_mean[i], 0.0)
    return lambda z, i: np.expm1(np.asarray(z) * q_std[i] + q_mean[i])


def train_model(model, ldr, m, out_dir, device):
    opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(m["epochs"]))
    loss_name = str(m.get("loss", "mse"))
    if loss_name == "huber_nse":
        # E1：Huber(reduction='none') × 样本权重（逐站方差归一），训练批加权、验证不加权
        delta = float(m.get("huber_delta", 1.0))
        print(f"损失 huber_nse（delta={delta}，逐站方差归一加权，仅训练）")
    elif loss_name == "huber":
        lossf = nn.HuberLoss(delta=float(m.get("huber_delta", 1.0)))
        print(f"损失 {loss_name}")
    else:
        lossf = nn.MSELoss()
        print(f"损失 {loss_name}")

    def run(loader, train):
        model.train(train)
        tot, n = 0.0, 0
        for x, hist, frain, idx, y, w in loader:
            x, hist, frain, idx, y = (t.to(device, non_blocking=True)
                                      for t in (x, hist, frain, idx, y))
            w = w.to(device, non_blocking=True)
            pred = model(x, hist, frain, idx)
            if loss_name == "huber_nse":
                l = nn.functional.huber_loss(pred, y, delta=delta, reduction="none")
                # 空间平均到样本级，与权重逐样本相乘
                l = l.mean(dim=tuple(range(1, l.ndim)))
                loss = (l * w).sum() / w.sum() if train else l.mean()
            else:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="key=value")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，直接加载 out_dir/best.pt 评估")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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

    # 非线性变换只做全局一份（各断面共用同一个 log / YJ lambda，或干脆不做），
    # 避免逐站拟合带来的站点特异性和逆变换损失；之后统一逐站线性标准化，
    # 让共享权重的模型在各断面量程（0.5~4000 m³/s）下可比。该标准化无损可逆。
    transform = str(m.get("transform", "log1p"))
    flow_pos = np.maximum(flow, 0)
    if transform == "yeo-johnson":
        from sklearn.preprocessing import PowerTransformer
        tr = flow_pos[:, :n_tr].ravel()
        tr = tr[np.isfinite(tr)].reshape(-1, 1)
        pt_g = PowerTransformer(method="yeo-johnson", standardize=False)
        pt_g.fit(tr)                                # 全部断面训练段 pooled 拟合一次
        lam = float(pt_g.lambdas_[0])
        lams = np.full(n_site, lam)
        log_flow = np.full_like(flow_pos, np.nan, dtype=np.float64)
        fin = np.isfinite(flow_pos)
        log_flow[fin] = pt_g.transform(flow_pos[fin].reshape(-1, 1)).ravel()
        print(f"变换 yeo-johnson（全局 lambda={lam:.3f}）")
    elif transform == "gstd":
        lams = None
        log_flow = flow_pos.astype(np.float64)
        print("变换 gstd：全局标准化（所有断面 pooled 只估一个均值/标准差）")
    elif transform == "std":
        lams = None
        log_flow = flow_pos.astype(np.float64)
        print("变换 std：无非线性变换，直接输出逐站标准化流量")
    else:
        lams = None
        log_flow = np.log1p(flow_pos)
        print("变换 log1p")
    q_mean = np.nanmean(log_flow[:, :n_tr], axis=1)
    q_std = np.nanstd(log_flow[:, :n_tr], axis=1)
    q_std[q_std < 1e-6] = 1.0
    if transform == "gstd":
        # 全局标准化：所有断面 pooled 一起估唯一的均值/标准差，
        # 不带任何站点私有参数（均值≈大断面量程，小断面被压到 0 附近）。
        gm = float(np.nanmean(log_flow[:, :n_tr]))
        gs = float(np.nanstd(log_flow[:, :n_tr])) or 1.0
        q_mean = np.full(n_site, gm)
        q_std = np.full(n_site, gs)
        print(f"全局均值 {gm:.2f}  全局标准差 {gs:.2f} m³/s")
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

    # E1：逐站方差归一化损失权重 = 全站平均方差 / 本站方差（裁剪防极端）
    w_train = None
    if str(m.get("loss")) == "huber_nse":
        v = np.nanvar(log_flow[:, :n_tr], axis=1)
        w_site = np.clip(v.mean() / np.clip(v, 1e-6, None), 1e-3, 1e3).astype(np.float32)
        w_train = w_site[np.asarray(samples["train"], dtype=np.int64)[:, 0]]
        print("逐站损失权重 " +
              " ".join(f"{names[i]}:{w_site[i]:.2f}" for i in range(n_site)))

    ds = {k: SiteDataset(cs, statics, flow_n, rain_n, v, cum_hours,
                         use_spatial, use_future, horizon, lookback,
                         weights=w_train if k == "train" else None)
          for k, v in samples.items()}
    ldr = {k: make_loader(v, int(m["batch_size"]), shuffle=(k == "train"),
                          device=device) for k, v in ds.items()}

    att_heads = int(m.get("att_heads", 0))
    att_mode = str(m.get("att_mode", "cat"))
    att_excl = bool(m.get("att_excl", False))
    use_tok = bool(m.get("use_tok", False))
    tok_dim = int(m.get("tok_dim", 8))
    use_feat = bool(m.get("use_feat", False))
    model = Net(len(cum_hours) + 4, len(cum_hours), use_spatial, use_site,
                int(m["hidden"]), int(m["spatial_dim"]), horizon,
                att_heads=att_heads, att_mode=att_mode, att_excl=att_excl,
                n_site=n_site if use_tok else 0, tok_dim=tok_dim,
                use_feat=use_feat, areas=areas if use_feat else None).to(device)
    print(f"设备 {device}  参数量 {sum(p.numel() for p in model.parameters()) / 1e3:.1f} 千  "
          f"空间分支 {'开' if use_spatial else '关'}  "
          f"预见期降雨 {'有' if use_future else '无'}  "
          f"自注意力 {att_heads or '无'}{'-' + att_mode if att_heads else ''}"
          f"{'-排己' if att_heads and att_excl else ''}  "
          f"站点token {f'{tok_dim}维' if use_tok else '无'}  "
          f"面积特征 {'开' if use_feat else '关'}")

    out_dir = os.path.join(ROOT, cfg["paths"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                         map_location=device))
        print(f"已加载 {out_dir}/best.pt，跳过训练")
    else:
        train_model(model, ldr, m, out_dir, device)
    # ---------- 评估 ----------
    model.eval()
    obs_l, sim_l, site_l = [], [], []
    with torch.no_grad():
        for x, hist, frain, idx, y, _w in ldr["test"]:
            x, hist, frain, idx, y = (t.to(device)
                                      for t in (x, hist, frain, idx, y))
            delta = model(x, hist, frain, idx)
            anchor = hist[:, -1, 0:1]
            obs_l.append((y + anchor).cpu().numpy())
            sim_l.append((delta + anchor).cpu().numpy())
            site_l.append(idx.cpu().numpy())
    obs_n, sim_n = np.concatenate(obs_l), np.concatenate(sim_l)
    st = np.concatenate(site_l)
    inv = make_inv(transform, q_mean, q_std,
                   lams if lams is not None else np.zeros(1))

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
        transform=np.array(transform),
        lams=np.array(lams if lams is not None else np.full(n_site, np.nan)),
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {cfg['paths']['out_dir']}")


if __name__ == "__main__":
    main()
