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
    """每个断面的汇水区掩膜：输入前用它裁剪降雨图（掩膜外归零）。"""
    return mask1km[:, None].astype(np.float32)


class SiteDataset(Dataset):
    """按需组装空间输入，避免把全部样本预先展开成巨大数组。"""

    def __init__(self, cs, statics, flow_n, rain_n, samples,
                 use_spatial, use_future, horizon, lookback, weights=None,
                 areal=False, mask_ch=False):
        self.cs, self.statics = cs, statics
        self.flow_n, self.rain_n = flow_n, rain_n
        self.samples = np.asarray(samples, dtype=np.int64)  # (n, 2) 便于整批索引
        self.use_spatial, self.use_future = use_spatial, use_future
        # DLinear 模式：不喂降雨图，改喂 3 个面雨量标量（见 __getitem__）
        self.areal = areal
        # 掩膜池化模式：x 增加第 3 通道 = 汇水区掩膜，供 pool2d 加权归一
        self.mask_ch = mask_ch
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
            # 空间支路：汇水区掩膜裁剪后的两张图——最近 1h 降雨、72h 累计降雨；
            # 预见期支路：同一张降雨图在预见期内的部分（裁剪）。
            # cs 为降雨图的逐时累积（cs[t]-cs[t-1] 即 t 时刻降雨图），GPU 常驻。
            m = self.statics[i, 0]                          # 汇水区掩膜
            if torch.is_tensor(self.cs):
                rnow = self.cs[t] - (self.cs[t - 1] if t > 0 else 0.0)
                lo = max(0, t - 71)
                rain72 = self.cs[t] - (self.cs[lo - 1] if lo > 0 else 0.0)
                x = torch.stack([rnow * m, rain72 * m]
                                + ([m] if self.mask_ch else []))
                fut = ((self.cs[t + self.horizon] - self.cs[t]) * m
                       if self.use_future else torch.zeros_like(rnow))
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                fut = torch.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)
                return (x,
                        torch.tensor(hist[:, None], dtype=torch.float32),
                        fut.unsqueeze(0),
                        i, torch.tensor(delta, dtype=torch.float32),
                        self.weights[k])
            rnow = self.cs[t] - (self.cs[t - 1] if t > 0 else 0.0)
            lo = max(0, t - 71)
            rain72 = self.cs[t] - (self.cs[lo - 1] if lo > 0 else 0.0)
            x = np.stack([rnow * m, rain72 * m]
                         + ([m] if self.mask_ch else []))
            fut = ((self.cs[t + self.horizon] - self.cs[t]) * m
                   if self.use_future else np.zeros_like(rnow))
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            fut = np.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = np.zeros((1, 1, 1), dtype=np.float32)
            if self.areal:
                # DLinear 的线性化降雨输入：当前 1h、72h 累计、下一小时面雨量
                # （均已全局标准化；下一小时为完美预报近似，与 S 系一致）
                lo = max(0, t - 71)
                r_nxt = self.rain_n[i, t + 1] if self.use_future else 0.0
                fut = np.array([self.rain_n[i, t],
                                self.rain_n[i, lo:t + 1].sum(), r_nxt],
                               dtype=np.float32)                       # (3,)
            else:
                fut = np.zeros((1, 1), dtype=np.float32)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(hist[:, None], dtype=torch.float32),
                torch.tensor(fut, dtype=torch.float32),
                i, torch.tensor(delta, dtype=torch.float32),
                self.weights[k])


def make_loader(ds, batch_size, shuffle, device, num_workers=0):
    """组装 DataLoader，并按需把常驻张量挪到 GPU（cuda:0）。

    数据量上来后，空间输入累计降雨 cs 与静态场 statics 占用数 GB 内存；
    放到 GPU 上既省内存又能让 __getitem__ 只回传小切片。
    """
    if device.type == "cuda":
        if ds.cs is not None:
            ds.cs = torch.from_numpy(ds.cs).to(device)      # 常驻 GPU
        ds.statics = torch.from_numpy(ds.statics).to(device)
    # 空间输入已常驻 GPU，其余小切片就地创建，无需 pin_memory
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)


def pool2d(f, m=None):
    """空间池化。 m=None：全局平均池化（旧行为）；m 为掩膜通道时：掩膜加权
    平均 Σ(f·m)/Σ(m)——每站除以自己的格数，小汇水区不再被 128×128 网格稀释。"""
    if m is None:
        return torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
    mr = torch.nn.functional.adaptive_avg_pool2d(m, f.shape[-2:])
    w = mr.sum(dim=(2, 3)).clamp(min=1e-6)                  # (B,1) 本站格数占比
    return (f * mr).sum(dim=(2, 3)) / w


def area_frac(m):
    """掩膜面积占比 → log 标量，给共享权重的下游提供本站尺度标定。"""
    return torch.log(m.mean(dim=(1, 2, 3)).clamp(min=1e-6)).unsqueeze(1)


class Net(nn.Module):
    def __init__(self, use_spatial, use_site, hidden,
                 spatial_dim, horizon, att_heads=0, att_mode="cat",
                 att_excl=False, n_site=0, tok_dim=8, use_feat=False,
                 areas=None, n_quant=1, masked_pool=False, delta_cap=0.0):
        super().__init__()
        self.use_spatial, self.use_site = use_spatial, use_site
        self.horizon = horizon
        self.masked_pool = masked_pool
        # 单步增量硬上限（标准化空间，tanh 饱和）：训练段最大小时跳变约 5.9σ，
        # 取 6 作顶——分布内不受限（tanh 在原点附近恒等），分布外反馈放大被刹车。
        # 0 = 关闭。对称 S.5/S.6/S.7 线性主干的天然有界性（Helene 不穿顶）。
        self.delta_cap = float(delta_cap)
        # 多分位数输出：forward 返回 (B, horizon, n_quant)，取中位分位点作点预报
        self.n_quant = n_quant
        self.mid = n_quant // 2
        if use_spatial:
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.proj = nn.Linear(32, spatial_dim)
            # 预见期降雨图（单通道）独立小卷积，各自投影后拼接
            self.fut_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.fproj = nn.Linear(32, spatial_dim)
        # 物理锚定的静态特征：log(面积/全域中位面积)，每时刻随行输入，
        # 同一函数作用于所有站（全局变换，无站点拟合参数），可迁移到新站。
        self.use_feat = use_feat
        in_dim = 1                                    # 仅流量
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
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)  # 历史流量
        self.att, self.att_mode, self.att_excl = None, att_mode, att_excl
        if att_heads:           # 历史支路自注意力：末态作查询，检索相似历史片段
            self.att = nn.MultiheadAttention(hidden, att_heads, batch_first=True)
        hist_dim = hidden * (2 if (self.att is not None and att_mode == "cat") else 1)
        fused = (hist_dim + (2 * spatial_dim if use_spatial else 0)
                 + (1 if use_spatial and masked_pool else 0)
                 + (3 if use_site else 0))
        self.head = nn.Sequential(nn.Linear(fused, 96), nn.ReLU(),
                                  nn.Linear(96, horizon * n_quant))

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
            row0 = torch.cat([torch.zeros(B, 1, hist.shape[-1], device=hist.device,
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
        if self.use_spatial:
            # 历史态两张图（当前 1h + 72h 平均）与预见期降雨图（下一小时）各自卷积
            m = x[:, 2:3] if self.masked_pool else None
            f = self.cnn(x[:, :2] if self.masked_pool else x)
            feats.append(self.proj(pool2d(f, m)))
            if self.masked_pool:
                feats.append(area_frac(m))
            if frain is not None:
                ff = self.fut_cnn(frain)
                feats.append(self.fproj(pool2d(ff, None)))
        if self.use_site and site is not None and site.dim() > 1:
            feats.append(site)
        out = self.head(torch.cat(feats, dim=-1))
        out = out.view(B, self.horizon, self.n_quant)
        if self.delta_cap > 0:
            out = self.delta_cap * torch.tanh(out / self.delta_cap)
        return out


class DLinearNet(nn.Module):
    """DLinear（Zeng et al., AAAI'23）共享权重移植：移动平均分解趋势/季节，
    两支各一个线性层相加。arch="dlinear" 时仅收流量历史 + 3 个面雨量标量
    （当前 1h、72h 均值、下一小时，均已标准化）；arch="dlinear_cnn" 时
    改收 S.4 的空间分支（2 通道降雨图 CNN + 下一小时降雨图 CNN，池化后
    拼进季节支）——把 S.4 的价值拆成"空间编码"与"时序主干"两部分对照。
    """

    def __init__(self, lookback, n_quant=3, kernel=25, n_rain=3,
                 use_spatial=False, spatial_dim=48, masked_pool=False,
                 delta_cap=0.0):
        super().__init__()
        self.lookback, self.kernel = lookback, kernel
        self.n_quant = n_quant
        self.mid = n_quant // 2
        self.use_spatial = use_spatial
        self.masked_pool = masked_pool
        self.delta_cap = float(delta_cap)
        self.lin_trend = nn.Linear(lookback, n_quant)
        season_in = lookback + (2 * spatial_dim if use_spatial else n_rain)
        season_in += 1 if use_spatial and masked_pool else 0   # log 面积占比
        self.lin_season = nn.Linear(season_in, n_quant)
        if use_spatial:
            # 与 Net 同构的空间分支（权重独立重训）
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.proj = nn.Linear(32, spatial_dim)
            self.fut_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.fproj = nn.Linear(32, spatial_dim)

    def forward(self, x, hist, frain, site):
        q = hist[:, :, 0]                                   # (B, LB) 标准化流量
        pad = self.kernel // 2
        qp = torch.cat([q[:, :1].expand(-1, pad), q,
                        q[:, -1:].expand(-1, pad)], dim=1)  # 端点延拓
        trend = nn.functional.avg_pool1d(qp.unsqueeze(1), self.kernel,
                                         stride=1).squeeze(1)
        season = q - trend
        if self.use_spatial:
            m = x[:, 2:3] if self.masked_pool else None
            f = self.cnn(x[:, :2] if self.masked_pool else x)
            extras = [self.proj(pool2d(f, m))]
            if self.masked_pool:
                extras.append(area_frac(m))
            if frain is not None:
                extras.append(self.fproj(pool2d(self.fut_cnn(frain), None)))
            else:
                extras.append(torch.zeros_like(extras[0]))
            season = torch.cat([season] + extras, -1)
        else:
            if frain is None:
                frain = torch.zeros(q.shape[0], 3, device=q.device)
            season = torch.cat([season, frain.reshape(q.shape[0], -1)], -1)
        out = self.lin_trend(trend) + self.lin_season(season)
        out = out.unsqueeze(1)                             # (B, 1, K)
        if self.delta_cap > 0:
            out = self.delta_cap * torch.tanh(out / self.delta_cap)
        return out


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


def unrolled_loss(model, idx, t0, flow_g, cs_g, st_g, w, K, lookback,
                  use_spatial, use_future, ar, delta):
    """把模型自己的预测喂回历史窗，滚动 K 步，返回每步损失均值（E1 加权）。

    与推理滚动完全同构：除起报窗外不用实测流量；损失打在滚动每一步上，
    涨水过程的每一步都有梯度，教师强迫的曝光偏差由此根治。
    """
    cur = flow_g[idx[:, None], t0[:, None] + ar[None, :]].clone()   # (B, LB)
    tot = 0.0
    for k in range(K):
        t = t0 + lookback - 1 + k                                   # 当前末刻 (B,)
        if use_spatial:
            mk = st_g[idx][:, 0]                                    # (B,G,G) 汇水区掩膜
            rnow = cs_g[t] - cs_g[t - 1]
            lo = (t - 71).clamp_min(0)
            base = cs_g[(lo - 1).clamp_min(0)] * (lo > 0).float()[:, None, None]
            rain72 = cs_g[t] - base                                  # 72h 累计
            x = torch.stack([rnow * mk, rain72 * mk], 1)             # (B,2,G,G)
            if model.masked_pool:
                x = torch.cat([x, mk.unsqueeze(1)], 1)               # 掩膜作第3通道
            fut = ((cs_g[t + 1] - cs_g[t]) * mk).unsqueeze(1)       # 下一小时降雨图
            if not use_future:
                fut = torch.zeros_like(fut)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            fut = torch.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = torch.zeros((len(idx), 1, 1, 1), device=flow_g.device)
            fut = None
        hist = cur.unsqueeze(-1)                                    # (B,LB,1) 仅流量
        d_pred = model(x, hist, fut, idx)[:, 0, model.mid]          # 中位分位点
        y = flow_g[idx, t + 1] - flow_g[idx, t]                     # 真值增量（锚定实测末刻）
        l = nn.functional.huber_loss(d_pred, y, delta=delta, reduction="none")
        tot = tot + (l * w).sum() / w.sum()
        nxt = cur[:, -1] + d_pred
        cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], 1)
    return tot / K


def train_model_unrolled(model, ds_tr, ds_va, m, out_dir, device):
    """展开式训练：训练/验证都在滚动条件下算损失，选模型与推理任务一致。"""
    K = int(m["unroll"])
    lookback = ds_tr.lookback
    use_spatial, use_future = ds_tr.use_spatial, ds_tr.use_future
    delta = float(m.get("huber_delta", 1.0))
    bs, epochs = int(m["batch_size"]), int(m["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ar = torch.arange(lookback, device=device)
    flow_tr = torch.from_numpy(np.nan_to_num(ds_tr.flow_n, nan=0.0)).float().to(device)
    flow_va = torch.from_numpy(np.nan_to_num(ds_va.flow_n, nan=0.0)).float().to(device)
    w_all = torch.as_tensor(ds_tr.weights, device=device)
    sam_tr = torch.as_tensor(ds_tr.samples, device=device)          # (n,2) 站号, t0
    sam_va = torch.as_tensor(ds_va.samples, device=device)
    cs_tr, st_tr = ds_tr.cs, ds_tr.statics                          # make_loader 已挪 GPU
    cs_va, st_va = ds_va.cs, ds_va.statics
    print(f"展开式训练 unroll={K}：滚动 {K} 步逐步回灌，损失打在每步上"
          f"（训练加权、验证不加权）")

    def run_train():
        model.train(True)
        perm = torch.randperm(len(sam_tr), device=device)
        tot, n = 0.0, 0
        for j in range(0, len(sam_tr), bs):
            sel = perm[j:j + bs]
            s = sam_tr[sel]
            loss = unrolled_loss(model, s[:, 0], s[:, 1], flow_tr, cs_tr, st_tr,
                                 w_all[sel], K, lookback, use_spatial,
                                 use_future, ar, delta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(sel)
            n += len(sel)
        return tot / max(n, 1)

    def run_val():
        model.train(False)
        tot, n = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(sam_va), bs):
                s = sam_va[j:j + bs]
                w = torch.ones(len(s), device=device)
                loss = unrolled_loss(model, s[:, 0], s[:, 1], flow_va, cs_va, st_va,
                                     w, K, lookback, use_spatial, use_future,
                                     ar, delta)
                tot += loss.item() * len(s)
                n += len(s)
        return tot / max(n, 1)

    best, best_state, wait = np.inf, None, 0
    for ep in range(1, epochs + 1):
        tr = run_train()
        sched.step()
        va = run_val()
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


def train_model(model, ldr, m, out_dir, device):
    opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(m["epochs"]))
    loss_name = str(m.get("loss", "mse"))
    if loss_name == "quantile":
        taus = torch.tensor([float(v) for v in m.get("quantiles", [0.1, 0.5, 0.9])],
                            device=device)
        qs = "/".join(f"{float(v):g}" for v in m.get("quantiles", [0.1, 0.5, 0.9]))
        print(f"损失 quantile（τ={qs}，pinball，逐站方差归一加权，仅训练）")
    elif loss_name == "huber_nse":
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
            pred = model(x, hist, frain, idx)          # (B, H, K)
            if loss_name == "quantile":
                err = pred - y.unsqueeze(-1)           # (B, H, K)
                l = torch.maximum(taus * err, (taus - 1) * err)
                l = l.mean(dim=tuple(range(1, l.ndim)))
                loss = (l * w).sum() / w.sum() if train else l.mean()
            else:
                pred = pred.squeeze(-1)                # (B, H)，单分位点
            if loss_name == "huber_nse":
                l = nn.functional.huber_loss(pred, y, delta=delta, reduction="none")
                # 空间平均到样本级，与权重逐样本相乘
                l = l.mean(dim=tuple(range(1, l.ndim)))
                loss = (l * w).sum() / w.sum() if train else l.mean()
            elif loss_name != "quantile":
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
    # rollout>0：单步模型（horizon=1）推理时递归滚动，把预报喂回历史窗口，
    # 逐步推出 rollout 小时；R 为评估/抽样用的实际预见期。
    rollout = int(m.get("rollout", 0))
    R = rollout or horizon
    stride, split = int(m["stride"]), list(m["split"])
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

    n_win = T - lookback - R + 1
    sd = m.get("split_dates")
    if sd:
        # 按日期切分（围绕洪水事件设计）：train < sd[0] ≤ val < sd[1] ≤ test
        b1 = int(times.searchsorted(pd.Timestamp(sd[0])))
        b2 = int(times.searchsorted(pd.Timestamp(sd[1])))
        b1 = min(max(b1, 1), n_win - 1)
        b2 = min(max(b2, b1 + 1), n_win)
        bounds = {"train": (0, b1), "val": (b1, b2), "test": (b2, n_win)}
        print(f"按日期划分  train 至 {times[b1]}  |  val 至 {times[b2]}  |  "
              f"test 至 {times[-1]}")
    else:
        n_tr = int(n_win * split[0])
        n_va = int(n_win * split[1])
        bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va),
                  "test": (n_tr + n_va, n_win)}
    n_tr = bounds["train"][1]      # 变换统计只用训练段

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
        step = stride if sp == "train" else R
        ss = []
        for i in range(n_site):
            for t0 in range(lo, hi, step):
                h = flow_n[i, t0:t0 + lookback]
                y = flow_n[i, t0 + lookback:t0 + lookback + R]
                r = rain_n[i, t0:t0 + lookback + R]
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

    masked_pool = bool(m.get("masked_pool", False))
    delta_cap = float(m.get("delta_cap", 0.0))
    ds = {k: SiteDataset(cs, statics, flow_n, rain_n, v,
                         use_spatial, use_future, horizon, lookback,
                         weights=w_train if k == "train" else None,
                         areal=(str(m.get("arch", "net")) == "dlinear"),
                         mask_ch=masked_pool)
          for k, v in samples.items()}
    ldr = {k: make_loader(v, int(m["batch_size"]), shuffle=(k == "train"),
                          device=device) for k, v in ds.items()}

    att_heads = int(m.get("att_heads", 0))
    att_mode = str(m.get("att_mode", "cat"))
    att_excl = bool(m.get("att_excl", False))
    use_tok = bool(m.get("use_tok", False))
    tok_dim = int(m.get("tok_dim", 8))
    use_feat = bool(m.get("use_feat", False))
    n_quant = (len(m.get("quantiles", [0.1, 0.5, 0.9]))
               if str(m.get("loss")) == "quantile" else 1)
    model_arch = str(m.get("arch", "net"))
    if model_arch == "dlinear":
        model = DLinearNet(lookback, n_quant=n_quant,
                           masked_pool=masked_pool, delta_cap=delta_cap).to(device)
    elif model_arch == "dlinear_cnn":
        model = DLinearNet(lookback, n_quant=n_quant, use_spatial=True,
                           spatial_dim=int(m["spatial_dim"]),
                           masked_pool=masked_pool, delta_cap=delta_cap).to(device)
    else:
        model = Net(use_spatial, use_site,
                    int(m["hidden"]), int(m["spatial_dim"]), horizon,
                    att_heads=att_heads, att_mode=att_mode, att_excl=att_excl,
                    n_site=n_site if use_tok else 0, tok_dim=tok_dim,
                    use_feat=use_feat, areas=areas if use_feat else None,
                    n_quant=n_quant, masked_pool=masked_pool,
                    delta_cap=delta_cap).to(device)
    print(f"设备 {device}  参数量 {sum(p.numel() for p in model.parameters()) / 1e3:.1f} 千  "
          f"结构 {model_arch}  "
          f"空间分支 {'开' if use_spatial else '关'}  "
          f"预见期降雨 {'有' if use_future else '无'}  "
          f"自注意力 {att_heads or '无'}{'-' + att_mode if att_heads else ''}"
          f"{'-排己' if att_heads and att_excl else ''}  "
          f"站点token {f'{tok_dim}维' if use_tok else '无'}  "
          f"面积特征 {'开' if use_feat else '关'}"
          + (f"  掩膜池化 开" if masked_pool else "")
          + (f"  增量限幅{delta_cap:g}σ" if delta_cap > 0 else "")
          + (f"  分位数×{n_quant}" if n_quant > 1 else "")
          + (f"  单步+滚动{rollout}h" if rollout else ""))

    out_dir = os.path.join(ROOT, cfg["paths"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                         map_location=device))
        print(f"已加载 {out_dir}/best.pt，跳过训练")
    elif int(m.get("unroll", 0)):
        train_model_unrolled(model, ds["train"], ds["val"], m, out_dir, device)
    else:
        train_model(model, ldr, m, out_dir, device)
    # ---------- 评估 ----------
    model.eval()

    if bool(m.get("train_fit", False)) and not args.eval_only:
        # 背数检查：训练段单步（教师强迫）拟合，物理量 m³/s 逐站 NSE
        inv_fit = make_inv(transform, q_mean, q_std,
                           lams if lams is not None else np.zeros(1))
        fo, fs, fi = [], [], []
        with torch.no_grad():
            for x, hist, frain, idx, y, _w in ldr["train"]:
                x, hist, frain, idx, y = (t.to(device)
                                          for t in (x, hist, frain, idx, y))
                dl = model(x, hist, frain, idx)[:, :, model.mid]
                anchor = hist[:, -1, 0:1]
                fo.append((y + anchor).cpu().numpy())
                fs.append((dl + anchor).cpu().numpy())
                fi.append(idx.cpu().numpy())
        fo, fs, fi = map(np.concatenate, (fo, fs, fi))
        print("\n训练段拟合（单步，教师强迫）")
        ns_fit = []
        for i in range(n_site):
            mk = fi == i
            if mk.sum() == 0:
                continue
            o = inv_fit(fo[mk], i).ravel()
            s = inv_fit(fs[mk], i).ravel()
            ns_fit.append(nse(o, s))
            print(f"{ids[i]:11s} {names[i]:10s} NSE {nse(o, s):6.3f}  "
                  f"MAE {np.mean(np.abs(o - s)):8.3f}")
        print(f"训练段中位 NSE {np.median(ns_fit):.3f}")

    obs_l, sim_l, site_l, t0_l = [], [], [], []
    if rollout:
        # 递归滚动评估：单步模型逐时把预报流量接回历史窗口（末刻锚定与训练一致），
        # 降雨图输入随滚动时刻前移，推出 R 小时预报；除起报窗外不用实测流量。
        from collections import defaultdict
        by_t0 = defaultdict(list)
        for i, t0 in samples["test"]:
            by_t0[int(t0)].append(int(i))
        flow_t = torch.from_numpy(np.nan_to_num(flow_n, nan=0.0)).float().to(device)
        rain_t = torch.from_numpy(np.nan_to_num(rain_n, nan=0.0)).float().to(device)
        cs_t, st_t = ds["test"].cs, ds["test"].statics   # make_loader 已挪到 GPU
        with torch.no_grad():
            for t0 in sorted(by_t0):
                idx = torch.tensor(by_t0[t0], device=device)
                B = len(idx)
                cur = flow_t[idx, t0:t0 + lookback].clone()   # (B, LB) 滚动历史
                sims = []
                for k in range(rollout):
                    t = t0 + lookback - 1 + k                 # 当前末刻
                    if use_spatial:
                        mk_ = st_t[idx][:, 0]                 # (B, G, G) 汇水区掩膜
                        rnow = cs_t[t] - (cs_t[t - 1] if t > 0 else 0.0)
                        lo2 = max(0, t - 71)
                        base = cs_t[lo2 - 1] if lo2 > 0 else 0.0
                        rain72 = cs_t[t] - base                      # 72h 累计
                        x = torch.stack([rnow * mk_, rain72 * mk_], 1)      # (B,2,G,G)
                        if masked_pool:
                            x = torch.cat([x, mk_.unsqueeze(1)], 1)   # 掩膜作第3通道
                        fut = ((cs_t[t + 1] - cs_t[t]) * mk_).unsqueeze(1)  # 下一小时降雨图
                        if not use_future:
                            fut = torch.zeros_like(fut)
                        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                        fut = torch.nan_to_num(fut, nan=0.0, posinf=0.0,
                                               neginf=0.0)
                    else:
                        x = torch.zeros((B, 1, 1, 1), device=device)
                        fut = None
                        if model_arch == "dlinear":
                            # 滚动时刻 t 的 3 个面雨量标量，与训练集 __getitem__ 同式
                            lo2 = max(0, t - 71)
                            fut = torch.stack([rain_t[idx, t],
                                               rain_t[idx, lo2:t + 1].mean(dim=1),
                                               rain_t[idx, t + 1]], 1)
                            if not use_future:
                                fut = torch.cat(
                                    [fut[:, :2],
                                     torch.zeros(B, 1, device=device)], 1)
                    hist = cur.unsqueeze(-1)              # (B, LB, 1) 仅流量
                    nxt = cur[:, -1] + model(x, hist, fut, idx)[:, 0, model.mid]
                    cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], 1)
                    sims.append(nxt)
                sim_l.append(torch.stack(sims, 1).cpu().numpy())
                ii = idx.cpu().numpy()
                obs_l.append(flow_n[ii, t0 + lookback:t0 + lookback + R])
                site_l.append(ii)
                t0_l.append(np.full(B, t0, dtype=np.int64))
    else:
        with torch.no_grad():
            for x, hist, frain, idx, y, _w in ldr["test"]:
                x, hist, frain, idx, y = (t.to(device)
                                          for t in (x, hist, frain, idx, y))
                delta = model(x, hist, frain, idx)[:, :, model.mid]   # 中位分位点
                anchor = hist[:, -1, 0:1]
                obs_l.append((y + anchor).cpu().numpy())
                sim_l.append((delta + anchor).cpu().numpy())
                site_l.append(idx.cpu().numpy())
        t0_l.append(np.array([t0 for _, t0 in samples["test"]], dtype=np.int64))
    obs_n, sim_n = np.concatenate(obs_l), np.concatenate(sim_l)
    st = np.concatenate(site_l)
    t0_arr = np.concatenate(t0_l)
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
        per = inv(np.repeat(obs_n[mk][:, :1], R, axis=1), i).ravel()
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

    # 分预见期精度：t+1/3/6/12/24 各列单独算逐站 NSE，再取中位。
    # 滚动预报只有第一步用实测末刻，之后各步输入均为模型自身预报（见上）。
    leads = [L for L in (1, 3, 6, 12, 24) if L <= R]
    lead_rows = []
    for i in range(n_site):
        mk = st == i
        if mk.sum() == 0:
            continue
        oo, ss = inv(obs_n[mk], i), inv(sim_n[mk], i)
        lead_rows.append([nse(oo[:, L - 1], ss[:, L - 1]) for L in leads])
    lr_arr = np.asarray(lead_rows)
    lead_med = np.median(lr_arr, axis=0)
    print("\n预见期  " + "  ".join(f"{L:+3d}h" for L in leads))
    print("中位NSE " + "  ".join(f"{v:5.3f}" for v in lead_med))
    summary["nse_by_lead"] = {f"{L}h": float(v) for L, v in zip(leads, lead_med)}

    np.savez_compressed(
        os.path.join(out_dir, "predictions.npz"),
        obs=obs_n, sim=sim_n, site=st, t0=t0_arr,
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
