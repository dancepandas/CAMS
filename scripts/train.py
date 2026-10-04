#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""断面流量预报：训练与评估。

一个共享权重的模型同时服务流域内多个断面。主模型的历史支路接收流量序列，
降雨由格点空间支路和预见期支路提供；DLinear 对照模型也可改用面雨量标量。
目标语义由 ``model.output_mode`` 决定：``delta`` 预测相对窗口末刻实测的增量，
``level`` 直接预测下一时刻的标准化流量数值。S.14/S.18 修正版使用后者。

训练入口先执行数据门禁：站序、时间、网格、坐标系以及面雨量的源降雨摘要
必须全部一致；训练目标必须早于 2023，2023 只作验证，2024 只作最终评估。

用法: python3 scripts/train.py [--config configs/pipeline.yaml]
                                 [--set model.epochs=8]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from data_contract import (MANIFEST_FORMAT, assert_calendar_splits,
                           validate_data_contract, write_training_manifest)


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


def load_inputs(cfg, contract=None):
    """读站点表、面雨量、各断面流量，按共同时间轴对齐。"""
    if contract is None:
        contract = validate_data_contract(cfg, ROOT)
    ids = list(contract.ids)
    names = list(contract.names)
    areas = contract.areas.copy()
    times = contract.times
    area_rain = contract.area_rain
    flow = contract.flow
    return ids, names, areas, times, area_rain, flow


def build_statics(cfg, mask1km):
    """每个断面的汇水区掩膜：输入前用它裁剪降雨图（掩膜外归零）。"""
    return mask1km[:, None].astype(np.float32)


class SiteDataset(Dataset):
    """按需组装空间输入，避免把全部样本预先展开成巨大数组。"""

    def __init__(self, cs, statics, flow_n, rain_n, samples,
                 use_spatial, use_future, horizon, lookback, weights=None,
                 areal=False, mask_ch=False, output_mode="delta"):
        self.cs, self.statics = cs, statics
        self.flow_n, self.rain_n = flow_n, rain_n
        # delta（默认）：目标是相对窗末实测的增量，持续性预报 = 输出全零。
        # level（S.14）：目标直接是下一时刻的流量数值本身，没有"输出零即持续"
        # 这个便利，模型必须自己学会贴着窗末实测值走。
        self.output_mode = str(output_mode)
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
        target = tgt if self.output_mode == "level" else tgt - hist[-1]

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
                        i, torch.tensor(target, dtype=torch.float32),
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
                i, torch.tensor(target, dtype=torch.float32),
                self.weights[k])


def make_loader(ds, batch_size, shuffle, device, num_workers=0):
    """组装 DataLoader，并按需把常驻张量挪到 GPU（cuda:0）。

    数据量上来后，空间输入累计降雨 cs 与静态场 statics 占用数 GB 内存；
    放到 GPU 上既省内存又能让 __getitem__ 只回传小切片。
    """
    if device.type == "cuda":
        # cs 可能已被上一处统一搬到显存了（三个数据集共用同一个张量），
        # 这里只在它还是 numpy 数组时才搬，避免同一块大数组被搬三份。
        if ds.cs is not None and not torch.is_tensor(ds.cs):
            ds.cs = torch.from_numpy(ds.cs).to(device)      # 常驻 GPU
        if not torch.is_tensor(ds.statics):
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
            # 历史态两张图（当前 1h + 过去最多 72h 累计）与预见期降雨图
            # （下一小时）各自卷积；这里不是 72h 平均值。
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
    （当前 1h、过去最多 72h 累计、下一小时，均已标准化）；arch="dlinear_cnn" 时
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


class MoENet(nn.Module):
    """双专家软门控（MoE）：两个完全独立的 Net 各出一套分位数增量预测，
    门控按 [当前流量z, 近24h均值z, log面积占比] 给两项输出加权融合。
    动机：大小站量程差 273 倍，训练时梯度互相干扰（C.9 损失加权的结构版对偶），
    让模型自由分工，不加任何强迫——跑完看门控实际怎么分（main 里有分工报告）。
    """

    def __init__(self, use_spatial, use_site, hidden, spatial_dim, horizon,
                 n_quant=3, masked_pool=False, delta_cap=0.0, areas=None):
        super().__init__()
        self.horizon, self.n_quant = horizon, n_quant
        self.mid = n_quant // 2
        self.experts = nn.ModuleList([
            Net(use_spatial, use_site, hidden, spatial_dim, horizon,
                n_quant=n_quant, masked_pool=masked_pool, delta_cap=delta_cap),
            Net(use_spatial, use_site, hidden, spatial_dim, horizon,
                n_quant=n_quant, masked_pool=masked_pool, delta_cap=delta_cap)])
        med = float(np.median(areas))
        self.register_buffer(
            "lg_area", torch.log(torch.as_tensor(areas, dtype=torch.float32) / med))
        # 门控只看 3 个可解释标量：现在多大水、近期多大水、站多大
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2))

    def gate_w(self, hist, site):
        """两项门控权重 (B, 2)，softmax 和为 1。单独成方法便于评估时复查分工。"""
        z = hist[:, -1, 0]
        z24 = hist[:, -24:, 0].mean(dim=1)
        g = torch.softmax(self.gate(torch.stack(
            [z, z24, self.lg_area[site]], 1)), dim=1)
        return g

    def forward(self, x, hist, frain, site):
        g = self.gate_w(hist, site)
        o1 = self.experts[0](x, hist, frain, site)
        o2 = self.experts[1](x, hist, frain, site)
        return g[:, 0, None, None] * o1 + g[:, 1, None, None] * o2


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


def unrolled_steps(model, idx, t0, flow_g, fin_g, cs_g, st_g, w, K, lookback,
                   use_spatial, use_future, ar, output_mode, taus=None,
                   huber_delta=1.0, bias_pen=0.0):
    """闭环滚动训练：逐步生成损失，每步把模型自己的预测喂回历史窗。

    与推理滚动完全同构：除起报窗外不用实测流量。和旧版 full-BPTT 展开的区别：

    - 支持 output_mode="level"：模型输出直接是流量数值本身，真值也取流量，
      回灌预测值时**不再加末刻锚**（加锚是 delta 语义，level 下是双重计数）。
    - 支持分位数损失（taus 非空时对全部量化头算 pinball，逐头取均值）。
    - 预测回灌默认断梯度（scheduled sampling 式截断反传）：模型照样"看见"
      自己的预测当输入、学到对偏差输入纠偏，但只需常驻单步计算图——
      全量 BPTT 要同时保住 K 步图，K=24 时显存直接翻倍起步。
    - 目标缺测防护：flow_g 里缺测已被 nan_to_num 填 0，必须乘回原始有限掩膜，
      否则缺测时刻的真值 0 会污染损失（教师强迫路径靠 Dataset 的掩膜，
      这里自己算）。
    - bias_pen（S.20 第一级，治滚动漂移的对称惩罚）：>0 时逐步额外罚
      「该提前量的有符号平均偏差绝对值」。为什么加它：纯闭环监督的隐性
      副作用是模型靠整体压低预测当漂移刹车（S.18 闭环实验洪峰 +10.3%→
      -16.1% 的代价就是这么来的）——单步 loss 只管"这一步报准"，看不见
      "偏差的符号会滚到深处"。这里把每个提前量的偏差绝对值直接摆进损失，
      且按提前量线性加权（越深罚越重，因为漂移随深度复利放大），惩罚是对
      称的：只罚"偏了"，不指定往哪偏，模型只能靠逐步报准来满足它。
      注意逐站分别算有符号偏差再取绝对值平均——批内大小站混算会让正负
      互相抵消，小站的漂移信号会被大站淹没。

    返回生成器，逐步 yield 加权平均损失（标量张量，带计算图）。
    """
    cur = flow_g[idx[:, None], t0[:, None] + ar[None, :]].clone()   # (B, LB)
    n_sites = flow_g.shape[0]
    for k in range(K):
        t = t0 + lookback - 1 + k                                   # 当前末刻 (B,)
        if use_spatial:
            mk = st_g[idx][:, 0]                                    # (B,G,G) 汇水区掩膜
            rnow = cs_g[t] - cs_g[t - 1]
            lo = (t - 71).clamp_min(0)
            base = cs_g[(lo - 1).clamp_min(0)] * (lo > 0).float()[:, None, None]
            rain72 = cs_g[t] - base                                  # 72h 累计
            x = torch.stack([rnow * mk, rain72 * mk], 1)             # (B,2,G,G)
            if getattr(model, "masked_pool", False):
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
        pred = model(x, hist, fut, idx)[:, 0, :]                    # (B,Q) 全部量化头
        y = flow_g[idx, t + 1]                                      # 下一时刻真值
        ok = fin_g[idx, t + 1].float()                              # 目标缺测掩膜
        if taus is not None:
            err = pred - y.unsqueeze(-1)                            # (B,Q)
            l = torch.maximum(taus * err, (taus - 1) * err).mean(-1)
        else:
            l = nn.functional.huber_loss(pred[:, model.mid], y,
                                         delta=huber_delta, reduction="none")
        # 缺测点损失置零；权重同样乘掩膜，缺测样本不再贡献分母
        denom = (w * ok).sum().clamp(min=1e-9)
        loss = ((l * ok) * w).sum() / denom
        if bias_pen > 0.0:
            # 逐站有符号偏差（标准化单位），再取绝对值平均；按提前量线性加权。
            # 只罚中位头——0.1/0.9 头的偏差是模型刻意留的不确定带，不该罚。
            err_mid = pred[:, model.mid] - y                       # (B,)
            b_sum = torch.zeros(n_sites, device=pred.device)
            b_den = torch.zeros(n_sites, device=pred.device)
            b_sum.index_add_(0, idx, err_mid * ok * w)
            b_den.index_add_(0, idx, ok * w)
            present = b_den > 0
            bias_site = (b_sum[present] / b_den[present].clamp(min=1e-9)).abs().mean()
            loss = loss + bias_pen * (k + 1) / K * bias_site
        yield loss
        # level：输出即流量数值，直接回灌（断梯度）；delta：末刻锚定 + 增量
        nxt = pred[:, model.mid] if output_mode == "level" \
            else cur[:, -1] + pred[:, model.mid]
        cur = torch.cat([cur[:, 1:], nxt.detach().unsqueeze(1)], 1)


def train_model_unrolled(model, ds_tr, ds_va, m, out_dir, device):
    """闭环滚动训练：训练/验证都在滚动条件下算损失，选模型与推理任务一致。

    逐步损失各自反向传播（梯度累加后统一裁剪、更新），任意时刻只常驻
    单步计算图——K=24 的闭环训练在 24 GB 显存上才跑得动。
    """
    K = int(m["unroll"])
    lookback = ds_tr.lookback
    use_spatial, use_future = ds_tr.use_spatial, ds_tr.use_future
    output_mode = ds_tr.output_mode
    every = max(1, int(m.get("unroll_every", 1)))
    delta = float(m.get("huber_delta", 1.0))
    bias_pen = float(m.get("bias_pen", 0.0))   # S.20：>0 逐步加对称漂移惩罚（见 unrolled_steps）
    loss_name = str(m.get("loss", "mse"))
    taus = None
    if loss_name == "quantile":
        taus = torch.tensor([float(v) for v in m.get("quantiles", [0.1, 0.5, 0.9])],
                            device=device)
    bs, epochs = int(m["batch_size"]), int(m["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ar = torch.arange(lookback, device=device)
    flow_tr = torch.from_numpy(np.nan_to_num(ds_tr.flow_n, nan=0.0)).float().to(device)
    fin_tr = torch.from_numpy(np.isfinite(ds_tr.flow_n)).to(device)
    flow_va = torch.from_numpy(np.nan_to_num(ds_va.flow_n, nan=0.0)).float().to(device)
    fin_va = torch.from_numpy(np.isfinite(ds_va.flow_n)).to(device)
    w_all = torch.as_tensor(ds_tr.weights, device=device)
    sam_tr = torch.as_tensor(ds_tr.samples, device=device)          # (n,2) 站号, t0
    sam_va = torch.as_tensor(ds_va.samples, device=device)
    if every > 1:                                   # 闭环 K 步成本≈K 倍单步，
        sam_tr = sam_tr[::every]                    # 抽稀训练样本把单轮耗时压回来；
    cs_tr, st_tr = ds_tr.cs, ds_tr.statics          # 抽样密度对早停选模无偏，只是方差略升
    cs_va, st_va = ds_va.cs, ds_va.statics
    print(f"闭环滚动训练 unroll={K}（每 {every} 取 1 样本，共 {len(sam_tr):,} 条）"
          f"：逐步回灌自身预测、断梯度，损失打在每步上（{loss_name}）"
          + (f"，对称漂移惩罚 bias_pen={bias_pen}" if bias_pen > 0 else ""))

    def run_train():
        model.train(True)
        perm = torch.randperm(len(sam_tr), device=device)
        tot, n = 0.0, 0
        for j in range(0, len(sam_tr), bs):
            sel = perm[j:j + bs]
            s = sam_tr[sel]
            opt.zero_grad()
            for step_loss in unrolled_steps(
                    model, s[:, 0], s[:, 1], flow_tr, fin_tr, cs_tr, st_tr,
                    w_all[sel], K, lookback, use_spatial, use_future, ar,
                    output_mode, taus, delta, bias_pen):
                step_loss.backward()                # 单步图用完即释放
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += step_loss.item() * len(sel)
            n += len(sel)
        return tot / max(n, 1)

    def run_val():
        model.train(False)
        tot, n = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(sam_va), bs):
                s = sam_va[j:j + bs]
                w = torch.ones(len(s), device=device)
                sl = list(unrolled_steps(
                    model, s[:, 0], s[:, 1], flow_va, fin_va, cs_va, st_va,
                    w, K, lookback, use_spatial, use_future, ar,
                    output_mode, taus, delta, bias_pen))
                tot += (sum(v.item() for v in sl) / len(sl)) * len(s)
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
            # 每刷新一次最优就落盘一次：闭环训练一轮要好几个小时，
            # 只在收尾存盘的话，中途断电就整夜白训。最优权重才 0.7 MB，写得起。
            torch.save(best_state, os.path.join(out_dir, "best.pt"))
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  第 {ep:3d} 轮  训练 {tr:.4f}  验证 {va:.4f}", flush=True)
        if wait >= int(m["patience"]):
            print(f"  第 {ep} 轮早停")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))


def unrolled_bptt_loss(model, idx, t0, flow_g, fin_g, cs_g, st_g, w, K, lookback,
                       use_spatial, use_future, ar, output_mode, taus=None,
                       huber_delta=1.0, bias_pen=0.0, adv_weight=1.0,
                       bias_pen_mode="abs", bias_pen_over=0.0, use_ckpt=True):
    """S.20 第二级：整条 K 步闭环轨迹不断梯度（full BPTT），直接优化部署成绩。

    与 unrolled_steps（断梯度逐步损失）的本质区别：

    - 回灌的预测值**不截断梯度**："第 1 步输出"能收到"第 20 步成绩"的反向
      传播。模型因此获得第二根杠杆——可以学会"看到历史窗里是自己的预测
      时做针对性修正"，而不是像逐步目标那样只剩"调全局输出水位"一根杠杆
      （S.18 闭环 / S.20 第一级两次"治好漂移但压掉洪峰"都是这根杠杆的杰作）。
    - 显存对策：逐步前向包在梯度检查点里（torch.utils.checkpoint），反向时
      重算激活。常驻的只有各步输入（历史窗 + 降雨图切片），K=24、批 128 时
      约 1 GB 量级；若不检查点，24 步 CNN 激活要常驻，批 128 也直接爆。
    - 目标函数三项（全部可导、全部在整条轨迹上算）：
        1. 逐步分位数/Huber 损失的逐步平均（保精度底）；
        2. adv_weight × 逐站"对持续性基准的 NSE 提升量"取负——和 S.16 PPO
           的奖励同一思想，但用精确梯度替代采样梯度：模型必须滚动 24 步后
           仍赢过"把起报末刻实测平推 24 小时"的笨基准，逐站算、逐站保护，
           不让大站的分差掩盖小站（S.17 被汇总中位数误导的教训）；
        3. bias_pen × 逐站逐 lead 有符号偏差绝对值（同第一级，随 lead
           线性加权），小权重留着——漂移的显性约束，权重不宜大（第一级
           证明深端大权重会把共享权重整个拉低）。

    返回 (标量损失, (三项分解))，损失带完整计算图，调用方自行 backward 一次。
    """
    import torch.utils.checkpoint as ckpt
    cur = flow_g[idx[:, None], t0[:, None] + ar[None, :]].clone()   # (B, LB)
    anchor = cur[:, -1]                                             # (B,) 持续性锚
    n_sites = flow_g.shape[0]

    def one_step(hist, k):
        """单步前向。空间输入构建放内部：检查点重算时顺带重建，带梯度的
        输入只有 hist（历史窗）；降雨切片从 cs_g 现取（数据无梯度）。"""
        t = t0 + lookback - 1 + k
        if use_spatial:
            mk = st_g[idx][:, 0]
            rnow = cs_g[t] - cs_g[t - 1]
            lo = (t - 71).clamp_min(0)
            base = cs_g[(lo - 1).clamp_min(0)] * (lo > 0).float()[:, None, None]
            rain72 = cs_g[t] - base
            x = torch.stack([rnow * mk, rain72 * mk], 1)
            if getattr(model, "masked_pool", False):
                x = torch.cat([x, mk.unsqueeze(1)], 1)
            fut = ((cs_g[t + 1] - cs_g[t]) * mk).unsqueeze(1)
            if not use_future:
                fut = torch.zeros_like(fut)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            fut = torch.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = torch.zeros((len(idx), 1, 1, 1), device=flow_g.device)
            fut = None
        return model(x, hist, fut, idx)[:, 0, :]                    # (B,Q)

    preds = []
    for k in range(K):
        hist = cur.unsqueeze(-1)                                    # (B,LB,1)
        if use_ckpt and torch.is_grad_enabled():
            pred = ckpt.checkpoint(one_step, hist, k, use_reentrant=False)
        else:
            pred = one_step(hist, k)
        preds.append(pred)
        nxt = pred[:, model.mid] if output_mode == "level" \
            else cur[:, -1] + pred[:, model.mid]
        cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], 1)          # 不断梯度
    P = torch.stack(preds, 1)                                       # (B,K,Q)

    # 真值/掩膜：目标下标 = t0 + lookback + k（与 unrolled_steps 一致）
    off = torch.arange(K, device=flow_g.device)
    t_end = t0[:, None] + lookback + off[None, :]                   # (B,K)
    Y = flow_g[idx[:, None], t_end]
    OK = fin_g[idx[:, None], t_end].float()

    # ---- 项 1：逐步损失平均（缺测置零、权重乘掩膜，口径同 unrolled_steps） ----
    if taus is not None:
        err = P - Y.unsqueeze(-1)                                   # (B,K,Q)
        l = torch.maximum(taus * err, (taus - 1) * err).mean(-1)    # (B,K)
    else:
        l = nn.functional.huber_loss(P[:, :, model.mid], Y,
                                     delta=huber_delta, reduction="none")
    denom = (w[:, None] * OK).sum(dim=0).clamp(min=1e-9)            # (K,)
    quant = (((l * OK) * w[:, None]).sum(dim=0) / denom).mean()

    # ---- 项 3：逐站逐 lead 有符号偏差惩罚（随 lead 线性加权） ----
    # 注意必须在项 2 之前算好 pen：项 2 的"整批被闸门剔光"早退分支要引用它。
    #
    # bias_pen_mode 三种（S.20 三级实验的教训：闭环漂移方向天生向上，"abs"
    # 对称惩罚的均衡点永远在零下——BPTT 版三小站全部负偏、洪峰 -17.8% 就是
    # 证据。模型永远选择"往下压"这条捷径）：
    #   "abs"  对称：罚 |偏差|（均衡点在零下，会牺牲洪峰）；
    #   "under"只罚低估（relu(-偏差)）：禁止往下压这条捷径，逼模型用真本事
    #          治漂移——代价是可能整体偏高、正向漂移失锁，靠逐步项拉住；
    #   "over" 只罚高估：反向对照（预计没用，留作消融）；
    #   "asym" 双边不同权重：under 侧用 bias_pen、over 侧用 bias_pen_over
    #          （< bias_pen）。单边罚实验里它是最优方向的微调——比特里溪的
    #          正漂移（+14.9%）需要 over 侧轻轻拉住，但权重必须明显小于
    #          under 侧，否则均衡点又掉回零下、洪峰再被连坐。
    mid = P[:, :, model.mid]                                        # (B,K)
    pen = torch.zeros((), device=mid.device)
    if bias_pen > 0.0:
        b_sum = torch.zeros(n_sites, K, device=mid.device)
        b_den = torch.zeros(n_sites, K, device=mid.device)
        b_sum.index_add_(0, idx, (mid - Y) * OK * w[:, None])
        b_den.index_add_(0, idx, OK * w[:, None])
        present = b_den > 0
        signed = torch.where(
            present, b_sum / b_den.clamp(min=1e-9), torch.zeros_like(b_sum))
        if bias_pen_mode == "under":
            bias_site = bias_pen * torch.relu(-signed)   # 只罚低估（负偏差）
        elif bias_pen_mode == "over":
            bias_site = bias_pen * torch.relu(signed)    # 只罚高估（正偏差）
        elif bias_pen_mode == "asym":
            # 双边不同权重：under 侧用 bias_pen、over 侧用 bias_pen_over
            # （必须明显小于 under 侧，否则均衡点又掉回零下、洪峰再被连坐）。
            # 单边罚实验里它是最优方向的微调——比特里溪的正漂移（+14.9%）
            # 需要 over 侧轻轻拉住。
            bias_site = (bias_pen * torch.relu(-signed)
                         + bias_pen_over * torch.relu(signed))
        else:                                            # "abs" 对称
            bias_site = bias_pen * signed.abs()
        # 缺测 lead 已置零：每站"惩罚总和 ÷ 有数据 lead 数"再跨站平均
        pen = (bias_site.sum(1)
               / present.sum(1).clamp(min=1).float()).mean()

    # ---- 项 2：逐站 NSE 提升量（相对持续性基准）取负 ----
    def site_sum(v):                                                # (B,K) → (站,)
        out = torch.zeros(n_sites, device=v.device)
        return out.index_add_(0, idx, (v * OK).sum(1))
    cnt = torch.zeros(n_sites, device=mid.device)
    cnt.index_add_(0, idx, OK.sum(1))
    sum_y = site_sum(Y)
    sse_sim = site_sum((mid - Y) ** 2)
    sse_per = site_sum((anchor[:, None] - Y) ** 2)
    sse_tot = site_sum(Y ** 2)
    good = cnt >= 4
    c = cnt[good].clamp(min=1.0)
    sst = sse_tot[good] - sum_y[good] ** 2 / c          # 该批内该站的离差平方和
    # 方差闸门：近常数窗口（死水/缺测碎片）上 NSE 是爆炸量纲——分母趋零时
    # 一点点误差被放大成天梯度的灾难点（首轮训练提升量冲到 -10 万就是
    # 比特里溪这类缺测小站触发的）。全局标准化下 1.0 = 全域标准差，
    # 窗口方差 < 1e-3（标准差 < 0.03）的水情没有预报意义，直接整站剔除。
    var = sst / c
    keep = var > 1e-3
    if not keep.any():          # 整批都被闸门剔光时退化为只算逐步损失，保住数值稳定
        zero = torch.zeros_like(quant)
        return quant + pen, (quant, zero, pen)
    sst = sst[keep].clamp(min=1e-9)
    nse_sim = 1.0 - sse_sim[good][keep] / sst
    nse_per = 1.0 - sse_per[good][keep] / sst
    # 提升量截断 [-2, 1]：赢过完美（1.0）没有额外信息，输穿 -2 的灾难站
    # 只保留 capped 梯度——它的绝对误差由逐步项继续惩罚，不许劫持整批梯度。
    advantage = (nse_sim - nse_per).clamp(min=-2.0, max=1.0)
    adv = -(adv_weight * advantage.mean())              # 要最大化提升量

    return quant + adv + pen, (quant, adv, pen)


def train_model_bptt(model, ds_tr, ds_va, m, out_dir, device):
    """S.20 第二级训练循环：结构对照 train_model_unrolled，差别只在

    - 每批样本整条 K 步轨迹只 backward 一次（计算图跨全部 K 步）；
    - 损失是 unrolled_bptt_loss 的三项轨迹目标；
    - 逐轮打印三项分解（逐步/提升量/漂移罚），方便看出模型在拿哪项换哪项。
    """
    K = int(m["unroll"])
    lookback = ds_tr.lookback
    use_spatial, use_future = ds_tr.use_spatial, ds_tr.use_future
    output_mode = ds_tr.output_mode
    every = max(1, int(m.get("unroll_every", 1)))
    delta = float(m.get("huber_delta", 1.0))
    bias_pen = float(m.get("bias_pen", 0.0))
    bias_mode = str(m.get("bias_pen_mode", "abs"))
    bias_over = float(m.get("bias_pen_over", 0.0))
    adv_w = float(m.get("adv_weight", 1.0))
    loss_name = str(m.get("loss", "quantile"))
    taus = None
    if loss_name == "quantile":
        taus = torch.tensor([float(v) for v in m.get("quantiles", [0.1, 0.5, 0.9])],
                            device=device)
    bs, epochs = int(m["batch_size"]), int(m["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ar = torch.arange(lookback, device=device)
    flow_tr = torch.from_numpy(np.nan_to_num(ds_tr.flow_n, nan=0.0)).float().to(device)
    fin_tr = torch.from_numpy(np.isfinite(ds_tr.flow_n)).to(device)
    flow_va = torch.from_numpy(np.nan_to_num(ds_va.flow_n, nan=0.0)).float().to(device)
    fin_va = torch.from_numpy(np.isfinite(ds_va.flow_n)).to(device)
    w_all = torch.as_tensor(ds_tr.weights, device=device)
    sam_tr = torch.as_tensor(ds_tr.samples, device=device)
    sam_va = torch.as_tensor(ds_va.samples, device=device)
    if every > 1:                                   # 闭环 K 步成本≈K 倍单步，
        sam_tr = sam_tr[::every]                    # 抽稀训练样本把单轮耗时压回来；
    cs_tr, st_tr = ds_tr.cs, ds_tr.statics          # 抽样密度对早停选模无偏，只是方差略升
    cs_va, st_va = ds_va.cs, ds_va.statics
    print(f"闭环 BPTT 训练 unroll={K}（每 {every} 取 1 样本，共 {len(sam_tr):,} 条）"
          f"：整条轨迹不断梯度，adv_weight={adv_w} bias_pen={bias_pen}"
          f"（{bias_mode}），逐步检查点保显存")

    def run_train():
        model.train(True)
        perm = torch.randperm(len(sam_tr), device=device)
        tot, n = 0.0, 0
        parts = [0.0, 0.0, 0.0]
        for j in range(0, len(sam_tr), bs):
            sel = perm[j:j + bs]
            s = sam_tr[sel]
            opt.zero_grad()
            loss, decomp = unrolled_bptt_loss(
                model, s[:, 0], s[:, 1], flow_tr, fin_tr, cs_tr, st_tr,
                w_all[sel], K, lookback, use_spatial, use_future, ar,
                output_mode, taus, delta, bias_pen, adv_w, bias_mode,
                bias_over)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(sel)
            for i_, v in enumerate(decomp):
                parts[i_] += v.item() * len(sel)
            n += len(sel)
        return tot / max(n, 1), [v / max(n, 1) for v in parts]

    def run_val():
        model.train(False)
        tot, n = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(sam_va), bs):
                s = sam_va[j:j + bs]
                w = torch.ones(len(s), device=device)
                loss, _ = unrolled_bptt_loss(
                    model, s[:, 0], s[:, 1], flow_va, fin_va, cs_va, st_va,
                    w, K, lookback, use_spatial, use_future, ar,
                    output_mode, taus, delta, bias_pen, adv_w, bias_mode,
                    bias_over, use_ckpt=False)
                tot += loss.item() * len(s)
                n += len(s)
        return tot / max(n, 1)

    best, best_state, wait = np.inf, None, 0
    for ep in range(1, epochs + 1):
        tr, tr_parts = run_train()
        sched.step()
        va = run_val()
        if va < best:
            best, wait = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # 每刷新一次最优就落盘一次：闭环训练一轮要好几个小时，
            # 只在收尾存盘的话，中途断电就整夜白训。最优权重才 0.7 MB，写得起。
            torch.save(best_state, os.path.join(out_dir, "best.pt"))
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  第 {ep:3d} 轮  训练 {tr:.4f}（逐步 {tr_parts[0]:.4f} "
                  f"提升量 {-tr_parts[1]:+.4f} 漂移罚 {tr_parts[2]:.4f}）"
                  f"  验证 {va:.4f}", flush=True)
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
    delta = float(m.get("huber_delta", 1.0))
    if loss_name == "quantile":
        taus = torch.tensor([float(v) for v in m.get("quantiles", [0.1, 0.5, 0.9])],
                            device=device)
        qs = "/".join(f"{float(v):g}" for v in m.get("quantiles", [0.1, 0.5, 0.9]))
        print(f"损失 quantile（τ={qs}，pinball，逐站方差归一加权，仅训练）")
    elif loss_name == "huber_nse":
        # E1：Huber(reduction='none') × 样本权重（逐站方差归一），训练批加权、验证不加权
        print(f"损失 huber_nse（delta={delta}，逐站方差归一加权，仅训练）")
    else:
        print(f"损失 {loss_name}")

    def run(loader, train):
        model.train(train)
        tot, n = 0.0, 0
        for x, hist, frain, idx, y, w in loader:
            x, hist, frain, idx, y = (t.to(device, non_blocking=True)
                                      for t in (x, hist, frain, idx, y))
            w = w.to(device, non_blocking=True)
            pred = model(x, hist, frain, idx)          # (B, H, K)
            # 目标缺测掩膜（S.12 起样本允许缺测目标，输入仍要求完整）：
            # 缺测点损失置零，按各样本有效点数归一，不再整窗丢弃
            mk = torch.isfinite(y)                     # (B, H)
            y0 = torch.where(mk, y, torch.zeros_like(y))
            cnt_raw = mk.sum(dim=tuple(range(1, mk.ndim)))
            has = cnt_raw > 0                          # 至少 1 个有效目标的样本
            cnt = cnt_raw.clamp(min=1)
            ldims = tuple(range(1, pred.ndim))
            if loss_name == "quantile":
                err = pred - y0.unsqueeze(-1)          # (B, H, K)
                l = (torch.maximum(taus * err, (taus - 1) * err)
                     * mk.unsqueeze(-1))
            else:
                p = pred.squeeze(-1)                   # (B, H)，单分位点
                if loss_name in ("huber_nse", "huber"):
                    l = nn.functional.huber_loss(p, y0, delta=delta,
                                                 reduction="none") * mk
                else:
                    l = ((p - y0) ** 2) * mk
            # 逐样本按有效点平均，再与权重逐样本相乘（分位数路 K 个头各自计点，
            # 与旧口径 mean(H,K) 一致，保证验证损失跨方案可比；完全无有效目标的
            # 样本整体剔除，不计 0 损失拖低均值）
            # l 在分位数损失下为 (B,H,K)，在单头损失下为 (B,H)；
            # 按损失张量自身的维度求和，避免单头路径多取一个不存在的 K 维。
            loss_dims = tuple(range(1, l.ndim))
            K = pred.shape[-1] if loss_name == "quantile" else 1
            ls = l.sum(dim=loss_dims) / (cnt * K)
            if train:
                wm = w * has
                loss = (ls * wm).sum() / wm.sum().clamp(min=1e-9)
            else:
                loss = ls[has].mean() if has.any() else ls.sum() * 0.0
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
            # 每刷新一次最优就落盘一次。这个循环可能跑好几个小时（S.18 每轮 5.2 分钟、
            # 上百轮），只在收尾时存盘的话，中途断电或进程被杀就整夜白训。最优权重
            # 才 0.7 MB，写得起。
            torch.save(best_state, os.path.join(out_dir, "best.pt"))
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
    ap.add_argument("--init-from", default=None, metavar="best.pt",
                    help="热启动：从指定权重继续训练（等价于延长训练计划），"
                         "输出目录仍写 out_dir，不影响被加载的源权重")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(os.path.join(ROOT, args.config)), args.set)
    # 先核对轻量元数据，再加载任何大数组；错误时在占用几十 GB 内存前退出。
    contract = validate_data_contract(cfg, ROOT)

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

    ids, names, areas, times, area_rain, flow = load_inputs(cfg, contract)
    mask1km = contract.mask1km
    statics = build_statics(cfg, mask1km)
    n_site, T = flow.shape
    print(f"断面 {n_site} 个  时间轴 {times[0]} ~ {times[-1]}  共 {T} 小时")

    cs = None
    if use_spatial:
        import xarray as xr
        # 延长后的逐小时 float32 网格约 20 GB。若把 astype / nan_to_num /
        # cumsum 串成一条表达式，多个全尺寸临时数组会同时驻留并撑爆内存。
        # 因此逐步原地处理，并在累计和生成后立即释放原始网格。
        rain_grid = xr.open_dataset(os.path.join(ROOT, cfg["paths"]["rain_nc"]))["rain"].values
        if rain_grid.dtype != np.float32:
            rain_grid = rain_grid.astype(np.float32)   # 已是 float32 就别白复制 20 GB
        np.nan_to_num(rain_grid, copy=False, nan=0.0)
        cs = np.cumsum(rain_grid, axis=0)
        del rain_grid
        print(f"累积降雨数组 {cs.shape}  {cs.nbytes / 1e9:.2f} GB")

    n_win = T - lookback - R + 1
    sd = m.get("split_dates")
    if sd:
        # 按日期切分（围绕洪水事件设计）。S.12 起按窗口右端点划线：每个样本的
        # 预测目标全部落在本段内，杜绝 val 目标窗越界伸进 test 头 3 天
        # （审计问题 B：早停选模偷看测试段）。
        b1 = int(times.searchsorted(pd.Timestamp(sd[0])))
        b2 = int(times.searchsorted(pd.Timestamp(sd[1])))
        b1 = min(max(b1, 1), n_win - 1)
        b2 = min(max(b2, b1 + 1), n_win)
        w = lookback + R - 1              # 窗口右端越过起报点的长度（小时）
        bounds = {"train": (0, b1 - w), "val": (b1, b2 - w), "test": (b2, n_win)}
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
                # 输入（历史流量+降雨）必须完整；目标允许缺测——损失与指标对
                # 缺测点逐点屏蔽（S.12 起；审计问题 A：整窗剔除会系统性丢掉
                # 缺测常与洪水同期的那部分窗口，测试集被悄悄换成容易版本）
                if (np.isfinite(h).all() and np.isfinite(y).any()
                        and np.isfinite(r).all()):
                    ss.append((i, t0))
        samples[sp] = ss
        print(f"{sp:5s} {len(ss)} 个样本")

    split_info = assert_calendar_splits(samples, times, lookback, R)
    train_boundary = int(times.searchsorted(pd.Timestamp("2023-01-01")))
    if not 0 < n_tr <= train_boundary:
        raise AssertionError(f"标准化截止下标越出训练段：{n_tr} > {train_boundary}")
    normalization_end = times[n_tr] if n_tr < T else times[-1] + pd.Timedelta(hours=1)

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
    # S.14：输出语义开关。level = 直接输出下一时刻的流量数值本身，不做 diff。
    output_mode = str(m.get("output_mode", "delta"))
    if output_mode not in ("delta", "level"):
        raise ValueError(f"未知 output_mode：{output_mode!r}")
    if output_mode == "level" and delta_cap > 0:
        # 增量限幅的语义是"单步增量不超过 ±delta_cap"，套到流量数值上会把输出
        # 压在 0 附近，直接毁掉模型。必须显式关掉，不能默默沿用。
        raise ValueError("output_mode=level 时 delta_cap 必须为 0")
    # 闭环滚动训练（unroll>0）现已同时支持 delta 与 level 两种输出语义
    # （unrolled_steps 按 output_mode 区分真值口径与回灌方式），分位数损失经
    # taus 传入。注意闭环验证损失是滚动多步的 pinball/huber，数值与单步
    # 教师强迫不可比，早停只在闭环口径内部自洽。
    # cs 是 GB 级的大数组，且 train/val/test 三个数据集共用同一个对象。若交给
    # 各自的加载器分别 .to(device)，显存里会留下三份拷贝：S.14 的 7.6 年是
    # 3×5.5 GB，侥幸没爆；训练段延长到 1990 年后单份就是 18.7 GiB，三份必炸。
    # 所以在这里统一搬一次，三个数据集共享同一个显存张量。
    if device.type == "cuda":
        if cs is not None:
            cs = torch.from_numpy(cs).to(device)
        statics = torch.from_numpy(statics).to(device)
    ds = {k: SiteDataset(cs, statics, flow_n, rain_n, v,
                         use_spatial, use_future, horizon, lookback,
                         weights=w_train if k == "train" else None,
                         areal=(str(m.get("arch", "net")) == "dlinear"),
                         mask_ch=masked_pool, output_mode=output_mode)
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
    elif model_arch == "moe":
        model = MoENet(use_spatial, use_site, int(m["hidden"]),
                       int(m["spatial_dim"]), horizon, n_quant=n_quant,
                       masked_pool=masked_pool, delta_cap=delta_cap,
                       areas=areas).to(device)
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
          + (f"  输出 流量数值" if output_mode == "level" else "  输出 增量")
          + (f"  单步+滚动{rollout}h" if rollout else ""))

    out_dir = os.path.join(ROOT, cfg["paths"]["out_dir"])
    manifest_path = Path(out_dir) / "training_manifest.json"
    # init-from 是有意的续训（延长训练计划），允许 out_dir 已有 best.pt；
    # 其余情况照旧拒绝覆盖，防止手滑冲掉一整夜训练成果。
    if not args.eval_only and not args.init_from and (Path(out_dir) / "best.pt").exists():
        raise FileExistsError(f"输出目录已有 best.pt，拒绝覆盖：{out_dir}")
    if args.eval_only and not manifest_path.is_file():
        raise FileNotFoundError(f"评估要求训练清单，但找不到：{manifest_path}")
    if args.eval_only:
        with open(manifest_path, encoding="utf-8") as f:
            saved_manifest = json.load(f)
        if saved_manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(
                f"评估要求 {MANIFEST_FORMAT} 训练清单；旧模型请重新训练")
        if saved_manifest.get("data_signature") != contract.signature:
            raise ValueError("当前数据签名与训练清单不一致，拒绝评估")
    os.makedirs(out_dir, exist_ok=True)
    if not args.eval_only:
        write_training_manifest(
            manifest_path, contract=contract, split_info=split_info,
            normalization_end_exclusive=normalization_end, config=cfg,
            extra={"lookback": lookback, "prediction_steps": R,
                   "normalization_prefix_end_index": n_tr})
        print(f"训练清单已写 {manifest_path}，数据签名 {contract.signature}")
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                         map_location=device))
        print(f"已加载 {out_dir}/best.pt，跳过训练")
    else:
        if args.init_from:
            # 热启动：先载入权重，再继续走正常训练分发——注意必须是"加载后
            # 继续训"，不是"加载后跳过训"（此前 elif 链把它放进了互斥分支，
            # 热启动会变成纯评估，是个潜伏 bug，S.20 第二级热启动时修掉）。
            src = (args.init_from if os.path.isabs(args.init_from)
                   else os.path.join(ROOT, args.init_from))
            model.load_state_dict(torch.load(src, map_location=device))
            print(f"热启动：已从 {src} 载入权重，新训练计划重新开始计时")
        if int(m.get("unroll", 0)) and bool(m.get("bptt", False)):
            train_model_bptt(model, ds["train"], ds["val"], m, out_dir, device)
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
                # level 模式下 y 和模型输出本身就已经是流量数值，不能再加锚点
                if output_mode == "level":
                    fo.append(y.cpu().numpy())
                    fs.append(dl.cpu().numpy())
                else:
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
            fin = np.isfinite(o) & np.isfinite(s)   # 缺测目标不计入背数检查
            ns_fit.append(nse(o[fin], s[fin]))
            print(f"{ids[i]:11s} {names[i]:10s} NSE {ns_fit[-1]:6.3f}  "
                  f"MAE {np.mean(np.abs(o[fin] - s[fin])):8.3f}")
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
                                               rain_t[idx, lo2:t + 1].sum(dim=1),
                                               rain_t[idx, t + 1]], 1)
                            if not use_future:
                                fut = torch.cat(
                                    [fut[:, :2],
                                     torch.zeros(B, 1, device=device)], 1)
                    hist = cur.unsqueeze(-1)              # (B, LB, 1) 仅流量
                    out = model(x, hist, fut, idx)[:, 0, model.mid]
                    # level：模型输出就是流量数值本身，绝不能再加一次当前流量
                    nxt = out if output_mode == "level" else cur[:, -1] + out
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
                pred = model(x, hist, frain, idx)[:, :, model.mid]   # 中位分位点
                if output_mode == "level":
                    obs_l.append(y.cpu().numpy())
                    sim_l.append(pred.cpu().numpy())
                else:
                    anchor = hist[:, -1, 0:1]
                    obs_l.append((y + anchor).cpu().numpy())
                    sim_l.append((pred + anchor).cpu().numpy())
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
        oo = inv(obs_n[mk], i)
        ss = inv(sim_n[mk], i)
        fin = np.isfinite(oo) & np.isfinite(ss)   # 目标缺测逐点剔除（S.12）
        obs, sim = oo[fin], ss[fin]
        if obs.size < 4:
            print(f"{ids[i]:11s} {names[i]:10s} {areas[i]:6.0f} 有效点不足，跳过")
            continue
        # 持续性 = 锚点实测（起报前最后 1h，t0+71）；不用 obs[:,0]（那是 t0+72，
        # 起报时不可能知道，等于偷看 1h——审计问题 C）
        anchor = flow_n[st[mk], t0_arr[mk] + lookback - 1]
        per = inv(np.repeat(anchor[:, None], R, axis=1), i)[fin]   # 与 obs/sim 同口径屏蔽
        a, b = nse(obs, sim), nse(obs, per)
        ka, r, al, be = kge(obs, sim)
        kp = kge(obs, per)[0]
        op = np.nanmax(oo, axis=1)
        sp = np.nanmax(ss, axis=1)
        big = op >= np.nanquantile(op, 0.9)
        if big.sum():
            of, sf = oo[big].ravel(), ss[big].ravel()
            finp = np.isfinite(of) & np.isfinite(sf)
            bo.append(of[finp])
            bs.append(sf[finp])
            # 峰偏差只在整行（R 步）都干净的样本上算
            ok = finp.reshape(int(big.sum()), -1).all(axis=1)
            pkb = (np.median((sp[big][ok] - op[big][ok])
                             / np.maximum(op[big][ok], 1e-6))
                   if ok.sum() else np.nan)
        else:
            pkb = np.nan
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
        row = []
        for L in leads:
            o, s = oo[:, L - 1], ss[:, L - 1]
            fin = np.isfinite(o) & np.isfinite(s)
            row.append(nse(o[fin], s[fin]) if fin.sum() >= 4 else np.nan)
        lead_rows.append(row)
    lr_arr = np.asarray(lead_rows)
    lead_med = np.median(lr_arr, axis=0)
    print("\n预见期  " + "  ".join(f"{L:+3d}h" for L in leads))
    print("中位NSE " + "  ".join(f"{v:5.3f}" for v in lead_med))
    summary["nse_by_lead"] = {f"{L}h": float(v) for L, v in zip(leads, lead_med)}

    if model_arch == "moe":
        # 分工报告：门控第二项（专家2）权重——按站、按流量档统计，
        # 回答"模型是否按预期把大小站/大小流分开"（软门控，权重和为 1）。
        gw, gs, gz = [], [], []
        with torch.no_grad():
            for x, hist, frain, idx, y, _w in ldr["test"]:
                x, hist, frain, idx = (t.to(device)
                                       for t in (x, hist, frain, idx))
                gw.append(model.gate_w(hist, idx)[:, 1].cpu().numpy())
                gs.append(idx.cpu().numpy())
                gz.append(hist[:, -1, 0].cpu().numpy())
        gw = np.concatenate(gw)
        gs = np.concatenate(gs)
        gz = np.concatenate(gz)
        print("\n门控分工报告（数值=专家2的平均权重，越大越靠专家2；按面积排序）")
        order = np.argsort(areas)
        for i in order:
            mk = gs == i
            if mk.sum():
                print(f"{ids[i]:11s} {names[i]:10s} {areas[i]:6.0f} km²  "
                      f"专家2权重 {gw[mk].mean():.3f}")
        qs = np.quantile(gz, [0.5, 0.9, 0.99])
        print("按当前流量分档（全站合并）：")
        print(f"  小水 z<{qs[0]:6.2f}      专家2权重 {gw[gz < qs[0]].mean():.3f}")
        for lo_, hi_, lab in [(qs[0], qs[1], "中水"), (qs[1], qs[2], "大水"),
                              (qs[2], np.inf, "特大")]:
            mk = (gz >= lo_) & (gz < hi_)
            if mk.sum():
                print(f"  {lab} z∈[{lo_:5.2f},{hi_:5.2f})  专家2权重 {gw[mk].mean():.3f}")

    np.savez_compressed(
        os.path.join(out_dir, "predictions.npz"),
        obs=obs_n, sim=sim_n, site=st, t0=t0_arr,
        times=np.array([str(t) for t in times]), q_mean=q_mean, q_std=q_std,
        ids=np.array(ids), names=np.array(names), areas=areas,
        transform=np.array(transform),
        lams=np.array(lams if lams is not None else np.full(n_site, np.nan)),
        output_mode=np.array(output_mode),
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {cfg['paths']['out_dir']}")


if __name__ == "__main__":
    main()
