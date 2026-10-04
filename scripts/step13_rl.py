#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S.13：在 S.10 上做 24 小时闭环微调所需的共用部件。

本文件只放模型、滚动、损失、PPO 数学和存档读写；训练入口在
``train_step13_rl.py``。所有流量都保持 S.10 的 gstd 标准化空间，只有汇报
指标时才换回 m³/s。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import MoENet

LOOKBACK = 72
ROLLOUT = 24
QUANTILES = (0.1, 0.5, 0.9)
NORMAL_Z_80 = 1.2815515655446004
S10_PARAMETER_COUNT = 162743
S13_FORMAT = "cams-s13-v1"


@dataclass(frozen=True)
class Normalization:
    """S.10 存档里的归一化信息，不允许重新估计。"""

    transform: str
    q_mean: np.ndarray
    q_std: np.ndarray
    lams: np.ndarray
    ids: Tuple[str, ...]
    names: Tuple[str, ...]
    areas: np.ndarray
    times: pd.DatetimeIndex


@dataclass(frozen=True)
class SplitOrigins:
    """起报点及目标边界；起报点 t0 的首个目标是 t0+lookback。"""

    train: np.ndarray
    val: np.ndarray
    train_target_stop: int
    val_target_start: int
    val_target_stop: int


@dataclass
class RolloutResult:
    levels: torch.Tensor
    actions: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    entropies: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None


@dataclass
class TrajectoryBatch:
    """PPO 所需的小张量。故意不含 128×128 降雨图。"""

    site: torch.Tensor
    time: torch.Tensor
    history: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    valid: torch.Tensor
    advantage: torch.Tensor
    return_: torch.Tensor

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        return TrajectoryBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def index(self, idx: torch.Tensor) -> "TrajectoryBatch":
        return TrajectoryBatch(**{k: v[idx] for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        return int(self.site.numel())


class TrajectoryBuffer:
    """按整条轨迹收集 PPO 数据，只保存历史流量和索引，不保存降雨图。"""

    forbidden_keys = frozenset({"x", "frain", "rain_image", "rain_images", "spatial"})

    def __init__(self) -> None:
        self._parts: List[Dict[str, torch.Tensor]] = []

    def add(self, **part: torch.Tensor) -> None:
        bad = self.forbidden_keys.intersection(part)
        if bad:
            raise ValueError(f"轨迹缓存不得保存降雨图字段：{sorted(bad)}")
        required = {"site", "time", "history", "action", "old_log_prob", "old_value",
                    "reward", "done", "valid", "advantage", "return_"}
        missing = required.difference(part)
        if missing:
            raise ValueError(f"轨迹字段不完整：{sorted(missing)}")
        bad_shapes = [k for k, v in part.items()
                      if k != "history" and (v.ndim != 1)]
        if part["history"].ndim != 2 or part["history"].shape[1] != LOOKBACK:
            bad_shapes.append("history")
        if bad_shapes:
            raise ValueError(f"轨迹字段形状不符：{sorted(set(bad_shapes))}")
        n = int(part["site"].shape[0])
        if any(int(v.shape[0]) != n for v in part.values()):
            raise ValueError("轨迹字段长度不一致")
        self._parts.append({k: v.detach().cpu() for k, v in part.items()})

    def clear(self) -> None:
        self._parts.clear()

    def __len__(self) -> int:
        return sum(int(p["site"].numel()) for p in self._parts)

    def as_batch(self) -> TrajectoryBatch:
        if not self._parts:
            raise ValueError("轨迹缓存为空")
        return TrajectoryBatch(**{
            k: torch.cat([p[k] for p in self._parts], dim=0)
            for k in self._parts[0]
        })

    @property
    def stores_rain_images(self) -> bool:
        return any(bool(self.forbidden_keys.intersection(p)) for p in self._parts)


def _scalar_text(value: np.ndarray | str) -> str:
    if isinstance(value, str):
        return value
    a = np.asarray(value)
    return str(a.item() if a.ndim == 0 else a.ravel()[0])


def file_sha256(path: os.PathLike[str] | str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_model_archive(run_dir: os.PathLike[str] | str,
                       device: torch.device | str = "cpu",
                       expected_output_mode: Optional[str] = None
                       ) -> Tuple[MoENet, Normalization, Dict[str, Any]]:
    """加载三分位 MoE 权重与归一化存档，并核对输出语义。"""
    run = Path(run_dir)
    pred_path, weight_path = run / "predictions.npz", run / "best.pt"
    if not pred_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"模型存档不完整：需要 {pred_path} 和 {weight_path}")
    with np.load(pred_path, allow_pickle=True) as p:
        required = {"q_mean", "q_std", "ids", "names", "areas", "times", "transform"}
        missing = required.difference(p.files)
        if missing:
            raise ValueError(f"predictions.npz 缺字段：{sorted(missing)}")
        transform = _scalar_text(p["transform"])
        output_mode = (_scalar_text(p["output_mode"]) if "output_mode" in p.files
                       else "delta")
        if transform != "gstd":
            raise ValueError(f"源模型必须使用 gstd，存档却是 {transform!r}")
        if output_mode not in ("delta", "level"):
            raise ValueError(f"源模型输出语义无效：{output_mode!r}")
        if expected_output_mode is not None and output_mode != expected_output_mode:
            raise ValueError(f"源模型输出语义应为 {expected_output_mode!r}，实际为 {output_mode!r}")
        ids = tuple(str(v) for v in p["ids"])
        names = tuple(str(v) for v in p["names"])
        areas = np.asarray(p["areas"], dtype=np.float64)
        q_mean = np.asarray(p["q_mean"], dtype=np.float64)
        q_std = np.asarray(p["q_std"], dtype=np.float64)
        lams = np.asarray(p["lams"] if "lams" in p.files else np.full(len(ids), np.nan))
        times = pd.DatetimeIndex([str(v) for v in p["times"]])
    n_site = len(ids)
    if not (len(names) == len(areas) == len(q_mean) == len(q_std) == n_site):
        raise ValueError("源模型站点元数据长度不一致")
    if not (np.all(np.isfinite(q_mean)) and np.all(np.isfinite(q_std)) and np.all(q_std > 0)):
        raise ValueError("源模型归一化参数含无效值")
    if not (np.allclose(q_mean, q_mean[0]) and np.allclose(q_std, q_std[0])):
        raise ValueError("源模型 gstd 应当只有一组全局均值和标准差")
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("源模型时间轴必须严格递增且无重复")

    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # 兼容旧版 PyTorch
        state = torch.load(weight_path, map_location="cpu")
    head = state.get("experts.0.head.2.weight") if isinstance(state, Mapping) else None
    if head is None or tuple(head.shape) != (3, 96):
        raise ValueError("源 best.pt 必须是带 3 个分位数输出头的 MoE 权重")
    model = MoENet(True, False, 96, 48, 1, n_quant=3,
                   masked_pool=False, delta_cap=0.0, areas=areas)
    if sum(v.numel() for v in model.state_dict().values()) != S10_PARAMETER_COUNT:
        raise RuntimeError("代码里的 MoE 结构已不再与源模型一致")
    model.load_state_dict(state, strict=True)
    model.to(device)
    norm = Normalization(transform, q_mean, q_std, lams, ids, names, areas, times)
    info = {
        "run_dir": str(run.resolve()),
        "predictions": str(pred_path.resolve()),
        "checkpoint": str(weight_path.resolve()),
        "predictions_sha256": file_sha256(pred_path),
        "checkpoint_sha256": file_sha256(weight_path),
        "output_mode": output_mode,
        "parameter_count": sum(v.numel() for v in model.state_dict().values()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, norm, info


def load_s10_archive(run_dir: os.PathLike[str] | str,
                     device: torch.device | str = "cpu") -> Tuple[MoENet, Normalization, Dict[str, Any]]:
    """兼容旧入口：加载增量输出的 S.10 存档。"""
    return load_model_archive(run_dir, device=device, expected_output_mode="delta")


def corrected_split_origins(times: Sequence[Any] | pd.DatetimeIndex,
                            lookback: int = LOOKBACK, rollout: int = ROLLOUT,
                            train_stride: int = 24, val_stride: int = 24,
                            train_stop: str = "2023-01-01",
                            val_stop: str = "2024-01-01") -> SplitOrigins:
    """按目标时刻切段：训练目标早于 2023，验证目标只在 2023。

    验证段首个起报点是 ``2023-01-01 - lookback``，不会像旧实现那样再空掉
    72 小时；末个目标严格早于 2024，训练程序因而没有测试段起报点。
    """
    ti = pd.DatetimeIndex(times)
    if lookback < 1 or rollout < 1:
        raise ValueError("lookback 和 rollout 必须为正数")
    b1 = int(ti.searchsorted(pd.Timestamp(train_stop)))
    b2 = int(ti.searchsorted(pd.Timestamp(val_stop)))
    if b1 < lookback or b2 <= b1 or b2 > len(ti):
        raise ValueError("时间轴不覆盖完整的训练/验证边界")
    tr_last_exclusive = b1 - lookback - rollout + 1
    va_first = b1 - lookback
    va_last_exclusive = b2 - lookback - rollout + 1
    if tr_last_exclusive <= 0 or va_last_exclusive <= va_first:
        raise ValueError("可用时段太短，无法构造完整目标窗")
    train = np.arange(0, tr_last_exclusive, train_stride, dtype=np.int64)
    val = np.arange(va_first, va_last_exclusive, val_stride, dtype=np.int64)
    # 这三条是防止将来改边界时悄悄越界。
    assert np.all(train + lookback + rollout <= b1)
    assert np.all(val + lookback >= b1)
    assert np.all(val + lookback + rollout <= b2)
    return SplitOrigins(train, val, b1, b1, b2)


def filter_site_origins(origins: Sequence[int], flow_n: np.ndarray,
                        rain_n: np.ndarray, lookback: int = LOOKBACK,
                        rollout: int = ROLLOUT,
                        require_target: bool = True) -> np.ndarray:
    """展开站点×起报点，并只检查输入完整；目标缺测允许逐点屏蔽。"""
    out: List[Tuple[int, int]] = []
    for i in range(flow_n.shape[0]):
        for t0_ in origins:
            t0 = int(t0_)
            hist = flow_n[i, t0:t0 + lookback]
            rain = rain_n[i, t0:t0 + lookback + rollout]
            target = flow_n[i, t0 + lookback:t0 + lookback + rollout]
            if (len(hist) == lookback and len(rain) == lookback + rollout
                    and np.isfinite(hist).all() and np.isfinite(rain).all()
                    and (np.isfinite(target).any() or not require_target)):
                out.append((i, t0))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def build_step_inputs(cs: torch.Tensor, statics: torch.Tensor,
                      site: torch.Tensor, time: torch.Tensor,
                      history: torch.Tensor, use_future: bool = True,
                      masked_pool: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """训练、采样和推理共用的单步输入组装器。

    ``time`` 是当前历史末刻。支持每个样本不同的时刻，避免三个入口各写一套
    容易错一小时的代码。
    """
    site = site.long()
    time = time.long()
    m = statics[site, 0]
    prev = (time - 1).clamp_min(0)
    rnow = cs[time] - cs[prev]
    rnow = torch.where((time > 0)[:, None, None], rnow, cs[time])
    lo = (time - 71).clamp_min(0)
    base_idx = (lo - 1).clamp_min(0)
    base = cs[base_idx] * (lo > 0).to(cs.dtype)[:, None, None]
    rain72 = cs[time] - base
    x = torch.stack([rnow * m, rain72 * m], dim=1)
    if masked_pool:
        x = torch.cat([x, m.unsqueeze(1)], dim=1)
    future = ((cs[time + 1] - cs[time]) * m).unsqueeze(1)
    if not use_future:
        future = torch.zeros_like(future)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    future = torch.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)
    if history.ndim == 2:
        history = history.unsqueeze(-1)
    return x, history, future


class S13ActorCritic(nn.Module):
    """S.10 策略、冻结参照策略、分位数方差和一个小价值头。"""

    def __init__(self, source: MoENet, min_std: float = 1e-3,
                 max_std: float = 2.0, base_output_mode: str = "delta") -> None:
        super().__init__()
        if base_output_mode not in ("delta", "level"):
            raise ValueError(f"未知源模型输出语义：{base_output_mode!r}")
        self.base_output_mode = base_output_mode
        self.policy = copy.deepcopy(source)
        self.reference = copy.deepcopy(source)
        self.min_std, self.max_std = float(min_std), float(max_std)
        # exp(0)=1：初始化时均值和 S.10 完全一致，分位数宽度也不改。
        self.log_std_scale = nn.Parameter(torch.zeros(()))
        # 价值只看当前/近24h流量、面积、门控和三分位增量，足够小且不拷贝 CNN。
        self.value_head = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        self.freeze_for_s13()

    @property
    def mid(self) -> int:
        return self.policy.mid

    @property
    def masked_pool(self) -> bool:
        return bool(self.policy.experts[0].masked_pool)

    def freeze_for_s13(self) -> None:
        for p in self.policy.parameters():
            p.requires_grad_(False)
        for expert in self.policy.experts:
            for p in expert.head.parameters():
                p.requires_grad_(True)
        for p in self.policy.gate.parameters():
            p.requires_grad_(True)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()
        self.log_std_scale.requires_grad_(True)
        for p in self.value_head.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True) -> "S13ActorCritic":
        super().train(mode)
        # 参照策略永远不进入训练态。
        self.reference.eval()
        return self

    def _distribution(self, net: MoENet, x: torch.Tensor, history: torch.Tensor,
                      future: torch.Tensor, site: torch.Tensor,
                      learned_scale: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quant = net(x, history, future, site)[:, 0, :]
        # RL 内部统一把动作定义成“相对当前流量的变化量”。S.14 的底模输出
        # 下一小时流量数值，因此先减去当前流量；之后的滚动、PPO 和奖励无需分叉。
        if self.base_output_mode == "level":
            quant = quant - history[:, -1, 0, None]
        mean = quant[:, net.mid]
        spread = (quant[:, -1] - quant[:, 0]).abs() / (2.0 * NORMAL_Z_80)
        scale = self.log_std_scale.exp() if learned_scale else mean.new_tensor(1.0)
        std = (spread * scale).clamp(self.min_std, self.max_std)
        return mean, std, quant

    def policy_stats(self, x: torch.Tensor, history: torch.Tensor,
                     future: torch.Tensor, site: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, quant = self._distribution(self.policy, x, history, future, site, True)
        gate = self.policy.gate_w(history, site)
        z = history[:, -1, 0]
        z24 = history[:, -24:, 0].mean(1)
        area = self.policy.lg_area[site]
        vf = torch.cat([z[:, None], z24[:, None], area[:, None], gate, quant], dim=1)
        # 价值回归只更新自己的小头，不借价值损失推着策略分位数和门控漂移。
        value = self.value_head(vf.detach()).squeeze(-1)
        return mean, std, value, quant

    def reference_stats(self, x: torch.Tensor, history: torch.Tensor,
                        future: torch.Tensor, site: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mean, std, _ = self._distribution(
                self.reference, x, history, future, site, False)
        return mean, std

    def forward(self, x: torch.Tensor, history: torch.Tensor,
                future: torch.Tensor, site: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value, _ = self.policy_stats(x, history, future, site)
        return mean, std, value

    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:
    std = std.clamp_min(torch.finfo(std.dtype).eps)
    return -0.5 * (((action - mean) / std) ** 2 + 2.0 * std.log() + math.log(2.0 * math.pi))


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    std = std.clamp_min(torch.finfo(std.dtype).eps)
    return std.log() + 0.5 * math.log(2.0 * math.pi * math.e)


def gaussian_kl(mean: torch.Tensor, std: torch.Tensor,
                ref_mean: torch.Tensor, ref_std: torch.Tensor) -> torch.Tensor:
    """KL[N(mean,std) || N(ref_mean,ref_std)]，逐元素返回。"""
    eps = torch.finfo(std.dtype).eps
    std, ref_std = std.clamp_min(eps), ref_std.clamp_min(eps)
    return torch.log(ref_std / std) + (std.square() + (mean - ref_mean).square()) / (
        2.0 * ref_std.square()) - 0.5


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(value.dtype)
    return (value * w).sum() / w.sum().clamp_min(1.0)


PEAK_TIME_MAX_SHIFT = 6.0
PEAK_TIME_GATE_K = 3.0


def site_activity_scale(flow_n: np.ndarray, times: Sequence[Any] | pd.DatetimeIndex,
                        train_stop: str = "2023-01-01") -> np.ndarray:
    """每站训练段的流量波动尺度，用来判断一个 24 小时窗是否真有涨水。

    归一化是全局的（所有站共用一组均值和标准差），固定阈值因而对大小站不公平：
    标准化后的 1.0 对纽波特是家常便饭，对比特里溪是不可能有的大洪水。这里按站
    给出各自的波动幅度，时间项的门槛才可比。只用训练段，不碰验证和测试。
    """
    ti = pd.DatetimeIndex(times)
    stop = int(ti.searchsorted(pd.Timestamp(train_stop)))
    if stop < 2:
        raise ValueError("训练段太短，无法估计站点波动尺度")
    seg = np.asarray(flow_n[:, :stop], dtype=np.float64)
    scale = np.nanstd(seg, axis=1)
    return np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)


def peak_time_terms(predicted: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, scale: torch.Tensor,
                    max_shift: float = PEAK_TIME_MAX_SHIFT,
                    gate_k: float = PEAK_TIME_GATE_K
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """峰现时间偏差与涨水显著程度，PPO 奖励和闭环损失共用。

    "峰现时间"取加权重心，不取最大值所在的位置：峰顶平坦时重心给出中间值，
    不会因数值抖动跳好几个小时；这条式子处处可导，监督模式也能直接用。

    实测和预测共用同一个基线（实测窗口最低水位），两者才可比，否则报偏低的
    模型重心会系统性地跟着漂。返回的两个量都在 0~1：``penalty`` 是归一化后的
    时间偏差，``gate`` 是窗口涨水幅度相对该站常态波动的比例——枯水窗口的"峰"
    只是噪声，``gate`` 趋近 0，时间项自动失效。``scale`` 是逐条样本的站点波动
    尺度，必须给，否则全局归一化下大小站不可比。
    """
    steps = predicted.shape[-1]
    idx = torch.arange(steps, dtype=predicted.dtype, device=predicted.device)
    has = mask.any(dim=-1)
    hi = torch.finfo(predicted.dtype).max
    obs_hi = torch.where(mask, target, target.new_full((), hi))
    obs_min = obs_hi.min(dim=-1).values
    obs_max = torch.where(mask, target, target.new_full((), -hi)).max(dim=-1).values
    keep = has.to(predicted.dtype)
    zero = torch.zeros_like(obs_min)
    obs_min = torch.where(has, obs_min, zero)
    obs_max = torch.where(has, obs_max, zero)

    base = obs_min.detach()
    w_obs = torch.where(mask, (target - base[:, None]).clamp_min(0.0),
                        torch.zeros_like(predicted))
    w_pred = torch.where(mask, (predicted - base[:, None]).clamp_min(0.0),
                         torch.zeros_like(predicted))
    t_obs = (w_obs * idx).sum(dim=-1) / w_obs.sum(dim=-1).clamp_min(1e-6)
    t_pred = (w_pred * idx).sum(dim=-1) / w_pred.sum(dim=-1).clamp_min(1e-6)

    penalty = (t_pred - t_obs).abs().clamp(max=max_shift) / max_shift
    gate = ((obs_max - obs_min) / (gate_k * scale)).clamp(0.0, 1.0)
    return penalty * keep, gate * keep


def reward_from_levels(predicted: torch.Tensor, target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       huber_delta: float = 1.0,
                       level_weight: float = 1.0,
                       delta_weight: float = 0.0,
                       peak_weight: float = 0.0,
                       peak_time_weight: float = 0.0,
                       site_scale: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐时轨迹奖励；缺测目标只屏蔽，不把 NaN 当成零流量。

    水位误差放在对应时刻；相邻变化误差放在后一时刻；整条轨迹的峰值误差放在
    最后一个有效时刻。这样 PPO 与 control/验证使用相同三项训练目标。
    """
    if mask is None:
        mask = torch.isfinite(target)
    else:
        mask = mask.bool() & torch.isfinite(target)
    safe_target = torch.where(mask, target, predicted.detach())
    level = F.huber_loss(predicted, safe_target, delta=huber_delta, reduction="none")
    reward = torch.where(mask, -level_weight * level, torch.zeros_like(level))

    if delta_weight and predicted.shape[-1] > 1:
        pair = mask[..., 1:] & mask[..., :-1]
        pd = predicted[..., 1:] - predicted[..., :-1]
        td = safe_target[..., 1:] - safe_target[..., :-1]
        dl = F.huber_loss(pd, td, delta=huber_delta, reduction="none")
        reward[..., 1:] = reward[..., 1:] + torch.where(
            pair, -delta_weight * dl, torch.zeros_like(dl))

    if peak_weight or peak_time_weight:
        has = mask.any(dim=-1)
        add = torch.zeros_like(reward)
        if has.any():
            rows = torch.arange(mask.shape[0], device=mask.device)[has]
            # 每条轨迹只记一次峰值代价，放到最后一个有效目标处。
            rev = torch.flip(mask, dims=(-1,)).to(torch.int64).argmax(dim=-1)
            last = mask.shape[-1] - 1 - rev
            if peak_weight:
                neg = torch.finfo(predicted.dtype).min
                pp = torch.where(mask, predicted,
                                 predicted.new_full((), neg)).max(dim=-1).values
                tp = torch.where(mask, safe_target,
                                 safe_target.new_full((), neg)).max(dim=-1).values
                pl = F.huber_loss(pp, tp, delta=huber_delta, reduction="none")
                add[rows, last[has]] = add[rows, last[has]] - peak_weight * pl[has]
            if peak_time_weight:
                # 时间偏差按涨水显著程度加权：枯水窗口的"峰"是噪声，不计。
                if site_scale is None:
                    raise ValueError("启用峰现时间项时必须给出站点波动尺度")
                pen, gate = peak_time_terms(predicted, safe_target, mask, site_scale)
                add[rows, last[has]] = add[rows, last[has]] - peak_time_weight * (
                    gate * pen)[has]
        reward = reward + add
    return reward, mask


def trajectory_loss(predicted: torch.Tensor, target: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    level_weight: float = 1.0, delta_weight: float = 0.2,
                    peak_weight: float = 0.1, huber_delta: float = 1.0,
                    peak_time_weight: float = 0.0,
                    site_scale: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """24 步物理量级（标准化流量水位）损失，所有分量都按有效点屏蔽。"""
    if mask is None:
        mask = torch.isfinite(target)
    else:
        mask = mask.bool() & torch.isfinite(target)
    safe_target = torch.where(mask, target, predicted.detach())
    level_raw = F.huber_loss(predicted, safe_target, delta=huber_delta, reduction="none")
    level = masked_mean(level_raw, mask)

    if predicted.shape[-1] > 1:
        pair_mask = mask[..., 1:] & mask[..., :-1]
        pd = predicted[..., 1:] - predicted[..., :-1]
        td = safe_target[..., 1:] - safe_target[..., :-1]
        delta = masked_mean(F.huber_loss(pd, td, delta=huber_delta, reduction="none"), pair_mask)
    else:
        delta = predicted.sum() * 0.0

    # 峰值只在一条轨迹至少有一个实测时计算；无效位置先压到极小值，不造假目标。
    has = mask.any(dim=-1)
    neg = torch.finfo(predicted.dtype).min
    pp = torch.where(mask, predicted, predicted.new_full((), neg)).max(dim=-1).values
    tp = torch.where(mask, safe_target, safe_target.new_full((), neg)).max(dim=-1).values
    if has.any():
        peak = F.huber_loss(pp[has], tp[has], delta=huber_delta, reduction="mean")
    else:
        peak = predicted.sum() * 0.0
    # 峰现时间走加权重心，这条式子是光滑的，监督模式可以直接求导。
    if peak_time_weight and has.any():
        if site_scale is None:
            raise ValueError("启用峰现时间项时必须给出站点波动尺度")
        pen, gate = peak_time_terms(predicted, safe_target, mask, site_scale)
        timing = (gate * pen)[has].mean()
    else:
        timing = predicted.sum() * 0.0
    total = (level_weight * level + delta_weight * delta + peak_weight * peak
             + peak_time_weight * timing)
    return total, {"total": total, "level": level, "delta": delta, "peak": peak,
                   "timing": timing,
                   "valid_points": mask.sum().detach()}


def compute_gae(reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor,
                valid: Optional[torch.Tensor] = None, next_value: Optional[torch.Tensor] = None,
                gamma: float = 0.99, lam: float = 0.95
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计。输入最后一维是时间，缺测步不会产生奖励或 TD 误差。"""
    if reward.shape != value.shape or reward.shape != done.shape:
        raise ValueError("reward/value/done 形状必须一致")
    if valid is None:
        valid = torch.ones_like(done, dtype=torch.bool)
    valid = valid.bool()
    # 缺测时刻是“没有监督”，不是一条轨迹的断点。GAE 从后往前时保留后续累计，
    # 只把该时刻自己的 TD 项置零；这样缺测前的动作仍能收到后续有效奖励。
    if next_value is None:
        next_value = torch.zeros_like(value[..., 0])
    adv = torch.zeros_like(reward)
    running = torch.zeros_like(next_value)
    for t in range(reward.shape[-1] - 1, -1, -1):
        nv = next_value if t == reward.shape[-1] - 1 else value[..., t + 1]
        cont = (~done[..., t]).to(reward.dtype)
        active = valid[..., t].to(reward.dtype)
        delta = active * (reward[..., t] + gamma * nv * cont - value[..., t])
        running = delta + gamma * lam * cont * running
        # 缺测时刻不直接进入 PPO 小批次，但累计回报继续向更早动作传播。
        adv[..., t] = running * active
    return adv, adv + value


def rollout_24(model: S13ActorCritic, cs: torch.Tensor, statics: torch.Tensor,
               flow_n: torch.Tensor, site: torch.Tensor, t0: torch.Tensor,
               steps: int = ROLLOUT, stochastic: bool = False,
               use_future: bool = True, target_flow: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> RolloutResult:
    """从实测历史窗起报，之后只把自己的流量预报接回去。"""
    site, t0 = site.long(), t0.long()
    ar = torch.arange(LOOKBACK, device=t0.device)
    cur = flow_n[site[:, None], t0[:, None] + ar[None, :]].clone()
    levels, actions, means, stds, logs, vals, ents = [], [], [], [], [], [], []
    for k in range(steps):
        now = t0 + LOOKBACK - 1 + k
        x, hist, fut = build_step_inputs(cs, statics, site, now, cur,
                                          use_future, model.masked_pool)
        mean, std, value = model(x, hist, fut, site)
        if stochastic:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                                generator=generator)
            action = mean + std * noise
        else:
            action = mean
        nxt = cur[:, -1] + action
        levels.append(nxt)
        actions.append(action)
        means.append(mean)
        stds.append(std)
        logs.append(gaussian_log_prob(action, mean, std))
        vals.append(value)
        ents.append(gaussian_entropy(std))
        cur = torch.cat([cur[:, 1:], nxt[:, None]], dim=1)
    result = RolloutResult(*[torch.stack(v, dim=1) for v in
                             (levels, actions, means, stds, logs, vals, ents)])
    if target_flow is not None:
        kk = torch.arange(steps, device=t0.device)
        target = target_flow[site[:, None], t0[:, None] + LOOKBACK + kk[None, :]]
        result.target = target
        result.target_mask = torch.isfinite(target)
    return result


def flatten_rollout_for_buffer(buffer: TrajectoryBuffer, result: RolloutResult,
                               flow_n: torch.Tensor, site: torch.Tensor,
                               t0: torch.Tensor, gamma: float = 0.99,
                               gae_lambda: float = 0.95,
                               level_weight: float = 1.0,
                               delta_weight: float = 0.0,
                               peak_weight: float = 0.0,
                               peak_time_weight: float = 0.0,
                               site_scale: Optional[torch.Tensor] = None,
                               huber_delta: float = 1.0) -> None:
    """把滚动结果转成 PPO 小样本；历史由动作重建，不放任何降雨图。"""
    if result.target is None or result.target_mask is None:
        raise ValueError("收集 PPO 轨迹时必须提供目标")
    rewards, target_valid = reward_from_levels(
        result.levels, result.target, result.target_mask,
        huber_delta=huber_delta, level_weight=level_weight,
        delta_weight=delta_weight, peak_weight=peak_weight,
        peak_time_weight=peak_time_weight, site_scale=site_scale)
    done = torch.zeros_like(target_valid)
    # 一条 24 步预报到末步即终止；终止态不做价值自举。
    done[:, -1] = True
    next_value = torch.zeros_like(result.values[:, 0])
    adv, ret = compute_gae(rewards, result.values.detach(), done, target_valid,
                           next_value=next_value, gamma=gamma, lam=gae_lambda)
    ar = torch.arange(LOOKBACK, device=t0.device)
    cur = flow_n[site[:, None], t0[:, None] + ar[None, :]].clone()
    histories = []
    for k in range(result.actions.shape[1]):
        histories.append(cur.clone())
        nxt = cur[:, -1] + result.actions[:, k].detach()
        cur = torch.cat([cur[:, 1:], nxt[:, None]], 1)
    history = torch.stack(histories, 1)
    b, h = result.actions.shape
    buffer.add(
        site=site[:, None].expand(b, h).reshape(-1),
        time=(t0[:, None] + LOOKBACK - 1
              + torch.arange(h, device=t0.device)[None, :]).reshape(-1),
        history=history.reshape(b * h, LOOKBACK),
        action=result.actions.detach().reshape(-1),
        old_log_prob=result.log_probs.detach().reshape(-1),
        old_value=result.values.detach().reshape(-1),
        reward=rewards.detach().reshape(-1),
        done=done.reshape(-1),
        valid=target_valid.reshape(-1),
        advantage=adv.detach().reshape(-1),
        return_=ret.detach().reshape(-1),
    )


def ppo_loss(model: S13ActorCritic, batch: TrajectoryBatch,
             cs: torch.Tensor, statics: torch.Tensor, clip_ratio: float = 0.2,
             value_coef: float = 0.5, entropy_coef: float = 0.001,
             kl_coef: float = 0.02) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x, hist, fut = build_step_inputs(cs, statics, batch.site, batch.time,
                                      batch.history, True, model.masked_pool)
    mean, std, value, _ = model.policy_stats(x, hist, fut, batch.site)
    ref_mean, ref_std = model.reference_stats(x, hist, fut, batch.site)
    logp = gaussian_log_prob(batch.action, mean, std)
    mask = batch.valid.bool()
    if mask.any():
        a = batch.advantage
        av = a[mask]
        norm_adv = torch.zeros_like(a)
        norm_adv[mask] = (av - av.mean()) / av.std(unbiased=False).clamp_min(1e-6)
        ratio = torch.exp(logp - batch.old_log_prob)
        unclipped = ratio * norm_adv
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * norm_adv
        policy_loss = -masked_mean(torch.minimum(unclipped, clipped), mask)
        value_loss = 0.5 * masked_mean((value - batch.return_) ** 2, mask)
        entropy = masked_mean(gaussian_entropy(std), mask)
        kl = masked_mean(gaussian_kl(mean, std, ref_mean, ref_std), mask)
        clip_fraction = masked_mean((torch.abs(ratio - 1.0) > clip_ratio).float(), mask)
    else:
        zero = mean.sum() * 0.0
        policy_loss = value_loss = entropy = kl = clip_fraction = zero
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy + kl_coef * kl
    return total, {"total": total, "policy": policy_loss, "value": value_loss,
                   "entropy": entropy, "kl": kl, "clip_fraction": clip_fraction,
                   "valid_points": mask.sum().detach()}


def iter_minibatches(batch: TrajectoryBatch, batch_size: int,
                     generator: Optional[torch.Generator] = None) -> Iterator[TrajectoryBatch]:
    # TrajectoryBatch 存在 CPU；CPU 生成器可跨 CPU/GPU 训练稳定复现。
    order = torch.randperm(len(batch), generator=generator, device="cpu")
    for start in range(0, len(batch), batch_size):
        yield batch.index(order[start:start + batch_size])


def ppo_update(model: S13ActorCritic, optimizer: torch.optim.Optimizer,
               batch: TrajectoryBatch, cs: torch.Tensor, statics: torch.Tensor,
               epochs: int = 4, batch_size: int = 256,
               max_grad_norm: float = 1.0,
               generator: Optional[torch.Generator] = None,
               **loss_kwargs: float) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    model.train(True)
    for _ in range(epochs):
        for mini in iter_minibatches(batch, batch_size, generator=generator):
            mini = mini.to(cs.device)
            loss, comp = ppo_loss(model, mini, cs, statics, **loss_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                     max_grad_norm)
            optimizer.step()
            for key, value in comp.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def checkpoint_payload(model: S13ActorCritic, mode: str, update: int,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       metrics: Optional[Mapping[str, float]] = None,
                       config: Optional[Mapping[str, Any]] = None,
                       source: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format": S13_FORMAT,
        "mode": str(mode),
        "update": int(update),
        "model_state": model.state_dict(),
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
        "source": dict(source or {}),
        "source_output_mode": model.base_output_mode,
        "action_mode": "delta",
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def save_checkpoint(path: os.PathLike[str] | str, model: S13ActorCritic,
                    mode: str, update: int,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    metrics: Optional[Mapping[str, float]] = None,
                    config: Optional[Mapping[str, Any]] = None,
                    source: Optional[Mapping[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(model, mode, update, optimizer, metrics, config, source), tmp)
    os.replace(tmp, path)


def load_s13_checkpoint(path: os.PathLike[str] | str, model: S13ActorCritic,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        map_location: torch.device | str = "cpu",
                        expected_source: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, Mapping) or obj.get("format") != S13_FORMAT:
        raise ValueError(f"不是 {S13_FORMAT} 结构化检查点：{path}")
    saved_output_mode = obj.get("source_output_mode")
    saved_action_mode = obj.get("action_mode")
    legacy_delta = (expected_source is None and model.base_output_mode == "delta"
                    and saved_output_mode is None and saved_action_mode is None)
    if not legacy_delta and (saved_output_mode != model.base_output_mode
                             or saved_action_mode != "delta"):
        raise ValueError("检查点的源模型输出语义或 RL 动作语义与当前模型不符")
    if expected_source is not None:
        saved_source = obj.get("source", {})
        for key in ("predictions_sha256", "checkpoint_sha256"):
            if saved_source.get(key) != expected_source.get(key):
                raise ValueError(f"检查点源模型不符：{key}")
    model.load_state_dict(obj["model_state"], strict=True)
    model.freeze_for_s13()
    if optimizer is not None and "optimizer_state" in obj:
        optimizer.load_state_dict(obj["optimizer_state"])
    return dict(obj)


def write_manifest(path: os.PathLike[str] | str, *, mode: str,
                   source: Mapping[str, Any], config: Mapping[str, Any],
                   artifacts: Mapping[str, str], best: Optional[Mapping[str, Any]] = None) -> None:
    doc = {
        "format": S13_FORMAT,
        # 方案名从配置里取：奖励一改就是另一版实验，写死会让清单说谎。
        "scheme": str(config.get("scheme", "S.16")),
        "mode": mode,
        "source_model": dict(source),
        "source_output_mode": source.get("output_mode", "delta"),
        "action_mode": "delta",
        "config": dict(config),
        "artifacts": dict(artifacts),
        "best": dict(best or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_manifest(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != S13_FORMAT:
        raise ValueError(f"清单格式不符：{path}")
    return doc
