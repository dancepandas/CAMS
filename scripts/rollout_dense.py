#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐小时（或指定步长）起报的密集滚动推理。

兼容旧命令 ``--run moe|step12``，也可用 ``--run-dir`` 指向 S.13 目录，并从
manifest.json + 结构化检查点加载。归一化参数只从已有预测存档读取，不再按当前
切分重新拟合。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_contract import MANIFEST_FORMAT, validate_data_contract
from step13_rl import (LOOKBACK, PEAK_TIME_GATE_K, S13ActorCritic, build_step_inputs,
                       load_model_archive, load_s13_checkpoint, read_manifest,
                       site_activity_scale)
from train import (DLinearNet, MoENet, Net, build_statics, kge, load_config,
                   load_inputs, make_inv, nse)

LEADS = (1, 3, 6, 12, 24)
LEGACY_RUNS = {"moe": "site_model_moe", "step12": "site_model_step12",
               "step14": "site_model_step14", "step15": "site_model_step15_huber",
               "step18": "site_model_step18"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S.10/S.12/S.13 密集滚动推理")
    ap.add_argument("--config", default="configs/pipeline.yaml",
                    help="管线配置；换用延长后的降雨数据时要指向新的配置")
    ap.add_argument("--run", choices=tuple(LEGACY_RUNS), default=None,
                    help="兼容旧入口：runs/site_model_<run>")
    ap.add_argument("--run-dir", default=None,
                    help="模型目录；支持普通监督训练目录或 S.13 目录")
    ap.add_argument("--checkpoint", default=None, help="权重路径；默认 run-dir/best.pt")
    ap.add_argument("--manifest", default=None,
                    help="训练清单；默认按目录中的 training_manifest.json 或 manifest.json 判断")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--roll", type=int, default=None, help="滚动小时数；S.13 默认24，旧模型默认12")
    ap.add_argument("--stride", type=int, default=1, help="起报间隔小时数")
    ap.add_argument("--output", default=None, help="输出 npz；默认写入模型目录")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return ap.parse_args()


def scalar_text(v: Any) -> str:
    a = np.asarray(v)
    return str(a.item() if a.ndim == 0 else a.ravel()[0])


def archived_metadata(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到归一化存档：{path}")
    with np.load(path, allow_pickle=True) as p:
        need = {"q_mean", "q_std", "ids", "names", "areas", "times", "transform"}
        miss = need.difference(p.files)
        if miss:
            raise ValueError(f"归一化存档缺字段：{sorted(miss)}")
        return {
            "q_mean": np.asarray(p["q_mean"], dtype=np.float64),
            "q_std": np.asarray(p["q_std"], dtype=np.float64),
            "ids": tuple(str(v) for v in p["ids"]),
            "names": tuple(str(v) for v in p["names"]),
            "areas": np.asarray(p["areas"], dtype=np.float64),
            "times": pd.DatetimeIndex([str(v) for v in p["times"]]),
            "transform": scalar_text(p["transform"]),
            "lams": np.asarray(p["lams"] if "lams" in p.files else np.full(len(p["ids"]), np.nan)),
            # 输出语义：旧存档没有这个字段，一律当增量处理；S.14 起为 level
            # （模型直接输出下一小时流量数值，滚动时不能再加当前流量）。
            "output_mode": (scalar_text(p["output_mode"]) if "output_mode" in p.files
                            else "delta"),
        }


def read_run_manifest(path: Path) -> Tuple[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到训练清单：{path}")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    format_name = document.get("format")
    if format_name == MANIFEST_FORMAT:
        return "supervised", document
    if format_name == "cams-s13-v1":
        return "s13", document
    if format_name == "cams-training-manifest-v1":
        raise ValueError("训练清单是旧 v1 格式，未绑定降雨和流量数值；请重新训练")
    raise ValueError(f"不支持的训练清单格式：{format_name!r}")


def resolve_paths(args: argparse.Namespace
                  ) -> Tuple[Path, Path, Path | None, str, Dict[str, Any] | None]:
    if args.run_dir and args.run:
        raise ValueError("--run-dir 和 --run 只能选一个")
    if not args.run_dir:
        run_name = args.run or "moe"
        run_dir = ROOT / "runs" / LEGACY_RUNS[run_name]
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        return run_dir, checkpoint, None, "legacy", None

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.is_absolute():
            manifest = ROOT / manifest
    else:
        choices = [run_dir / "training_manifest.json", run_dir / "manifest.json"]
        existing = [path for path in choices if path.is_file()]
        if len(existing) != 1:
            if not existing:
                raise FileNotFoundError(
                    f"模型目录缺少 training_manifest.json 或 manifest.json：{run_dir}")
            raise ValueError("模型目录同时有两种训练清单；请用 --manifest 明确指定")
        manifest = existing[0]
    kind, document = read_run_manifest(manifest)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    return run_dir, checkpoint, manifest, kind, document


def supervised_config(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    config = manifest.get("effective_config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict) \
            or not isinstance(config.get("paths"), dict):
        raise ValueError("普通监督训练清单缺少完整 effective_config")
    return config


def build_supervised_model(model_cfg: Mapping[str, Any], areas: np.ndarray,
                           n_site: int) -> torch.nn.Module:
    lookback = int(model_cfg["lookback"])
    horizon = int(model_cfg["horizon"])
    use_spatial = bool(model_cfg["use_spatial"])
    use_site = bool(model_cfg.get("use_site", False))
    masked_pool = bool(model_cfg.get("masked_pool", False))
    delta_cap = float(model_cfg.get("delta_cap", 0.0))
    n_quant = (len(model_cfg.get("quantiles", [0.1, 0.5, 0.9]))
               if str(model_cfg.get("loss")) == "quantile" else 1)
    arch = str(model_cfg.get("arch", "net"))
    if arch == "dlinear":
        return DLinearNet(lookback, n_quant=n_quant, masked_pool=masked_pool,
                          delta_cap=delta_cap)
    if arch == "dlinear_cnn":
        return DLinearNet(lookback, n_quant=n_quant, use_spatial=True,
                          spatial_dim=int(model_cfg["spatial_dim"]),
                          masked_pool=masked_pool, delta_cap=delta_cap)
    if arch == "moe":
        return MoENet(use_spatial, use_site, int(model_cfg["hidden"]),
                      int(model_cfg["spatial_dim"]), horizon, n_quant=n_quant,
                      masked_pool=masked_pool, delta_cap=delta_cap, areas=areas)
    return Net(use_spatial, use_site, int(model_cfg["hidden"]),
               int(model_cfg["spatial_dim"]), horizon,
               att_heads=int(model_cfg.get("att_heads", 0)),
               att_mode=str(model_cfg.get("att_mode", "cat")),
               att_excl=bool(model_cfg.get("att_excl", False)),
               n_site=n_site if bool(model_cfg.get("use_tok", False)) else 0,
               tok_dim=int(model_cfg.get("tok_dim", 8)),
               use_feat=bool(model_cfg.get("use_feat", False)),
               areas=areas if bool(model_cfg.get("use_feat", False)) else None,
               n_quant=n_quant, masked_pool=masked_pool, delta_cap=delta_cap)


def require_output_mode(meta: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> str:
    expected = str(model_cfg.get("output_mode", "delta"))
    actual = str(meta["output_mode"])
    if actual != expected:
        raise ValueError(
            f"预测存档 output_mode={actual} 与训练清单 {expected} 不一致")
    return expected


def model_masked_pool(model: torch.nn.Module) -> bool:
    return bool(model.experts[0].masked_pool) if isinstance(model, MoENet) \
        else bool(model.masked_pool)


def split_origins(times: pd.DatetimeIndex, begin: pd.Timestamp,
                  end_exclusive: pd.Timestamp, lookback: int,
                  roll: int, stride: int) -> np.ndarray:
    """按目标时段返回历史窗口起始下标 ``t0``。

    历史窗口为 ``[t0, t0+lookback)``；最后一个已知流量（物理起报时刻）是
    ``times[t0+lookback-1]``，首个预测目标是下一小时
    ``times[t0+lookback]``。``begin/end_exclusive`` 约束的是目标时刻，不是
    ``t0`` 本身。因此不同历史起点的数据集可以按相同物理目标时刻公平对齐。
    """
    target_lo = int(times.searchsorted(pd.Timestamp(begin)))
    target_hi = int(times.searchsorted(pd.Timestamp(end_exclusive)))
    first = target_lo - lookback
    stop = target_hi - lookback - roll + 1
    if first < 0 or stop <= first:
        raise ValueError(f"时间轴无法构造 {begin} 至 {end_exclusive} 的完整窗口")
    origins = np.arange(first, stop, stride, dtype=np.int64)
    assert np.all(origins + lookback >= target_lo)
    assert np.all(origins + lookback + roll <= target_hi)
    return origins


def legacy_split_origins(times: pd.DatetimeIndex, split: str, roll: int,
                         stride: int) -> np.ndarray:
    """复现旧脚本按起报点切段的网格，仅供 --run moe|step12 回归。"""
    begin = pd.Timestamp("2023-01-01" if split == "val" else "2024-01-01")
    end = pd.Timestamp("2024-01-01" if split == "val" else "2025-01-01")
    first = int(times.searchsorted(begin))
    last_target = int(times.searchsorted(end))
    stop = min(last_target, len(times)) - LOOKBACK - roll + 1
    if stop <= first:
        raise ValueError(f"时间轴无法构造旧口径 {split} 窗口")
    return np.arange(first, stop, stride, dtype=np.int64)


def valid_samples(origins: Sequence[int], flow_n: np.ndarray, area_rain: np.ndarray,
                  lookback: int, roll: int) -> np.ndarray:
    rows: List[Tuple[int, int]] = []
    for i in range(flow_n.shape[0]):
        for t0_ in origins:
            t0 = int(t0_)
            hist = flow_n[i, t0:t0 + lookback]
            rain = area_rain[i, t0:t0 + lookback + roll]
            if len(hist) == lookback and len(rain) == lookback + roll \
                    and np.isfinite(hist).all() and np.isfinite(rain).all():
                rows.append((i, t0))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def safe_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    good = np.isfinite(obs) & np.isfinite(sim)
    return nse(obs[good], sim[good]) if good.sum() >= 4 else float("nan")


def peak_timing_metrics(obs: np.ndarray, sim: np.ndarray, scale: float,
                        gate_min: float = 0.25,
                        gate_k: float = PEAK_TIME_GATE_K) -> Dict[str, float]:
    """峰现时间与涨退水段重心偏差；只用实测确有涨水的场次。

    ``obs``/``sim`` 形状 (n, roll)，已还原为物理流量。峰现时间取"高出窗口最低
    水位那部分"的时间重心。``scale`` 是该站的常态波动尺度，用来排除枯水窗口——
    那里的"峰"只是噪声。判定方式和训练奖励完全一致，否则评估和训练看的不是
    同一件事。
    """
    steps = obs.shape[1]
    idx = np.arange(steps, dtype=float)[None, :]
    good = np.isfinite(obs)
    nan = float("nan")
    out = {"peak_time_bias_median": nan, "rise_centroid_bias_median": nan,
           "fall_centroid_bias_median": nan, "n_peak_samples": 0}
    keep = good.sum(axis=1) >= 4
    if not keep.any():
        return out
    # 先在有效行上取极值：缺测太多的场次整行都是 NaN，直接算极值会刷一屏告警。
    obs, sim, good = obs[keep], sim[keep], good[keep]
    o = np.where(good, obs, nan)
    o_min = np.nanmin(o, axis=1)
    o_max = np.nanmax(o, axis=1)
    gate = np.clip((o_max - o_min) / (gate_k * max(scale, 1e-6)), 0.0, 1.0)
    base = o_min[:, None]
    wo = np.where(good, np.maximum(o - base, 0.0), 0.0)
    ws = np.where(good, np.maximum(sim - base, 0.0), 0.0)

    def centroid(w: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        ww = w if seg is None else w * seg
        total = ww.sum(1)
        return (ww * idx).sum(1) / np.where(total > 1e-6, total, nan)

    to, ts = centroid(wo), centroid(ws)
    sel = np.where(gate >= gate_min)[0]
    if not len(sel):
        return out
    # 以实测峰现时间为界切开，分别看涨水段和退水段的重心位置。
    half = np.clip(np.rint(np.nan_to_num(to[sel], nan=steps / 2)).astype(int), 1, steps - 2)
    rise = idx <= half[:, None]
    fall = idx >= half[:, None]
    out["peak_time_bias_median"] = float(np.nanmedian(np.abs(ts[sel] - to[sel])))
    out["rise_centroid_bias_median"] = float(
        np.nanmedian(np.abs(centroid(ws[sel], rise) - centroid(wo[sel], rise))))
    out["fall_centroid_bias_median"] = float(
        np.nanmedian(np.abs(centroid(ws[sel], fall) - centroid(wo[sel], fall))))
    out["n_peak_samples"] = int(len(sel))
    return out


def robust_metrics(obs_n: np.ndarray, sim_n: np.ndarray, site: np.ndarray,
                   t0: np.ndarray, flow_n: np.ndarray, meta: Mapping[str, Any],
                   lookback: int, roll: int) -> Dict[str, Any]:
    inv = make_inv(meta["transform"], meta["q_mean"], meta["q_std"], meta["lams"])
    # 峰现时间项的门槛要按站给尺度，和训练奖励用同一条规则（只用训练段）。
    scale_n = site_activity_scale(flow_n, meta["times"])
    rows, lead_rows, timing_rows = [], [], []
    print(f"\n{'站号':11s} {'名称':10s} {'NSE':>8s} {'持续':>8s} {'KGE':>8s} "
          f"{'MAE':>10s} {'RMSE':>10s} {'偏差':>9s} {'峰现':>7s}")
    for i, sid in enumerate(meta["ids"]):
        take = site == i
        if not take.any():
            continue
        oo, ss = inv(obs_n[take], i), inv(sim_n[take], i)
        anchor_n = flow_n[i, t0[take] + lookback - 1]
        pp = inv(np.repeat(anchor_n[:, None], roll, axis=1), i)
        good = np.isfinite(oo) & np.isfinite(ss)
        if good.sum() < 4:
            continue
        o, s, p = oo[good], ss[good], pp[good]
        ns, npers = nse(o, s), nse(o, p)
        kg = kge(o, s)[0]
        mae = float(np.mean(np.abs(o - s)))
        rmse = float(np.sqrt(np.mean((o - s) ** 2)))
        bias = float(np.mean(s - o))
        rows.append((ns, npers, kg, mae, rmse, bias))
        lr = [safe_nse(oo[:, lead - 1], ss[:, lead - 1])
              for lead in LEADS if lead <= roll]
        lead_rows.append(lr)
        timing = peak_timing_metrics(oo, ss, float(scale_n[i]))
        timing_rows.append(timing)
        print(f"{sid:11s} {meta['names'][i]:10s} {ns:8.3f} {npers:8.3f} {kg:8.3f} "
              f"{mae:10.2f} {rmse:10.2f} {bias:9.2f} "
              f"{timing['peak_time_bias_median']:7.2f}")
    arr = np.asarray(rows, dtype=float)
    lead_arr = np.asarray(lead_rows, dtype=float)
    leads = [v for v in LEADS if v <= roll]
    if arr.size == 0:
        summary = {
            "n_sites": 0, "n_samples": int(len(site)),
            "n_valid_targets": int((np.isfinite(obs_n) & np.isfinite(sim_n)).sum()),
            "nse_median": float("nan"), "persistence_nse_median": float("nan"),
            "nse_wins": 0, "kge_median": float("nan"),
            "mae_median": float("nan"), "rmse_median": float("nan"),
            "bias_median": float("nan"),
            "nse_by_lead": {f"{lead}h": float("nan") for lead in leads},
            "peak_time_bias_median": float("nan"),
            "rise_centroid_bias_median": float("nan"),
            "fall_centroid_bias_median": float("nan"),
            "n_peak_samples": 0,
        }
        print("所有站的有效目标都不足 4 点，已保存预测但不计算汇总指标")
        return summary
    summary: Dict[str, Any] = {
        "n_sites": int(len(rows)),
        "n_samples": int(len(site)),
        "n_valid_targets": int((np.isfinite(obs_n) & np.isfinite(sim_n)).sum()),
        "nse_median": float(np.nanmedian(arr[:, 0])),
        "persistence_nse_median": float(np.nanmedian(arr[:, 1])),
        "nse_wins": int(np.sum(arr[:, 0] > arr[:, 1])),
        "kge_median": float(np.nanmedian(arr[:, 2])),
        "mae_median": float(np.nanmedian(arr[:, 3])),
        "rmse_median": float(np.nanmedian(arr[:, 4])),
        "bias_median": float(np.nanmedian(arr[:, 5])),
        "nse_by_lead": {f"{lead}h": float(v) for lead, v in
                        zip(leads, np.nanmedian(lead_arr, axis=0))}
                        if lead_arr.size else {f"{lead}h": float("nan") for lead in leads},
        "peak_time_bias_median": float(np.nanmedian(
            [t["peak_time_bias_median"] for t in timing_rows])),
        "rise_centroid_bias_median": float(np.nanmedian(
            [t["rise_centroid_bias_median"] for t in timing_rows])),
        "fall_centroid_bias_median": float(np.nanmedian(
            [t["fall_centroid_bias_median"] for t in timing_rows])),
        "n_peak_samples": int(np.sum([t["n_peak_samples"] for t in timing_rows])),
    }
    print(f"中位 NSE {summary['nse_median']:.3f}，持续性 {summary['persistence_nse_median']:.3f}，"
          f"胜出 {summary['nse_wins']}/{summary['n_sites']}")
    print("分预见期中位 NSE：" + "  ".join(
        f"{k} {v:.3f}" for k, v in summary["nse_by_lead"].items()))
    print(f"峰现时间偏差中位 {summary['peak_time_bias_median']:.2f} 小时"
          f"（涨水段 {summary['rise_centroid_bias_median']:.2f}，"
          f"退水段 {summary['fall_centroid_bias_median']:.2f}），"
          f"统计了 {summary['n_peak_samples']} 个有涨水的起报")
    return summary


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride 必须至少为 1")
    run_dir, checkpoint, manifest_path, run_kind, manifest = resolve_paths(args)
    if run_kind == "supervised":
        assert manifest is not None
        cfg = supervised_config(manifest)
        model_cfg = cfg["model"]
        default_roll = int(manifest.get("prediction_steps") or
                           model_cfg.get("rollout", 0) or model_cfg["horizon"])
        lookback = int(model_cfg["lookback"])
    else:
        cfg = load_config(ROOT / args.config)
        model_cfg = cfg["model"]
        default_roll = 12 if run_kind == "legacy" else int(
            manifest.get("config", {}).get("rollout", 24))
        lookback = LOOKBACK
    roll = args.roll if args.roll is not None else default_roll
    if roll < 1:
        raise ValueError("--roll 必须至少为 1")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))

    contract = validate_data_contract(cfg, ROOT)
    ids, names, areas, times, area_rain, flow = load_inputs(cfg, contract)
    mask = contract.mask1km
    manifest_data_signature = None
    statics_t = torch.from_numpy(build_statics(cfg, mask)).to(device)

    if run_kind == "legacy":
        train_manifest_path = run_dir / "training_manifest.json"
        if not train_manifest_path.is_file():
            raise FileNotFoundError(f"推理要求训练清单，但找不到：{train_manifest_path}")
        with open(train_manifest_path, encoding="utf-8") as f:
            train_manifest = json.load(f)
        manifest_data_signature = train_manifest.get("data_signature")
        if manifest_data_signature != contract.signature:
            raise ValueError("当前数据签名与训练清单不一致，拒绝推理")
        pred_path = run_dir / "predictions.npz"
        meta = archived_metadata(pred_path)
        if tuple(ids) != meta["ids"] or not times.equals(meta["times"]):
            raise ValueError("当前数据与模型预测存档的站点或时间轴不一致")
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        # 分位数模型有 3 个输出头；Huber/MSE 模型只有 1 个。按检查点构造，
        # 避免评估脚本把单头权重硬塞进三头结构。
        head_w = state.get("experts.0.head.2.weight")
        if head_w is None:
            raise ValueError("检查点缺少 MoE 输出头权重")
        n_quant = int(head_w.shape[0])
        model = MoENet(True, False, 96, 48, 1, n_quant=n_quant,
                       masked_pool=False, delta_cap=0.0, areas=areas).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        actor = None
        print(f"已加载旧模型 {args.run or 'moe'}，归一化来自 {pred_path}")
    elif run_kind == "supervised":
        assert manifest is not None
        manifest_data_signature = manifest.get("data_signature")
        if manifest_data_signature != contract.signature:
            raise ValueError("当前数据签名与普通监督训练清单不一致，拒绝推理")
        pred_path = run_dir / "predictions.npz"
        meta = archived_metadata(pred_path)
        if tuple(ids) != meta["ids"] or not times.equals(meta["times"]):
            raise ValueError("当前数据与模型预测存档的站点或时间轴不一致")
        require_output_mode(meta, model_cfg)
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        model = build_supervised_model(model_cfg, areas, len(ids)).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        actor = None
        print(f"已加载普通监督模型 {checkpoint}，归一化来自 {pred_path}")
    else:
        if manifest_path is None or manifest is None:
            raise FileNotFoundError(f"找不到 S.13 清单：{manifest_path}")
        # resolve_paths 已按格式辨认；这里继续复用 S.13 的字段级严格校验。
        manifest = read_manifest(manifest_path)
        manifest_data_signature = manifest.get("config", {}).get("data_signature")
        if not manifest_data_signature:
            raise ValueError("S.13/S.16 清单没有数据签名，拒绝推理")
        if manifest_data_signature != contract.signature:
            raise ValueError("当前数据签名与训练清单不一致，拒绝推理")
        source_doc = manifest.get("source_model", manifest.get("source_s10"))
        new_manifest = "source_model" in manifest
        if source_doc is None:
            raise ValueError("S.13/S.16 清单缺少源模型信息")
        source_run = Path(source_doc["run_dir"])
        if not source_run.is_absolute():
            source_run = ROOT / source_run
        source_pred = Path(source_doc["predictions"])
        if not source_pred.is_absolute():
            source_pred = ROOT / source_pred
        meta = archived_metadata(source_pred)
        source_model, _, source_info = load_model_archive(source_run, device="cpu")
        actor = S13ActorCritic(source_model, base_output_mode=meta["output_mode"]).to(device)
        load_s13_checkpoint(checkpoint, actor, map_location=device,
                            expected_source=source_info if new_manifest else None)
        actor.eval()
        model = None
        if tuple(ids) != meta["ids"] or not times.equals(meta["times"]):
            raise ValueError("当前数据与源模型存档的站点或时间轴不一致")
        print(f"已加载 S.13 {checkpoint}，归一化来自 {source_pred}")

    flow_pos = np.maximum(flow, 0.0)
    flow_n = (flow_pos - meta["q_mean"][:, None]) / meta["q_std"][:, None]
    if run_kind == "legacy":
        origins = legacy_split_origins(times, args.split, roll, args.stride)
    elif run_kind == "supervised":
        split_doc = manifest.get("splits", {}).get(args.split, {})
        if "target_start" not in split_doc or "target_end" not in split_doc:
            raise ValueError(f"普通监督训练清单缺少 {args.split} 的物理时间边界")
        origins = split_origins(
            times, pd.Timestamp(split_doc["target_start"]),
            pd.Timestamp(split_doc["target_end"]) + pd.Timedelta(hours=1),
            lookback, roll, args.stride)
    else:
        begin = pd.Timestamp("2023-01-01" if args.split == "val" else "2024-01-01")
        end = pd.Timestamp("2024-01-01" if args.split == "val" else "2025-01-01")
        origins = split_origins(times, begin, end, lookback, roll, args.stride)
    samples = valid_samples(origins, flow_n, area_rain, lookback, roll)
    if not len(samples):
        raise RuntimeError(f"{args.split} 段没有满足输入完整条件的起报点")
    print(f"{args.split} 段起报 {len(samples)} 条，间隔 {args.stride} 小时，滚动 {roll} 小时")

    # 推理仍需完整原始降雨时间轴；不做任何统计拟合。
    import xarray as xr
    da = xr.open_dataset(ROOT / cfg["paths"]["rain_nc"])["rain"]
    try:
        rain_grid = da.values
    finally:
        da.close()
    # 延长后的网格是 20 GB，必须逐步原地处理：原来 astype / nan_to_num / cumsum
    # 串成一条链，三个中间结果同时驻留会到 80 GB，把内存撑爆。
    if rain_grid.dtype != np.float32:
        rain_grid = rain_grid.astype(np.float32)
    np.nan_to_num(rain_grid, copy=False, nan=0.0)
    cs_np = np.cumsum(rain_grid, axis=0, dtype=np.float32)
    del rain_grid
    cs_t = torch.from_numpy(cs_np).to(device)
    del cs_np
    flow_t = torch.from_numpy(np.nan_to_num(flow_n, nan=0.0).astype(np.float32)).to(device)

    use_future = bool(model_cfg.get("use_future_rain", True))
    by_t0: Dict[int, List[int]] = {}
    for i, t0_ in samples:
        by_t0.setdefault(int(t0_), []).append(int(i))
    obs_l, sim_l, site_l, t0_l = [], [], [], []
    with torch.no_grad():
        for t0_, sites in sorted(by_t0.items()):
            site_t = torch.as_tensor(sites, dtype=torch.long, device=device)
            cur = flow_t[site_t, t0_:t0_ + lookback].clone()
            levels = []
            for k in range(roll):
                now = torch.full_like(site_t, t0_ + lookback - 1 + k)
                x, hist, fut = build_step_inputs(
                    cs_t, statics_t, site_t, now, cur, use_future,
                    bool(actor.masked_pool) if actor else model_masked_pool(model))
                if actor is not None:
                    out = actor(x, hist, fut, site_t)[0]
                    nxt = cur[:, -1] + out          # S.13 的动作始终是增量
                else:
                    out = model(x, hist, fut, site_t)[:, 0, model.mid]
                    # level：模型输出就是流量数值本身，绝不能再加一次当前流量
                    nxt = out if meta["output_mode"] == "level" else cur[:, -1] + out
                levels.append(nxt)
                cur = torch.cat([cur[:, 1:], nxt[:, None]], dim=1)
            sim_l.append(torch.stack(levels, dim=1).cpu().numpy())
            ii = np.asarray(sites, dtype=np.int64)
            obs_l.append(flow_n[ii, t0_ + lookback:t0_ + lookback + roll])
            site_l.append(ii)
            t0_l.append(np.full(len(ii), t0_, dtype=np.int64))
    obs_n = np.concatenate(obs_l).astype(np.float32)
    sim_n = np.concatenate(sim_l).astype(np.float32)
    site = np.concatenate(site_l)
    t0 = np.concatenate(t0_l)
    full = np.isfinite(obs_n).all(axis=1)

    if args.output:
        output = Path(args.output)
    elif run_kind == "legacy":
        output = run_dir / "predictions_dense.npz"
    else:
        output = run_dir / f"predictions_dense_{args.split}.npz"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, obs=obs_n, sim=sim_n, site=site, t0=t0,
                        times=np.array([str(v) for v in times]), ids=np.array(ids),
                        names=np.array(names), areas=areas, q_mean=meta["q_mean"],
                        q_std=meta["q_std"], transform=np.array(meta["transform"]),
                        lams=meta["lams"], full=full, R=np.array(roll),
                        output_mode=np.array("level"),
                        source_output_mode=np.array(meta["output_mode"]),
                        action_mode=np.array("delta" if actor is not None else "not_applicable"),
                        stride=np.array(args.stride), split=np.array(args.split))
    summary = robust_metrics(obs_n, sim_n, site, t0, flow_n, meta, lookback, roll)
    summary.update({"split": args.split, "roll": roll, "stride": args.stride,
                    "data_signature": contract.signature,
                    "checkpoint": str(checkpoint.resolve()), "output": str(output.resolve())})
    summary_path = output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"已写 {output}")
    print(f"已写 {summary_path}")


if __name__ == "__main__":
    main()
