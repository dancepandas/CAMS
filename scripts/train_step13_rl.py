#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S.13 训练入口：24 步闭环监督对照或自写 PyTorch PPO。

训练和选模只看 2023 年及以前：训练目标早于 2023-01-01，验证目标只落在
2023 年。脚本不构造测试集，也不输出测试指标。

示例：
  python scripts/train_step13_rl.py --mode control --max-updates 200
  python scripts/train_step13_rl.py --mode ppo --max-updates 100
  python scripts/train_step13_rl.py --mode control --smoke --max-updates 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_contract import (DataContract, assert_calendar_splits,
                           target_time_range, validate_data_contract)
from step13_rl import (LOOKBACK, ROLLOUT, S13ActorCritic, TrajectoryBuffer,
                       corrected_split_origins, filter_site_origins,
                       flatten_rollout_for_buffer, load_model_archive,
                       load_s13_checkpoint, ppo_update, rollout_24,
                       save_checkpoint, site_activity_scale, trajectory_loss,
                       write_manifest)
from train import build_statics, load_config, make_inv, nse

TRAIN_STOP = pd.Timestamp("2023-01-01")
VALIDATION_STOP = TRAIN_STOP + pd.DateOffset(years=1)

DEFAULTS: Dict[str, Any] = {
    "lookback": LOOKBACK,
    "rollout": ROLLOUT,
    "train_stride": 24,
    "val_stride": 24,
    "batch_size": 8,             # 单次显存中的路线数
    "val_batch_size": 64,
    "collection_batch": 8,
    "trajectories_per_update": 64,  # 每次更新实际看 64 条路线（8×8 梯度累积）
    "ppo_epochs": 4,
    "ppo_batch_size": 256,
    "lr": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.001,
    "kl_coef": 0.02,
    "level_weight": 1.0,
    "delta_weight": 0.4,        # S.17 起：从 0.2 提高，更看重涨落节奏
    "peak_weight": 0.3,         # S.17 起：从 0.1 提高，洪峰报低要罚得动
    "peak_time_weight": 0.2,    # S.17 新增：峰现时间偏差（加权重心，枯水期自动失效）
    "huber_delta": 1.0,
    "scheme": "S.17",
    "validate_every": 5,
    "max_grad_norm": 1.0,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S.13 闭环微调（训练只到 2023 验证段）")
    ap.add_argument("--mode", choices=("control", "ppo"), default="control",
                    help="control=24步监督闭环；ppo=随机轨迹PPO")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="只取极少起报点检查流程")
    ap.add_argument("--run-dir", default=None, help="输出目录；默认 runs/site_model_s13_<mode>")
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--resume", nargs="?", const="last.pt", default=None,
                    help="续跑检查点；不写路径时用 run-dir/last.pt")
    ap.add_argument("--force", action="store_true",
                    help="允许在已有产物目录里从头覆盖；默认拒绝误覆盖")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前 PyTorch 看不到显卡")
    return torch.device(name)


def load_pretest_data(cfg: Mapping[str, Any], contract: DataContract,
                      norm_times: pd.DatetimeIndex, archive_ids: Sequence[str]
                      ) -> Tuple[List[str], List[str], np.ndarray, pd.DatetimeIndex,
                                 np.ndarray, np.ndarray, int]:
    """从源文件直接读到验证段末；训练进程从不加载后面的流量或面雨量。"""
    ids = list(contract.ids)
    names = list(contract.names)
    areas = contract.areas.copy()
    all_times = contract.times
    stop = int(all_times.searchsorted(VALIDATION_STOP))
    times = all_times[:stop]
    area_rain = np.asarray(contract.area_rain[:, :stop], dtype=np.float64)
    flow = np.full((len(ids), stop), np.nan, dtype=np.float64)
    for i, sid in enumerate(ids):
        q = pd.read_csv(ROOT / cfg["paths"]["sites"] / f"{sid}.csv",
                        index_col=0, parse_dates=True)["flow_m3s"]
        if getattr(q.index, "tz", None) is not None:
            q.index = q.index.tz_localize(None)
        flow[i] = q.reindex(times).to_numpy()
    if tuple(ids) != tuple(str(v) for v in archive_ids):
        raise ValueError("当前站点顺序与源模型存档不一致")
    if stop < 1 or not times.equals(norm_times[:stop]):
        raise ValueError("当前数据时间轴与源模型存档不一致")
    return ids, names, areas, times, area_rain, flow, stop


def load_spatial_prefix(cfg: Mapping[str, Any], contract: DataContract, stop: int,
                        device: torch.device, smoke: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """只读取验证段末以前的降雨格点；这里是全程序最大的内存块。"""
    import xarray as xr

    rain_path = ROOT / cfg["paths"]["rain_nc"]
    da = xr.open_dataset(rain_path)["rain"]
    try:
        rain = da.isel(time=slice(0, stop)).values.astype(np.float32)
    finally:
        da.close()
    cs = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0).cumsum(axis=0,
                                                                       dtype=np.float32)
    del rain
    mask = contract.mask1km
    statics = build_statics(cfg, mask)
    cs_t = torch.from_numpy(cs).to(device)
    st_t = torch.from_numpy(statics).to(device)
    return cs_t, st_t


def grouped_batches(samples: np.ndarray, batch_size: int, rng: np.random.Generator,
                    shuffle: bool = True) -> Iterable[np.ndarray]:
    order = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
    for start in range(0, len(order), batch_size):
        yield samples[order[start:start + batch_size]]


def validation(model: S13ActorCritic, samples: np.ndarray, flow_t: torch.Tensor,
               cs_t: torch.Tensor, statics_t: torch.Tensor, norm: Any,
               batch_size: int, scale_t: torch.Tensor
               ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    losses: List[float] = []
    obs_l, sim_l, site_l, t0_l = [], [], [], []
    with torch.no_grad():
        for part in grouped_batches(samples, batch_size, np.random.default_rng(0), False):
            site = torch.as_tensor(part[:, 0], dtype=torch.long, device=flow_t.device)
            t0 = torch.as_tensor(part[:, 1], dtype=torch.long, device=flow_t.device)
            out = rollout_24(model, cs_t, statics_t, flow_t, site, t0,
                             stochastic=False, target_flow=flow_t)
            loss, _ = trajectory_loss(out.levels, out.target, out.target_mask,
                                      level_weight=DEFAULTS["level_weight"],
                                      delta_weight=DEFAULTS["delta_weight"],
                                      peak_weight=DEFAULTS["peak_weight"],
                                      peak_time_weight=DEFAULTS["peak_time_weight"],
                                      site_scale=scale_t[site],
                                      huber_delta=DEFAULTS["huber_delta"])
            losses.append(float(loss.cpu()))
            obs_l.append(out.target.cpu().numpy())
            sim_l.append(out.levels.cpu().numpy())
            site_l.append(part[:, 0].copy())
            t0_l.append(part[:, 1].copy())
    obs = np.concatenate(obs_l)
    sim = np.concatenate(sim_l)
    site = np.concatenate(site_l)
    t0 = np.concatenate(t0_l)
    inv = make_inv(norm.transform, norm.q_mean, norm.q_std, norm.lams)
    per_site, mae = [], []
    lead_rows = []
    for i in range(len(norm.ids)):
        mk = site == i
        if not mk.any():
            continue
        oo, ss = inv(obs[mk], i), inv(sim[mk], i)
        good = np.isfinite(oo) & np.isfinite(ss)
        if good.sum() >= 4:
            per_site.append(nse(oo[good], ss[good]))
            mae.append(float(np.mean(np.abs(oo[good] - ss[good]))))
        row = []
        for lead in (1, 3, 6, 12, 24):
            g = np.isfinite(oo[:, lead - 1]) & np.isfinite(ss[:, lead - 1])
            row.append(nse(oo[g, lead - 1], ss[g, lead - 1]) if g.sum() >= 4 else np.nan)
        lead_rows.append(row)
    metrics = {
        "val_loss": float(np.mean(losses)),
        "val_nse_median": float(np.nanmedian(per_site)),
        "val_mae_median": float(np.nanmedian(mae)),
    }
    med = np.nanmedian(np.asarray(lead_rows), axis=0)
    metrics.update({f"val_nse_{lead}h": float(v)
                    for lead, v in zip((1, 3, 6, 12, 24), med)})
    arrays = {"obs": obs.astype(np.float32), "sim": sim.astype(np.float32),
              "site": site, "t0": t0}
    return metrics, arrays


def save_val_predictions(path: Path, arrays: Mapping[str, np.ndarray], norm: Any,
                         times: pd.DatetimeIndex) -> None:
    np.savez_compressed(path, **arrays,
                        times=np.array([str(v) for v in times]),
                        ids=np.array(norm.ids), names=np.array(norm.names), areas=norm.areas,
                        q_mean=norm.q_mean, q_std=norm.q_std,
                        transform=np.array(norm.transform), lams=norm.lams,
                        output_mode=np.array("level"), action_mode=np.array("delta"),
                        source_output_mode=np.array("level"),
                        split=np.array("val"), R=np.array(ROLLOUT))


def append_history(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("--max-updates 必须至少为 1")
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / f"site_model_step17_{args.mode}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if (run_dir / "manifest.json").is_file() and not args.resume and not args.force:
        raise FileExistsError(
            f"{run_dir} 已有训练产物；请用 --resume 续跑，或明确加 --force 从头覆盖")
    run_dir.mkdir(parents=True, exist_ok=True)

    source_model, norm, source = load_model_archive(
        ROOT / "runs" / "site_model_step14", device="cpu", expected_output_mode="level")
    cfg = load_config(ROOT / args.config)
    contract = validate_data_contract(cfg, ROOT)
    ids, names, areas, times, area_rain, flow, stop = load_pretest_data(
        cfg, contract, norm.times, norm.ids)
    if len(ids) != len(norm.ids) or not np.allclose(areas, norm.areas):
        raise ValueError("当前站点面积或数量与源模型存档不一致")

    # 流量归一化严格复用 S.14 存档；不在新切分上重估。
    flow_pos = np.maximum(flow, 0.0)
    flow_n = (flow_pos - norm.q_mean[:, None]) / norm.q_std[:, None]
    splits = corrected_split_origins(times, train_stride=DEFAULTS["train_stride"],
                                     val_stride=DEFAULTS["val_stride"],
                                     train_stop=str(TRAIN_STOP),
                                     val_stop=str(VALIDATION_STOP))
    train_samples = filter_site_origins(splits.train, flow_n, area_rain)
    val_samples = filter_site_origins(splits.val, flow_n, area_rain)
    # S.13 不读测试目标，但仍用完整时间元数据建立清单并硬断言边界。
    test_first = int(contract.times.searchsorted(VALIDATION_STOP)) - LOOKBACK
    test_last = len(contract.times) - LOOKBACK - ROLLOUT
    calendar_samples = {
        "train": train_samples,
        "val": val_samples,
        "test": np.array([[0, test_first], [0, test_last]], dtype=np.int64),
    }
    split_info = assert_calendar_splits(calendar_samples, contract.times, LOOKBACK, ROLLOUT)
    split_info["test"]["samples"] = "not loaded by this training process"
    if args.smoke:
        # 同时保留不同站，确保门控和站点索引也走到；不是缩短 24 步。
        train_samples = train_samples[:min(4, len(train_samples))]
        val_samples = val_samples[:min(4, len(val_samples))]
    if not len(train_samples) or not len(val_samples):
        raise RuntimeError("训练或验证样本为空")
    print(f"设备 {device}  训练起报 {len(train_samples)} 条  验证起报 {len(val_samples)} 条")
    print("边界已锁定：训练目标早于 2023，验证目标仅为 2023；没有构造测试集")
    print(f"载入降雨格点到验证段末，预计约 {stop * 128 * 128 * 4 / 1e9:.2f} GB")
    cs_t, statics_t = load_spatial_prefix(cfg, contract, stop, device, args.smoke)
    flow_t = torch.from_numpy(flow_n.astype(np.float32)).to(device)
    # 峰现时间项的门槛按站给尺度：全局归一化下固定阈值对大小站不公平。
    scale_t = torch.from_numpy(
        site_activity_scale(flow_n, times, str(TRAIN_STOP)).astype(np.float32)).to(device)

    model = S13ActorCritic(source_model, base_output_mode=source["output_mode"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=DEFAULTS["lr"])
    start_update = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = run_dir / resume_path
        saved = load_s13_checkpoint(resume_path, model, optimizer, map_location=device,
                                    expected_source=source)
        if saved.get("mode") != args.mode:
            raise ValueError(f"续跑模式不符：存档 {saved.get('mode')}，当前 {args.mode}")
        start_update = int(saved["update"])
        print(f"已续跑 {resume_path}，从第 {start_update + 1} 次更新开始")

    max_updates = args.max_updates or (1 if args.smoke else (200 if args.mode == "control" else 100))
    settings = dict(DEFAULTS, mode=args.mode, seed=args.seed, smoke=bool(args.smoke),
                    max_updates=max_updates, train_stop=str(TRAIN_STOP),
                    validation_stop=str(VALIDATION_STOP), normalization_source="S.14 archive",
                    normalization_end_exclusive=str(TRAIN_STOP),
                    data_signature=contract.signature, splits=split_info,
                    source_output_mode="level", action_mode="delta")
    artifacts = {"last_checkpoint": "last.pt", "best_checkpoint": "best.pt",
                 "history": "history.jsonl", "validation_predictions": "predictions_val.npz"}
    manifest_path = run_dir / "manifest.json"
    history_path = run_dir / "history.jsonl"
    if args.force and not args.resume and history_path.exists():
        history_path.unlink()
    best_score = -float("inf")
    best_info: Dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            old_manifest = json.load(f)
        old_signature = old_manifest.get("config", {}).get("data_signature")
        if old_signature != contract.signature:
            raise ValueError("当前数据签名与续跑清单不一致，拒绝续跑")
        best_score = float(old_manifest.get("best", {}).get("val_score", best_score))
        best_info = dict(old_manifest.get("best", {}))
    write_manifest(manifest_path, mode=args.mode, source=source, config=settings,
                   artifacts=artifacts, best=best_info)

    rng = np.random.default_rng(args.seed + start_update)
    ppo_generator = torch.Generator(device="cpu")
    ppo_generator.manual_seed(args.seed + start_update)
    train_cursor = 0
    train_order = rng.permutation(len(train_samples))
    for update in range(start_update + 1, max_updates + 1):
        if args.mode == "control":
            # 一次更新累计 64 条路线的梯度；每个小批仍为 8，避免 24 步展开撑爆显存。
            optimizer.zero_grad(set_to_none=True)
            comp_sum: Dict[str, float] = {}
            seen = 0
            remaining = DEFAULTS["trajectories_per_update"]
            while remaining > 0:
                if train_cursor >= len(train_order):
                    train_order = rng.permutation(len(train_samples))
                    train_cursor = 0
                take = min(DEFAULTS["batch_size"], remaining,
                           len(train_order) - train_cursor)
                sel = train_order[train_cursor:train_cursor + take]
                train_cursor += take
                remaining -= take
                part = train_samples[sel]
                site = torch.as_tensor(part[:, 0], dtype=torch.long, device=device)
                t0 = torch.as_tensor(part[:, 1], dtype=torch.long, device=device)
                model.train(True)
                try:
                    out = rollout_24(model, cs_t, statics_t, flow_t, site, t0,
                                     stochastic=False, target_flow=flow_t)
                    loss, components = trajectory_loss(
                        out.levels, out.target, out.target_mask,
                        level_weight=DEFAULTS["level_weight"],
                        delta_weight=DEFAULTS["delta_weight"],
                        peak_weight=DEFAULTS["peak_weight"],
                        peak_time_weight=DEFAULTS["peak_time_weight"],
                        site_scale=scale_t[site],
                        huber_delta=DEFAULTS["huber_delta"])
                    (loss * (take / DEFAULTS["trajectories_per_update"])).backward()
                except torch.cuda.OutOfMemoryError:
                    raise RuntimeError("显存不足：请减小脚本 DEFAULTS 里的 batch_size") from None
                for key, value in components.items():
                    comp_sum[key] = comp_sum.get(key, 0.0) + float(value.detach().cpu()) * take
                seen += take
            torch.nn.utils.clip_grad_norm_(trainable, DEFAULTS["max_grad_norm"])
            optimizer.step()
            train_metrics = {f"train_{k}": v / max(seen, 1)
                             for k, v in comp_sum.items()}
        else:
            n_take = min(DEFAULTS["trajectories_per_update"], len(train_samples))
            chosen = train_samples[rng.choice(len(train_samples), n_take, replace=False)]
            buffer = TrajectoryBuffer()
            model.eval()
            with torch.no_grad():
                for start in range(0, len(chosen), DEFAULTS["collection_batch"]):
                    part = chosen[start:start + DEFAULTS["collection_batch"]]
                    site = torch.as_tensor(part[:, 0], dtype=torch.long, device=device)
                    t0 = torch.as_tensor(part[:, 1], dtype=torch.long, device=device)
                    out = rollout_24(model, cs_t, statics_t, flow_t, site, t0,
                                     stochastic=True, target_flow=flow_t)
                    flatten_rollout_for_buffer(buffer, out, flow_t, site, t0,
                                               gamma=DEFAULTS["gamma"],
                                               gae_lambda=DEFAULTS["gae_lambda"],
                                               level_weight=DEFAULTS["level_weight"],
                                               delta_weight=DEFAULTS["delta_weight"],
                                               peak_weight=DEFAULTS["peak_weight"],
                                               peak_time_weight=DEFAULTS["peak_time_weight"],
                                               site_scale=scale_t[site],
                                               huber_delta=DEFAULTS["huber_delta"])
            train_metrics = {f"train_{k}": v for k, v in ppo_update(
                model, optimizer, buffer.as_batch(), cs_t, statics_t,
                epochs=DEFAULTS["ppo_epochs"], batch_size=DEFAULTS["ppo_batch_size"],
                max_grad_norm=DEFAULTS["max_grad_norm"], generator=ppo_generator,
                clip_ratio=DEFAULTS["clip_ratio"], value_coef=DEFAULTS["value_coef"],
                entropy_coef=DEFAULTS["entropy_coef"], kl_coef=DEFAULTS["kl_coef"]).items()}
            train_metrics["trajectory_steps"] = len(buffer)

        do_val = update % DEFAULTS["validate_every"] == 0 or update == max_updates
        metrics: Dict[str, Any] = {"update": update, "mode": args.mode, **train_metrics}
        if do_val:
            val_metrics, val_arrays = validation(model, val_samples, flow_t, cs_t,
                                                 statics_t, norm,
                                                 DEFAULTS["val_batch_size"], scale_t)
            val_score = (0.25 * val_metrics["val_nse_6h"]
                         + 0.25 * val_metrics["val_nse_12h"]
                         + 0.50 * val_metrics["val_nse_24h"])
            val_metrics["val_score"] = float(val_score)
            metrics.update(val_metrics)
            print(f"第 {update:4d} 次更新  训练损失 {train_metrics.get('train_total', float('nan')):.5f}  "
                  f"验证分 {val_score:.3f}  验证中位NSE {val_metrics['val_nse_median']:.3f}")
            if np.isfinite(val_score) and val_score > best_score:
                best_score = float(val_score)
                best_info = {"update": update, **val_metrics}
                save_checkpoint(run_dir / "best.pt", model, args.mode, update, optimizer,
                                val_metrics, settings, source)
                save_val_predictions(run_dir / "predictions_val.npz", val_arrays, norm,
                                     times)
        append_history(history_path, metrics)
        save_checkpoint(run_dir / "last.pt", model, args.mode, update, optimizer,
                        {k: v for k, v in metrics.items() if k.startswith("val_")}, settings,
                        source)
        write_manifest(manifest_path, mode=args.mode, source=source, config=settings,
                       artifacts=artifacts, best=best_info)

    print(f"完成。产物在 {run_dir}；这里只做了训练段和 2023 验证段。")


if __name__ == "__main__":
    main()
