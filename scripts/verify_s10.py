# -*- coding: utf-8 -*-
"""独立验证 S.10 MoE 模型预报指标，不依赖 train.py 的评估代码。"""
import json
import numpy as np
import pandas as pd

npz = np.load(r"D:\chengs\CAMS\runs\site_model_moe\predictions.npz", allow_pickle=True)
obs, sim = npz["obs"], npz["sim"]            # (4819,24) 标准化空间
site, t0 = npz["site"], npz["t0"]
times = pd.to_datetime(npz["times"])         # 小时轴
q_mean, q_std = npz["q_mean"], npz["q_std"]
ids, names, areas = npz["ids"], npz["names"], npz["areas"]
print("transform =", npz["transform"], " lams =", npz["lams"])
print(f"obs {obs.shape}  sites {len(ids)}  times {times[0]} ~ {times[-1]}")
print("q_mean/q_std 唯一值:", np.unique(q_mean.round(6)), np.unique(q_std.round(6)))

LB = 72  # lookback，train 配置


def nse(o, s):
    o = np.asarray(o, float); s = np.asarray(s, float)
    return float(1 - ((o - s) ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-9))


def inv(z, i):
    return np.maximum(np.asarray(z, float) * q_std[i] + q_mean[i], 0.0)


# ---------- 1) 读原始 CSV，重建各站小时流量（对齐 times 轴） ----------
raw = {}
for sid in ids:
    df = pd.read_csv(rf"D:\chengs\CAMS\data\sites\{sid}.csv",
                     parse_dates=["datetime"])
    idx = df["datetime"]
    if getattr(idx.dt, "tz", None) is not None:
        idx = idx.dt.tz_localize(None)
    s = pd.Series(df["flow_m3s"].values, index=idx).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    raw[sid] = s.reindex(times)  # NaN=缺测
print("CSV 重采样完成")

# ---------- 2) 时轴验证 ----------
# 假设: obs[样本,k] == ( flow_raw[site, times[t0+72+k]] - q_mean ) / q_std
for shift, label in [(72, "t0+72+k (主张)"), (71, "t0+71+k (错位-1)"),
                     (73, "t0+73+k (错位+1)")]:
    errs, n_tot, n_nan = [], 0, 0
    for i, sid in enumerate(ids):
        mk = site == i
        tt = t0[mk][:, None] + shift + np.arange(24)[None, :]   # (n,24)
        q = raw[sid].values[tt]                                  # m³/s
        zcsv = (q - q_mean[i]) / q_std[i]
        m = np.isfinite(zcsv)
        n_nan += (~m).sum(); n_tot += m.size
        errs.append(np.abs(zcsv[m] - obs[mk][m]))
    e = np.concatenate(errs)
    full_scale = 8.0  # z 空间大致量程（全局标准化后 obs 约 ∈[-0.6,8]）
    p05 = float((e < 0.005 * full_scale).mean())
    p_exact = float((e < 1e-6).mean())
    p_1pct = float((e < 0.01 * full_scale).mean())
    print(f"[对齐] {label}: 有效点 {e.size}/{n_tot} (CSV缺测 {n_nan}) | "
          f"误差中位 {np.median(e):.2e} 最大 {e.max():.3e} | "
          f"<0.5%量程 {p05:.4f}  <1%量程 {p_1pct:.4f}  <1e-6 {p_exact:.4f}")

# ---------- 3) 逆变换回 m³/s ----------
Qobs = np.empty_like(obs); Qsim = np.empty_like(sim)
for i in range(len(ids)):
    mk = site == i
    Qobs[mk] = inv(obs[mk], i); Qsim[mk] = inv(sim[mk], i)

# ---------- 4) 分预见期 NSE（逐站 -> 15 站中位） ----------
leads = [1, 3, 6, 12, 24]
lead_nse = {L: [] for L in leads}
per_site_nse, per_site_persist = [], []
for i in range(len(ids)):
    mk = site == i
    oo, ss = Qobs[mk], Qsim[mk]
    per_site_nse.append(nse(oo.ravel(), ss.ravel()))
    per = np.repeat(oo[:, :1], 24, axis=1)
    per_site_persist.append(nse(oo.ravel(), per.ravel()))
    for L in leads:
        lead_nse[L].append(nse(oo[:, L - 1], ss[:, L - 1]))

print("\n[分预见期中位 NSE]  独立复算 vs 日志")
log_lead = {"1h": 0.990, "3h": 0.905, "6h": 0.877, "12h": 0.625, "24h": 0.249}
for L in leads:
    v = float(np.median(lead_nse[L])); lg = log_lead[f"{L}h"]
    flag = "  <<< 差异>0.01" if abs(v - lg) > 0.01 else ""
    print(f"  +{L:>2d}h  复算 {v:.4f}   日志 {lg:.4f}   差 {v - lg:+.4f}{flag}")

# ---------- 5) 2024 全年逐站 NSE / 中位 / 持续 / 大洪水 ----------
print("\n[2024 全年逐站 NSE]  独立复算 vs 日志")
log_site = {"03455000": 0.336, "03454500": 0.555, "03453500": 0.622,
            "03451500": 0.718, "03447687": 0.893, "03443000": 0.802,
            "03453000": 0.362, "03451000": 0.697, "0344878100": 0.794,
            "03439000": -0.960, "03446000": 0.256, "0344632850": 0.564,
            "03441000": 0.463, "0344894205": 0.698, "03450000": 0.140}
log_persist = {"03455000": 0.865, "03454500": 0.702, "03453500": 0.603,
               "03451500": 0.713, "03447687": 0.804, "03443000": 0.804,
               "03453000": 0.127, "03451000": 0.507, "0344878100": 0.787,
               "03439000": 0.487, "03446000": 0.297, "0344632850": 0.357,
               "03441000": 0.619, "0344894205": 0.178, "03450000": 0.347}
bad = 0
for i, sid in enumerate(ids):
    a, b = per_site_nse[i], per_site_persist[i]
    la, lb = log_site[sid], log_persist[sid]
    f1 = " <<<" if abs(a - la) > 0.01 else ""
    f2 = " <<<" if abs(b - lb) > 0.01 else ""
    bad += abs(a - la) > 0.01
    print(f"  {sid:11s} {names[i]:8s} NSE 复算 {a:7.4f} 日志 {la:7.3f} ({a-la:+.4f}{f1})"
          f"  持续 复算 {b:7.4f} 日志 {lb:7.3f} ({b-lb:+.4f}{f2})")
med = float(np.median(per_site_nse)); medp = float(np.median(per_site_persist))
print(f"  中位 NSE 复算 {med:.4f}  日志/summary 0.5642776  差 {med-0.5642776:+.4f}"
      f"{'  <<< 差异>0.01' if abs(med-0.5642776)>0.01 else ''}")
print(f"  中位持续 复算 {medp:.4f}  日志/summary 0.6031442  差 {medp-0.6031442:+.4f}"
      f"{'  <<<' if abs(medp-0.6031442)>0.01 else ''}")
print(f"  NSE 胜出站点数 复算 {int(np.sum(np.array(per_site_nse)>np.array(per_site_persist)))}/15  日志 8/15")

# 大洪水 NSE：每站取 obs 24 步峰值的 top10% 样本，合并全部 24 步
bo, bs = [], []
for i in range(len(ids)):
    mk = site == i
    op = Qobs[mk].max(axis=1)
    big = op >= np.quantile(op, 0.9)
    if big.sum():
        bo.append(Qobs[mk][big].ravel()); bs.append(Qsim[mk][big].ravel())
nse_peak = nse(np.concatenate(bo), np.concatenate(bs))
print(f"\n[大洪水 NSE] 复算 {nse_peak:.4f}  日志/summary 0.5968354  "
      f"差 {nse_peak-0.5968354:+.4f}{'  <<< 差异>0.01' if abs(nse_peak-0.5968354)>0.01 else ''}")

# ---------- 6) 剔除 Helene 窗口 (2024-09-25 ~ 2024-10-01) ----------
h_lo = pd.Timestamp("2024-09-25"); h_hi = pd.Timestamp("2024-10-01 23:00")
tgt = times.values[t0[:, None] + LB + np.arange(24)[None, :]]   # 每个样本的24个目标时刻
in_helene = ((tgt >= np.datetime64(h_lo)) & (tgt <= np.datetime64(h_hi))).any(axis=1)
print(f"\n[Helene 窗口] 涉及样本 {in_helene.sum()}/{len(in_helene)}")
nse_h = []
for i in range(len(ids)):
    mk = (site == i) & (~in_helene)
    nse_h.append(nse(Qobs[mk].ravel(), Qsim[mk].ravel()))
print(f"  剔除 Helene 后中位 NSE = {np.median(nse_h):.4f}  (日志无此项，供参考)")
nse_hel = []
for i in range(len(ids)):
    mk = (site == i) & in_helene
    nse_hel.append(nse(Qobs[mk].ravel(), Qsim[mk].ravel()) if mk.sum() > 0 else np.nan)
print(f"  仅 Helene 窗内逐站中位 NSE = {np.nanmedian(nse_hel):.4f}")
print(f"  仅 Helene 窗内合并 NSE = {nse(Qobs[in_helene].ravel(), Qsim[in_helene].ravel()):.4f}")

# ---------- 7) 事件窗专项：每站 2024 最大值时刻 ± 窗 ----------
print("\n[事件窗 NSE] 窗口=[峰-144h, 峰+96h]，仅窗内有预报点的样本")
print(f"{'站号':11s} {'名称':8s} {'峰值时刻':17s} {'峰流量':>9s} {'样本数':>5s}"
      f" {'NSE@1h':>8s} {'NSE@6h':>8s} {'NSE@24h':>8s}")
for i, sid in enumerate(ids):
    s = raw[sid]
    s2024 = s[(s.index >= "2024-01-01") & (s.index <= "2024-12-31 23:00")]
    peak_t = s2024.idxmax(); peak_q = s2024.max()
    w_lo, w_hi = peak_t - pd.Timedelta(hours=144), peak_t + pd.Timedelta(hours=96)
    inw = (tgt >= np.datetime64(w_lo)) & (tgt <= np.datetime64(w_hi))
    mk = (site == i)
    ns = int((mk & inw.any(axis=1)).sum())
    row = f"  {sid:11s} {names[i]:8s} {str(peak_t):17s} {peak_q:9.1f} {ns:5d}"
    for L in (1, 6, 24):
        m2 = mk & inw[:, L - 1]
        row += f" {nse(Qobs[m2, L-1], Qsim[m2, L-1]):8.4f}" if m2.sum() > 2 else f" {'--':>8s}"
    print(row)
print("\n完成")
