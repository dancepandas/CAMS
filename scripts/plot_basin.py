#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流域概览图：地形晕渲 + 水系 + 15 个断面位置与汇水区。

读 data/terrain30.npz（30m DEM/流向/上游面积/水系级别）、
data/catchments.npz（1km 汇水区掩膜）与 configs/sites_french_broad.csv，
出一张 basin_overview.png。

用法: python scripts/plot_basin.py [--out experiments]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 与 catchments.py 一致的降水网格仿射：左上 (lon0-step/2, lat1+step/2)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 35.065, 36.335, -83.485, -82.215
G, STEP = 128, 0.01


def hillshade(elev, azimuth=315, altitude=45):
    """简单晕渲：由坡度坡向加权光照强度。"""
    x, y = np.gradient(elev)
    slope = np.pi / 2 - np.arctan(np.hypot(x, y))
    aspect = np.arctan2(-x, y)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = (np.sin(alt) * np.sin(slope)
             + np.cos(alt) * np.cos(slope) * np.cos(az - np.pi - aspect))
    return (shade + 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    t = np.load(os.path.join(ROOT, "data", "terrain30.npz"), allow_pickle=True)
    elev, flwdir, uparea, order = t["elevtn"], t["flwdir"], t["uparea"], t["order"]
    lon0, lat0, lon1, lat1 = t["bounds"]
    nrow, ncol = elev.shape
    extent = [lon0, lon0 + ncol / 3600, lat0, lat0 + nrow / 3600]

    c = np.load(os.path.join(ROOT, "data", "catchments.npz"), allow_pickle=True)
    mask1km, site_ids, names = c["mask1km"], c["site_ids"], c["names"]

    sites = pd.read_csv(os.path.join(ROOT, "configs", "sites_french_broad.csv"),
                        dtype={"site_id": str})

    # 全流域 30m 掩膜：从各断面 outlet 在流向场上反溯汇水区再取并集，
    # 水系只画在流域内，避免田纳西河平原等流域外大河干扰
    import pyflwdir
    from rasterio.transform import Affine
    SEC = 3600.0
    tf30 = Affine(1 / SEC, 0, lon0, 0, -1 / SEC, lat0 + nrow / SEC)
    flw = pyflwdir.from_array(flwdir, ftype="d8", transform=tf30, latlon=True)
    outlets = []
    for _, row in sites.iterrows():
        cc = int(np.clip(round((row["lon"] - lon0) * SEC - 0.5), 0, ncol - 1))
        rr = int(np.clip(round((lat0 + nrow / SEC - row["lat"]) * SEC - 0.5),
                         0, nrow - 1))
        win = uparea[max(0, rr - 10):rr + 11, max(0, cc - 10):cc + 11]
        wr, wc = np.unravel_index(np.nanargmax(win), win.shape)
        outlets.append((max(0, rr - 10) + wr) * ncol + max(0, cc - 10) + wc)
    basins = flw.basins(idxs=np.array(outlets))
    basin_m = (basins > 0)

    fig, ax = plt.subplots(figsize=(12.5, 11))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06)

    # 地形晕渲（自定义陆表色标：低地绿 → 丘陵黄绿 → 山地棕 → 山顶白）
    hs = hillshade(elev)
    land = mcolors.LinearSegmentedColormap.from_list(
        "land", ["#7fae7f", "#b8c47a", "#c9a961", "#96693d", "#e9e4da"])
    norm = mcolors.Normalize(vmin=250, vmax=1800)
    rgb = land(norm(elev))[:, :, :3]
    for k in range(3):
        rgb[:, :, k] = rgb[:, :, k] * (0.6 + 0.4 * hs)
    rgb[~basin_m] = rgb[~basin_m] * 0.55 + 0.25   # 流域外压暗
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear")

    # 水系：流域内按上游面积分三级叠加；30m 细线直接画会亚像素消失，
    # 按 6×6 块取最大聚合后再画（线宽约 200m，视觉上成线）
    f = 6
    rr, cc = nrow // f * f, ncol // f * f
    up_d = uparea[:rr, :cc].reshape(rr // f, f, cc // f, f).max(axis=(1, 3))
    bs_d = basin_m[:rr, :cc].reshape(rr // f, f, cc // f, f).max(axis=(1, 3))
    ext_d = [lon0, lon0 + cc / 3600, lat0 + (nrow - rr) / 3600, lat0 + nrow / 3600]
    for thr, color, alpha in ((1000, "#08306b", 1.0),
                              (300, "#2171b5", 0.9),
                              (60, "#6baed6", 0.65)):
        st = np.where((up_d > thr) & (bs_d > 0), 1.0, np.nan)
        ax.imshow(np.ma.masked_invalid(st), extent=ext_d, origin="upper",
                  cmap=mcolors.ListedColormap([color]), alpha=alpha,
                  interpolation="nearest", aspect="auto")

    # 全流域外轮廓（粗）+ 各断面汇水区边界（细）
    gx = np.linspace(LON_MIN - STEP / 2, LON_MIN - STEP / 2 + G * STEP, G)
    gy = np.linspace(LAT_MAX + STEP / 2 - G * STEP, LAT_MAX + STEP / 2, G)
    union = mask1km.max(axis=0)
    ax.contour(gx, gy, union, levels=[0.5], colors=["k"], linewidths=1.8)
    for i in range(len(site_ids)):
        ax.contour(gx, gy, mask1km[i], levels=[0.5], colors=["white"],
                   linewidths=0.7, alpha=0.8)

    # 断面位置 + 名称标注
    for _, row in sites.iterrows():
        ax.plot(row["lon"], row["lat"], marker="v", ms=9, mfc="crimson",
                mec="white", mew=0.8, zorder=5)
        dx, dy = 0.012, 0.012
        if row["site_id"] == "03455000":
            dy = -0.028
        if row["site_id"] == "03439000":
            dx = -0.10
        if row["site_id"] == "0344894205":
            dx, dy = 0.012, -0.028
        if row["site_id"] == "03451000":
            dx, dy = 0.05, -0.01
        if row["site_id"] == "0344878100":
            dx, dy = -0.075, 0.028
        if row["site_id"] == "03453500":
            dx, dy = -0.06, 0.03
        ax.annotate(f"{row['name']}\n{row['area_km2']:.0f} km²",
                    (row["lon"], row["lat"]), xytext=(row["lon"] + dx, row["lat"] + dy),
                    fontsize=9, fontweight="bold", color="black", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75,
                              ec="none"))

    # 比例尺（约 50 km）
    km50 = 50 / 111.32 / np.cos(np.radians((LAT_MIN + LAT_MAX) / 2))
    sx, sy = LON_MIN + 0.06, LAT_MIN + 0.06
    ax.plot([sx, sx + km50], [sy, sy], color="black", lw=3)
    ax.text(sx + km50 / 2, sy + 0.015, "50 km", ha="center", fontsize=10)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("French Broad 流域：地形、水系与 15 个预报断面", fontsize=14)
    ax.grid(alpha=0.2)

    # 指北针
    ax.annotate("N", xy=(0.965, 0.90), xycoords="axes fraction", fontsize=13,
                fontweight="bold", ha="center")
    ax.annotate("", xy=(0.965, 0.965), xytext=(0.965, 0.885),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    out = os.path.join(ROOT, args.out, "basin_overview.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"已写 {out}")


if __name__ == "__main__":
    main()
