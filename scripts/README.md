# scripts/ — 数据处理与模型代码

本目录是 French Broad River 流域 15 个 USGS 断面逐小时流量预报项目的全部
代码。所有脚本直接用项目根目录为工作目录运行，例如：

```bash
python scripts/fetch_rain.py --config configs/pipeline.yaml
```

约定（所有脚本共同遵守）：

- 注释和文档用中文；汇报术语翻译成人话。
- 大产物至少采用“同目录临时文件 → 原子替换”；关键降雨和面雨量产物还会在
  发布前或发布后重开校验。各脚本的具体保证见模块文档。
- 网格真源在 `spatial_grid.py`：MRMS 全球网格南北 54.995°、东西 -129.995°、
  步长 0.01°；项目裁剪网格 128×128（lat 35.065–36.335，lon -83.485–-82.215，
  EPSG:4326，纬度严格递增、第 0 行在最南）。
- 训练评估边界：训练目标严格早于 2023-01-01，2023 只用于早停，
  2024 只做最终评估。

## 数据获取（fetch_*）

| 文件 | 作用 |
|---|---|
| `fetch_rain.py` | 下载 MRMS 逐小时降水（2015-06 起，爱荷华州立镜像），裁剪成 NetCDF；支持断点续传、坐标迁移（`--migrate-coordinates`，必须显式给 `--backup`）、从已有正式文件安全扩展 |
| `fetch_rain_aorc.py` | 下载 1990–2015 年 AORC 逐小时降水（NOAA 公开 S3 Zarr），按重叠面积权重重采样到项目网格；任一块下载失败立即报错，绝不静默补零 |
| `fetch_dem.py` | 下载 30 米 SRTM 高程图幅（AWS elevation-tiles-prod） |
| `fetch_flow.py` | 下载 USGS 逐小时流量（参数码 00060，15 分钟值取整点，ft³/s→m³/s） |

## 空间数据链（terrain / catchments / area_rain）

| 文件 | 作用 |
|---|---|
| `spatial_grid.py` | 网格共同定义：MRMS 真源常量、网格指纹 SHA256、坐标校验、安全写盘 |
| `terrain.py` | 30 米 DEM 填洼、D8 流向、汇流面积、坡度、河网分级，裁剪到降水网格 |
| `catchments.py` | 按断面位置划上游汇水区掩膜，重投影到降水网格（纬度翻转后与 rain 一致） |
| `area_rain.py` | 按掩膜覆盖比例加权，把格点降雨折算成各断面逐小时面雨量 |

## 扩展降雨（S.18 长历史）

| 文件 | 作用 |
|---|---|
| `build_rain_extended.py` | 把 1990–2015 AORC 年度件与 2015-06 起 MRMS 拼成 `data/rain_1990_2024.nc`（约 20 GB，分块写入，接缝与 MRMS 尾段逐点校验） |

## 训练与推理

| 文件 | 作用 |
|---|---|
| `data_contract.py` | 训练/推理入口共用的数据门禁与训练清单（manifest v2：绑定降雨实值和流量 CSV 的 SHA256） |
| `train.py` | 通用监督训练入口：支持 Net / DLinear / MoE，以及预测增量（delta）/ 直接预测流量数值（level）两种语义；S.14/S.18 启动脚本选用 MoE、分位数损失、全局标准化和 level |
| `rollout_dense.py` | 密集滚动推理：2024 年逐小时起报、每次滚动 24 小时；区分 legacy / supervised / s13 三类运行 |
| `step13_rl.py` | S.13 闭环微调共用部件（模型、滚动、损失、PPO 数学、存档读写） |
| `train_step13_rl.py` | S.13 训练入口（24 步闭环监督对照或自写 PPO） |

## 绘图与对比

| 文件 | 作用 |
|---|---|
| `plot_basin.py` | 流域概览图：地形晕渲 + 水系 + 站点与汇水区 |
| `plot_forecast.py` | 单模型预报 vs 实测 + 降雨 |
| `plot_compare.py` | 多模型逐站对比（实测 / 预报 / 持续性 / 面雨量） |
| `plot_event.py` | 测试段最大洪水事件全过程对比（涨水-峰-退水） |
| `plot_step17.py` | S.14 / S.16 / S.17 三方对比 |
| `plot_step18.py` | S.14 修正版 / S.18 修正版公平对比（按物理起报时刻对齐，出汇总、覆盖、事件图） |
| `pipeline_final.sh` | **定稿流水线（一条命令）**：测试 → 数据合同 → 训底模 → BPTT 非对称罚微调 → 两年 dense 推理 → 对比出图；产物 `runs/site_model_final*` 与 `experiments/final_eval/`，已存在拒绝覆盖 |
| `plot_unroll.py` | S.20 闭环训练四版对比图；支持 `--models "名=目录,名=目录"` 画任意组合（定稿流水线用它画底模 vs 定稿）；`--split test`=2024 超极端年（图进 `experiments/step18_compare/`）、`--split val`=2023 正常年（图进 `experiments/step18_val_compare/`）；汇总六联 + 四大站/三小站事件图 |
| `compare_step18.py` | S.14 与 S.18 在 2024 测试段的逐项指标对比（按“站号 + 物理起报时间”对齐） |
| `verify_s10.py` | 独立复算 S.10 指标，不依赖 train.py 的评估代码 |
| `run_all.py` | 把完整管道按顺序串起来（各步骤产物落盘，可单独重跑） |
