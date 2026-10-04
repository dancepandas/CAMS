# tests/ — 单元测试

不联网、不读取大文件（降雨 NetCDF、5.5 GB 数据一律不碰），用小型临时
fixture 验证数据链和模型的关键安全性质。

运行方式（在项目根目录）：

```bash
python -m unittest discover -s tests -v
# 或单个文件
python -m unittest tests.test_spatial_alignment -v
```

## 文件说明

| 文件 | 覆盖内容 |
|---|---|
| `test_spatial_alignment.py` | 空间坐标、文件契约与原子写盘：纬度必须严格递增且第 0 行在最南、半格偏移拒绝、网格指纹校验、人工掩膜重投影方向正确、纬度翻转/站序互换/CRS 或 shape 不符时下游立即报错 |
| `test_rain_extended.py` | AORC 下载与扩展降雨拼接：404/超时/短块/解压长度错误导致整年失败且不替换旧文件、合法 NaN 按剩余有效面积重新归一（不补零）、float32 与完整年份校验、原子发布、接缝规则 |
| `test_data_contract.py` | 数据合同与训练清单：rain/catchment/area-rain/流量 CSV 一致性、日历切分断言（训练 <2023-01-01，2024 不参与训练/标准化/早停）、manifest v2 写入与哈希绑定 |
| `test_rollout_dense.py` | 密集推理入口：manifest 识别（supervised/s13/v1 拒绝/双清单歧义）、从 effective_config 重建模型并加载权重、不同内部下标映射到相同物理起报时刻、output_mode 不一致拒绝 |
| `test_step13_rl.py` | S.13 共用数学与安全边界：PPO 数学、滚动语义、损失边界 |

## 背景

这批测试对应 2026-09 的“S.18 审计”修出的三个正确性问题：

1. 汇水区掩膜南北颠倒 + MRMS 坐标各偏半格；
2. AORC 下载块失败被静默写成零降雨；
3. 必须用修正后的共同数据重建 S.14/S.18 面雨量并公平重训。

每条修复都有对应测试防止回退。
