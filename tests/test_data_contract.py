# -*- coding: utf-8 -*-
"""数据合同与训练清单的快速测试；不联网、不读取大文件。

覆盖三类门禁：
- 降雨/汇水区/面雨量/流量 CSV 之间的坐标、站序、时间轴一致性校验；
- 日历切分断言（训练必须早于 2023-01-01，2024 不得参与训练/标准化/早停）；
- 训练清单（manifest v2）的写入、重开校验与哈希绑定。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_contract import (RAIN_VALUES_FORMAT, assert_calendar_splits,
                           rain_values_sha256, validate_data_contract)
from spatial_grid import grid_signature


class DataContractTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict:
        (root / "sites").mkdir()
        pd.DataFrame({
            "site_id": ["001", "002"], "name": ["甲", "乙"],
            "lat": [1.0, 2.0], "lon": [3.0, 4.0], "area_km2": [10.0, 20.0],
        }).to_csv(root / "sites.csv", index=False)
        times = pd.date_range("2022-12-20", "2024-01-10 23:00", freq="h")
        lat = np.array([35.0, 35.01])
        lon = np.array([-83.0, -82.99, -82.98])
        crs = "EPSG:4326"
        signature = grid_signature(lat, lon, crs)
        mask = np.ones((2, 2, 3), dtype=np.float32)
        np.savez_compressed(root / "catch.npz", mask1km=mask,
                            site_ids=np.array(["001", "002"]),
                            names=np.array(["甲", "乙"]),
                            area_obs=np.array([10.0, 20.0]), lat=lat, lon=lon,
                            crs=np.array(crs), mask_dims=np.array(["site", "lat", "lon"]))
        rain_values = np.zeros((len(times), len(lat), len(lon)), dtype=np.float32)
        rain_digest = rain_values_sha256(rain_values, chunk_hours=17)
        np.savez_compressed(root / "area.npz", area_rain=np.ones((2, len(times))),
                            times=np.array([str(v) for v in times]),
                            site_ids=np.array(["001", "002"]),
                            names=np.array(["甲", "乙"]), areas=np.array([10.0, 20.0]),
                            lat=lat, lon=lon, crs=np.array(crs),
                            grid_signature=np.array(signature),
                            source_rain_values_format=np.array(RAIN_VALUES_FORMAT),
                            source_rain_values_sha256=np.array(rain_digest))
        ds = xr.Dataset(
            {"rain": (("time", "lat", "lon"), rain_values)},
            coords={"time": times, "lat": lat, "lon": lon}, attrs={"crs": crs})
        ds.to_netcdf(root / "rain.nc")
        for sid in ("001", "002"):
            pd.DataFrame({"flow_m3s": np.ones(len(times))}, index=times).to_csv(
                root / "sites" / f"{sid}.csv")
        return {"paths": {"sites_csv": "sites.csv", "sites": "sites",
                           "catchments": "catch.npz", "area_rain": "area.npz",
                           "rain_nc": "rain.nc"}}

    def test_valid_contract_and_string_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.make_fixture(Path(tmp))
            contract = validate_data_contract(cfg, tmp, hash_files=False)
            self.assertEqual(contract.ids, ("001", "002"))
            self.assertEqual(contract.mask1km.shape, (2, 2, 3))
            self.assertEqual(contract.flow.shape, contract.area_rain.shape)

    def test_rejects_station_reordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_fixture(root)
            with np.load(root / "area.npz", allow_pickle=True) as old:
                values = {key: old[key] for key in old.files}
            values["site_ids"] = np.array(["002", "001"])
            np.savez_compressed(root / "area.npz", **values)
            with self.assertRaisesRegex(ValueError, "站号或顺序"):
                validate_data_contract(cfg, root, hash_files=False)

    def test_rejects_one_hour_time_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_fixture(root)
            with np.load(root / "area.npz", allow_pickle=True) as old:
                values = {key: old[key] for key in old.files}
            times = pd.DatetimeIndex([str(v) for v in values["times"]])
            values["times"] = np.array([str(v) for v in times + pd.Timedelta(hours=1)])
            np.savez_compressed(root / "area.npz", **values)
            with self.assertRaisesRegex(ValueError, "逐点一致"):
                validate_data_contract(cfg, root, hash_files=False)

    def test_rejects_missing_spatial_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_fixture(root)
            np.savez_compressed(root / "catch.npz", mask1km=np.ones((2, 2, 3)),
                                site_ids=np.array(["001", "002"]))
            with self.assertRaisesRegex(ValueError, "空间契约字段"):
                validate_data_contract(cfg, root, hash_files=False)

    def test_rejects_rain_value_change_with_same_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_fixture(root)
            with xr.open_dataset(root / "rain.nc") as old:
                changed = old.load()
            changed["rain"].values[len(changed.time) // 2, 0, 0] = 1.0
            changed.to_netcdf(root / "rain.nc", mode="w")
            with self.assertRaisesRegex(ValueError, "不是由当前 rain 数值生成"):
                validate_data_contract(cfg, root, hash_files=False)

    def test_rain_digest_is_chunk_invariant_and_distinguishes_nan(self):
        values = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
        self.assertEqual(rain_values_sha256(values, 1),
                         rain_values_sha256(values, 4))
        with_nan = values.copy()
        with_nan[2, 0, 0] = np.nan
        with_zero = values.copy()
        with_zero[2, 0, 0] = 0.0
        self.assertNotEqual(rain_values_sha256(with_nan, 2),
                            rain_values_sha256(with_zero, 2))

    def test_flow_change_updates_data_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_fixture(root)
            before = validate_data_contract(cfg, root).signature
            path = root / "sites" / "001.csv"
            table = pd.read_csv(path, index_col=0)
            table.iloc[len(table) // 2, 0] = 5.0
            table.to_csv(path)
            after = validate_data_contract(cfg, root).signature
            self.assertNotEqual(before, after)

    def test_real_configs_use_calendar_boundaries(self):
        import yaml
        for name in ("pipeline.yaml", "pipeline_rain1990.yaml"):
            with open(ROOT / "configs" / name, encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["model"]["split_dates"],
                             ["2023-01-01", "2024-01-01"])
            self.assertEqual(str(config["period"]["end"]), "2024-12-31 23:00")

    def test_split_assertions_and_manifest(self):
        times = pd.date_range("2022-12-20", "2024-01-10 23:00", freq="h")
        idx23 = int(times.searchsorted(pd.Timestamp("2023-01-01")))
        idx24 = int(times.searchsorted(pd.Timestamp("2024-01-01")))
        samples = {
            "train": [(0, idx23 - 73)],
            "val": [(0, idx23 - 72), (0, idx24 - 73)],
            "test": [(0, idx24 - 72)],
        }
        info = assert_calendar_splits(samples, times, 72, 1)
        self.assertLess(pd.Timestamp(info["train"]["target_end"]), pd.Timestamp("2023-01-01"))
        bad = dict(samples, train=[(0, idx23 - 72)])
        with self.assertRaisesRegex(AssertionError, "训练目标"):
            assert_calendar_splits(bad, times, 72, 1)


if __name__ == "__main__":
    unittest.main()
