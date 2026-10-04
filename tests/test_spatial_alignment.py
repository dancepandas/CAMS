# -*- coding: utf-8 -*-
"""空间坐标、文件契约与原子写盘的快速测试。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:  # 轻量环境仍可运行不依赖 NetCDF 的测试
    xr = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from spatial_grid import (atomic_savez, grid_signature, mrms_coordinates,
                          mrms_window, validate_regular_coordinate)

if xr is not None:
    try:
        from fetch_rain import (_compare_migrated_netcdf, _load_checkpoint,
                                _raise_retry_failures, _validate_complete_hours)
    except ImportError:
        _compare_migrated_netcdf = None
else:
    _compare_migrated_netcdf = None


class MrmsGridTests(unittest.TestCase):
    def test_coordinates_come_from_global_indices_and_match_south_north_data(self):
        rows, cols = mrms_window(35.065, 36.335, -83.485, -82.215)
        lat, lon = mrms_coordinates(rows, cols)
        self.assertEqual((rows.start, rows.stop), (1866, 1994))
        self.assertEqual((cols.start, cols.stop), (4651, 4779))
        self.assertEqual((len(lat), len(lon)), (128, 128))
        self.assertAlmostEqual(lat[0], 35.065)
        self.assertAlmostEqual(lat[-1], 36.335)
        self.assertAlmostEqual(lon[0], -83.485)
        self.assertAlmostEqual(lon[-1], -82.215)
        self.assertTrue(np.all(np.diff(lat) > 0))

    def test_non_grid_aligned_bounds_resolve_to_real_mrms_centers(self):
        rows, cols = mrms_window(35.064, 36.334, -83.484, -82.214)
        lat, lon = mrms_coordinates(rows, cols)
        self.assertAlmostEqual(lat[0], 35.065)
        self.assertAlmostEqual(lon[0], -83.485)


class ContractTests(unittest.TestCase):
    def test_signature_changes_with_coordinate_or_crs(self):
        lat = np.array([1.0, 2.0])
        lon = np.array([3.0, 4.0])
        sig = grid_signature(lat, lon)
        self.assertTrue(sig.startswith("sha256:"))
        self.assertNotEqual(sig, grid_signature(lat + 0.01, lon))
        self.assertNotEqual(sig, grid_signature(lat, lon, "EPSG:3857"))

    def test_coordinates_must_be_regular_and_increasing(self):
        with self.assertRaisesRegex(ValueError, "严格递增"):
            validate_regular_coordinate([2.0, 1.0], "lat")
        with self.assertRaisesRegex(ValueError, "等间隔"):
            validate_regular_coordinate([1.0, 2.0, 3.1], "lat")

class MigrationComparisonTests(unittest.TestCase):
    @unittest.skipIf(_compare_migrated_netcdf is None, "缺少 NetCDF/eccodes 测试依赖")
    def test_detects_changed_middle_timestep(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = Path(tmp) / "old.nc"
            new_path = Path(tmp) / "new.nc"
            times = np.arange("2020-01-01T00", "2020-01-01T05",
                              dtype="datetime64[h]")
            lat = np.array([35.0, 35.01])
            lon = np.array([-83.0, -82.99])
            rain = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
            rain[1, 0, 0] = np.nan
            product = np.array([0, 0, 1, 1, 1], dtype=np.int8)
            ds = xr.Dataset(
                {"rain": (("time", "lat", "lon"), rain)},
                coords={"time": times, "lat": lat, "lon": lon,
                        "product": ("time", product)})
            ds.to_netcdf(old_path)
            changed = rain.copy()
            changed[2, 1, 1] += 1.0
            ds.assign(rain=(("time", "lat", "lon"), changed)).to_netcdf(new_path)
            with self.assertRaisesRegex(ValueError, "rain 数值改变"):
                _compare_migrated_netcdf(old_path, new_path, chunk_hours=2)

    @unittest.skipIf(_compare_migrated_netcdf is None, "缺少 NetCDF/eccodes 测试依赖")
    def test_accepts_exact_values_with_only_coordinate_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = Path(tmp) / "old.nc"
            new_path = Path(tmp) / "new.nc"
            times = np.arange("2020-01-01T00", "2020-01-01T03",
                              dtype="datetime64[h]")
            rain = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
            rain[1, 0, 0] = np.nan
            ds = xr.Dataset(
                {"rain": (("time", "lat", "lon"), rain)},
                coords={"time": times, "lat": [35.0, 35.01],
                        "lon": [-83.0, -82.99],
                        "product": ("time", np.array([0, 1, 1], dtype=np.int8))})
            ds.to_netcdf(old_path)
            ds.assign_coords(lat=[35.005, 35.015]).to_netcdf(new_path)
            _compare_migrated_netcdf(old_path, new_path, chunk_hours=1)


class MrmsCheckpointTests(unittest.TestCase):
    @unittest.skipIf(_compare_migrated_netcdf is None, "缺少 MRMS 测试依赖")
    def test_checkpoint_requires_complete_matching_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rain_path = root / "partial.npy"
            product_path = root / "product.npy"
            metadata_path = root / "metadata.json"
            expected = {"format": "cams-mrms-checkpoint-v1", "hours": 2}
            shape = (2, 2, 2)
            defaults = np.array(["a", "a"])

            np.save(rain_path, np.zeros(shape, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "不完整"):
                _load_checkpoint(rain_path, product_path, metadata_path,
                                 expected, shape, defaults)

            np.save(product_path, defaults)
            metadata_path.write_text(json.dumps({**expected, "hours": 3}),
                                     encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "参数"):
                _load_checkpoint(rain_path, product_path, metadata_path,
                                 expected, shape, defaults)

            metadata_path.write_text(json.dumps(expected), encoding="utf-8")
            rain, products = _load_checkpoint(
                rain_path, product_path, metadata_path, expected, shape, defaults)
            self.assertEqual(rain.dtype, np.float32)
            np.testing.assert_array_equal(products, defaults)

    @unittest.skipIf(_compare_migrated_netcdf is None, "缺少 MRMS 测试依赖")
    def test_failed_or_empty_hour_stops_before_publish(self):
        with self.assertRaisesRegex(RuntimeError, "正式文件未发布"):
            _raise_retry_failures([("2024-12-31 01:00", "HTTP 404")])
        values = np.ones((2, 2, 2), dtype=np.float32)
        values[1] = np.nan
        times = pd.date_range("2024-12-31", periods=2, freq="h")
        with self.assertRaisesRegex(RuntimeError, "整小时全部缺测"):
            _validate_complete_hours(values, times)


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_savez_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.npz"
            atomic_savez(path, value=np.array([1, 2, 3]))
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["value"], [1, 2, 3])
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_atomic_savez_preserves_old_file_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.npz"
            atomic_savez(path, value=np.array([1]))
            real_replace = os.replace

            def fail_only_publish(src, dst):
                if Path(dst) == path:
                    raise OSError("simulated")
                return real_replace(src, dst)

            with mock.patch("spatial_grid.os.replace", side_effect=fail_only_publish):
                with self.assertRaises(OSError):
                    atomic_savez(path, value=np.array([2]))
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["value"], [1])
            self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
