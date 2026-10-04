# -*- coding: utf-8 -*-
"""AORC 下载和扩展降雨拼接的快速测试；不联网、不读取大文件。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import netCDF4
import numpy as np
import pandas as pd
import requests
import xarray as xr
from numcodecs import Zstd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_rain_extended as build
import fetch_rain_aorc as fetch


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class FetchTests(unittest.TestCase):
    def test_http_404_raises_immediately(self):
        reader = fetch.AorcReader(threads=1, attempts=3, timeout=0.01)
        self.addCleanup(reader.close)
        reader.session.get = mock.Mock(return_value=FakeResponse(404))
        with self.assertRaises(fetch.AorcDownloadError):
            reader._get("https://example.test/missing")
        self.assertEqual(reader.session.get.call_count, 1)

    def test_network_failure_raises_after_all_retries(self):
        reader = fetch.AorcReader(threads=1, attempts=2, timeout=0.01)
        self.addCleanup(reader.close)
        reader.session.get = mock.Mock(side_effect=requests.ConnectionError("offline"))
        with mock.patch.object(fetch._time, "sleep"):
            with self.assertRaises(fetch.AorcDownloadError):
                reader._get("https://example.test/fail")
        self.assertEqual(reader.session.get.call_count, 2)

    def test_chunks_propagates_any_failed_request(self):
        reader = fetch.AorcReader(threads=2, attempts=1)
        self.addCleanup(reader.close)
        reader._get = mock.Mock(side_effect=[b"ok", fetch.AorcDownloadError("bad")])
        with self.assertRaises(fetch.AorcDownloadError):
            reader.chunks(2000, [(0, 0, 0), (0, 0, 1)])

    def test_chunk_wait_allows_all_configured_request_attempts(self):
        reader = fetch.AorcReader(threads=1, attempts=3, timeout=90.0)
        self.addCleanup(reader.close)
        future = mock.Mock()
        future.result.return_value = b"ok"
        with mock.patch.object(reader.pool, "submit", return_value=future):
            self.assertEqual(reader.chunks(2000, [(0, 0, 0)]), [b"ok"])
        # 请求自己的预算为 304.5 秒；整批保险下限为 15 分钟，避免排队中的
        # future 被当作网络失败提前终止。
        future.result.assert_called_once_with(timeout=900.0)

    def test_coord_supports_zlib_and_bad_network_still_raises(self):
        values = np.arange(4, dtype="<i8")
        meta = (b'{"shape":[4],"chunks":[4],"dtype":"<i8",'
                b'"compressor":{"id":"zlib"}}')
        reader = fetch.AorcReader(threads=1, attempts=1)
        self.addCleanup(reader.close)
        reader._get = mock.Mock(side_effect=[meta, zlib.compress(values.tobytes())])
        np.testing.assert_array_equal(reader.coord(2000, "time"), values)

    def test_assemble_rejects_missing_or_wrong_length_block(self):
        with self.assertRaises(ValueError):
            fetch.assemble([], [0], [0])
        bad = Zstd().encode(np.zeros(10, dtype="<i2").tobytes())
        with self.assertRaisesRegex(ValueError, "应为"):
            fetch.assemble([bad], [0], [0])

    def test_missing_values_are_renormalized_not_zero_filled(self):
        slab = np.array([[[2.0, np.nan]]], dtype=np.float32)
        wy = csr_matrix(np.array([[1.0]], dtype=np.float64))
        wx = csr_matrix(np.array([[0.5, 0.5]], dtype=np.float64))
        result = fetch.apply_regrid(slab, wy, wx)
        self.assertAlmostEqual(float(result[0, 0, 0]), 2.0)
        all_missing = fetch.apply_regrid(
            np.full_like(slab, np.nan), wy, wx)
        self.assertTrue(np.isnan(all_missing[0, 0, 0]))

    def test_descending_coordinate_is_rejected(self):
        source_lat = np.array([35.0, 35.01, 35.02])
        source_lon = np.array([-83.0, -82.99, -82.98])
        target_lat = np.array([35.01, 35.00])
        target_lon = np.array([-83.0, -82.99])
        with self.assertRaisesRegex(ValueError, "严格递增"):
            fetch.build_regridder(source_lat, source_lon,
                                  target_lat, target_lon)

    def test_existing_year_is_only_skipped_after_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "aorc_2001.npz"
            np.savez(output, apcp=np.zeros((1, 1, 1), np.float32),
                     time=np.array([0]), lat=np.array([1.0]), lon=np.array([2.0]))
            with mock.patch.object(fetch, "target_grid",
                                   return_value=(np.array([1.0, 1.01]),
                                                 np.array([2.0, 2.01]))):
                with self.assertRaises(ValueError):
                    fetch.fetch_year(2001, directory, mock.Mock(), force=False)

    def test_annual_validation_requires_float32_full_year(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aorc_2001.npz"
            times = fetch.expected_year_times(2001)
            rain = np.zeros((len(times), 2, 2), dtype=np.float16)
            np.savez(path, apcp=rain, time=times,
                     lat=np.array([1.0, 1.01]), lon=np.array([2.0, 2.01]))
            with self.assertRaisesRegex(ValueError, "float32"):
                fetch.validate_annual_file(path, 2001)

    def test_annual_validation_rejects_descending_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aorc_2001.npz"
            times = fetch.expected_year_times(2001)
            rain = np.zeros((len(times), 2, 2), dtype=np.float32)
            np.savez(path, apcp=rain, time=times,
                     lat=np.array([35.01, 35.00]),
                     lon=np.array([-83.00, -82.99]))
            with self.assertRaisesRegex(ValueError, "严格递增"):
                fetch.validate_annual_file(path, 2001)

    def test_annual_validation_rejects_empty_hour_and_all_missing_year(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aorc_2001.npz"
            times = fetch.expected_year_times(2001)
            lat = np.array([35.00, 35.01])
            lon = np.array([-83.00, -82.99])
            rain = np.zeros((len(times), 2, 2), dtype=np.float32)
            rain[17] = np.nan
            np.savez(path, apcp=rain, time=times, lat=lat, lon=lon)
            with self.assertRaisesRegex(ValueError, "第 17 小时全部缺测"):
                fetch.validate_annual_file(path, 2001)

            rain[:] = np.nan
            np.savez(path, apcp=rain, time=times, lat=lat, lon=lon)
            with self.assertRaisesRegex(ValueError, "全部缺测"):
                fetch.validate_annual_file(path, 2001)

    def test_annual_validation_rejects_negative_and_low_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aorc_2001.npz"
            times = fetch.expected_year_times(2001)
            lat = np.array([35.00, 35.01])
            lon = np.array([-83.00, -82.99])
            rain = np.zeros((len(times), 2, 2), dtype=np.float32)
            rain[5, 0, 0] = -0.1
            np.savez(path, apcp=rain, time=times, lat=lat, lon=lon)
            with self.assertRaisesRegex(ValueError, "负降雨"):
                fetch.validate_annual_file(path, 2001)

            # 把第 5 小时打到只剩 1/4 有效（0.25 < 0.50 下限），
            # 验证有效格点比例过低被拒；不能用 3/4——0.75 已高于下限，不会报错。
            rain[5, :, :] = np.nan
            rain[5, 0, 1] = 0.1
            np.savez(path, apcp=rain, time=times, lat=lat, lon=lon)
            with self.assertRaisesRegex(ValueError, "有效格点比例"):
                fetch.validate_annual_file(path, 2001)

    def test_target_grid_rejects_wrong_grid_even_with_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.nc"
            lat = np.array([35.0, 35.01])
            lon = np.array([-83.0, -82.99])
            signature = fetch.grid_signature(lat, lon, "EPSG:4326")
            xr.Dataset(coords={"lat": lat, "lon": lon}, attrs={
                "crs": "EPSG:4326", "grid_signature": signature,
            }).to_netcdf(path)
            with self.assertRaisesRegex(ValueError, "不是已修正"):
                fetch.target_grid(path)


class BuildTests(unittest.TestCase):
    def test_contract_requires_all_1990_through_2015_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for year in range(build.FIRST_YEAR, build.LAST_YEAR + 1):
                if year != 2003:
                    (directory / f"aorc_{year}.npz").touch()
            with self.assertRaisesRegex(FileNotFoundError, "aorc_2003.npz"):
                build.expected_aorc_files(directory)

    def test_inspect_accepts_exact_float_epoch_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            times = build.expected_year_seconds(2001).astype(np.float64)
            lat = np.array([35.0, 35.01])
            lon = np.array([-83.0, -82.99])
            rain = np.zeros((len(times), 2, 2), dtype=np.float32)
            np.savez(directory / "aorc_2001.npz", apcp=rain, time=times,
                     lat=lat, lon=lon)
            with mock.patch.object(build, "FIRST_YEAR", 2001), \
                    mock.patch.object(build, "LAST_YEAR", 2001):
                files, actual_lat, actual_lon = build.inspect_aorc_files(directory)
            self.assertEqual(files, [directory / "aorc_2001.npz"])
            np.testing.assert_array_equal(actual_lat, lat)
            np.testing.assert_array_equal(actual_lon, lon)

    def test_inspect_rejects_fractional_epoch_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            times = build.expected_year_seconds(2001).astype(np.float64)
            times[5] += 0.5
            rain = np.zeros((len(times), 2, 2), dtype=np.float32)
            np.savez(directory / "aorc_2001.npz", apcp=rain, time=times,
                     lat=np.array([35.0, 35.01]),
                     lon=np.array([-83.0, -82.99]))
            with mock.patch.object(build, "FIRST_YEAR", 2001), \
                    mock.patch.object(build, "LAST_YEAR", 2001):
                with self.assertRaisesRegex(ValueError, "完整覆盖"):
                    build.inspect_aorc_files(directory)

    def test_small_chunked_build_is_atomic_and_keeps_mrms_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aorc_dir = directory / "aorc"
            aorc_dir.mkdir()
            lat = np.array([35.0, 35.1], dtype=np.float64)
            lon = np.array([-83.0, -82.9], dtype=np.float64)
            # 缩短年份范围和拼接日期，让完整路径测试只写几十个值。
            years = [2000, 2001]
            split = pd.Timestamp("2001-01-01 03:00")
            for year in years:
                times = build.expected_year_seconds(year)
                rain = np.full((len(times), 2, 2), year, dtype=np.float32)
                np.savez(aorc_dir / f"aorc_{year}.npz", apcp=rain,
                         time=times, lat=lat, lon=lon)

            mrms_times = pd.date_range(split, periods=5, freq="h")
            mrms_values = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
            mrms_values[1, 0, 1] = np.nan
            mrms_nc = directory / "mrms.nc"
            xr.Dataset(
                {"rain": (("time", "lat", "lon"), mrms_values)},
                coords={"time": mrms_times, "lat": lat, "lon": lon},
                attrs={"crs": "EPSG:4326",
                       "grid_signature": build.grid_signature(lat, lon)},
            ).to_netcdf(mrms_nc)
            output = directory / "combined.nc"
            staging = output.with_name(output.name + ".staging")

            real_replace = os.replace
            replace_calls = []

            def record_replace(source, destination):
                self.assertEqual(Path(source), staging)
                with xr.open_dataset(source) as staged:
                    self.assertEqual(staged["rain"].dtype, np.float32)
                replace_calls.append((Path(source), Path(destination)))
                real_replace(source, destination)

            with mock.patch.object(build, "FIRST_YEAR", years[0]), \
                    mock.patch.object(build, "LAST_YEAR", years[-1]), \
                    mock.patch.object(build, "START", pd.Timestamp("2000-01-01")), \
                    mock.patch.object(build, "SPLIT", split), \
                    mock.patch.object(build.os, "replace", side_effect=record_replace):
                build.build(aorc_dir, mrms_nc, output, chunk_hours=2)

            self.assertEqual(len(replace_calls), 1)
            self.assertTrue(output.exists())
            self.assertFalse(staging.exists())
            with xr.open_dataset(output) as result:
                tail = result["rain"].isel(time=slice(-5, None)).values
                self.assertTrue(np.array_equal(tail, mrms_values, equal_nan=True))
                self.assertEqual(result["rain"].dtype, np.float32)

    def test_existing_output_gets_timestamped_hardlink_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rain.nc"
            output.write_bytes(b"old rain")
            backup = build.backup_existing_output(output)
            self.assertIsNotNone(backup)
            self.assertRegex(backup.name, r"rain\.nc\.bak\.\d{8}_\d{6}_\d{6}")
            self.assertEqual(backup.read_bytes(), b"old rain")
            self.assertEqual(os.stat(output).st_ino, os.stat(backup).st_ino)

    def test_backup_failure_aborts_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rain.nc"
            output.write_bytes(b"old rain")
            with mock.patch.object(build.os, "link", side_effect=OSError("no hardlinks")), \
                    mock.patch("shutil.copy2") as copy:
                with self.assertRaisesRegex(RuntimeError, "硬链接备份"):
                    build.backup_existing_output(output)
                copy.assert_not_called()
            self.assertEqual(output.read_bytes(), b"old rain")

    def test_failed_reopen_validation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.nc"
            staging = output.with_name(output.name + ".staging")
            aorc_dir = Path(temporary) / "aorc"
            aorc_dir.mkdir()
            mrms = Path(temporary) / "mrms.nc"
            mrms.touch()
            with mock.patch.object(build, "inspect_aorc_files",
                                   return_value=([], np.array([1.0]), np.array([2.0]))), \
                    mock.patch.object(build.xr, "open_dataset") as open_dataset, \
                    mock.patch.object(build, "validate_mrms", return_value=(np.array([0]), 0)), \
                    mock.patch.object(build, "write_staging", return_value=1), \
                    mock.patch.object(build, "validate_staging", side_effect=ValueError("bad")), \
                    mock.patch.object(build.os, "replace") as replace:
                staging.touch()
                fake = mock.MagicMock()
                open_dataset.return_value.__enter__.return_value = fake
                with self.assertRaisesRegex(ValueError, "bad"):
                    build.build(aorc_dir, mrms, output, chunk_hours=1)
                replace.assert_not_called()
                self.assertFalse(staging.exists())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
