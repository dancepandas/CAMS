# -*- coding: utf-8 -*-
"""普通监督、旧模型和 S.13 密集推理入口的快速测试。"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rollout_dense as rollout
from data_contract import MANIFEST_FORMAT
from train import MoENet


def args_for(run_dir: Path, manifest: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(run=None, run_dir=str(run_dir), checkpoint=None,
                              manifest=None if manifest is None else str(manifest))


class ManifestTests(unittest.TestCase):
    def write_manifest(self, path: Path, format_name: str) -> None:
        path.write_text(json.dumps({"format": format_name}), encoding="utf-8")

    def test_detects_supervised_and_s13_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            supervised = run_dir / "training_manifest.json"
            self.write_manifest(supervised, MANIFEST_FORMAT)
            result = rollout.resolve_paths(args_for(run_dir))
            self.assertEqual(result[3], "supervised")
            supervised.unlink()

            s13 = run_dir / "manifest.json"
            self.write_manifest(s13, "cams-s13-v1")
            result = rollout.resolve_paths(args_for(run_dir))
            self.assertEqual(result[3], "s13")

    def test_two_default_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.write_manifest(run_dir / "training_manifest.json", MANIFEST_FORMAT)
            self.write_manifest(run_dir / "manifest.json", "cams-s13-v1")
            with self.assertRaisesRegex(ValueError, "同时有两种"):
                rollout.resolve_paths(args_for(run_dir))

    def test_v1_training_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_manifest.json"
            self.write_manifest(path, "cams-training-manifest-v1")
            with self.assertRaisesRegex(ValueError, "旧 v1"):
                rollout.read_run_manifest(path)


class SupervisedModelTests(unittest.TestCase):
    def model_config(self) -> dict:
        return {
            "arch": "moe", "lookback": 72, "horizon": 1,
            "use_spatial": True, "use_site": False, "hidden": 96,
            "spatial_dim": 48, "loss": "quantile",
            "quantiles": [0.1, 0.5, 0.9], "masked_pool": False,
            "delta_cap": 0.0, "output_mode": "level",
        }

    def test_builds_moe_from_effective_config_and_loads_raw_state(self):
        cfg = self.model_config()
        areas = np.array([10.0, 20.0])
        original = rollout.build_supervised_model(cfg, areas, 2)
        self.assertIsInstance(original, MoENet)
        self.assertEqual(original.n_quant, 3)
        rebuilt = rollout.build_supervised_model(cfg, areas, 2)
        rebuilt.load_state_dict(original.state_dict(), strict=True)
        for left, right in zip(original.parameters(), rebuilt.parameters()):
            self.assertTrue(torch.equal(left, right))

        with mock.patch.object(rollout, "load_s13_checkpoint") as s13_loader:
            rebuilt.load_state_dict(original.state_dict(), strict=True)
            s13_loader.assert_not_called()

    def test_output_mode_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_mode"):
            rollout.require_output_mode({"output_mode": "delta"},
                                        self.model_config())


class OriginTests(unittest.TestCase):
    def test_different_internal_indices_map_to_same_physical_origin(self):
        short = pd.date_range("2015-06-01", "2024-01-02", freq="h")
        long = pd.date_range("1990-01-01", "2024-01-02", freq="h")
        begin = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-02")
        short_origins = rollout.split_origins(short, begin, end, 72, 24, 1)
        long_origins = rollout.split_origins(long, begin, end, 72, 24, 1)
        self.assertNotEqual(int(short_origins[0]), int(long_origins[0]))
        short_targets = short[short_origins + 72]
        long_targets = long[long_origins + 72]
        self.assertTrue(short_targets.equals(long_targets))
        self.assertEqual(short_targets[0], begin)


if __name__ == "__main__":
    unittest.main()
