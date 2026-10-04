# -*- coding: utf-8 -*-
"""S.13 共用数学与安全边界测试；默认不读 5.5 GB 降雨数据。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from step13_rl import (S13ActorCritic, TrajectoryBuffer, compute_gae,
                       corrected_split_origins, gaussian_kl, gaussian_log_prob,
                       load_s10_archive, peak_time_terms, reward_from_levels,
                       site_activity_scale, trajectory_loss)
from train import MoENet

STEPS = 24


def triangle_flood(peak_at: int, amp: float = 8.0, base: float = 0.0,
                   steps: int = STEPS) -> torch.Tensor:
    """三角形洪水过程线，峰出现在 peak_at 时刻。"""
    t = torch.arange(steps, dtype=torch.float32)
    return torch.clamp(amp - (t - peak_at).abs() * (amp / 6.0), min=0.0) + base


class PeakTimeTests(unittest.TestCase):
    """峰现时间项：奖励用它，闭环损失也用它，两边必须是同一套算法。"""

    def terms(self, pred, target, scale=1.0, mask=None):
        mask = torch.ones(1, STEPS, dtype=torch.bool) if mask is None else mask
        return peak_time_terms(pred[None], target[None], mask,
                               torch.tensor([scale], dtype=torch.float32))

    def test_identical_trajectory_is_not_penalized(self):
        pen, gate = self.terms(triangle_flood(10), triangle_flood(10))
        self.assertAlmostEqual(float(pen), 0.0, places=5)
        self.assertAlmostEqual(float(gate), 1.0, places=5)

    def test_shift_is_linear_in_hours_and_capped(self):
        _, gate = self.terms(triangle_flood(10), triangle_flood(10))
        self.assertAlmostEqual(float(gate), 1.0, places=5)
        pen3, _ = self.terms(triangle_flood(13), triangle_flood(10))
        self.assertAlmostEqual(float(pen3), 3.0 / 6.0, places=4)
        pen9, _ = self.terms(triangle_flood(19), triangle_flood(10))
        self.assertAlmostEqual(float(pen9), 1.0, places=4)

    def test_flat_recession_window_is_gated_out(self):
        """枯水窗口的"峰"只是噪声，绝不能拿它去训练模型。"""
        obs = triangle_flood(11, amp=0.02, base=0.5)
        sim = triangle_flood(12, amp=0.02, base=0.5)
        _, gate = self.terms(sim, obs, scale=1.0)
        self.assertLess(float(gate), 0.02)
        # 同样幅度的起伏放在大站（常态波动大）上，门槛只会更严。
        _, big = self.terms(sim, obs, scale=3.0)
        self.assertLess(float(big), float(gate))

    def test_missing_target_does_not_fabricate_a_peak(self):
        mask = torch.ones(1, STEPS, dtype=torch.bool)
        mask[0, 5:9] = False
        clean = triangle_flood(10)
        dirty = clean.clone()
        dirty[5:9] = 999.0            # 缺测位置若被当真实值会造假峰
        p1, g1 = self.terms(clean, clean, mask=mask)
        p2, g2 = self.terms(clean, dirty, mask=mask)
        self.assertAlmostEqual(float(p1), float(p2), places=5)
        self.assertAlmostEqual(float(g1), float(g2), places=5)

    def test_all_missing_is_inert(self):
        mask = torch.zeros(1, STEPS, dtype=torch.bool)
        pen, gate = peak_time_terms(triangle_flood(10)[None],
                                    torch.full((1, STEPS), float("nan")), mask,
                                    torch.tensor([1.0]))
        self.assertEqual(float(pen), 0.0)
        self.assertEqual(float(gate), 0.0)

    def test_differentiable_for_the_supervised_mode(self):
        pred = triangle_flood(13)[None].clone().requires_grad_(True)
        pen, gate = peak_time_terms(pred, triangle_flood(10)[None],
                                    torch.ones(1, STEPS, dtype=torch.bool),
                                    torch.tensor([1.0]))
        (pen * gate).mean().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0.0)

    def test_reward_punishes_only_the_mistimed_trajectory(self):
        target = triangle_flood(10).repeat(2, 1)
        pred = torch.stack([triangle_flood(10), triangle_flood(16)])
        off = dict(level_weight=0.0, delta_weight=0.0, peak_weight=0.0)
        base, _ = reward_from_levels(pred, target, None, **off)
        self.assertEqual(float(base.sum()), 0.0)
        timed, _ = reward_from_levels(pred, target, None, **off,
                                      peak_time_weight=0.2,
                                      site_scale=torch.ones(2))
        self.assertEqual(float(timed[0].sum()), 0.0)
        self.assertLess(float(timed[1].sum()), 0.0)

    def test_timing_term_must_have_a_station_scale(self):
        """全局归一化下不给站点尺度就会悄悄退化成大小站不公平，必须报错。"""
        target = triangle_flood(10).repeat(2, 1)
        pred = torch.stack([triangle_flood(10), triangle_flood(16)])
        with self.assertRaises(ValueError):
            reward_from_levels(pred, target, None, level_weight=0.0,
                               peak_time_weight=0.2)
        with self.assertRaises(ValueError):
            trajectory_loss(pred, target, None, peak_time_weight=0.2)

    def test_trajectory_loss_reports_timing_component(self):
        target = triangle_flood(10).repeat(2, 1)
        pred = torch.stack([triangle_flood(10), triangle_flood(16)])
        _, comp = trajectory_loss(pred, target, None, peak_time_weight=0.2,
                                  site_scale=torch.ones(2))
        self.assertIn("timing", comp)
        self.assertGreater(float(comp["timing"]), 0.0)

    def test_site_activity_scale_is_positive_and_per_site(self):
        times = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h")
        flow_n = np.random.default_rng(0).normal(size=(3, len(times)))
        flow_n[0] *= 5.0
        scale = site_activity_scale(flow_n, times)
        self.assertEqual(scale.shape, (3,))
        self.assertTrue(np.all(scale > 0))
        self.assertGreater(scale[0], scale[2])


class RewardMaskTests(unittest.TestCase):
    def test_missing_target_does_not_become_zero(self):
        pred = torch.tensor([[3.0, 100.0, 5.0]])
        target = torch.tensor([[2.0, float("nan"), 7.0]])
        reward, mask = reward_from_levels(pred, target)
        self.assertEqual(mask.tolist(), [[True, False, True]])
        self.assertEqual(float(reward[0, 1]), 0.0)
        # 有效点 Huber: |1| -> 0.5，|2| -> 1.5。
        self.assertTrue(torch.allclose(reward[[0], [0]], torch.tensor([-0.5])))
        self.assertTrue(torch.allclose(reward[[0], [2]], torch.tensor([-1.5])))

    def test_trajectory_loss_ignores_nan_prediction_position(self):
        pred_a = torch.tensor([[1.0, 9999.0, 4.0]], requires_grad=True)
        pred_b = torch.tensor([[1.0, -9999.0, 4.0]], requires_grad=True)
        target = torch.tensor([[2.0, float("nan"), 3.0]])
        la, ca = trajectory_loss(pred_a, target, delta_weight=0.0, peak_weight=0.0)
        lb, cb = trajectory_loss(pred_b, target, delta_weight=0.0, peak_weight=0.0)
        self.assertAlmostEqual(float(la.detach()), float(lb.detach()), places=7)
        self.assertEqual(int(ca["valid_points"]), 2)
        la.backward()
        self.assertEqual(float(pred_a.grad[0, 1]), 0.0)


class ProbabilityTests(unittest.TestCase):
    def test_gaussian_logprob_known_value(self):
        x = torch.tensor([0.0])
        lp = gaussian_log_prob(x, x, torch.ones_like(x))
        self.assertAlmostEqual(float(lp), -0.5 * np.log(2 * np.pi), places=6)

    def test_gaussian_kl_identity_and_shift(self):
        m = torch.tensor([0.0, 1.0])
        s = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.allclose(gaussian_kl(m, s, m, s), torch.zeros(2), atol=1e-7))
        shifted = gaussian_kl(torch.tensor([1.0]), torch.tensor([1.0]),
                              torch.tensor([0.0]), torch.tensor([1.0]))
        self.assertAlmostEqual(float(shifted), 0.5, places=6)

    def test_gae_terminal_and_mask(self):
        reward = torch.tensor([[1.0, 1.0, 9.0]])
        value = torch.zeros_like(reward)
        done = torch.tensor([[False, True, True]])
        valid = torch.tensor([[True, True, False]])
        adv, ret = compute_gae(reward, value, done, valid, gamma=1.0, lam=1.0)
        self.assertTrue(torch.allclose(adv, torch.tensor([[2.0, 1.0, 0.0]])))
        self.assertTrue(torch.allclose(ret, adv))
    def test_gae_gap_keeps_future_credit_for_prior_action(self):
        reward = torch.tensor([[1.0, 0.0, 2.0]])
        value = torch.zeros_like(reward)
        done = torch.tensor([[False, False, True]])
        valid = torch.tensor([[True, False, True]])
        adv, _ = compute_gae(reward, value, done, valid, gamma=1.0, lam=1.0)
        # 中间缺测动作不训练，但第一个动作仍影响第三步水位，因此收到后续奖励。
        self.assertTrue(torch.allclose(adv, torch.tensor([[3.0, 0.0, 2.0]])))


class SplitTests(unittest.TestCase):
    def test_targets_stay_inside_years(self):
        times = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h")
        split = corrected_split_origins(times, train_stride=17, val_stride=13)
        b1 = int(times.searchsorted(pd.Timestamp("2023-01-01")))
        b2 = int(times.searchsorted(pd.Timestamp("2024-01-01")))
        self.assertTrue(np.all(split.train + 72 + 24 <= b1))
        self.assertTrue(np.all(split.val + 72 >= b1))
        self.assertTrue(np.all(split.val + 72 + 24 <= b2))
        self.assertLess(int(split.val[-1] + 72 + 23), b2)


class ModelTests(unittest.TestCase):
    @staticmethod
    def make_model() -> S13ActorCritic:
        base = MoENet(True, False, 96, 48, 1, n_quant=3,
                      masked_pool=False, delta_cap=0.0,
                      areas=np.array([10.0, 20.0, 100.0]))
        return S13ActorCritic(base)

    def test_only_heads_gate_value_and_scale_train(self):
        model = self.make_model()
        trainable = set(model.trainable_parameter_names())
        self.assertIn("log_std_scale", trainable)
        self.assertTrue(any(n.startswith("policy.experts.0.head.") for n in trainable))
        self.assertTrue(any(n.startswith("policy.gate.") for n in trainable))
        self.assertTrue(any(n.startswith("value_head.") for n in trainable))
        self.assertFalse(any(n.startswith("reference.") for n in trainable))
        self.assertFalse(any("policy.experts.0.lstm" in n for n in trainable))
        self.assertFalse(any("policy.experts.0.cnn" in n for n in trainable))

    def test_initial_policy_exactly_matches_reference_mean(self):
        model = self.make_model().eval()
        x = torch.randn(2, 2, 16, 16)
        hist = torch.randn(2, 72, 1)
        future = torch.randn(2, 1, 16, 16)
        site = torch.tensor([0, 2])
        with torch.no_grad():
            pm, _, _, _ = model.policy_stats(x, hist, future, site)
            rm, _ = model.reference_stats(x, hist, future, site)
        self.assertTrue(torch.equal(pm, rm))

    def test_level_source_is_converted_to_delta_action(self):
        base = MoENet(True, False, 96, 48, 1, n_quant=3,
                      masked_pool=False, delta_cap=0.0,
                      areas=np.array([10.0, 20.0, 100.0]))
        delta_model = S13ActorCritic(base, base_output_mode="delta").eval()
        level_model = S13ActorCritic(base, base_output_mode="level").eval()
        x = torch.randn(2, 2, 16, 16)
        hist = torch.randn(2, 72, 1)
        future = torch.randn(2, 1, 16, 16)
        site = torch.tensor([0, 2])
        anchor = hist[:, -1, 0]
        with torch.no_grad():
            dm, ds, _, dq = delta_model.policy_stats(x, hist, future, site)
            lm, ls, _, lq = level_model.policy_stats(x, hist, future, site)
        self.assertTrue(torch.allclose(lq, dq - anchor[:, None], atol=1e-6))
        self.assertTrue(torch.allclose(lm, dm - anchor, atol=1e-6))
        # 三个分位数同时减去锚点，间距和策略标准差不应改变。
        self.assertTrue(torch.allclose(ls, ds, atol=1e-6))

    def test_buffer_rejects_rain_images(self):
        buffer = TrajectoryBuffer()
        with self.assertRaises(ValueError):
            buffer.add(rain_images=torch.zeros(1))
        self.assertFalse(buffer.stores_rain_images)


class ArchiveTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "runs" / "site_model_moe" / "best.pt").is_file(),
                         "本机没有 S.10 存档")
    def test_real_s10_archive_strict_load_and_identity(self):
        base, norm, info = load_s10_archive(ROOT / "runs" / "site_model_moe")
        self.assertEqual(info["parameter_count"], 162743)
        self.assertEqual(norm.transform, "gstd")
        self.assertEqual(len(norm.ids), 15)
        actor = S13ActorCritic(base)
        p = actor.policy.state_dict()
        r = actor.reference.state_dict()
        self.assertEqual(p.keys(), r.keys())
        self.assertTrue(all(torch.equal(p[k], r[k]) for k in p))


if __name__ == "__main__":
    unittest.main()
