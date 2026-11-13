"""Pooled distribution diagnostics and HRV readers (numpy only)."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distribution_metrics import (composite_score, ks_statistic, pooled_distribution_metrics,
                                  stage_per_sample, wasserstein1, window_kl_summary)
from hrv_metrics import read_beats, summarize_hrv, window_hrv


class DistributionMetricTests(unittest.TestCase):
    def test_identical_samples_have_zero_distance(self):
        x = np.linspace(0.6, 1.8, 500)
        self.assertEqual(wasserstein1(x, x), 0.0)
        self.assertEqual(ks_statistic(x, x), 0.0)

    def test_shift_gives_w1_equal_to_shift(self):
        x = np.random.default_rng(0).normal(1.0, 0.1, 1000)
        self.assertAlmostEqual(wasserstein1(x, x + 0.05), 0.05, places=12)

    def test_matches_scipy_when_available(self):
        try:
            from scipy.stats import ks_2samp, wasserstein_distance
        except ImportError:
            self.skipTest("scipy not installed")
        rng = np.random.default_rng(1)
        a, b = rng.normal(0.9, 0.2, 800), rng.normal(1.1, 0.3, 800)
        self.assertAlmostEqual(wasserstein1(a, b), wasserstein_distance(a, b), places=12)
        self.assertAlmostEqual(ks_statistic(a, b), ks_2samp(a, b).statistic, places=12)

    def test_pooled_metrics_use_observed_positions_and_stages(self):
        reference = np.array([[0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 0.0, 2.0]])
        generated = np.array([[5.0, 1.1, 1.1, 2.1], [1.1, 1.1, -3.0, 2.1]])
        mask = reference > 0
        stages = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        result = pooled_distribution_metrics(generated, reference, mask, stages)
        self.assertEqual(result["samples"], 6)
        self.assertAlmostEqual(result["pooled_w1_ms"], 100.0, places=9)
        self.assertAlmostEqual(result["mean_bias_ms"], 100.0, places=9)
        self.assertEqual(result["per_stage"]["W"]["samples"], 3)
        self.assertAlmostEqual(result["per_stage"]["W"]["w1_ms"], 100.0, places=9)
        self.assertEqual(result["per_stage"]["REM"]["samples"], 0)

    def test_rejects_nonfinite_generated_and_bad_masks(self):
        with self.assertRaises(ValueError):
            pooled_distribution_metrics(np.array([1.0, np.nan]), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            pooled_distribution_metrics(np.ones(3), np.ones(3), observed_mask=np.ones(3))

    def test_stage_per_sample_and_kl_summary(self):
        onehot = np.eye(4)[[0, 3, 2, 1]]
        np.testing.assert_array_equal(stage_per_sample(onehot), [0, 3, 2, 1])
        summary = window_kl_summary([0.1, 0.2, 3.0, 0.4])
        self.assertAlmostEqual(summary["median"], 0.3)
        self.assertAlmostEqual(summary["share_above_1"], 0.25)

    def test_composite_ranking_is_unit_invariant(self):
        a = composite_score(0.5, 100.0, 0.2)
        b = composite_score(0.4, 150.0, 0.2)
        self.assertEqual(a < b, (0.5 * 100 * 0.2) < (0.4 * 150 * 0.2))
        self.assertGreater(composite_score(0.0, 0.0, 0.0), 0.0)


class HrvTests(unittest.TestCase):
    def test_reader_recovers_sample_and_hold_beats(self):
        beats = [0.8, 1.0, 0.8, 1.2]
        series = np.concatenate([np.full(int(round(b * 25)), b) for b in beats])
        values, segments = read_beats(series)
        np.testing.assert_allclose(values, beats)
        self.assertTrue((segments == 0).all())

    def test_missing_marker_breaks_successive_differences(self):
        series = np.concatenate([np.full(20, 0.8), [0.0], np.full(25, 1.0), np.full(20, 0.8)])
        values, segments = read_beats(series)
        np.testing.assert_allclose(values, [0.8, 1.0, 0.8])
        self.assertEqual(segments.tolist(), [0, 1, 1])

    def test_window_hrv_values(self):
        beats = [1.0, 1.1] * 10
        series = np.concatenate([np.full(int(round(b * 25)), b) for b in beats])
        result = window_hrv(series)
        self.assertEqual(result["beats"], 20)
        self.assertAlmostEqual(result["rmssd_ms"], 100.0, places=6)
        self.assertAlmostEqual(result["pnn50_percent"], 100.0)
        self.assertAlmostEqual(result["mean_hr_bpm"], 60 / 1.05, places=6)

    def test_summary_shapes_and_short_windows(self):
        reference = np.full((2, 100), 0.8)
        generated = np.zeros((2, 100))
        summary = summarize_hrv(generated, reference)
        self.assertEqual(summary["windows"], 2)
        self.assertIsNone(summary["generated"]["sdnn_ms"])
        self.assertIsNotNone(summary["real"]["mean_hr_bpm"])


if __name__ == "__main__":
    unittest.main()
