"""Sample-and-hold structure diagnostics, paired W1 and optional beat reconstruction."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibi_structure_metrics import step_structure, summarize_windows, to_sample_and_hold, w1_paired


def held_beats(beats, fs=25):
    """25 Hz sample-and-hold series where each beat value lasts its own duration."""
    return np.concatenate([np.full(int(round(value * fs)), value) for value in beats])


class StructureTests(unittest.TestCase):
    def test_sample_and_hold_signal_is_self_consistent(self):
        series = held_beats([1.6, 1.8, 1.72, 1.64, 1.76, 1.8])
        result = step_structure(series)
        self.assertGreater(result["hold_fraction"], .95)
        self.assertEqual(result["plateaus"], 6)
        self.assertLess(result["median_abs_duration_minus_value_seconds"], .03)

    def test_continuously_varying_signal_has_no_plateaus(self):
        rng = np.random.default_rng(0)
        series = 1.7 + .05 * rng.standard_normal(3000)
        result = step_structure(series)
        self.assertEqual(result["hold_fraction"], 0.)
        self.assertEqual(result["plateaus"], 3000)
        self.assertAlmostEqual(result["median_plateau_seconds"], .04)

    def test_invalid_windows_are_rejected(self):
        for bad in (np.array([1.]), np.ones((2, 3)), np.array([1., np.nan, 1.])):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                step_structure(bad)


class W1Tests(unittest.TestCase):
    def test_matches_scipy_on_observed_positions(self):
        rng = np.random.default_rng(1)
        reference, generated = rng.normal(1.6, .2, 500), rng.normal(1.5, .3, 500)
        mask = rng.random(500) > .1
        self.assertAlmostEqual(w1_paired(reference, generated, mask),
                               wasserstein_distance(reference[mask], generated[mask]), places=12)
        self.assertEqual(w1_paired(reference, reference), 0.)
        with self.assertRaises(ValueError):
            w1_paired(reference, generated, np.zeros(500, dtype=bool))


class SummaryTests(unittest.TestCase):
    def test_window_summary_reports_medians_and_never_selects(self):
        reference = np.stack([held_beats([1.6, 1.7, 1.8, 1.7, 1.6, 1.8])[:150]] * 2)
        rng = np.random.default_rng(2)
        generated = reference + .02 * rng.standard_normal(reference.shape)
        summary = summarize_windows(generated, reference)
        self.assertEqual(summary["windows"], 2)
        self.assertGreater(summary["reference_hold_fraction_median"], .9)
        self.assertEqual(summary["generated_hold_fraction_median"], 0.)
        self.assertGreater(summary["w1_ms_median"], 0.)
        self.assertFalse(summary["used_for_checkpoint_selection"])
        with self.assertRaises(ValueError):
            summarize_windows(generated[:1], reference)


class SampleAndHoldTests(unittest.TestCase):
    def test_reconstruction_restores_plateaus_and_marks_missing(self):
        rng = np.random.default_rng(3)
        noisy = 1.7 + .03 * rng.standard_normal(500)
        noisy[100:105] = (-.05, 0., .1, .15, .19)
        rebuilt = to_sample_and_hold(noisy)
        self.assertEqual(rebuilt.shape, noisy.shape)
        self.assertGreater(step_structure(rebuilt)["hold_fraction"], .9)
        self.assertTrue(np.all((rebuilt == 0) | (rebuilt > 1.5)))
        self.assertIs(type(to_sample_and_hold(held_beats([1.6, 1.8]))), np.ndarray)
        for bad in (0., .31):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                to_sample_and_hold(noisy, missing_threshold=bad)


if __name__ == "__main__":
    unittest.main()
