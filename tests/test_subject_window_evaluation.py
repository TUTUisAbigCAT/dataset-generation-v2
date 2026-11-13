"""Any-participant window evaluation and condition-effect checks on synthetic data (CPU, fake sampler)."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor
from night_checkpoint import load_inference_config
from night_data import prepare_night
from scripts import check_condition_effect, evaluate_subject_windows
import subject_windows as sw

# Four-class mean IBI per stage (W, Light, REM, Deep) and an AHI effect, in seconds.
STAGE_IBI = np.array([0.80, 1.00, 0.90, 1.10])
AHI_EFFECT = 0.004
RAW_STAGES = np.array([0, 2, 2, 4, 3, 3, 2, 0, 1, 4, 3, 2, 2, 0, 4, 2])   # W N2 N2 REM Deep Deep ...
FOUR = np.array([0, 1, 1, 2, 3, 3, 1, 0, 1, 2, 3, 1, 1, 0, 2, 1])
SUBJECTS = (("S001", 5, 22, 30, "F"), ("S002", 40, 30, 55, "M"), ("S003", 15, 26, 45, "F"), ("S092", 60, 33, 60, "M"))


def write_dataset(directory):
    records = Path(directory) / "records"
    records.mkdir()
    for sid, ahi, *_ in SUBJECTS:
        values = np.repeat(STAGE_IBI[FOUR] + AHI_EFFECT * ahi, 750).astype(np.float32)
        values[::750] = 0.0                                   # one missing marker per epoch
        np.savez_compressed(records / f"{sid}.npz", data=values[:, None], stages=np.repeat(RAW_STAGES, 750),
                            fs=np.array(25), ahi=np.array(float(ahi)))
    with (records / "participant_info.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["SID", "AHI", "BMI", "AGE", "GENDER"])
        writer.writeheader()
        for sid, ahi, bmi, age, gender in SUBJECTS:
            writer.writerow(dict(SID=sid, AHI=ahi, BMI=bmi, AGE=age, GENDER=gender))
    (records / "splits.json").write_text(json.dumps(dict(train=["S001.npz", "S002.npz", "S003.npz"],
                                                         val=["S092.npz"], test=[])), encoding="utf-8")
    config = Path(directory) / "config.json"
    config.write_text(json.dumps(dict(data_dir=str(records), window_epochs=4,
                                      signal_normalization=dict(method="minmax", min_seconds=0.0, max_seconds=2.0))),
                      encoding="utf-8")
    return config


def conditional_sampler(scaler, use_stage=True, use_ahi=True):
    """Fake sampler: generated seconds follow the given conditions exactly."""
    from signal_normalization import signal_affine_parameters
    center, scale = signal_affine_parameters(scaler)
    mean, std = scaler["continuous_mean"][0], scaler["continuous_std"][0]

    def sample(model, clean, schedule, args, device):
        stages = clean[:, :, 1:5].argmax(dim=2).cpu().numpy()
        seconds = STAGE_IBI[stages] if use_stage else np.full(stages.shape, STAGE_IBI.mean())
        if use_ahi:
            seconds = seconds + AHI_EFFECT * (clean[:, :, 5].cpu().numpy() * std + mean)
        return torch.as_tensor((seconds - center) / scale), {}
    return sample


def fake_loader(args, checkpoint, device):
    return None, dict(path=str(checkpoint), sha256="fake")


class SubjectWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = write_dataset(self.directory)
        self.checkpoint = self.directory / "model.pth"
        self.checkpoint.write_bytes(b"placeholder")
        args, data_config, _ = load_inference_config(self.config)
        self.args = args
        self.preprocessor = Preprocessor("DREAMT", data_config=data_config)

    def test_window_selection_and_condition_replacement(self):
        prepared = prepare_night(self.preprocessor, "S092", allow_time_gaps=True)
        starts = sw.select_window_starts(prepared, 3000, 3000)
        np.testing.assert_array_equal(starts, [0, 3000, 6000, 9000])
        self.assertEqual(len(sw.select_window_starts(prepared, 3000, 750)), 13)
        self.assertEqual(len(sw.select_window_starts(prepared, 3000, 750, max_windows=3)), 3)
        reference, mask, stage_samples, epoch_stages = sw.window_arrays(prepared, starts, 3000)
        self.assertEqual(reference.shape, (4, 3000))
        self.assertEqual(int((~mask).sum()), 16)
        np.testing.assert_array_equal(epoch_stages.reshape(-1), FOUR)
        clean = sw.build_clean_windows(prepared["conditions"], starts, 3000)
        self.assertTrue((clean[:, :, 0] == 0).all())
        swapped = sw.replace_stages(clean, np.full(stage_samples.shape, 3))
        self.assertTrue((swapped[:, :, 4] == 1).all())
        self.assertTrue(np.array_equal(swapped[:, :, 5:], clean[:, :, 5:]))
        donor = sw.subject_demographics(prepare_night(self.preprocessor, "S001", allow_time_gaps=True))
        self.assertTrue((sw.replace_demographics(clean, donor)[:, :, 5:10] == donor).all())

    def test_evaluate_subject_windows_end_to_end_with_perfect_fake_sampler(self):
        output = self.directory / "eval"
        cli = evaluate_subject_windows.parse_args([
            "--checkpoint", str(self.checkpoint), "--config", str(self.config), "--subjects", "S092", "S001",
            "--output-dir", str(output), "--batch-size", "3"])
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_subject_windows.run(cli, sampler=conditional_sampler(self.preprocessor.scaler),
                                                  model_loader=fake_loader, device=torch.device("cpu"))
        self.assertEqual(report["status"], "completed")
        rows = list(csv.DictReader((output / "subjects_summary.csv").open(encoding="utf-8")))
        self.assertEqual([row["subject"] for row in rows], ["S092", "S001"])
        self.assertEqual(rows[0]["role"], "val")
        for row in rows:
            self.assertLess(float(row["pooled_w1_ms"]), 1e-3)
            self.assertLess(float(row["window_kl_mean"]), 1e-3)
        summary = json.loads((output / "subject_S092" / "summary.json").read_text())
        self.assertEqual(summary["windows"], 4)
        self.assertIn("per_stage", summary["pooled_distribution"])
        with np.load(output / "subject_S092" / "inputs.npz") as inputs:
            self.assertEqual(inputs["start_samples"].tolist(), [0, 3000, 6000, 9000])
        from scripts import plot_s092_all_windows_kde
        with contextlib.redirect_stdout(io.StringIO()):
            plot_s092_all_windows_kde.main([str(output), "--folder", "subject_S092", "--grid", "4"])
        self.assertTrue((output / "plots_subject_S092_windows" / "window_waveform_grid.png").is_file())

    def test_check_only_needs_no_model(self):
        output = self.directory / "check"
        cli = evaluate_subject_windows.parse_args([
            "--checkpoint", str(self.checkpoint), "--config", str(self.config), "--subjects", "S003",
            "--output-dir", str(output), "--check-only", "--window-stride-seconds", "30"])
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_subject_windows.run(cli)
        self.assertEqual(report["status"], "checked_only")
        self.assertEqual(report["subjects"]["S003"]["windows"], 13)


class ConditionEffectTests(SubjectWindowTests):
    def run_checks(self, sampler, checks, name):
        output = self.directory / name
        argv = ["--checkpoint", str(self.checkpoint), "--config", str(self.config), "--subject", "S092",
                "--subjects", "S001", "S002", "S003", "S092", "--output-dir", str(output),
                "--checks", *checks, "--max-windows", "4"]
        cli = check_condition_effect.parse_args(argv)
        with contextlib.redirect_stdout(io.StringIO()):
            return check_condition_effect.run(cli, sampler=sampler, model_loader=fake_loader,
                                              device=torch.device("cpu"))

    def test_real_r2_reflects_stage_and_identity_effects(self):
        report = self.run_checks(None, ["real-r2"], "r2")
        r2 = report["results"]["real_r2"]["r2"]
        self.assertEqual(report["results"]["real_r2"]["participants"], 4)
        self.assertGreater(r2["participant_identity_and_stage"], 0.99)
        self.assertGreater(r2["stage_and_demographics"], r2["stage"])
        self.assertAlmostEqual(report["results"]["real_r2"]["real_per_stage_ibi"]["W"]["mean_s"],
                               0.80 + AHI_EFFECT * np.mean([5, 40, 15, 60]), places=6)

    def test_conditioned_sampler_passes_all_model_checks(self):
        report = self.run_checks(conditional_sampler(self.preprocessor.scaler),
                                 ["stage-swap", "shuffle", "demographic-swap"], "pass")
        results = report["results"]
        self.assertTrue(results["stage_swap"]["order_matches"])
        self.assertAlmostEqual(results["stage_swap"]["spread_ratio"], 1.0, places=6)
        self.assertTrue(results["shuffle"]["passed"])
        self.assertLess(results["shuffle"]["w1_true_ms"], 1e-3)
        self.assertAlmostEqual(results["shuffle"]["pooled_w1_shuffled_ms"], results["shuffle"]["pooled_w1_true_ms"], places=3)
        self.assertEqual(results["demographic_swap"]["diagonal_share"], 1.0)
        self.assertTrue(report["decision"]["conditions_sufficient"])
        self.assertTrue((self.directory / "pass" / "condition_checks.png").is_file())

    def test_sampler_ignoring_conditions_fails(self):
        report = self.run_checks(conditional_sampler(self.preprocessor.scaler, use_stage=False, use_ahi=False),
                                 ["stage-swap", "shuffle", "demographic-swap"], "fail")
        results = report["results"]
        self.assertFalse(results["stage_swap"]["passed"])
        self.assertAlmostEqual(results["shuffle"]["w1_ratio"], 1.0, places=6)
        self.assertFalse(results["shuffle"]["passed"])
        self.assertFalse(report["decision"]["conditions_sufficient"])

    def test_arguments_require_checkpoint_and_two_subjects(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            check_condition_effect.parse_args(["--config", str(self.config), "--output-dir", "x"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            check_condition_effect.parse_args(["--config", str(self.config), "--output-dir", "x",
                                               "--checkpoint", "m.pth", "--subjects", "S001"])


if __name__ == "__main__":
    unittest.main()
