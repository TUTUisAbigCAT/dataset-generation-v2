"""Opt-in learning-rate schedule, gradient clipping, model size and EMA final weights."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_config import MODEL_DEFAULTS, model_cli_arguments, model_options
from night_checkpoint import load_inference_config
from training_ema import TrainingEMA
from training_schedule import TrainingSchedule, create_training_schedule, schedule_options
from scripts.verify_upstream_boundary import (
    TRAINING_SCHEDULE_EDITS, UPSTREAM, expected_training_source, remove_training_ema_edits,
    remove_training_schedule_edits)


def tiny():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


class ScheduleTests(unittest.TestCase):
    def test_defaults_are_inactive_and_never_touch_the_optimizer(self):
        model, optimizer = tiny()
        schedule = TrainingSchedule(optimizer, model, 5, 2, schedule_options({}))
        self.assertFalse(schedule.active)
        for _ in range(7):
            schedule.before_step()
            schedule.after_step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(schedule.updates, 0)

    def test_warmup_then_cosine_learning_rates(self):
        model, optimizer = tiny()
        options = dict(warmup_steps=2, decay="cosine", min_lr_ratio=0.1)
        schedule = TrainingSchedule(optimizer, model, 3, 2, schedule_options({"training_schedule": options}))
        seen = []
        for _ in range(6):
            seen.append(optimizer.param_groups[0]["lr"])
            schedule.after_step()
        self.assertAlmostEqual(seen[0], 0.5e-3)
        self.assertAlmostEqual(seen[1], 1e-3)
        self.assertAlmostEqual(seen[2], 1e-3)                  # cosine progress 0
        expected_last = 1e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 3 / 4)))
        self.assertAlmostEqual(seen[5], expected_last)
        self.assertTrue(all(b <= a + 1e-15 for a, b in zip(seen[2:], seen[3:])))

    def test_gradient_clipping_limits_the_norm(self):
        model, optimizer = tiny()
        schedule = TrainingSchedule(optimizer, model, 1, 1, schedule_options({"training_schedule": {"clip_grad_norm": 0.5}}))
        model(torch.full((4, 3), 100.)).sum().backward()
        schedule.before_step()
        norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
        self.assertLessEqual(float(norm), 0.5 + 1e-6)
        self.assertEqual(schedule.clipped_updates, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_invalid_options_are_rejected(self):
        for invalid in ({"warmup_steps": -1}, {"decay": "step"}, {"min_lr_ratio": 2}, {"clip_grad_norm": 0},
                        {"unknown": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                schedule_options({"training_schedule": invalid})

    def test_factory_reads_the_selected_configuration(self):
        model, optimizer = tiny()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"training_schedule": {"warmup_steps": 4}}), encoding="utf-8")
            with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(path)}):
                schedule = create_training_schedule(optimizer, model, 2, 2)
        self.assertTrue(schedule.lr_active)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.25e-3)


class ScheduleBoundaryTests(unittest.TestCase):
    def test_current_source_has_exactly_the_schedule_additions(self):
        original = (UPSTREAM / "training_wavestitch.py").read_bytes()
        actual = (ROOT / "training_wavestitch.py").read_bytes()
        self.assertEqual(actual, expected_training_source(original))
        without = expected_training_source(original, include_schedule=False)
        self.assertEqual(remove_training_schedule_edits(actual), without)
        self.assertIs(remove_training_schedule_edits(without), without)
        self.assertEqual(remove_training_ema_edits(actual), expected_training_source(original, include_ema=False))
        self.assertIn(b"            training_schedule.before_step()\n            optimizer.step()", actual)
        self.assertEqual(actual.count(b"optimizer.step()"), 1)
        partial = actual.replace(b"            training_schedule.after_step()\n", b"", 1)
        with self.assertRaises(ValueError):
            remove_training_schedule_edits(partial)
        self.assertEqual(len(TRAINING_SCHEDULE_EDITS), 5)


class ModelSizeTests(unittest.TestCase):
    def test_defaults_and_cli_arguments(self):
        self.assertEqual(model_options({}), MODEL_DEFAULTS)
        self.assertEqual(model_cli_arguments({}), [])
        self.assertEqual(model_cli_arguments({"model": {"num_res_layers": 8, "res_channels": 128}}),
                         ["-num_res_layers", "8", "-res_channels", "128", "-skip_channels", "64", "-s4_dstate", "64"])
        for invalid in ({"num_res_layers": 0}, {"res_channels": True}, {"depth": 3}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                model_options({"model": invalid})

    def test_inference_configuration_uses_model_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"window_epochs": 4, "model": {"num_res_layers": 8, "res_channels": 128,
                                                                      "skip_channels": 128}}), encoding="utf-8")
            args, _, provenance = load_inference_config(path)
            self.assertEqual((args.num_res_layers, args.res_channels, args.skip_channels, args.s4_dstate), (8, 128, 128, 64))
            self.assertEqual(provenance["model"]["num_res_layers"], 8)
            path.write_text(json.dumps({"window_epochs": 4}), encoding="utf-8")
            args, _, _ = load_inference_config(path)
            self.assertEqual((args.num_res_layers, args.res_channels), (4, 64))


class EmaFinalWeightsTests(unittest.TestCase):
    def test_inactive_ema_saves_live_weights_and_active_saves_average(self):
        model, optimizer = tiny()
        with tempfile.TemporaryDirectory() as directory:
            inactive = TrainingEMA(model, 0.)
            record = inactive.save_evaluation_weights(model, Path(directory) / "a.pth")
            saved = torch.load(record["path"], weights_only=True)
            self.assertTrue(all(torch.equal(saved[k], v) for k, v in model.state_dict().items()))
            active = TrainingEMA(model, .9)
            model.weight.data.add_(1.0)
            active.update(model)
            record = active.save_evaluation_weights(model, Path(directory) / "b.pth")
            saved = torch.load(record["path"], weights_only=True)
            self.assertTrue(record["ema_active"])
            self.assertTrue(torch.equal(saved["weight"], active.model.weight))
            self.assertFalse(torch.equal(saved["weight"], model.weight))
            self.assertEqual(active.report()["final_evaluation_weights"]["sha256"], record["sha256"])


if __name__ == "__main__":
    unittest.main()
