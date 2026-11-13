"""Opt-in evaluation EMA: configuration, exact no-op default, averaging and allowlist."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epoch_validation import validation_options
from training_ema import TrainingEMA, create_training_ema, ema_options
from scripts.verify_upstream_boundary import (
    EMA_EDITS, UPSTREAM, expected_training_source, remove_training_ema_edits,
    remove_training_input_dropout_edits)
from training_input_dropout import TrainingInputDropout
from tests.test_training_input_dropout import training_loop
NO_SCHEDULE = __import__('types').SimpleNamespace(before_step=lambda: None, after_step=lambda: None)


class BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.7, -.2]))
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.register_buffer("fixed", torch.tensor([1., 2., 3.]))
        self.inputs = []

    def forward(self, values, timesteps):
        self.inputs.append(values.detach().clone())
        return (values[:, :, 0] * self.weight[0] + self.bias).unsqueeze(1)


class EMAOptionTests(unittest.TestCase):
    def test_configuration_is_strict_and_defaults_to_disabled(self):
        self.assertEqual(ema_options({}), dict(decay=0.))
        self.assertEqual(ema_options({"training_ema": {"decay": .9999}}), dict(decay=.9999))
        for invalid in (None, [], {"decay": -.1}, {"decay": 1.}, {"decay": float("nan")},
                        {"decay": True}, {"decay": "0.9"}, {"beta": .9}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ema_options({"training_ema": invalid})

    def test_factory_reads_selected_configuration_and_rejects_missing_file(self):
        model = BufferedModel()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"training_ema": {"decay": .99}}), encoding="utf-8")
            with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(path)}):
                ema = create_training_ema(model)
            self.assertTrue(ema.active)
            self.assertEqual(ema.decay, .99)
            with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(path.with_name("absent.json"))}), \
                    self.assertRaises(FileNotFoundError):
                create_training_ema(model)


class EMABehaviourTests(unittest.TestCase):
    def test_decay_zero_is_an_exact_no_op(self):
        model = BufferedModel()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        with mock.patch.object(copy, "deepcopy", side_effect=AssertionError("decay 0 must not copy the model")):
            ema = TrainingEMA(model, 0.)
            ema.update(model)
        self.assertFalse(ema.active)
        self.assertIs(ema.evaluation_model(model), model)
        self.assertEqual(ema.updates, 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_active_ema_follows_warmup_formula_and_copies_buffers(self):
        model = BufferedModel()
        ema = TrainingEMA(model, .9)
        self.assertIsNot(ema.evaluation_model(model), model)
        self.assertEqual(set(ema.model.state_dict()), set(model.state_dict()))
        expected = model.weight.detach().clone()
        for update in range(1, 6):
            with torch.no_grad():
                model.weight.add_(1.)
                model.fixed.add_(10.)
            ema.update(model)
            decay = min(.9, (1 + update) / (10 + update))
            expected = decay * expected + (1 - decay) * model.weight.detach()
            torch.testing.assert_close(ema.model.weight, expected)
            self.assertTrue(torch.equal(ema.model.fixed, model.fixed))
        self.assertEqual(ema.updates, 5)
        self.assertFalse(any(parameter.requires_grad for parameter in ema.model.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        report = ema.report()
        self.assertTrue(report["active"])
        self.assertFalse(report["random_streams_consumed"])


class EMATrainingLoopTests(unittest.TestCase):
    def run_loop(self, decay):
        source = (ROOT / "training_wavestitch.py").read_bytes()
        evaluated = []
        with torch.random.fork_rng(), contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(42)
            model = BufferedModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            dataset = torch.arange(5 * 12 * 10, dtype=torch.float32).reshape(5, 12, 10) / 100 + 1
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
            ema = TrainingEMA(model, decay)
            losses = []

            def criterion(prediction, target):
                loss = torch.nn.functional.mse_loss(prediction, target)
                losses.append(float(loss.detach()))
                return loss

            def run(epoch, evaluated_model, mean_loss):
                evaluated.append((evaluated_model, {k: v.clone() for k, v in evaluated_model.state_dict().items()}))

            namespace = dict(torch=torch, np=np, model=model, optimizer=optimizer, criterion=criterion,
                             device=torch.device("cpu"), args=SimpleNamespace(epochs=2), dataloader=dataloader,
                             diffusion_config={"T": 3, "alpha_bars": torch.tensor([[.9], [.5], [.1]])},
                             non_hier_cols=torch.tensor([0]), randint=torch.randint, normal=torch.normal,
                             sqrt=torch.sqrt, training_input_dropout=TrainingInputDropout(dict(probability=0.), "cpu"),
                             training_ema=ema, training_schedule=NO_SCHEDULE, validator=SimpleNamespace(run=run, should_stop=False))
            exec(training_loop(source), namespace)
            rng = torch.random.get_rng_state().clone()
        return model, ema, losses, evaluated, rng

    def test_ema_never_changes_training_and_is_what_validation_receives(self):
        live0, ema0, losses0, evaluated0, rng0 = self.run_loop(0.)
        live1, ema1, losses1, evaluated1, rng1 = self.run_loop(.5)
        self.assertEqual(losses0, losses1)
        self.assertTrue(torch.equal(rng0, rng1))
        for key, value in live0.state_dict().items():
            self.assertTrue(torch.equal(value, live1.state_dict()[key]))
        self.assertEqual(len(evaluated0), 2)
        self.assertTrue(all(model is live0 for model, _ in evaluated0))
        self.assertTrue(all(model is ema1.model for model, _ in evaluated1))
        self.assertEqual(ema1.updates, 6)  # 3 batches x 2 epochs
        self.assertFalse(torch.equal(evaluated1[-1][1]["weight"], live1.state_dict()["weight"]))


class EMAAllowlistTests(unittest.TestCase):
    def test_source_is_exact_allowlist_and_reversals_are_strict(self):
        original = (UPSTREAM / "training_wavestitch.py").read_bytes()
        actual = (ROOT / "training_wavestitch.py").read_bytes()
        self.assertEqual(actual, expected_training_source(original))
        without_ema = expected_training_source(original, include_ema=False)
        prior = expected_training_source(original, include_input_dropout=False)
        self.assertEqual(remove_training_ema_edits(actual), without_ema)
        self.assertIs(remove_training_ema_edits(without_ema), without_ema)
        self.assertEqual(remove_training_input_dropout_edits(actual), prior)
        self.assertEqual(remove_training_input_dropout_edits(without_ema), prior)
        with self.assertRaises(ValueError):
            expected_training_source(original, include_input_dropout=False, include_ema=True)
        for before, after in EMA_EDITS:
            partial = actual.replace(after.encode(), before.encode(), 1)
            with self.subTest(partial=before[:40]), self.assertRaises(ValueError):
                remove_training_ema_edits(partial)
        rewritten = actual.replace(b"training_ema.update(model)", b"training_ema.update(model); optimizer.step()")
        with self.assertRaises(ValueError):
            remove_training_ema_edits(rewritten)


class LauncherEMACheckpointTests(unittest.TestCase):
    def test_launcher_validates_ema_and_saves_every_epoch_checkpoint(self):
        # Actual training script, trace observer, validator and file I/O on CPU;
        # CUDA guards and generation are mocked exactly as in the early-stopping launcher test.
        from epoch_validation import EpochValidator
        from scripts import run_dreamt, run_simulation
        from tests.test_early_stopping import TinyTestModel
        from validation_metrics import summarize_distribution

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            workspace, output = base / "workspace", base / "training"
            run_dreamt.snapshot_source(workspace)
            self.assertTrue((workspace / "training_ema.py").is_file())
            config = dict(data_dir=str(ROOT / "example_data"), split_seed=42,
                          early_stopping=dict(enabled=False), training_ema=dict(decay=.5),
                          validation=dict(save_epoch_checkpoints=True))
            config_path = workspace / "dataset_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            env = {key: value for key, value in os.environ.items() if not key.startswith("DREAMT_")}
            env["DREAMT_CONFIG_PATH"] = str(config_path)
            argv = ["run_simulation.py", "--dataset", "DREAMT", "--epochs", "2",
                    "--batch-size", "4", "--output-dir", str(output)]
            validated = []

            def generate(validator, model):
                validated.append(model)
                return summarize_distribution(validator.reference, validator.reference), {"unit_test_mock": True}, []

            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(run_simulation, "ROOT", workspace), \
                 mock.patch.object(run_simulation, "require_cuda", return_value=torch.device("cpu")), \
                 mock.patch.object(run_simulation, "cuda_environment", return_value={"device": "cpu_test_mock"}), \
                 mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cpu")), \
                 mock.patch("cuda_runtime.assert_cuda_model"), \
                 mock.patch("epoch_validation._require_device", return_value=torch.device("cpu")), \
                 mock.patch("training_utils.fetchModel", side_effect=lambda *args: TinyTestModel()), \
                 mock.patch.object(EpochValidator, "_generate", generate), \
                 mock.patch("torch.cuda.synchronize"), contextlib.redirect_stdout(io.StringIO()):
                run_simulation.main()

            read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
            report, manifest = read("training_report.json"), read("run_manifest.json")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(manifest["training_ema"], dict(options=dict(decay=.5)))
            self.assertTrue(report["training_ema"]["active"])
            self.assertEqual(report["training_ema"]["updates"], report["updates"])
            self.assertEqual(len(validated), 2)
            self.assertTrue(all(model is validated[0] for model in validated))
            for epoch in (1, 2):
                epoch_dir = output / "validation" / f"epoch_{epoch:04d}"
                self.assertTrue((epoch_dir / "model.pth").is_file())
                metrics = json.loads((epoch_dir / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["epoch_checkpoint"]["path"], str(epoch_dir / "model.pth"))
            last = torch.load(output / "validation" / "epoch_0002" / "model.pth", map_location="cpu", weights_only=True)
            final = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(set(last), set(final))
            self.assertFalse(torch.equal(last["weight"], final["weight"]))  # EMA copy vs live weights


class EpochCheckpointOptionTests(unittest.TestCase):
    def test_save_epoch_checkpoints_is_optional_and_boolean(self):
        self.assertNotIn("save_epoch_checkpoints", validation_options({}))
        self.assertTrue(validation_options({"validation": {"save_epoch_checkpoints": True}})["save_epoch_checkpoints"])
        for invalid in (1, "yes", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validation_options({"validation": {"save_epoch_checkpoints": invalid}})


if __name__ == "__main__":
    unittest.main()
