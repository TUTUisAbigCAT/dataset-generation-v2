"""configs/dreamt_recommended.json parses with every option validator and differs only as documented."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor
from epoch_validation import early_stopping_options, validation_options
from s4_dropout_config import s4_dropout_options
from signal_normalization import normalization_spec
from training_ema import ema_options
from training_input_dropout import input_dropout_options

RECOMMENDED = ROOT / "configs/dreamt_recommended.json"


class RecommendedConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(RECOMMENDED.read_text(encoding="utf-8"))
        self.base = json.loads((ROOT / "dataset_config.json").read_text(encoding="utf-8-sig"))

    def test_all_validators_accept_the_recommended_settings(self):
        self.assertEqual(ema_options(self.config), dict(decay=.9999))
        self.assertEqual(input_dropout_options(self.config), dict(probability=0., seed=42))
        self.assertEqual(s4_dropout_options(self.config), .03)
        self.assertTrue(validation_options(self.config)["save_epoch_checkpoints"])
        early_stopping_options(self.config)
        self.assertEqual(normalization_spec(self.config)["method"], "minmax")
        local = dict(self.config, data_dir=str(ROOT / "example_data"))
        local.pop("evaluation_protocol")
        processor = Preprocessor("DREAMT", config_override=local)
        self.assertEqual(processor.missing_value_policy, "preserve")

    def test_only_the_documented_keys_differ_from_dataset_config(self):
        changed = {key for key in set(self.config) | set(self.base) if self.config.get(key) != self.base.get(key)}
        # Input dropout is already 0 in dataset_config.json; the recommended file keeps it explicit.
        self.assertEqual(changed, {"s4_dropout", "training_ema", "validation"})
        self.assertEqual(self.config["training_input_dropout"]["probability"], 0.)
        validation_changes = {key for key in self.config["validation"]
                              if self.config["validation"][key] != self.base["validation"].get(key)}
        self.assertEqual(validation_changes, {"save_epoch_checkpoints"})

    def test_submission_scripts_match_the_recorded_trillium_environment(self):
        self.assertFalse((ROOT / "submit_recommended.sh").exists())
        for name in ("submit_v2_recommended.sh", "submit_v2_gpu_tests.sh", "submit_sweep_v3.sh"):
            source = (ROOT / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertTrue(source.startswith("#!/bin/bash\n"))
                self.assertNotIn("\r", source)
                for line in ("set -euo pipefail", 'cd "${SLURM_SUBMIT_DIR}"', "module load python/3.11",
                             "module load scipy-stack/2024a", "source ./.venv/bin/activate",
                             "export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1"):
                    self.assertIn(line, source)
        training = (ROOT / "submit_v2_recommended.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=24:00:00", training)
        self.assertIn('EPOCHS="${EPOCHS:-50}"', training)
        self.assertIn('CONFIG="${CONFIG:-configs/dreamt_recommended.json}"', training)
        self.assertIn('--config "$CONFIG" --output-dir "$OUTPUT"', training)
        self.assertIn("/links/scratch/", training.split("OUTPUT=", 1)[1].splitlines()[0])
        tests = (ROOT / "submit_v2_gpu_tests.sh").read_text(encoding="utf-8")
        self.assertIn('COPY="${SLURM_TMPDIR}/dataset-generation-v2"', tests)
        self.assertIn('export TMPDIR="${SLURM_TMPDIR}"', tests)
        self.assertLess(tests.index('cd "${COPY}"'), tests.index("python -B -m unittest discover -s tests -v"))


class FollowUpConfigTests(unittest.TestCase):
    def test_b1_and_large_extend_the_recommended_configuration(self):
        from model_config import model_options
        from training_schedule import schedule_options
        recommended = json.loads(RECOMMENDED.read_text(encoding="utf-8"))
        b1 = json.loads((ROOT / "configs/dreamt_b1.json").read_text(encoding="utf-8"))
        large = json.loads((ROOT / "configs/dreamt_large.json").read_text(encoding="utf-8"))
        self.assertEqual({k for k in set(b1) | set(recommended) if b1.get(k) != recommended.get(k)},
                         {"training_schedule", "validation"})
        self.assertEqual(validation_options(b1)["selection_metric"], "w1")
        self.assertEqual(schedule_options(b1)["decay"], "cosine")
        self.assertEqual({k for k in set(large) | set(b1) if large.get(k) != b1.get(k)}, {"model"})
        self.assertEqual(model_options(large)["num_res_layers"], 8)
        for config in (recommended, b1, large):
            self.assertNotIn("output_range", config["signal_normalization"])
            self.assertEqual(config["missing_value_policy"], "preserve")


if __name__ == "__main__":
    unittest.main()
