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
        self.assertEqual(processor.missing_value_policy, "masked")

    def test_only_the_documented_keys_differ_from_dataset_config(self):
        changed = {key for key in set(self.config) | set(self.base) if self.config.get(key) != self.base.get(key)}
        # Input dropout is already 0 in dataset_config.json; the recommended file keeps it explicit.
        self.assertEqual(changed, {"missing_value_policy", "s4_dropout", "training_ema", "validation"})
        self.assertEqual(self.config["training_input_dropout"]["probability"], 0.)
        validation_changes = {key for key in self.config["validation"]
                              if self.config["validation"][key] != self.base["validation"].get(key)}
        self.assertEqual(validation_changes, {"save_epoch_checkpoints"})

    def test_submission_script_uses_the_recommended_config(self):
        source = (ROOT / "submit_recommended.sh").read_text(encoding="utf-8")
        self.assertIn("source ./.venv/bin/activate", source)
        self.assertIn("--config configs/dreamt_recommended.json", source)
        self.assertIn("module load python/3.11", source)


if __name__ == "__main__":
    unittest.main()
