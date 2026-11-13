"""Deduplicated dropout sweep grid (configs/sweep_v3.json) and its config writer."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s4_dropout_config import s4_dropout_options
from training_input_dropout import input_dropout_options
from scripts import sweep_config


class SweepConfigTests(unittest.TestCase):
    def test_grid_has_five_unique_settings_on_minus_one_to_one(self):
        _, base, rows = sweep_config.load_grid(ROOT / "configs/sweep_v3.json")
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({(r["input_probability"], r["s4_probability"]) for r in rows}), 5)
        self.assertNotIn("output_range", base["signal_normalization"])

    def test_written_configuration_changes_only_the_dropouts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "config_3.json"
            with contextlib.redirect_stdout(io.StringIO()):
                sweep_config.main(["--grid", str(ROOT / "configs/sweep_v3.json"), "--index", "3", "--output", str(out)])
            config = json.loads(out.read_text())
            base = json.loads((ROOT / "configs/dreamt_recommended.json").read_text())
            self.assertEqual(s4_dropout_options(config), .03)
            self.assertEqual(input_dropout_options(config)["probability"], 0.)
            changed = {k for k in set(config) | set(base) if config.get(k) != base.get(k)}
            self.assertEqual(changed, {"sweep"})          # base already has s4 0.03 / input 0
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                sweep_config.main(["--grid", str(ROOT / "configs/sweep_v3.json"), "--index", "3", "--output", str(out)])

    def test_duplicates_and_unit_interval_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = json.loads((ROOT / "configs/sweep_v3.json").read_text())
            grid["configurations"].append(dict(id=5, name="dup", input_probability=0.0, s4_probability=0.0))
            path = Path(directory) / "grid.json"
            path.write_text(json.dumps(grid))
            with self.assertRaisesRegex(ValueError, "repeats configuration 0"):
                sweep_config.load_grid(path)
            base = json.loads((ROOT / "configs/dreamt_recommended.json").read_text())
            base["signal_normalization"]["output_range"] = [0, 1]
            base_path = Path(directory) / "base.json"
            base_path.write_text(json.dumps(base))
            grid["configurations"].pop()
            grid["base"] = str(base_path)
            path.write_text(json.dumps(grid))
            with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
                sweep_config.load_grid(path)


if __name__ == "__main__":
    unittest.main()
