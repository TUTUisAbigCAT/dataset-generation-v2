"""Held-out-subject split: no leakage into training or the scaler."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor
from scripts.make_holdout_split import main


class HoldoutSplitTests(unittest.TestCase):
    def test_holdout_config_excludes_subject_from_training_and_scaler(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base.json"
            base.write_text(json.dumps(dict(
                evaluation_protocol=dict(mode="all_subjects_train_single_subject_eval",
                                         subject="EXAMPLE004", expected_subjects=4),
                window_epochs=4)), encoding="utf-8")
            output = Path(temp) / "holdout"
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["--subject", "EXAMPLE004", "--base-config", str(base),
                               "--data-dir", str(ROOT / "example_data"), "--output-dir", str(output)])
            config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertNotIn("evaluation_protocol", config)
            self.assertEqual(config["split_path"], "splits_holdout_EXAMPLE004.json")
            self.assertFalse(config["generate_splits"])
            with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": result["config_path"]}):
                processor = Preprocessor("DREAMT")
            self.assertEqual(processor.evaluation_filenames, ["EXAMPLE004.npz"])
            self.assertNotIn("EXAMPLE004.npz", processor.splits["train"])
            self.assertNotIn("EXAMPLE004", processor.scaler_subjects)
            self.assertEqual(processor.split_source, "split_file")
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                main(["--subject", "EXAMPLE004", "--base-config", str(base),
                      "--data-dir", str(ROOT / "example_data"), "--output-dir", str(output)])
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                main(["--subject", "S999", "--base-config", str(base),
                      "--data-dir", str(ROOT / "example_data"), "--output-dir", str(Path(temp) / "other")])


if __name__ == "__main__":
    unittest.main()
