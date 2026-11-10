"""Write a held-out-subject split and a matching dataset configuration.

The all-subject protocol trains on every participant, including the evaluation
subject (for example S092), so its KL measures fitting, not generalization.
This script writes:

  splits_holdout_<SID>.json          train = every other NPZ, val = [<SID>.npz], test = []
  dataset_config_holdout_<SID>.json  the base configuration without evaluation_protocol,
                                     with split_path pointing at the split file

With that configuration the signal/metadata scaler is fitted on the training
subjects only and validation runs on the held-out subject. Existing files are
never overwritten.

Example:
  python scripts/make_holdout_split.py --subject S092 --output-dir configs/holdout_S092
  python scripts/run_dreamt.py --config configs/holdout_S092/dataset_config_holdout_S092.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("data_dir", "metadata_path", "output_root")


def _absolute(value, base):
    if value is None or not isinstance(value, str) or not value.strip():
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def build_holdout(base_config_path, subject, output_dir, data_dir=None):
    base_config_path = Path(base_config_path).resolve()
    config = json.loads(base_config_path.read_text(encoding="utf-8-sig")) if base_config_path.is_file() else {}
    if not isinstance(config, dict):
        raise ValueError("Base configuration must be a JSON object")
    for key in PATH_KEYS:
        if key in config:
            config[key] = _absolute(config[key], base_config_path.parent)
    if data_dir is not None:
        config["data_dir"] = str(Path(data_dir).expanduser().resolve())
    if not config.get("data_dir"):
        raise ValueError("A data_dir is required (in the base configuration or via --data-dir)")
    records = Path(config["data_dir"])
    names = sorted(path.name for path in records.glob("*.npz") if path.is_file())
    sid = subject.strip()
    if sid.casefold().endswith(".npz"):
        sid = sid[:-4]
    target = f"{sid}.npz"
    if target not in names:
        raise ValueError(f"Held-out subject {target} is not in {records}")
    if len(names) < 2:
        raise ValueError("At least one training subject besides the held-out subject is required")
    split = dict(train=[name for name in names if name != target], val=[target], test=[])
    config.pop("evaluation_protocol", None)
    split_name = f"splits_holdout_{sid}.json"
    config.update(split_path=split_name, generate_splits=False)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / split_name
    config_path = output_dir / f"dataset_config_holdout_{sid}.json"
    for path, value in ((split_path, split), (config_path, config)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    return dict(split_path=str(split_path), config_path=str(config_path), held_out=target,
                training_subjects=len(split["train"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="S092", help="Participant ID to hold out (default: S092)")
    parser.add_argument("--base-config", type=Path, default=ROOT / "dataset_config.json")
    parser.add_argument("--data-dir", type=Path, help="Override the NPZ folder of the base configuration")
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args(argv)
    result = build_holdout(cli.base_config, cli.subject, cli.output_dir, cli.data_dir)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
