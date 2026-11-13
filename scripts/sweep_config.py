"""Write one training configuration of a deduplicated dropout sweep grid.

The grid (for example configs/sweep_v3.json) names a base configuration and
rows that change only training_input_dropout.probability and s4_dropout.
Rows whose effective settings coincide are rejected: a dropout probability of
0 has no location, so "no dropout" may appear only once. Normalization must be
the base's minmax [-1, 1] (output_range absent or [-1, 1]).

  python scripts/sweep_config.py --grid configs/sweep_v3.json --index 3 --output OUT/config_3.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW_FIELDS = {"id", "name", "input_probability", "s4_probability"}


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a probability in [0, 1)")
    return float(value)


def load_grid(path):
    path = Path(path)
    grid = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(grid, dict) or not isinstance(grid.get("configurations"), list) or "base" not in grid:
        raise ValueError("Grid needs base and configurations")
    base_path = Path(grid["base"])
    base_path = base_path if base_path.is_absolute() else ROOT / base_path
    base = json.loads(base_path.read_text(encoding="utf-8-sig"))
    output_range = base.get("signal_normalization", {}).get("output_range", [-1, 1])
    if [float(v) for v in output_range] != [-1.0, 1.0]:
        raise ValueError("The sweep base must use minmax output_range [-1, 1]")
    rows, seen_ids, seen_settings = [], set(), {}
    for row in grid["configurations"]:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError("Each configuration needs exactly " + ", ".join(sorted(ROW_FIELDS)))
        if row["id"] in seen_ids or row["id"] != len(rows):
            raise ValueError("Configuration ids must be 0, 1, 2, ... without gaps or repeats")
        settings = (_probability(row["input_probability"], "input_probability"),
                    _probability(row["s4_probability"], "s4_probability"))
        if settings in seen_settings:
            raise ValueError(f"Configuration {row['id']} repeats configuration {seen_settings[settings]} "
                             "(identical effective dropout settings)")
        seen_ids.add(row["id"])
        seen_settings[settings] = row["id"]
        rows.append(dict(row, input_probability=settings[0], s4_probability=settings[1]))
    return base_path, base, rows


def configuration(grid_path, index):
    base_path, base, rows = load_grid(grid_path)
    if not 0 <= index < len(rows):
        raise ValueError(f"index must be in [0, {len(rows) - 1}]")
    row = rows[index]
    config = copy.deepcopy(base)
    config.setdefault("training_input_dropout", {"probability": 0.0, "seed": 42})
    config["training_input_dropout"]["probability"] = row["input_probability"]
    config["s4_dropout"] = row["s4_probability"]
    config["sweep"] = dict(grid=str(Path(grid_path).resolve()), base=str(base_path), id=row["id"], name=row["name"])
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/sweep_v3.json")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args(argv)
    config = configuration(cli.grid, cli.index)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    if cli.output.exists():
        raise FileExistsError(f"Refusing to overwrite {cli.output}")
    cli.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="written", output=str(cli.output), sweep=config["sweep"])))
    return config


if __name__ == "__main__":
    main()
