"""Optional network-size configuration (residual layers, channels, S4 state).

An absent "model" block keeps the historical command line and network
(4 residual layers, 64 residual/skip channels, S4 state 64) exactly.
"""
from __future__ import annotations

MODEL_DEFAULTS = dict(num_res_layers=4, res_channels=64, skip_channels=64, s4_dstate=64)
_LIMITS = dict(num_res_layers=64, res_channels=1024, skip_channels=1024, s4_dstate=1024)


def model_options(config):
    """Return the complete validated network size; defaults when the block is absent."""
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a JSON object")
    configured = config.get("model", {})
    if not isinstance(configured, dict) or set(configured) - set(MODEL_DEFAULTS):
        raise ValueError("model accepts only " + ", ".join(MODEL_DEFAULTS))
    options = {**MODEL_DEFAULTS, **configured}
    for key, value in options.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _LIMITS[key]:
            raise ValueError(f"model.{key} must be an integer in [1, {_LIMITS[key]}]")
    return options


def model_cli_arguments(config):
    """Training CLI flags; empty when the block is absent (historical command)."""
    if "model" not in config:
        return []
    options = model_options(config)
    arguments = []
    for key in MODEL_DEFAULTS:
        arguments += ["-" + key, str(options[key])]
    return arguments
