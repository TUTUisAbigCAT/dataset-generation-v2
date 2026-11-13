"""Opt-in exponential moving average (EMA) of model parameters.

The EMA copy is used only for epoch validation, the checkpoints written by
the validator (best_kl_model.pth and optional per-epoch model.pth) and the
additional saved_models/<dataset>/model_ema.pth written after training.
Training, the optimizer, the loss, the forward-noising draws and the final
saved_models/<dataset>/model.pth keep using the live training model.

decay=0 (the default) creates no copy and performs no work, so legacy runs stay
bitwise identical. Parameters are averaged; buffers are copied from the live
model (S4 buffers are fixed at construction, so state_dict keys and buffer
values remain identical to a seed-42 reconstruction).
"""
from __future__ import annotations

import copy
import json
import math
import os
from numbers import Real
from pathlib import Path


def ema_options(config):
    """Canonical configuration; an absent block keeps exact inactive behavior."""
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a JSON object")
    configured = config.get("training_ema", {})
    if not isinstance(configured, dict) or set(configured) - {"decay"}:
        raise ValueError("training_ema accepts only decay")
    decay = configured.get("decay", 0.0)
    if (isinstance(decay, bool) or not isinstance(decay, Real)
            or not math.isfinite(decay) or not 0.0 <= decay < 1.0):
        raise ValueError("training_ema.decay must be finite and within [0, 1)")
    return dict(decay=float(decay))


class TrainingEMA:
    """Parameter EMA with the standard (1 + n) / (10 + n) warm-up."""

    def __init__(self, model, decay):
        self.decay = ema_options({"training_ema": {"decay": decay}})["decay"]
        self.updates = 0
        self.model = None
        if self.decay == 0.0:
            return
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)

    @property
    def active(self):
        return self.model is not None

    def update(self, model):
        if self.model is None:
            return
        import torch

        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        with torch.no_grad():
            ema_parameters = list(self.model.parameters())
            live_parameters = list(model.parameters())
            if len(ema_parameters) != len(live_parameters):
                raise RuntimeError("EMA and training model parameter lists differ")
            for ema_param, param in zip(ema_parameters, live_parameters):
                ema_param.lerp_(param.detach(), 1.0 - decay)
            ema_buffers = list(self.model.buffers())
            live_buffers = list(model.buffers())
            if len(ema_buffers) != len(live_buffers):
                raise RuntimeError("EMA and training model buffer lists differ")
            for ema_buffer, buffer in zip(ema_buffers, live_buffers):
                ema_buffer.copy_(buffer)

    def evaluation_model(self, model):
        """Return the EMA model when active, otherwise the live model itself."""
        return model if self.model is None else self.model

    def save_evaluation_weights(self, model, path):
        """Write the evaluation weights (EMA when active, else the live model) atomically.

        Called once after training. With decay=0 the file equals model.pth.
        """
        import hashlib
        import torch

        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.evaluation_model(model).state_dict(), temporary)
        temporary.replace(path)
        self.saved_evaluation_weights = dict(path=str(path), ema_active=self.active,
                                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return self.saved_evaluation_weights

    def report(self):
        return dict(decay=self.decay, active=self.active, updates=self.updates,
                    warmup="effective decay = min(decay, (1 + n) / (10 + n)) at update n",
                    averaged="model.parameters()", buffers="copied from the live training model after every update",
                    used_for="epoch validation and validator checkpoints (best_kl_model.pth, epoch_NNNN/model.pth)",
                    final_saved_model="saved_models/<dataset>/model.pth remains the live training weights",
                    final_evaluation_weights=getattr(self, "saved_evaluation_weights", None),
                    random_streams_consumed=False)


def create_training_ema(model):
    """Read the same selected JSON as Preprocessor; no independent overrides."""
    selected = os.environ.get("DREAMT_CONFIG_PATH")
    path = (Path(selected).expanduser().resolve() if selected
            else Path(__file__).resolve().with_name("dataset_config.json"))
    if selected and not path.is_file():
        raise FileNotFoundError(f"Selected DREAMT configuration does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    return TrainingEMA(model, ema_options(config)["decay"])
