"""Opt-in learning-rate schedule and gradient clipping for the training loop.

Configuration block "training_schedule" (all optional):
    warmup_steps   linear warm-up length in optimizer updates (default 0)
    decay          "none" (default) or "cosine" after warm-up
    min_lr_ratio   final learning rate / base learning rate for cosine (default 0.0)
    clip_grad_norm maximum total gradient L2 norm, or null (default: no clipping)

With the defaults the schedule is inactive: before_step() and after_step() do
nothing, the optimizer's learning rate is never written, and training stays
bitwise identical. No random numbers are drawn.

Learning rate at update t (0-based), base rate b, warm-up W, total updates N:
    t < W:  b * (t + 1) / W
    else:   b                                                    (decay="none")
            b * (r + (1 - r) * 0.5 * (1 + cos(pi * (t - W) / max(1, N - W))))  (cosine)
"""
from __future__ import annotations

import json
import math
import os
from numbers import Real
from pathlib import Path

DEFAULTS = dict(warmup_steps=0, decay="none", min_lr_ratio=0.0, clip_grad_norm=None)


def schedule_options(config):
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a JSON object")
    configured = config.get("training_schedule", {})
    if not isinstance(configured, dict) or set(configured) - set(DEFAULTS):
        raise ValueError("training_schedule accepts only " + ", ".join(DEFAULTS))
    options = {**DEFAULTS, **configured}
    warmup = options["warmup_steps"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("training_schedule.warmup_steps must be a nonnegative integer")
    if options["decay"] not in ("none", "cosine"):
        raise ValueError("training_schedule.decay must be none or cosine")
    ratio = options["min_lr_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("training_schedule.min_lr_ratio must lie in [0, 1]")
    options["min_lr_ratio"] = float(ratio)
    clip = options["clip_grad_norm"]
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, Real) or not math.isfinite(clip) or clip <= 0:
            raise ValueError("training_schedule.clip_grad_norm must be positive or null")
        options["clip_grad_norm"] = float(clip)
    return options


class TrainingSchedule:
    def __init__(self, optimizer, model, steps_per_epoch, epochs, options):
        self.options = schedule_options({"training_schedule": options})
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be a positive integer")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        self.optimizer, self.model = optimizer, model
        self.total_steps = steps_per_epoch * epochs
        self.lr_active = self.options["warmup_steps"] > 0 or self.options["decay"] != "none"
        self.clip_active = self.options["clip_grad_norm"] is not None
        self.updates = 0
        self.clipped_updates = 0
        self.max_grad_norm_seen = None
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        if self.lr_active:
            self._apply(0)

    @property
    def active(self):
        return self.lr_active or self.clip_active

    def factor(self, step):
        warmup = self.options["warmup_steps"]
        if step < warmup:
            return (step + 1) / warmup
        if self.options["decay"] == "none":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, self.total_steps - warmup))
        ratio = self.options["min_lr_ratio"]
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self, step):
        factor = self.factor(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def current_lrs(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def before_step(self):
        """Clip gradients (if configured) after backward and before optimizer.step()."""
        if not self.clip_active:
            return
        import torch

        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options["clip_grad_norm"]))
        if norm > self.options["clip_grad_norm"]:
            self.clipped_updates += 1
        if math.isfinite(norm):
            self.max_grad_norm_seen = norm if self.max_grad_norm_seen is None else max(self.max_grad_norm_seen, norm)

    def after_step(self):
        """Advance to the learning rate of the next update."""
        if not self.active:
            return
        self.updates += 1
        if self.lr_active:
            self._apply(self.updates)

    def report(self):
        return dict(options=dict(self.options), active=self.active, lr_schedule_active=self.lr_active,
                    gradient_clipping_active=self.clip_active, updates=self.updates,
                    planned_updates=self.total_steps, base_learning_rates=list(self.base_lrs),
                    final_learning_rates=self.current_lrs(), clipped_updates=self.clipped_updates,
                    max_grad_norm_seen=self.max_grad_norm_seen, random_streams_consumed=False)


def create_training_schedule(optimizer, model, steps_per_epoch, epochs):
    """Read the same selected JSON as Preprocessor; no independent overrides."""
    selected = os.environ.get("DREAMT_CONFIG_PATH")
    path = (Path(selected).expanduser().resolve() if selected
            else Path(__file__).resolve().with_name("dataset_config.json"))
    if selected and not path.is_file():
        raise FileNotFoundError(f"Selected DREAMT configuration does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    return TrainingSchedule(optimizer, model, steps_per_epoch, epochs, schedule_options(config))
