"""Pooled real-vs-generated IBI distribution metrics (numpy only, diagnostics).

All functions compare values at the same reference-observed positions, so the
real and generated samples always have equal counts. Values are in seconds;
reported distances use milliseconds where stated. Nothing here draws random
numbers, clips generated values or touches the training state.

Definitions
-----------
* pooled W1: Wasserstein-1 distance between the pooled real and generated
  value distributions. With equal sample counts this is the mean absolute
  difference between the sorted samples.
* pooled KS: largest absolute difference between the two empirical CDFs.
* per-stage W1: the same W1 restricted to positions whose 30-second epoch has
  that sleep-stage label (W, Light, REM, Deep).
* window KL summary: median, 90th percentile, share above 1 and mean of the
  per-window Gaussian KL values that validation already computes.
"""
from __future__ import annotations

import math

import numpy as np

STAGE_NAMES = ("W", "Light", "REM", "Deep")


def _as_pairs(generated_seconds, reference_seconds, observed_mask=None):
    generated = np.asarray(generated_seconds, dtype=np.float64)
    reference = np.asarray(reference_seconds, dtype=np.float64)
    if generated.shape != reference.shape or generated.ndim not in (1, 2):
        raise ValueError("generated and reference must have the same 1-D or [windows, samples] shape")
    if observed_mask is None:
        mask = np.ones(generated.shape, dtype=bool)
    else:
        mask = np.asarray(observed_mask)
        if mask.dtype != bool or mask.shape != generated.shape:
            raise ValueError("observed_mask must be a bool array matching the waveforms")
    if not np.isfinite(generated).all():
        raise ValueError("Generated values must be finite everywhere")
    if not np.isfinite(reference[mask]).all():
        raise ValueError("Reference values must be finite at observed positions")
    return generated, reference, mask


def wasserstein1(a, b):
    """W1 between two equal-size samples (seconds in, seconds out)."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if a.size != b.size or a.size == 0:
        raise ValueError("wasserstein1 expects two nonempty samples of equal size")
    return float(np.mean(np.abs(a - b)))


def wasserstein1_unequal(a, b, quantiles=2001):
    """W1 between samples of any sizes, from their quantile functions (seconds)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("wasserstein1_unequal expects two nonempty samples")
    if a.size == b.size:
        return wasserstein1(a, b)
    levels = (np.arange(quantiles) + 0.5) / quantiles
    return float(np.mean(np.abs(np.quantile(a, levels) - np.quantile(b, levels))))


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (largest empirical CDF gap)."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        raise ValueError("ks_statistic expects two nonempty samples")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def stage_per_sample(stage_onehot):
    """[..., 4] one-hot stage channels -> integer stage index per sample."""
    onehot = np.asarray(stage_onehot)
    if onehot.shape[-1] != 4:
        raise ValueError("Expected four sleep-stage one-hot channels")
    if not np.allclose(onehot.sum(axis=-1), 1.0):
        raise ValueError("Each sample needs exactly one active sleep-stage channel")
    return onehot.argmax(axis=-1).astype(np.int64)


def pooled_distribution_metrics(generated_seconds, reference_seconds, observed_mask=None, stages=None):
    """Pooled W1/KS, mean bias, spread ratio and optional per-stage W1."""
    generated, reference, mask = _as_pairs(generated_seconds, reference_seconds, observed_mask)
    real, gen = reference[mask], generated[mask]
    if real.size < 2:
        raise ValueError("Need at least two observed positions")
    real_std = float(real.std())
    result = dict(
        samples=int(real.size),
        pooled_w1_ms=wasserstein1(real, gen) * 1000.0,
        pooled_ks=ks_statistic(real, gen),
        real_mean_s=float(real.mean()), generated_mean_s=float(gen.mean()),
        mean_bias_ms=float(gen.mean() - real.mean()) * 1000.0,
        real_std_s=real_std, generated_std_s=float(gen.std()),
        std_ratio=float(gen.std() / real_std) if real_std > 0 else None,
        generated_nonpositive_fraction_observed=float((gen <= 0).mean()),
        definition="Same reference-observed positions for real and generated; W1 in ms; KS dimensionless; "
                   "generated values are not clipped",
    )
    if stages is not None:
        stage_index = np.asarray(stages, dtype=np.int64)
        if stage_index.shape != generated.shape:
            raise ValueError("stages must give one stage index per sample")
        per_stage = {}
        for code, name in enumerate(STAGE_NAMES):
            selected = mask & (stage_index == code)
            count = int(selected.sum())
            if count < 2:
                per_stage[name] = dict(samples=count)
                continue
            r, g = reference[selected], generated[selected]
            per_stage[name] = dict(samples=count, w1_ms=wasserstein1(r, g) * 1000.0,
                                   ks=ks_statistic(r, g),
                                   real_mean_s=float(r.mean()), generated_mean_s=float(g.mean()))
        result["per_stage"] = per_stage
    return result


def window_kl_summary(window_kl):
    values = np.asarray(window_kl, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Window KL values must be nonempty and finite")
    return dict(windows=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), share_above_1=float((values > 1.0).mean()))


def composite_score(kl, w1_ms, ks, floor=1e-12):
    """Geometric mean of KL, W1 (seconds) and KS.

    Ranking epochs by this score is independent of the units chosen for each
    term, because rescaling any term multiplies every epoch's score equally.
    """
    terms = (float(kl), float(w1_ms) / 1000.0, float(ks))
    if any(not math.isfinite(term) or term < 0 for term in terms):
        raise ValueError("Composite terms must be finite and nonnegative")
    return float(math.exp(sum(math.log(max(term, floor)) for term in terms) / 3.0))
