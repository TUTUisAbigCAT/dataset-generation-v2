"""Diagnostics for the temporal structure of 25 Hz IBI and paired distribution distance.

Real DREAMT IBI at 25 Hz is sample-and-hold: each beat interval is held for about
its own duration (S092: ~97% of consecutive samples are exactly equal and the
plateau duration matches the plateau value to ~0.01 s). The per-window Gaussian KL
cannot see this, so these numpy-only diagnostics are reported next to it. They
are diagnostics only and never select checkpoints.
"""
from __future__ import annotations

import numpy as np

FS = 25


def step_structure(x, fs=FS, tol=1e-6):
    """Plateau diagnostics for one IBI window in seconds.

    hold_fraction: share of consecutive samples that are equal (within tol).
    median_abs_duration_minus_value_seconds: for interior plateaus with a
    positive value, median |plateau duration - plateau value|. Edge plateaus are
    truncated by the window and are excluded.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("step_structure expects one finite window with at least two samples")
    d = np.abs(np.diff(x))
    edges = np.flatnonzero(d > tol) + 1
    bounds = np.concatenate(([0], edges, [len(x)]))
    seconds = np.diff(bounds) / fs
    values = x[bounds[:-1]]
    inner_s, inner_v = seconds[1:-1], values[1:-1]
    keep = inner_v > 0
    mismatch = float(np.median(np.abs(inner_s[keep] - inner_v[keep]))) if keep.any() else float("nan")
    return dict(hold_fraction=float((d <= tol).mean()), plateaus=int(len(values)),
                median_plateau_seconds=float(np.median(seconds)),
                median_abs_duration_minus_value_seconds=mismatch)


def w1_paired(reference, generated, observed=None):
    """Wasserstein-1 between the two value distributions at the same positions (seconds)."""
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if reference.shape != generated.shape or reference.ndim != 1:
        raise ValueError("w1_paired expects two equal-length one-dimensional windows")
    if observed is None:
        observed = np.ones(reference.shape, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != reference.shape or not observed.any():
        raise ValueError("w1_paired needs a matching mask with at least one observed position")
    r = np.sort(reference[observed])
    g = np.sort(generated[observed])
    return float(np.mean(np.abs(r - g)))


def summarize_windows(generated_seconds, reference_seconds, observed_mask=None, fs=FS):
    """Median-over-windows structure and W1 diagnostics for [windows, samples] arrays."""
    generated = np.asarray(generated_seconds, dtype=np.float64)
    reference = np.asarray(reference_seconds, dtype=np.float64)
    if generated.ndim != 2 or generated.shape != reference.shape:
        raise ValueError("Expected matching [windows, samples] generated and reference arrays")
    masks = (np.ones(generated.shape, dtype=bool) if observed_mask is None
             else np.asarray(observed_mask, dtype=bool))
    if masks.shape != generated.shape:
        raise ValueError("observed_mask must match [windows, samples]")
    gen = [step_structure(row, fs=fs) for row in generated]
    ref = [step_structure(np.where(np.isfinite(row), row, 0.0), fs=fs) for row in reference]
    w1 = [w1_paired(r, g, m) for r, g, m in zip(reference, generated, masks)]

    def median(rows, key):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else None

    return dict(
        windows=int(generated.shape[0]),
        w1_ms_median=float(np.median(w1)) * 1000.0,
        generated_hold_fraction_median=median(gen, "hold_fraction"),
        reference_hold_fraction_median=median(ref, "hold_fraction"),
        generated_plateau_duration_minus_value_seconds_median=median(gen, "median_abs_duration_minus_value_seconds"),
        reference_plateau_duration_minus_value_seconds_median=median(ref, "median_abs_duration_minus_value_seconds"),
        definition="Medians over windows. W1 on reference-observed positions; hold fraction = share of equal consecutive 25 Hz samples.",
        used_for_checkpoint_selection=False)


def to_sample_and_hold(generated, fs=FS, missing_threshold=0.2):
    """Optional post-processing: hold each generated beat value for its own duration.

    Values below missing_threshold become the missing marker 0 for one sample.
    This rebuilds the sample-and-hold structure only; it does not correct the
    model's value distribution. Never applied implicitly.
    """
    g = np.asarray(generated, dtype=np.float64)
    if g.ndim != 1 or not np.isfinite(g).all():
        raise ValueError("to_sample_and_hold expects one finite 1-D series in seconds")
    if not 0.0 < missing_threshold <= 0.3:
        raise ValueError("missing_threshold must lie in (0, 0.3] seconds")
    out = np.empty_like(g)
    i = 0
    while i < len(g):
        v = g[i]
        if v < missing_threshold:
            out[i] = 0.0
            i += 1
            continue
        n = max(1, int(round(v * fs)))
        out[i:i + n] = v
        i += n
    return out
