"""Heart-rate-variability diagnostics from 25 Hz IBI series (numpy only).

The beat sequence is read the way a 25 Hz sample-and-hold IBI series encodes
it: take the value v at the current sample as one beat, then jump forward
round(v * fs) samples to the next beat. Values below ``missing_threshold``
(including the stored missing marker 0) are not beats; they break the
sequence, and successive differences are never taken across a break.

The same reader is applied to real and generated windows, so both sides are
summarized identically. This is a diagnostic only: it does not change any
waveform that is saved or returned.

Per window: mean HR (bpm), SDNN (ms), RMSSD (ms), pNN50 (%). Windows with fewer
than ``min_beats`` beats (or fewer than two successive differences for
RMSSD/pNN50) report None for the affected values.
"""
from __future__ import annotations

import numpy as np

FS = 25
DEFAULT_MISSING_THRESHOLD = 0.3   # seconds; 200 bpm
DEFAULT_MIN_BEATS = 5


def read_beats(series, fs=FS, missing_threshold=DEFAULT_MISSING_THRESHOLD):
    """Return (beats_seconds, segment_ids) read from one 1-D 25 Hz series."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("read_beats expects one 1-D series")
    beats, segments = [], []
    segment, index = 0, 0
    while index < values.size:
        value = values[index]
        if not np.isfinite(value) or value < missing_threshold:
            segment += 1
            index += 1
            continue
        beats.append(value)
        segments.append(segment)
        index += max(1, int(round(value * fs)))
    return np.asarray(beats, dtype=np.float64), np.asarray(segments, dtype=np.int64)


def window_hrv(series, fs=FS, missing_threshold=DEFAULT_MISSING_THRESHOLD, min_beats=DEFAULT_MIN_BEATS):
    beats, segments = read_beats(series, fs, missing_threshold)
    result = dict(beats=int(beats.size), mean_hr_bpm=None, sdnn_ms=None, rmssd_ms=None, pnn50_percent=None)
    if beats.size < min_beats:
        return result
    result["mean_hr_bpm"] = float(60.0 / beats.mean())
    result["sdnn_ms"] = float(beats.std() * 1000.0)
    same_segment = segments[1:] == segments[:-1]
    differences = np.diff(beats)[same_segment] * 1000.0
    if differences.size >= 2:
        result["rmssd_ms"] = float(np.sqrt(np.mean(differences ** 2)))
        result["pnn50_percent"] = float(np.mean(np.abs(differences) > 50.0) * 100.0)
    return result


def _median(rows, key):
    values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
    return float(np.median(values)) if values.size else None


def summarize_hrv(generated_seconds, reference_seconds, fs=FS, missing_threshold=DEFAULT_MISSING_THRESHOLD,
                  min_beats=DEFAULT_MIN_BEATS):
    """Median-over-windows HRV for [windows, samples] real and generated arrays."""
    generated = np.asarray(generated_seconds, dtype=np.float64)
    reference = np.asarray(reference_seconds, dtype=np.float64)
    if generated.ndim != 2 or generated.shape != reference.shape:
        raise ValueError("Expected matching [windows, samples] arrays")
    real_rows = [window_hrv(row, fs, missing_threshold, min_beats) for row in reference]
    gen_rows = [window_hrv(row, fs, missing_threshold, min_beats) for row in generated]
    keys = ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_percent")
    summary = dict(windows=int(generated.shape[0]),
                   real={key: _median(real_rows, key) for key in keys},
                   generated={key: _median(gen_rows, key) for key in keys},
                   real_beats_median=_median(real_rows, "beats"),
                   generated_beats_median=_median(gen_rows, "beats"),
                   definition="Medians over windows; beats read as value-then-jump round(value * fs); "
                              f"values below {missing_threshold} s break the beat sequence; diagnostics only")
    summary["generated_minus_real"] = {
        key: (None if summary["real"][key] is None or summary["generated"][key] is None
              else summary["generated"][key] - summary["real"][key]) for key in keys}
    return summary
