"""Independent 2-minute windows from any configured participant (evaluation only).

Shared by scripts/evaluate_subject_windows.py and scripts/check_condition_effect.py.
The training data, scaler and partitions are not changed: conditions come from
night_data.prepare_night with the configured training scaler, and reference
IBI never enters the sampler. Each window is generated independently with the
same corrected sampler used by epoch validation.

Window rule: windows start every ``stride`` samples (a whole number of 30-second
epochs) from the first stored epoch. A window is used when all of its epochs
are structurally valid, it does not cross a known source-time gap, and it has
at least two reference-observed (IBI > 0) samples.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from window_config import EPOCH_SAMPLES, FS

CONDITION_COLUMNS = ("stage_W", "stage_Light", "stage_REM", "stage_Deep",
                     "AHI", "BMI", "AGE", "gender_F", "gender_M")
EXPECTED_COLUMNS = ("IBI",) + CONDITION_COLUMNS


def check_preprocessor_columns(preprocessor):
    if (tuple(preprocessor.columns) != EXPECTED_COLUMNS
            or list(preprocessor.hierarchical_features_cyclic) != list(EXPECTED_COLUMNS[1:])):
        raise ValueError("Expected one IBI channel and the recorded nine DREAMT conditions")


def select_window_starts(prepared, window_size, stride_samples, *, min_observed=2, max_windows=None):
    if window_size % EPOCH_SAMPLES or stride_samples % EPOCH_SAMPLES or stride_samples < EPOCH_SAMPLES:
        raise ValueError("Window size and stride must be whole numbers of 30-second epochs")
    record = prepared["record"]
    reference = np.asarray(prepared["reference_seconds"], dtype=np.float64)
    valid = np.asarray(record["valid_epochs"], dtype=bool)
    adjacent = np.asarray(record["adjacent"], dtype=bool)
    epochs = window_size // EPOCH_SAMPLES
    starts = []
    for start in range(0, len(reference) - window_size + 1, stride_samples):
        first = start // EPOCH_SAMPLES
        if not valid[first:first + epochs].all():
            continue
        if epochs > 1 and not adjacent[first:first + epochs - 1].all():
            continue
        segment = reference[start:start + window_size]
        if np.count_nonzero(np.isfinite(segment) & (segment > 0)) < min_observed:
            continue
        starts.append(start)
    starts = np.asarray(starts, dtype=np.int64)
    if max_windows is not None and len(starts) > max_windows:
        keep = np.unique(np.linspace(0, len(starts) - 1, max_windows).round().astype(np.int64))
        starts = starts[keep]
    if not len(starts):
        raise ValueError(f"No usable windows for {prepared.get('subject_id')}")
    return starts


def window_arrays(prepared, starts, window_size):
    """Reference seconds, observed mask, per-sample stage index and per-window epoch stages."""
    reference = np.asarray(prepared["reference_seconds"], dtype=np.float64)
    stage_epochs = np.asarray(prepared["stage_epochs_four"], dtype=np.int64)
    epochs = window_size // EPOCH_SAMPLES
    ref = np.stack([reference[s:s + window_size] for s in starts])
    mask = np.isfinite(ref) & (ref > 0)
    epoch_stages = np.stack([stage_epochs[s // EPOCH_SAMPLES:s // EPOCH_SAMPLES + epochs] for s in starts])
    stage_samples = np.repeat(epoch_stages, EPOCH_SAMPLES, axis=1)
    return ref, mask, stage_samples, epoch_stages


def build_clean_windows(conditions, starts, window_size):
    """[windows, window_size, 10] float32: IBI channel unknown (0), nine clean conditions."""
    conditions = np.asarray(conditions, dtype=np.float32)
    windows = np.zeros((len(starts), window_size, 10), dtype=np.float32)
    for index, start in enumerate(starts):
        windows[index, :, 1:] = conditions[start:start + window_size]
    return windows


def replace_stages(clean_windows, stage_samples):
    """Copy with the four stage one-hot channels set from [windows, samples] stage indices."""
    result = np.array(clean_windows, copy=True)
    result[:, :, 1:5] = np.eye(4, dtype=np.float32)[np.asarray(stage_samples, dtype=np.int64)]
    return result


def replace_demographics(clean_windows, demographics):
    """Copy with the five demographic channels (AHI, BMI, AGE z-scores, gender F/M) replaced."""
    values = np.asarray(demographics, dtype=np.float32)
    if values.shape != (5,):
        raise ValueError("demographics must hold five standardized values")
    result = np.array(clean_windows, copy=True)
    result[:, :, 5:10] = values
    return result


def subject_demographics(prepared):
    values = np.asarray(prepared["conditions"], dtype=np.float32)[:, 4:9]
    if not np.all(values == values[0]):
        raise ValueError("Participant demographics must be constant across the night")
    return values[0].copy()


def load_frozen_model(args, checkpoint, device):
    """Seed-42 construction, then strict restoration of every parameter and buffer."""
    import torch
    from cuda_runtime import assert_cuda_model
    from night_checkpoint import load_checkpoint_state
    from training_utils import fetchModel

    np.random.seed(42)
    torch.manual_seed(42)
    model = fetchModel(10, 1, args).to(device)
    assert_cuda_model(model)
    data = Path(checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    restoration = load_checkpoint_state(model, saved)
    model.eval()
    return model, dict(path=str(Path(checkpoint).resolve()), sha256=hashlib.sha256(data).hexdigest(),
                       restoration=restoration)


def default_sampler(model, clean, schedule, args, device):
    from validation_sampling import sample_validation_windows
    return sample_validation_windows(model, clean, schedule, args, device)


def sample_windows(model, clean_windows, schedule, args, device, *, batch_size, seed, sampler=None):
    """Standardized generated IBI [windows, samples] (float64 numpy); one seed for the whole set."""
    import torch

    sampler = default_sampler if sampler is None else sampler
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = []
    for start in range(0, len(clean_windows), batch_size):
        clean = torch.as_tensor(clean_windows[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.device(device):
            generated, _ = sampler(model, clean, schedule, args, device)
        parts.append(generated.detach().to(torch.float64).cpu().numpy())
    return np.concatenate(parts, axis=0)


def to_seconds(standardized, scaler):
    from signal_normalization import signal_affine_parameters
    center, scale = signal_affine_parameters(scaler)
    return np.asarray(standardized, dtype=np.float64) * scale + center


def evaluate_windows(generated_seconds, reference_seconds, mask, stage_samples, *, kl_variance_floor_ms2=1.0):
    """Per-window rows and pooled summaries for one set of windows."""
    import torch
    from distribution_metrics import pooled_distribution_metrics, window_kl_summary
    from hrv_metrics import summarize_hrv
    from ibi_structure_metrics import summarize_windows
    from validation_metrics import window_statistics

    statistics = window_statistics(torch.as_tensor(generated_seconds), torch.as_tensor(reference_seconds),
                                   observed_mask=torch.as_tensor(mask),
                                   kl_variance_floor_ms2=kl_variance_floor_ms2)
    statistics = {key: value.cpu().numpy() for key, value in statistics.items()}
    rows = []
    for index in range(len(generated_seconds)):
        row = {key: float(value[index]) if value.dtype.kind == "f" else int(value[index])
               for key, value in statistics.items()}
        row["generated_nonpositive_samples_all"] = int((generated_seconds[index] <= 0).sum())
        rows.append(row)
    summary = dict(
        windows=int(len(generated_seconds)),
        pooled_distribution=pooled_distribution_metrics(generated_seconds, reference_seconds, mask, stage_samples),
        window_kl=window_kl_summary(statistics["gaussian_kl_real_to_generated"]),
        hrv=summarize_hrv(generated_seconds, reference_seconds),
        ibi_structure=summarize_windows(generated_seconds, np.where(mask, reference_seconds, 0.0), mask),
        generated_nonpositive_fraction_all=float((np.asarray(generated_seconds) <= 0).mean()),
    )
    return rows, summary


def save_window_outputs(folder, *, reference, mask, epoch_stages, starts, generated, conditions, seed,
                        rows, summary, extra=None):
    """Write files readable by scripts/plot_s092_all_windows_kde.py (--folder) plus metrics."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    window_size = reference.shape[1]
    np.savez_compressed(folder / "inputs.npz", reference_ibi_seconds=reference.reshape(-1),
                        conditions=conditions.reshape(-1, conditions.shape[-1]),
                        scoring_mask=mask.reshape(-1), sleep_stages=epoch_stages.reshape(-1),
                        start_second=np.array(int(starts[0]) // FS), sample_rate_hz=np.array(FS),
                        window_size=np.array(window_size), start_samples=np.asarray(starts, dtype=np.int64))
    np.savez_compressed(folder / "waveforms.npz", generated_ibi_seconds=generated.reshape(-1),
                        seed=np.array(seed), window_size=np.array(window_size),
                        start_samples=np.asarray(starts, dtype=np.int64))
    with (folder / "per_window_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["window_index", "start_sample", "start_second", "stages"] + list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        names = ("W", "Light", "REM", "Deep")
        for index, (start, row) in enumerate(zip(starts, rows)):
            writer.writerow(dict(window_index=index, start_sample=int(start), start_second=int(start) / FS,
                                 stages="/".join(names[s] for s in epoch_stages[index]), **row))
    payload = dict(summary, seed=seed, **(extra or {}))
    (folder / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
