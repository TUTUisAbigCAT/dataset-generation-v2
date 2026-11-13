"""Evaluate a checkpoint on independent 2-minute windows of any configured participants.

For each requested participant the script selects windows (default: every
120 s, i.e. non-overlapping), generates each window independently from that
participant's sleep-stage and demographic conditions, and compares the result
with the real IBI at the same reference-observed positions.

Outputs (in --output-dir):
  subject_<SID>/inputs.npz, waveforms.npz   plot with scripts/plot_s092_all_windows_kde.py --folder subject_<SID>
  subject_<SID>/per_window_metrics.csv      per-window Gaussian KL and moments
  subject_<SID>/summary.json                pooled W1/KS, per-stage W1, window-KL quantiles, HRV, structure
  subjects_summary.csv                      one row per participant
  summary.json                              run provenance

The configuration must match the checkpoint's training setup (window length,
diffusion schedule, model size, normalization); its training split is used to
reconstruct the scaler. Sampling requires CUDA; --check-only does not.

Example:
  python scripts/evaluate_subject_windows.py --checkpoint RUN/training/best_kl_model.pth \
      --config RUN/workspace/dataset_config.json --subjects S092 S002 S010 --output-dir OUT
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRIDE_CHOICES = (30, 60, 90, 120)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True, help="Participant IDs, for example S092 S002")
    parser.add_argument("--window-stride-seconds", type=int, choices=STRIDE_CHOICES, default=120)
    parser.add_argument("--max-windows", type=int, default=None, help="Optional evenly spaced subset per participant")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate data and window selection without a model or GPU")
    cli = parser.parse_args(argv)
    if cli.batch_size < 1 or cli.threads < 1 or (cli.max_windows is not None and cli.max_windows < 1):
        parser.error("--batch-size, --threads and --max-windows must be positive")
    if not 0 <= cli.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if len({s.casefold() for s in cli.subjects}) != len(cli.subjects):
        parser.error("--subjects contains duplicates")
    return cli


SUMMARY_FIELDS = ("subject", "role", "windows", "pooled_w1_ms", "pooled_ks", "mean_bias_ms", "std_ratio",
                  "window_kl_mean", "window_kl_median", "window_kl_p90", "window_kl_share_above_1",
                  "real_rmssd_ms", "generated_rmssd_ms", "real_sdnn_ms", "generated_sdnn_ms",
                  "real_mean_hr_bpm", "generated_mean_hr_bpm", "generated_nonpositive_fraction_all")


def summary_row(subject, role, summary):
    pooled, kl, hrv = summary["pooled_distribution"], summary["window_kl"], summary["hrv"]
    return dict(subject=subject, role=role, windows=summary["windows"], pooled_w1_ms=pooled["pooled_w1_ms"],
                pooled_ks=pooled["pooled_ks"], mean_bias_ms=pooled["mean_bias_ms"], std_ratio=pooled["std_ratio"],
                window_kl_mean=kl["mean"], window_kl_median=kl["median"], window_kl_p90=kl["p90"],
                window_kl_share_above_1=kl["share_above_1"],
                real_rmssd_ms=hrv["real"]["rmssd_ms"], generated_rmssd_ms=hrv["generated"]["rmssd_ms"],
                real_sdnn_ms=hrv["real"]["sdnn_ms"], generated_sdnn_ms=hrv["generated"]["sdnn_ms"],
                real_mean_hr_bpm=hrv["real"]["mean_hr_bpm"], generated_mean_hr_bpm=hrv["generated"]["mean_hr_bpm"],
                generated_nonpositive_fraction_all=summary["generated_nonpositive_fraction_all"])


def run(cli, *, sampler=None, model_loader=None, device=None):
    """Testable body. sampler/model_loader/device default to the CUDA production path."""
    from scripts.evaluate_denoising import save_json, sha256
    from night_checkpoint import load_inference_config
    from data_utils import Preprocessor
    from diffusion_schedule import diffusion_identity
    from night_data import prepare_night
    import subject_windows as sw

    output = cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="dreamt_subject_window_evaluation_v1", status="starting", check_only=cli.check_only,
                  requested_subjects=list(cli.subjects), seed=cli.seed,
                  window_stride_seconds=cli.window_stride_seconds, max_windows=cli.max_windows,
                  reference_ibi_passed_to_sampler=False, output_dir=str(output))
    save_json(output / "summary.json", report)
    try:
        if device is None and not cli.check_only:
            from cuda_runtime import require_cuda
            device = require_cuda()
        checkpoint = cli.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint file does not exist: " + str(checkpoint))
        args, data_config, config_provenance = load_inference_config(cli.config)
        preprocessor = Preprocessor("DREAMT", data_config=data_config)
        sw.check_preprocessor_columns(preprocessor)
        stride = cli.window_stride_seconds * 25
        plans = []
        for subject in cli.subjects:
            prepared = prepare_night(preprocessor, subject, allow_time_gaps=True)
            starts = sw.select_window_starts(prepared, args.window_size, stride, max_windows=cli.max_windows)
            plans.append((prepared, starts))
        report.update(checkpoint=str(checkpoint), checkpoint_preflight_sha256=sha256(checkpoint),
                      config_provenance=config_provenance, model_args=vars(args),
                      diffusion_identity=diffusion_identity(args), scaler=preprocessor.scaler,
                      subjects={p["subject_id"]: dict(role=p["provenance"]["subject_role"], windows=int(len(s)),
                                                      in_scaler_training_split=p["provenance"]["subject_in_original_training_scaler"])
                                for p, s in plans})
        if cli.check_only:
            report.update(status="checked_only", model_constructed=False, sampling_performed=False)
            save_json(output / "summary.json", report)
            print(json.dumps(dict(status=report["status"], subjects=report["subjects"])), flush=True)
            return report

        import torch
        from diffusion_schedule import build_diffusion_config
        torch.set_num_threads(cli.threads)
        loader = sw.load_frozen_model if model_loader is None else model_loader
        model, checkpoint_identity = loader(args, checkpoint, device)
        schedule = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in build_diffusion_config(args).items()}
        rows_out = []
        for prepared, starts in plans:
            sid = prepared["subject_id"]
            reference, mask, stage_samples, epoch_stages = sw.window_arrays(prepared, starts, args.window_size)
            clean = sw.build_clean_windows(prepared["conditions"], starts, args.window_size)
            standardized = sw.sample_windows(model, clean, schedule, args, device, batch_size=cli.batch_size,
                                             seed=cli.seed, sampler=sampler)
            generated = sw.to_seconds(standardized, preprocessor.scaler)
            rows, summary = sw.evaluate_windows(generated, reference, mask, stage_samples)
            sw.save_window_outputs(output / f"subject_{sid}", reference=reference, mask=mask,
                                   epoch_stages=epoch_stages, starts=starts, generated=generated,
                                   conditions=clean[:, :, 1:], seed=cli.seed, rows=rows, summary=summary,
                                   extra=dict(subject=sid, role=prepared["provenance"]["subject_role"],
                                              checkpoint=checkpoint_identity))
            rows_out.append(summary_row(sid, prepared["provenance"]["subject_role"], summary))
            print(json.dumps(dict(event="subject_evaluated", subject=sid, windows=int(len(starts)),
                                  pooled_w1_ms=summary["pooled_distribution"]["pooled_w1_ms"])), flush=True)
        with (output / "subjects_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows_out)
        report.update(status="completed", model_constructed=True, sampling_performed=True,
                      checkpoint_identity=checkpoint_identity, results=rows_out)
        save_json(output / "summary.json", report)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save_json(output / "summary.json", report)
        raise


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
