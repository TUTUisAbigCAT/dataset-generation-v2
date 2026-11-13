"""Check whether the conditions (sleep stage, demographics) are enough to guide generation.

Four checks; all use one frozen checkpoint and change only the conditions given
to the model. Every model check uses the same sampling seed for all variants,
so differences come from the conditions, not from the noise.

real-r2           No model. Over all configured participants, how much of the
                  variation in 2-minute window mean IBI is explained by sleep
                  stage and demographics (OLS R^2), compared with a participant
                  identity model. This is the ceiling for what these conditions
                  can explain.
stage-swap        Same windows of --subject generated with the stage channels set
                  to W, Light, REM or Deep throughout. Is the order of the four
                  generated means the same as the real per-stage means, and is the
                  spread between stages at least half the real spread?
shuffle           Same windows generated with the window stage sequences shuffled
                  among windows. Compare stage-aligned W1 (positions grouped by
                  their true stage) with the unshuffled generation. If it barely
                  changes, the model is not using the stage conditions. (Pooled W1
                  cannot detect this: shuffling keeps the overall stage mix.)
demographic-swap  Windows of each participant in --subjects generated with each
                  participant's demographics (AHI, BMI, AGE, gender). Matrix of
                  W1(generated with i's demographics, real j). Is the diagonal the
                  smallest entry in each column?

Suggested criteria (summary.json "decision"):
  stage-swap: order matches and spread ratio >= 0.5
  shuffle: W1(shuffled) / W1(true) >= 1.2
  demographic-swap: diagonal is the column minimum for a majority of participants
All evaluated model checks passing -> conditions judged sufficient.

Example:
  python scripts/check_condition_effect.py --checkpoint RUN/training/best_kl_model.pth \
      --config RUN/workspace/dataset_config.json --subject S092 --subjects S092 S002 S010 S020 \
      --output-dir OUT
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKS = ("real-r2", "stage-swap", "shuffle", "demographic-swap")
STAGES = ("W", "Light", "REM", "Deep")
THRESHOLDS = dict(stage_spread_ratio=0.5, shuffle_w1_ratio=1.2, demographic_majority=0.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", default="S092", help="Participant for stage-swap and shuffle")
    parser.add_argument("--subjects", nargs="+", default=None, help="Participants for demographic-swap (at least two)")
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    parser.add_argument("--window-stride-seconds", type=int, choices=(30, 60, 90, 120), default=120)
    parser.add_argument("--max-windows", type=int, default=60, help="Windows per participant for model checks")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and window selection without a model")
    cli = parser.parse_args(argv)
    model_checks = set(cli.checks) - {"real-r2"}
    if model_checks and not cli.check_only and cli.checkpoint is None:
        parser.error("--checkpoint is required for stage-swap, shuffle and demographic-swap")
    if "demographic-swap" in cli.checks and (cli.subjects is None or len(cli.subjects) < 2):
        parser.error("demographic-swap needs --subjects with at least two participants")
    if cli.max_windows < 1 or cli.batch_size < 1 or cli.threads < 1:
        parser.error("--max-windows, --batch-size and --threads must be positive")
    return cli


# --------------------------------------------------------------------------- real-r2
def _ols_r2(y, design):
    design = np.column_stack([np.ones(len(y)), design]) if design.size else np.ones((len(y), 1))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = ((y - y.mean()) ** 2).sum()
    return float(1.0 - (residual ** 2).sum() / total) if total > 0 else None


def real_r2(preprocessor, window_size, stride, prepare, sw):
    """Window-level OLS R^2 of mean observed IBI on stage composition, demographics and identity."""
    names = sorted(set(preprocessor.splits["train"]) | set(preprocessor.splits["val"]))
    rows, skipped = [], {}
    for name in names:
        try:
            prepared = prepare(preprocessor, Path(name).stem, allow_time_gaps=True)
            starts = sw.select_window_starts(prepared, window_size, stride)
        except ValueError as error:
            skipped[Path(name).stem] = str(error)
            continue
        reference, mask, stage_samples, _ = sw.window_arrays(prepared, starts, window_size)
        demographics = sw.subject_demographics(prepared)
        for index in range(len(starts)):
            observed = mask[index]
            stage_fraction = np.bincount(stage_samples[index][observed], minlength=4) / observed.sum()
            rows.append((prepared["subject_id"], float(reference[index][observed].mean()), stage_fraction,
                         demographics[:4], stage_samples[index][observed], reference[index][observed]))
    if len(rows) < 3:
        raise ValueError("Too few windows for the real-data R^2 check")
    subjects = sorted({row[0] for row in rows})
    y = np.array([row[1] for row in rows])
    stage = np.array([row[2][1:] for row in rows])                  # W is the reference category
    demo = np.array([row[3] for row in rows])                       # AHI, BMI, AGE z, gender_F
    identity = np.array([[row[0] == s for s in subjects[1:]] for row in rows], dtype=float)
    per_stage_values = {name: [] for name in STAGES}
    for row in rows:
        for code, name in enumerate(STAGES):
            per_stage_values[name].append(row[5][row[4] == code])
    per_stage = {}
    for name, parts in per_stage_values.items():
        values = np.concatenate(parts) if parts else np.array([])
        per_stage[name] = dict(samples=int(values.size),
                               mean_s=float(values.mean()) if values.size else None,
                               median_s=float(np.median(values)) if values.size else None)
    return dict(
        participants=len(subjects), windows=int(len(y)), skipped_participants=skipped,
        window_mean_ibi_sd_s=float(y.std()),
        r2=dict(stage=_ols_r2(y, stage), demographics=_ols_r2(y, demo),
                stage_and_demographics=_ols_r2(y, np.column_stack([stage, demo])),
                participant_identity=_ols_r2(y, identity),
                participant_identity_and_stage=_ols_r2(y, np.column_stack([identity, stage]))),
        real_per_stage_ibi=per_stage,
        interpretation="stage_and_demographics is the share of window-mean variation the model's conditions can "
                       "explain at best; the gap to participant_identity_and_stage is what these conditions miss")


# --------------------------------------------------------------------------- model checks
def _generate(context, clean):
    standardized = context["sw"].sample_windows(context["model"], clean, context["schedule"], context["args"],
                                                context["device"], batch_size=context["batch_size"],
                                                seed=context["seed"], sampler=context["sampler"])
    return context["sw"].to_seconds(standardized, context["scaler"])


def stage_swap(context, prepared, starts):
    from distribution_metrics import wasserstein1_unequal
    sw = context["sw"]
    window_size = context["args"].window_size
    reference, mask, stage_samples, _ = sw.window_arrays(prepared, starts, window_size)
    clean = sw.build_clean_windows(prepared["conditions"], starts, window_size)
    real_by_stage = {name: reference[mask & (stage_samples == code)] for code, name in enumerate(STAGES)}
    generated_by_stage = {}
    for code, name in enumerate(STAGES):
        generated = _generate(context, sw.replace_stages(clean, np.full(stage_samples.shape, code)))
        generated_by_stage[name] = generated[mask]
    present = [name for name in STAGES if real_by_stage[name].size >= 2]
    real_means = {name: float(real_by_stage[name].mean()) for name in present}
    generated_means = {name: float(generated_by_stage[name].mean()) for name in STAGES}
    order_real = sorted(present, key=lambda name: real_means[name])
    order_generated = sorted(present, key=lambda name: generated_means[name])
    real_spread = max(real_means.values()) - min(real_means.values()) if len(present) >= 2 else 0.0
    generated_spread = (max(generated_means[n] for n in present) - min(generated_means[n] for n in present)
                        if len(present) >= 2 else 0.0)
    ratio = generated_spread / real_spread if real_spread > 0 else None
    between = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            between[f"{a}-{b}"] = dict(
                real_w1_ms=wasserstein1_unequal(real_by_stage[a], real_by_stage[b]) * 1000,
                generated_w1_ms=wasserstein1_unequal(generated_by_stage[a], generated_by_stage[b]) * 1000)
    rank = {name: index for index, name in enumerate(order_real)}
    generated_rank = {name: index for index, name in enumerate(order_generated)}
    if len(present) >= 2:
        d = np.array([rank[n] - generated_rank[n] for n in present], dtype=float)
        k = len(present)
        spearman = float(1 - 6 * (d ** 2).sum() / (k * (k * k - 1)))
    else:
        spearman = None
    passed = (len(present) >= 2 and order_real == order_generated
              and ratio is not None and ratio >= THRESHOLDS["stage_spread_ratio"])
    return dict(windows=int(len(starts)), stages_present_in_real=present,
                real_stage_mean_s=real_means, generated_stage_mean_s=generated_means,
                real_order_low_to_high=order_real, generated_order_low_to_high=order_generated,
                order_matches=order_real == order_generated, spearman=spearman,
                real_spread_s=real_spread, generated_spread_s=generated_spread, spread_ratio=ratio,
                between_stage_w1=between, passed=bool(passed))


def _stage_aligned_w1_ms(reference, generated, mask, stage_samples):
    """Sample-weighted mean of per-stage W1, grouping positions by their TRUE stage label.

    Pooled W1 cannot detect shuffled stage conditions (shuffling preserves the
    pooled mix of stages); grouping by the true label at each position can.
    """
    from distribution_metrics import pooled_distribution_metrics
    per_stage = pooled_distribution_metrics(generated, reference, mask, stage_samples)["per_stage"]
    rows = [(value["samples"], value["w1_ms"]) for value in per_stage.values() if "w1_ms" in value]
    total = sum(count for count, _ in rows)
    return sum(count * w1 for count, w1 in rows) / total, per_stage


def shuffle(context, prepared, starts):
    from distribution_metrics import wasserstein1
    sw = context["sw"]
    window_size = context["args"].window_size
    reference, mask, stage_samples, _ = sw.window_arrays(prepared, starts, window_size)
    clean = sw.build_clean_windows(prepared["conditions"], starts, window_size)
    rng = np.random.default_rng(context["seed"])
    permutation = rng.permutation(len(starts))
    if len(starts) > 1:
        while np.array_equal(permutation, np.arange(len(starts))):
            permutation = rng.permutation(len(starts))
    true_generated = _generate(context, clean)
    shuffled_generated = _generate(context, sw.replace_stages(clean, stage_samples[permutation]))
    w1_true, per_stage_true = _stage_aligned_w1_ms(reference, true_generated, mask, stage_samples)
    w1_shuffled, per_stage_shuffled = _stage_aligned_w1_ms(reference, shuffled_generated, mask, stage_samples)
    changed = float((stage_samples[permutation] != stage_samples)[mask].mean())
    if w1_true > 0:
        ratio = w1_shuffled / w1_true
    else:
        ratio = float("inf") if w1_shuffled > 0 else 1.0
    return dict(windows=int(len(starts)), metric="stage-aligned W1 (positions grouped by true stage)",
                w1_true_ms=w1_true, w1_shuffled_ms=w1_shuffled, w1_ratio=ratio if np.isfinite(ratio) else None,
                w1_ratio_infinite=bool(not np.isfinite(ratio)),
                pooled_w1_true_ms=wasserstein1(reference[mask], true_generated[mask]) * 1000,
                pooled_w1_shuffled_ms=wasserstein1(reference[mask], shuffled_generated[mask]) * 1000,
                per_stage_true=per_stage_true, per_stage_shuffled=per_stage_shuffled,
                share_of_positions_with_changed_stage=changed,
                passed=bool(ratio >= THRESHOLDS["shuffle_w1_ratio"]))


def demographic_swap(context, plans):
    from distribution_metrics import wasserstein1
    sw = context["sw"]
    window_size = context["args"].window_size
    subjects = [prepared["subject_id"] for prepared, _ in plans]
    donors = [sw.subject_demographics(prepared) for prepared, _ in plans]
    matrix = np.zeros((len(plans), len(plans)))
    for j, (prepared, starts) in enumerate(plans):
        reference, mask, _, _ = sw.window_arrays(prepared, starts, window_size)
        clean = sw.build_clean_windows(prepared["conditions"], starts, window_size)
        for i, demographics in enumerate(donors):
            generated = _generate(context, sw.replace_demographics(clean, demographics))
            matrix[i, j] = wasserstein1(reference[mask], generated[mask]) * 1000
    diagonal_best = [bool(np.argmin(matrix[:, j]) == j) for j in range(len(plans))]
    share = float(np.mean(diagonal_best))
    return dict(subjects=subjects, w1_ms_matrix_rows_donor_columns_target=matrix.tolist(),
                diagonal_is_column_minimum=dict(zip(subjects, diagonal_best)), diagonal_share=share,
                passed=bool(share > THRESHOLDS["demographic_majority"]))


def _plot(output, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    panels = [key for key in ("stage_swap", "shuffle", "demographic_swap") if key in results]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)
    for ax, key in zip(axes[0], panels):
        result = results[key]
        if key == "stage_swap":
            x = np.arange(len(STAGES))
            ax.bar(x - 0.2, [result["real_stage_mean_s"].get(n, np.nan) for n in STAGES], 0.4, label="Real", color="#2a78d6")
            ax.bar(x + 0.2, [result["generated_stage_mean_s"][n] for n in STAGES], 0.4, label="Generated (stage set)", color="#eb6834")
            ax.set_xticks(x, STAGES)
            ax.set_ylabel("Mean IBI (s)")
            ax.set_title(f"Stage swap: order match {result['order_matches']}, spread ratio "
                         f"{result['spread_ratio'] if result['spread_ratio'] is None else round(result['spread_ratio'], 2)}",
                         fontsize=10, loc="left")
            ax.legend(frameon=False)
            low = min([v for v in result["real_stage_mean_s"].values()] + list(result["generated_stage_mean_s"].values()))
            ax.set_ylim(bottom=max(0, low - 0.2))
        elif key == "shuffle":
            ax.bar(["true stages", "shuffled stages"], [result["w1_true_ms"], result["w1_shuffled_ms"]], color=["#2a78d6", "#8a8a85"])
            ax.set_ylabel("Stage-aligned W1 to real (ms)")
            ratio_text = "inf" if result["w1_ratio_infinite"] else f"{result['w1_ratio']:.2f}"
            ax.set_title(f"Shuffle: ratio {ratio_text}", fontsize=10, loc="left")
        else:
            matrix = np.array(result["w1_ms_matrix_rows_donor_columns_target"])
            image = ax.imshow(matrix, cmap="viridis")
            ax.set_xticks(range(len(result["subjects"])), result["subjects"], rotation=45)
            ax.set_yticks(range(len(result["subjects"])), result["subjects"])
            ax.set_xlabel("Real participant (target)")
            ax.set_ylabel("Demographics used (donor)")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(image, ax=ax, label="W1 (ms)")
            ax.set_title(f"Demographic swap: diagonal best {result['diagonal_share']:.0%}", fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = Path(output) / "condition_checks.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def run(cli, *, sampler=None, model_loader=None, device=None):
    from scripts.evaluate_denoising import save_json
    from night_checkpoint import load_inference_config
    from data_utils import Preprocessor
    from night_data import prepare_night
    import subject_windows as sw

    output = cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="dreamt_condition_effect_v1", status="starting", checks=list(cli.checks),
                  seed=cli.seed, thresholds=THRESHOLDS, check_only=cli.check_only, output_dir=str(output))
    save_json(output / "summary.json", report)
    try:
        args, data_config, config_provenance = load_inference_config(cli.config)
        preprocessor = Preprocessor("DREAMT", data_config=data_config)
        sw.check_preprocessor_columns(preprocessor)
        stride = cli.window_stride_seconds * 25
        report.update(config_provenance=config_provenance, model_args=vars(args))
        results = {}
        if "real-r2" in cli.checks:
            results["real_r2"] = real_r2(preprocessor, args.window_size, stride, prepare_night, sw)
            save_json(output / "summary.json", dict(report, results=results))
        model_checks = [check for check in cli.checks if check != "real-r2"]
        main_plan = None
        if {"stage-swap", "shuffle"} & set(model_checks):
            prepared = prepare_night(preprocessor, cli.subject, allow_time_gaps=True)
            main_plan = (prepared, sw.select_window_starts(prepared, args.window_size, stride, max_windows=cli.max_windows))
        swap_plans = None
        if "demographic-swap" in model_checks:
            swap_plans = []
            for subject in cli.subjects:
                prepared = prepare_night(preprocessor, subject, allow_time_gaps=True)
                swap_plans.append((prepared, sw.select_window_starts(prepared, args.window_size, stride,
                                                                     max_windows=cli.max_windows)))
        if cli.check_only or not model_checks:
            report.update(status="checked_only" if cli.check_only else "completed", results=results,
                          windows=dict(main=None if main_plan is None else int(len(main_plan[1])),
                                       demographic_swap=None if swap_plans is None else
                                       {p["subject_id"]: int(len(s)) for p, s in swap_plans}))
            save_json(output / "summary.json", report)
            return report

        import torch
        from diffusion_schedule import build_diffusion_config
        if device is None:
            from cuda_runtime import require_cuda
            device = require_cuda()
        torch.set_num_threads(cli.threads)
        loader = sw.load_frozen_model if model_loader is None else model_loader
        checkpoint = cli.checkpoint.expanduser().resolve()
        model, checkpoint_identity = loader(args, checkpoint, device)
        schedule = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in build_diffusion_config(args).items()}
        context = dict(sw=sw, model=model, schedule=schedule, args=args, device=device, batch_size=cli.batch_size,
                       seed=cli.seed, sampler=sampler, scaler=preprocessor.scaler)
        if "stage-swap" in model_checks:
            results["stage_swap"] = stage_swap(context, *main_plan)
        if "shuffle" in model_checks:
            results["shuffle"] = shuffle(context, *main_plan)
        if "demographic-swap" in model_checks:
            results["demographic_swap"] = demographic_swap(context, swap_plans)
        evaluated = {key: results[key]["passed"] for key in ("stage_swap", "shuffle", "demographic_swap") if key in results}
        decision = dict(checks_passed=evaluated, conditions_sufficient=bool(evaluated) and all(evaluated.values()),
                        rule="all evaluated model checks pass; real_r2 is reported as context")
        figure = _plot(output, results)
        if "demographic_swap" in results:
            with (output / "demographic_swap_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                subjects = results["demographic_swap"]["subjects"]
                writer.writerow(["donor\\target"] + subjects)
                for name, row in zip(subjects, results["demographic_swap"]["w1_ms_matrix_rows_donor_columns_target"]):
                    writer.writerow([name] + row)
        report.update(status="completed", results=results, decision=decision, figure=figure,
                      checkpoint=str(checkpoint), checkpoint_identity=checkpoint_identity,
                      subject=main_plan[0]["subject_id"] if main_plan else None)
        save_json(output / "summary.json", report)
        print(json.dumps(dict(status="completed", decision=decision)), flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save_json(output / "summary.json", report)
        raise


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
