"""Plot real vs generated IBI for every saved S092 window of one configuration.

Reads an output directory written by scripts/compare_s092_sweep_best.py, either in
--all-windows mode (many windows, possibly overlapping) or in the consecutive
--start-second/--windows mode. No model or GPU is used.

Only reference-observed positions (scoring_mask, real IBI > 0) are compared; these
are the same positions that the per-window Gaussian KL uses. Values are in seconds.

Outputs (in --output-dir):
  window_metrics.csv               one row per window: time, stages, means, variances, KL
  window_metrics_timeline.png      per-window means and KL across the night, with hypnogram
  pooled_and_stage_kde.png         KDE pooled over windows: all stages, W, Light, REM, Deep
  window_kde_grid.png              KDE of evenly spaced windows (overview)
  all_windows_kde.pdf              KDE of every plotted window, 16 per page

Usage:
  python scripts/plot_s092_all_windows_kde.py RESULT_DIR --config-id 1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STAGES = ("W", "Light", "REM", "Deep")
STAGE_LEVEL = np.array([3, 1, 2, 0])          # hypnogram height: W on top, Deep at the bottom
EPOCH_SAMPLES = 750                            # 30 s at 25 Hz
REAL, GEN = "#2a78d6", "#eb6834"
KDE_MAX_POINTS = 60000                         # pooled KDE subsample (fixed seed); windows use all points


def load(result_dir, config_id):
    root = Path(result_dir).expanduser().resolve()
    folder = root / f"config_{config_id}"
    with np.load(folder / "inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    with np.load(folder / "waveforms.npz", allow_pickle=False) as saved:
        waves = {key: saved[key].copy() for key in saved.files}
    fs, w = int(inputs["sample_rate_hz"]), int(inputs["window_size"])
    ref = inputs["reference_ibi_seconds"].astype(np.float64)
    gen = waves["generated_ibi_seconds"].astype(np.float64)
    mask = inputs["scoring_mask"].astype(bool)
    if ref.size % w or gen.shape != ref.shape or mask.shape != ref.shape:
        raise ValueError("inputs.npz and waveforms.npz do not hold matching complete windows")
    n = ref.size // w
    if "start_samples" in inputs:              # --all-windows mode
        starts = inputs["start_samples"].astype(np.int64)
        if "start_samples" in waves and not np.array_equal(waves["start_samples"], starts):
            raise ValueError("Window starts differ between inputs.npz and waveforms.npz")
    else:                                      # consecutive mode
        starts = int(inputs["start_second"]) * fs + np.arange(n, dtype=np.int64) * w
    if len(starts) != n:
        raise ValueError("Number of window starts differs from the saved windows")
    stages = inputs["sleep_stages"].astype(np.int64).reshape(n, w // EPOCH_SAMPLES)
    meta = {}
    summary_path = folder / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        meta = dict(epoch=summary.get("epoch"), seed=summary.get("seed"),
                    name=(summary.get("configuration") or {}).get("name"))
    return dict(root=root, fs=fs, w=w, starts=starts, ref=ref.reshape(n, w), gen=gen.reshape(n, w),
                mask=mask.reshape(n, w), stages=stages, meta=meta, config_id=config_id)


def gaussian_kl_ms(real_s, gen_s, floor_ms2=1.0):
    """KL(real || generated) of univariate Gaussian fits, ms units, variance floor 1 ms^2 (as in the repo)."""
    r, g = real_s * 1000.0, gen_s * 1000.0
    vr, vg = max(r.var(), floor_ms2), max(g.var(), floor_ms2)
    return 0.5 * (vr / vg - 1.0 - np.log(vr / vg)) + 0.5 * (r.mean() - g.mean()) ** 2 / vg


def nonoverlapping(starts, w):
    keep, last = [], None
    for index, start in enumerate(starts):
        if last is None or start >= last + w:
            keep.append(index)
            last = start
    return np.asarray(keep, dtype=np.int64)


def clock(seconds):
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def kde_curve(values, xs):
    from scipy.stats import gaussian_kde
    if values.size < 2 or np.ptp(values) < 1e-9:
        return None                            # single value: drawn as a vertical line instead
    return gaussian_kde(values)(xs)


def draw_kde(ax, real, gen, xs, title, legend=True, fontsize=9):
    for values, color, style, label in ((real, REAL, "-", "Real"), (gen, GEN, "--", "Generated")):
        curve = kde_curve(values, xs)
        text = f"{label} (mean {values.mean():.3f} s)" if values.size else label
        if curve is None:
            ax.axvline(values.mean() if values.size else np.nan, color=color, ls=style, lw=1.8, label=text)
        else:
            ax.fill_between(xs, curve, color=color, alpha=.12, lw=0)
            ax.plot(xs, curve, color=color, ls=style, lw=1.8, label=text)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=fontsize, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=fontsize - 1, loc="upper left")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--config-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-overlapping", action="store_true",
                        help="Also plot overlapping windows in the KDE figures/PDF (default: non-overlapping only; "
                             "the timeline and CSV always use every window)")
    parser.add_argument("--grid", type=int, default=16, help="Windows in the overview grid (default 16)")
    cli = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    d = load(cli.result_dir, cli.config_id)
    fs, w, starts = d["fs"], d["w"], d["starts"]
    out = (cli.output_dir or d["root"] / f"plots_config_{cli.config_id}_windows").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = len(starts)
    selected = np.arange(n) if cli.include_overlapping else nonoverlapping(starts, w)
    stride = int(np.min(np.diff(starts))) // fs if n > 1 else w // fs

    # Per-window statistics on reference-observed positions.
    rows = []
    selected_set = set(selected.tolist())
    for i in range(n):
        m = d["mask"][i]
        r, g = d["ref"][i][m], d["gen"][i][m]
        rows.append(dict(window_index=i, start_second=starts[i] / fs, start_clock=clock(starts[i] / fs),
                         stages="/".join(STAGES[s] for s in d["stages"][i]), observed_samples=int(m.sum()),
                         real_mean_s=r.mean(), generated_mean_s=g.mean(),
                         real_std_s=r.std(), generated_std_s=g.std(),
                         mean_error_ms=(g.mean() - r.mean()) * 1000, gaussian_kl_real_to_generated=gaussian_kl_ms(r, g),
                         generated_nonpositive_fraction=float((d["gen"][i] <= 0).mean()),
                         in_kde_plots=i in selected_set))
    with (out / "window_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    kl = np.array([row["gaussian_kl_real_to_generated"] for row in rows])

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                         "figure.facecolor": "white", "axes.edgecolor": "#b5b4ad",
                         "axes.labelcolor": "#3f3e3b", "xtick.color": "#3f3e3b", "ytick.color": "#3f3e3b"})
    meta = d["meta"]
    header = (f"S092 | config_{cli.config_id}" + (f" ({meta['name']})" if meta.get("name") else "")
              + (f" | checkpoint epoch {meta['epoch']}" if meta.get("epoch") else "")
              + (f" | seed {meta['seed']}" if meta.get("seed") is not None else ""))
    note = (f"{n} windows of {w // fs} s, start step {stride} s; KDE figures use "
            + ("every window" if cli.include_overlapping else f"{len(selected)} non-overlapping windows")
            + ". Reference-observed positions only; IBI in seconds; generated values not clipped.")

    # Common x-range for all KDE panels (axis limits only; densities use all points).
    sel_mask = d["mask"][selected]
    pooled_r, pooled_g = d["ref"][selected][sel_mask], d["gen"][selected][sel_mask]
    both = np.concatenate([pooled_r, pooled_g])
    lo, hi = np.percentile(both, [0.05, 99.95])
    pad = max(.05, (hi - lo) * .05)
    xs = np.linspace(lo - pad, hi + pad, 600)

    # 1) Timeline: window means, KL, hypnogram.
    hours = starts / fs / 3600
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True, height_ratios=[2.2, 1.6, 1])
    axes[0].plot(hours, [row["real_mean_s"] for row in rows], ".", ms=3.5, color=REAL, label="Real window mean")
    axes[0].plot(hours, [row["generated_mean_s"] for row in rows], ".", ms=3.5, color=GEN, label="Generated window mean")
    axes[0].set_ylabel("Mean IBI (s)")
    axes[0].legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    axes[1].plot(hours, kl, ".", ms=3.5, color="#3f3e3b")
    axes[1].axhline(np.median(kl), color="#888", lw=.8, ls="--", label=f"median {np.median(kl):.3g}")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gaussian KL\n(real || gen)")
    axes[1].legend(frameon=False, loc="upper right")
    epochs = {}
    for i in range(n):
        for k, stage in enumerate(d["stages"][i]):
            epochs[int(starts[i]) // EPOCH_SAMPLES + k] = int(stage)
    for epoch, stage in epochs.items():
        t0 = epoch * EPOCH_SAMPLES / fs / 3600
        axes[2].hlines(STAGE_LEVEL[stage], t0, t0 + 30 / 3600, color="#334155", lw=2.2)
    axes[2].set_yticks([0, 1, 2, 3], ["Deep", "Light", "REM", "W"])
    axes[2].set_ylim(-.6, 3.6)
    axes[2].set_xlabel("Time since start of S092 record (h)")
    for ax in axes:
        ax.grid(axis="y", color="#e6e5e0", lw=.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(header + " | per-window comparison across the night", x=.01, ha="left", fontsize=13)
    fig.text(.01, .005, note + " Timeline and CSV use every window.", fontsize=8.5, color="#55544f")
    fig.tight_layout(rect=(0, .02, 1, .97))
    fig.savefig(out / "window_metrics_timeline.png", dpi=170)
    plt.close(fig)

    # 2) Pooled and per-stage KDE (stage = label of the 30 s epoch containing each sample).
    rng = np.random.default_rng(0)
    sample_stage = np.repeat(d["stages"][selected], EPOCH_SAMPLES, axis=1)[sel_mask]
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6), sharex=True)
    for ax, (label, keep) in zip(axes, [("All stages", np.ones_like(sample_stage, bool))]
                                 + [(name, sample_stage == s) for s, name in enumerate(STAGES)]):
        r, g = pooled_r[keep], pooled_g[keep]
        if r.size == 0:
            ax.set_title(f"{label}: no observed samples", fontsize=10, loc="left")
            continue
        take = rng.choice(r.size, KDE_MAX_POINTS, replace=False) if r.size > KDE_MAX_POINTS else slice(None)
        windows_here = (len(selected) if label not in STAGES
                        else int((d["stages"][selected] == STAGES.index(label)).any(axis=1).sum()))
        draw_kde(ax, r[take], g[take], xs, f"{label}\n{r.size:,} samples · {windows_here} window{'s' if windows_here != 1 else ''}", fontsize=10)
        ax.set_xlabel("IBI (s)")
    axes[0].set_ylabel("Density")
    fig.suptitle(header + " | pooled distribution over windows", x=.01, ha="left", fontsize=13)
    fig.text(.01, .01, note + f" KDE on at most {KDE_MAX_POINTS:,} samples per panel (fixed seed).",
             fontsize=8.5, color="#55544f")
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(out / "pooled_and_stage_kde.png", dpi=170)
    plt.close(fig)

    def window_panel(ax, i, fontsize=8.5):
        m = d["mask"][i]
        title = (f"#{i}  {clock(starts[i] / fs)}–{clock(starts[i] / fs + w / fs)}\n"
                 f"{rows[i]['stages']} · KL {kl[i]:.3g}")
        draw_kde(ax, d["ref"][i][m], d["gen"][i][m], xs, title, legend=True, fontsize=fontsize)

    # 3) Overview grid of evenly spaced windows.
    count = min(cli.grid, len(selected))
    chosen = selected[np.unique(np.linspace(0, len(selected) - 1, count).round().astype(int))]
    cols = 4
    rows_n = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 3.6 * rows_n), squeeze=False, sharex=True)
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    for ax, i in zip(axes.flat, chosen):
        window_panel(ax, int(i))
    for ax in axes[-1]:
        ax.set_xlabel("IBI (s)")
    fig.suptitle(header + " | evenly spaced windows", x=.01, ha="left", fontsize=13)
    fig.text(.01, .005, note, fontsize=8.5, color="#55544f")
    fig.tight_layout(rect=(0, .02, 1, .97))
    fig.savefig(out / "window_kde_grid.png", dpi=150)
    plt.close(fig)

    # 4) Every plotted window, 16 per page.
    with PdfPages(out / "all_windows_kde.pdf") as pdf:
        for page in range(0, len(selected), 16):
            block = selected[page:page + 16]
            fig, axes = plt.subplots(4, 4, figsize=(17, 12.5), squeeze=False, sharex=True)
            for ax in axes.flat[len(block):]:
                ax.axis("off")
            for ax, i in zip(axes.flat, block):
                window_panel(ax, int(i), fontsize=8)
            fig.suptitle(f"{header} | windows {page + 1}–{page + len(block)} of {len(selected)}",
                         x=.01, ha="left", fontsize=12)
            fig.text(.01, .005, note, fontsize=8, color="#55544f")
            fig.tight_layout(rect=(0, .02, 1, .97))
            pdf.savefig(fig)
            plt.close(fig)

    print(json.dumps(dict(status="completed", output_dir=str(out), windows=n, plotted_windows=int(len(selected)),
                          start_step_seconds=stride, median_kl=float(np.median(kl)), mean_kl=float(kl.mean()))))


if __name__ == "__main__":
    main()
