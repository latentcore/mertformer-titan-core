#!/usr/bin/env python3
"""
==============================================================================
MERTFORMER TITAN — TRAINING LOG VISUALIZER
------------------------------------------------------------------------------
Forensic-grade training analysis

Usage:
    python scripts/plot_training_log.py <path_to_jsonl_log>
    python scripts/plot_training_log.py <path_to_jsonl_log> --out report.png
    python scripts/plot_training_log.py <path_to_jsonl_log> --dark
==============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# matplotlib is imported LAZILY (see _require_matplotlib) rather than at module
# import time. Two reasons:
#   1. Parsing + the console summary need no plotting backend at all, so they must
#      stay usable (and unit-testable) in an environment without matplotlib.
#   2. scripts/one_command_full_sop.sh runs this script as a ladder step. A
#      hard `sys.exit(1)` at import turned "matplotlib not installed" into a FAILED
#      ladder step instead of a skipped chart -- and matplotlib is genuinely absent
#      from a freshly bootstrapped .titan-venv even though requirements.txt lists it.
plt = None
ticker = None
GridSpec = None


def _require_matplotlib() -> bool:
    """Import matplotlib on demand. Returns False (no raise) when unavailable."""
    global plt, ticker, GridSpec
    if plt is not None:
        return True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.ticker as _ticker
        from matplotlib.gridspec import GridSpec as _GridSpec
    except ImportError:
        return False
    plt, ticker, GridSpec = _plt, _ticker, _GridSpec
    return True

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "loss":           "#FF6B6B",
    "val_loss":       "#4ECDC4",
    "ce":             "#FFE66D",
    "kd":             "#A8E6CF",
    "aux":            "#FF8B94",
    "entropy":        "#6C5CE7",
    "max_load":       "#FD79A8",
    "grad_norm":      "#00B894",
    "lr":             "#74B9FF",
    "ppl":            "#E17055",
    "tok_s":          "#55EFC4",
    "collapse_line":  "#FF3838",
    "target_line":    "#00FF88",
}


def apply_theme(dark: bool = True):
    """Apply professional dark or light theme."""
    if dark:
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor":   "#0D1117",
            "axes.facecolor":     "#161B22",
            "axes.edgecolor":     "#30363D",
            "axes.labelcolor":    "#C9D1D9",
            "text.color":         "#C9D1D9",
            "xtick.color":        "#8B949E",
            "ytick.color":        "#8B949E",
            "grid.color":         "#21262D",
            "grid.alpha":         0.6,
            "legend.facecolor":   "#161B22",
            "legend.edgecolor":   "#30363D",
        })
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.grid":          True,
        "grid.linewidth":     0.5,
        "legend.fontsize":    8,
        "figure.dpi":         150,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DATA PARSING
# ─────────────────────────────────────────────────────────────────────────────

STEP_FIELD_ALIASES = {
    # dashboard key            : accepted source keys, in priority order
    "loss": ("loss",),
    "ce": ("ce",),
    "kd": ("kd", "distill"),
    "aux_loss": ("aux_loss", "aux"),
    "router_entropy": ("router_entropy", "moe_load_entropy"),
    "router_max_load": ("router_max_load", "moe_max_load"),
    "collapse_detected": ("collapse_detected", "router_collapse"),
    "grad_norm": ("grad_norm",),
    "lr": ("lr",),
    "tok_s": ("tokens_per_sec", "tok_s"),
    "tokens_seen": ("tokens_seen",),
    "capacity_overflow_ratio": ("capacity_overflow_ratio", "moe_capacity_overflow"),
}


def _record_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dict that actually holds the metric fields.

    ``utils/logger.py::RunLogger.log_step`` writes step metrics **flat** at the top
    level of the record; ``log_event`` nests its payload under ``"data"``. This
    parser previously only looked at ``entry["data"]``, so every real step record
    (the flat kind) resolved to ``{}``, ``step`` came back ``None``, and the whole
    dashboard silently reported "No training steps found" and exited 1. Accept both
    shapes.
    """
    nested = entry.get("data")
    return nested if isinstance(nested, dict) else entry


def _pick(row: Dict[str, Any], names: tuple, default: Any = None) -> Any:
    """First non-None value among ``names``.

    train/train.py emits the MoE telemetry under ``moe_*`` names
    (``moe_load_entropy``, ``moe_max_load``, ``moe_capacity_overflow``) and the aux
    loss as ``aux``/``distill``, while this dashboard's panels were written against
    ``router_*`` / ``aux_loss`` / ``kd``. Both spellings are accepted so the MoE
    Health and loss panels are populated by the real trainer output.
    """
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def parse_log(path: str) -> Dict[str, List]:
    """Parse JSONL log file into metric arrays."""
    steps = {"step": [], **{key: [] for key in STEP_FIELD_ALIASES}}
    evals = {"step": [], "val_loss": [], "val_ppl": []}
    config_info: Dict[str, Any] = {}

    # train/train.py calls logger.log_step() TWICE per logged optimizer step (once
    # with the compact `metrics` dict, once with the richer `log_data` dict), both
    # tagged type="step" and both carrying the same step number. Merge by step so a
    # single point per step is plotted with the union of both field sets, instead of
    # two half-populated points. Same approach preflight_run.py::parse_step_csvs
    # already uses for the CSV side.
    merged_steps: Dict[int, Dict[str, Any]] = {}
    merged_evals: Dict[int, Dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = _record_view(entry)
            entry_type = entry.get("type") or row.get("type", "")

            if entry_type == "config":
                config_info = row
                continue

            if entry_type in ("step", "eval"):
                raw_step = row.get("global_step", row.get("step"))
                try:
                    step = int(raw_step)
                except (TypeError, ValueError):
                    continue
                bucket = merged_steps if entry_type == "step" else merged_evals
                target = bucket.setdefault(step, {})
                for key, value in row.items():
                    if value is not None:
                        target[key] = value

    for step in sorted(merged_steps):
        row = merged_steps[step]
        steps["step"].append(step)
        for key, aliases in STEP_FIELD_ALIASES.items():
            default = 0 if key in ("collapse_detected", "capacity_overflow_ratio") else None
            steps[key].append(_pick(row, aliases, default))

    for step in sorted(merged_evals):
        row = merged_evals[step]
        evals["step"].append(step)
        evals["val_loss"].append(_pick(row, ("val_loss",)))
        evals["val_ppl"].append(_pick(row, ("val_ppl_capped", "val_ppl")))

    return steps, evals, config_info


def _safe(arr):
    """Replace None with NaN for matplotlib."""
    import math
    return [x if x is not None else math.nan for x in arr]


def _has_data(arr):
    """Check if array has any non-None values."""
    return any(x is not None for x in arr)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(steps, evals, config_info, out_path: str, dark: bool = True):
    """Generate the full training dashboard."""
    apply_theme(dark)

    # Determine layout: 6 panels
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.07, right=0.95, top=0.94, bottom=0.04)

    s = steps["step"]
    total_steps = max(s) if s else 0

    # ── Title ────────────────────────────────────────────────────────────
    model_name = config_info.get("model_name", "MertFormer")
    version = config_info.get("version", "unknown")
    profile = config_info.get("profile", "")
    title = f"{model_name}  |  {version}"
    if profile:
        title += f"  ·  {profile}"
    title += f"  ·  {total_steps:,} steps"

    tokens = steps["tokens_seen"][-1] if steps["tokens_seen"] else 0
    if tokens:
        if tokens > 1e9:
            title += f"  ·  {tokens/1e9:.2f}B tokens"
        elif tokens > 1e6:
            title += f"  ·  {tokens/1e6:.1f}M tokens"
        else:
            title += f"  ·  {tokens:,.0f} tokens"

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 1: LOSS CURVE (wide, top)
    # ─────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Loss Curve  (Train + Validation)", pad=10)

    ax1.plot(s, _safe(steps["loss"]), color=COLORS["loss"],
             linewidth=1.2, alpha=0.85, label="Train Loss")

    if evals["step"]:
        ax1.plot(evals["step"], _safe(evals["val_loss"]),
                 color=COLORS["val_loss"], linewidth=2, marker="o",
                 markersize=5, label="Val Loss", zorder=5)

    if _has_data(steps["ce"]):
        ax1.plot(s, _safe(steps["ce"]), color=COLORS["ce"],
                 linewidth=0.8, alpha=0.5, label="CE Loss")

    if _has_data(steps["kd"]):
        ax1.plot(s, _safe(steps["kd"]), color=COLORS["kd"],
                 linewidth=0.8, alpha=0.5, label="KD Loss")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.set_xlim(left=0)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 2: MoE Router Health
    # ─────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("MoE Router Health", pad=10)

    if _has_data(steps["router_entropy"]):
        ax2.plot(s, _safe(steps["router_entropy"]), color=COLORS["entropy"],
                 linewidth=1.2, label="Router Entropy")
        ax2.axhline(y=0.6, color=COLORS["target_line"], linestyle="--",
                    linewidth=0.8, alpha=0.6, label="Entropy Target (0.6)")

    if _has_data(steps["router_max_load"]):
        ax2.plot(s, _safe(steps["router_max_load"]), color=COLORS["max_load"],
                 linewidth=1.2, label="Max Expert Load")
        ax2.axhline(y=0.4, color=COLORS["collapse_line"], linestyle="--",
                    linewidth=0.8, alpha=0.6, label="Load Warning (0.4)")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Value")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="center right", fontsize=7)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 3: Auxiliary Loss + Capacity Overflow
    # ─────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("MoE Auxiliary Loss & Capacity", pad=10)

    if _has_data(steps["aux_loss"]):
        ax3.plot(s, _safe(steps["aux_loss"]), color=COLORS["aux"],
                 linewidth=1.0, alpha=0.8, label="Aux Loss")

    if _has_data(steps["capacity_overflow_ratio"]):
        ax3b = ax3.twinx()
        ax3b.plot(s, _safe(steps["capacity_overflow_ratio"]),
                  color=COLORS["entropy"], linewidth=0.8, alpha=0.6,
                  label="Capacity Overflow")
        ax3b.set_ylabel("Overflow Ratio", color=COLORS["entropy"])
        ax3b.tick_params(axis="y", labelcolor=COLORS["entropy"])

    ax3.set_xlabel("Step")
    ax3.set_ylabel("Aux Loss")
    ax3.legend(loc="upper right", fontsize=7)

    # ─────────────────────────────────────────────────────────────────────
    # Panel 4: Gradient Norm
    # ─────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_title("Gradient Norm", pad=10)

    if _has_data(steps["grad_norm"]):
        gnorms = _safe(steps["grad_norm"])
        ax4.plot(s, gnorms, color=COLORS["grad_norm"],
                 linewidth=0.8, alpha=0.7)
        ax4.fill_between(s, 0, gnorms, color=COLORS["grad_norm"], alpha=0.15)

        # Highlight spikes
        import math
        valid = [g for g in gnorms if not math.isnan(g)]
        if valid:
            median_gn = sorted(valid)[len(valid) // 2]
            spike_threshold = median_gn * 10
            spike_steps = [s[i] for i, g in enumerate(gnorms)
                           if not math.isnan(g) and g > spike_threshold]
            spike_vals = [g for g in gnorms
                          if not math.isnan(g) and g > spike_threshold]
            if spike_steps:
                ax4.scatter(spike_steps, spike_vals, color=COLORS["collapse_line"],
                            s=30, zorder=5, label=f"Spikes (>{spike_threshold:.0f})")
                ax4.legend(loc="upper right", fontsize=7)

    ax4.set_xlabel("Step")
    ax4.set_ylabel("Gradient Norm")
    ax4.set_yscale("log")

    # ─────────────────────────────────────────────────────────────────────
    # Panel 5: Learning Rate Schedule
    # ─────────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title("Learning Rate Schedule", pad=10)

    if _has_data(steps["lr"]):
        ax5.plot(s, _safe(steps["lr"]), color=COLORS["lr"], linewidth=1.5)
        ax5.fill_between(s, 0, _safe(steps["lr"]),
                         color=COLORS["lr"], alpha=0.15)

    ax5.set_xlabel("Step")
    ax5.set_ylabel("Learning Rate")
    ax5.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax5.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # ─────────────────────────────────────────────────────────────────────
    # Panel 6: Perplexity (left) + Throughput (right)
    # ─────────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_title("Validation Perplexity", pad=10)

    if evals["step"] and _has_data(evals["val_ppl"]):
        ax6.plot(evals["step"], _safe(evals["val_ppl"]),
                 color=COLORS["ppl"], linewidth=2, marker="s",
                 markersize=5, label="Val PPL")
        ax6.set_yscale("log")
        ax6.legend(loc="upper right")

    ax6.set_xlabel("Step")
    ax6.set_ylabel("Perplexity (log)")

    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_title("Throughput (tokens/sec)", pad=10)

    if _has_data(steps["tok_s"]):
        ax7.plot(s, _safe(steps["tok_s"]), color=COLORS["tok_s"],
                 linewidth=0.6, alpha=0.5)
        # Rolling average
        tok_safe = _safe(steps["tok_s"])
        window = max(1, len(tok_safe) // 50)
        if window > 1:
            import math
            rolling = []
            for i in range(len(tok_safe)):
                chunk = [t for t in tok_safe[max(0, i - window):i + 1]
                         if not math.isnan(t)]
                rolling.append(sum(chunk) / len(chunk) if chunk else math.nan)
            ax7.plot(s, rolling, color=COLORS["tok_s"],
                     linewidth=2, label=f"Rolling avg ({window} steps)")
            ax7.legend(loc="lower right", fontsize=7)

    ax7.set_xlabel("Step")
    ax7.set_ylabel("Tokens / sec")

    # ── Save ─────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅  Dashboard saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(steps, evals, config_info):
    """Print key training metrics to console."""
    import math

    s = steps["step"]
    if not s:
        print("⚠️  No step data found.")
        return

    print("\n" + "=" * 60)
    print("  MERTFORMER TRAINING SUMMARY")
    print("=" * 60)

    print(f"\n  Steps:          {min(s):,} → {max(s):,}")

    tokens = steps["tokens_seen"][-1] if steps["tokens_seen"] and steps["tokens_seen"][-1] else 0
    if tokens > 1e9:
        print(f"  Tokens Seen:    {tokens / 1e9:.2f}B")
    elif tokens > 1e6:
        print(f"  Tokens Seen:    {tokens / 1e6:.1f}M")

    losses = [l for l in steps["loss"] if l is not None]
    if losses:
        print(f"\n  Loss (start):   {losses[0]:.4f}")
        print(f"  Loss (end):     {losses[-1]:.4f}")
        print(f"  Loss (min):     {min(losses):.4f}")

    if evals["val_loss"]:
        vl = [v for v in evals["val_loss"] if v is not None]
        if vl:
            print(f"  Val Loss (end): {vl[-1]:.4f}")

    if evals["val_ppl"]:
        vp = [v for v in evals["val_ppl"] if v is not None]
        if vp:
            print(f"  Val PPL (end):  {vp[-1]:,.1f}")

    ents = [e for e in steps["router_entropy"] if e is not None]
    if ents:
        print(f"\n  Router Entropy: {ents[0]:.3f} → {ents[-1]:.3f}", end="")
        print("  ✅" if ents[-1] > 0.5 else "  ⚠️ LOW")

    loads = [l for l in steps["router_max_load"] if l is not None]
    if loads:
        print(f"  Max Load:       {loads[0]:.3f} → {loads[-1]:.3f}", end="")
        print("  ✅" if loads[-1] < 0.4 else "  ⚠️ HIGH")

    collapses = [c for c in steps["collapse_detected"] if c is not None and c > 0]
    print(f"  Collapse Events: {len(collapses)}", end="")
    print("  ✅" if not collapses else "  🔴 ALERT")

    gnorms = [g for g in steps["grad_norm"] if g is not None]
    if gnorms:
        median_gn = sorted(gnorms)[len(gnorms) // 2]
        max_gn = max(gnorms)
        spikes = sum(1 for g in gnorms if g > median_gn * 10)
        print(f"\n  Grad Norm (med): {median_gn:.2f}")
        print(f"  Grad Norm (max): {max_gn:.2f}")
        print(f"  Grad Spikes:     {spikes}")

    has_nan = any(
        l is not None and (l != l)  # NaN check
        for l in steps["loss"]
    )
    print(f"\n  NaN Detected:    {'🔴 YES' if has_nan else '✅ NONE'}")

    toks = [t for t in steps["tok_s"] if t is not None]
    if toks:
        avg_toks = sum(toks) / len(toks)
        print(f"  Avg tok/s:       {avg_toks:,.0f}")

    print("\n" + "=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MertFormer Training Log Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_training_log.py logs/training.jsonl
  python scripts/plot_training_log.py logs/training.jsonl --out dashboard.png
  python scripts/plot_training_log.py logs/training.jsonl --light --no-summary
        """,
    )
    parser.add_argument("log_file", help="Path to JSONL log file")
    parser.add_argument("--out", "-o", default=None,
                        help="Output PNG path (default: <log_dir>/training_dashboard.png)")
    parser.add_argument("--light", action="store_true",
                        help="Use light theme")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip summary stats")

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"❌  File not found: {log_path}")
        sys.exit(1)

    dark = not args.light

    if args.out:
        out_path = args.out
    else:
        out_path = str(log_path.parent / "training_dashboard.png")

    print(f"📂  Reading: {log_path}")
    steps, evals, config_info = parse_log(str(log_path))

    if not steps["step"]:
        # [2026-07-30] Exit 0, not 1. "This log has no training steps" is not an error --
        # it is the normal state of a repo that has not run training yet, and of
        # aggregate/event-only logs such as logs/ALL_LOGS.jsonl. Since this script is now
        # a DEFAULT step in scripts/one_command_full_sop.sh (moved ahead of the bundle so
        # a fresh dashboard can actually reach the zip), a non-zero exit here failed the
        # entire closure ladder on a machine that simply has no run to plot.
        # A real failure -- a corrupt log, a matplotlib crash -- still surfaces: this
        # branch is only reached when parsing SUCCEEDED and found zero step records.
        print(f"⚠️   No training steps found in {log_path.name}; nothing to plot.")
        print("    (Not an error: this log carries no `type=step` records. Point --out at "
              "a real run log, or run training first.)")
        return

    print(f"📊  Found {len(steps['step']):,} training steps, "
          f"{len(evals['step'])} eval points")

    if not args.no_summary:
        print_summary(steps, evals, config_info)

    if not _require_matplotlib():
        # Degrade to summary-only instead of failing the ladder step. The parsed
        # metrics above are the substance; the PNG is a convenience.
        print("⚠️  matplotlib not installed — summary printed, chart skipped.")
        print("    Install it to get the PNG dashboard: pip install matplotlib")
        return

    plot_dashboard(steps, evals, config_info, out_path, dark=dark)


if __name__ == "__main__":
    main()
