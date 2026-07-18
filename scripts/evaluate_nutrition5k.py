#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent test-set evaluation for the Nutrition5k vision side-experiment.

This is deliberately NOT a re-read of train_nutrition5k.py's own REPORT.md.
That report is trustworthy (it's computed from the same real held-out split
during training), but this script exists to independently reproduce the
numbers on a different machine, against the delivered checkpoint, using
fresh code -- so "the model is real and works" is not resting on a single
self-reported log.

What it does:
    1. Downloads (or reuses, if already present) the real Nutrition5k
       overhead-RGB dataset via train_nutrition5k.py's own verified
       download path (same GCS bucket, same resumability).
    2. Loads the trained checkpoint (checkpoints/nutrition5k_best.pt, or
       nutrition5k_work/checkpoints/nutrition5k_best.pt if run in the
       original training folder).
    3. Runs the model on every image in the REAL Nutri-Test split (the
       official Nutrition5k train/test partition, not a custom split).
    4. Compares each prediction against the REAL ground-truth nutrition
       values for that exact dish (from Nutrition5k's own metadata CSVs,
       not synthetic/assumed data).
    5. Reports MAE / MAE% per target, computed the same way the paper
       defines it (MAE as a percent of the mean ground-truth value over
       the evaluation set), next to the paper's own verified Table 3
       baselines for direct comparison.

Usage:
    python evaluate_nutrition5k.py

Zero required arguments. Must be run from the same folder as
train_nutrition5k.py (imports it as a library).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_nutrition5k as trainer  # noqa: E402


def find_checkpoint() -> Path:
    candidates = [
        trainer.CKPT_DIR / "nutrition5k_best.pt",
        SCRIPT_DIR / "checkpoints" / "nutrition5k_best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    trainer.fail(
        "No trained checkpoint found in either "
        f"{candidates[0]} or {candidates[1]}. Extract the full delivered "
        "output ZIP (must include the checkpoints/ folder), or run "
        "train_nutrition5k.py here first."
    )
    raise AssertionError("unreachable")


def main() -> int:
    t0 = time.time()
    trainer.log("=" * 72)
    trainer.log("NUTRITION5K INDEPENDENT EVALUATION (zero arguments)")
    trainer.log("Re-runs the real checkpoint against the real held-out test split.")
    trainer.log("=" * 72)

    trainer.bootstrap_dependencies()
    trainer.discover_or_vendor_repo()

    ckpt_path = find_checkpoint()
    trainer.log(f"Using checkpoint: {ckpt_path}")

    state = trainer.load_state()
    # Test split only -- this script never trains, so the ~4000 train-split
    # images (~1.6 GB) would be a pure disk cost for nothing.
    dataset_manifest = trainer.phase_dataset(state, splits=("test",))

    if not trainer.TEST_INDEX_CSV.exists():
        trainer.fail(f"Test index not found at {trainer.TEST_INDEX_CSV} after dataset staging.")

    test_rows = trainer._load_index(trainer.TEST_INDEX_CSV)
    trainer.log(f"Evaluating on {len(test_rows):,} real held-out test dishes (official Nutri-Test split).")

    trainer.phase_config()

    import torch  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = trainer.build_model()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    Nutrition5kDataset = trainer.Nutrition5kDataset
    ds = Nutrition5kDataset(test_rows, trainer.IMAGE_SIZE)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=trainer.BATCH_SIZE, shuffle=False, num_workers=0
    )

    sums = {t: 0.0 for t in trainer.TARGETS}
    gt_sum = {t: 0.0 for t in trainer.TARGETS}
    n = 0
    t_eval0 = time.time()
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)
            preds, _ = model(images)
            bsz = images.size(0)
            n += bsz
            for i, t in enumerate(trainer.TARGETS):
                sums[t] += (preds[t].float() - targets[:, i]).abs().sum().item()
                gt_sum[t] += targets[:, i].sum().item()
            if (step + 1) % 5 == 0:
                trainer.log(f"  ... {n}/{len(test_rows)} images evaluated")

    results = {}
    for t in trainer.TARGETS:
        mae = sums[t] / max(1, n)
        gt_mean = gt_sum[t] / max(1, n)
        mae_pct = (mae / gt_mean * 100.0) if gt_mean > 0 else float("nan")
        results[t] = {"mae": mae, "mae_pct": mae_pct, "gt_mean": gt_mean}

    trainer.log(f"Evaluation done in {trainer.human_duration(time.time() - t_eval0)}.")

    lines = []
    add = lines.append
    add("# Nutrition5k Independent Evaluation (Mac-side, fresh code)")
    add("")
    add(f"Checkpoint: `{ckpt_path}` (epoch {ckpt.get('epoch', '?')})")
    add(f"Real held-out test dishes evaluated: {n:,} (official Nutri-Test split)")
    add("")
    add("| Target | This model MAE | This model MAE% | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |")
    add("|---|---:|---:|---:|---:|")
    for t in trainer.TARGETS:
        r = results[t]
        pb = trainer.PAPER_BASELINE_2D_DIRECT[t]
        mb = trainer.PAPER_ALWAYS_PREDICT_MEAN_BASELINE[t]
        unit = "kcal" if t == "calories" else "g"
        add(
            f"| {t} | {r['mae']:.1f} {unit} | {r['mae_pct']:.1f}% | "
            f"{pb['mae_pct']:.1f}% | {mb['mae_pct']:.1f}% |"
        )
    report_text = "\n".join(lines)

    print("\n" + report_text + "\n")

    out_path = SCRIPT_DIR / "evaluation_independent.md"
    out_path.write_text(report_text + "\n", encoding="utf-8")
    trainer.log(f"Written: {out_path}")
    trainer.log(f"Total wall time: {trainer.human_duration(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
