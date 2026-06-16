# Ablation artifacts

Canonical, human-readable ablation results live in [`/ABLATION.md`](../../ABLATION.md)
(Türkçe: [`/ABLATION_TR.md`](../../ABLATION_TR.md)).

## Liquid / CfC mixer

- **CANONICAL (FINAL):** `liquid_ablation_final_20260615/` — 12-seed ON/OFF ablation
  (`final_summary.json` + sha256-chained `MANIFEST.json` + plots). Verdict: no accuracy benefit,
  ~30% slower per wall-clock, inconclusive at toy scale. See `/ABLATION.md`.
- **SUPERSEDED (history, retained):** the 2026-06-14 single-seed pilot —
  `liquid_ablation_kaggle_20260614.json`, `liquid_ablation_results.json`,
  `liquid_ablation_pilot_curve.png`. The single-seed Δ(off−on)=+0.50 signal was largely one lucky
  seed and is superseded by the 12-seed final above. Kept for provenance only.

Raw heavy artifacts (telemetry CSV, per-seed JSONL logs) are intentionally not committed; the
sha256 `MANIFEST.json` is the tracked proof (see `.gitignore`).
