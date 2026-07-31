# Logits Integrity Report

- generated_utc: `2026-07-31T10:35:56Z`
- logits_root: `datasets/logits`
- shard_count: `0`

## Result

- Precomputed logits are the canonical requirement for the offline-clean lane in this closure pass.
- If shard coverage is incomplete, `HF_TOKEN` plus a successful Phase-0 precompute remains the only claim-safe path back to green.
