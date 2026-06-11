# Offline 4060 Demo — Architecture-Validation Smoke Evidence

- date: `2026-06-11`
- claim mode: `measured` (local architecture-validation smoke only; NOT a capability, benchmark, or readiness claim)
- entrypoint: `scripts/offline_4060_demo_train.py`
- device: MPS (Apple M4, local)

## What this run is
A demo-scale smoke of the **canonical model stack** (`model/transformers.py` → `layers/` BitLinear + Sparse MoE + Liquid + MLA), carrying the 2026-06-11 reliability fixes, run to validate that the real architecture trains end-to-end, converges, and checkpoints on real hardware. Teacherless (`distill_alpha=0`), local byte-level demo tokenizer (vocab 259), demo corpus (99 train / 128 val rows), 300 steps.

## Measured
- loss: step 1 `255.06` → step 300 `3.44`; best val `4.44`
- MoE auxiliary (router) loss active throughout (~1.0)
- gradient norm stable (≈25 → ≈6)
- checkpoint written (best + latest), reload-safe (65 MB)
- elapsed: ~79 s for 300 steps

## Explicit non-claims (boundary kept)
- This is **not** a trained-model, benchmark-verified, capable, mobile-ready, production, or readiness claim.
- The model is demo-scale (hidden 256, 2 layers, vocab 259) on 99 rows; it cannot learn language at this scale.
- Free-form generation collapses to the EOS / most-frequent token at this scale — this is an expected scale limitation, **not** a defect (and explicitly not the prior tokenizer-mismatch bug, which is fixed). A coherent / "talkable" checkpoint remains the post-run evidence class: it requires a real corpus + real compute and is still pending.

## What it demonstrates (and only this)
The canonical architecture trains, converges, checkpoints, and reloads end-to-end on real hardware. Capability stays `target` / post-run.

artifacts: `reports/offline_4060_demo_summary.json`; `logs/offline_4060_demo.jsonl` and `checkpoints/offline_4060_demo/*.pt` (both gitignored — large/ephemeral).
