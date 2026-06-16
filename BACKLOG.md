# BACKLOG — deferred work

Canonical backlog entry point. Turkish: [BACKLOG_TR.md](BACKLOG_TR.md). Seed detail: [V2_BACKLOG_SEED.md](V2_BACKLOG_SEED.md).

## Hard gate (not a code edit)
- **45K GPU run (K1)** — train the canonical 3.67B model on H100/H200; produce the first real loss curve, checkpoint, model card. This is the only thing that converts "target/vision" claims to "measured". See [STATUS.md](STATUS.md).

## Post-45K — verified findings, deliberately NOT changed pre-run (so the run is un-confounded)
Each is documented with its mechanism in [DECISIONS.md](DECISIONS.md):
- **z-loss effective weight ≈ 2e-6** (1e-4 × 0.02 double-multiply, ~50× below intended). Pick the intended effective weight and apply once; re-validate with `cfc_moe_tolerance_check.py`.
- **Liquid `dt` fixed at 1.0** → canonical CfC is a gated RNN, not continuous-time. Either wire variable `dt` and ablate, or relabel.
- **GPU-perf only:** sequential MoE dispatch skips `capacity_mask` (correct output, wasted FLOPs); per-micro-step `.item()` host-device syncs. Measurable only on real GPU.
- **`liquid_warmup_steps`** has no env override (hardcoded 10000) — add for parity with other tunables.
- **`mark_weights_updated()`** uncalled cache-invalidation hook — confirm whether the eval cache needs it (do NOT delete blindly).

## Cosmetic / housekeeping (safe, non-behavioral)
- `mark_weights_updated` dead-method review; `_compute_weight_version` relies on private torch `._version` (guard across versions); RoPE cache lazy-grow comment clarity; `iter_packed_sequences` greedy-buffer note.
- See `V2_BACKLOG_SEED.md` Track A–F for the full list (compile policy, distributed contract, optimizer matrix, etc.).

## Out of scope (documented, not pursued in repo)
- AGI/ASI capability rows in `reports/closure_57_matrix.md` are **out-of-scope pending** (require benchmark / long-horizon evidence) — see [INTERNAL_AGI_GAP.md](INTERNAL_AGI_GAP.md).
