# BACKLOG — deferred work

Canonical backlog entry point. Turkish: [BACKLOG_TR.md](BACKLOG_TR.md). Seed detail: [V2_BACKLOG_SEED.md](V2_BACKLOG_SEED.md).

## Hard gate (not a code edit)
- **45K GPU run (K1)** — train the canonical 3.67B model on H100/H200; produce the first real loss curve, checkpoint, model card. This is the only thing that converts "target/vision" claims to "measured". **No intermediate 250M/500M run** — straight to 45K (see [DECISIONS.md](DECISIONS.md)). See [STATUS.md](STATUS.md).

## Post-45K — verified findings, deliberately NOT changed pre-run (so the run is un-confounded)
Each is documented with its mechanism in [DECISIONS.md](DECISIONS.md):
- **z-loss effective weight ≈ 2e-6** (1e-4 × 0.02 double-multiply, ~50× below intended). Pick the intended effective weight and apply once; re-validate with `cfc_moe_tolerance_check.py`.
- **Liquid `dt` fixed at 1.0** → canonical CfC is a gated RNN, not continuous-time. Either wire variable `dt` and ablate, or relabel.
- **GPU-perf only:** sequential MoE dispatch skips `capacity_mask` (correct output, wasted FLOPs); per-micro-step `.item()` host-device syncs. Measurable only on real GPU.
- **`liquid_warmup_steps`** has no env override (hardcoded 10000) — add for parity with other tunables.
- **`mark_weights_updated()`** uncalled cache-invalidation hook — confirm whether the eval cache needs it (do NOT delete blindly).

## Post-45K — holistic-read (EK-4) additions
- **Evidence-gated banner WRITER (design post-run):** build the actual `PRE-TRAINING (UNVERIFIED)` → measured-status writer once a *real* 45K checkpoint + a *real* (non-zero) eval exist, so the gate can be designed/tested against genuine artifacts. A pre-built auto-writer is unsafe (a stray demo checkpoint + a stub `summary.json` "status: ready" with zero counts satisfies a naive gate and stamps a false "TRAINED" claim across ~50 files). The shipped `scripts/flip_status_banner.py` is therefore **report-only** for now; the writer (with explicit human confirm + non-zero-metric gate + frozen-path banners) is this deferred item.
- **Unit-test re-baseline pass (separate):** add `decoupled_rope` / top-p / MoE-capacity / quant-parity tests. Deferred because it re-baselines the `370/4` count and cascades through every "370 passed" claim surface — do it as its own pass.
- **Liquid training loop perf:** the `for t in range(T)` recurrence (seq=4096, 3 layers) is a real throughput cost; measurable/optimizable only on real GPU (profile-then-optimize, post-run).
- **`ffn_dropout` intent:** not in `config.py` → silently `0.0` in `layers/ffn.py` while `attention_dropout`/`dropout` are `0.1`. Confirm whether FFN-dropout-off is intended before any tuning.

## Cosmetic / housekeeping (safe, non-behavioral)
- `mark_weights_updated` dead-method review; `_compute_weight_version` relies on private torch `._version` (guard across versions); RoPE cache lazy-grow comment clarity; `iter_packed_sequences` greedy-buffer note.
- See `V2_BACKLOG_SEED.md` Track A–F for the full list (compile policy, distributed contract, optimizer matrix, etc.).

## Out of scope (documented, not pursued in repo)
- AGI/ASI capability rows in `reports/closure_57_matrix.md` are **out-of-scope pending** (require benchmark / long-horizon evidence) — see [INTERNAL_AGI_GAP.md](INTERNAL_AGI_GAP.md).
- The `orchestrator/` cognitive runtime + flag-off layers (`cognitive_extensions.py`, `world_model_head.py`, `lifelong_safety.py`, `qinn.py`) are **inert on the canonical 45K path** — out-of-scope, not part of the trained model. Documented (not deleted); see [ARCHITECTURE.md](ARCHITECTURE.md) "out-of-scope surfaces".

## Pre-45K — laptop preflight run-feedback (2026-07-02, real run signal)
Source: `evidence/2026-07-02-laptop-preflight/` (RTX 5070, 8 GB, commit `5fc5068`). The pre-flight validated the infrastructure (atomic checkpoint at step 500; guards live; graceful interrupt-save at step 981) AND produced a decisive negative training-dynamics finding: the run **diverged** (loss ~10.4 → ~15.0; grad_norm → `inf`, sustained `1e11`–`1e16`, survived only via `clip=2.0`; MoE load entropy 0.99 → 0.74). These are the pre-45K stabilization items — **documented as run-feedback; the frozen training path is NOT changed until each is applied and re-verified on a real run:**
- **LR regime:** `1.5e-3` is empirically fatal for this arch at this scale → sweep from `3e-4`, drop the router ×1.5 LR multiplier, lengthen warmup; target clip-hit rate `< 5%` (not "survive despite clip").
- **Liquid spike threshold:** absolute `loss>5.0` is scale-blind (never releases at this loss range → the Liquid layer is effectively untrained). Make it **relative** (`EMA×1.5`) + cooldown.
- **`generate()` Liquid state:** thread the Liquid hidden state through decode (as router state already is) + add a full-forward↔incremental-decode **parity test**.
- **Held-out perplexity harness:** fixed corpus + fixed seed + fixed script (today's eval cannot tell whether a run learned anything beyond its own loss curve).
- **100–300M pilot** before the real 45K (gate: clip-hit `<5%`, no persistent Liquid freeze, no MoE collapse, monotone held-out ppl).
