# Ocean 2x H200 1024-First Launch Profile

- generated_utc: `manual-update-2026-05-24`
- scope: target-machine launch profile for the canonical 45K architecture validation run
- claim boundary: this is launch readiness infrastructure, not trained-model, benchmark, latency, energy, deployment, or production evidence

## Protected 45K Boundary

This profile does not change teacher, tokenizer, dataset policy, loss, canonical prompt surface, or the BitNet + MoE + Liquid architecture. Ocean/OnCompute is treated only as the target execution environment. Older Ocean arithmetic proof sidecars are not imported into the canonical 45K training path.

## Runtime Profile

```bash
TITAN_OCEAN_45K_LAUNCH=1
TITAN_OFFLINE=0
TITAN_INSTALL=1
TITAN_PROFILE=stable
TITAN_BATCH_SIZE=1024
TITAN_BATCH_SIZE_FALLBACKS=1024,512,256
TITAN_DATALOADER_PIN=1
TITAN_DATALOADER_NONBLOCKING=1
TITAN_FFN_PACK=1
TITAN_MOE_PACK=1
TITAN_MLA_KV_PACK=1
TITAN_LIQUID_FAST_PATH=0
TITAN_LIQUID_TRAIN_IMPL=packed_pair
MERTFORMER_LOWBIT_KERNEL=0
MERTFORMER_FUSED_BACKWARD=0
```

`packed_pair_compile` is intentionally not part of the first long run. `ACCELERATE_CONFIG_FILE` stays optional and should point only to a target-matching profile under `repro/`.

## Token Budget (canonical 23.6B vs this 1024 profile)

`config/config.py` defaults to `batch_size=128` and `target_tokens_min=23.6B`
(45,000 × 128 × 4096). This 1024-first profile multiplies the global batch by 8, so
the same 45,000 steps process **~188B tokens (~51 tok/param) — 8× the 23.6B default**.
The WSD learning-rate schedule and the 5-stage curriculum are **step-based and are NOT
rescaled** by batch size, so each stage simply sees ~8× more tokens.

- **Canonical budget: 23.6B** (config default). Use it by keeping `TITAN_BATCH_SIZE=128`
  (omit the override). This is the `DECISIONS.md` canonical 45K target.
- **Opt-in higher-throughput budget: ~188B** via `TITAN_BATCH_SIZE=1024` (this profile).
  A deliberate compute-budget choice, not a silent default.
- **Guard:** set `TITAN_STRICT_TOKEN_BUDGET=1` to make `config/config.py` **hard-fail**
  (instead of stderr-warn) when planned tokens exceed `target_tokens_min` by >5% — so a
  1024 launch that was meant to be 23.6B cannot start by accident.

## Failure Policy

`scripts/final_orchestrator.py` starts with batch `1024`. It retries `512` and then `256` only when the child result contains a clear OOM signal such as CUDA out-of-memory or `oom_backoff_exhausted`. Teacher/data/config/auth/test failures are not batch-fallback events and should stop the launch.

## Required Secrets

The Ocean package must inject `ONCOMPUTE_REPO_KEY` for encrypted repo materialization and `HF_TOKEN` for the remote-bootstrap 45K path. Secrets must stay runtime-only and must not be committed into tracked artifacts.

## Verification

```bash
python3 -m pytest -q tests/test_final_orchestrator_cli.py tests/test_packed_projection_equivalence.py tests/test_liquid_safeguard.py
bash scripts/verify_all.sh
bash zero_touch_start.sh --check-only
git diff --check
```

## Expected Launch Outputs

- `reports/start_gate_report.json`
- `reports/train_readiness_decision.json`
- `reports/final_orchestrator_status.json`
- training logs and checkpoints produced by the canonical training path
- post-train evidence artifacts only after a real checkpoint exists
