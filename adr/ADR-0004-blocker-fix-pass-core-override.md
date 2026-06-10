# ADR-0004 Authorized Blocker-Fix Pass: Deliberate Override of the Architecture-Freeze / Protected-Core Constraints

- Date: `2026-06-11`
- Owner: `repo maintainer (explicit user authorization)`
- Status: `accepted`

## Context
- A premise-verified multi-agent audit of the working tree found real bugs that the existing test suite did not catch and that would crash a paid GPU run, hang DDP, silently corrupt distillation, or produce a garbage checkpoint.
- `AGENTS.md` constrains closure work: "No large architecture rewrite" (Non-Goals), "Preserve the MoE 8e2 decision unless a verified blocker explicitly requires a change" (Protected Core), and "Do not silently change teacher, tokenizer, or dataset policy".
- The user explicitly authorized, on `2026-06-11`, doing **everything including architecture-class rewrites** for this single pass. This ADR records that override so the freeze-breach is auditable (per the change-control SOP a freeze-breach requires an ADR or explicit governance note).

## Verified issues fixed in this pass
- **BLOCKER B1** — gradient checkpointing + MoE crashed the first backward (`CheckpointError`). Config defaults `use_gradient_checkpointing=True` and `use_moe=True`.
- **BLOCKER B2** — precompute↔teacher-logit alignment was structurally fragile (dynamic padding mismatch, no per-sample identity, produced-count resume, divergent skip predicates) → silent KD corruption.
- **BLOCKER B3** — the `cpp_cpu` BitLinear backend ran an unquantized matmul (full-precision instead of BitNet b1.58).
- **HIGH H1** — per-rank NaN/OOM `continue` desynced DDP collectives (NCCL hang).
- **HIGH H2** — when `pad_token_id == eos_token_id` (Llama/online lane) EOS was masked out of the loss → model never learns to stop.
- **HIGH H3** — config advertised GaLore/8-bit optimizer but the live path was plain full AdamW (`rebuild_optimizer` was dead code) → ~2-4× VRAM vs plan.
- **HIGH H4** — the public SDK `api.load_model` decoded a trained checkpoint with a silently-substituted tokenizer (the original tokenizer bug, still live in the SDK), uncovered by the 305 tests.
- **HIGH H6** + MED/LOW — eval generation never stopped on EOS and decoded the prompt; checkpoint RNG/dataloader state not saved; length-biased curriculum sampler; kernel strict-mode not enforcing; `triton_ternary` STE gap; `generate()` temperature zero-guard.

## Alternatives Considered
- Ship the "pure" MoE-telemetry hoist for B1 — **rejected**: it edits the frozen MoE core with a large regression surface, and a verified non-core fix exists.
- Flip optimizer config defaults to honest plain AdamW (H3) — **rejected**: abandons the intended memory feature; building the real optimizer + logging the active class both keeps the feature and tells the truth.
- Leave the length-biased sampler and the val-set leakage as documented-only — **rejected for this pass**: the user authorized the data-policy changes.

## Decision (which protected surfaces are touched, and why)
- **MoE 8e2 core: NOT touched.** B1 is fixed at the single checkpoint call site (`model/transformers.py` → `use_reentrant=True`). The frozen MoE telemetry writes are deliberately left intact.
- **Data / packing semantics:** introduced `train/packing.py` (deterministic EOS-separated packing + per-sequence identity), repacked precompute and the precomputed-curriculum dataset, switched the token budget to count real (non-pad) tokens, and switched the loss mask to the canonical `-100` ignore index. New flags `cfg.sequence_packing`, `cfg.verify_logit_alignment` (both default on).
- **Tokenizer policy (H4):** `api.load_model` is brought into line with the existing single-source policy (`utils/tokenizer_resolver`); this *enforces* `AGENTS.md` "do not silently change tokenizer policy" rather than violating it.
- **Optimizer policy (H3):** optimizer is now constructed honoring `use_galore`/`use_8bit_adam` with graceful fallback and an explicit startup log of the active class (config == reality).
- **Loop control (H1):** NaN skip is a DDP-collective decision (reusing the existing reduce/broadcast idiom); multi-GPU OOM escalates to a clean safety-brake stop instead of a desynced retry.
- **Dataset policy (sampler):** the online curriculum sampler moves from byte-seek to uniform line-index sampling (removes long-line bias).
- **Validation set (H5):** val candidates are fingerprint-excluded against the on-disk training stage JSONLs (offline-safe, reusing `RollingDeduper`), with a `validation_provenance.json` artifact.
- **Kernel numerics (B3):** `cpp_cpu` now quantizes before the kernel call — corrects numerics, no architecture change.
- A new hard preflight gate `scripts/validate_logit_alignment.py` enforces teacher/student identity alignment (wired into `titan_preflight` strict_offline, `has_precomputed_logits(verify_alignment=...)`, `zero_touch_start.sh`, and the start-gate transfer list).

## Tradeoffs / Rollback Impact
- Each change is independently revertible. B1 rollback = restore `use_reentrant=False` (re-introduces the crash). B3 rollback = drop the quant calls (wrong numerics). H3 rollback = restore hard-coded AdamW. H4 rollback = restore silent teacher fallback (re-introduces the wrong-tokenizer bug). Packing rollback = `TITAN_SEQUENCE_PACKING=0` / `TITAN_ALLOW_LEGACY_LOGIT_REALIGN=1`. Sampler rollback = restore byte-seek.
- `b1.58` numeric behavior on CPU now matches the STE reference; previously-recorded cpp_cpu numbers are not comparable.

## Reports / Truth-Sync Implication
- New tests push the suite above `305 passed, 4 skipped`. `scripts/verify_all.sh` auto-captures the new stat and `scripts/sync_test_stat_claims.py` rewrites `README.md`, `README_TR.md`, `reports/release_snapshot*.md`, and `reports/one_command_full_sop_summary.md`; the count is generated, not hand-edited.
- The token-budget provenance changed (real tokens vs pad-inflated); any token-budget claims in README/TRAINING_PLAN/TECHNICAL_REPORT must reflect real-token accounting.
- `scripts/build_closure_governance_pack.py` should be refreshed so the ADR index lists ADR-0004.

## Benchmark / Product / Ops / Legal-Security Impact
- B3 and H6 change measured eval numbers — any prior benchmark snapshot referencing the cpp_cpu path or GSM8K accuracy is stale and must be re-run. No improved-score claim is made without re-running the actual eval.
- No new external capability claims. Strategic / legal / security decisions remain human-reviewed (consistent with ADR-0002).

## Compatibility Impact
- Existing precomputed logits were missing/incomplete, so the `topk_sparse_v1` → `topk_packed_v1` shard format change requires no real-artifact migration; the legacy reader remains available behind `TITAN_ALLOW_LEGACY_LOGIT_REALIGN=1`.
