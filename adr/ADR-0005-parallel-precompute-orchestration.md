# ADR-0005 Multi-GPU Data-Parallel Teacher-Logit Precompute (Additive Orchestration)

- Date: `2026-06-11`
- Owner: `repo maintainer (explicit user authorization)`
- Status: `accepted`

## Context
- Teacher-logit precompute (a full 70B forward over every unique packed sequence) is the dominant wall-clock cost of the offline_clean KD lane. The canonical `scripts/precompute_logits_topk.py` is single-process (`device_map="auto"`, `--batch-size 4`): on an 8-GPU node it does **not** use the other GPUs for throughput, so a real ~6.1B-unique-token run takes days where it could take a working day.
- Top-K sparse storage, 4-bit teacher load, resume-by-raw-line, and per-sequence identity (ADR-0004 / B2) reduce **disk** and **risk** but not the teacher **compute** count — that floor is `unique_tokens / throughput`, and the only lever left in-repo is data-parallelism across GPUs.
- The user authorized, on `2026-06-11`, adding a production-grade parallel precompute that is 100% compatible with every existing launch command and architecture, with an explicit "do not break anything / add no risk" constraint. This is **additive tooling**, not an architecture-freeze breach (no protected-core/MoE/tokenizer/dataset-policy change), recorded here for auditability.

## Decision
- **New worker sharding (additive, opt-in) in `scripts/precompute_logits_topk.py`:** flags `--num-shards N` / `--shard-id i` select a *block-cyclic* slice. A worker re-packs the **whole** stream from line 0 with the identical deterministic packer (so every sequence's identity is unchanged) and teacher-forwards **only** blocks where `(seq_index // chunk_size) % N == i`. Each owned block is written as one shard named by its first global `seq_index` (`stageK_train_part_{block*chunk}.pt`).
- **Why order/identity survive:** the train reader (`PrecomputedLogitsIterable` / `PrecomputedCurriculumDataset`) and `scripts/validate_logit_alignment.py` sort shards by integer part index. Naming each shard by its first global `seq_index` makes the union of all workers' shards read back in **exact global order**, so the train-side per-sequence identity HARD-assert and the preflight validator hold **verbatim, unchanged**.
- **Default path is byte-identical:** `--num-shards 1` (the default) dispatches to the original `precompute_stage` with its original chunk-index naming and state file. No existing call site passes the new flags, so the canonical single-process lane is untouched.
- **New orchestrator `scripts/precompute_logits_parallel.py`:** launches one worker subprocess per GPU (`CUDA_VISIBLE_DEVICES` pinned; a 4-bit 70B fits one modern GPU), monitors, retries a failed worker (resume is idempotent), then per stage verifies **coverage** (every block present, no gaps), writes the **canonical** `_save_resume_state` so `_stage_complete` / `has_precomputed_logits` / `titan_preflight` see completion, and (unless `--no-validate`) runs `validate_logit_alignment.validate_stage` which must PASS. Exit codes follow the repo convention (`0/1/4/5/130`); a JSON report lands under `logs/precompute_parallel/`.

## Alternatives Considered
- **Round-robin by `seq_index % N`** — rejected: interleaves seq_indices across shards, so first-seq-index naming no longer yields contiguous, globally ordered shards (read order would break the identity assert).
- **Contiguous raw-line-range split per worker** — rejected: the greedy packer carries a cross-row buffer, so starting mid-file diverges from the single-pass packing at the seam (the exact tier-2 seam bug ADR-0004 fixed). Block-cyclic on `seq_index` avoids it because every worker packs from line 0.
- **A pre-computed "pack plan" of input_ids shared to workers** — rejected for this pass: larger new on-disk format + reader, more surface/risk; re-packing from 0 per worker is the already-accepted cost in this codebase and the prefix-tokenization overhead is a small fraction of teacher forward.
- **Multi-node / pipeline-sharded single model** — out of scope; data-parallel independent teacher copies is simpler and saturates an 8-GPU node.

## Tradeoffs / Rollback Impact
- Fully revertible and isolatable: delete `scripts/precompute_logits_parallel.py` and stop passing `--num-shards/--shard-id`; the single-process lane is unaffected. No config default changed.
- **One-mode-per-`logits_dir`:** parallel shards are named by first-seq-index, single-process shards by chunk-index; both are integer-part-sortable and the reader handles either, but the two schemes must not be mixed in the **same** directory in the **same** run. The orchestrator owns a clean run and writes the canonical state at the end.
- Each worker loads its own teacher copy (N× teacher VRAM across N GPUs) — intended (independent data-parallel forwards), and a 4-bit 70B fits a single modern GPU.
- Re-packing from line 0 in every worker re-tokenizes the prefix up to its blocks (small vs the teacher forward it skips).

## Reports / Truth-Sync Implication
- New file `tests/test_precompute_parallel.py` (9 tests) pushes the suite above the prior `338 passed, 4 skipped`; `scripts/verify_all.sh` auto-captures the new stat and `scripts/sync_test_stat_claims.py` propagates it to `README.md`, `README_TR.md`, and the snapshot/SOP summaries — generated, not hand-edited.
- `scripts/build_closure_governance_pack.py` should be refreshed so the ADR index lists ADR-0005.

## Benchmark / Product / Ops / Legal-Security Impact
- No model-quality, benchmark, or capability claim. The output shards are bit-for-bit the same teacher Top-K logits the single-process lane would produce (same packer, same identity), only computed across more GPUs — so KD results are unaffected; only wall-clock changes.
- No new external capability claims; no tokenizer/teacher/dataset-policy change.

## Adversarial review fixes shipped in this pass
A multi-agent adversarial review (4 lenses, every finding independently refute-verified) ran before closure. Six findings were confirmed and the four actionable ones are fixed in this pass:
- **HIGH (review) — `_stage_shards` lexicographic sort.** `scripts/precompute_logits_topk.py:_stage_shards` sorted shard files as strings, so the alignment validator (`validate_logit_alignment` / `has_precomputed_logits` / `titan_preflight`) read identities out of global order at ≥6 parallel blocks (`part_10000 < part_2000`) or ≥11 single-process shards (`part_10 < part_2`) and **false-FAILed a byte-correct shard set**. Fixed to sort by integer part index (matching the train reader `distillation_manager._part_index`); repairs the latent single-process bug too. Regression: `test_stage_shards_sorted_by_integer_part_index`, `test_validator_passes_across_digit_boundary`.
- **HIGH (review) — SIGINT relaunch.** The orchestrator's `_run_stage` retry loop did not check the interrupt flag, so a Ctrl-C'd worker wave was relaunched. Fixed by threading the interrupt flag and short-circuiting. Regression: `test_run_stage_honors_preset_interrupt`.
- **BLOCKER (review, pre-existing) — sparse-KD multi-GPU dispatch hang.** Independent of the parallel feature: on >1 GPU the canonical offline_clean lane is an `IterableDataset`, so Accelerate dispatches batches by calling `accelerate.utils.concatenate()` on the collated batch on rank 0; the sparse teacher payload carried `format` (str) / `vocab_size` / `top_k` (int) keys, so `concatenate()` raised `TypeError` on rank 0 while peers hung at the next collective (NCCL hang on the first batch). Fixed by making the **collated** payload tensor-only (`train/train.py:_stack_teacher_payloads`) and detecting sparse-ness from the tensors (`_is_sparse_topk_payload`) — `format`/`vocab_size`/`top_k` are not read downstream of collate. Dispatch (and thus the offline data-distribution semantics) is unchanged. Regression: `test_collated_sparse_payload_survives_accelerate_dispatch_concat` calls the exact `accelerate.utils.concatenate` path on CPU.
- **LOW (review, documented, not changed) — `tokens_seen_total` estimate.** `train/train.py` estimates `tokens_seen_total` from rank 0's slice × `num_processes` rather than a collective reduce. It is inert in the default `token_budget_mode="fixed_steps"` (the saturation gate is off) and only skews a provenance number under the non-default `open_ended` mode. Left unchanged this pass: the correct fix is a per-step collective that cannot be hardware-validated on the single-device build, so changing it blind would add the very multi-GPU risk this ADR avoids. Tracked as a known minor limitation.

**Validation boundary (honest):** the BLOCKER fix is verified by exercising the real `accelerate.utils.concatenate` on CPU and by the full green suite; the true multi-GPU NCCL path was **not** hardware-validated (no multi-GPU on the build host). A short 2-GPU smoke of the offline_clean lane on the rented machine before the full run is recommended.

## Compatibility Impact
- Train reader, `validate_logit_alignment.py`, `titan_preflight.py`, `start_gate.py`, `zero_touch_start.sh`, `final_orchestrator.py` and every other launch command are unchanged and consume the parallel output exactly as before — guaranteed by the global-order shard naming and a test that runs the real validator against sharded output.
