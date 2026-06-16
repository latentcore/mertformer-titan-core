# Architecture Decisions

- **Config discipline**: Keep `config.py` as runtime default; allow optional YAML overlays.
- **Swarm architecture**: Documented as target; not enforced in core training.
- **Multimodal**: Deferred until text core is proven.
- **Canonical KD lane**: Keep the external lane name `offline_clean`, but treat it as the strict precomputed-KD path for the 45K closure run.
- **Teacher surface**: Pin the teacher to `meta-llama/Llama-3.3-70B-Instruct`; do not silently swap teacher IDs on the canonical path.
- **Fallback boundary**: Remove teacherless fallback from the canonical `offline_clean` training path. If logits are missing, fail with the exact blocker instead of silently downgrading the run.
- **Prompt surface**: Keep `prompts/system_v1.txt` as the only canonical system prompt surface until post-training behavioral evidence justifies expansion.
- **Artifact strategy**: Preserve the existing release/repo zips and add a separate training outputs bundle zip for real run retrieval (`artifacts/mertformer_training_outputs_bundle.zip` + SHA256 + manifests).

## Audited decisions (2026-06-12 zero-contradiction closure) — document, do NOT "re-fix"

A multi-agent audit verified these against current code. They are deliberate, NOT bugs;
a future reviewer/AI should not "fix" them:

- **P5/P8 — README performance table**: throughput/latency figures (~1.2s/step, ~107 tok/s,
  11,000 tok/s aggregate) are explicitly labeled "Projected / Est. / Not Measured /
  architecture simulation" — claim-discipline-compliant projections, not measured claims.
- **P13 — utils/safety.py `kill_if_non_finite`**: exported public utility (`utils/__init__.py`
  `__all__`), not orphaned. Absence of an internal caller is not a defect.
- **P18 — `config.auto_configure_batch_size` nvidia-smi**: no naive LOCAL_RANK guard is added —
  a rank-0-only guard would desync per-rank micro-batch; Method-A (`torch.cuda.get_device_properties`)
  already avoids nvidia-smi on real GPU nodes, and prints are env-gated (`TITAN_CONFIG_VERBOSE`).
- **P11 — `trust_remote_code=True`** on the gated Meta `Llama-3.3-70B-Instruct` teacher load is
  deliberate (trusted first-party repo). Production dataset loads do not enable it.
- **Measured param count `3,672,982,022` (~3.67B)** is the canonical measured runtime total;
  explicit `moe_intermediate=8192` (config/config.py) holds it. `economics/flops_estimator.py`
  `DEFAULT_PARAMS=2.64e9` is the separate design-target and is preserved. See reports/FACTS.json.
- **D3 — Stage-1 code source**: switched `bigcode/the-stack-v2` → `bigcode/the-stack-dedup`. v2 rows
  carry `blob_id`/`src_encoding` (the `content` field is empty and needs Software-Heritage S3
  resolution), which silently starved Stage-1's ~28% code corpus; the-stack-dedup exposes a real
  `content` field. A preflight source smoke-probe now fails loudly (`SOURCE_FIELD_EMPTY`) on any
  empty-field source. `datasets/hashes.json` revision/sha256 for the-stack-dedup must be re-pinned on
  the training machine (network required). Deliberate dataset-policy change.
- **Llama naming**: the Llama 3.3 Community License requests derivative model NAMES include "Llama".
  Current posture: name "MertFormer Titan" + the "Built with Llama" NOTICE; full naming compliance is
  EXTERNAL-PENDING legal (see NOTICE and reports/teacher_output_license_assessment.md).

## Mac-doable backlog dispositions (2026-06-13, Pass 7)

Closed as won't-change-now WITH reasons, so the backlog has no silently-"open" rows:
- **33 topk in bf16**: rejected — bf16 lowers top-k routing precision and would break the
  `scripts/cfc_moe_tolerance_check.py` gate (the fp32 cast at layers/moe.py is deliberate).
- **14 Liquid impl default**: already selectable via `TITAN_LIQUID_TRAIN_IMPL`
  (baseline / precompute_input / packed_pair / packed_pair_compile). Flipping the DEFAULT is the
  #1 MFU lever and needs pilot timing data → bound to the pilot, not changed blind.
- **17 MoE capacity-loop vectorization**: GPU-perf only (benefit shows on CUDA); do it on the
  training box where the speedup is measurable and the tolerance gate validates the numerics.
- **20 teacher pad-mask**: saves compute on one sub-max-seq sequence per stage (already masked
  downstream at train time) — GPU-perf only; do it on the teacher-precompute machine.
- **36 legacy logit-realign path**: kept as a deliberate env-gated escape hatch
  (`TITAN_ALLOW_LEGACY_LOGIT_REALIGN`, off by default); the canonical packing path is the default.
- **48 sharded checkpoint (accelerator.save_state)**: needs real multi-GPU to implement+validate;
  belongs on the training box. Single-GPU/Mac uses the standard checkpoint save.
- **23 MLA→GQA rename — DONE (2026-06-16) in the canonical path.** The attention class in
  `layers/mla.py` is now `GQA` (it was always grouped-query — `num_kv_heads` projection + KV head
  replication — never a latent-MLA low-rank bottleneck). Updated in the same change: importers
  (`layers/__init__.py`, `layers/mertformer_block.py`), tests (`test_mla_regressions.py`,
  `test_architecture_integrity.py`), the 57-matrix grep (item 18), the doc-claim-consistency
  required phrase, and the README / ARCHITECTURE / TECHNICAL_REPORT prose (EN + TR). Deliberately
  kept: the filename `layers/mla.py` (path stability for manifests/mypy list) and the public env
  knob `TITAN_MLA_KV_PACK`. The 5 vendored *onefile* delivery copies (`scripts/*onefile*`,
  `chess_5080`, `kaggle_*`) keep their internal `MLA` symbol on purpose — they are frozen,
  self-contained delivery bundles whose sha256 manifests / AST singleton tests would otherwise
  break for zero functional gain.
- **34 train.py split / 68 whole-repo mypy**: permanently left — high blast radius (AST singleton
  tests / red CI gate) for ~zero functional gain.
- **40 drift telemetry**: `lifelong_safety.safety_metrics()` already exposes `last_drift`; wiring it
  into the run logger is purely additive and the layer is off by default → deferred (no behavior risk).
- **24 / 25 / 31 / 38 test backlog**: additive pure-CPU coverage (RoPE position, block-order schema,
  SIGTERM handler, packing resume-seam). No defect — `consumed_through` monotonicity and the freeze/RNG/
  qinn/MoE-tolerance paths are already tested; these extend coverage and are deferred without risk.

## Repo curation — commercial / strategy layer disposition (2026-06-16)

The main repo is the **technical / evidence** surface (it stays PRIVATE; the only public surface is
the README-derived Gist). Business material is kept out of the reviewer-facing tree. **Nothing was
hard-deleted** — everything is either moved under the already-ignored `/private/` tree (local copy
kept, git history intact) or untracked-but-regenerated.

- **Moved to `private/anthropic_internal/` (gitignored):** `application_strategy`, `interview_prep`
  ("Gaps To Say Out Loud"), `science_of_scaling_cv_seed` ("Do Not Claim"), `tokens_variant_notes`,
  `strongest_stories`, `performance_engineer_fallback`. These are internal positioning notes, never
  shipped to a reviewer; the Anthropic packet (`final_one_shot.sh` `refresh_anthropic_packet`) now
  bundles only the four evidence-facing files.
- **Moved to `private/commercial/` (gitignored):** `PITCH(.md/_TR)`, and the hand-authored
  pure-pitch reports `sales_funnel_90d(±TR)`, `ip_licensing_split(±TR)`, `founders_hub_application(±TR)`.
  No generator produces these, so the move is stable. Reviewer-facing index links in README /
  README_TR were removed in the same change.
- **Untracked (regenerated locally, generator unchanged):** `reports/investor_deck.pptx` +
  `_TR` — `git rm --cached` + `.gitignore`. `scripts/build_investor_deck.py` still emits them
  locally; the binary PPTX just leaves the tracked/reviewer-visible tree (same pattern as the heavy
  ablation CSV/jsonl).
- **Left in place — generated, tightly coupled to the verified closure ladder — DOCUMENT, do NOT
  rewire:** `reports/gtm_master_plan.md`, `reports/commercial_handover_pack.md`,
  `reports/legal_ip_pack.md` (all emitted by `scripts/build_offline_closure_pack.py`
  `build_operating_docs`), and `reports/commercial_handover/` (cross-referenced by that pack and by
  `build_closure_governance_pack.py`). Extracting these would require rewiring the green
  closure-ladder builders — a behavior change to a verified pipeline — so per the safe-fix /
  document policy they remain as **internal closure-governance artifacts** (not reviewer-facing).
  The canonical home for active commercial work is the separate private **dealroom** repo, which
  receives the current copies; the main-repo copies are auto-generated byproducts.

## Verified findings — documented now, behavior fixes deferred to post-45K (2026-06-16)

A live source pass (this repo, not analysis transcripts) verified the following. They are real,
but intentionally NOT changed in this pass, because touching the loss / training behavior would
confound the first 45K architecture-validation run. Each is logged with the exact mechanism and
the post-45K action (see `V2_BACKLOG_SEED.md`).

- **z-loss effective weight is ~`2e-6` (double-multiplied), not `1e-4`.** `layers/moe.py:775`
  computes `z_loss = (z*z).mean() * cfg.z_loss_coef` (`z_loss_coef = 1e-4`, config.py:271) and folds
  it into `aux_loss`; `train/train.py:1272` then multiplies the whole aux term again by
  `router_aux_loss_coef = 0.02` (config.py:244). Net z-loss weight = `1e-4 * 0.02 = 2e-6`, ~50x below
  the apparent `1e-4`. The Switch load-balance term is correct (paper alpha). z-loss only guards
  router-logit blow-up; other stabilizers (BitLinear, jitter, capacity) currently cover for it.
  Post-45K: pick the intended effective weight and apply it once (re-validate with
  `scripts/cfc_moe_tolerance_check.py`).
- **Liquid `dt` is fixed at `1.0` → the canonical CfC path is a gated RNN, not continuous-time.**
  `layers/liquid.py` `LiquidCell.forward(..., dt=1.0)` is always called with the default; the
  input-dependent exponential forget gate works, but the continuous-time advantage only appears with
  variable `dt`. README/ARCHITECTURE describe it honestly as a research bet. Post-45K: either wire a
  variable `dt` and ablate it, or relabel it as a gated RNN.
- **`epoch_mode` defaults to `False`** (config.py:447 and `train.py` `getattr(..., False)`): verified
  NOT a live footgun. If manually enabled it can inflate `max_steps` after the config-finalize
  token-budget guard; left off by default, documented for awareness.
- **GPU-perf only, deferred to the training box:** sequential MoE dispatch does not pass
  `capacity_mask` (output is correct — `topk_vals` are renormalized — only wasted FLOPs);
  per-micro-step `.item()` host-device syncs in the train loop. Both only matter (and are measurable)
  on real GPU.
- **`LiquidCell.mark_weights_updated()` (`layers/liquid.py:286`) is currently uncalled.** It is a
  cache-invalidation API (`_weight_version += 1; _cache_ready = False`) — NOT dead code to delete, but
  a *potential* JIT/quant-eval-cache staleness gap. Post-45K: confirm whether the eval cache needs this
  hook; do NOT remove it.
- **`rope_theta` is informational only** (GQA attention reads `rope_base`); annotated in config.py.

None of the above is changed in this pass — no loss/behavior change before the 45K run.
