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

## Repo curation — second pass (2026-06-17): single-persona + canonical scaffold
- **Canonical root scaffold added** (`STATUS/TRUTH_MATRIX/BACKLOG/GOVERNANCE/REPRODUCE.md` + `_TR`)
  so a reviewer has one clear entry instead of competing `reports/` truth matrices. `archive/`
  created; `docs/tr/` intentionally NOT created (TR twins stay co-located per the EN↔TR parity rule).
- **README cut** 178KB→~4KB (full pre-cut README → `archive/`); `sync_manifest.py` now injects a
  pointer to `docs/PROJECT_STRUCTURE.md` instead of re-bloating the front page with the full tree.
- **Anthropic application packet** → `private/anthropic_internal/` (gitignored); **all remaining
  commercial/GTM/outreach** → `private/commercial/` (hand-authored) or gitignored (generated:
  gtm/commercial_handover_pack/legal_ip/investable). The core repo now reads as one persona —
  technical/evidence.
- **Closure cluster:** the redundant/superseded generated reports (duplicate_source_of_truth_report,
  final_truth_constitution, final_master_plan_freeze, release_closure_note, final_freeze_manifest)
  are gitignored (regenerate locally); the hand-authored superseded ones (report_truth_matrix,
  closure_report_build30_v2, presentation_readiness_final) → `archive/closure/`. The remaining
  generated closure/governance reports under `reports/` are intentionally retained as **secondary
  generated detail** that the root scaffold points to — gitignoring the full set would churn the
  verified closure ladder for no reviewer-facing gain. Empty template stubs → `archive/templates/`;
  the ASCII-Turkish `codex_deep_audit_DE` (redundant with `_TR`) → `archive/audits/`.
- **Conflicts fixed at source/generator:** test-count (370), 3.70B→3.67B, the-stack-dedup,
  closure_57 AGI row 53 → out-of-scope + legend, tracked `/Users/` path leaks + guard blind spot.

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

## Training scale — no intermediate run, straight to 45K (2026-06-17)
- **Decision:** there will be **no intermediate-scale model run (e.g. 250M / 500M)** before or
  after the canonical run — not for TÜBİTAK or any other milestone. The next and only training
  job is the **canonical 3.67B 45K run (K1)**.
- **Why:** a smaller *model* has its own training dynamics (MoE 8-top2 and the Liquid/CfC mixer
  behave differently at small scale — the toy-scale ablation already showed the Liquid effect
  washes out into noise), so it does **not** predict 45K behaviour and is not worth the
  compute/time as a "proof" surface.
- **NOT covered by this decision (kept):** the **free pre-45K smoke is not a sub-scale model.**
  It is the *canonical 3.67B architecture itself* run for a few steps via `TITAN_MAX_STEPS=2`
  (no behaviour-param change) plus the K4 checkpoint chain (save → restore → resume). That only
  verifies the code imports/instantiates/steps and that artifacts round-trip — cheap insurance
  against burning expensive 45K compute on a broken import/checkpoint chain (the last H200 run
  died at artifact retrieval). It is a "does the real code run" check, not a "250M deneme".

## Pre-45K de-risk pass — operational hardening applied, training-behavior still deferred (2026-06-17)

Two independent senior reviews (a doc/candidate review and a 6-round 45K code audit) were run; findings
were verified live (file:line) and split into a *safe* set (applied) and a *training-behavior* set (kept
document-only per the rule above). The code review's verdict: no guaranteed crash on the canonical
single-pass path; the real residual de-risk is a 2-GPU smoke test + the run itself.

**Applied — correctness / operational hardening (does NOT change training math, so it cannot confound the run):**
- **grad_clip ratchet removed (`train/train.py`).** The old code permanently mutated `cfg.grad_clip *= 0.7`
  (floor 0.1, never recovering) on any grad-norm spike — a few early BitNet-STE spikes could latch the clip
  at 0.1 and silently over-clip the rest of 45K. The spike is now logged only; the clip stays at the
  configured value (transient). Latent-bug fix, not a tuning change.
- **Atomic checkpoint save (`train/train.py`):** temp file + `os.replace`, so a mid-write kill can no longer
  leave a corrupt `.pt` that breaks the next resume (the exact provider failure we hit before).
- **Resume `torch.load(..., weights_only=False)` (`train/train.py`):** explicit, so GaLore8bit/bnb optimizer
  state unpickles and the crash-recovery path actually works on pinned torch.
- **MoE runtime buffers `persistent=False` (`layers/moe.py`):** `collapse_detected` / `expert_activity_mask`
  + telemetry buffers no longer leak into checkpoints (stale flag across a resume boundary).
- **Removed the in-`MoE.forward` `all_reduce` (`layers/moe.py`):** a collective inside forward re-fires during
  gradient-checkpointing recompute and can interleave with DDP's gradient all_reduces (NCCL-deadlock hazard on
  multi-GPU) + a per-step host sync. The collapse flag is a local heuristic; kept rank-local.
- **Telemetry throttle (`config.py` log_interval 1→10; `train.py` host snapshot every `TITAN_TELEMETRY_INTERVAL=100`):**
  drops the per-step subprocess across 45K. Logging only; no math change.

**Token budget (Q2): 23.6B canonical + 1024 opt-in.** `config/config.py` keeps `batch_size=128` / `23.6B`.
`TITAN_BATCH_SIZE=1024` (Ocean 1024-first profile) is an EXPLICIT opt-in that yields ~188B (8×); the LR schedule
and curriculum are step-based and not rescaled. The `TITAN_STRICT_TOKEN_BUDGET=1` guard hard-fails a >5%
overshoot. Documented in `config/config.py` and `reports/ocean_2xh200_1024_first_launch_profile.md`.

**Still deferred — training-behavior, document-only (same rule as the section above):**
- z-loss effective `2e-6` and Liquid `dt=1.0` (documented above).
- **dropout 0.1 + attention_dropout 0.1 + label_smoothing 0.1** on top of BitNet-STE noise: for an undertrained
  regime (~6.4 tok/param at 23.6B) this stacks four regularizers and may slow learning (LLaMA/PaLM use dropout
  0.0). NOT changed — a run-config tuning call decided at launch.
- **early-stop `patience=5` × `val_check_interval=1000`** could cut 45K early across curriculum-stage shifts.
  NOT changed — monitor at run time / tune at launch.

**Doc / presentation (Q3 — frontier-lab polish):** `TECHNICAL_REPORT.md(+_TR)` reviewer-facing language was made
clinical (dropped the "Synaptic / wisdom / fluid intelligence / living heart / Emotional Weighting / Onyx Storm"
register; §10 reframed as "speculative, not implemented, out of scope"); `ARCHITECTURE.md` Liquid section was
truth-synced to the 12-seed verdict; the inert `orchestrator/` AGI runtime + flag-off cognitive layers are
documented as out-of-scope (not deleted); the MoE-intermediate doc typo was fixed (8192, not 5632 — 5632 is the
dense FFN). closure_57 item 20 renamed from "gqa" to "kv-head sharing" (de-dup vs item 18).

**Teacher-free smoke knob (behavior-neutral):** `config.py` `distill_alpha` is now env-overridable
(`TITAN_DISTILL_ALPHA`, default 0.8 — unchanged default behavior). `TITAN_DISTILL_ALPHA=0` lets the
canonical training entry run a smoke WITHOUT loading/downloading the 70B teacher (train.py only loads
it when `distill_alpha > 0`; loss falls back to pure cross-entropy). This is the one-command teacher-free
pre-45K smoke documented in REPRODUCE(.md/_TR). Not a training-dynamics change for the real run (default
stays 0.8).

**License contradiction resolved → Proprietary (2026-06-17).** README/README_TR footers claimed
`Apache-2.0` while the authoritative `LICENSE` file is **Proprietary & Confidential, all rights reserved**
(and `CONTRIBUTING` closes external contribution, `CITATION.cff` already says `Proprietary`, `pyproject.toml`
points `license = {file = "LICENSE"}`, repo is private + commercial dealroom). The README footers were the
lone error and are fixed to "Proprietary & Confidential — all rights reserved; see LICENSE". Resolved toward
proprietary deliberately: it matches every other metadata surface and is the safe direction (a stray
`Apache-2.0` would be an accidental open-source grant — irreversible). Llama obligations are unchanged and
remain disclosed via `NOTICE` + `MODEL_LICENSE.md` ("Built with Llama"). NOT legal advice; a future decision
to actually open-source (and reconcile Llama-derived components) is a separate, deliberate call.

**External review pass — 5 flagged items, verified live (2026-06-17).** A separate review flagged 5 items;
each was checked against the live repo and only the genuinely-real ones were fixed:
- ✅ **FIXED:** `pyproject.toml` was missing `mertformer_sdk.kernels.cpp` (its `cpp/__init__.py` exists
  alongside metal/vulkan/npu) → added, so `pip install -e .` installs the subpackage.
- ✅ **FIXED (claim discipline):** `TECHNICAL_REPORT(+_TR)` §3.1 "12x acceleration" is now labeled
  **Target/estimate** (not measured at 45K); §7 SHA256 step-chaining reframed to **"designed"** (no completed
  45K chain exists yet).
- ✅ **Self-documented:** `scripts/apply_github_policy.sh` now writes a `note` into `github_policy_report.json`,
  so `return_code: 1` reads as the **EXPECTED best-effort 403** (private/non-Pro branch-protection, §0.8), not a real failure.
- ❌ **NOT a bug (reviewer's stale-knowledge call):** `ci.yml` uses `actions/checkout@v6` + `setup-python@v6`.
  These **exist** (released by mid-2026) and the repo's **own CI shows green `success` runs today** — downgrading
  to v4/v5 would regress a working pipeline. Left as-is.
- ✅ **Already done:** README/README_TR `Apache-2.0`→`Proprietary` was fixed in the prior pass.
- **Llama naming (EXTERNAL-PENDING legal):** the Llama 3.3 Community License carries a "Built with Llama" /
  naming clause. MertFormer Titan uses Llama-3.3 only as a **KD teacher** + the Llama-3 tokenizer (no Llama
  weights), so whether the student name must carry "Llama" is a genuine legal-judgment question, not a clear
  rename. Disclosed via `NOTICE` + `MODEL_LICENSE.md`; flagged for IP/legal review before any public model
  release. Not renamed on an uncertain reading.

**Findings-list cleanup + whitepaper fast-forward sync (2026-06-17, Pass 4.5).** An app/web Claude added an
arXiv submission section (Section 8) to `WHITE_PAPER_LIQUIDROUTER(.md/_TR.md)` and pushed via the GitHub API
(commits `eed0a4d` EN + `0e8f040` TR); the local working copy was a **clean fast-forward behind**, synced with
`git merge --ff-only origin/main` (no divergence/conflict). A separate "final findings list" (F/P/D/C/L/S/B/A)
was verified live and only the genuinely-real, safe items were fixed:
- **Docs (EN+TR):** WHITE_PAPER_TR §2 "Statik"→"Durumsuz (Stateless)"; CHANGELOG (+TR) gained a 2026-06-17
  section (June work); IMPLEMENTATION_PLAN (+TR) Founders-Hub line de-staled (submitted 2026-05-31); TASK (+TR)
  repointed moved-to-`private/commercial/` refs; README_CHECKLIST (+TR) refreshed (Antigravity/2026-02-05 +
  removed Mermaid/ASCII rows post-4KB-cut); `ablations/results(.md/_TR)` + `no_liquid/README(.md/_TR)`
  truth-synced to the 12-seed verdict (single-seed +0.50 pilot labeled **superseded**); AGENTS.md + START_HERE.md
  annotate the closure-ladder-generated (gitignored) `reports/final_*` files as "regenerate locally".
- **Config/security:** `.pre-commit-config.yaml` ruff→v0.15.5 (matches `constraints.txt`); `config/base.yaml`
  log_interval 1→10; `Dockerfile` now applies `constraints.txt`; `registry/mertformer_v0.1.json` v27.0→Build 30 V2;
  `scripts/secret_scan.py` + `policy/allow_deny_policy.yaml` GitHub-token patterns extended to gho_/ghu_/ghs_/ghr_ +
  fine-grained PAT (kept in sync); `repro/pip_freeze.txt` chess line added.
- **Verified NOT-a-bug (left as-is):** moe_dispatch_mode parallel-vs-DECISIONS-sequential (no contradiction —
  sequential is documented GPU-perf-only deferred); secret_scan↔policy regex "mismatch" (patterns were identical —
  the real gap was the missing `gho_`, now fixed in both); CI `actions/*@v6` (exists; repo CI green).
- **Verified intentional (left):** env.lock/cuda.lock placeholders (filled on training hardware); `warmup_steps`
  dead field (already self-documented "informational only" — not touched pre-45K); ADR-0001's reference to the
  gitignored `final_truth_constitution.md` (historical ADR kept immutable); dataset license-TBD entries (disabled,
  verified at enable time).
- **Pre-45K freeze (untouched):** all post-45K backlog code items (z-loss/dt/dispatch/`.item()`/opset/RMSNorm/
  residual_scale/GaLore/8bit-Adam/val_steps) and architecture observations — document-only, to avoid confounding the run.
- **User action (cannot be done from here):** revoke the `gho_` OAuth token visible in the parallel chat
  (GitHub → Settings → Developer settings → revoke). Not committed to the repo; repo risk low.

**NİHAİ EKSİKLİKLER cleanup — verified subset (2026-06-18, Pass 4.6).** A parallel app/web Claude compiled an
all-day "final deficiency list" (~207 items spanning repo + GitHub + 2 gists + life/personal/career). Each was
re-verified live (3 Explore agents); only the genuinely-real, safe items were fixed (training-math untouched,
pytest 370 unchanged):
- **Stale version markers (cosmetic, runtime-zero):** removed `BUILD 27` / `V27.0` prefixes from 14 comment/print/
  docstring sites (`config/config.py`, `train/train.py`, `layers/mla.py`, `layers/mertformer_block.py`, `run.sh`,
  `scripts/smart_runner.py`, `scripts/mac_simulation.py`) — comments only, no logic. (`version_checker.py` keeps its
  banned-token data.)
- **Reviewer-facing identity docs (truth-sync):** `CITATION.cff` — removed false `date-released: 2026-03-13` (model
  not released), dropped "(Onyx Storm)" to match README, added author affiliation. `LICENSE`(+`_TR`) — "MertFormer
  AI Team" → "Mert Yünlü" (solo author), softened "partners/investors" overclaim, removed the self-contradicting
  "confidential Swarm Orchestrator / LiquidRouter" naming (LiquidRouter is public in the whitepaper).
  `SECURITY.md`(+`_TR`) — stripped internal jargon ("45K ship gate", "Medium Refine", "Build 30 Max Closure"), added
  a real vuln-reporting channel (GitHub Security Advisories). `CONTRIBUTING.md`(+`_TR`) — `codex/`→`feature/`/`fix/`,
  added an honest AI-assistance disclosure (covers the `Co-Authored-By: Claude` git-history concern without rewriting
  history). `requirements.txt` header → Build 30 V2.
- **Observability (additive, canonical path):** `orchestrator/distillation_manager.py` resume-state `except: pass`
  now logs the failure before the (unchanged) shard-count fallback — useful for a long 45K resume.
- **Docs/parity:** `TROUBLESHOOTING.md`(+`_TR`) gained NCCL-hang / checkpoint-corruption / tokenizer-mismatch
  scenarios; created EN twins for two tiny TR-orphans (`training_dynamics/cold_vs_warm.md`,
  `experiments/exp_001_baseline/notes.md`).
- **Verified NOT-a-bug / FALSE finding (left):** "git tag not pushed" is FALSE (all 13 tags incl. `v1.0-TITAN-BUILD30-V2`
  are on remote); moe_dispatch parallel-vs-sequential, CI `@v6`, secret_scan↔policy regex — all previously confirmed.
  Pass-4.5 fixes (secret_scan `gho_`, base.yaml, registry, pip_freeze, ruff, CHANGELOG) re-verified present.
- **Verified real but LEFT (quarantine / freeze / external):** `orchestrator/core.py` gpt2-fallback +
  `self_audit.py` silent-except are in the inert out-of-scope AGI scaffold (quarantined, not in the canonical path).
  All pre-45K-freeze architecture items (z-loss/dt/dispatch/mark_weights_updated/warmup_steps/dropout/GaLore/
  8bit-Adam/val_steps/opset/LiquidRouter-naming) untouched. Post-45K/infra (22 test gaps, ML-science ablations, MLOps,
  ethics/RAI, attack analyses) deferred. TR-orphans whose EN lives in `archive/templates/` (economics) left as legacy;
  EN-only docs (ARCHITECTURE/START_HERE/...) intentional (English-primary).
- **Skipped with reason:** README CI badge — repo is PRIVATE, a live Actions badge 404s externally and would force a
  gist `4_README` re-sync; the inline "370 passed" status already conveys it.
- **External / user action:** life/personal/career items live in the external Grand Master life doc (app sandbox), not
  this repo; the `gho_` token revoke remains the user's to do.
