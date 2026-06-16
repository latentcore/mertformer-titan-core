# Start Here

This is the shortest truthful path through the repository.

## What This Repo Is
MertFormer Titan is an offline-first, auditable AI systems project with three connected review lanes:
- systems and performance infrastructure
- offline assistant and RAG foundations
- chess proof and teaching discipline

## Current Exact Status
- Current maturity: `pilot-ready pre-training baseline`
- Current repo-side readiness: `TRAIN_ALLOWED`
- Current readiness reason code: `READY_REMOTE_BOOTSTRAP`
- Preferred serious validation target: `45K`
- Exact `45K` is not the only acceptable application threshold
- Recommended rented-machine semantics: `remote_bootstrap` with runtime-injected `HF_TOKEN` and target-machine dataset/bootstrap flow
- Strict local `offline_clean` semantics: strict precomputed KD with fixed teacher surface `meta-llama/Llama-3.3-70B-Instruct`
- Real owned training run + checkpoint-bound evidence are still required for application readiness
- Remaining non-winning blockers: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`
- Real trained checkpoints, benchmark outputs, trained demo bundles, and trained export measurements remain post-run evidence

## What To Read In Order
1. `README_SUMMARY.md` - shortest external summary
2. `docs/PROJECT_MASTER_TRUTH.md` - compact whole-repo truth table
3. `reports/final_truth_matrix.md` - claim-by-claim evidence map
4. `reports/known_limits_v1.md` - current hard limits and missing evidence boundary
5. `reports/systems_performance_case_study.md` - systems and low-bit runtime story
6. `reports/offline_assistant_case_study.md` - offline assistant and RAG story
7. `reports/chess_proof_teaching_case_study.md` - benchmark honesty and teaching discipline story
8. `reports/final_backlog_missing_items.md` - exact remaining blocker classes
9. `STATUS.md` - canonical status entry point
10. `MISSION.md` - mission framing and claim philosophy

## Canonical Commands
- Verify the repo: `bash scripts/verify_all.sh`
- Check readiness only: `bash zero_touch_start.sh --check-only`
- Launch the canonical owned training lane: `bash zero_touch_start.sh`
- Launch the canonical Kaggle closure lane: `bash zero_touch_start.sh --kaggle-onefile --mode train-end --profile auto`
- Verify Kaggle wiring only: `bash zero_touch_start.sh --kaggle-onefile --mode verify`
- macOS one-click launcher: `launch_mertformer_kaggle_closure.command`
- Refresh closure, artifacts, and hashes: `bash scripts/final_one_shot.sh`
- One-command closure flow (`SOP` = `Standard Operating Procedure`): `bash scripts/one_command_full_sop.sh`
- Optional Phase-0 helper: `python3 scripts/precompute_logits_topk.py --all-stages`

## Phase-0 Note
- `zero_touch_start.sh` now skips optional Phase-0 precompute for `--plan-only`, `--dry-run`, `--check-only`, and post-only invocations.
- Real training invocations may auto-start Top-K teacher logit precompute when stage datasets exist and `HF_TOKEN` is supplied.
- Canonical `offline_clean` does not silently fall back to teacherless mode; if logits are still incomplete and Phase-0 is not actionable, the lane stays blocked.
- Override knobs: `TITAN_SKIP_PHASE0=1`, `TITAN_TOP_K=<n>`, `TITAN_PRECOMPUTE_BATCH=<n>`.

## Anthropic-Relevant Signals
- explicit measured vs unmeasured claim boundary
- training efficiency and experiment-discipline story
- low-bit runtime plus backend-routing honesty
- offline-first, governance-gated assistant foundations
- benchmark discipline that keeps product mode and proof mode separate

## What Is Still Open
The remaining high-value open class is not missing folders or missing scripts. It is the missing post-run evidence class:
- real owned training run
- trained final weights
- best/latest checkpoint proof
- checkpoint-bound benchmark outputs
- trained demo bundle
- export/device evidence as a strong plus
- downloadable `artifacts/mertformer_training_outputs_bundle.zip` from a real target-machine run

Everything else in this closure pack is there to keep that boundary explicit, reviewer-friendly, and hard to overclaim.

## Kaggle Canon Notes
- The canonical Kaggle one-file surface is `scripts/kaggle_onefile_closure_build30.py`.
- Legacy/reference lanes remain available in `scripts/kaggle_onefile_demo_build30.py` and `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py`.
- Runtime GPU availability is account-dependent; the canonical lane detects `GPU P100`, `GPU T4 x2`, or fallback hardware at runtime instead of claiming a fixed Kaggle entitlement.
