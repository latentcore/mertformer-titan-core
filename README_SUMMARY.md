![MertFormer Titan Header](assets/header.png)

Language: [English](README_SUMMARY.md) | [Turkce](README_SUMMARY_TR.md)

---

# MertFormer Titan - External Summary (Build 30 V2)

## What This Project Is
MertFormer Titan is an offline-first, auditable AI systems repository built around low-bit runtime infrastructure, local assistant foundations, and disciplined evaluation surfaces.

Long-term target: reduce the cost barrier for auditable AI training and inference for compute-constrained developers, small teams, and local institutions. This is a target, not a benchmark claim, until checkpoint-bound runs and target-hardware measurements exist.

## Current Exact State
- Stage: `pilot-ready pre-training baseline`
- Repo-side readiness: `TRAIN_ALLOWED`
- Exact reason code: `READY_REMOTE_BOOTSTRAP`
- Recommended repo-side lane: `remote_bootstrap`
- Strict local lane: `offline_clean`
- Preferred serious validation target: `45K`
- Application gate: real owned training run + checkpoint-bound evidence
- Remaining non-winning blockers: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`

## What Matters For Review
- No trained checkpoint claims are made yet.
- Benchmark status remains `NOT ELIGIBLE FOR CLAIM` until a trained checkpoint exists.
- Exact `45K` is preferred, but application readiness is defined by a meaningful real training run with checkpoint-bound evidence.
- Export/device evidence is a strong plus, not a hard blocker.

## Shortest Review Path
1. [START_HERE.md](START_HERE.md)
2. [docs/PROJECT_MASTER_TRUTH.md](docs/PROJECT_MASTER_TRUTH.md)
3. [reports/final_truth_matrix.md](reports/final_truth_matrix.md)
4. [reports/known_limits_v1.md](reports/known_limits_v1.md)
5. [reports/systems_performance_case_study.md](reports/systems_performance_case_study.md)
6. [reports/offline_assistant_case_study.md](reports/offline_assistant_case_study.md)
7. [reports/chess_proof_teaching_case_study.md](reports/chess_proof_teaching_case_study.md)
8. [STATUS.md](STATUS.md)

## Canonical Commands
```bash
bash scripts/bootstrap_venv.sh
bash scripts/verify_all.sh
bash zero_touch_start.sh --check-only
bash zero_touch_start.sh
bash scripts/final_one_shot.sh
```

## Strongest Signals
- training efficiency and systems-debugging discipline
- backend routing and fallback honesty
- offline-first, governance-gated assistant foundations
- claim-safe verification and repo truth sync
- compute-accessibility positioning for lower-cost auditable training/inference, with target status kept separate from measured evidence

## Open Post-Run Evidence Class
- trained final weights
- best/latest checkpoint proof
- checkpoint-bound benchmark outputs
- trained demo bundle
- trained export/device measurements
