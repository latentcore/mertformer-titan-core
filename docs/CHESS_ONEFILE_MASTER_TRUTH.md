# Chess Onefile Master Truth

This document summarizes the current repo-side closure status of the chess onefile lane.

It is intentionally strict about the difference between:

- repo-side implementation completeness
- actual trained evidence
- external validation
- release-grade closure

## Current Rule

- `repo-side complete` means the code, tests, and documentation surface exist and are wired into the onefile closure chain.
- `real closure blocked` means a real run, measured artifact, external sign-off, or management decision is still missing.

## Aggregated Master Table

| Lane | Repo-Side Complete | Real Closure Blocked | Why It Is Still Blocked |
|---|---|---|---|
| `release_registry` | `true` | `true` | Internal registry surfaces exist, but release-grade proof is not granted by local artifacts alone. |
| `external_closure` | `true` | `true` | External reproducibility, legal, security, and pilot closure are still external work. |
| `operational_closure` | `true` | `true` | Operator handbook, DR evidence, retention policy, and blind handoff still need real rehearsal. |
| `release_governance` | `true` | `true` | Release notes, freeze sign-off, changelog review, and maintenance policy still need formal governance. |
| `device_packaging` | `true` | `true` | Export truth, device validation, packaging validation, and installer validation still need measured runs. |
| `benchmark_closure` | `true` | `true` | Raw outputs, compare reports, benchmark summaries, and locked manifests are still not real measured closure. |
| `training_accounting` | `true` | `true` | Training report, token accounting, compute accounting, and cost reporting still need real measured inputs. |
| `trained_artifact_truth` | `true` | `true` | Final weights, best/latest checkpoint truth, and trained artifact registry still need validated trained outputs. |
| `management_closure` | `true` | `true` | Core-complete, research separation, maintenance-only, and final closure decisions still need management sign-off. |
| `truth_docs_alignment` | `true` | `false` | Canonical chess/project truth docs and generated truth reports are currently synchronized. |
| `project_actionability` | `true` | `false` | Project blockers now have an execution order plus 40+ derivative actionability reports (dependency graph, lane board, phase plan, owner accountability, critical path, release-path, cutover, and signoff-cutset reports, among others) wired into the repo-side truth chain. |
| `generated_truth_consistency` | `true` | `false` | Generated summary reports and blocker/action-plan layers are currently internally consistent. |

## Repo-Side Complete Surfaces

- Feature-bundle and feature-flag system
- Auxiliary chess heads
- Self-play, tournament, and replay-buffer reporting surfaces
- Closure manifests
- Release/evidence registry
- Claim registry, known limits, support matrix, release gate summary
- RC/golden/handoff stub surfaces
- External, pilot, security, legal stub surfaces
- Operator handbook, DR, backup retention, blind handoff stub surfaces
- Release-governance stub surfaces
- Device/export/packaging stub surfaces
- Benchmark-closure stub surfaces
- Training/accounting stub surfaces
- Trained-artifact-truth stub surfaces
- Management-closure stub surfaces
- Master summary and aggregated truth surfaces

## Related, separate package: `ChessFormerAI/chessformer`

An independent rewrite of this lane (`ChessFormerAI/chessformer`, developed against a
read-only mirror of this repo's `layers/`; not part of this repo's own training pipeline or
test suite, though a minimal inference+eval subset is vendored at
`evidence/2026-08-02-chess-searchless-5070/chessformer/` for reproducibility) has a real, checkpoint-bound
training run on target consumer hardware (RTX 5070) with real measured evaluation: puzzle
accuracy 45.78% (directly comparable to DeepMind's Searchless Chess, arXiv:2402.04494) and an
Elo estimate of 1509 on Stockfish's own internal rating scale (not comparable to DeepMind's
2895 Lichess-blitz-vs-humans figure). Full detail: `evidence/2026-08-02-chess-searchless-5070/`,
`BACKLOG.md`, `DECISIONS.md`.

This does **not** change anything in this document's own table above — `scripts/
chess_5080_onefile.py` itself is unchanged, still never run, and the two packages have
different report schemas, different governance apparatus, and (as of 2026-08-06) a deliberate
decision to stay separate rather than merge. `chessformer` fixed four real bugs found in
`scripts/chess_5080_onefile.py` by an earlier independent review (`is_causal` mislabeling, a
Liquid clamp train/eval mismatch, a MoE dispatch capacity drift, a Liquid-state-discard bug);
those bugs remain live in `chess_5080_onefile.py` itself.

## Real Remaining Core Work

The following remain actual blockers even after repo-side closure is strong:

- `external_strength_unproven`
- `real_training_outputs_pending`
- `external_reproduction_pending`
- `security_legal_pilot_pending`
- `operator_handoff_dr_pending`
- `rc_golden_final_release_pending`
- `export_device_packaging_pending`
- `benchmark_evidence_pending`
- `trained_artifact_truth_pending`
- `management_closure_pending`

## Bottom Line

- The chess onefile lane now has a strong repo-side closure framework.
- The repo is much closer to being operationally auditable than it was before.
- It is still not honest to call the lane fully finished.
- The main missing class is no longer "missing code surface".
- The main missing class is now "real evidence, external proof, and final closure decisions".
