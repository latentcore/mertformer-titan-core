# Changelog

All notable changes to this project are tracked in this file.

## Unreleased - 2026-06-17

### Added
- WHITE_PAPER_LIQUIDROUTER (EN+TR): arXiv submission section (Section 8) with proposed title — gated post-45K.
- `TITAN_DISTILL_ALPHA` env knob enabling a teacher-free pre-45K smoke (the 70B teacher is never downloaded when set to 0).
- README_SUMMARY (EN+TR): "Architecture at a glance" component table (BitNet / GQA / sparse MoE / Liquid-CfC).
- Canonical root scaffold (STATUS / TRUTH_MATRIX / BACKLOG / GOVERNANCE / REPRODUCE, EN+TR) as the single reviewer entry point.

### Changed
- Attention class renamed MLA → GQA (always grouped-query; filename `layers/mla.py` kept for manifest/SHA stability).
- 45K-pre-run operational hardening: atomic checkpoint save (`os.replace`), removed the in-forward MoE collapse-flag `all_reduce`, telemetry buffers `persistent=False`, removed the permanent grad-clip ratchet (now transient), telemetry throttle.
- Liquid ablation canonicalized to the 12-seed verdict (OFF 96.32% / ON 94.69%, Δ−1.63 pp, p=0.305, inconclusive — no measured benefit, ~30% slower); the single-seed +0.50 pilot is superseded across ablation surfaces.
- TECHNICAL_REPORT (EN+TR) clinical rewrite; §3.1 "12x" relabeled Target/estimate; §7 SHA256 step-chaining relabeled "designed"; MoE expert intermediate corrected to 8192.
- README cut 178KB → ~4KB (full snapshot archived); single-persona technical/evidence surface; commercial/GTM material moved under `private/`.
- License surface resolved to Proprietary & Confidential across README / README_TR (matching LICENSE).

### Fixed
- `pyproject.toml`: added the missing `mertformer_sdk.kernels.cpp` package.
- `.pre-commit-config.yaml`: ruff pinned to v0.15.5 (matches constraints.txt).
- `Dockerfile`: now applies `constraints.txt` for reproducible builds.
- `registry/mertformer_v0.1.json`: version synced to Build 30 V2 (was v27.0).
- `scripts/secret_scan.py` + `policy/allow_deny_policy.yaml`: GitHub-token patterns extended to gho_/ghu_/ghs_/ghr_ and fine-grained PAT.

### Validation
- `370 passed, 4 skipped` (offline-first pytest); readiness `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. No trained/benchmark claim — the one remaining gap is a real 45K GPU run.

## Unreleased - 2026-05-24

### Added
- Optional, default-off packed projection controls for FFN, MoE BitSwiGLU, and MLA K/V training paths.
- Environment-overridable training controls for batch size, log interval, validation interval, checkpoint interval, and DataLoader transfer behavior.
- Optional 8-GPU Accelerate profile under `repro/accelerate_8xgpu.yaml`.
- Equivalence coverage for packed projection and Liquid train implementation variants.

### Changed
- README, usage guide, training plan, feature-flag governance, script catalog, and verification matrix now document the optional speed-control surface with explicit claim boundaries.
- Documentation clarifies that `repro/` holds reproducibility/run configs, while `configs/` remains reserved for stable named configuration contracts.

### Validation
- Optional speed flags remain off by default and require equivalence tests plus target-machine logs before any speed claim.

## v1.0.0-build30-v2 - 2026-03-13

### Added
- Cross-dataset deduplication path in data pipeline.
- MoE parallel dispatch mode and CfC fast path toggle.
- Onefile demo CLI enhancements + training log dashboard script.
- SOP tolerance check for CfC/MoE loss parity.

### Changed
- Build 30 V2 version sync across core docs and model metadata.
- Training token budget defaulted to fixed-steps gating.

### Validation
- SOP full run (verify_all, md_quality, linkcheck, sync_manifest) PASS.

## v0.1.0-pilot-ready - 2026-02-08

### Added
- Pilot report contract: `interfaces/pilot_report_v1.schema.json`.
- SDK pilot helpers and CLI commands:
  - `mertformer verify`
  - `mertformer pilot-report --out <json>`
- SITL proof flow for drone-class offline evidence:
  - `scripts/drone_sitl_demo.py`
  - `reports/drone_sitl_demo.md`
  - `reports/drone_sitl_demo_TR.md`
  - `reports/pilots/README.md`
  - `reports/pilots/README_TR.md`
- Pilot business docs:
  - `reports/pilot_readiness_kit.md` + `_TR`
  - `reports/pilot_offer_packages.md` + `_TR`
  - `reports/sales_funnel_90d.md` + `_TR`
- Clean-room verification report:
  - `reports/cleanroom_verification.md`
  - `reports/cleanroom_verification_TR.md`
- Pilot acceptance signature template:
  - `reports/pilot_acceptance_signoff.md`
  - `reports/pilot_acceptance_signoff_TR.md`

### Changed
- Benchmark claim safety gate is strict: missing checkpoint now reports `NOT ELIGIBLE FOR CLAIM`.
- README claim language tightened: measured vs target/estimate separated.
- Absolute Desktop paths removed from tracked artifacts.
- Turkish document parity and wording normalization improved across TR docs.

### Fixed
- Strict checkpoint guard in SDK load path to prevent accidental random-weight pilot runs.
- Docs index and project structure blocks synchronized with tracked files.

### Validation
- `python3 -m pytest -q` passed.
- `ruff check .` passed.
- `bash scripts/verify_all.sh` passed.

## Pass 7 (2026-06-13) — Mac-doable backlog zeroed + $0 Kaggle pilot
- Added `scripts/run_liquid_ablation.py` + `docs/KAGGLE_PILOT.md`: a free LiquidRouter ON-vs-OFF
  ablation pilot (~80–100M, pure CE, no teacher) — the single domino that unlocks the GPU-gated work.
- LatentODE per-batch reset during training (no cross-batch state leak); MoE collapse flag DDP
  all-reduce (guarded, no-op off-DDP); liquid-impl benchmark script; coverage config.
- Docs: ARCHITECTURE.md Projections + stage-3 note; CPU quickstart. Backlog dispositions in DECISIONS.md.
- Invariants held: param count locked; pytest green; ruff + scoped mypy + verify_all green.
