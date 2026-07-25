# Changelog

All notable changes to this project are tracked in this file.

> **Maintenance note (added 2026-07-25):** this file is hand-maintained, not auto-regenerated — it drifted a full month (2026-06-28 → 2026-07-25) before this note existed. Any closure pass that lands a real `BACKLOG.md`/`DECISIONS.md` entry should also add/update the current `## Unreleased - <date>` section here (EN) and in `CHANGELOG_TR.md` (TR) — a short summary is enough, full detail stays in `BACKLOG.md`/`DECISIONS.md`. See `reports/change_control_sop.md`.

## Unreleased - 2026-07-25

### Added
- `scripts/pre45k_gate.py`/`.sh` + `scripts/ddp_smoke.py`: chains the offline preflight, a dry-run preview, and a real 2-GPU DDP smoke test into one pre-spend launch-readiness gate; writes `reports/pre45k_gate_report.{json,md}`.
- `scripts/kaggle_batch_runner.py`: unattended multi-job Kaggle orchestrator; produced 4 real evidence sets under `evidence/2026-07-25-*` (Nutrition5k Liquid-OFF/MoE-OFF ablations, 36M/171M LM re-verification).
- `utils/divergence_guard.py` gained an independent grad-norm EMA co-trigger ("C1") alongside the existing loss-based brake — confirmed firing correctly on real 36M/171M hardware.
- `scripts/offsite_backup_watcher.py`, `runbooks/checkpoint_offsite_backup.md`, `train/trainer_core.py::get_rewarmup_schedule()` (post-45K LR re-warmup).
- `tests/test_atomic_write_hygiene.py`: atomic (temp+`os.replace`) writes for 5 pipeline files previously trusted via a bare `.exists()` check.
- `model/nutrition_vision.py` + `scripts/{train,predict,evaluate}_nutrition5k.py`: a bounded vision side-experiment reusing the real BitLinear/MoE/Liquid trunk unmodified; a real trained + independently-re-verified checkpoint, then a real comparative ablation (see Changed).

### Fixed
- z-loss effective weight: an accidental double-multiply left it ~500x below the Switch-Transformer/ST-MoE convention; `z_loss_coef` corrected `1e-4 → 0.05`.
- `generate()` never threaded the Liquid/CfC hidden state across decode steps — a silent no-op during generation; fixed, with a full-forward↔incremental-decode parity test.
- `bigcode/the-stack-dedup` revision/sha256 finally pinned (a dataset-ref scanner false-positive had blocked it for months).
- `scripts/kaggle_batch_runner.py::run_chess()` invocation bug (wrong `sys.path`) found live during a real Kaggle run and fixed.
- `layers/moe.py` MoE dispatch-parallel `torch.bincount` → `scatter_add_` (MPS/older-torch portability).

### Changed
- LR regime (`1.5e-3 → 3e-4`, sweep start not verified-safe), Liquid spike guard (absolute → EMA-relative), WSD scheduler clamp — all candidate fixes, applied and re-tested on real RTX-5070/Kaggle hardware but not yet proven sufficient (see Validation).
- Eight launch-time decisions locked (`DECISIONS.md`): lane = `online_teacher`, Liquid = Keep, model size = 3.67B canonical, `top_k` = 32 (not 256), 2 dead Stage-5 datasets replaced with a verified live one, 3 license-TBD datasets kept-and-documented, Stage-3 TR/synthetic ratio ratified, INT-KERNEL claim relabeled honest (fp-simulation, no real ternary kernel yet).
- Public gist reorganized: Nutrition5k promoted to the front, a real z-loss arithmetic error corrected (`~50x` → `~500x`), the one-pager's pitch/investor framing replaced with research framing.

### Validation
- `622 passed, 5 skipped` (was `370 passed, 4 skipped` at the last entry). Third real-hardware confirmation (2026-07-02 / 07-12 / 07-25) that this architecture still diverges at small scale without further LR/optimizer work — the new grad-norm safety brake (C1) is now confirmed catching it cleanly at two scales instead of an uncontrolled explosion. Readiness unchanged: `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. No trained/benchmark claim — the one remaining gap is a real 45K GPU run. Full pass-by-pass detail: `BACKLOG.md`, `DECISIONS.md`.

## Unreleased - 2026-06-28

### Added
- `scripts/flip_status_banner.py`: **report-only** status-banner auditor — lists tracked files carrying the pre-training banner and reports post-flip eligibility (a real, non-zero eval metric). It has **no write path**: the actual evidence-gated flip is a deliberate post-run task (a naive "checkpoint+summary exists" gate is satisfied by a stray demo checkpoint + stub summary, so a pre-built auto-writer is unsafe). See BACKLOG.
- `ENV_VARS.md`: single index of the canonical training/precompute/orchestration environment variables with defaults.

### Fixed
- `eval/gsm8k.py`: checkpoint load now uses `weights_only=False` (+ `_orig_mod.` key normalization, non-strict load), mirroring the documented `train.py` resume path — prevents a torch≥2.6 `UnpicklingError` when evaluating a real training checkpoint (optimizer/GaLore state) in the post-45K GSM8K benchmark.

### Changed
- Banner/version hygiene: normalized non-frozen `Status` / `Version` / `__version__` banners to the canonical Build-30-V2 form (`utils/logger.py`, `orchestrator/*`, `scripts/*`). Comment/metadata only, zero runtime change. Frozen-path banners (`model/`, `train/`, `layers/`) are deliberately left for the post-45K evidence-gated flip.

### Validation
- `370 passed, 4 skipped` (offline-first pytest, unchanged); readiness `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. No trained/benchmark claim — the one remaining gap is a real 45K GPU run.

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
