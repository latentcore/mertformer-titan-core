# Changelog

All notable changes to this project are tracked in this file.

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
