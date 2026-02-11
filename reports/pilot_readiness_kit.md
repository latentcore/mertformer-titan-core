# Pilot Readiness Kit (B2B)

## Purpose
This kit defines the minimum technical evidence package required before any paid pilot starts.

## Scope
- Offline-first verification and operational safety gates
- Deliverable structure for enterprise pilot acceptance
- Risk and limitation disclosure for pre-training stage

## Mandatory Technical Evidence
1. `bash scripts/verify_all.sh` result: PASS
2. `TITAN_OFFLINE=1 bash run.sh --test` result: PASS
3. Operator-mode gate summary with step-level statuses
4. `pilot_report_v1` JSON output (`mertformer pilot-report --out <path>`)

## Offline Procedure (Customer Environment)
1. Create environment with `bash scripts/bootstrap_venv.sh`
2. Run `mertformer verify` (offline-only)
3. Run `mertformer pilot-report --out reports/pilot_report.json`
4. Attach logs and report to pilot acceptance package

## Acceptance Criteria (Technical)
1. Secret scan, pytest, preflight, operator gate all pass
2. No network dependency required for verification commands
3. Gate outputs and pilot report fields are consistent
4. No machine-specific absolute paths in tracked docs

## Risk and Limits (Pre-Training)
1. Full model quality benchmarks are not claim-eligible without a trained checkpoint
2. Device latency/energy claims remain target estimates until measured
3. Low-bit kernel is experimental and opt-in

