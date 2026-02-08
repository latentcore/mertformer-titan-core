# Drone SITL Demo Protocol (Proof-of-System)

## Scope
This demo is a software-in-the-loop (SITL) **evidence flow** for autonomous drone-class behavior under constraints. It is not a real-flight claim.

## Goal
Produce reproducible logs showing:
1. Offline decision loop
2. Deterministic fail-safe fallback on confidence breach
3. Recovery after transient sensor fault

## Runner
- Script: `scripts/drone_sitl_demo.py`
- Output root: `reports/pilots/<pilot_id>/sitl_<timestamp>/`

## Command
```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --runs 3 --steps 120
```

## Expected Outputs
1. `sitl_events.jsonl` (step-level event log)
2. `sitl_summary.json` (run-level and aggregate summary)
3. `sitl_report.md` (human-readable proof note)

## Pass Criteria
1. Each run must trigger at least one fail-safe fallback
2. Recovery must be observed after injected fault window
3. Aggregate status must be `all_green=true`

## Notes
- This is a deterministic simulator for pilot evidence.
- Real UAV integration remains out of scope in this closure cycle.
