# Logs Overview

This directory contains **generated artifacts** (run logs, preflight output, operator-gate evidence).

Policy:
- `logs/` is **gitignored by default** (artifacts are not committed).
- The only tracked file under `logs/` is this `logs/README.md`.

## Common Artifacts (Untracked)
- `logs/preflight/titan_preflight.log` (preflight diagnostics)
- `logs/operator_mode/*.jsonl` + `*.manifest.json` (operator gate evidence)
- `logs/run_*.jsonl` and `logs/run_*.csv` (training loop metrics)
- `logs/production_run.log` (can be large)
- `logs/ALL_LOGS.jsonl` (unified logbook)

## Unified Logbook
- Build/append: `.titan-venv/bin/python scripts/logbook_build.py --append`
- Rebuild from scratch: `.titan-venv/bin/python scripts/logbook_build.py --rebuild`
- The logbook is **append-only** and includes source metadata for each imported log line.

## Notes
- Logs are **sanitized** (tokens are masked in preflight output).
- For fresh runs, regenerate artifacts via `run.sh --test` or `run.sh`.
