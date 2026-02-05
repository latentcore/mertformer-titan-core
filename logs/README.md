# Logs Overview

This directory contains **selected** run logs for reproducibility and audit trails.

## Included in GitHub
- `logs/preflight/` (preflight diagnostics)
- `logs/operator_mode/` (operator gate evidence)
- `logs/run_*.jsonl` and `logs/run_*.csv` (lightweight local run logs)
- `logs/ALL_LOGS.jsonl` (unified logbook, all logs in one file)

## Not included by default
- `logs/production_run.log` (can be large; kept local unless explicitly requested)

## Unified Logbook
- Build/append: `python3 scripts/logbook_build.py --append`
- Rebuild from scratch: `python3 scripts/logbook_build.py --rebuild`
- The logbook is **append-only** and includes source metadata for each log line.

## Notes
- Logs are **sanitized** (tokens are masked in preflight output).
- For fresh runs, regenerate logs via `run.sh --test` or `run.sh`.
