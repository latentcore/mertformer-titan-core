# Logs Overview

This directory contains **selected** run logs for reproducibility and audit trails.

## Included in GitHub
- `logs/preflight/` (preflight diagnostics)
- `logs/operator_mode/` (operator gate evidence)
- `logs/run_*.jsonl` and `logs/run_*.csv` (lightweight local run logs)

## Not included by default
- `logs/production_run.log` (can be large; kept local unless explicitly requested)

## Notes
- Logs are **sanitized** (tokens are masked in preflight output).
- For fresh runs, regenerate logs via `run.sh --test` or `run.sh`.
