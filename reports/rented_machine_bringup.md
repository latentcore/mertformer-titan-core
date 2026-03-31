# Rented Machine Bring-Up

1. Transfer the repo and package artifacts.
2. Run `bash scripts/bootstrap_venv.sh` if the venv is absent.
3. Run `bash zero_touch_start.sh --check-only` on the target machine.
4. Run `bash zero_touch_start.sh` using the intended lane and credentials.
5. Collect `reports/final_orchestrator_status.json`, checkpoints, benchmark outputs, and final evidence pack.
