# Post-Train Automation Contract

This contract is the canonical post-train closure state machine for the current working tree.

## Modes

- `full`: benchmark -> golden eval -> demo manifest -> export -> logbook -> docs sync -> governance pack -> release zip -> training outputs bundle -> evidence pack.
- `bench-only`: checkpoint resolution plus benchmark and golden eval.
- `export-only`: checkpoint resolution plus export refresh.
- `demo-only`: checkpoint resolution plus demo manifest and evidence pack summary.
- `readme-update-only`: manifest sync, doc claim gate, governance refresh, and evidence pack summary.
- `check-only`: checkpoint resolution only.
- `plan-only`: emit this contract and the state machine without running commands.
- `dry-run`: emit the execution plan with current checkpoint resolution but do not run mutating commands.

## Failure Policy

- Missing checkpoint is an error for benchmark-driven modes unless `--allow-missing-checkpoint` is set.
- If a command step fails in `full` mode, remaining steps are skipped and the JSON status is marked `failed`.
- Internal manifest-writing steps still run in `demo-only` and `readme-update-only` modes even if no checkpoint is available, but they mark the missing checkpoint explicitly.
- This script never claims trained evidence exists if the checkpoint is missing.
