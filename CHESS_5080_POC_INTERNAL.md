# Chess 5080 PoC Internal

This lane is a private operator-facing chess proof flow.

## Entrypoints
- `bash run.sh --chess-5080-poc`
- `bash zero_touch_start.sh --chess-5080-poc`
- `python scripts/chess_5080_onefile.py`
- `python scripts/export_chess_5080_share.py`

## Intent
- Train a standalone chess-only policy/value model on filtered Lichess data.
- Keep legal move masking mandatory.
- Finish on a single RTX 5080 desktop in roughly 1-4 hours.
- Produce a zipped proof bundle with logs, config, provenance, checkpoints, and SHA256.

## Claim Boundary
- `1600+` is a target, not a guaranteed claim.
- Only measured results may be reported as `verified`.
- If Stockfish comparison is unavailable, the run must be marked `target-not-verified` or `not-run`.
- This lane does not upgrade the core 45K repo claim surface by itself.

## Share Export
- The canonical readable script stays in repo.
- `scripts/export_chess_5080_share.py` generates:
  - open copy
  - obfuscated share copy
  - delivery zip
  - SHA256 file
- The share wrapper may self-delete only after a successful packaged run.
- The repo copy must never self-delete.
