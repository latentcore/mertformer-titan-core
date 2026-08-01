# MertFormer Chess GUI

Local browser-based GUI for playing against a chess checkpoint once one exists. **No chess
checkpoint has been trained yet** — `scripts/chess_5080_onefile.py`'s training profiles are
built and `READY_FOR_TRAINING` (see `reports/chess_training_readiness_report.md`), but no run
has been launched (see `BACKLOG.md`/the pre/post-45K list). This GUI is the "how to play once
you have `best_by_val_loss.pt`" surface, not a claim that one currently exists.

## Launch

Double-click:
- `launch_mertformer_chess_gui.command`

Or run manually:

```bash
cd apps/chess_gui && ../../.titan-venv/bin/python play_mertformer_chess_web.py
```

## Notes

- Repo location: `apps/chess_gui`
- Uses the local checkpoint under `checkpoints/best_by_val_loss.pt`
- Uses the local training summary under `assets/run_summary.json`
- Heavy runtime files under `checkpoints/` and `logs/` stay local and are intentionally not tracked by git
- Canonical source of truth for the chess engine remains `scripts/chess_5080_onefile.py`
- Device auto-selection (`choose_device()`): CUDA first, then MPS, then CPU — whichever this machine actually has, not Mac-specific (this repo now also runs on Windows/CUDA, per the 2026-07-31 Windows-portability pass)
- Open the printed local URL if the browser does not auto-open
- Stop with `Ctrl+C` in the terminal that launched it
