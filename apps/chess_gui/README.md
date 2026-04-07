# MertFormer Chess GUI

Local browser-based GUI for the trained `best_by_val_loss.pt` checkpoint.

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
- Chooses `mps` automatically on this Mac when available
- Open the printed local URL if the browser does not auto-open
- Stop with `Ctrl+C` in the terminal that launched it
