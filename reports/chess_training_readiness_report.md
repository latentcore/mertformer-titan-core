# Chess Training Readiness Report

- final_status: `READY_FOR_TRAINING`
- required_green: `7/7`
- canonical_train_command: `python3 scripts/chess_5080_onefile.py --mode train`
- canonical_verify_command: `python3 scripts/chess_5080_onefile.py --mode verify`

## Checks

| Check | Required | Status | Detail |
| --- | --- | --- | --- |
| `canonical_onefile` | `True` | `True` | <REPO_ROOT>/scripts/chess_5080_onefile.py |
| `share_export` | `True` | `True` | <REPO_ROOT>/scripts/export_chess_5080_share.py |
| `windows_builder` | `True` | `True` | <REPO_ROOT>/scripts/build_chess_5080_windows_delivery.py |
| `gui_app` | `True` | `True` | <REPO_ROOT>/apps/chess_gui/play_mertformer_chess_web.py |
| `gui_sync_state` | `True` | `True` | canonical_fallback_ready |
| `teaching_contract_smoke` | `True` | `True` | true |
| `onefile_extension_report` | `True` | `True` | READY |
| `stockfish_anchor_optional` | `False` | `True` | Stockfish stays optional for training start and required later for anchor benchmark runs. |

## Notes

- This report covers repo-side training readiness only.
- Real training, checkpoints, and benchmark outputs remain post-run evidence.
- A missing local GUI onefile copy is acceptable when repo-canonical fallback is available.
