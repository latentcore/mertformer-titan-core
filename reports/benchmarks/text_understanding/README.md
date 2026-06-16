# Text Understanding PoC Artifacts (Build30 V2)

Canonical run: `run_20260315_180151`

## Status
- FINAL_STATUS: gate_pass
- Method: rule-based extraction over synthetic Turkish long-text records

## Included (Repo-Tracked)
- `run_20260315_180151_summary.json`
- `run_20260315_180151_compare.json`
- `run_20260315_180151_compare.md`
- `run_20260315_180151_compare.csv`
- `run_20260315_180151_health.txt`
- `run_20260315_180151_run_log.jsonl`
- `run_20260315_180151_artifact_index.json`

## External Artifacts (Git-ignored; hash recorded)
- Evidence zip:
  - Path: `<HOME>/Downloads/content/mertformer_outputs/runs/run_20260315_180151/run_20260315_180151_evidence.zip`
  - SHA256: `922c1ef0bc1bb9f020726066997c0404acba383e3be79fbaff0717f6c882bd17`
- Synthetic dataset JSONL files:
  - `run_20260315_180151_train.jsonl`
  - `run_20260315_180151_val.jsonl`
  - `run_20260315_180151_test.jsonl`
  - `run_20260315_180151_unseen.jsonl`

## Reproduce
```
python3 scripts/kaggle_onefile_demo_build30_text_understanding.py
```
