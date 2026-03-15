# Math Fastproof Artifacts (Build30 V2)

Canonical run: `run_20260315_050133`

## Status
- FINAL_STATUS: gate_fail (accuracy gate)
- Note: No retrain performed in this closure pass per explicit request. Artifacts below preserve the last canonical run.

## Included (Repo-Tracked)
- `run_20260315_050133_summary.json`
- `run_20260315_050133_compare.json`
- `run_20260315_050133_compare.md`
- `run_20260315_050133_compare.csv`
- `run_20260315_050133_health.txt`
- `run_20260315_050133_run_log.jsonl`
- `run_20260315_050133_step_metrics.csv`
- `run_20260315_050133_artifact_index.json`
- `zip_manifest.json`

## External Artifacts (Git-ignored; hash recorded)
- Evidence zip:
  - Path: `/Users/mertyunlu/Downloads/content/mertformer_outputs/runs/run_20260315_050133/run_20260315_050133_evidence.zip`
  - SHA256: `26c7cdea097c88e6b1ce5a4400c16e9f417887d1a06fcc257111db80c5254be7`
- Final checkpoint:
  - Path: `/Users/mertyunlu/Downloads/content/mertformer_outputs/runs/run_20260315_050133/run_20260315_050133_model_final.pt`
  - SHA256: `e39819c37b78900118667172bc94669d5c6548d46d29b9cae177cf457860d797`

## Reproduce
```
MERTFORMER_SUPPRESS_TORCH_FX_WARNINGS=1 \
python3 scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py
```

If a new run is required, update this folder with the new `run_id` and hashes.
