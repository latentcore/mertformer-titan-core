# Text Understanding PoC Artifactları (Build30 V2)

Kanonik koşu: `run_20260315_180151`

## Durum
- FINAL_STATUS: gate_pass
- Yöntem: sentetik TR uzun metin üzerinde kural tabanlı çıkarım

## Dahil (Repo-Tracked)
- `run_20260315_180151_summary.json`
- `run_20260315_180151_compare.json`
- `run_20260315_180151_compare.md`
- `run_20260315_180151_compare.csv`
- `run_20260315_180151_health.txt`
- `run_20260315_180151_run_log.jsonl`
- `run_20260315_180151_artifact_index.json`

## Harici Artifactlar (Git-ignored; hash kayıtlı)
- Evidence zip:
  - Yol: `<HOME>/Downloads/content/mertformer_outputs/runs/run_20260315_180151/run_20260315_180151_evidence.zip`
  - SHA256: `922c1ef0bc1bb9f020726066997c0404acba383e3be79fbaff0717f6c882bd17`
- Sentetik dataset JSONL:
  - `run_20260315_180151_train.jsonl`
  - `run_20260315_180151_val.jsonl`
  - `run_20260315_180151_test.jsonl`
  - `run_20260315_180151_unseen.jsonl`

## Yeniden Üretim
```
python3 scripts/kaggle_onefile_demo_build30_text_understanding.py
```
