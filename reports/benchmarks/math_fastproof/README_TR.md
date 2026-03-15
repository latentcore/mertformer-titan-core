# Math Fastproof Artifactları (Build30 V2)

Kanonik koşu: `run_20260315_050133`

## Durum
- FINAL_STATUS: gate_fail (accuracy gate)
- Not: Bu kapanış geçişinde yeniden eğitim yapılmadı (talep üzerine). Aşağıdaki artifactlar son kanonik koşuyu temsil eder.

## Dahil (Repo-Tracked)
- `run_20260315_050133_summary.json`
- `run_20260315_050133_compare.json`
- `run_20260315_050133_compare.md`
- `run_20260315_050133_compare.csv`
- `run_20260315_050133_health.txt`
- `run_20260315_050133_run_log.jsonl`
- `run_20260315_050133_step_metrics.csv`
- `run_20260315_050133_artifact_index.json`
- `zip_manifest.json`

## Harici Artifactlar (Git-ignored; hash kayıtlı)
- Evidence zip:
  - Yol: `/Users/mertyunlu/Downloads/content/mertformer_outputs/runs/run_20260315_050133/run_20260315_050133_evidence.zip`
  - SHA256: `26c7cdea097c88e6b1ce5a4400c16e9f417887d1a06fcc257111db80c5254be7`
- Final checkpoint:
  - Yol: `/Users/mertyunlu/Downloads/content/mertformer_outputs/runs/run_20260315_050133/run_20260315_050133_model_final.pt`
  - SHA256: `e39819c37b78900118667172bc94669d5c6548d46d29b9cae177cf457860d797`

## Yeniden Üretim
```
MERTFORMER_SUPPRESS_TORCH_FX_WARNINGS=1 \
python3 scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py
```

Yeni koşu gerekiyorsa bu klasörü yeni `run_id` ve hash’lerle güncelleyin.
