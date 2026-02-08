# Benchmark Çıktıları

Bu klasor benchmark üretim çıktılarıni ve ozetlerini tutar. Aksi belirtilmedikce
buradaki dosyalar **eğitim öncesi** kabul edilir ve tam eğitim sonrasi yeniden uretilmelidir.

## Dosyalar
- `humaneval_outputs.jsonl` ve `mbpp_outputs.jsonl`
  Ureten: `python scripts/benchmarks_internal.py --run`
- `gsm8k_outputs.jsonl` ve `gsm8k_summary.json`
  Ureten: `python eval/gsm8k.py --run` (ozet için `--score-only`)
- `smoke_train_metrics.json`
  Ureten: `python scripts/smoke_train_benchmark.py` (dokumantasyon için smoke; sentetik token)
- `summary.json`
  Ureten: `python eval/report_builder.py`

## Notlar
- Hizli smoke kontrol için `--samples` kullanin.
- Rastgele agirliklari onlemek için `--ckpt` ile gerçek checkpoint verin.
