# Benchmark Çıktıları

Bu klasör benchmark üretim çıktılarıni ve ozetlerini tutar. Aksi belirtilmedikce
buradaki dosyalar **eğitim öncesi** kabul edilir ve tam eğitim sonrası yeniden üretilmelidir.

## Dosyalar
- `humaneval_outputs.jsonl` ve `mbpp_outputs.jsonl`
  Üreten: `python scripts/benchmarks_internal.py --run`
- `gsm8k_outputs.jsonl` ve `gsm8k_summary.json`
  Üreten: `python eval/gsm8k.py --run` (özet için `--score-only`)
- `smoke_train_metrics.json`
  Üreten: `python scripts/smoke_train_benchmark.py` (dokümantasyon için smoke; sentetik token)
- `summary.json`
  Üreten: `python eval/report_builder.py`
- `math_fastproof/`
  Canonical math-fastproof artifactları (README’ye bakın).
- `text_understanding/`
  Text-understanding PoC artifactları (README’ye bakın).

## Notlar
- Hızlı smoke kontrol için `--samples` kullanın.
- Rastgele ağırlıkları önlemek için `--ckpt` ile gerçek checkpoint verin.
