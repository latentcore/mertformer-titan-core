# Benchmark Ciktilari

Bu klasor benchmark uretim ciktilarini ve ozetlerini tutar. Aksi belirtilmedikce
buradaki dosyalar **egitim oncesi** kabul edilir ve tam egitim sonrasi yeniden uretilmelidir.

## Dosyalar
- `humaneval_outputs.jsonl` ve `mbpp_outputs.jsonl`
  Ureten: `python scripts/benchmarks_internal.py --run`
- `gsm8k_outputs.jsonl` ve `gsm8k_summary.json`
  Ureten: `python eval/gsm8k.py --run` (ozet icin `--score-only`)
- `smoke_train_metrics.json`
  Ureten: `python scripts/smoke_train_benchmark.py` (dokumantasyon icin smoke; sentetik token)
- `summary.json`
  Ureten: `python eval/report_builder.py`

## Notlar
- Hizli smoke kontrol icin `--samples` kullanin.
- Rastgele agirliklari onlemek icin `--ckpt` ile gercek checkpoint verin.
