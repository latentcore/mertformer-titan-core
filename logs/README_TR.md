# Loglar (Genel Bakış)

Bu dizin, **üretilen artefaktları** (run logları, preflight çıktıları, operator-gate kanıtları) içerir.

Politika:
- `logs/` varsayılan olarak **gitignore** edilir (artefaktlar commit'lenmez).
- `logs/` altında track edilen dosyalar: `logs/README.md` ve bu dosya (`logs/README_TR.md`).

## Sık Görülen Artefaktlar (Untracked)
- `logs/preflight/titan_preflight.log` (preflight teşhis çıktısı)
- `logs/operator_mode/*.jsonl` + `*.manifest.json` (operator gate kanıtı)
- `logs/run_*.jsonl` ve `logs/run_*.csv` (training loop metrikleri)
- `logs/production_run.log` (büyük olabilir)
- `logs/ALL_LOGS.jsonl` (birleşik logbook)

## Birleşik Logbook
- Build/append: `.titan-venv/bin/python scripts/logbook_build.py --append`
- Baştan rebuild: `.titan-venv/bin/python scripts/logbook_build.py --rebuild`
- Logbook **append-only**'dir ve her içe aktarılan satır için kaynak metadatası taşır.

## Notlar
- Loglar **sanitize** edilir (preflight çıktısında token'lar maskelenir).
- Yeni bir koşu için artefaktları `run.sh --test` veya `run.sh` ile yeniden üret.

