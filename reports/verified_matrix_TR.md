# Verified vs Target Matrisi (Mühendislik Gerçeği)

> **Dış inceleme notu:** Compute sponsorship değerlendiriyorsanız önce
> `private/commercial/outreach_compute_sponsorship_messages.md` okunmalıdır. Bu matris,
> doğrulanmış yüzeyleri hedeflerden ayırır ve claim-boundary referansı olarak
> kullanılmalıdır.

Bu doküman şu ayrımı katı şekilde yapar:
- **Verified (Run)**: komut çalıştırılarak yerelde veya CI'da doğrulandı
- **Verified (Code)**: kodda görülerek doğrulandı (ama çalıştırılmadı)
- **Target / Claim**: hedef/iddia; doğru olması için eğitim/benchmark gerekir

Legend:
- ✅ Verified (Run)
- 🔎 Verified (Code)
- 🎯 Target / Claim (henüz doğrulanmadı)
- ⏭️ SKIP (bu ortam için geçerli değil)

## Doğrulama Baseline

- Baseline Python: **3.11** (bkz: `repro/python_TR.md`)
- Varsayılan çalışma şekli: **offline-first** (`TITAN_OFFLINE=1`)
- Tek komut doğrulama: `bash scripts/verify_all.sh`
- Son yerel doğrulama (örnek): 2026-02-06 (macOS, Python 3.11.14)

## Matris

| Yetenek | Durum | Kanıt |
| --- | --- | --- |
| Track'li dosyalarda secret scan gate | ✅ Verified (Run) | `python scripts/secret_scan.py` |
| Unit/integration testler | ✅ Verified (Run) | `python -m pytest -q` |
| Preflight (offline-safe) | ✅ Verified (Run) | `TITAN_OFFLINE=1 python scripts/titan_preflight.py` |
| Operator mode gate (safe, offline) | ✅ Verified (Run) | `python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` |
| `run.sh --test` offline-first (external login/download yok) | ✅ Verified (Run) | `TITAN_OFFLINE=1 bash run.sh --test` |
| Koddan dataset ID envanteri | ✅ Verified (Run) | `python scripts/extract_dataset_refs.py` → `datasets/inventory*` |
| Dataset kaynakları dokümante | ✅ Verified (Code) | `datasets/SOURCES*.md` |
| Dataset lisans checklisti | 🔎 Verified (Code) | `datasets/LICENSES*.md` (çekirdek eğitim datasetleri doğrulandı; opsiyonel/demo girdiler etkinleştirilmeden önce `TBD` kalabilir) |
| Dataset snapshot hash kayıtları | ✅ Verified (Run) | `python scripts/record_dataset_hashes.py` → `datasets/hashes.json` |
| Training “tiny smoke” (CPU/MPS) | ✅ Verified (Run) | `python scripts/train_smoke.py --cleanup` |
| Tam eğitim koşusu (uçtan uca) | 🎯 Target / Claim | Eğitim donanımı + gerçek dataset snapshotları gerektirir |
| Benchmarklar (HumanEval/MBPP) | 🔎 Verified (Code) | `scripts/benchmarks_internal.py` (checkpoint yoksa SKIP) |
| ONNX export doğruluğu | ✅ Verified (Run) | `pytest` içinde `scripts/test_onnx_export.py::test_export` çalışır |
| CI pipeline | 🔎 Verified (Code) | `.github/workflows/ci.yml` |
| Demo (pygame autoplayer) | 🔎 Verified (Code) | `snake_demo.py` ( `pip install -e '.[demo]'` veya `bootstrap_venv.sh --demo` ) |

## Notlar / Blokerler (Tarafsız)

- Dataset uyum kapısı (lisans + snapshot registry) şu an pinlenen HF revision'ları için sağlanmıştır (bkz: `datasets/hashes.json`).
- Performans rakamları, tam eğitim + benchmark raporu olmadan **hedef** olarak kalır.
