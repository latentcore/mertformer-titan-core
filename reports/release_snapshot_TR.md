# Release Snapshot (Review-Ready) [TR]

Bu dosya, repo'nun inceleme anındaki durumunu (point-in-time) insan-okur bir formatta özetler.

## Snapshot

- Tarih (lokal): 2026-02-06
- Base Git SHA (kısa): 63e47a2
- Baseline Python: 3.11 (bkz: `repro/python_TR.md`)
- Varsayılan mod: offline-first (`TITAN_OFFLINE=1`)

## Verified (Run)

Runbook:

```bash
bash scripts/bootstrap_venv.sh
bash scripts/verify_all.sh
```

Beklenen:
- Secret scan: PASS
- Pytest: PASS (`21 passed, 4 skipped`)
- Preflight (offline): PASS
- Operator gate (safe, offline): PASS

Ek spot-check:

```bash
TITAN_OFFLINE=1 bash run.sh --test
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Önemli Dokümanlar

- Mühendislik denetimi: `reports/codex_deep_audit_DE.md`, `reports/codex_deep_audit_TR.md`
- Verified vs Target matrisi: `reports/verified_matrix.md`, `reports/verified_matrix_TR.md`
- Dış inceleme checklisti: `reports/review_checklist.md`, `reports/review_checklist_TR.md`
- Dataset uyum dokümanları: `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`, `datasets/inventory*`

## Bilinen Gate / Blokerler

- Dataset uyum kapısı:
  - ✅ Lisanslar doğrulandı (`datasets/LICENSES*.md` içinde `TBD` yok)
  - ✅ Snapshot registry `datasets/hashes.json` içinde kayitli (pinlenen revision + manifest fingerprint)
- Kalan iş, operasyonel ve eğitim sonrası:
  - Hedef donanımda üretim eğitimini çalıştırmak
  - Üretilen checkpoint'lerden benchmark raporlarını almak
