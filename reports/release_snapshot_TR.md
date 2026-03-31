# Release Snapshot (Review-Ready) [TR]

Bu dosya, repo'nun inceleme anındaki durumunu (point-in-time) insan-okur bir formatta özetler.

## Snapshot

- Tarih (yerel): 2026-03-04
- Base Git SHA (kısa): `6f41827`
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
- Pytest: PASS (`122 passed, 3 skipped`)
- Preflight (offline): PASS
- Operator gate (safe, offline): PASS

Ek spot-check:

```bash
TITAN_OFFLINE=1 bash run.sh --test
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Önemli Dokümanlar

- Mühendislik denetimi: `reports/codex_deep_audit_EN.md`, `reports/codex_deep_audit_DE.md`, `reports/codex_deep_audit_TR.md`
- TR denetim dosyaları yönlendirme (pointer) dosyalarıdır: `reports/codex_deep_audit_EN_TR.md`, `reports/codex_deep_audit_DE_TR.md` (kanonik TR içerik: `reports/codex_deep_audit_TR.md`)
- Verified vs Target matrisi: `reports/verified_matrix.md`, `reports/verified_matrix_TR.md`
- Dış inceleme checklisti: `reports/review_checklist.md`, `reports/review_checklist_TR.md`
- Final senkron matrisi: `reports/final_sync_matrix.md`, `reports/final_sync_matrix_TR.md`
- Verimlilik yakınsama analizi: `reports/efficiency_convergence_analysis.md`, `reports/efficiency_convergence_analysis_TR.md`
- Kullanım kılavuzu: `USAGE_GUIDE.md`, `USAGE_GUIDE_TR.md`
- Demo kanıt videosu: `assets/snake_demo_proof.mp4`
- Clean-room doğrulama: `reports/cleanroom_verification_TR.md`
- Dataset uyum dokümanları: `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`, `datasets/inventory*`

## Release Artefaktları (Desktop)

- `MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip`
- `MertFormer_Titan_OnyxStorm_v2.0_B30_Locked.secure.age`
- Locked artefakt durumu: `skipped (expected: AGE_RECIPIENT_FILE missing)`
- SHA-256:
  - `d62a7a537014cdc8a06d457ea4d2bd73dba6fd51c6eb8f41665991c40b096f6d` (`MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip`)
  - `SKIPPED` (`MertFormer_Titan_OnyxStorm_v2.0_B30_Locked.secure.age`)

## Bilinen Gate / Blokerler

- Dataset uyum kapısı:
  - 🟡 Çekirdek eğitim datasetleri için lisanslar doğrulandı; opsiyonel/demo girdiler `datasets/LICENSES*.md` içinde `TBD` olarak kalır (etkinleştirilmeden önce doğrulanmalı)
  - ✅ Snapshot registry `datasets/hashes.json` içinde kayıtlı (pinlenen revision + manifest fingerprint)
- Kalan iş, operasyonel ve eğitim sonrası:
  - Hedef donanımda üretim eğitimini çalıştırmak
  - Üretilen checkpoint'lerden benchmark raporlarını almak
