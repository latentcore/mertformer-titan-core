# Release Snapshot (Review-Ready) [TR]

Bu dosya, repo'nun inceleme anındaki durumunu (point-in-time) insan-okur bir formatta özetler.

## Snapshot

- Tarih (yerel): 2026-02-18
- Base Git SHA (kısa): `git rev-parse --short HEAD`
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
- Pytest: PASS (`58 passed, 3 skipped`)
- Preflight (offline): PASS
- Operator gate (safe, offline): PASS

Ek spot-check:

```bash
TITAN_OFFLINE=1 bash run.sh --test
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Önemli Dokümanlar

- Mühendislik denetimi: `reports/codex_deep_audit_EN.md`, `reports/codex_deep_audit_DE.md`, `reports/codex_deep_audit_TR.md`
- Denetim raporlarının TR eşleri: `reports/codex_deep_audit_EN_TR.md`, `reports/codex_deep_audit_DE_TR.md`
- Verified vs Target matrisi: `reports/verified_matrix.md`, `reports/verified_matrix_TR.md`
- Dış inceleme checklisti: `reports/review_checklist.md`, `reports/review_checklist_TR.md`
- Final senkron matrisi: `reports/final_sync_matrix.md`, `reports/final_sync_matrix_TR.md`
- Verimlilik yakınsama analizi: `reports/efficiency_convergence_analysis.md`, `reports/efficiency_convergence_analysis_TR.md`
- Kullanım kılavuzu: `USAGE_GUIDE.md`, `USAGE_GUIDE_TR.md`
- Demo kanıt videosu: `assets/snake_demo_proof.mp4`
- Clean-room doğrulama: `reports/cleanroom_verification_TR.md`
- Dataset uyum dokümanları: `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`, `datasets/inventory*`

## Release Artefaktları (Desktop)

- `MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`
- `MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age`
- `MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`
- SHA-256:
  - `3efc60d316d7caccdd65786c74a667cf74ef3687cd8d8be7e19e4088e6c3880e` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`)
  - `2fc620ae2a122626992e302a313111d8ab1781b1acaa7b7613e9fbf5a88183bd` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age`)
  - `9ec43d6bf41c251346e7337068f097b2d8be4481bebfe652813108c091db01a2` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)

## Bilinen Gate / Blokerler

- Dataset uyum kapısı:
  - ✅ Lisanslar doğrulandı (`datasets/LICENSES*.md` içinde `TBD` yok)
  - ✅ Snapshot registry `datasets/hashes.json` içinde kayitli (pinlenen revision + manifest fingerprint)
- Kalan iş, operasyonel ve eğitim sonrası:
  - Hedef donanımda üretim eğitimini çalıştırmak
  - Üretilen checkpoint'lerden benchmark raporlarını almak
