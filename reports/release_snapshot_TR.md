# Release Snapshot (Review-Ready) [TR]

Bu dosya, repo'nun inceleme anındaki durumunu (point-in-time) insan-okur bir formatta özetler.

## Snapshot

- Tarih (yerel): 2026-02-09
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
- Pytest: PASS (`30 passed, 4 skipped`)
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

- `MertFormer_Titan_OnyxStorm_v1.0_B27_Release.zip`
- `MertFormer_Titan_OnyxStorm_v1.0_B27_Hamdi_Package_Release.zip`
- `MertFormer_Titan_OnyxStorm_v1.0_B27_Locked.secure.age`
- SHA-256:
  - `785854fafafb2120e5373e4d8bdcbac0f02d5d269a38883f0fa09255648b03b9` (`MertFormer_Titan_OnyxStorm_v1.0_B27_Release.zip`)
  - `49c95d2bebefdbcb184044acfaaaabec7800026116f3f5d1836707485b1bcc40` (`MertFormer_Titan_OnyxStorm_v1.0_B27_Hamdi_Package_Release.zip`)
  - `e260307e32b6d1f9aa940ef08a500b4d832168c861598e23022d693b97ed515b` (`MertFormer_Titan_OnyxStorm_v1.0_B27_Locked.secure.age`)

## Bilinen Gate / Blokerler

- Dataset uyum kapısı:
  - ✅ Lisanslar doğrulandı (`datasets/LICENSES*.md` içinde `TBD` yok)
  - ✅ Snapshot registry `datasets/hashes.json` içinde kayitli (pinlenen revision + manifest fingerprint)
- Kalan iş, operasyonel ve eğitim sonrası:
  - Hedef donanımda üretim eğitimini çalıştırmak
  - Üretilen checkpoint'lerden benchmark raporlarını almak
