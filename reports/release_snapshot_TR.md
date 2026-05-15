# Release Snapshot (Review-Ready) [TR]

Bu dosya, repo'nun inceleme anındaki durumunu (point-in-time) insan-okur bir formatta özetler.

## Snapshot

- Snapshot tazeliği: en güncel closure koşu aralığı için `reports/one_command_full_sop_summary.md` dosyasına bakın.
- Güncel Git SHA (lokal): bu çalışma ağacında `git rev-parse --short HEAD` çalıştırın.
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
- Pytest: PASS (`250 passed, 4 skipped`)
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
- Repo-side closure scorecard: `reports/repo_closure_scorecard.md`, `reports/repo_closure_scorecard.json`
- Closure freeze ve known limits: `reports/final_master_plan_freeze.md`, `reports/known_limits_v1.md`
- Bakım, kalite ve doğrulama kontratı: `reports/support_maintenance_policy.md`, `reports/quality_gate_matrix.md`, `reports/test_verification_matrix.md`
- ADR zinciri: `reports/adr_index.md`, `adr/ADR-0001-source-of-truth-and-claim-boundary.md`
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
  - `f1076a8513297c397fe7c567401f5b8c7d623bf3607271f5e029e37ee50e301f` (`MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip`)
  - `SKIPPED` (`MertFormer_Titan_OnyxStorm_v2.0_B30_Locked.secure.age`)

## Bilinen Gate / Blokerler

- Dataset uyum kapısı:
  - 🟡 Çekirdek eğitim datasetleri için lisanslar doğrulandı; opsiyonel/demo girdiler `datasets/LICENSES*.md` içinde `TBD` olarak kalır (etkinleştirilmeden önce doğrulanmalı)
  - ✅ Snapshot registry `datasets/hashes.json` içinde kayıtlı (pinlenen revision + manifest fingerprint)
- Kalan iş, operasyonel ve eğitim sonrası:
  - Hedef donanımda üretim eğitimini çalıştırmak
  - Üretilen checkpoint'lerden benchmark raporlarını almak
