# Final Senkron Matris (Build 30)

Bu dosya, doküman eşleşmesi, sürüm tutarlılığı ve doğrulama koşularının son durum kaydıdır.

## 1) EN/TR Markdown Eşleşmesi

Kural: uygun olan her tracked `X.md` dosyasının `X_TR.md` karşılığı olmalı.

| Kontrol | Sonuç |
| --- | --- |
| Eksik eşleşme | 0 |
| Yetim `_TR.md` dosyası | 1 (`reports/codex_deep_audit_TR.md`, bilinçli ana TR denetim dosyası) |
| Derin denetim pointer politikası | EN_TR + DE_TR dosyaları `reports/codex_deep_audit_TR.md` için pointer dosyalaridir |
| Bu final geçişinde eklenenler | `reports/pilot_readiness_kit_TR.md`, `reports/pilot_offer_packages_TR.md`, `reports/sales_funnel_90d_TR.md`, `reports/drone_sitl_demo_TR.md`, `reports/cleanroom_verification_TR.md`, `reports/go_status_matrix_TR.md` |

## 2) Build Etiketi Tutarlılığı

| Kapsam | Sonuç |
| --- | --- |
| Kullanıcıya görünen sürüm referansları | Build 30 uyumlu |
| Aktif kullanıcı scriptlerinde legacy referanslar | normalize edildi |
| Soy/köken (lineage) notları | sadece açıkça roadmap/tarihçe olarak bırakıldı |

## 3) Doğrulama Koşuları (Release Candidate)

| Komut | Durum |
| --- | --- |
| `.titan-venv/bin/python -m pytest -q` | PASS (`158 passed, 4 skipped`) |
| `bash scripts/verify_all.sh` | PASS |
| `TITAN_OFFLINE=1 bash run.sh --test` | PASS |
| `.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | PASS |

## 4) Demo Kanıtı

| Artefakt | Durum |
| --- | --- |
| `assets/snake_demo_proof.mp4` | üretildi |
| `snake_demo.py` auto-restart + telemetry | doğrulandı |
| README linkleri | güncellendi |

## 5) Paketleme Kapıları

| Kapı | Durum |
| --- | --- |
| Temiz zip içinde venv/cache/log/.env yok | PASS |
| `.age` paket kapısı | PASS veya SKIPPED (`AGE_RECIPIENT_FILE` yoksa beklenen) |
| 2 paket için SHA-256 özeti üretildi | PASS |

## 6) Release Kimlikleri

- Final git SHA (main): `git rev-parse --short HEAD`
- Release zip: `MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip`
- Kilitli güvenli paket: `MertFormer_Titan_OnyxStorm_v2.0_B30_Locked.secure.age`
- SHA-256 kayıtları: `reports/release_snapshot_TR.md` (Release Artefaktları bölümü).

## 7) Taşınabilir Eğitim Hazırlığı (Build30)

| Kontrol | Sonuç |
| --- | --- |
| `run.sh --train-ready` strict modu | Eklendi (`strict_online_training_readiness`) |
| `run.sh` profil sözleşmesi | Eklendi (`TITAN_PROFILE=stable|max_arch`) |
| Teacher hard-fail politikası | Aktif (`require_gated_teacher=true`) |
| Müfredat tek kaynak oranları | Aktif (`config.curriculum_stage_ratios`) |
| Golden assertion skorlayıcı | Eklendi (`scripts/golden_score.py`) |
| Readiness manifest/rapor çıktıları | Eklendi (`reports/training_readiness_manifest.json`, `logs/preflight/train_ready_status.json`) |
