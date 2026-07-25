# GO Durum Matrisi (Pilot vs Ürün Claim)

## Kapsam
Bu matris, repoda tamamen biten işleri ve dış/operasyonel olarak kalan işleri ayırır.

## A) Pilot Teslim Hazırlığı (A1-A20)

| ID | Durum | Kanıt / Gerekçe |
| --- | --- | --- |
| A1 | ✅ | Repo temiz ve senkron (`git status --short` hem Desktop hem canonical depo kökü temiz). |
| A2 | ✅ | `strict_checkpoint=True` varsayılanı: `mertformer_sdk/api.py`. |
| A3 | ✅ | `mertformer verify` komutu: `mertformer_sdk/cli.py`. |
| A4 | ✅ | `mertformer pilot-report --out ...` komutu: `mertformer_sdk/cli.py`. |
| A5 | ✅ | Pilot şeması mevcut: `interfaces/pilot_report_v1.schema.json`. |
| A6 | ✅ | Claim kapısı var: `scripts/benchmarks_internal.py` içinde `NOT ELIGIBLE FOR CLAIM`. |
| A7 | ✅ | Desktop absolute-path temiz (takipli dosyalarda hardcoded Desktop yolu yok). |
| A8 | ✅ | Claim policy + pilot checklist net: `README.md`. |
| A9 | ✅ | README config örneği kodla uyumlu (`use_torch_compile = False`): `config/config.py`. |
| A10 | ✅ | Pilot doküman seti hazır: `reports/pilot_readiness_kit.md`, `reports/pilot_offer_packages.md`, `reports/sales_funnel_90d.md`. |
| A11 | ✅ | Test kapısı geçti: `578 passed, 5 skipped`. |
| A12 | ✅ | Lint kapısı geçti: `ruff check` yeşil. |
| A13 | ✅ | Full gate geçti: `bash scripts/verify_all.sh` -> `[verify] OK`. |
| A14 | ✅ | SDK EN/TR senkron: `SDK_GUIDE.md`, `SDK_GUIDE_TR.md`. |
| A15 | ✅ | Release commitleri `main` üstünde push edildi (bkz: `git log --oneline -n 1`). |
| A16 | ✅ | Tag/release işareti push edildi (`v0.1.3-review-fix`) + changelog dosyaları mevcut. |
| A17 | ✅ | Clean-room doğrulaması tamam: `reports/cleanroom_verification.md`. |
| A18 | ✅ | Pilot teslim klasör standardı mevcut: `reports/pilots/`. |
| A19 | ❌ | Dahili cleanroom hukuki kaydı mevcut (`reports/legal_cleanroom_signoff_internal.md`), ancak dış hukuk imzası hala beklemede. |
| A20 | ❌ | 2 ücretli pilot veya 2 imzalı LOI gerekir; repo dışı ticari icradır. |

## B) Ürün/Benchmark Claim Hazırlığı (B1-B10)

| ID | Durum | Kanıt / Gerekçe |
| --- | --- | --- |
| B1 | ❌ | Gerçek stage snapshot + final pinli hash üretim veri akışından gelmelidir. |
| B2 | ❌ | Hedef donanımda tam pretrain/finetune koşusu gerekir. |
| B3 | ❌ | Eğitilmiş üretim checkpoint artefaktı gerekir. |
| B4 | ❌ | Benchmark çıktıları eğitilmiş checkpoint üzerinden üretilmelidir. |
| B5 | ❌ | Gerçek cihazda latency/power ölçümü gerekir. |
| B6 | ✅ | Claim dili dokümanda ayrıştırıldı (ölçülmüş vs hedef/tahmin). |
| B7 | ❌ | Üçüncü taraf tekrar doğrulama dış doğrulayıcı ekip gerektirir. |
| B8 | ❌ | Dahili teacher/output değerlendirmesi mevcut (`reports/teacher_output_license_assessment.md`), ancak nihai ticari hukuk onayı dış bağımlılık olarak beklemede. |
| B9 | ❌ | Güvenlik/pentest raporu için harici güvenlik test kapsamı gerekir. |
| B10 | ❌ | SLA/incident/rollback sözleşme paketi iş/hukuk icrası gerektirir. |

## C) Pilot Teslim Paketi (Müşteri Başına)

| Kalem | Durum | Kanıt |
| --- | --- | --- |
| verify_all özet/log | ✅ | `scripts/verify_all.sh` + pilot report payload |
| operator gate JSON özeti | ✅ | `scripts/operator_mode_gate.py` çıktısı |
| pilot_report_v1 JSON | ✅ | `interfaces/pilot_report_v1.schema.json` |
| offline tekrar çalıştırma adımları | ✅ | `USAGE_GUIDE.md` |
| risk/limit notu | ✅ | README + benchmark eligibility kapısı |
| kabul imza sayfası | ✅ | `reports/pilot_acceptance_signoff_TR.md` |

## Güncel Kapı Kararı
- **Pilot teknik hazırlık:** GO
- **Ticari pilot kapanış hazırlığı:** A19 + A20 beklemede
- **Ürün/benchmark claim hazırlığı:** B1-B5, B7-B10 beklemede
