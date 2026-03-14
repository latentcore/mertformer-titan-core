# Değişiklik Kaydı

Bu dosya projedeki önemli değişiklikleri takip eder.


## v1.0.0-build30-v2 - 2026-03-13

### Eklenenler
- Veri pipeline'ında cross-dataset deduplication hattı eklendi.
- MoE paralel dispatch modu ve CfC fast path toggle eklendi.
- Onefile demo CLI geliştirmeleri + training log dashboard scripti.
- CfC/MoE loss parity için SOP tolerans kontrolü.

### Değişenler
- Build 30 V2 sürüm senkronu core dokümanlar ve model metadata genelinde yapıldı.
- Training token budget varsayılanı fixed-steps gating olarak ayarlandı.

### Doğrulama
- SOP tam koşu (verify_all, md_quality, linkcheck, sync_manifest) PASS.

## v0.1.0-pilot-ready - 2026-02-08

### Eklenenler
- Pilot raporu sözleşmesi: `interfaces/pilot_report_v1.schema.json`.
- SDK pilot yardımcıları ve CLI komutları:
  - `mertformer verify`
  - `mertformer pilot-report --out <json>`
- Drone sınıfı offline kanıt akışı için SITL akışı:
  - `scripts/drone_sitl_demo.py`
  - `reports/drone_sitl_demo.md`
  - `reports/drone_sitl_demo_TR.md`
  - `reports/pilots/README.md`
  - `reports/pilots/README_TR.md`
- Pilot ticari doküman seti:
  - `reports/pilot_readiness_kit.md` + `_TR`
  - `reports/pilot_offer_packages.md` + `_TR`
  - `reports/sales_funnel_90d.md` + `_TR`
- Clean-room doğrulama raporu:
  - `reports/cleanroom_verification.md`
  - `reports/cleanroom_verification_TR.md`
- Pilot kabul imza şablonu:
  - `reports/pilot_acceptance_signoff.md`
  - `reports/pilot_acceptance_signoff_TR.md`

### Değişenler
- Benchmark claim güvenlik kapısı sıkılaştırıldı: checkpoint yoksa `NOT ELIGIBLE FOR CLAIM` döner.
- README claim dili netleştirildi: ölçülmüş değer ile hedef/tahmin ayrımı yapıldı.
- Track edilen artefaktlardan mutlak Desktop path referansları kaldırıldı.
- TR dokümanlarda eşleşme ve yazım tutarlılığı iyileştirildi.

### Düzeltmeler
- SDK yükleme yolunda strict checkpoint koruması eklendi; rastgele ağırlıkla pilot koşusu engellendi.
- Docs index ve project structure blokları gerçek dosya setiyle senkronlandı.

### Doğrulama
- `python3 -m pytest -q` geçti.
- `ruff check .` geçti.
- `bash scripts/verify_all.sh` geçti.
