# Değişiklik Kaydı

Bu dosya projedeki önemli değişiklikleri takip eder.


## Unreleased - 2026-06-17

### Eklenenler
- WHITE_PAPER_LIQUIDROUTER (EN+TR): arXiv gönderim bölümü (Section 8) + önerilen başlık — 45K sonrasına kapılı.
- `TITAN_DISTILL_ALPHA` env knob'u: öğretmensiz 45K-öncesi smoke (0 olduğunda 70B teacher hiç indirilmez).
- README_SUMMARY (EN+TR): "Architecture at a glance" bileşen tablosu (BitNet / GQA / seyrek MoE / Liquid-CfC).
- Kök kanonik scaffold (STATUS / TRUTH_MATRIX / BACKLOG / GOVERNANCE / REPRODUCE, EN+TR) — tek reviewer giriş noktası.

### Değişenler
- Attention sınıfı MLA → GQA yeniden adlandırıldı (her zaman grouped-query; `layers/mla.py` dosya adı manifest/SHA kararlılığı için korundu).
- 45K-öncesi operasyonel sertleştirme: atomik checkpoint (`os.replace`), forward-içi MoE collapse-flag `all_reduce` kaldırıldı, telemetri buffer'ları `persistent=False`, kalıcı grad-clip ratchet kaldırıldı (artık transient), telemetri throttle.
- Liquid ablasyonu 12-seed verdict'e kanonikleştirildi (OFF %96.32 / ON %94.69, Δ−1.63 pp, p=0.305, inconclusive — ölçülen fayda yok, ~%30 yavaş); tek-seed +0.50 pilotu tüm ablasyon yüzeylerinde ezildi.
- TECHNICAL_REPORT (EN+TR) klinik rewrite; §3.1 "12x" → Target/estimate; §7 SHA256 → "designed"; MoE expert intermediate 8192'ye düzeltildi.
- README 178KB → ~4KB kesildi (tam snapshot arşivlendi); tek-persona teknik/kanıt yüzeyi; ticari/GTM materyali `private/` altına taşındı.
- Lisans yüzeyi README / README_TR genelinde Proprietary & Confidential'e çözüldü (LICENSE ile uyumlu).

### Düzeltilenler
- `pyproject.toml`: eksik `mertformer_sdk.kernels.cpp` paketi eklendi.
- `.pre-commit-config.yaml`: ruff v0.15.5'e pinlendi (constraints.txt ile uyumlu).
- `Dockerfile`: yeniden üretilebilir build için artık `constraints.txt` uygular.
- `registry/mertformer_v0.1.json`: sürüm Build 30 V2'ye senklendi (v27.0 idi).
- `scripts/secret_scan.py` + `policy/allow_deny_policy.yaml`: GitHub token pattern'leri gho_/ghu_/ghs_/ghr_ ve fine-grained PAT'e genişletildi.

### Doğrulama
- `370 passed, 4 skipped` (offline-first pytest); readiness `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. Trained/benchmark iddiası yok — kalan tek boşluk gerçek 45K GPU koşusu.

## Unreleased - 2026-05-24

### Eklenenler
- FFN, MoE BitSwiGLU ve MLA K/V eğitim yolları için opsiyonel, varsayılan kapalı packed projection kontrolleri.
- Batch size, log aralığı, validation aralığı, checkpoint aralığı ve DataLoader transfer davranışı için environment üzerinden değiştirilebilir eğitim kontrolleri.
- `repro/accelerate_8xgpu.yaml` altında opsiyonel 8 GPU Accelerate profili.
- Packed projection ve Liquid eğitim implementasyonu varyantları için equivalence test kapsamı.

### Değişenler
- README, kullanım kılavuzu, eğitim planı, feature-flag governance, script kataloğu ve doğrulama matrisi; opsiyonel hız kontrol yüzeyini açık iddia sınırlarıyla dokümante edecek şekilde güncellendi.
- Dokümantasyon, `repro/` klasörünün yeniden üretilebilirlik/koşu config'leri için; `configs/` klasörünün ise stabil isimlendirilmiş config sözleşmeleri için kullanıldığını netleştirdi.

### Doğrulama
- Opsiyonel hız flag'leri varsayılan kapalı kalır ve herhangi bir hız iddiasından önce equivalence testleri ile hedef makine logları gerektirir.

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

## Pass 7 (2026-06-13) — Mac'te yapılabilir backlog sıfırlandı + $0 Kaggle pilotu
- `scripts/run_liquid_ablation.py` + `docs/KAGGLE_PILOT.md` eklendi: ücretsiz LiquidRouter ON-vs-OFF
  ablasyon pilotu (~80–100M, saf CE, teacher yok) — GPU'ya kapılı işi açan tek domino.
- Eğitim sırasında LatentODE per-batch reset (batch'ler arası state sızıntısı yok); MoE collapse flag DDP
  all-reduce (guarded, DDP dışında no-op); liquid-impl benchmark scripti; coverage config'i.
- Dokümanlar: ARCHITECTURE.md Projections + stage-3 notu; CPU quickstart. Backlog dispozisyonları DECISIONS.md'de.
- İnvaryantlar korundu: parametre sayısı kilitli; pytest yeşil; ruff + scoped mypy + verify_all yeşil.
