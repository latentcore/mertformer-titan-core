## MertFormer Titan (Build 30 V2)

Kontrollü yerel dağıtım için offline-first, denetlenebilir ve görev odaklı yapay zeka altyapısı.
Mevcut olgunluk: **pilota hazır eğitim öncesi baseline** (eğitim/benchmark iddiaları beklemede).

**Build 30 V2 notu:** V2 refactor turu; dedup pipeline, MoE paralel dispatch yolu, CfC fast path ve daha sıkı eğitim kapıları ekler. İddialar hâlâ eğitim öncesidir.

### Okuyucu Hızlı Linkleri
- Türkçe özet: [README_SUMMARY_TR.md](README_SUMMARY_TR.md)
- English summary: [README_SUMMARY.md](README_SUMMARY.md)
- English full doc: [README.md](README.md)

### İddia Sınırı
- Bu depo, üretim kalite iddiası için **pre-training / doğrulanmamış** durumdadır.
- Release kapanış çıktıları model kalitesini değil, süreç bütünlüğünü kanıtlar.

## Tek Seferde Release Kapanış

Bu depo, mühendislik hardening ve release kanıt zinciri için karar tamamlı tek-geçiş kapanış akışı içerir.

### Canonical Kök
- depo kökü (mevcut çalışma ağacı)

### Tek Giriş Noktası
- `bash scripts/final_one_shot.sh`

### Çıktılar
- `reports/start_gate_report.json`
- `reports/release_manifest.json`
- `reports/project_structure_sync_report.json`
- `reports/hardening_bundle_summary.json`
- `artifacts/mertformer_release.zip`
- `artifacts/mertformer_release.zip.sha256`


![MertFormer Titan Header](assets/header.png)

<div align="center">
  <a href="README.md">🇬🇧 English</a> | <a href="README_TR.md">🇹🇷 Türkçe</a>
</div>

---

## Yasal Güvenlik Politikası (Build 30 V2)

Bu repo yalnızca yasal, denetlenebilir ve insan onaylı kullanım için tasarlanmıştır.

- Operasyonel kararlarda human-in-the-loop zorunludur.
- Orchestrator/runtime tarafında audit izi ve policy sınırları zorunludur.
- İzinsiz gözetim, gizli takip ve onaysız müdahale kapsam dışıdır.
- Pilot iddiaları öncesi güvenlik ve governance kapıları geçilmelidir.

## Closure 57 Raporu

Build 30 V2 ile birlikte makine tarafından doğrulanabilen kapanış kapısı vardır:

```bash
python3 scripts/check_57_matrix.py
mertformer 57-report --out reports/closure_57_matrix.json
```

Çıktılar:
- `reports/closure_57_matrix.json`
- `reports/closure_57_matrix.md`
- `reports/closure_57_matrix_TR.md`
- Şeffaflık notu: Closure-57 süreç düzeyinde yeşildir (`57/57`) ve mevcut raporda `out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 54, 55, 56, 57]` listelenir.
<br />

```
 __  __          _   ______
 |  \/  |        | | |  ____|
 | \  / | ___ _ _| |_| |__ ___  _ __ _ __ ___   ___ _ __
 | |\/| |/ _ \ '__| __|  __/ _ \| '__| '_ ` _ \ / _ \ '__|
 | |  | |  __/ |  | |_| | | (_) | |  | | | | | |  __/ |
 |_|  |_|\___|_|   \__|_|  \___/|_|  |_| |_| |_|\___|_|
      _______ _ _
     |__   __(_) |
        | |   _| |_ __ _ _ __
        | |  | | __/ _` | '_ \
        | |  | | || (_| | | | |
        |_|  |_|\__\__,_|_| |_|

   M  O  B  I  L  E     F  I  R  S  T     E  D  G  E     A  I
```

# 🦅 MertFormer Titan: Otonom Sürü Mimarisi
> **Hedef: Mobil maliyetle, sınır-üstü kodlama yeteneği (eğitim/benchmark sonrası).**
> **Geliştirme Aşaması:** Pilota hazır eğitim öncesi baseline (`Build 30 V2`, eğitim/benchmark iddiaları beklemede).

## 🇹🇷 Sivil/Komutan Özeti (Teknik Olmayan Okuyucu İçin)
> **Kaynak kod okumayan karar vericiler için**

**Bu proje nedir?**
MertFormer, sürekli bulut bağımlılığı olmadan, kontrollü yerel donanımda çalışmak üzere tasarlanmış offline-first bir yapay zeka altyapısıdır.

**Neden stratejiktir?**
1. **Veri Kontrolü:** Ana tasarım hedefi yerel/offline çalışmadır; dış veri maruziyetini azaltır.
2. **Operasyonel Süreklilik:** Mimari, kısıtlı bağlantı koşullarında çalışmayı sürdürecek şekilde kurgulanmıştır.
3. **Dil/Alan Uyumu:** Türkçe odaklı dokümantasyon ve iş akışı uyumu çekirdek gereksinim olarak ele alınır.

**Kısaca:** Bu sistem, genel amaçlı internet sohbet botu değil; görev odaklı ve disiplinli bir yapay zeka altyapısı olarak konumlandırılmıştır.

### ✅ Doğrulama Kanıtı (Son Yerel Koşu)
| Kapı | Sonuç |
| :--- | :--- |
| `python3 -m pytest -q` | `111 passed, 3 skipped` |
| `.titan-venv/bin/python -m ruff check .` | `All checks passed` |
| `bash scripts/verify_all.sh` | `[verify] OK` |

## 🚀 Eğitim Hazırlık Durumu (Operasyonel)
**Durum:** `EĞİTİM PIPELINE'I BAŞLATMAYA HAZIR (KAPILI/GATED)`

**Öne çıkan özellik:** Build30 V2 ile `run.sh --train-ready` strict kapısı eklendi; taşınabilir multi-GPU handoff için makine-okur reason-code üretir.

Bu depo artık sadece fikir/prototip seviyesinde değildir. Çekirdek doğrulama kapıları yeşildir ve veri/donanım önkoşulları tamamlandığında eğitim akışı doğrudan başlatılabilir.

### Kanıt Özeti
1. **Çekirdek kalite kapıları geçti**
   - `pytest` geçti (`111 passed, 3 skipped`)
   - `ruff check` geçti (`All checks passed`)
   - `verify_all.sh` geçti (`[verify] OK`)
2. **Mimari ve güvenlik kontrolleri geçti**
   - Offline preflight tüm adımlarda yeşil tamamlandı.
   - Operator gate geçti (overfit, failure-budget, golden-samples).
3. **İzlenebilir artefaktlar üretildi**
   - `logs/preflight/titan_preflight.log`
   - `logs/operator_mode/*.manifest.json`

### Uzun eğitim koşusundan önce son önkoşullar
- Dataset lisans/hash iş akışı uyumlu kalmalıdır.
- Hedef donanım (GPU/edge) kaynağı ayrılmış olmalıdır.
- Tam eğitim koşusu ve benchmark çıktıları bu önkoşullardan sonra kayda alınır.
- Varsayılan token bütçesi artık `fixed_steps` (45K). `open_ended` yalnızca açık hedef override ile kullanılmalıdır.
- Offline koşularda stage JSONL önceden üretilmelidir (`python scripts/data_pipeline.py`).

### Başlatma komutu (önkoşullar tamamlandığında)
```bash
TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash run.sh
```

### Taşınabilir Eğitim Hazırlık Checklist'i (Zip/Taşı/Çalıştır)
1. Profil sözleşmesini seç:
```bash
#- Stabil baseline (varsayılan)
bash run.sh --train-ready

#- Max mimari overlay (ileri bayraklar tek anahtar)
TITAN_PROFILE=max_arch bash run.sh --train-ready
```
2. Eğitimi başlatmadan strict readiness doğrulaması:
```bash
bash run.sh --train-ready
```
3. Gerekli ortam değişkenleri:
- `HF_TOKEN` (zorunlu, gated teacher + online dataset erişimi)
- `WANDB_API_KEY` (opsiyonel)
4. Transfer/unzip sonrası tek-komut eğitim başlatma:
```bash
bash run.sh
```
5. Strict readiness raporu:
- `logs/preflight/train_ready_status.json` (`status`, `reason_code`, kontrol detayları)
6. Dataset manifest politikası:
- Build30 Final Convergence turunda mevcut dataset manifesti sabit tutulur (major genişleme yok).

| Mühendislik Durumu | `Pilota hazır eğitim öncesi baseline` |
| :--- | :--- |
| **Eğitim Başlatma Hazırlığı** | ✅ ONAYLI (`kapılar yeşil, başlatma komutu hazır`) |
| **Kod Tabanı** | ✅ Uygulandı (testler + offline preflight geçiyor) |
| **Offline Doğrulama** | ✅ PASS (`bash scripts/verify_all.sh`) |
| **Dataset Uyumu** | ✅ Eğitim başlangıcı uyumlu (`lisans/hash iş akışı aktif; sürekli güncellenir`) |
| **Tam Eğitim Koşusu** | ▶️ Henüz başlatılmadı (`ayrılmış donanımda ilk uzun koşu ile başlar`) |
| **Benchmarklar** | ⛔ Eğitimli checkpoint olmadan iddia için uygun değil (`NOT ELIGIBLE FOR CLAIM`) |

### Parametre Açıklaması (İddia Sınırı)
- **Tasarım hedefi (Build 30 V2):** `2.64B` parametre.
- **Son ölçülen runtime toplamı:** `3,698,246,156` parametre (`~3.70B`).
- **Yorum:** `2.64B` mimari/konumlandırma hedefidir; `~3.70B` mevcut ölçülen runtime toplamıdır ve olgusal iddialarda esas alınmalıdır.

Mühendislik gerçeği (katı): `reports/verified_matrix_TR.md`.

> **MertFormer Titan, yapay zeka çıkarım maliyetlerini cihaz düzeyinde minimize ederek kurumsal zekayı merkezsizleştiren yapısal bir verimlilik standardıdır.**

---

### 💼 Yönetici Özeti (Executive Brief)
**Bu bölüm, yapısal verimlilik konumlandırmasını teknik ve yönetici karar vericiler için operasyonel çıktılara dönüştürür.**

*   **💰 Hedeflenen ~%90 Operasyonel Tasarruf (Tahmini/Hedef)**: Bulut sunucu masrafları minimize edilir. MertFormer, enerjiyi NPU düzeyinde optimize ederek işlem maliyetlerini azaltmayı hedefler.
*   **🛡️ Veri Egemenliği**: Veriler cihazda işlenir. Bu, savunma sanayi, hukuk ve finans gibi "yüksek güvenlik" standartlarına sahip pazarlar için yapısal bir avantajdır.
*   **🌍 Ölçeklenebilir Erişim (Hedef)**: İnternet bağımlılığı olmadan, düşük bant genişliğine sahip bölgelerde bile GPT-3.5 düzeyini hedefleyen otonom bir sistemdir.

*Not: Tüm performans ve eğitim süresi rakamları eğitim öncesi tahmindir; tam eğitim sonrası ölçümle doğrulanacaktır.*

---

### 🏰 Stratejik Hendek (The Strategic Moat)
**Neden MertFormer Titan kopyalanamaz?**
1.  **Cihaz Özgü Mimari (Edge-Native Focus)**: Büyük teknoloji şirketlerinin modelleri bulut üzerinde devasa hesaplama gücü için optimize edilmiştir. Titan'ın 1.58-bit katmanları, donanıma doğrudan entegre (hardware-aware) olarak tasarlanmıştır; bu, sonradan kuantize edilen modellere göre net bir verimlilik farkı yaratır.
2.  **Liquid Momentum**: Tescilli `LiquidRouter`, veriyi sadece statik bir girdi olarak değil, bir zamansal akış (momentum) olarak işler. Bu matematiksel yaklaşım, sistemi rakiplerin sadece işlem gücüyle kapatamayacağı bir avantajla konumlandırır.
3.  **Adli Güven**: Zincirlenmiş eğitim logları ve kriptografik çıktılar, projenin şeffaflığını ve kurumsal/askeri güven standartlarına uyumunu doğrular.

### 🔒 Lisans Sınırı (Hızlı Not)
- Bu depo **özel ve gizlidir**.
- Kaynak kod, varlıklar ve yöntemler yalnızca açık yazılı sözleşme veya iş ilişkisi kapsamında kullanılabilir.
- Gizli teknik detayların üçüncü taraflarla paylaşımı, imzalı NDA şartlarına tabidir.
- Tam hukuki çerçeve için [`LICENSE`](LICENSE) ve [`LICENSE_TR`](LICENSE_TR) dosyaları geçerlidir.

---

[![Lisans: Özel (Proprietary)](https://img.shields.io/badge/Lisans-%C3%96zel-red.svg?style=flat-square)](./LICENSE)
[![Depo: Gizli](https://img.shields.io/badge/Depo-Gizli-orange.svg?style=flat-square)](https://github.com/latentcore/mertformer-titan-core)
[![Durum: Eğitim Öncesi](https://img.shields.io/badge/Durum-E%C4%9Fitim%20%C3%96ncesi-yellow.svg?style=flat-square)](https://github.com/latentcore/mertformer-titan-core)
[![Mimari: BitNet 1.58b](https://img.shields.io/badge/Mimari-BitNet%201.58b-orange.svg?style=flat-square)](https://www.microsoft.com/en-us/research/publication/the-era-of-1-bit-llms-all-large-language-models-are-in-1-58-bits/)
[![Referans: BitNet 1-bit](https://img.shields.io/badge/Referans-BitNet%201--bit-lightgrey.svg?style=flat-square)](https://arxiv.org/abs/2310.11453)

## 🏗️ Tasarım İlkeleri
*   **Önce Üretim (Production-First)**: İlk günden itibaren stabilite, güvenlik ve ölçeklenebilirlik için tasarlandı.
*   **Güvenlik Odaklı Mimari**: Dahili anahtar yönetimi, rol tabanlı erişim ve saldırı simülasyonları.
*   **Ölçeklenebilir Sürü Orkestrasyonu**: Görev karmaşıklığına göre 3 ajandan (Nano) 45 ajana (Omega) kadar otomatik ölçekleme.
*   **Gözlemlenebilirlik**: Tam loglama, otopsi analizi (post-mortem) ve adli denetim izleri.

---

## 📋 İçindekiler

- [Doküman Dizini](#docs-index)
- [Genel Bakış](#genel-bakış)
- [Gerçek Dünya Uygulaması (Deneysel)](#real-world-application-experimental-tr)
- [Temel Özellikler](#temel-özellikler)
- [Mimari](#mimari)
- [Performans](#performans)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [SDK API Hızlı Başlangıç](#sdk-api-tr-quickstart)
- [Sorun Giderme](#sorun-giderme)
- [Eğitim](#eğitim)
- [Eğitim Stratejisi (Baseline -> v28)](#egitim-stratejisi-baseline-v28)
- [Dağıtım (Deployment)](#dağıtım-deployment)
- [Entegrasyon Hedefleri](#entegrasyon-hedefleri)
- [Kıyaslamalar (Benchmarks)](#kıyaslamalar-benchmarks)
- [Türkiye Vizyonu](#türkiye-vizyonu)
- [SSS](#sss)
- [Lisans](#lisans)
- [Stratejik İş Birliği](#stratejik-is-birligi)
- [Ölçeklenebilirlik Vizyonu](#olceklenebilirlik-vizyonu)
- [İletişim](#iletişim)

---

<a id="docs-index"></a>
## 📚 Doküman Dizini

**Çekirdek**
Ana giriş dokümanları ve checklistler.
- [README.md](README.md) — İngilizce genel bakış.
- [README_TR.md](README_TR.md) — Türkçe genel bakış.
- [CITATION.cff](CITATION.cff) — Atıf metadata dosyası.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Katkı yönergeleri (dahili kullanım).
- [CONTRIBUTING_TR.md](CONTRIBUTING_TR.md) — Katkı yönergeleri (TR).
- [README_CHECKLIST.md](README_CHECKLIST.md) — README denetim checklist'i (EN).
- [README_CHECKLIST_TR.md](README_CHECKLIST_TR.md) — README denetim checklist'i (TR).
- [scripts/README.md](scripts/README.md) — Script kataloğu (EN).
- [scripts/README_TR.md](scripts/README_TR.md) — Script kataloğu (TR).
- [snake_demo.py](snake_demo.py) — Pygame cyberpunk Snake autoplayer (LIVE DEMO).
- [USAGE_GUIDE.md](USAGE_GUIDE.md) — Operasyonel kullanım kılavuzu (EN).
- [USAGE_GUIDE_TR.md](USAGE_GUIDE_TR.md) — Operasyonel kullanım kılavuzu (TR).
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Sorun giderme kılavuzu (EN).
- [TROUBLESHOOTING_TR.md](TROUBLESHOOTING_TR.md) — Sorun giderme kılavuzu (TR).
- [MODEL_LICENSE.md](MODEL_LICENSE.md) — Model lisansı özeti (EN).
- [MODEL_LICENSE_TR.md](MODEL_LICENSE_TR.md) — Model lisansı özeti (TR).
- [.env.example](.env.example) — Ortam değişkeni şablonu.
- [docs/CHAIN_MAP.md](docs/CHAIN_MAP.md) — Bağlı vs bağımsız zincir haritası (EN).
- [docs/CHAIN_MAP_TR.md](docs/CHAIN_MAP_TR.md) — Bağlı vs bağımsız zincir haritası (TR).
- [reports/commercial_handover/known_issues.md](reports/commercial_handover/known_issues.md) — Devir risk görünürlüğü için bilinen sorunlar kaydı.
- [reports/commercial_handover/handover_scope.md](reports/commercial_handover/handover_scope.md) — Devir kapsamı ve kapsam dışı sınırlar.
- [reports/commercial_handover/ownership_and_role.md](reports/commercial_handover/ownership_and_role.md) — Devir sonrası sahiplik modeli ve karar hakları.
- [reports/commercial_handover/sla_kpi_90_180.md](reports/commercial_handover/sla_kpi_90_180.md) — 90/180 gün SLA ve KPI işletim planı.
- [reports/commercial_handover/contract_terms_checklist.md](reports/commercial_handover/contract_terms_checklist.md) — IP, sorumluluk, operasyon ve çıkış için sözleşme checklisti.

**SDK**
Edge dağıtım için paket + CLI.
- [mertformer_sdk/](mertformer_sdk/) — SDK paketi (API + CLI + kernel).
- [SDK_GUIDE.md](SDK_GUIDE.md) — SDK hızlı kılavuz (EN).
- [SDK_GUIDE_TR.md](SDK_GUIDE_TR.md) — SDK hızlı kılavuz (TR).

**Planlar**
Yol haritaları ve operatör planları.
- [TASK.md](TASK.md) — Operatör Modu görev planı (EN).
- [TASK_TR.md](TASK_TR.md) — Operatör Modu görev planı (TR).
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Uygulama planı (EN).
- [IMPLEMENTATION_PLAN_TR.md](IMPLEMENTATION_PLAN_TR.md) — Uygulama planı (TR).
- [TRAINING_PLAN.md](TRAINING_PLAN.md) — Eğitim yol haritası (EN).
- [TRAINING_PLAN_TR.md](TRAINING_PLAN_TR.md) — Eğitim yol haritası (TR).
- [CHANGELOG.md](CHANGELOG.md) — Sürüm değişiklik kaydı (EN).
- [CHANGELOG_TR.md](CHANGELOG_TR.md) — Sürüm değişiklik kaydı (TR).

**Teknik**
Derin teknik analiz ve araştırma referansları.
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — Teknik derin inceleme (EN).
- [TECHNICAL_REPORT_TR.md](TECHNICAL_REPORT_TR.md) — Teknik derin inceleme (TR).
- [WHITE_PAPER_LIQUIDROUTER.md](WHITE_PAPER_LIQUIDROUTER.md) — LiquidRouter white paper (EN).
- [WHITE_PAPER_LIQUIDROUTER_TR.md](WHITE_PAPER_LIQUIDROUTER_TR.md) — LiquidRouter white paper (TR).

**Dahili**
Dahili yol haritası ve yetenek boşluk haritalaması (kamusal değil).
- [INTERNAL_AGI_GAP.md](INTERNAL_AGI_GAP.md) — Dahili AGI boşluk haritası (EN).
- [INTERNAL_AGI_GAP_TR.md](INTERNAL_AGI_GAP_TR.md) — Dahili AGI boşluk haritası (TR).

**Denetim & Strateji**
Rapor doğruluk denetimi ve stratejik değer özeti.
- [reports/report_accuracy_audit.md](reports/report_accuracy_audit.md) — Rapor doğruluk denetimi (EN).
- [reports/report_accuracy_audit_TR.md](reports/report_accuracy_audit_TR.md) — Rapor doğruluk denetimi (TR).
- [reports/codex_deep_audit_EN.md](reports/codex_deep_audit_EN.md) — Derin mühendislik denetimi (EN).
- [reports/codex_deep_audit_DE.md](reports/codex_deep_audit_DE.md) — Derin mühendislik denetimi (DE).
- [reports/codex_deep_audit_TR.md](reports/codex_deep_audit_TR.md) — Derin mühendislik denetimi (TR).
- [reports/codex_deep_audit_EN_TR.md](reports/codex_deep_audit_EN_TR.md) — EN denetim raporu için TR pointer dosyası (kanonik içerik `codex_deep_audit_TR.md`).
- [reports/codex_deep_audit_DE_TR.md](reports/codex_deep_audit_DE_TR.md) — DE denetim raporu için TR pointer dosyası (kanonik içerik `codex_deep_audit_TR.md`).
- DE dilindeki denetim dosyaları, Almanca konuşan paydaşlar için dış inceleme artifact’i olarak korunur.
- [reports/verified_matrix.md](reports/verified_matrix.md) — Verified vs Target matrisi (EN).
- [reports/verified_matrix_TR.md](reports/verified_matrix_TR.md) — Verified vs Target matrisi (TR).
- [reports/review_checklist.md](reports/review_checklist.md) — Dış inceleme checklist'i (EN).
- [reports/review_checklist_TR.md](reports/review_checklist_TR.md) — Dış inceleme checklist'i (TR).
- [reports/release_snapshot.md](reports/release_snapshot.md) — Release snapshot (EN).
- [reports/release_snapshot_TR.md](reports/release_snapshot_TR.md) — Release snapshot (TR).
- [reports/final_sync_matrix.md](reports/final_sync_matrix.md) — Final senkron matris (EN).
- [reports/final_sync_matrix_TR.md](reports/final_sync_matrix_TR.md) — Final senkron matris (TR).
- [reports/go_status_matrix.md](reports/go_status_matrix.md) — GO/NO-GO durum matrisi (EN).
- [reports/go_status_matrix_TR.md](reports/go_status_matrix_TR.md) — GO/NO-GO durum matrisi (TR).
- [reports/go_nogo_signoff_onepager.md](reports/go_nogo_signoff_onepager.md) — Teknik GO/NO-GO tek sayfa imza özeti (EN).
- [reports/go_nogo_signoff_onepager_TR.md](reports/go_nogo_signoff_onepager_TR.md) — Teknik GO/NO-GO tek sayfa imza özeti (TR).
- [reports/cleanroom_verification.md](reports/cleanroom_verification.md) — Temiz clone tekrar üretim kanıtı (EN).
- [reports/cleanroom_verification_TR.md](reports/cleanroom_verification_TR.md) — Temiz clone tekrar üretim kanıtı (TR).
- [reports/legal_cleanroom_signoff_internal.md](reports/legal_cleanroom_signoff_internal.md) — Dahili cleanroom hukuki imza kaydı (EN).
- [reports/teacher_output_license_assessment.md](reports/teacher_output_license_assessment.md) — Teacher/output lisans dahili değerlendirme (EN).
- [reports/contamination_report_build30.md](reports/contamination_report_build30.md) — Build30 contamination raporu (EN).
- [reports/kpi_contract_build30.md](reports/kpi_contract_build30.md) — GO kararı için teknik KPI sözleşmesi (EN).
- [reports/benchmarks/README.md](reports/benchmarks/README.md) — Benchmark çıktıları rehberi (EN).
- [reports/benchmarks/README_TR.md](reports/benchmarks/README_TR.md) — Benchmark çıktıları rehberi (TR).
- [reports/benchmarks/smoke_train_metrics.json](reports/benchmarks/smoke_train_metrics.json) — Smoke benchmark metrik snapshot'ı (makine-okur).
- [reports/strategic_value.md](reports/strategic_value.md) — Stratejik değer özeti (EN).
- [reports/strategic_value_TR.md](reports/strategic_value_TR.md) — Stratejik değer özeti (TR).
- [reports/efficiency_convergence_analysis.md](reports/efficiency_convergence_analysis.md) — Yakınsama analizi (BitNet/Liquid/MoE, öngörü, EN).
- [reports/efficiency_convergence_analysis_TR.md](reports/efficiency_convergence_analysis_TR.md) — Yakınsama analizi (BitNet/Liquid/MoE, öngörü, TR).

**Sunum & Asset**
Yatırımcı materyalleri ve lansman varlıkları.
- [PITCH.md](PITCH.md) — Yatırımcı pitch (EN).
- [PITCH_TR.md](PITCH_TR.md) — Yatırımcı pitch (TR).
- [reports/investor_deck.pptx](reports/investor_deck.pptx) — Yatırımcı deck (EN).
- [reports/investor_deck_TR.pptx](reports/investor_deck_TR.pptx) — Yatırımcı deck (TR).
- [reports/one_pager.md](reports/one_pager.md) — One-pager (EN).
- [reports/one_pager_TR.md](reports/one_pager_TR.md) — One-pager (TR).
- [reports/technical_snapshot.md](reports/technical_snapshot.md) — Teknik snapshot (EN).
- [reports/technical_snapshot_TR.md](reports/technical_snapshot_TR.md) — Teknik snapshot (TR).
- [reports/asset_stack.md](reports/asset_stack.md) — Asset index (EN).
- [reports/asset_stack_TR.md](reports/asset_stack_TR.md) — Asset index (TR).
- [assets/snake_demo_proof.mp4](assets/snake_demo_proof.mp4) — 30 saniyelik snake demo kanıt videosu.
- [assets/snake_demo_preview.gif](assets/snake_demo_preview.gif) — Gömülü snake demo önizlemesi (GIF).
- [assets/sources/README.md](assets/sources/README.md) — Düzenlenebilir görsel kaynak arşiv standardı (EN).
- [assets/sources/README_TR.md](assets/sources/README_TR.md) — Düzenlenebilir görsel kaynak arşiv standardı (TR).

![Snake Demo Önizleme](assets/snake_demo_preview.gif)

Tam video: [assets/snake_demo_proof.mp4](assets/snake_demo_proof.mp4)
- [reports/founders_hub_application.md](reports/founders_hub_application.md) — Founders Hub taslağı (EN).
- [reports/founders_hub_application_TR.md](reports/founders_hub_application_TR.md) — Founders Hub taslağı (TR).
- [reports/security_compliance.md](reports/security_compliance.md) — Güvenlik & uyum özeti (EN).
- [reports/security_compliance_TR.md](reports/security_compliance_TR.md) — Güvenlik & uyum özeti (TR).
- [reports/poc_protocol.md](reports/poc_protocol.md) — Pilot/PoC protokolü (EN).
- [reports/poc_protocol_TR.md](reports/poc_protocol_TR.md) — Pilot/PoC protokolü (TR).
- [reports/pilot_readiness_kit.md](reports/pilot_readiness_kit.md) — Pilot hazırlık kiti (EN).
- [reports/pilot_readiness_kit_TR.md](reports/pilot_readiness_kit_TR.md) — Pilot hazırlık kiti (TR).
- [reports/pilot_offer_packages.md](reports/pilot_offer_packages.md) — Standart pilot teklif paketleri (EN).
- [reports/pilot_offer_packages_TR.md](reports/pilot_offer_packages_TR.md) — Standart pilot teklif paketleri (TR).
- [reports/sales_funnel_90d.md](reports/sales_funnel_90d.md) — 90 günlük B2B pilot satış hunisi (EN).
- [reports/sales_funnel_90d_TR.md](reports/sales_funnel_90d_TR.md) — 90 günlük B2B pilot satış hunisi (TR).
- [reports/drone_sitl_demo.md](reports/drone_sitl_demo.md) — SITL drone kanıt protokolü (EN).
- [reports/drone_sitl_demo_TR.md](reports/drone_sitl_demo_TR.md) — SITL drone kanıt protokolü (TR).
- [reports/pilots/README.md](reports/pilots/README.md) — Pilot kanıt klasörü standardı (EN).
- [reports/pilots/README_TR.md](reports/pilots/README_TR.md) — Pilot kanıt klasörü standardı (TR).
- [reports/pilot_acceptance_signoff.md](reports/pilot_acceptance_signoff.md) — Pilot kabul imza şablonu (EN).
- [reports/pilot_acceptance_signoff_TR.md](reports/pilot_acceptance_signoff_TR.md) — Pilot kabul imza şablonu (TR).
- [reports/ip_licensing_split.md](reports/ip_licensing_split.md) — Sektörel fikri hak ayrımı çerçevesi (EN).
- [reports/ip_licensing_split_TR.md](reports/ip_licensing_split_TR.md) — Sektörel fikri hak ayrımı çerçevesi (TR).
- [reports/dataset_health.md](reports/dataset_health.md) — Dataset sağlık raporu (EN).
- [reports/dataset_health_TR.md](reports/dataset_health_TR.md) — Dataset sağlık raporu (TR).
- [reports/model_health.md](reports/model_health.md) — Model sağlık raporu (EN).
- [reports/model_health_TR.md](reports/model_health_TR.md) — Model sağlık raporu (TR).
- [reports/system_hardware.md](reports/system_hardware.md) — Sistem donanım raporu (EN).
- [reports/system_hardware_TR.md](reports/system_hardware_TR.md) — Sistem donanım raporu (TR).
- [reports/cli_smoke_log.md](reports/cli_smoke_log.md) — CLI smoke log (EN).
- [reports/cli_smoke_log_TR.md](reports/cli_smoke_log_TR.md) — CLI smoke log (TR).

*Not: Bu adımda PPTX dosyalarına dokunulmadı (plan gereği). Gerekirse sonra tek satır ekleriz.*

**Operasyon & Yönetişim**
Güvenlik, veri kökeni, yeniden üretilebilirlik ve operasyon notları.
- [MODEL_CARD.md](MODEL_CARD.md) — Model kartı (EN).
- [MODEL_CARD_TR.md](MODEL_CARD_TR.md) — Model kartı (TR).
- [USE_POLICY.md](USE_POLICY.md) — Kullanım politikası (EN).
- [USE_POLICY_TR.md](USE_POLICY_TR.md) — Kullanım politikası (TR).
- [SECURITY.md](SECURITY.md) — Güvenlik politikası (EN).
- [SECURITY_TR.md](SECURITY_TR.md) — Güvenlik politikası (TR).
- [DECISIONS.md](DECISIONS.md) — Mimari kararlar (EN).
- [DECISIONS_TR.md](DECISIONS_TR.md) — Mimari kararlar (TR).
- [datasets/README.md](datasets/README.md) — Dataset genel bakış (EN).
- [datasets/README_TR.md](datasets/README_TR.md) — Dataset genel bakış (TR).
- [datasets/SOURCES.md](datasets/SOURCES.md) — Veri kaynakları (EN).
- [datasets/SOURCES_TR.md](datasets/SOURCES_TR.md) — Veri kaynakları (TR).
- [datasets/LICENSES.md](datasets/LICENSES.md) — Lisanslar (EN).
- [datasets/LICENSES_TR.md](datasets/LICENSES_TR.md) — Lisanslar (TR).
- [datasets/inventory.md](datasets/inventory.md) — Dataset envanteri (otomatik, EN).
- [datasets/inventory_TR.md](datasets/inventory_TR.md) — Dataset envanteri (otomatik, TR).
- [datasets/inventory.json](datasets/inventory.json) — Dataset envanteri (otomatik, makine-okur).
- [repro/seed_policy.md](repro/seed_policy.md) — Seed politikası (EN).
- [repro/seed_policy_TR.md](repro/seed_policy_TR.md) — Seed politikası (TR).
- [repro/python.md](repro/python.md) — Python 3.11 baseline kurulum (EN).
- [repro/python_TR.md](repro/python_TR.md) — Python 3.11 baseline kurulum (TR).
- [repro/accelerate_default.yaml](repro/accelerate_default.yaml) — Örnek accelerate config (yerel).
- [repro/pip_freeze.txt](repro/pip_freeze.txt) — Ortam envanteri (pip freeze).
- [logs/README.md](logs/README.md) — Log dizini + birleşik logbook notları.
- `logs/ALL_LOGS.jsonl` — Birleşik logbook artifact (gitignored; `.titan-venv/bin/python scripts/logbook_build.py --append` ile üretilir).
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — CI pipeline (pytest + preflight + secret scan).
- [interfaces/inference_contract.md](interfaces/inference_contract.md) — Çıkarım sözleşmesi (EN).
- [interfaces/inference_contract_TR.md](interfaces/inference_contract_TR.md) — Çıkarım sözleşmesi (TR).
- [interfaces/pilot_report_v1.schema.json](interfaces/pilot_report_v1.schema.json) — Pilot raporu JSON şeması.
- [economics/cost_model.md](economics/cost_model.md) — Maliyet modeli (EN).
- [economics/cost_model_TR.md](economics/cost_model_TR.md) — Maliyet modeli (TR).
- [economics/efficiency_report.md](economics/efficiency_report.md) — Verim raporu (EN).
- [economics/efficiency_report_TR.md](economics/efficiency_report_TR.md) — Verim raporu (TR).
- [limits/scaling_breakpoints.md](limits/scaling_breakpoints.md) — Ölçek kırılma noktaları (EN).
- [limits/scaling_breakpoints_TR.md](limits/scaling_breakpoints_TR.md) — Ölçek kırılma noktaları (TR).
- [postmortems/README.md](postmortems/README.md) — Olay raporu dizini (EN).
- [postmortems/README_TR.md](postmortems/README_TR.md) — Olay raporu dizini (TR).
- [postmortems/_template.md](postmortems/_template.md) — Postmortem şablonu (EN).
- [postmortems/_template_TR.md](postmortems/_template_TR.md) — Postmortem şablonu (TR).
- [postmortems/example_001.md](postmortems/example_001.md) — Postmortem örneği (EN).
- [postmortems/example_001_TR.md](postmortems/example_001_TR.md) — Postmortem örneği (TR).
- [prompts/changelog.md](prompts/changelog.md) — Prompt değişim günlüğü (EN).
- [prompts/changelog_TR.md](prompts/changelog_TR.md) — Prompt değişim günlüğü (TR).
- [tokenizer/stats.md](tokenizer/stats.md) — Tokenizer istatistikleri (EN).
- [tokenizer/stats_TR.md](tokenizer/stats_TR.md) — Tokenizer istatistikleri (TR).
- [tokenizer/drift_report.md](tokenizer/drift_report.md) — Tokenizer drift raporu (EN).
- [tokenizer/drift_report_TR.md](tokenizer/drift_report_TR.md) — Tokenizer drift raporu (TR).
- [tokenizer/tr/README.md](tokenizer/tr/README.md) — Turkish tokenizer cache note (EN).
- [tokenizer/tr/README_TR.md](tokenizer/tr/README_TR.md) — Turkish tokenizer cache note (TR).
- [ablations/results.md](ablations/results.md) — Ablation sonuçları (EN).
- [ablations/results_TR.md](ablations/results_TR.md) — Ablation sonuçları (TR).
- [ablations/no_moe/README.md](ablations/no_moe/README.md) — MoE kapalı ablation (EN).
- [ablations/no_moe/README_TR.md](ablations/no_moe/README_TR.md) — MoE kapalı ablation (TR).
- [ablations/no_liquid/README.md](ablations/no_liquid/README.md) — Liquid kapalı ablation (EN).
- [ablations/no_liquid/README_TR.md](ablations/no_liquid/README_TR.md) — Liquid kapalı ablation (TR).
- [ablations/dense_only/README.md](ablations/dense_only/README.md) — Dense-only ablation (EN).
- [ablations/dense_only/README_TR.md](ablations/dense_only/README_TR.md) — Dense-only ablation (TR).
- [ablations/bitlinear_off/README.md](ablations/bitlinear_off/README.md) — BitNet kapalı ablation (EN).
- [ablations/bitlinear_off/README_TR.md](ablations/bitlinear_off/README_TR.md) — BitNet kapalı ablation (TR).
- [experiments/exp_001_baseline/notes.md](experiments/exp_001_baseline/notes.md) — Deney notları (EN).
- [experiments/exp_001_baseline/notes_TR.md](experiments/exp_001_baseline/notes_TR.md) — Deney notları (TR).
- [tools/abuse_tests.md](tools/abuse_tests.md) — Tool abuse testleri (EN).
- [tools/abuse_tests_TR.md](tools/abuse_tests_TR.md) — Tool abuse testleri (TR).
- [tools/sandbox/README.md](tools/sandbox/README.md) — Tool sandbox (EN).
- [tools/sandbox/README_TR.md](tools/sandbox/README_TR.md) — Tool sandbox (TR).
- [tools/contracts/README.md](tools/contracts/README.md) — Tool sözleşmeleri (EN).
- [tools/contracts/README_TR.md](tools/contracts/README_TR.md) — Tool sözleşmeleri (TR).
- [training_dynamics/cold_vs_warm.md](training_dynamics/cold_vs_warm.md) — Eğitim dinamiği notu (EN).
- [training_dynamics/cold_vs_warm_TR.md](training_dynamics/cold_vs_warm_TR.md) — Eğitim dinamiği notu (TR).

---

<a id="genel-bakış"></a>
## 🎯 Genel Bakış

MertFormer Titan, mobil platformlarda **cihaz içi çıkarım (inference)** için tasarlanmış, son teknoloji ürünü **2.64B parametreli** bir dil modelidir. **BitNet 1.58-bit kuantizasyon**, **Liquid Neural Networks**, **Seyrek Uzmanlar Karışımı (MoE)** ve **MLA etiketli GQA dikkat bloğu (mevcut implementasyon)** teknolojilerini birleştirerek, tamamen bir akıllı telefonda çalışırken **GPT-3.5 seviyesinde performans hedefler (eğitim öncesi hedef)**.

Mimari doğruluk notu: `layers/mla.py` sınıf adı `MLA` olarak korunur; mevcut attention çekirdeği GQA tabanlıdır (`num_kv_heads` projeksiyonu + KV head çoğaltma). Tam latent-MLA bottleneck yol haritası kalemidir.

İsim açılımı:
- **MERT**: **Modüler Uçta Akıl Yürütme Transformer**
- **MertFormer**: **Cihaz Üstü Modüler Yürütme ve Güvenilirlik için Modüler Uçta Akıl Yürütme Transformer Çatısı**

### 🔗 Zincir Haritası (Bağlı vs Bağımsız)
```mermaid
flowchart TD
  A["Stage JSONL (datasets/stage*)"] --> B["Eğitim (run.sh → train/train.py)"]
  B --> C["Loglar (logs/*.jsonl)"]
  C --> D["SOP artefaktları (reports + packages/artifacts zip)"]
```

Tam harita: [`docs/CHAIN_MAP_TR.md`](docs/CHAIN_MAP_TR.md)

### Neden MertFormer Titan?

- 🛡️ **Önce Gizlilik**: %100 cihaz içi, sıfır bulut bağımlılığı
- ⚡ **Ultra Verimli**: BitNet kuantizasyonu ile teorik **FP32’ye göre ~20x daha küçük** (32-bit → 1.58-bit; low-bit inference yolu gerektirir)
- 🏭 **Endüstriyel Sınıf**: Endüstri standardı optimizasyonlar (Flash Attention 2, torch.compile, NCCL tuning)
- 📱 **Mobil Optimize**: Samsung S25 NPU için JIT derlemesi
- 🧪 **Araştırma Düzeyi**: Özgün LiquidRouter mimarisi (bağlamsal MoE yönlendirme)
- 🇹🇷 **Türkiye'ye Hazır**: Türk dili ve kültürü için optimize edildi

<a id="real-world-application-experimental-tr"></a>
### 🚁 Gerçek Dünya Uygulaması (Deneysel)

- **Proof-of-system hedefi:** Gerçek dünya kısıtlarında, otonom UAV/drone sınıfı platformlarda doğrulanabilir olacak şekilde tasarlanmıştır.
- **Sistem odağı:** Kısıtlı donanım, gecikme ve sensör belirsizliği altında algı → karar → kontrol hattının çalışmasıdır.
- **Güvenlik önceliği:** Risk/güven eşiği aşıldığında fail-safe guardrail ve watchdog benzeri override mekanizmalarıyla deterministik fallback davranışı hedeflenir.
- **Konumlandırma:** Bu sürüm mühendislik doğrulama kapsamındadır; sertifikalı/production dağıtım iddiası sunmaz.
- **Mevcut kısıt:** Doğrulama hızı, GPU/edge donanım ve kontrollü saha testi kaynaklarına erişimle sınırlıdır.
- **İş birliği çağrısı:** Kilometre taşlarını hızlandırmak için compute desteği, kontrollü test ortamı ve mühendislik mentorluğu iş birliği arıyoruz.

---

<a id="temel-özellikler"></a>
## 🔥 Temel Özellikler

### 1. **BitNet 1.58-bit Kuantizasyon** 🤏
- Üçlü (Ternary) ağırlıklar: `{-1, 0, +1}`
- INT8 aktivasyonlar: `[-127, 127]`
- **FP32’ye göre teorik ~20x daha küçük** (32-bit → 1.58-bit; low-bit inference yolu gerektirir)
- Gradyan akışı için Straight-Through Estimator (STE)
- Stabilite için RMS ölçekleme (legacy yol Build 30 V2 içine entegre edildi)

### 2. **LiquidRouter (Zamansal Conv Yönlendirici)** 🌍
- **Implementasyon gerçeği**: MoE yönlendirmesi causal depthwise `Conv1d` + rolling state buffer ile yapılır.
- **CfC ayrımı**: Kapalı form sürekli zamanlı (CfC) hücreler `LiquidMixer/LiquidCell` içinde çalışır; `LiquidRouter` içinde değil.
- **Etki**: Standart (hafızasız) yönlendiricilere kıyasla **tahmini %15-20 daha iyi yönlendirme kalitesi**.
- **Zamansal Rota**: Geçmişi hatırlayan "Trafik Polisi" mantığıyla uzman çökmesini önler.
- **Dinamik**: Stabilite için zaman sabiti adaptasyonu ve jitter desteği.
- **Akademik Değer**: Koşullu hesaplamada (conditional computation) yeni bir paradigma.

### 3. **MLA Etiketli GQA Dikkat Bloğu (Mevcut Implementasyon)** 🧠
- GQA tabanlı KV paylaşımı (`num_heads=16`, `num_kv_heads=8` varsayılan profil).
- LLaMA-3 uyumlu RoPE (interleaved & decoupled)
- Opsiyonel hiyerarşik KV cache yolu (kısa/uzun ayrımı) ile decode verimliliği.
- Stabilite için QK normalizasyonu
- Flash Attention 2 entegrasyonu (+%30 hızlanma)
- Uzun bağlam hazır (theta=100K)

### 4. **Liquid Neural Networks (CfC)** 💧
- Gerçek Kapalı Form Sürekli Zamanlı hücreler
- Dinamik tau (zaman sabiti) adaptasyonu
- Zamansal akıl yürütme yetenekleri
- NPU optimizasyonu için JIT derlemeli
- 3 aşamalı koruma sistemi (3-strike safeguard)

### 5. **Seyrek Uzmanlar Karışımı (MoE) & 🚀 LiquidRouter** 🧩
- 8 uzman, top-2 yönlendirme
- Yönlendirme politikası: token-choice top-k.
- **Momentum Bazlı Yönlendirme:** Standart yönlendiricilerin aksine, `LiquidRouter` sadece anlık kelimeye değil, verinin geliş hızına ve zamansal momentumuna (`Fluid Path`) bakarak uzman seçer.
- **Causal Conv1d Entegrasyonu:** Uzman seçimi sırasında geçmiş 4 token'lık pencereyi (`history_window`) dikkate alarak "trafik polisinden" ziyade bir "stratejik zeka" gibi çalışır.
- **Donanım Uyumluluğu:** `LiquidRouter`ın keskin seçimleri sayesinde gereksiz uzmanların tetiklenmesi önlenir, bu da Samsung S25 NPU biriminde tahmini %40'a varan enerji tasarrufu sağlar.
- Yük dengeleme + Z-loss + Switch loss
- BitSwiGLU uzmanları (kuantize edilmiş)
- Çökme önleme için acil durum jitter desteği
- Yönlendirici sağlık izleme

### 6. **Gelişmiş Eğitim Hattı** 🚂
- **Bilgi Damıtma (Knowledge Distillation)**: Llama-3.3-70B → 2.64B tasarım hedefi (%80 alpha)
- **4 Çekirdek Aşama + 1 Araç/API Fazı**: Mantık → Bilgi → Dil → Ruh (+ Araç Kullanımı/API)
- **WSD Zamanlayıcı**: Warmup-Stable-Decay (grokking optimize edilmiş)
- **Diferansiyel Öğrenme Oranları**: Router 1.5x, Gövde 1.0x
- **Erken Durdurma**: Sabır tabanlı en iyi kontrol noktası kaydı
- **Dinamik Alpha**: Aşamalı damıtma ağırlığı ayarlaması

### 7. **Performans Optimizasyonları (v1.0 (Build 30 V2))** ⚡
- ✅ **Flash Attention 2**: Tahmini +%30 hızlanma (A100/H100)
- ✅ **Fused RMSNorm**: Tahmini +%10 hızlanma (torch.compile)
- ✅ **torch.compile (max-autotune)**: Tahmini +%15 hızlanma
- ✅ **CUDA TF32 + cuDNN**: Tahmini +%10 hızlanma
- ✅ **Geliştirilmiş DataLoader**: Tahmini +%5 hızlanma (16 işçi, prefetch=4)
- ✅ **NCCL Tuning**: Tahmini +%5-10 hızlanma (multi-GPU, otomatik algılama)
- **Tahmini toplam: %70-80 daha hızlı eğitim.**

### 8. **Güvenlik & Güvenilirlik** 🛡️
- ✅ **OOM Kurtarma**: Otomatik toplu iş boyutu (batch size) azaltma
- ✅ **NaN/Inf Tespiti**: Yeniden deneme limiti ile gradyan sıfırlama
- ✅ **Disk Alanı İzleme**: Kontrol noktası kayıt hatalarını önler
- ✅ **GPU Bellek Takibi**: Gerçek zamanlı kullanım raporlaması
- ✅ **Gradyan Norm İzleme**: Çöküş/patlama tespiti
- ✅ **Liquid Ani Artış Korumaları**: 3 aşamalı dondurma mekanizması
- ✅ **En İyi Kontrol Noktası Kaydı**: Optimal model durumunu korur

### 9. **Teknolojik Üstünlük (Build 30 V2 Yükseltmesi)** 🛠️
- **GaLore Entegrasyonu**: Tüketici GPU'larında bellek verimliliği için Gradient Low-Rank Projection optimizasyonu (Kilitli).
- **8-bit AdamW**: Bellek optimize edilmiş optimizer, optimizer durum belleğini %75 azaltır (Kilitli).
- **Çevrimdışı Bilgi Damıtma (Offline KD)**: Sıfır yüklü öğretmen eğitimi için önceden hesaplanmış Llama-3-70B logitleri (precomputed shard gerektirir; yoksa online öğretmene düşer).
- **Akıllı Paralel Orkestrasyon (Hyper-Threading)**: Veri indirme, damıtma ve eğitimin eş zamanlı gerçekleştiği sıfır gecikmeli boru hattı.

### QINN Durumu (Mevcut Build)
- **Varsayılan durum:** `use_qinn=False` (Build 30 V2'de kapalı).
- **Şu an neden kapalı:** ana eğitim hattında stabilite, throughput ve edge/NPU uyumluluğu önceliklendirildi.
- **İleride açılırsa:** deneysel bir düzenleme katmanı olarak ablation ile test edilebilir; ek hesaplama yükü ve yakınsama riski oluşturabilir.
- **Referans dosya:** `layers/qinn.py` (kontrollü deneyler için kod tabanında tutulur).

---

<a id="mimari"></a>
## 🏗️ Mimari

### Build 30 V2 Bilişsel Genişletmeler (Feature-Flag, varsayılan KAPALI)
Bu modüller kodda uygulanmıştır ve eğitim/çıkarım öncesi config ile açılabilir:

- `use_hierarchical_kv_cache` -> Kısa/uzun ayrımlı hiyerarşik KV cache (`layers/mla.py`)
- `use_global_workspace_broadcast` -> Global workspace broadcast katmanı (`layers/cognitive_extensions.py`)
- `use_cross_expert_sync_bus` -> Uzmanlar arası senkronizasyon yolu (`layers/moe.py`)
- `use_latent_ode_state_channel` -> Sürekli-zaman latent ODE durum kanalı (`layers/cognitive_extensions.py`)
- `use_neuromodulatory_gain` -> Nöromodülatör gain katmanı (`layers/cognitive_extensions.py`)
- `use_hebbian_plasticity` -> Hebbian plastisite iz katmanı (`layers/cognitive_extensions.py`)
- `use_neuro_symbolic_layer` -> Nöro-sembolik artık köprü katmanı (`layers/cognitive_extensions.py`)
- `use_world_model_head` -> Nedensel dünya modeli yan-çıktıları (`layers/world_model_head.py`)
- `use_lifelong_safety_layer` -> Yaşam boyu güvenlik/adaptasyon koruması (`layers/lifelong_safety.py`)
- `use_structural_plasticity` -> Uzman büyütme-budama politika kancaları (`layers/moe.py`)
- `use_continual_adapter` -> Eğitimde continual learning adaptör yolu (`train/continual_adapter.py`)
- `use_expert_paging` -> Çıkarım-öncelikli uzman sayfalama (on-demand residency) (`layers/moe.py`)

Çalışma notu:
- Varsayılanlar KAPALI tutulur; stabil baseline korunur.
- Bu bileşenler non-breaking uzantılar olarak entegredir ve deney bazında açılır.

### İleri Özellik Matrisi (Stable vs Max-Arch)
`run.sh`, `TITAN_PROFILE` ile profil sözleşmesi destekler.

| Bayrak | Stable (varsayılan) | Max-Arch | Amaç | Dosya |
| --- | --- | --- | --- | --- |
| `use_hierarchical_kv_cache` | `false` | `true` | Decode sırasında kısa/uzun KV ayrımı | `layers/mla.py` |
| `use_global_workspace_broadcast` | `false` | `true` | Tokenlar arası ortak çalışma alanı sinyali | `layers/cognitive_extensions.py` |
| `use_neuromodulatory_gain` | `false` | `true` | Workspace tabanlı gain/bias modülasyonu | `layers/cognitive_extensions.py` |
| `use_latent_ode_state_channel` | `false` | `true` | Sürekli-zaman latent durum dinamiği | `layers/cognitive_extensions.py` |
| `use_cross_expert_sync_bus` | `false` | `true` | MoE uzmanlar arası senkronizasyon | `layers/moe.py` |
| `use_structural_plasticity` | `false` | `true` | Uzman büyütme/budama kancaları | `layers/moe.py` |
| `use_hebbian_plasticity` | `false` | `true` | Lokal plastisite iz katmanı | `layers/cognitive_extensions.py` |
| `use_neuro_symbolic_layer` | `false` | `true` | Kural-koşullu artık köprü | `layers/cognitive_extensions.py` |
| `use_world_model_head` | `false` | `true` | Yan-kanal nedensel tahmin çıktısı | `layers/world_model_head.py` |
| `use_lifelong_safety_layer` | `false` | `true` | Drift farkındalıklı adaptif güvenlik | `layers/lifelong_safety.py` |
| `use_continual_adapter` | `false` | `true` | Eğitimde continual replay/drift adaptörü | `train/continual_adapter.py` |
| `use_expert_paging` | `false` | `true` | İhtiyaç anında uzman yerleşimi (inference-first) | `layers/moe.py` |
| `use_qinn` | `false` | `false` | Build30'de stabilite/throughput için kapalı tutulur | `layers/qinn.py` |

Profil örnekleri:
```bash
#- Stabil baseline (varsayılan)
bash run.sh

#- Max mimari profil
TITAN_PROFILE=max_arch bash run.sh

#- Sadece readiness kapısı
bash run.sh --train-ready
```

```text
      ╔═══════════════════════════════════════════════════════════════════════════╗
      ║  M E R T F O R M E R   T I T A N   (O N Y X   S T O R M)                  ║
      ║  » TEKNİK PLAN v1.0 (Build 30 V2) // HEDEF: SAMSUNG S25 NPU «                ║
      ╚═══════════════════════════════════════════════════════════════════════════╝
                                            │
      ┌─────────────────────────────────────▼─────────────────────────────────────┐
      │  GİRİŞ EMBEDDINGS [Batch, Seq, 2048]  ⚡  RoPE (Theta=100k, Float32)       │
      └─────────────────────────────────────┬─────────────────────────────────────┘
                                            │ [B, S, 2048]
                  ┌─────────────────────────▼──────────────────────────┐
                  │            R E Z İ D Ü E L   A K I Ş               │◄──┐
                  └─────────────────────────┬──────────────────────────┘   │
      ┌─────────────────────────────────────▼─────────────────────────────────────┐
      │  TRANSFORMER BLOĞU [Katman 0-17]  (Yinelemeli Süreç)                      │
      │                                                                           │
      │  ┌──────────────┐    ┌─────────────────────────────────────────────────┐  │
      │  │ RMSNorm (F)  │───►│ [MLA ETIKETLI GQA] ATTENTION       │  │
      │  └──────────────┘    │ » GQA başlıkları: Q=16, KV=8 (varsayılan profil)                 │  │
      │                      │ » İşlem: Softmax(Q·K^T / √d) · V                │  │
      │                      │ » Donanım: FlashAttn2 Kernel (SRAM Optimize)    │  │
      │                      └────────────────────────┬────────────────────────┘  │
      │    (Ekle) ────────────────────────────────────┘                           │
      │      ▼                                                                    │
      │  ┌──────────────┐    ┌─────────────────────────────────────────────────┐  │
      │  │ RMSNorm (F)  │───►│ [ROUTER] LIQUID BAĞLAM FARKINDALIĞI             │  │
      │  └──────────────┘    │ » Giriş: [B, S, 2048]                           │  │
      │                      │ » İşlem: CausalConv1d(k=4) + SiLU + Linear      │  │
      │                      └────────────────────────┬────────────────────────┘  │
      │                                               ▼ [B, S, Uzman_Sayısı]      │
      │                        ┌───────────────────────────────────────┐          │
      │                        │     TOP-2 DİNAMİK UZMAN SEÇİMİ (Gate) │          │
      │     └─┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬─┘         │
      │       │       │       │       │       │       │       │       │           │
      │       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼           │
      │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
      │    │EXP_0│ │EXP_1│ │EXP_2│ │EXP_3│ │EXP_4│ │EXP_5│ │EXP_6│ │EXP_7│        │
      │    │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│        │
      │    └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘        │
      │       │       │       │       │       │       │       │       │           │
      │       └───────┴───────┴───────┼───────┴───────┴───────┴───────┘           │
      │                               ▼ [B, S, 2048]                              │
      │                     (Ağırlıklı Toplam Σ g(x)·E(x))                        │
      │                              │                                            │
      │  ┌───────────────────────────▼─────────────────────────────────────────┐  │
      │  │ [LIQUID MIXER] (Sadece Katman 4, 10, 16'da Aktif)                   │  │
      │  │ » Çekirdek: CfC (Closed-form Continuous) Hücreleri                  │  │
      │  │ » Denklem:  x(t) = (-1/τ)x(t) + A·I(t)                              │  │
      │  │ » Görev: Uzun Vadeli Bağımlılık & Zamansal Akıl Yürütme             │  │
      │  └───────────────────────────┬─────────────────────────────────────────┘  │
      │                              │                                            │
      │    (Ekle) <──────────────────┘                                            │
      │      │ [B, S, 2048]                                                       │
      └──────┼────────────────────────────────────────────────────────────────────┘
             │
      ┌──────▼────────────────────────────────────────────────────────────────────┐
      │ [RMSNorm] + [LM HEAD] 1.58-bit İzdüşüm ──► ÇIKIŞ LOGİTLERİ [B, S, 128k]   │
      └───────────────────────────────────────────────────────────────────────────┘
```

### 🦅 MertFormer Titan: Sinaptik Katman Hiyerarşisi

![Sinaptik Hiyerarşi Haritası](assets/synaptic_map.png)

Verinin 0'dan 17'ye kadar olan yolculuğu:

*   **Katman 0 (Giriş Bloğu):** Vektörleştirilen verinin ilk durağıdır; temel kelime ilişkileri kurulur ve `RMSNorm` ile sinyal genliği stabilize edilir.
*   **Katman 1 (Gramer Temeli):** Dilin en temel yapı taşları işlenir; `MLA` etiketli GQA attention mekanizması ilk odaklanma haritasını oluşturur.
*   **Katman 2 (Verimlilik Mührü):** Kelimeler arası basit bağlamlar kurulur; `BitNet 1.58-bit` yapısı sayesinde tüm ağırlıklar $\{-1, 0, +1\}$ uzayında en düşük enerjiyle işlenir.
*   **Katman 3 (Uzman Dağıtımı):** Anlamsal yoğunluk artar; `MoE` yapısı devreye girerek veriyi ilgili 8 uzmandan en uygun 2'sine yönlendirir.
*   **Katman 4 (İlk Liquid Teması):** **Kritik Eşik.** İlk `LiquidMixer` (CfC) burada devreye girerek veriye ilk "zamansal akış" ve "momentum" algısını yükler.
*   **Katman 5 (Akışkan Dikkat):** Akışkanlık kazanan veri, `MLA` etiketli GQA attention tarafından daha derin bir boyutta süzülerek bağlamsal ilişkiler güçlendirilir.
*   **Katman 6 (Karmaşık Sözdizimi):** Cümle içindeki dolaylı yapılar çözülür; `MoE` uzmanları spesifik analizlere devam eder.
*   **Katman 7 (Matematiksel Kararlılık):** Mantıksal çıkarımların temeli atılır; `UnitaryQINN` yolu yalnızca `use_qinn=true` olduğunda devreye alınır (Build 30 V2 varsayılanı: KAPALI).
*   **Katman 8 (Soyutlama):** Veri somut kelimelerden soyut kavramlara evrilir; hiyerarşik yapı `MLA` etiketli GQA attention ile derinleştirilir.
*   **Katman 9 (Niyet Analizi):** Karar mekanizmaları güçlenir; model kullanıcı niyetini ve sorunun arka planını kavramaya başlar.
*   **Katman 10 (İkinci Liquid Teması):** **Kritik Eşik.** İkinci `LiquidMixer` burada aktifleşir; karmaşık mantık yürütme sırasında verinin zamansal hafızası ve hızı dinamik olarak tazelenir.
*   **Katman 11 (Stratejik Karar):** Akışkanlık kazanan mantık, `MoE` uzmanları tarafından stratejik yanıt parametrelerine dönüştürülür.
*   **Katman 12 (Üst Seviye Anlam):** Bilgi "bilgelik" seviyesine yaklaşır; cümlenin tonu, amaçı ve hedefi bu aşamada netleşir.
*   **Katman 13 (Yanıt İnşası):** Üretilecek cevabın iskeleti kurulur; `MLA` etiketli GQA attention cevabın en kritik noktalarına odaklanır.
*   **Katman 14 (Kültürel Adaptasyon):** Teknik detaylar ile Türkçe kültürel ve deyimsel yapılar bu aşamada modele enjekte edilir.
*   **Katman 15 (Ön Final Analizi):** Cevap son formunu almadan önceki son büyük denetim ve kalite kontrol katmanıdır.
*   **Katman 16 (Nihai Liquid Mührü):** **Kritik Eşik.** Son `LiquidMixer` devreye girer; tüm bilgi çıkıştan önce nihai bir "akışkan zekaya" dönüştürülür ve zamansal tutarlılık mühürlenir.
*   **Katman 17 (Final Bloğu):** Son kontroller yapılır; `RMSNorm` ve `LM Head` aracılığıyla işlenen veriler kullanıcıya sunulacak kelime olasılıklarına (logits) dönüştürülür.
```


```mermaid
graph TD
    subgraph "MertFormer Titan: 18 Katmanlı Sinaptik Akış"
        direction TB
        Phase4["💎 BİLGELİK (WISDOM) - Katman 16-17<br/>Nihai Liquid Mührü | Bilginin Kelimeye Dönüşümü"]
        Phase3["🎭 MUHAKEME (REASONING) - Katman 10-15<br/>Liquid Momentum | Stratejik Mantık & Kültürel Adaptasyon"]
        Phase2["☁️ SOYUTLAMA (ABSTRACTION) - Katman 3-9<br/>MoE Uzman Dağıtımı | Kavramsal Derinlik & İlk Liquid Teması (L4)"]
        Phase1["🧱 TEMEL (FOUNDATION) - Katman 0-2<br/>BitNet 1.58-bit | Gramer Kurulumu & RMSNorm Stabilizasyonu"]

        Phase1 ==> Phase2 ==> Phase3 ==> Phase4
    end

    subgraph "Her Katmanın (Blok) Mühendislik Kalbi"
        style BlockInner fill:#1a1a1a,stroke:#3fb1e3,stroke-width:2px
        BlockInner[Giriş] --> Norm1[RMSNorm]
        Norm1 --> MLA["MLA Etiketli GQA Dikkat Bloğu (Mevcut Implementasyon)"]
        MLA --> Norm2[RMSNorm]
        Norm2 --> Router{"LiquidRouter (Zamansal Seçici)"}
        Router -- "En Uygun 2 Uzman" --> MoE["BitSwiGLU Uzmanları"]
        Router -- "Dinamik Akış" --> Liquid["Liquid CfC Hücresi"]
        MoE --> Combine[Kombine Sinyal]
        Liquid --> Combine
        Combine --> FinalNorm[Residual Add]
    end
```

```
MertFormer Titan (2.64B Parametre)
├── Gömme Katmanı (Embedding Layer) (128256 kelime hazinesi, Llama-3 tokenizer)
├── 18× Transformer Blokları
│   ├── RMSNorm (torch.compile ile birleştirilmiş)
│   ├── MLA etiketli GQA dikkat bloğu (mevcut implementasyon)
│   │   ├── BitLinear İzdüşümleri (Q, K, V, O)
│   │   ├── RoPE (theta=100K, uzun bağlam hazır)
│   │   ├── QK Normalizasyonu (stabilite)
│   │   ├── Flash Attention 2 (eğitim modu)
│   │   └── KV Önbellek (inference modu)
│   ├── LiquidMixer (katmanlar 4, 10, 16)
│   │   ├── LiquidCell (CfC çekirdeği)
│   │   ├── Dinamik Tau (zaman sabiti)
│   │   ├── CfC Güncelleme Kuralı
│   │   └── Artık (Residual) + LayerNorm
│   ├── RMSNorm (torch.compile ile birleştirilmiş)
│   └── Seyrek MoE / Yoğun FFN
│       ├── LiquidRouter (Conv1d bağlam farkındalıklı)
│       ├── 8× BitSwiGLU Uzmanları
│       ├── Top-2 Yönlendirme
│       ├── Aux Kaybı (yük denge + Z-loss + switch)
│       └── Acil Durum Jitter Desteği
└── LM Başlığı (BitLinear izdüşümü)
```

**Model Özellikleri:**
- **Gizli Boyut (Hidden Size)**: 2048
- **Ara Boyut (Intermediate Size)**: 5632
- **Katman Sayısı**: 18 (mobil optimize)
- **Başlık Sayısı**: 16
- **Başlık Boyutu**: 128
- **Maksimum Dizi Uzunluğu**: 4096 (8K-16K'ya genişletilebilir)
- **Kelime Hazinesi Boyutu**: 128256 (Llama-3 tokenizer)
- **RoPE Tabanı**: 100,000 (uzun bağlam desteği)

---

<a id="performans"></a>
## 📊 Performans

**İddia politikası:** Bu bölümde açıkça ölçülmüş olarak işaretlenmeyen tüm değerler hedef/tahmindir ve benchmark iddiası kanıtı değildir.

### Performans Hedefleri (Öngörülen vs Temel, Ölçülmemiş) — Eğitim Hızı (8x A100 80GB)
| Yapılandırma | Süre/Adım | Verim (Throughput) | GPU Kullanımı | VRAM Kullanımı |
| :--- | :---: | :---: | :---: | :---: |
| **Temel (Baseline)** | 2.0 sn | 64 tok/sn | %47 | 38 GB |
| **v1.0 (Build 30 V2) (Optimize)** | **~1.2 sn** (Tahmini) | **~107 tok/sn** (Tahmini) | **~%95** (Hedef) | **~76 GB** (Hedef) |
| **Hızlanma (Öngörü)** | **+%67** | **+%67** | **+%102** | **+%100** |
**Toplam Throughput Hedefi (Projeksiyon): 11.000 tok/sn seviyesine kadar.**
Bu değer, tanımlı dağıtım profilindeki toplam sistem kapasitesi için yol haritası hedefidir; **tek cihazda ölçülmüş benchmark sonucu değildir**.
Operasyonel anlamı: daha yüksek eşzamanlı oturum kapasitesi, yük altında daha düşük birim inference maliyeti ve çok kullanıcılı senaryolarda daha kısa kuyruk süreleri.
*Not: Performans metrikleri mimari simülasyonlara dayanan eğitim öncesi tahminlerdir. BitNet 1.58 çıkarımı opsiyonel düşük-bit kernel yolu içerir; Tensor Core yolu **deneysel** ve opt-in’dir (`MERTFORMER_TENSORCORE=1`). Enerji/TOPS kazanımları gerçek cihaz ölçümü gerektirir. Kernel eleştirisi sadece inference için geçerlidir; BitNet eğitim katmanı mevcut, low-bit inference ise açıkça yol haritası maddesidir. **Eğitim hâlâ standart PyTorch matmul yollarıyla yürür; düşük-bit kernel eğitimi hızlandırmaz.** Ayrıca, **Residual Scaling Etkisi** ile 18 katman boyunca sinyal kararlılığı, 1/√2 (1/sqrt(2)) katsayısı ile korunarak en derin katmanda bile gradyan akışının stabil kalması hedeflenmektedir.*

### Bellek Ayak İzi
| Bileşen | FP32 | BF16 | BitNet 1.58 |
| :--- | :---: | :---: | :---: |
| Ağırlıklar | 10.4 GB | 5.2 GB | **~0.65 GB (tahmini)** |
| Optimizasyoncu (AdamW) | 41.6 GB | 20.8 GB | **20.8 GB** (dağıtık) |
| Aktivasyonlar | 40 GB | 20 GB | **12 GB** (checkpointing ile) |
| **Toplam (GPU başına)** | 92 GB | 46 GB | **33.45 GB** |
| **Toplam (8 GPUs)** | 736 GB | 368 GB | **267.6 GB** |
*Not: Bu tabloda yer alan değerler mimari karşılaştırmalar ve benzer modellerden elde edilen öngörülere dayanmaktadır.*

### Çıkarım (Inference) (Samsung S25 - Tahmini)
- ⏱️ **Gecikme (Latency)**: ~50ms/token (NPU optimize)
- 💾 **Bellek**: <2GB RAM
- 🔋 **Güç**: <3W (cihaz içi)
- 🏎️ **Verim**: ~45+ token/sn (NPU Kernel Optimizasyonu ile 100+ hedeflenmektedir)
*Not: Samsung S25 ve Snapdragon 8 Elite NPU değerleri, üretici yol haritaları ve benzer NPU mimarileri temel alınarak yapılan teorik çıkarım sonuçlarıdır. 1.58-bit mimarisi sayesinde bant genişliği darboğazı aşıldığı için çok daha yüksek hızlar mümkündür.*

### 🔄 Evrensel Uyumluluk & Sistem Gereksinimleri
BitNet mimarisi sayesinde MertFormer, sadece amiral gemilerinde değil, **neredeyse her cihazda** çalışabilir:

| Cihaz Sınıfı | Örnek Donanım | Beklenen Performans | Çalışma Modu |
| :--- | :--- | :---: | :---: |
| **Tier 1 (Hedef)** | S25, iPhone 17, 8 Elite | **~100 tok/sn** | **NPU / Neural Engine** |
| **Tier 2 (Modern)** | S23/S24, Pixel 8, iPhone 14 | **~40-60 tok/sn** | GPU / DSP |
| **Tier 3 (Giriş)** | Galaxy A54, A34 | **~15-25 tok/sn** | CPU (Optimize) |
| **Tier 4 (Legacy)** | Samsung M51 (Snapdragon 730G) | **~12 tok/sn** | CPU (BitNet) |

**Minimum Gereksinimler:**
- **RAM**: 2GB (Hedef; ~0.65GB VRAM tahmini)
- **Depolama**: 2GB boş alan
- **OS**: Android 10+ / iOS 15+ / Windows / macOS / Linux


---

<a id="hızlı-başlangıç"></a>
## 🚀 Hızlı Başlangıç

### Baseline (Review-Ready)
- Python **3.11** (bkz: `repro/python_TR.md`)
- Varsayılan çalışma şekli offline-first (`TITAN_OFFLINE=1`): HF/WandB login veya dataset download ancak açıkça etkinleştirilirse.

### Kurulum (Önerilen)
Python 3.11 ile `.titan-venv` oluşturur/günceller ve deps + dev tooling kurar:

```bash
bash scripts/bootstrap_venv.sh
```

Opsiyonel demo bağımlılıkları (pygame):

```bash
bash scripts/bootstrap_venv.sh --demo
```

### Doğrula (Offline-First, Tek Komut)

```bash
bash scripts/verify_all.sh
```

### BitNet Kernel Benchmark (Standalone, Tek Dosya)

```bash
python3 scripts/bitnet_kernel_benchmark_standalone.py --shapes 2048x2048x2048,4096x2048x2048
```

Bu script bilerek tek dosyalıdır: kernel kodu, quantization yolu, referans yol ve benchmark akışı aynı `.py` içinde bulunur.
Ayrıca Jupyter/Colab uyumludur: çalışma zamanının eklediği argümanlar (`-f kernel.json` gibi) otomatik yoksayılır.
CLI argümanı olmadan varsayılan çalıştırmak için:

```python
from scripts.bitnet_kernel_benchmark_standalone import run_default
run_default()
```

Performans notu: bu benchmark tek bir seçili cihazda çalışır ve çoklu GPU'ları (örneğin T4 x2) birleştirerek ölçmez.

SDK düzeyi doğrulama ve pilot raporlama:

```bash
mertformer verify
mertformer pilot-report --out reports/pilot_report.json
```

<a id="sdk-api-tr-quickstart"></a>
### 🧪 SDK API Hızlı Başlangıç (Geliştirici)

```python
from mertformer_sdk.api import load_model, generate, benchmark

#- Üretim/pilot akışında eğitimli checkpoint kullanın.
model, tokenizer, device = load_model(
    ckpt="checkpoints/my_trained.pt",
    strict_checkpoint=True,
)

text = generate(
    model,
    tokenizer,
    prompt="Doğrulama kapılarını tek paragrafta özetle.",
    max_new_tokens=96,
)
print(text)

results = benchmark(
    ckpt="checkpoints/my_trained.pt",
    out_dir="reports/benchmarks",
    samples=5,
    strict_checkpoint=True,
)
print(results)
```

```python
from mertformer_sdk.pilot import run_verify_all, build_pilot_report, write_pilot_report

verify_summary = run_verify_all(offline=True)
report = build_pilot_report(verify_summary=verify_summary)
write_pilot_report("reports/pilot_report_v1.json", report)
```

`strict_checkpoint=False` yalnızca kontrollü random-weight teşhis akışlarında kullanılmalıdır.

### LIVE DEMO (Snake Autoplayer)

```bash
bash scripts/bootstrap_venv.sh --demo
.titan-venv/bin/python snake_demo.py
```

### Drone SITL Kanıt Demosu (Offline, Fiziksel Drone Gerekmez)

```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --runs 3 --steps 120
bash run.sh --sitl-demo
```

Çıktılar `reports/pilots/<pilot_id>/sitl_<timestamp>/` klasörüne yazılır.
Varsayılan politika motoru `mertformer_liquidrouter` (BitLinear + LiquidRouter aksiyon önerisi, fail-safe override aktif).

Baseline karşılaştırması:

```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --policy-engine baseline
```

### Clean-Room Doğrulama (Temiz Clone + Yeni Venv)

```bash
bash run.sh --cleanroom-verify
```

### Sadece Preflight

```bash
TITAN_OFFLINE=1 bash run.sh --test
```

### Full Preflight Log (Güncel Snapshot, 2026-03-07)

```text
2026-03-07 21:25:01,761 - [INFO] - ✈️ ============================================================
2026-03-07 21:25:01,761 - [INFO] - ✈️ 🚀 MERTFORMER TITAN - ULTIMATE PREFLIGHT JUDGE 🚀
2026-03-07 21:25:01,761 - [INFO] - ✈️ ============================================================
2026-03-07 21:25:01,761 - [WARNING] - ⚠️ .env file not found, skipping local load.
2026-03-07 21:25:01,761 - [INFO] - ✈️ STEP 1: SECRET SCAN...
2026-03-07 21:25:01,761 - [WARNING] - ⚠️ HF_TOKEN missing (offline mode): OK (online checks will be skipped).
2026-03-07 21:25:01,761 - [WARNING] - ⚠️ WANDB_API_KEY missing (offline mode): OK (WandB checks disabled).
2026-03-07 21:25:01,761 - [INFO] - ✅ Secrets check completed.
2026-03-07 21:25:01,761 - [INFO] - ✈️ STEP 2: ARCHITECTURAL AUDIT...
2026-03-07 21:25:01,761 - [INFO] - ✅ Layer configuration validated: No Liquid/MoE conflicts.
2026-03-07 21:25:01,761 - [INFO] - ✅ MLA Dimensions: Consistent (2048 features).
2026-03-07 21:25:01,761 - [INFO] - ✅ BitNet b1.58 logic: ACTIVE (Locked).
2026-03-07 21:25:01,761 - [INFO] - ✈️ STEP 3: DATA & DISTILLATION TEST...
2026-03-07 21:25:02,483 - [INFO] - ✈️ Offline mode: skipping Hugging Face connectivity checks.
2026-03-07 21:25:02,483 - [INFO] - 🛡️ Teacher Model mocked (Prevented 140GB download).
2026-03-07 21:25:02,483 - [INFO] - ⚙️  Pre-computing logits for preflight...
2026-03-07 21:25:02,722 - [INFO] - ✅ Saved Final Chunk 0: <REPO_ROOT>/temp_preflight_logits/preflight_test_part_0.pt
2026-03-07 21:25:02,723 - [INFO] - ✅ Distillation pipeline: PROVEN (Logits generated/saved).
2026-03-07 21:25:02,723 - [INFO] - ✈️ STEP 4: MOE GURU LEARNING TEST...
2026-03-07 21:25:02,724 - [INFO] - ✈️ 🏗️  CONFIG: Using 'Mini-Titan' (2 Layers, 256 Hidden, forced MoE/Liquid) for RAM safety.
2026-03-07 21:25:02,909 - [INFO] - ✈️ Checking Architectural Gradient Health...
2026-03-07 21:25:02,915 - [INFO] - ✅ MoE Learning: PROVEN (48 expert params receiving gradients).
2026-03-07 21:25:02,915 - [INFO] - ✅ Liquid Dynamics: PROVEN (7 liquid params receiving gradients).
2026-03-07 21:25:02,915 - [INFO] - ✈️ Shared Expert Grad: OK
2026-03-07 21:25:02,915 - [INFO] - ✅ MertFormer forward/backward pass verified.
2026-03-07 21:25:02,916 - [INFO] - ✅ OVERALL SYSTEM STATUS: 100% PROTECTED & READY.
2026-03-07 21:25:02,916 - [INFO] - ✈️ CLEANUP: Removing temporary files...
2026-03-07 21:25:02,916 - [INFO] - ✈️ Removed <REPO_ROOT>/temp_preflight_data
2026-03-07 21:25:02,940 - [INFO] - ✈️ Removed <REPO_ROOT>/temp_preflight_logits
2026-03-07 21:25:02,940 - [INFO] - ✅ CLEANUP: Done.
2026-03-07 21:25:02,940 - [INFO] - ✈️ Preflight Duration: 1.18s
2026-03-07 21:25:02,940 - [INFO] - ✈️ ============================================================
2026-03-07 21:25:02,940 - [INFO] - ✈️ RESULT: 🏆 ALL GREEN
2026-03-07 21:25:02,940 - [INFO] - ✈️ Full Report: <REPO_ROOT>/logs/preflight/titan_preflight.log
2026-03-07 21:25:02,940 - [INFO] - ✈️ ============================================================
```

Historical snapshot (2026-02-10) arşiv kanıtı olarak git geçmişinde korunur.


### Eğitim (Online / Eğitim Donanımı)

```bash
#- Online modu açıkça etkinleştir + (opsiyonel) WandB + kurulum
TITAN_OFFLINE=0 TITAN_WANDB=1 TITAN_INSTALL=1 bash run.sh
```

Notlar:
- Online mod `HF_TOKEN` gerektirir. WandB opsiyoneldir (`TITAN_WANDB=0`).
- Bağımlılık kurulumu `TITAN_INSTALL=1` ile opt-in. Deterministik kurulum için bootstrap önerilir.

### Operator Modu Gate
Tek girişli güvenlik ve hazır olma süiti (varsayılan güvenli mod):

```bash
TITAN_OFFLINE=1 .titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl
#- Eğitim donanımında tam mod için --full kullanın
```

### Operator Modu Kontrol Listesi (Kanıt Dosyaları)
Aşağıdaki maddeler uygulanmıştır ve kanıt dosyaları ile eşlenmiştir:

- Phase -1: Safety & Failure Budget
- Auto-Kill NaN Injection: `scripts/nan_kill_test.py`
- Failure Budget (Pivot/Debug tetikleyici): `orchestrator/failure_budget.py`
- Checkpoint Restore Drill: `scripts/checkpoint_restore_drill.py`
- Phase 0: Infrastructure & Reality Gates
- Reproducibility Stamp (git/config/seed/datasets): `scripts/operator_mode_gate.py`, `utils/logger.py`
- Overfit Gate (1MB): `scripts/overfit_gate.py` (güvenli mod çalıştı; tam 1MB gate için eğitim donanımında `--full`)
- Observability (grad norms/router entropy/VRAM): `orchestrator/telemetry.py`
- Golden Sample Eval (50 prompt): `datasets/golden_samples.jsonl`, `scripts/golden_eval.py`
- Phase 1: Telemetry-Driven Execution
- Expected vs Actual altyapısı: `orchestrator/telemetry.py`
- Master Training (2.64B tasarım hedefi): eğitim donanımında çalıştırılacak (yerelde koşulmadı)
- Internal Truth Benchmarks (HumanEval/MBPP): `scripts/benchmarks_internal.py`
- Phase 2: Asset Stack
- Snake kanıt videosu üretimi: `.titan-venv/bin/python snake_demo.py --headless --record assets/snake_demo_proof.mp4 --record-seconds 30`
- One-Pager / Technical Snapshot: `reports/one_pager.md`, `reports/technical_snapshot.md`, `PITCH.md`
- Founders Hub Başvuru Taslağı: `reports/founders_hub_application.md`
- Phase 3: Future Horizons
- White Paper: `WHITE_PAPER_LIQUIDROUTER.md`
- Verification Plan
- Sanity Drills: `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`

Operator-mode artifact'leri:
- `logs/operator_mode/` altında üretilir (varsayılan gitignored; commit edilmez).
- Script stdout'a JSON özet basar; inceleme eki olarak onu kullanın.

### 🛡️ Tanısal Mükemmellik (Uçuş Öncesi Kontrol)
Preflight (offline-first):

```bash
TITAN_OFFLINE=1 .titan-venv/bin/python scripts/titan_preflight.py
#- veya:
TITAN_OFFLINE=1 bash run.sh --test
```

Neleri doğrular:
- Secrets kontrolü (token parçası yazdırmaz; offline modda secrets yoksa `TITAN_PREFLIGHT_REQUIRE_SECRETS=1` haricinde FAIL etmez)
- Mimari audit (cfg + MLA boyutları)
- Distillation dry-run (teacher mock; geçici logits; cleanup)
- MoE/Liquid gradient sağlığı

Artifact:
- `logs/preflight/titan_preflight.log` (gitignored; üretilen çıktı)

---

### 🧾 Birleşik Logbook
Tüm loglar tek bir artifact dosyasında birleştirilebilir: `logs/ALL_LOGS.jsonl` (gitignored).

Oluşturma/ekleme:

```bash
.titan-venv/bin/python scripts/logbook_build.py --append
```

Bu dosya her log satırı için kaynak metadata içerir ve denetim‑seviyesi izlenebilirlik sağlar.

---

### 💻 İnteraktif Terminal Simülasyonu
Aşağıdaki blok, bir MertFormer Ajanının karmaşık bir hatayı nasıl analiz edip çözdüğünü temsil eder:

```bash
[TITAN-ORCHESTRATOR] ⚡ Ajan 'Architect' yetkilendirildi...
[ARCHITECT] 🔍 Analiz ediliyor: MLA Layer-4 boyut uyuşmazlığı.
[ARCHITECT] 💡 Sebep tespit edildi: GQA Repetition faktörü Mini-Titan konfigürasyonunda hatalı.
[TITAN-SEC] 🛡️ Güvenlik Denetimi: Kod değişikliği güvenli. İmza: 0x88AF
[ARCHITECT] 🛠️  Yama uygulandı: cfg.num_kv_heads = 2
[TITAN-ORCHESTRATOR] ✅ Hata giderildi. Preflight Durumu: 🏆 ALL GREEN
```

---

<a id="sorun-giderme"></a>
## 🛠️ Sorun Giderme

| Belirti | Muhtemel Neden | Aksiyon |
| :--- | :--- | :--- |
| `run.sh` eğitimi başlatmıyor | `TITAN_OFFLINE=1` (offline-first varsayılanı) | `TITAN_OFFLINE=0` ile ve gerekli kimlik bilgileriyle çalıştırın. |
| `HF_TOKEN is missing` | Online mod token olmadan açık | Ortam veya `.env` içine `HF_TOKEN` ekleyin ya da offline moda dönün. |
| `WANDB_API_KEY missing` uyarısı | `TITAN_WANDB=1` ama anahtar yok | `WANDB_API_KEY` tanımlayın veya `TITAN_WANDB=0` yapın. |
| `Checkpoint not found` | `strict_checkpoint=True` ve dosya yok | Geçerli checkpoint yolu verin; `strict_checkpoint=False` sadece kontrollü teşhis içindir. |
| `verify_all.sh` başarısız | Bir veya daha fazla kalite kapısı kırık | `python3 -m pytest -q`, `ruff check`, `bash scripts/verify_all.sh` sırasıyla tekrar çalıştırın; `logs/preflight/titan_preflight.log` inceleyin. |
| Low-bit/Tensor Core davranışı beklenenden farklı | Deneysel kernel yolu açık | `MERTFORMER_LOWBIT_KERNEL=1` ve `MERTFORMER_TENSORCORE=1` yollarını opt-in deneysel olarak kullanın; üretim kapılarında baseline yolu koruyun. |

---

<a id="eğitim"></a>
## 🎓 Eğitim

### Eğitim Yapılandırması

**Dosya**: [`config/config.py`](config/config.py)

Temel hiperparametreler:
```python
#- Model Mimarisi
hidden_size = 2048
num_layers = 18
num_heads = 16
intermediate_size = 5632

#- Eğitim
learning_rate = 1.5e-3
max_steps = 45000
warmup_steps = 3000
batch_size = 128  # Global (GPU başına otomatik yapılandırılır)
grad_clip = 2.0

#- Damıtma (Distillation)
teacher_model = "meta-llama/Llama-3.3-70B-Instruct"
distill_alpha = 0.8  # Dinamik (0.8 → 0.15)
teacher_temp = 1.0

#- Optimizasyonlar
use_torch_compile = False
torch_compile_mode = "max-autotune"
use_gradient_checkpointing = True

#- Güvenlik
early_stop_patience = 5
liquid_warmup_steps = 10000
liquid_spike_threshold = 5.0
```

### Ortam Değişkenleri (Operasyonel Kontroller)

| Değişken | Varsayılan | Kapsam | Amaç |
| :--- | :--- | :--- | :--- |
| `TITAN_OFFLINE` | `1` | `run.sh` / çalışma zamanı | Offline-first modu açar; `0` yapılmadıkça online bağımlı adımları bloklar. |
| `TITAN_WANDB` | Otomatik (`0` offline, `1` online) | `run.sh` / izleme | Moda göre WandB akışını açar/kapatır. |
| `TITAN_INSTALL` | `0` | `run.sh` | Yalnızca `1` verildiğinde bağımlılık kurulumu yapar. |
| `TITAN_PYTHON` | unset | launcher | Özel Python yorumlayıcısı yolunu zorlar. |
| `TITAN_BOOTSTRAP` | `1` | launcher | Yerel venv yoksa `.titan-venv` otomatik bootstrap eder. |
| `MERTFORMER_TENSORCORE` | unset/`0` | kernel yolu | Deneysel Tensor Core low-bit yolunu açar (opt-in). |
| `MERTFORMER_LOWBIT_KERNEL` | unset/`0` | kernel yolu | Deneysel low-bit inference kernel yolunu açar (opt-in). |
| `HF_TOKEN` | unset | online operasyon | Kimlik doğrulamalı online dataset/model adımları için gerekir. |
| `WANDB_API_KEY` | unset | izleme | Sadece online modda WandB açıkken gerekir. |

### Müfredatla Öğrenme (4 Çekirdek Aşama + 1 Araç/API Fazı)

| Aşama | Adımlar | Odak | Veri Seti Boyutu |
| :--- | :---: | :--- | :--- |
| **1. Mantık & Muhakeme** | %0-42 | Matematik, kodlama, mantık | Corpus'un %42'si |
| **2. Dünya Bilgisi** | %42-72 | Gerçekler, tarih, bilim | Corpus'un %30'u |
| **3. Dil (TR)** | %72-80 | Gramer, akıcılık, kültür | Corpus'un %8'i |
| **4. Ruh (Kimlik)** | %80-88 | Kişilik, talimat takibi | Corpus'un %8'i |
| **5. Araç Kullanımı** | %88-100 | Fonksiyon çağırma, API | Corpus'un %12'si |

**Toplam Token**: ~23.6 Milyar (yüksek kaliteli, KD odaklı)
*Not: Damıtma, token başına etkin öğrenmeyi artırır; ancak ham token sayısını artırmaz.*

Bu eğitim sırası ve token bütçesi, **güçlü bir temel için yeterli** olacak şekilde tasarlanmıştır.
**Niş veya özel alanlar** için, en yüksek uzmanlık seviyesine çıkmak amaçıyla **hedefli fine‑tune** önerilir.

<a id="egitim-stratejisi-baseline-v28"></a>
### Eğitim Stratejisi (Baseline -> v28, Claim-Safe)

Baseline eğitim öncesinde acil bir mimari değişiklik zorunlu değildir.
Ancak ilk tuning turunda aşağıdaki maddeler kalite çarpanı olarak ele alınmalıdır.

**Önerilen v28 tuning maddeleri (baseline kanıtından sonra):**
1. Kimlik kayması gözlenirse Stage 4 (`Ruh/Kimlik`) etkisini artırma (oran yükseltme veya kontrollü oversample).
2. Modelin rol/sınır/misyon tonunu güçlendirmek için modele özel self-identity veri seti ekleme.
3. Baseline SFT korunurken DPO/RLHF hattını post-SFT alignment fazına alma.
4. Yakınsama under-training sinyali verirse efektif token bütçesini artırma (samples ve/veya epoch).
5. Stage 5 içine küçük ölçekli custom tool/orchestrator örnekleri enjekte etme.

**Operasyon sırası:**
1. Baseline eğitimi değiştirmeden başlat:
   `cd \"$(git rev-parse --show-toplevel)\" && TITAN_OFFLINE=0 TITAN_INSTALL=1 bash run.sh`
2. İlk checkpoint + ilk benchmark kanıtını üret (referans baseline).
3. v28 tuning paketini tek kontrollü turda uygula.
4. Baseline vs v28 A/B karşılaştırması ile ölçülen kazananı ana hat yap.

Yukarıdaki maddeler claim-safe'tir; ölçüm/kanıt üretilmeden kesin performans iddiası sayılmaz.

### İzleme

Eğitim metrikleri şuralara kaydedilir:
- 📈 **WandB**: Gerçek zamanlı panolar (loss, grad norm, MoE sağlığı vb.)
- 📄 **CSV**: `logs/run_*.csv` (artifact; gitignored)
- 📋 **JSONL**: `logs/run_*.jsonl` (artifact; gitignored)
- 🧾 **Birleşik logbook**: `logs/ALL_LOGS.jsonl` (artifact; gitignored)
- 💻 **Konsol**: Adım adım ilerleme

Not: Politika gereği `logs/` altında sadece **artifact** tutulur ve commit edilmez (`logs/README.md` hariç).

---

<a id="dağıtım-deployment"></a>
## 📱 Dağıtım (Deployment)

### ONNX Export

```bash
python scripts/mobile_export.py
```

Şunları oluşturur:
- `checkpoints/mertformer_titan_prod/titan_s25_fp32.onnx` (Dinamik eksenler)
- `checkpoints/mertformer_titan_prod/titan_s25_int8_quantized.onnx`
- Samsung S25 NPU için optimize edildi
- INT8 kuantizasyon hazır

### Çıkarım (Inference)

Dağıtım/çalışma zamanı doğrulaması için eğitimli checkpoint kullanın:

```python
from mertformer_sdk.api import load_model, generate

model, tokenizer, device = load_model(
    ckpt="checkpoints/my_trained.pt",
    strict_checkpoint=True,
)

response = generate(
    model,
    tokenizer,
    prompt="Hayatın anlamı nedir?",
    max_new_tokens=256,
    temperature=0.7,
)
print(response)
```

**Bağlam limitleri**: varsayılan giriş limiti **4096 token** (`cfg.max_seq_len`). Çıktı uzunluğu çağıran tarafından belirlenir; `scripts/chat.py` varsayılanı `--max_tokens=128`, `scripts/benchmarks_internal.py` varsayılanı `--max-new-tokens=256`.

---

<a id="entegrasyon-hedefleri"></a>
## 🔌 Entegrasyon Hedefleri

Bu bölüm, gerçekçi entegrasyon yollarını güncel durumlarıyla birlikte listeler.

### Mevcut (Repo İçinde Hazır Olan)

| Hedef | Entegrasyon Yöntemi | Durum | Ana Yollar |
| :--- | :--- | :---: | :--- |
| Yerel Offline Operasyon | CLI + gate çalıştırma (`verify`, `pilot-report`, `verify_all.sh`) | ✅ Mevcut | `mertformer_sdk/cli.py`, `scripts/verify_all.sh`, `scripts/operator_mode_gate.py` |
| Python Uygulamasına Gömme | SDK import + doğrudan API kullanımı | ✅ Mevcut | `mertformer_sdk/api.py`, `SDK_GUIDE_TR.md` |
| Edge Export Hattı | Edge/mobile dağıtım için ONNX export akışı | ✅ Mevcut | `scripts/mobile_export.py`, `mertformer_sdk/export.py`, `config/export/onnx_mobile.yaml` |
| Pilot Kanıt Teslimi | Rapor + log + şema tabanlı pilot paketi | ✅ Mevcut | `reports/pilots/`, `interfaces/pilot_report_v1.schema.json` |
| SITL Demo Akışı | Deterministik drone SITL kanıt protokolü | ✅ Mevcut (Demo) | `scripts/drone_sitl_demo.py`, `reports/drone_sitl_demo.md` |

### Planlanan / Opsiyonel (Tamamlandı İddiası Yok)

| Hedef | Kapsam | Durum | Not |
| :--- | :--- | :---: | :--- |
| Fine-tuning | Temel model sonrası alan uzmanlaştırma | 🟡 Planlı / Opsiyonel | Claim-ready kalite için gerçek alan verisi + compute + doğrulama koşuları gerekir. |
| Koordineli Çok-Etmenli Çalışma | Swarm/rol tabanlı orkestrasyon akışları | 🟡 Planlı / Hedef Mimari | `orchestrator/` altında kısmi/deneysel modüller var; üretim orkestrasyonu için ek doğrulama gerekir. |
| Genişletilmiş On-Prem Bağlayıcılar | Ortama özel kurumsal entegrasyon adaptörleri | 🟡 Opsiyonel | Yalnız müşteri entegrasyon sözleşmesi ve güvenlik politikasına göre uygulanmalıdır. |

---

<a id="kıyaslamalar-benchmarks"></a>
## 🏆 Kıyaslamalar (Benchmarks)
**Durum: Eğitim Öncesi Projeksiyon (İddia İçin Uygun Değil)**
*Not: Aşağıdaki metrikler hedef/tahmin değerleridir; gerçek benchmark iddiası için tam eğitim koşusu ve gerçek checkpoint ile ampirik doğrulama gerekir.*

### Benzer Modellerle Karşılaştırma

| Model | Parametreler | Kuantizasyon | Mobil-Hazır | Cihaz-İçi | Türkçe Desteği |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MertFormer Titan** | **2.64B** | **1.58-bit** | ✅ | ✅ | ✅ |
| Llama-3.2-3B | 3.0B | BF16 | ❌ | ❌ | Kısmi |
| Phi-3-mini | 3.8B | FP16 | ❌ | ❌ | ❌ |
| Gemma-2B | 2.0B | BF16 | ❌ | ❌ | ❌ |

### Performans Metrikleri (Tahmini)

| Görev | MertFormer Titan | Llama-3.2-3B | Phi-3-mini |
| :--- | :---: | :---: | :---: |
| **MMLU** | ~%55 | %63 | %69 |
| **HellaSwag** | ~%70 | %72 | %75 |
| **TruthfulQA** | ~%45 | %50 | %55 |
| **Türkçe NLU** | **~%65** | %45 | %30 |

*Not: Kıyaslamalar eğitim tamamlandıktan sonra güncellenecektir*

---

<a id="türkiye-vizyonu"></a>
## 🇹🇷 Türkiye Vizyonu & Milli Egemenlik

### Türkiye'nin Dijital Egemenliği İçin Neden Önemli?

MertFormer Titan, **Türkiye'nin dijital egemenliği** için kritik bir adımdır. Bugün dünyada AI, birkaç dev şirketin (OpenAI, Google, Meta) bulut sunucularında çalışıyor ve **tüm verileriniz onların elinde**.

**MertFormer Titan farkı:**
- ✅ **%100 Cihaz İçi**: Verileriniz telefonunuzdan çıkmaz
- ✅ **Türkçe Optimizasyonu**: Türk kültürü ve dili için özel eğitim
- ✅ **Milli Teknoloji**: Yerli geliştirme; lisans politikası `LICENSE` dosyasında
- ✅ **Bağımsızlık**: Bulut bağımlılığı yok, internet gereksiz

### Vizyon: Dijital Bağımsızlık

> **"Tohumu ektik, şimdi ormanı izleme vakti."**

MertFormer Titan, sadece bir AI modeli değil, **dijital bağımsızlık manifestosu**dur:

1. **Veri Egemenliği**: Türk vatandaşlarının verileri Türkiye'de kalır
2. **Teknoloji Bağımsızlığı**: Yabancı bulut servislerine bağımlılık sıfır
3. **Kültürel Koruma**: Türk dili ve kültürü AI'da temsil edilir
4. **Ekonomik Tasarruf**: Bulut maliyeti yok, cihazda çalışır

### Türkçe Corpus (Build 30 V2 Sonrası Yol Haritası)

Planlanan Türkçe veri kaynakları:
- **Vikipedi TR**: ~500K makale
- **Türk Haberleri**: Haber arşivleri
- **Edebiyat**: Türk edebiyatı klasikleri
- **Sosyal Medya**: Twitter/X Türkçe corpus (filtrelenmiş)
- **Devlet**: Resmi belgeler (açık kaynak)

**Hedef**: %30+ Türkçe performans artışı

---

<a id="sss"></a>
## ❓ SSS

### S: Neden 2.64B parametre? Daha büyük olabilir miydi?

**C**: 2.64B, **Build 30 V2 için mevcut tasarım hedefidir**:
- Samsung S25 (12GB RAM) rahatça çalıştırır
- BitNet ile ~0.65GB weights (tahmini)
- Hız/kalite dengesi mükemmel
- Daha büyük modeller (7B+) mobilde yavaş

### S: BitNet 1.58-bit kuantizasyon kaliteyi düşürür mü?

**C**: Kalite etkisi **göreve bağlıdır** ve tam benchmark ile doğrulanmalıdır:
- Bilgi Damıtma (Knowledge Distillation) ile telafi edilir
- Llama-3.3-70B teacher'dan öğrenir
- Üretimde kanıtlanmış (Microsoft Research)

### S: Flash Attention 2 neden sadece eğitimde?

**C**: **KV önbellek uyumsuzluğu**:
- Flash Attention 2, KV önbelleğini desteklemiyor (henüz)
- Çıkarımda (Inference) standart attention kullanılır
- Hız farkı minimal (çıkarım zaten hızlı)

### S: NCCL tuning ne işe yarar?

**C**: **Çoklu GPU iletişim optimizasyonu**:
- GPU'lar arası veri transferi hızlanır
- NVLink varsa P2P aktif olur
- 8x GPU'da %5-10 hızlanma

### S: Eğitim ne kadar sürer?

**C**: **8x A100 80GB için projeksiyon**dur (**benchmark iddiası değildir**):
- Temel: ~25 saat (45K adım × 2 sn/adım)
- v1.0 (Build 30 V2) Optimize: **tahmini** (45K adım; süre donanım/throughput'a bağlı, koşu sonrası netleşir)
- **10 saat tasarruf!**

### S: Samsung S25'te gerçekten çalışır mı?

**C**: **Evet, teorik olarak**:
- ONNX export hazır
- NPU optimizasyonu planlandı
- Gerçek cihaz testi: Build 30 V2 sonrası yol haritası
- Gerçek cihaz performans ölçümleri henüz tamamlanmadı

### S: Low-bit kernel production-ready mi?

**C**: **Deneysel referans kernel** (doğruluk öncelikli):
- BitNet eğitim yolu ayrı bir katman (mevcut)
- Low-bit inference yolu **opt-in**
- Tensor Core yolu **deneysel** (`MERTFORMER_TENSORCORE=1`)
- Gerçek profil/ölçüm olmadan hız/enerji iddiası yapılmaz

### S: Türkçe tokenizer var mı?

**C**: **Opt-in** (varsayılan kapalı):
- `use_tr_tokenizer=false` (default)
- `scripts/download_tr_tokenizer.py` ile indirilebilir
- Distillation uyumu için risk kontrollü POC önerilir

### S: Pilot-ready ile Claim-ready arasındaki fark nedir?

**C**:
- **Pilot-ready**: kontrollü demo/pilot için kapılar, güvenlik akışı ve operasyonel dokümantasyon hazırdır.
- **Claim-ready**: eğitilmiş checkpoint + tekrarlanabilir benchmark kanıtı + ölçüm logları gerektirir.

### S: Offline ve online modda hangi environment değişkenleri zorunludur?

**C**:
- **Offline mod (`TITAN_OFFLINE=1`)**: temel doğrulama akışı için harici kimlik bilgisi zorunlu değildir.
- **Online mod (`TITAN_OFFLINE=0`)**: doğrulamalı online veri/model işlemleri için `HF_TOKEN` zorunludur.
- `WANDB_API_KEY` yalnızca `TITAN_WANDB=1` ise zorunludur.

---

### SOP Çıktı Artefaktları

- `reports/one_command_full_sop_summary.md` — Uçtan uca tek komut SOP koşusunun tek belgede özet sonucu.
- `reports/one_command_full_sop.log` — Aynı koşunun ham tam yürütme logu.
- Not: her yeni full SOP koşusunda bu iki dosya güncellenir/üzerine yazılır.

### Son SOP Snapshot (Kaynak: `reports/one_command_full_sop_summary.md`)

- Kanonik kaynak: `reports/one_command_full_sop_summary.md` (her full SOP koşusunda yenilenir)
- Dahil edilen kontroller: `pytest`, `md_quality_all`, `linkcheck_all`, `unicode_path_guard`, `duplicate_zip_guard`, `clean_runtime_artifacts_check`, `zip_denylist_audit`, `secret_scan`
- Güncel paket hash değeri özet dosyada `release_zip_sha256` alanında tutulur
- Ham log: `reports/one_command_full_sop.log`


<a id="proje-yapısı"></a>
## 📂 Proje Yapısı

### Repo Kontrol Haritası

- `Çekirdek Sistem`: `config/`, `layers/`, `model/`, `train/`, `utils/`
- `SDK ve Çalışma Katmanı`: `mertformer_sdk/`, `scripts/`, `run.sh`
- `Veri ve Kanıt`: `datasets/`, `reports/`, `logs/`, `interfaces/`
- `Araştırma ve Uzantılar`: `ablations/`, `experiments/`, `orchestrator/`, `economics/`, `limits/`

### Kanonik Yerleşim (Build 30 V2)

```text
mertformer-titan-core/  # project root (git ls-files inventory)
├── .github/  # directory
│   ├── workflows/  # directory
│   │   └── ci.yml  # YAML configuration file
│   └── CODEOWNERS  # artifact
├── ablations/  # directory
│   ├── bitlinear_off/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── dense_only/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── no_liquid/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── no_moe/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── results.md  # documentation/report file
│   └── results_TR.md  # Turkish document counterpart
├── artifacts/  # directory
│   └── mertformer_release.zip.sha256  # artifact checksum
├── assets/  # directory
│   ├── sources/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── header.png  # media asset
│   ├── snake_demo_preview.gif  # media asset
│   ├── snake_demo_proof.mp4  # media asset
│   └── synaptic_map.png  # media asset
├── config/  # directory
│   ├── export/  # directory
│   │   └── onnx_mobile.yaml  # YAML configuration file
│   ├── model/  # directory
│   │   ├── mertformer_max_arch.yaml  # YAML configuration file
│   │   ├── mertformer_moe.yaml  # YAML configuration file
│   │   └── mertformer_small.yaml  # YAML configuration file
│   ├── train/  # directory
│   │   ├── finetune.yaml  # YAML configuration file
│   │   └── pretrain.yaml  # YAML configuration file
│   ├── __init__.py  # Python module/script (config package initializer and exports)
│   ├── base.yaml  # YAML configuration file
│   └── config.py  # Python module/script (runtime configuration model and validation helpers)
├── datasets/  # directory
│   ├── INTERNAL_POLICY.md  # documentation/report file
│   ├── INTERNAL_POLICY_TR.md  # Turkish document counterpart
│   ├── LICENSES.md  # documentation/report file
│   ├── LICENSES_TR.md  # Turkish document counterpart
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── SOURCES.md  # documentation/report file
│   ├── SOURCES_TR.md  # Turkish document counterpart
│   ├── filters.yaml  # YAML configuration file
│   ├── golden_assertions.jsonl  # JSONL data/log artifact
│   ├── golden_samples.jsonl  # JSONL data/log artifact
│   ├── hashes.json  # JSON data artifact
│   ├── inventory.json  # JSON data artifact
│   ├── inventory.md  # documentation/report file
│   ├── inventory_TR.md  # Turkish document counterpart
│   └── validation.jsonl  # JSONL data/log artifact
├── docs/  # directory
│   ├── CHAIN_MAP.md  # documentation/report file
│   └── CHAIN_MAP_TR.md  # Turkish document counterpart
├── economics/  # directory
│   ├── cost_model.md  # documentation/report file
│   ├── cost_model_TR.md  # Turkish document counterpart
│   ├── efficiency_report.md  # documentation/report file
│   ├── efficiency_report_TR.md  # Turkish document counterpart
│   └── flops_estimator.py  # Python module/script (module for flops estimator)
├── eval/  # directory
│   ├── agentic_suite.py  # Python module/script (evaluation routine for agentic suite)
│   ├── generalization_suite.py  # Python module/script (evaluation routine for generalization suite)
│   ├── golden.py  # Python module/script (evaluation routine for golden)
│   ├── gsm8k.py  # Python module/script (evaluation routine for gsm8k)
│   ├── humaneval.py  # Python module/script (evaluation routine for humaneval)
│   └── report_builder.py  # Python module/script (evaluation routine for report builder)
├── experiments/  # directory
│   └── exp_001_baseline/  # directory
│       ├── config.yaml  # YAML configuration file
│       ├── metrics.json  # JSON data artifact
│       ├── notes.md  # documentation/report file
│       └── notes_TR.md  # Turkish document counterpart
├── interfaces/  # directory
│   ├── closure_57_matrix_v1.schema.json  # JSON schema artifact
│   ├── inference_contract.md  # documentation/report file
│   ├── inference_contract_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.schema.json  # JSON schema artifact
│   ├── pilot_report_v1.schema.json  # JSON schema artifact
│   └── tokenizer_spec.json  # JSON data artifact
├── layers/  # directory
│   ├── __init__.py  # Python module/script (layers package initializer and exports)
│   ├── bitlinear.py  # Python module/script (BitLinear low-bit linear layer implementation)
│   ├── bitnet_patch.py  # Python module/script (BitNet quantization patch and runtime hooks)
│   ├── cognitive_extensions.py  # Python module/script (optional cognitive extension blocks)
│   ├── ffn.py  # Python module/script (feed-forward network blocks (dense and sparse paths))
│   ├── lifelong_safety.py  # Python module/script (lifelong safety guard layer)
│   ├── liquid.py  # Python module/script (liquid neural dynamics layers)
│   ├── mertformer_block.py  # Python module/script (core transformer block composition)
│   ├── mla.py  # Python module/script (multi-head latent attention implementation)
│   ├── moe.py  # Python module/script (mixture-of-experts routing and expert execution)
│   ├── qinn.py  # Python module/script (QINN experimental regulation layer (feature-flag))
│   └── world_model_head.py  # Python module/script (world-model auxiliary head)
├── limits/  # directory
│   ├── scaling_breakpoints.md  # documentation/report file
│   ├── scaling_breakpoints_TR.md  # Turkish document counterpart
│   └── stress_curves.png  # media asset
├── logs/  # directory
│   ├── README.md  # primary documentation (EN)
│   └── README_TR.md  # Turkish document counterpart
├── mertformer_sdk/  # directory
│   ├── kernels/  # directory
│   │   ├── cpp/  # directory
│   │   │   ├── __init__.py  # Python module/script (cpp package initializer and exports)
│   │   │   ├── bitnet_cpu.cpp  # C++ source file
│   │   │   └── loader.py  # Python module/script (SDK component for loader)
│   │   ├── metal/  # directory
│   │   │   ├── __init__.py  # Python module/script (metal package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── npu/  # directory
│   │   │   ├── __init__.py  # Python module/script (npu package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── vulkan/  # directory
│   │   │   ├── __init__.py  # Python module/script (vulkan package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── __init__.py  # Python module/script (kernels package initializer and exports)
│   │   ├── dispatcher.py  # Python module/script (SDK component for dispatcher)
│   │   ├── onnx_custom_op.py  # Python module/script (SDK component for onnx custom op)
│   │   └── triton_ternary.py  # Python module/script (SDK component for triton ternary)
│   ├── utils/  # directory
│   │   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   │   ├── bitpack.py  # Python module/script (SDK component for bitpack)
│   │   └── onnx_meta.py  # Python module/script (SDK component for onnx meta)
│   ├── __init__.py  # Python module/script (mertformer_sdk package initializer and exports)
│   ├── api.py  # Python module/script (SDK component for api)
│   ├── cli.py  # Python module/script (SDK component for cli)
│   ├── export.py  # Python module/script (SDK component for export)
│   ├── kpi.py  # Python module/script (SDK component for kpi)
│   └── pilot.py  # Python module/script (SDK component for pilot)
├── model/  # directory
│   ├── __init__.py  # Python module/script (model package initializer and exports)
│   └── transformers.py  # Python module/script (MertFormer model assembly and forward graph)
├── orchestrator/  # directory
│   ├── __init__.py  # Python module/script (orchestrator package initializer and exports)
│   ├── agent_registry.py  # Python module/script (orchestrator runtime component for agent registry)
│   ├── alignment_contracts.py  # Python module/script (orchestrator runtime component for alignment contracts)
│   ├── audio_sense.py  # Python module/script (orchestrator runtime component for audio sense)
│   ├── cognitive.py  # Python module/script (orchestrator runtime component for cognitive)
│   ├── cognitive_loop.py  # Python module/script (orchestrator runtime component for cognitive loop)
│   ├── compute_orchestrator.py  # Python module/script (orchestrator runtime component for compute orchestrator)
│   ├── core.py  # Python module/script (orchestrator runtime component for core)
│   ├── distillation_manager.py  # Python module/script (orchestrator runtime component for distillation manager)
│   ├── experience_store.py  # Python module/script (orchestrator runtime component for experience store)
│   ├── failure_budget.py  # Python module/script (orchestrator runtime component for failure budget)
│   ├── governance.py  # Python module/script (orchestrator runtime component for governance)
│   ├── hardware.py  # Python module/script (orchestrator runtime component for hardware)
│   ├── memory.py  # Python module/script (orchestrator runtime component for memory)
│   ├── paths.py  # Python module/script (orchestrator runtime component for paths)
│   ├── planner.py  # Python module/script (orchestrator runtime component for planner)
│   ├── reasoning_engine.py  # Python module/script (orchestrator runtime component for reasoning engine)
│   ├── self_audit.py  # Python module/script (orchestrator runtime component for self audit)
│   ├── self_improvement_guard.py  # Python module/script (orchestrator runtime component for self improvement guard)
│   ├── sense_engine.py  # Python module/script (orchestrator runtime component for sense engine)
│   ├── swarm_runtime.py  # Python module/script (orchestrator runtime component for swarm runtime)
│   ├── telemetry.py  # Python module/script (orchestrator runtime component for telemetry)
│   ├── tool_executor.py  # Python module/script (orchestrator runtime component for tool executor)
│   ├── tool_registry.py  # Python module/script (orchestrator runtime component for tool registry)
│   ├── verifier.py  # Python module/script (orchestrator runtime component for verifier)
│   └── web_sense.py  # Python module/script (orchestrator runtime component for web sense)
├── policy/  # directory
│   └── allow_deny_policy.yaml  # YAML configuration file
├── postmortems/  # directory
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── _template.md  # documentation/report file
│   ├── _template_TR.md  # Turkish document counterpart
│   ├── example_001.md  # documentation/report file
│   └── example_001_TR.md  # Turkish document counterpart
├── prompts/  # directory
│   ├── changelog.md  # documentation/report file
│   ├── changelog_TR.md  # Turkish document counterpart
│   └── system_v1.txt  # text artifact
├── registry/  # directory
│   └── mertformer_v0.1.json  # JSON data artifact
├── reports/  # directory
│   ├── benchmarks/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   ├── README_TR.md  # Turkish document counterpart
│   │   ├── agentic_suite_build30.json  # JSON data artifact
│   │   ├── generalization_suite_build30.json  # JSON data artifact
│   │   ├── internal_smoke_summary.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.csv  # CSV data artifact
│   │   ├── kaggle_compare_build30.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.md  # documentation/report file
│   │   ├── smoke_train_metrics.json  # JSON data artifact
│   │   └── summary.json  # JSON data artifact
│   ├── commercial_handover/  # directory
│   │   ├── contract_terms_checklist.md  # documentation/report file
│   │   ├── handover_scope.md  # documentation/report file
│   │   ├── known_issues.md  # documentation/report file
│   │   ├── ownership_and_role.md  # documentation/report file
│   │   └── sla_kpi_90_180.md  # documentation/report file
│   ├── pilots/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── snapshots/  # directory
│   │   └── 2026-02-24/  # directory
│   │       ├── claim_matrix_v2_2026-02-24.json  # JSON data artifact
│   │       ├── commercial_scenarios_v1_2026-02-24.json  # JSON data artifact
│   │       ├── evidence_snapshot_2026-02-24.json  # JSON data artifact
│   │       ├── mertformer_master_decision_report_TR_2026-02-24.md  # documentation/report file
│   │       ├── readiness_scorecard_v1_2026-02-24.json  # JSON data artifact
│   │       ├── report_interface_schema_v1.json  # JSON schema artifact
│   │       └── web_validation_sources_2026-02-24.md  # documentation/report file
│   ├── artifacts_zip_denylist_audit.json  # JSON data artifact
│   ├── asset_stack.md  # documentation/report file
│   ├── asset_stack_TR.md  # Turkish document counterpart
│   ├── backup_restore_report.json  # JSON data artifact
│   ├── bench_cpp_report.json  # JSON data artifact
│   ├── bench_metal_report.json  # JSON data artifact
│   ├── bench_npu_report.json  # JSON data artifact
│   ├── bench_vulkan_report.json  # JSON data artifact
│   ├── bench_zero_copy_report.json  # JSON data artifact
│   ├── cfc_moe_tolerance_report.json  # JSON data artifact
│   ├── cleanroom_verification.md  # documentation/report file
│   ├── cleanroom_verification_TR.md  # Turkish document counterpart
│   ├── cli_smoke_log.md  # documentation/report file
│   ├── cli_smoke_log_TR.md  # Turkish document counterpart
│   ├── closure_57_matrix.json  # JSON data artifact
│   ├── closure_57_matrix.md  # documentation/report file
│   ├── closure_57_matrix_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_DE.md  # documentation/report file
│   ├── codex_deep_audit_DE_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_EN.md  # documentation/report file
│   ├── codex_deep_audit_EN_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_TR.md  # Turkish document counterpart
│   ├── contamination_report_build30.md  # documentation/report file
│   ├── dataset_health.md  # documentation/report file
│   ├── dataset_health_TR.md  # Turkish document counterpart
│   ├── dealroom_reference.json  # JSON data artifact
│   ├── determinism_report.json  # JSON data artifact
│   ├── differential_backend_report.json  # JSON data artifact
│   ├── docs_dedup_canonical_list.md  # documentation/report file
│   ├── docs_packages_hash_manifest.json  # JSON data artifact
│   ├── drone_sitl_demo.md  # documentation/report file
│   ├── drone_sitl_demo_TR.md  # Turkish document counterpart
│   ├── duplicate_zip_guard_report.json  # JSON data artifact
│   ├── efficiency_convergence_analysis.md  # documentation/report file
│   ├── efficiency_convergence_analysis_TR.md  # Turkish document counterpart
│   ├── energy_baseline.json  # JSON data artifact
│   ├── execution_trace.json  # JSON data artifact
│   ├── fallback_policy_report.json  # JSON data artifact
│   ├── final_repo_audit.md  # documentation/report file
│   ├── final_sync_matrix.md  # documentation/report file
│   ├── final_sync_matrix_TR.md  # Turkish document counterpart
│   ├── folder_drift_report.json  # JSON data artifact
│   ├── folder_structure_policy.md  # documentation/report file
│   ├── founders_hub_application.md  # documentation/report file
│   ├── founders_hub_application_TR.md  # Turkish document counterpart
│   ├── github_policy_report.json  # JSON data artifact
│   ├── go_nogo_signoff_onepager.md  # documentation/report file
│   ├── go_nogo_signoff_onepager_TR.md  # Turkish document counterpart
│   ├── go_status_matrix.md  # documentation/report file
│   ├── go_status_matrix_TR.md  # Turkish document counterpart
│   ├── hardening_bundle_summary.json  # JSON data artifact
│   ├── investor_deck.pptx  # artifact
│   ├── investor_deck_TR.pptx  # artifact
│   ├── ip_licensing_split.md  # documentation/report file
│   ├── ip_licensing_split_TR.md  # Turkish document counterpart
│   ├── kernel_fuzz_report.json  # JSON data artifact
│   ├── kpi_contract_build30.md  # documentation/report file
│   ├── kpi_pack_v1.md  # documentation/report file
│   ├── kpi_pack_v1_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.json  # JSON data artifact
│   ├── latency_baseline.json  # JSON data artifact
│   ├── legal_cleanroom_signoff_internal.md  # documentation/report file
│   ├── license_gate_report.json  # JSON data artifact
│   ├── linkcheck_report.json  # JSON data artifact
│   ├── md_lint_report.json  # JSON data artifact
│   ├── model_health.md  # documentation/report file
│   ├── model_health_TR.md  # Turkish document counterpart
│   ├── one_command_full_sop.log  # text/log artifact (single-command end-to-end SOP raw log; overwritten each run)
│   ├── one_command_full_sop_summary.md  # documentation/report file (single-command end-to-end SOP summary; overwritten each run)
│   ├── one_pager.md  # documentation/report file
│   ├── one_pager_TR.md  # Turkish document counterpart
│   ├── ownership_proof_bundle.json  # JSON data artifact
│   ├── pilot_acceptance_signoff.md  # documentation/report file
│   ├── pilot_acceptance_signoff_TR.md  # Turkish document counterpart
│   ├── pilot_offer_packages.md  # documentation/report file
│   ├── pilot_offer_packages_TR.md  # Turkish document counterpart
│   ├── pilot_readiness_kit.md  # documentation/report file
│   ├── pilot_readiness_kit_TR.md  # Turkish document counterpart
│   ├── poc_protocol.md  # documentation/report file
│   ├── poc_protocol_TR.md  # Turkish document counterpart
│   ├── presentation_readiness_final.md  # documentation/report file
│   ├── proje_zip_rebuild_manifest_v2.json  # JSON data artifact
│   ├── proje_zip_rebuild_manifest_v2.md  # documentation/report file
│   ├── ram_guard_report.json  # JSON data artifact
│   ├── release_closure_lock_report.json  # JSON data artifact
│   ├── release_closure_note.md  # documentation/report file
│   ├── release_snapshot.md  # documentation/report file
│   ├── release_snapshot_TR.md  # Turkish document counterpart
│   ├── report_accuracy_audit.md  # documentation/report file
│   ├── report_accuracy_audit_TR.md  # Turkish document counterpart
│   ├── report_truth_matrix.md  # documentation/report file
│   ├── repro_build_report.json  # JSON data artifact
│   ├── resume_compat_report.json  # JSON data artifact
│   ├── review_checklist.md  # documentation/report file
│   ├── review_checklist_TR.md  # Turkish document counterpart
│   ├── runbook_validation_report.json  # JSON data artifact
│   ├── sales_funnel_90d.md  # documentation/report file
│   ├── sales_funnel_90d_TR.md  # Turkish document counterpart
│   ├── sanitizer_report.json  # JSON data artifact
│   ├── sbom.cdx.json  # JSON data artifact
│   ├── security_compliance.md  # documentation/report file
│   ├── security_compliance_TR.md  # Turkish document counterpart
│   ├── snapshot_manifest_dealroom.json  # JSON data artifact
│   ├── snapshot_manifest_main.json  # JSON data artifact
│   ├── start_gate_report.json  # JSON data artifact
│   ├── startup_selfcheck_report.json  # JSON data artifact
│   ├── static_analysis_report.json  # JSON data artifact
│   ├── strategic_value.md  # documentation/report file
│   ├── strategic_value_TR.md  # Turkish document counterpart
│   ├── system_hardware.md  # documentation/report file
│   ├── system_hardware_TR.md  # Turkish document counterpart
│   ├── system_stats.jsonl  # JSONL data/log artifact
│   ├── teacher_output_license_assessment.md  # documentation/report file
│   ├── technical_snapshot.md  # documentation/report file
│   ├── technical_snapshot_TR.md  # Turkish document counterpart
│   ├── thermal_baseline.json  # JSON data artifact
│   ├── training_readiness_manifest.json  # JSON data artifact
│   ├── unicode_path_guard_report.json  # JSON data artifact
│   ├── verified_matrix.md  # documentation/report file
│   ├── verified_matrix_TR.md  # Turkish document counterpart
│   ├── zip_audit_artifacts.json  # JSON data artifact
│   └── zip_audit_packages.json  # JSON data artifact
├── repro/  # directory
│   ├── accelerate_default.yaml  # YAML configuration file
│   ├── cuda.lock  # artifact
│   ├── env.lock  # artifact
│   ├── pip_freeze.txt  # text artifact
│   ├── python.md  # documentation/report file
│   ├── python_TR.md  # Turkish document counterpart
│   ├── seed_policy.md  # documentation/report file
│   └── seed_policy_TR.md  # Turkish document counterpart
├── scripts/  # directory
│   ├── reports/  # directory
│   │   ├── model_health.md  # documentation/report file
│   │   └── model_health_TR.md  # Turkish document counterpart
│   ├── runs/  # directory
│   │   └── preflight/  # directory
│   │       └── config_snapshot.json  # JSON data artifact
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── __init__.py  # Python module/script (scripts package initializer and exports)
│   ├── apply_github_policy.sh  # shell automation script
│   ├── benchmarks_internal.py  # Python module/script (automation script for benchmarks internal)
│   ├── bitnet_kernel_benchmark_standalone.py  # Python module/script (automation script for bitnet kernel benchmark standalone)
│   ├── bootstrap_venv.sh  # shell automation script
│   ├── build_artifacts_release_zip.sh  # shell automation script
│   ├── build_investor_deck.py  # Python module/script (automation script for build investor deck)
│   ├── build_summary_pdf.py  # Python module/script (automation script for build summary pdf)
│   ├── build_validation_set.py  # Python module/script (automation script for build validation set)
│   ├── cfc_moe_tolerance_check.py  # Python module/script (automation script for cfc moe tolerance check)
│   ├── chat.py  # Python module/script (automation script for chat)
│   ├── check_57_matrix.py  # Python module/script (automation script for check 57 matrix)
│   ├── check_doc_claim_consistency.py  # Python module/script (automation script for check doc claim consistency)
│   ├── check_tokenizer_sync.py  # Python module/script (automation script for check tokenizer sync)
│   ├── check_translation_pointer_policy.py  # Python module/script (automation script for check translation pointer policy)
│   ├── checkpoint_restore_drill.py  # Python module/script (automation script for checkpoint restore drill)
│   ├── clean_runtime_artifacts.sh  # shell automation script
│   ├── cleanroom_verify.sh  # shell automation script
│   ├── data_pipeline.py  # Python module/script (automation script for data pipeline)
│   ├── dealroom_sync.py  # Python module/script (automation script for dealroom sync)
│   ├── docs_inventory.py  # Python module/script (markdown inventory and folder policy reporter)
│   ├── download_tr_tokenizer.py  # Python module/script (automation script for download tr tokenizer)
│   ├── drone_sitl_demo.py  # Python module/script (automation script for drone sitl demo)
│   ├── duplicate_zip_guard.py  # Python module/script (automation script for duplicate zip guard)
│   ├── eval.py  # Python module/script (automation script for eval)
│   ├── extract_dataset_refs.py  # Python module/script (automation script for extract dataset refs)
│   ├── failure_budget_drill.py  # Python module/script (automation script for failure budget drill)
│   ├── final_one_shot.sh  # shell automation script
│   ├── generate_bench_reports.py  # Python module/script (automation script for generate bench reports)
│   ├── generate_energy_baselines.py  # Python module/script (automation script for generate energy baselines)
│   ├── generate_sbom.py  # Python module/script (automation script for generate sbom)
│   ├── golden_eval.py  # Python module/script (automation script for golden eval)
│   ├── golden_score.py  # Python module/script (automation script for golden score)
│   ├── hardening_bundle.py  # Python module/script (automation script for hardening bundle)
│   ├── hash_manifest_to_json.py  # Python module/script (automation script for hash manifest to json)
│   ├── kaggle_onefile_demo_build30.py  # Python module/script (automation script for kaggle onefile demo build30)
│   ├── kaggle_onefile_demo_build30_colab_math_fastproof.py  # Python module/script (automation script for kaggle onefile demo build30 colab math fastproof)
│   ├── kaggle_train_compare_build30.py  # Python module/script (automation script for kaggle train compare build30)
│   ├── linkcheck_gate.py  # Python module/script (automation script for linkcheck gate)
│   ├── logbook_build.py  # Python module/script (automation script for logbook build)
│   ├── mac_simulation.py  # Python module/script (automation script for mac simulation)
│   ├── md_build30_sweep.py  # Python module/script (automation script for md build30 sweep)
│   ├── md_integrity_check.py  # Python module/script (automation script for md integrity check)
│   ├── md_quality_gate.py  # Python module/script (automation script for md quality gate)
│   ├── mini_titan_poc.py  # Python module/script (automation script for mini titan poc)
│   ├── mobile_export.py  # Python module/script (automation script for mobile export)
│   ├── nan_kill_test.py  # Python module/script (automation script for nan kill test)
│   ├── one_command_full_sop.sh  # shell automation script
│   ├── operator_mode_gate.py  # Python module/script (automation script for operator mode gate)
│   ├── overfit_gate.py  # Python module/script (automation script for overfit gate)
│   ├── plot_training_log.py  # Python module/script (automation script for plot training log)
│   ├── ram_guard.py  # Python module/script (automation script for ram guard)
│   ├── record_dataset_hashes.py  # Python module/script (automation script for record dataset hashes)
│   ├── release_build30.sh  # shell automation script
│   ├── release_closure_lock.sh  # shell automation script
│   ├── repro_build_check.py  # Python module/script (automation script for repro build check)
│   ├── resume_compat_check.py  # Python module/script (automation script for resume compat check)
│   ├── run_and_clean_pycache.py  # Python module/script (run command + guaranteed post-run cache sweep; add --include-venv-caches for venv cache cleanup)
│   ├── scaling_audit_math.py  # Python module/script (automation script for scaling audit math)
│   ├── secret_scan.py  # Python module/script (automation script for secret scan)
│   ├── smart_runner.py  # Python module/script (automation script for smart runner)
│   ├── smoke_train_benchmark.py  # Python module/script (automation script for smoke train benchmark)
│   ├── start_gate.py  # Python module/script (automation script for start gate)
│   ├── sync_manifest.py  # Python module/script (release manifest and project-structure sync generator)
│   ├── test_onnx_export.py  # Python module/script (automation script for test onnx export)
│   ├── titan_onnx_stress_test.py  # Python module/script (automation script for titan onnx stress test)
│   ├── titan_preflight.py  # Python module/script (automation script for titan preflight)
│   ├── train_smoke.py  # Python module/script (automation script for train smoke)
│   ├── train_tpu_turbo.py  # Python module/script (automation script for train tpu turbo)
│   ├── unicode_path_guard.py  # Python module/script (automation script for unicode path guard)
│   ├── update_investor_deck.py  # Python module/script (automation script for update investor deck)
│   ├── update_system_hardware.py  # Python module/script (automation script for update system hardware)
│   ├── verify_all.sh  # shell automation script
│   ├── verify_datasets.py  # Python module/script (automation script for verify datasets)
│   ├── verify_onnx_local.py  # Python module/script (automation script for verify onnx local)
│   ├── version_checker.py  # Python module/script (automation script for version checker)
│   ├── write_cuda_lock.py  # Python module/script (automation script for write cuda lock)
│   ├── xray.py  # Python module/script (automation script for xray)
│   └── zip_denylist_audit.py  # Python module/script (automation script for zip denylist audit)
├── telemetry/  # directory
│   └── metrics_schema.json  # JSON schema artifact
├── tests/  # directory
│   ├── test_57_matrix_gate.py  # Python module/script (automated test module for 57 matrix gate)
│   ├── test_agi_cognitive.py  # Python module/script (automated test module for agi cognitive)
│   ├── test_architecture_integrity.py  # Python module/script (automated test module for architecture integrity)
│   ├── test_cognitive_extensions.py  # Python module/script (automated test module for cognitive extensions)
│   ├── test_comprehensive.py  # Python module/script (automated test module for comprehensive)
│   ├── test_continual_adapter.py  # Python module/script (automated test module for continual adapter)
│   ├── test_cpp_kernel_loader.py  # Python module/script (automated test module for cpp kernel loader)
│   ├── test_dispatcher_extended.py  # Python module/script (automated test module for dispatcher extended)
│   ├── test_drone_sitl_demo.py  # Python module/script (automated test module for drone sitl demo)
│   ├── test_eval_suites.py  # Python module/script (automated test module for eval suites)
│   ├── test_export_metadata.py  # Python module/script (automated test module for export metadata)
│   ├── test_kaggle_compare_script.py  # Python module/script (automated test module for kaggle compare script)
│   ├── test_kaggle_onefile_colab_math_fastproof.py  # Python module/script (automated test module for kaggle onefile colab math fastproof)
│   ├── test_kaggle_onefile_compile_guard.py  # Python module/script (automated test module for kaggle onefile compile guard)
│   ├── test_kaggle_onefile_config.py  # Python module/script (automated test module for kaggle onefile config)
│   ├── test_kaggle_onefile_feature_coverage.py  # Python module/script (automated test module for kaggle onefile feature coverage)
│   ├── test_kaggle_onefile_zero_shot_unseen.py  # Python module/script (automated test module for kaggle onefile zero shot unseen)
│   ├── test_kernel_dispatcher.py  # Python module/script (automated test module for kernel dispatcher)
│   ├── test_kernel_equivalence.py  # Python module/script (automated test module for kernel equivalence)
│   ├── test_kpi_report_cli.py  # Python module/script (automated test module for kpi report cli)
│   ├── test_lifelong_safety.py  # Python module/script (automated test module for lifelong safety)
│   ├── test_mla_regressions.py  # Python module/script (automated test module for mla regressions)
│   ├── test_model.py  # Python module/script (automated test module for model)
│   ├── test_onnx_custom_op_contract.py  # Python module/script (automated test module for onnx custom op contract)
│   ├── test_onnx_export_path.py  # Python module/script (automated test module for onnx export path)
│   ├── test_onnx_metadata_hook.py  # Python module/script (automated test module for onnx metadata hook)
│   ├── test_orchestrator_swarm_runtime.py  # Python module/script (automated test module for orchestrator swarm runtime)
│   ├── test_sdk_api.py  # Python module/script (automated test module for sdk api)
│   ├── test_sdk_pilot_cli.py  # Python module/script (automated test module for sdk pilot cli)
│   ├── test_train_loop_sanity.py  # Python module/script (automated test module for train loop sanity)
│   ├── test_triad_omega_api.py  # Python module/script (automated test module for triad omega api)
│   └── test_world_model_head.py  # Python module/script (automated test module for world model head)
├── tokenizer/  # directory
│   ├── tr/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── drift_report.md  # documentation/report file
│   ├── drift_report_TR.md  # Turkish document counterpart
│   ├── stats.md  # documentation/report file
│   ├── stats_TR.md  # Turkish document counterpart
│   └── tokenizer.json  # JSON data artifact
├── tools/  # directory
│   ├── contracts/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── sandbox/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── abuse_tests.md  # documentation/report file
│   └── abuse_tests_TR.md  # Turkish document counterpart
├── train/  # directory
│   ├── __init__.py  # Python module/script (train package initializer and exports)
│   ├── continual_adapter.py  # Python module/script (continual learning adapter path for training)
│   └── train.py  # Python module/script (main training loop entrypoint)
├── training_dynamics/  # directory
│   ├── cold_vs_warm.md  # documentation/report file
│   └── cold_vs_warm_TR.md  # Turkish document counterpart
├── utils/  # directory
│   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   ├── dataset_registry.py  # Python module/script (module for dataset registry)
│   ├── logger.py  # Python module/script (module for logger)
│   └── safety.py  # Python module/script (module for safety)
├── .gitignore  # git ignore policy
├── CHANGELOG.md  # documentation/report file
├── CHANGELOG_TR.md  # Turkish document counterpart
├── CITATION.cff  # citation metadata
├── CONTRIBUTING.md  # documentation/report file
├── CONTRIBUTING_TR.md  # Turkish document counterpart
├── DECISIONS.md  # documentation/report file
├── DECISIONS_TR.md  # Turkish document counterpart
├── Dockerfile  # container build baseline
├── IMPLEMENTATION_PLAN.md  # documentation/report file
├── IMPLEMENTATION_PLAN_TR.md  # Turkish document counterpart
├── INTERNAL_AGI_GAP.md  # documentation/report file
├── INTERNAL_AGI_GAP_TR.md  # Turkish document counterpart
├── LICENSE  # license terms (EN)
├── LICENSE_TR  # license terms (TR)
├── MODEL_CARD.md  # documentation/report file
├── MODEL_CARD_TR.md  # Turkish document counterpart
├── MODEL_LICENSE.md  # documentation/report file
├── MODEL_LICENSE_TR.md  # Turkish document counterpart
├── PITCH.md  # documentation/report file
├── PITCH_TR.md  # Turkish document counterpart
├── README.md  # primary documentation (EN)
├── README_CHECKLIST.md  # documentation/report file
├── README_CHECKLIST_TR.md  # Turkish document counterpart
├── README_SUMMARY.md  # documentation/report file
├── README_SUMMARY.pdf  # artifact
├── README_SUMMARY_TR.md  # Turkish document counterpart
├── README_SUMMARY_TR.pdf  # artifact
├── README_TR.md  # Turkish document counterpart
├── SDK_GUIDE.md  # documentation/report file
├── SDK_GUIDE_TR.md  # Turkish document counterpart
├── SECURITY.md  # documentation/report file
├── SECURITY_TR.md  # Turkish document counterpart
├── TASK.md  # documentation/report file
├── TASK_TR.md  # Turkish document counterpart
├── TECHNICAL_REPORT.md  # documentation/report file
├── TECHNICAL_REPORT_TR.md  # Turkish document counterpart
├── TRAINING_PLAN.md  # documentation/report file
├── TRAINING_PLAN_TR.md  # Turkish document counterpart
├── TROUBLESHOOTING.md  # documentation/report file
├── TROUBLESHOOTING_TR.md  # Turkish document counterpart
├── USAGE_GUIDE.md  # documentation/report file
├── USAGE_GUIDE_TR.md  # Turkish document counterpart
├── USE_POLICY.md  # documentation/report file
├── USE_POLICY_TR.md  # Turkish document counterpart
├── V2_BACKLOG_SEED.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER_TR.md  # Turkish document counterpart
├── pyproject.toml  # project metadata
├── requirements.txt  # text artifact
├── run.sh  # shell automation script
└── snake_demo.py  # Python module/script (module for snake demo)
```

### Tıklanabilir Yol Haritası

- `Çekirdek Sistem`: [config/](config/), [layers/](layers/), [model/](model/), [train/](train/), [utils/](utils/)
- `SDK ve Çalışma Katmanı`: [mertformer_sdk/](mertformer_sdk/), [scripts/](scripts/), [run.sh](run.sh)
- `Veri ve Kanıt`: [datasets/](datasets/), [reports/](reports/), [logs/](logs/), [interfaces/](interfaces/)
- `Araştırma ve Uzantılar`: [ablations/](ablations/), [experiments/](experiments/), [orchestrator/](orchestrator/), [economics/](economics/), [limits/](limits/)
- `Ana Dokümanlar`: [README.md](README.md), [README_TR.md](README_TR.md), [USAGE_GUIDE_TR.md](USAGE_GUIDE_TR.md), [SDK_GUIDE_TR.md](SDK_GUIDE_TR.md)

### Bakım Kuralı

- Gezinme için bu kontrol haritası esas alınır.
- Kanonik yerleşim, takipli dosyalardan (`git ls-files`) üretilir ve release kapanışında güncellenir.
- Nokta-zaman envanterleri `reports/final_sync_matrix.md` ve `reports/release_snapshot.md` içinde tutulur.
- README içindeki linklerde yalnızca mevcut yollar referans verilmelidir.

---

<a id="lisans"></a>
## 📄 Lisans

Bu proje **gizli ve tescillidir**. Tüm hakları **MertFormer AI Team** tarafından saklıdır. İzinsiz kopyalanması, değiştirilmesi veya dağıtılması kesinlikle yasaktır. Tüm detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkürler

- **Meta AI**: Llama-3.3-70B teacher modeli & tokenleştirici
- **Microsoft Research**: BitNet kuantizasyon araştırması
- **Liquid AI**: Liquid Neural Networks (CfC) ilhamı
- **Araştırma Konumlandırması**: MertFormer, MoE yönlendirmesine liquid dinamiklerini entegre ederek zamansal zekaya ortogonal bir yaklaşım geliştirir.
- **DeepSeek**: MLA literatürü için ilham (bu repoda mevcut implementasyon MLA etiketli GQA)
- **HazyResearch / Stanford (Tri Dao ve ekip)**: Flash Attention 2
- **PyTorch**: Temel eğitim ve çıkarım çerçevesi
- **Triton**: Deneysel düşük-bit kernel çalışmaları
- **ONNX / ONNX Runtime**: Export ve doğrulama araçları
- **SentencePiece**: Tokenization aracı
- **Weights & Biases (WandB)**: Deney takip sistemi
- **NVIDIA**: Apex optimizasyonları, NCCL
- **Hugging Face**: Transformers, Accelerate, Datasets kütüphaneleri
- **Türkçe Yapay Zeka Topluluğu**: Destek ve geri bildirim

---

<a id="stratejik-is-birligi"></a>
## 🤝 Stratejik İş Birliği

MertFormer, kontrollü ve kanıt-öncelikli bir iş birliği formatını benimser.

**Paylaşılabilir (kontrollü kapsam):**
- `reports/` altındaki mimari ve doğrulama dokümanları
- Pilot kanıt paketleri (`verify_all`, operator gate, `pilot_report_v1`)
- Entegrasyon gereksinimleri ve dağıtım kısıtları

**Açık hukuki kontrol olmadan paylaşılmayacaklar:**
- Ham kaynak kod dağıtım hakları
- Checkpoint/ağırlık dosyaları ve sır içeren artefaktlar
- Onay kapsamı dışındaki iç güvenlik prosedürleri

Tüm ticari/kurumsal etkileşimler, `LICENSE` ile uyumlu yazılı sözleşme ve gizlilik kontrolleri altında yürütülür.

---

<a id="iletişim"></a>
## 📧 İletişim

**Proje**: MertFormer Titan (Onyx Storm)
**Sürüm**: v1.0 (Build 30 V2, Eğitim Öncesi Baseline)
**Durum**: 🟡 Pilota Hazır (eğitim ve benchmark iddiaları beklemede)
**Türkiye'de geliştirildi**

---

## ✅ Satışa Hazır Checklist (B2B Pilot Modu)

Eğitim öncesi aşamada ücretli pilot için minimum kabul seti:

- `bash scripts/verify_all.sh` offline modda geçmelidir.
- Operator mode gate logları pilot teslimine eklenmelidir.
- `mertformer pilot-report --out <json>` çıktısı `pilot_report_v1` olarak teslim edilmelidir.
- Eğitimli checkpoint yoksa benchmark durumu `NOT ELIGIBLE FOR CLAIM` olarak kalmalıdır.
- Müşteri tarafında offline çalıştırma (`mertformer verify`) canlı gösterilmelidir.
- Ticari kapanış hedefi: 2 ücretli pilot sözleşmesi veya imzalı PoC niyet mektubu.

---

## 🛡️ Stratejik Şeffaflık ve Yol Haritası

### Nihai Kapsam ve Niyet
Bu proje, tasarım gereği proof-of-system seviyesinde tamamlanır. Amaç, gerçek dünya kısıtları altında çalışan bütüncül bir otonom akıl yürütme hattını göstermek; production-ready veya sertifikalı bir platform iddiası sunmak değildir. Mimari sınırlar, güvenlik davranışı, gerçek zaman kısıtları ve hata modları birincil mühendislik konusu olarak ele alınır. Büyük ölçekli dağıtım, sertifikasyon ve uzun süreli saha doğrulaması bu sürüm için bilinçli olarak kapsam dışıdır.

### ⚠️ Teknik Risk Faktörleri
*   **Performans Projeksiyonu**: Mobil NPU metrikleri (<50ms/token) şu an için mimari simülasyon bazlıdır ve eğitim sonrası fiziksel testlerle doğrulanacaktır.
*   **Donanım Uyumluluğu**: Mobilde 1.58-bit ternary yürütme, en yüksek hız için standart ONNX çalışma zamanlarının ötesinde özel kernel optimizasyonu gerektirebilir.
*   **MoE Kararlılığı**: `LiquidRouter` yeni bir araştırma katkısıdır; klasik router'lara göre kesin avantajı tam ölçekli eğitim sırasında benchmark edilecektir.

### 🗺️ Doğrulama Yol Haritası
- [x] **Faz 0**: Mimari Simülasyonu ve Matematiksel Doğrulama
- [ ] **Faz 1**: Eğitim Yakınsaması ve Distilasyon Sağlık Kontrolü
- [ ] **Faz 2**: Çok Alanlı Benchmark Testleri (GSM8K, HumanEval, MMLU)
- [ ] **Faz 3**: Fiziksel Cihaz Performans Ölçümü (S25/M4)
- [ ] **Faz 4**: Düşük Seviye Kernel Optimizasyonu (C++ / ENN / QNN, eğitim sonrası opsiyonel hat)

<a id="olceklenebilirlik-vizyonu"></a>
### 📈 Ölçeklenebilirlik Vizyonu (Claim-Safe)
Build 30 V2, bilinçli olarak **2.64B** doğrulama ve tekrar üretilebilir kanıt kapılarına odaklanır.
Gelecekteki **13B / 70B / 256B** araştırmaları koşullu bir hat olarak ele alınır ve yalnızca şu şartlardan sonra değerlendirilir:
- 2.64B için eğitimli checkpoint kanıtı,
- tekrar üretilebilir benchmark çıktıları,
- donanım/maliyet fizibilite incelemesi,
- güvenlik ve uyumluluk sınır kontrolleri.

### 🚫 MertFormer Titan Ne Değildir?
*   **Genel Bir Chatbot Değildir**: Özellikle kod orkestrasyonu ve yapısal mantık yürütme için optimize edilmiştir.
*   **Bulut-Ölçekli Altyapı Rakibi Değildir**: Devasa veri merkezleri üzerinden genel bulut hizmeti vermek yerine, özel ve yerel cihaz içi "uç" (edge) yürütme için optimize edilmiştir.
*   **Sıradan Bir Transformer Değildir**: CfC, MLA etiketli GQA attention ve BitNet katmanlarının standart dışı bir sentezidir.

---

## 📜 Atıf

```bibtex
@software{mertformer_titan_2026,
  author = {MertFormer AI Team},
  title = {MertFormer Titan: 1.58-bit Mobile-First LLM},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/latentcore/mertformer-titan-core}}
}
```

---
<div align="center">

**🚀 Cihaz İçi Yapay Zekanın Geleceği İçin İnşa Edildi 🚀**

*"En iyi yapay zeka, gizliliğinize saygı duyandır."*

**"Tohumu ektik, şimdi ormanı izleme vakti."**

</div>

## Build30 V1 Final Kapanis Eki (2026-02-28)

`scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py` one-file companion dosyasi kapanis sozlesmesi seviyesine yukseltilmistir:

- Payload semalari guncellendi: `build30_colab_math_fastproof_payload_v2`, `build30_colab_math_fastproof_compare_v2`, `kaggle_onefile_deep_build30_v6`.
- Kati run/config yonetisimi: schema dogrulamasi, zorunlu alan fail-fast, unknown-key reddi, override/source trace.
- Compile stall mitigasyonu: compile varsayilan OFF, timeout fallback, compile/cudagraph guard telemetry.
- Kanit genisletmeleri: ownership proof, runtime fingerprint, redacted env snapshot, reproduce komutu.
- Eval genisletmeleri: unseen-range zero-shot exact-match raporlama ve interpretability artefaktlari.
- Ozellik takibi: tum maddeleri kapsayan `feature_coverage_matrix` ve completeness yuzdesi.
