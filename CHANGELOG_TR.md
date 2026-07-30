# Değişiklik Kaydı

Bu dosya projedeki önemli değişiklikleri takip eder.

> **Bakım notu (2026-07-25'te eklendi):** bu dosya elle bakımı yapılan, otomatik yeniden üretilmeyen bir dosya — bu not var olmadan önce tam bir ay (2026-06-28 → 2026-07-25) bayat kaldı. Gerçek bir `BACKLOG.md`/`DECISIONS.md` girdisi bırakan her closure pass'i, burada (TR) ve `CHANGELOG.md`'de (EN) güncel `## Unreleased - <tarih>` bölümünü de eklemeli/güncellemeli — kısa bir özet yeterli, tam detay `BACKLOG.md`/`DECISIONS.md`'de kalır. Bkz. `reports/change_control_sop.md`.
> Girdiler sıkı ters-kronolojik sırada tutulur (en yeni en üstte); 2026-07-27 pass'i "Pass 7 (2026-06-13)"in 2026-03-13/2026-02-08 etiketli sürümlerden sonraya yanlış dosyalandığını buldu ve doğru kronolojik yerine taşıdı — ayrıntı için aşağıdaki girdiye bakın.

## Unreleased - 2026-07-31

### Düzeltilenler
- `tests/test_pre45k_gate.py::test_offline_preflight_reports_the_missing_corpus_rather_than_passing`: ambient `GITHUB_ACTIONS`/`CI`/`TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL` env değişkenlerini kendi subprocess çağrısına miras alıyordu; çıplak yerel checkout'ta geçiyor ama gerçek GitHub Actions CI'da (`bash scripts/verify_all.sh` üzerinden koşan) kırılıyordu. Artık kapıyı çağırmadan önce üçünü de `monkeypatch.delenv(...)` ile temizliyor; iki production escape hatch de (`scripts/titan_preflight.py`, `scripts/verify_all.sh`) değişmedi. Tam detay: `BACKLOG.md`.

### Doğrulama
- Test sayısı değişmedi (`726 passed, 5 skipped` yerelde, önceki girdiyle aynı) — bu, mevcut bir testin ortam izolasyonunu düzeltiyor, test eklemiyor/silmiyor. Doğrulandı: CI ambient kirliliği simüle edildiğinde (`GITHUB_ACTIONS=true CI=true TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL=1`) düzeltme-öncesi hata tekrarlanıyor, düzeltme-sonrası geçiyor.

## Unreleased - 2026-07-30

### Eklenenler
- `CODE_OF_CONDUCT.md`/`_TR` (Contributor Covenant 2.1, artı proje-özgü measured/target/vision iddia-disiplini maddesi) — `README.md` artık dış katkı davet ettiğine göre GitHub community-standards kontrol listesindeki son boşluğu kapatıyor.

### Düzeltilenler
- Audit wave 1-5 (bağımsız 2026-07-27 statik denetimi, 20+ gerçek bulgu): MoE capacity host-sync'leri kaldırıldı, `train/packing.py` resume-sayacı desenkronizasyonu düzeltildi, teacher-logit identity sidecar'ı eklendi, param tahmincileri ölçülen sayıyı birebir üretiyor, feature-flag'li drift dedektörleri erişilebilir kılındı, 2 ölü script silindi, artı structure-check/config-validator/alias-guard/PoC-hashing düzeltmeleri ve 45K dashboard bağlantısı. Tam pass-pass detay: `BACKLOG.md`.
- Lisans-başlığı çelişkileri: Apache 2.0 relicensing'den sonra 50 `*.py` dosyası ve `run.sh` hâlâ All-Rights-Reserved başlığı taşıyordu; düzeltildi, her tracked dosya tipinde tam-repo grep ile doğrulandı.
- `NOTICE`: relicensing'in yanlış bıraktığı iki cümle (bayat "proprietary" ifadesi, bayat takım adı) düzeltildi; Llama attribution katmanı dokunulmadı.
- `SECURITY.md`/`_TR`: birincil GitHub Security Advisories kanalının yanına açık bir contact-email fallback eklendi.
- `tests/test_pre45k_gate.py::test_run_offline_preflight_against_real_repo`: gitignore'lu eğitim korpusuna bağımlıydı ve her fresh clone/CI koşucusunda kırılıyordu; artık korpus yokken skip ediyor, dürüst-hata yönünü kilitleyen yeni bir kardeş test eklendi (o kardeş testin kendi takip-düzeltmesi için yukarıdaki 2026-07-31 girdisine bakın).

### Değişenler
- Kod, public release için Apache 2.0 altında relicense edildi; `README.md`/`README_TR.md`'ye Hiring ve Contribution bölümleri eklendi. Bkz. `DECISIONS.md`.

### Doğrulama
- `726 passed, 5 skipped`. `bash scripts/final_one_shot.sh` yeşildi (bkz. `BACKLOG.md` "Public yayın kapanışı"). Eğitim-matematiği, readiness veya iddia sınırı değişikliği yok.

## Unreleased - 2026-07-27

### Düzeltilenler
- `scripts/scaling_audit_math.py` + `config/config.py::_estimate_total_params()`: iki bağımsız analitik param-sayısı tahmincisi de MoE expert'leri için dense-FFN `intermediate_size`'ı kullanıyordu (gerçek, daha büyük `moe_intermediate` yerine) ve ikisi de `layers/moe.py`'nin her zaman aktif "shared expert"ini tamamen atlıyordu — aktif parametreleri ~%44, toplam parametreleri ~%8 eksik sayıyordu. Gerçek mimariye uyacak şekilde düzeltildi; `estimate_params()` artık ~3.698B toplam / ~1.886B aktif raporluyor, `ARCHITECTURE.md`'nin bağımsız olarak belirttiği ~1.86B aktif rakamıyla eşleşiyor. 4 yeni regresyon testi (`tests/test_scaling_audit_math.py`, `tests/test_config_dynamic_param_count.py`).
- `CHANGELOG.md`/`CHANGELOG_TR.md`: "Pass 7 (2026-06-13)", 2026-03-13 ve 2026-02-08 girdilerinden sonraya dosyalanmıştı, ters-kronolojik sırayı bozuyordu — doğru yerine (2026-06-17 ile 2026-05-24 arasına) taşındı.
- `ENV_VARS.md`: repodaki her `os.environ.get`/`os.getenv` çağrısına karşı canlı `grep` ile yeniden senkronlandı — dosya 2026-07-08→07-25 kararlılık çalışmasının tam bir ay gerisinde kalmıştı ve tüm LR/warmup-sweep ailesini (`TITAN_LEARNING_RATE`, `TITAN_ROUTER_LR_MULT`, `TITAN_WARMUP_RATIO`, `TITAN_WARMUP_STEPS`, `TITAN_LIQUID_WARMUP_STEPS`), `TITAN_DIVERGENCE_GUARD`'ı, re-warmup ailesini, off-site backup ailesini, `TITAN_PREFLIGHT_*` ailesini, `TITAN_DETERMINISTIC`'i, dataloader bayraklarını, `TITAN_PROFILE`/`TITAN_INSTALL`'ı ve `MERTFORMER_DDP_SMOKE_SECONDS`/`MERTFORMER_FUSED_BACKWARD`'ı eksik listeliyordu.
- `V2_BACKLOG_SEED.md` Track F: 3 madde (`liquid_warmup_steps` env override, z-loss çift-çarpım, `mark_weights_updated()` cache sorusu) `BACKLOG.md`/`DECISIONS.md` zaten çözülmüş/incelenmiş kaydettiği halde hâlâ açık gösteriliyordu — çapraz-referanslarla çözülmüş olarak işaretlendi.
- `CHESS_5080_POC_INTERNAL.md` (EN), kendi metninde EN dosyayı bayat diye işaretleyen `CHESS_5080_POC_INTERNAL_TR.md` ile içerik paritesine getirildi (Windows build workspace, EXE delivery, Stockfish auto-fetch cache, curated position suite, synthetic teaching corpus yalnız TR'de vardı).
- `TECHNICAL_REPORT.md`/`_TR`: başlık tarihi 2026-06-18'de donmuştu, gövde zaten 2026-07-19 tarihli bir revizyon taşıyordu (INT-KERNEL relabel) — sessizce tarihi geri değiştirmek yerine açık bir son-revizyon notu eklendi.
- `README.md`/`README_TR.md`: `AGENTS.md` (repo'nun kendi source-of-truth sırasında kendini 1. sıraya koyan dosya) hiçbir kök dokümandan link almıyordu; `START_HERE.md`/`README_SUMMARY.md` (dış-reviewer onboarding yolu) da `README.md`'den hiç linklenmiyordu. İkisi de Canonical surfaces listesine eklendi.

### Doğrulama
- `626 passed, 5 skipped` (son kayıtta `622 passed, 5 skipped` idi — +4, bu pass'in kendi yeni regresyon testleri). Bunun dışında yalnız dokümantasyon değişikliği (Master Protokol'e göre Sınıf A); `bash scripts/verify_all.sh` yeniden koşuldu, yeşil. Eğitim-matematiği, readiness veya iddia sınırı değişikliği yok.

## Unreleased - 2026-07-25

### Eklenenler
- `scripts/pre45k_gate.py`/`.sh` + `scripts/ddp_smoke.py`: offline preflight, dry-run önizlemesi ve gerçek bir 2-GPU DDP smoke testini tek, harcamadan-önce bir launch-hazırlık kapısında zincirliyor; `reports/pre45k_gate_report.{json,md}` yazıyor.
- `scripts/kaggle_batch_runner.py`: gözetimsiz çoklu-iş Kaggle orkestratörü; `evidence/2026-07-25-*` altında 4 gerçek kanıt seti üretti (Nutrition5k Liquid-OFF/MoE-OFF ablasyonları, 36M/171M LM yeniden-doğrulaması).
- `utils/divergence_guard.py`'ye mevcut loss-tabanlı frenin yanına bağımsız bir grad-norm EMA eş-tetikleyicisi ("C1") eklendi — gerçek 36M/171M donanımında doğru tetiklendiği doğrulandı.
- `scripts/offsite_backup_watcher.py`, `runbooks/checkpoint_offsite_backup.md`, `train/trainer_core.py::get_rewarmup_schedule()` (post-45K LR re-warmup).
- `tests/test_atomic_write_hygiene.py`: daha önce salt `.exists()` kontrolüyle güvenilen 5 pipeline dosyası için atomik (temp+`os.replace`) yazımlar.
- `model/nutrition_vision.py` + `scripts/{train,predict,evaluate}_nutrition5k.py`: gerçek BitLinear/MoE/Liquid gövdesini değiştirmeden yeniden kullanan sınırlı bir görsel yan-deney; gerçek eğitilmiş + bağımsız-yeniden-doğrulanmış checkpoint, sonra gerçek bir karşılaştırmalı ablasyon (bkz. Değişenler).

### Düzeltilenler
- z-loss efektif ağırlığı: kazara bir çift-çarpım onu Switch-Transformer/ST-MoE konvansiyonunun ~500x altına düşürmüştü; `z_loss_coef` `1e-4 → 0.05` olarak düzeltildi.
- `generate()` Liquid/CfC gizli durumunu decode adımları arasında hiç taşımıyordu — üretimde sessiz bir no-op; düzeltildi, tam-forward↔artımlı-decode parite testiyle birlikte.
- `bigcode/the-stack-dedup` revision/sha256 sonunda pinlendi (bir dataset-ref tarayıcı yanlış-pozitifi aylardır bunu engelliyordu).
- `scripts/kaggle_batch_runner.py::run_chess()` invocation bug'ı (yanlış `sys.path`) gerçek bir Kaggle koşusu sırasında canlı bulunup düzeltildi.
- `layers/moe.py` MoE dispatch-parallel `torch.bincount` → `scatter_add_` (MPS/eski-torch taşınabilirliği).

### Değişenler
- LR rejimi (`1.5e-3 → 3e-4`, sweep başlangıcı, doğrulanmış-güvenli değil), Liquid spike guard'ı (mutlak → EMA-göreli), WSD scheduler clamp'i — hepsi aday düzeltmeler, gerçek RTX-5070/Kaggle donanımında uygulanıp yeniden test edildi ama henüz yeterli kanıtlanmadı (bkz. Doğrulama).
- Sekiz launch-anı kararı kilitlendi (`DECISIONS.md`): lane = `online_teacher`, Liquid = Keep, model boyutu = 3.67B kanonik, `top_k` = 32 (256 değil), 2 ölü Stage-5 dataset'i doğrulanmış canlı biriyle değiştirildi, 3 lisans-TBD dataset tutuldu-ve-belgelendi, Stage-3 TR/sentetik oranı ratifiye edildi, INT-KERNEL iddiası dürüstçe relabel edildi (fp-simülasyon, henüz gerçek ternary kernel yok).
- Public gist yeniden düzenlendi: Nutrition5k öne alındı, gerçek bir z-loss aritmetik hatası düzeltildi (`~50x` → `~500x`), one-pager'ın pitch/yatırımcı çerçevesi araştırma çerçevesiyle değiştirildi.

### Doğrulama
- `622 passed, 5 skipped` (son kayıtta `370 passed, 4 skipped` idi). 2026-07-02/07-12/07-25 tarihlerinde üçüncü gerçek-donanım teyidi: bu mimari küçük ölçekte, daha fazla LR/optimizer çalışması olmadan hâlâ diverge ediyor — yeni grad-norm güvenlik freni (C1) artık iki ölçekte, kontrolsüz patlama yerine, temiz şekilde yakaladığı doğrulandı. Readiness değişmedi: `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. Trained/benchmark iddiası yok — kalan tek boşluk gerçek 45K GPU koşusu. Tam pass-pass detay: `BACKLOG.md`, `DECISIONS.md`.

## Unreleased - 2026-06-28

### Eklenenler
- `scripts/flip_status_banner.py`: **yalnızca-rapor** durum-banner denetleyicisi — pre-training banner'ı taşıyan tracked dosyaları listeler ve flip-sonrası uygunluğu (gerçek, sıfır-olmayan bir eval metriği) raporlar. **Yazma yolu yok**: gerçek kanıt-kapılı flip bilinçli bir post-run görevdir (naif bir "checkpoint+summary var" kapısı, başıboş bir demo checkpoint + stub summary ile tatmin olur, bu yüzden ön-inşa edilmiş bir auto-writer güvensizdir). Bkz. BACKLOG.
- `ENV_VARS.md`: kanonik training/precompute/orchestration environment değişkenlerinin varsayılanlarıyla birlikte tek indeksi.

### Düzeltilenler
- `eval/gsm8k.py`: checkpoint yükleme artık `weights_only=False` kullanıyor (+ `_orig_mod.` key normalizasyonu, non-strict load), belgelenmiş `train.py` resume yolunu yansıtıyor — post-45K GSM8K benchmark'ında gerçek bir training checkpoint'ini (optimizer/GaLore state) değerlendirirken torch≥2.6 `UnpicklingError`'ı önler.

### Değişenler
- Banner/sürüm hijyeni: frozen-olmayan `Status` / `Version` / `__version__` banner'ları kanonik Build-30-V2 formuna normalize edildi (`utils/logger.py`, `orchestrator/*`, `scripts/*`). Yalnız yorum/metadata, sıfır runtime değişikliği. Frozen-path banner'ları (`model/`, `train/`, `layers/`) bilinçli olarak post-45K kanıt-kapılı flip'e bırakıldı.

### Doğrulama
- `370 passed, 4 skipped` (offline-first pytest, değişmedi); readiness `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. Trained/benchmark iddiası yok — kalan tek boşluk gerçek 45K GPU koşusu.

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

## Pass 7 (2026-06-13) — Mac'te yapılabilir backlog sıfırlandı + $0 Kaggle pilotu
- `scripts/run_liquid_ablation.py` + `docs/KAGGLE_PILOT.md` eklendi: ücretsiz LiquidRouter ON-vs-OFF
  ablasyon pilotu (~80–100M, saf CE, teacher yok) — GPU'ya kapılı işi açan tek domino.
- Eğitim sırasında LatentODE per-batch reset (batch'ler arası state sızıntısı yok); MoE collapse flag DDP
  all-reduce (guarded, DDP dışında no-op); liquid-impl benchmark scripti; coverage config'i.
- Dokümanlar: ARCHITECTURE.md Projections + stage-3 notu; CPU quickstart. Backlog dispozisyonları DECISIONS.md'de.
- İnvaryantlar korundu: parametre sayısı kilitli; pytest yeşil; ruff + scoped mypy + verify_all yeşil.

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
