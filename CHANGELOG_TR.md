# Değişiklik Kaydı

Bu dosya projedeki önemli değişiklikleri takip eder.

> **Bakım notu (2026-07-25'te eklendi):** bu dosya elle bakımı yapılan, otomatik yeniden üretilmeyen bir dosya — bu not var olmadan önce tam bir ay (2026-06-28 → 2026-07-25) bayat kaldı. Gerçek bir `BACKLOG.md`/`DECISIONS.md` girdisi bırakan her closure pass'i, burada (TR) ve `CHANGELOG.md`'de (EN) güncel `## Unreleased - <tarih>` bölümünü de eklemeli/güncellemeli — kısa bir özet yeterli, tam detay `BACKLOG.md`/`DECISIONS.md`'de kalır. Bkz. `reports/change_control_sop.md`.
> Girdiler sıkı ters-kronolojik sırada tutulur (en yeni en üstte); 2026-07-27 pass'i "Pass 7 (2026-06-13)"in 2026-03-13/2026-02-08 etiketli sürümlerden sonraya yanlış dosyalandığını buldu ve doğru kronolojik yerine taşıdı — ayrıntı için aşağıdaki girdiye bakın.

## Unreleased - 2026-07-31

### Eklenenler
- Liquid/CfC duvar-saati maliyeti üzerine 2026-07-31 tarihli dış sinyal belgelendi (bağımsız, küçük-ölçekli `layers/liquid.py`/`layers/mla.py` testi, farklı donanım, bileşen seviyesinde ~9.4x `LiquidMixer`-vs-`GQA`, artı daha önce yazılmamış bir mekanizma: tekrarlama maliyeti `seq_len` ile ölçekleniyor, attention'ınki ölçeklenmiyor) — `BACKLOG.md`/`BACKLOG_TR.md`, `ABLATION.md`/`ABLATION_TR.md`, `WHITE_PAPER_LIQUIDROUTER.md`, `reports/liquid_keep_or_drop_brief.md`, `reports/blog_liquid_ablation_draft.md`, `reports/paper_outline_draft.md`, `reports/publication_readiness_kit.md`, ve `ablations/no_liquid/README.md`/`README_TR.md` genelinde. Sadece bilgilendirme — 2026-07-19 tarihli `DECIDED: Keep` kararını yeniden açmıyor, hiçbir ölçülmüş iddiayı değiştirmiyor, koda dokunulmadı. Tam detay: `BACKLOG_TR.md`.
- Yukarıdaki girdiye simetrik bir inference-tarafı karşı-nokta eklendi, aynı `BACKLOG.md`/`BACKLOG_TR.md`/`ABLATION.md`/`ABLATION_TR.md` girdisinde: `generate()`'in durum-tutan `LiquidMixer` decode yolu (`h_init`/`return_state`) zaten uygulanmış ve doğruluğu test edilmiş (`tests/test_liquid_generate_parity.py`'de full-forward'a karşı `<1e-8` parite, 2026-07-08'de düzeltildi), ve mimari olarak büyüyen-KV-cache attention'ın aksine context uzunluğundan bağımsız bir token-başı decode maliyetine işaret ediyor — ama decode-modu hızı hiç benchmark edilmedi. Eğitim-tarafı bulguyla aynı 45K-öncesi doğrulama maddesine katlandı; yeni bir iddia yok, koda dokunulmadı.
- Yeni `scripts/liquid_vs_gqa_canonical_benchmark.py` eklendi (bileşen-seviyeli `GQA`-vs-`LiquidMixer` train-mode + decode-mode mikro-benchmark'ı, `scripts/liquid_train_impl_benchmark.py` ile aynı üslup, CUDA/MPS/CPU) ve ikisi de bu makinenin kendi RTX 4060'ında (8GB VRAM) kanonik `hidden_size=2048`'de gerçekten koşuldu — yukarıdaki iki girdinin 45K-öncesi doğrulama maddesi, planlanmaktan ölçülmeye geçti. Sonuçlar: train-mode (`seq_len=512`, `batch_size=1`, `--fast-path 0` — bu Windows makinesinde çalışan bir Triton kurulumu yok) `LiquidMixer`'ı `GQA`'dan ~797-1620 kat yavaş ölçtü (`packed_pair`, `baseline`'dan yavaş), ve `seq_len=2048`/`4096` bu GPU'da `batch_size=1`'de bile doğrudan OOM verdi (sadece duvar-saati değil, daha önce hiç yazılmamış bir bellek-ölçekli maliyet). Decode-mode, `LiquidMixer`'ın token-başı maliyetini 24 kat'lık context taramasında sabit ve `GQA`'dan 8-23 kat daha hızlı ölçtü, yukarıdaki durum-tutan-decode hipotezini doğrulayarak. Tam sayılar ve hedge'ler: `BACKLOG.md`/`BACKLOG_TR.md` (aynı girdi), `ABLATION.md`/`ABLATION_TR.md` ekleri. Ham JSON: `reports/benchmarks/liquid_train_impl_canonical_512.json`, `reports/benchmarks/liquid_vs_gqa_canonical_decode.json`. Hâlâ tüketici-GPU/tek-koşu/kanonik-olmayan-ölçek kanıtı; `DECIDED: Keep`'i yeniden açmıyor. Model/eğitim koduna dokunulmadı.

### Düzeltilenler
- `layers/moe.py::MoE._dispatch_parallel()`: `torch.arange(N, device=topk_idx.device).repeat_interleave(k)`, `torch.onnx.export`'un tracer'ı altında cuda/cpu cihaz uyuşmazlığıyla çöküyordu (`repeat_interleave`'in ONNX sembolik dışa aktarımı kaynak cihazı taşımıyor) — bu makinenin CUDA GPU'sunda `tests/test_comprehensive.py::TestONNXCycle::test_onnx_export_import`'ı gerçekten koşunca ilk kez ortaya çıktı (daha önce hep `skipif` ile CPU-only/Mac makinelerde atlanmıştı). Cihaz-güvenli eşdeğeriyle düzeltildi: `torch.arange(N, device=topk_idx.device).unsqueeze(1).expand(N, k).reshape(-1)` — birebir aynı çıktı. `tests/test_moe_dispatch_parallel_counts.py` (12 test) etkilenmedi. Aynı test, ikinci ve daha derin bir ONNX-export uyumsuzluğu için de (`LiquidMixer`'ın eval-önbelleğindeki in-place buffer mutasyonu, `layers/liquid.py::_set_cache`) `xfail` işaretlendi (düzeltilmedi, tarihli sebep, Mert'in kararı) — `generate()`'in doğruluğu test edilmiş decode yolunun dayandığı aynı önbellek, gerçek bir mimari sorusu, aynı gece yapılacak bir düzeltme değil. Tam detay: `BACKLOG.md`.
- `tests/test_pre45k_gate.py::test_offline_preflight_reports_the_missing_corpus_rather_than_passing`: ambient `GITHUB_ACTIONS`/`CI`/`TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL` env değişkenlerini kendi subprocess çağrısına miras alıyordu; çıplak yerel checkout'ta geçiyor ama gerçek GitHub Actions CI'da (`bash scripts/verify_all.sh` üzerinden koşan) kırılıyordu. Artık kapıyı çağırmadan önce üçünü de `monkeypatch.delenv(...)` ile temizliyor; iki production escape hatch de (`scripts/titan_preflight.py`, `scripts/verify_all.sh`) değişmedi. Tam detay: `BACKLOG.md`.
- Windows taşınabilirlik geçişi: `bash scripts/final_one_shot.sh` Windows'ta hiç çalıştırılmamıştı ve defalarca başarısız oldu, her başarısızlık gerçek, tekrarlanabilir bir platformlar-arası hataydı. Kategoriye göre kök nedene inip düzeltildi: 9 test dosyasında `python3`-hardcode, `.titan-venv` Windows venv-düzeni algılama, manifest/senkron script'lerinde `Path.relative_to()` ters-bölü-çizgisi-vs-düz-bölü-çizgisi anahtar uyuşmazlıkları, 39 dosyanın `subprocess.run(text=True)` çağrılarında eksik `encoding="utf-8"` artı 3 dosyanın `sys.stdout`/`sys.stderr` UTF-8 `.reconfigure()` gerektirmesi (`train/train.py`'de bir `wandb`-console-wrapper çökmesini düzeltiyor), bir `os.isatty()` test varsayımı, `sys.platform == "darwin"` arkasına alınan yalnızca-Mac'e-özgü hardcode `/Applications` yolları, taşınabilir olmayan bir `mktemp -t` şablonu, en-iyi-çaba yapılan iki koşulsuz-ölümcül `dealroom_sync` çağrısı (reponun kardeş-repo-opsiyonel adımlar için zaten sahip olduğu deseniyle eşleşecek şekilde), bir `start_gate`/`check_doc_claim_consistency` kendi-kendini-değiştirdikten-sonra-gereksiz-yeniden-doğrulama sıralama hatası, göreli bir venv yorumlayıcı yolunu `subprocess.Popen()`/`subprocess.run()` argv'sine geçiren 3 script (`sys.executable` / mutlak yola düzeltildi), eksik bir `ruff` dev bağımlılığı (pinlenmiş sürümde kuruldu) ve 4 cihaz-seçim çağrı noktasında (`scripts/train_smoke.py`, `tests/test_moe_dispatch_parallel_counts.py`, `tests/test_architecture_integrity.py`, `apps/chess_gui/play_mertformer_chess_web.py`) hiç kontrol edilmeyen CUDA (yalnızca MPS-veya-CPU) — dördü de artık önce CUDA, sonra MPS, sonra CPU kontrol ediyor, `config/config.py`'nin mevcut varsayılanıyla eşleşerek hem bu makinede hem Mac'te davranış-nötr kalıyor. Kategori başına tam kök-neden detayı: `BACKLOG.md`.
- `tests/test_kaggle_onefile_colab_math_fastproof.py::test_logging_artifacts_written_and_compare_schema`, `resolve_runtime_config()`'in Colab-yolu-yansıtan varsayılanının testin kendi geçersiz kılmasından önce çözülmesi yüzünden her pytest koşusunda (herhangi bir işletim sisteminde) yan etki olarak gerçek, boş bir `~/Downloads/content/mertformer_outputs/` dizini oluşturuyordu; `out_dir`/`artifact_root`'u çağrıya *geçirilen* config sözlüğüne önceden kapsamlandırarak düzeltildi.
- `scripts/final_one_shot.sh`'nin `chess_5080_share_export` adımı, önceki paketleri temizlemeden her merdiven geçişinde benzersiz zaman damgalı bir Desktop teslimat zip'i yeniden inşa ediyordu, sessizce yinelenenler biriktiriyordu; `TITAN_CHESS_5080_EXPORT=1` ile opt-in yapıldı (varsayılan: atlandı). Platformdan-bağımsız bir davranış değişikliği, Windows'a özgü değil.
- `BACKLOG_TR.md`'nin kendi Windows-taşınabilirlik girdisi, literal pytest sayısını 3 yerde Türkçeye çevirmişti, dosyanın araç çıktısını birebir alıntılama konvansiyonunu bozuyordu; düzeltildi.
- Windows-taşınabilirlik commit'inin `git add -A`'sı ayrıca, dokümante edilmiş kanonik durum (`PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`) yerine bu makinenin kendi sıfır-korpus durumunu (`STAGE_JSONL_MISSING`) yansıtan yeniden üretilmiş bir `reports/train_readiness_decision.json`/`.md`'yi de süpürmüştü — bu, GitHub Actions'ın "Documentation claim consistency gate"ini kırdı: `scripts/verify_all.sh`, dokümanları kendi tazeleme adımı çalışmadan önce zaten commit'lenmiş olan readiness enstantanesine karşı kontrol ediyor. İki dosya da commit-öncesi içeriğine geri alındı ve bağımlı `reports/target_machine_handoff_manifest.json`/`.md` hash-tutarlılığı için yeniden üretildi. Kod/mantık hatası değil; tam detay `BACKLOG.md`'de.
- 2026-07-29 relicensing pass'inden kalan yazar-adı yazım regresyonu: yeni `LICENSE` copyright satırı (ve ondan kopyalanan her şey — `NOTICE`, `LICENSE_TR`, `README.md`/`README_TR.md`, ve 51 kaynak-dosya lisans header'ı) ASCII harf çevirisi `Mert Yunlu`'yu kullanıyordu, `DECISIONS.md`'nin başka bir yerinde belgeli, daha önceki bilinçli bir kararı (Türkçe yazım `Mert Yünlü`'yü standardize etme) sessizce geri alıyordu. 56 canlı örneğin hepsinde düzeltildi; `DECISIONS.md`/`DECISIONS_TR.md`'nin 2026-07-29 `LICENSE` metnini alıntılayan iki tarihli satırı bilerek doğru tarihsel kayıt olarak bırakıldı. Yalnızca yorum/metin değişikliği, sıfır çalışma-zamanı etkisi.

### Doğrulama
- Test sayısı değişmedi (`726 passed, 5 skipped` yerelde, önceki girdiyle aynı) — bu, mevcut bir testin ortam izolasyonunu düzeltiyor, test eklemiyor/silmiyor. Doğrulandı: CI ambient kirliliği simüle edildiğinde (`GITHUB_ACTIONS=true CI=true TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL=1`) düzeltme-öncesi hata tekrarlanıyor, düzeltme-sonrası geçiyor.
- `bash scripts/final_one_shot.sh` artık Windows/Git-Bash'te uçtan uca temiz tamamlanıyor: `[final] COMPLETED`, çıkış kodu 0. Bu makinenin dürüst ölçümü `721 passed, 10 skipped`, önceki Mac-ölçümü `726 passed, 5 skipped`'e karşı atlama-atlama denetlendi: 10'dan 2'si gerçekten platforma özgü (`test_architecture_integrity.py`/`test_comprehensive.py`'deki MPS-yokluğu atlamaları, Mac'te geçerdi), diğer 8'i bu reponun kendi zaten dokümante edilmiş temiz-klon-atlaması emsaliyle eşleşiyor (`c2c212e2`) — eksik yerel tokenizer artefaktları ve önceden var olan CUDA/MPS/QINN/korpus ortam atlamaları. Bilinçli olarak `726/5`'e geri döndürülmedi; bu, bu makinenin gerçek ölçümünü yanlış temsil ederdi. `md_integrity_check.py` (373 dosya, 0 bulgu) ve `secret_scan.py` (953 dosya, temiz) ikisi de geçiyor; eksik `sanitize_path()` kapsamı yüzünden yerel Windows kullanıcı adını/mutlak yolları sızdırdığı bulunan 5 otomatik-üretilmiş rapor dosyası commit'ten hariç tutuldu (yeniden üretilebilir artefaktlar, kaynak değil).

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
