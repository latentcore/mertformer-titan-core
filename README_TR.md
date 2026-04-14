## MertFormer Titan (Build 30 V2)

Kontrollu yerel dagitim ve durust ML systems kaniti icin offline-first, denetlenebilir yapay zeka altyapisi.

Mevcut olgunluk: **pilota hazir egitim oncesi baseline**.
Mevcut exact repo-side readiness: `TRAIN_ALLOWED`, reason `READY_OFFLINE_CLEAN`.

### Ilk Bakista Bilinmesi Gerekenler
- Basvuru acisindan ana kapi: gercek owned training run + checkpoint-bound evidence.
- Exact `45K`, tercih edilen ciddi dogrulama hedefidir; tek kabul edilebilir basvuru esigi degildir.
- Kanonik repo-side lane: `offline-clean`.
- Opsiyonel gated blocker: `online_teacher:MISSING_HF_TOKEN`.
- Hala acik olan post-run evidence sinifi: trained final weights, best/latest checkpoint proof, checkpoint-bound benchmark outputs, trained demo bundle ve trained export/device measurements.

### En Kisa Dogru Okuma Sirasi
1. [START_HERE.md](START_HERE.md)
2. [docs/PROJECT_MASTER_TRUTH.md](docs/PROJECT_MASTER_TRUTH.md)
3. [reports/final_truth_matrix.md](reports/final_truth_matrix.md)
4. [reports/known_limits_v1.md](reports/known_limits_v1.md)
5. [reports/systems_performance_case_study.md](reports/systems_performance_case_study.md)
6. [reports/offline_assistant_case_study.md](reports/offline_assistant_case_study.md)
7. [reports/chess_proof_teaching_case_study.md](reports/chess_proof_teaching_case_study.md)
8. [applications/anthropic/README.md](applications/anthropic/README.md)

### Anthropic Icin Yuksek-Sinyal Noktalar
- training efficiency ve systems debugging disiplini
- measured vs unmeasured claim sinirinin acik tutulmasi
- low-bit runtime ve backend-routing durustlugu
- governance-gated tool execution ile offline-first assistant foundation
- eksik post-run kaniti tamamlanmis gibi anlatmayan benchmark disiplini

### Kanonik Komutlar
- Repo dogrulamasi: `bash scripts/verify_all.sh`
- Sadece readiness kontrolu: `bash zero_touch_start.sh --check-only`
- Kanonik owned training lane baslatma: `bash zero_touch_start.sh`
- Final sync, release artifacts ve hash yenileme: `bash scripts/final_one_shot.sh`

### Uyumluluk Anchor'lari
- Closure companion script: `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py`
- Sabit README config anchor'i: `use_torch_compile = False`

### Dogruluk Siniri
- `measured` / `target` / `vision` ayri claim etiketleridir.
- `verified` / `hypothesis` / `creative_or_folklore` ayri cikti modlaridir.
- Varsayilan mod `verified`'dir.
- Varsayılan mod `verified`
- Bu depo, production-scale kalite iddialari icin **pre-training / dogrulanmamis** durumdadir.
- Trained checkpoint olusmadan benchmark durumu `NOT ELIGIBLE FOR CLAIM` olarak kalir.
- Bu gecis final urun iddiasi degil, proof-of-system muhendislik release'idir.
- Buyuk compute, kisisel finansman zorunlulugu degildir; asil kapi durust verified evidence ve coherent systems signal'dir.

### Acik Teknik Notlar
- `MLA-labeled GQA attention (current implementation)`
- `Routing policy: token-choice top-k.`
- Closure-57 seffaflik notu: `out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 54, 55, 56, 57]`
- `MLA etiketli GQA dikkat bloğu (mevcut implementasyon)`
- `Yönlendirme politikası: token-choice top-k.`

![MertFormer Titan Header](assets/header.png)

<div align="center">
  <a href="README.md">English</a> | <a href="README_TR.md">Turkce</a>
</div>

---

## Bu Repo Gercekte Neyi Kanitliyor
MertFormer Titan, uc bagli yuzeye sahip bir systems-and-evidence reposudur:
- low-bit systems ve runtime infrastructure
- offline assistant ve RAG foundation
- evaluation discipline ve training-system davranisini gosteren sinirli bir chess proof lane

Buradaki ana sinyal, headline trained-model benchmark'i degildir. Ana sinyal; readiness kapilarinin acik olmasi, verification'in script-owned olmasi, fallback davranisinin belgelenmesi ve claim boundary'nin anlatimla degil sistemle korunmasidir.

## Guncel Exact Durum
- Repo-wide verification ve closure yuzeyleri wired durumda.
- Guncel readiness verdict `TRAIN_ALLOWED`.
- Guncel readiness reason `READY_OFFLINE_CLEAN`.
- Kanonik yol `offline-clean`; teacher lane opsiyonel ve gated.
- Exact `45K` hala tercih edilen ciddi validation hedefidir; ama basvuru readiness, step sayisindan ziyade gercek owned training evidence + checkpoint bagina gore tanimlanir.
- Bu repoda henuz trained main checkpoint claim'i yoktur.
- Bu repoda henuz main model icin claim-grade headline benchmark claim'i yoktur.
- Chess evidence vardir; fakat trained ana dil modeli checkpoint'inin yerine gecen bir claim olarak anlatilmaz.

## Inceleme Yollari
### 1. Systems ve Scaling
Repo; low-bit runtime path, backend routing, verification gate, telemetry surface ve train-readiness contract ile verimlilik ve failure mode'lari incelenebilir hale getirmeyi hedefler.

### 2. Offline Assistant Foundation
Tool governance, local retrieval, observability ve assistant-side infrastructure gercek sistem yuzeyleri olarak vardir. Bunlar bitmis urun degil, foundation olarak anlatilir.

### 3. Chess Proof Lane
Chess lane; sinirli bir workload uzerinde training behavior, evaluation honesty ve evidence discipline gostermek icin vardir. Ana modelin frontier-scale language-model training'ini bitirdigi iddiasi olarak okunmamaktadir.

## Mimari Ozeti
Kod ve truth docs tarafinda su cekirdek fikirler gorunur:
- BitNet-style low-bit layer ve runtime disiplini
- sparse MoE akisinda temporal routing icin `LiquidRouter`
- mevcut implementasyonu acikca GQA-based olarak belirtilmis `MLA`-labeled attention
- manifests, reports, hashes ve closure script'leri etrafinda kurulan training/release governance

## Guncel Olculmus Kanit
- Son canonical test stat marker: `203 passed, 3 skipped`
- `python3 -m pytest -q` son closure pass'te gecti.
- `.titan-venv/bin/python -m ruff check .` son closure pass'te gecti.
- `bash scripts/verify_all.sh` kanonik repo verification gate'idir ve son closure pass'te gecmektedir.
- Offline-clean lane icin repo-side training readiness yesildir.
- Closure artifacts, manifests ve package hashes script ile yeniden uretilir; el ile tutulmaz.

Exact evidence boundary icin bak:
- [reports/known_limits_v1.md](reports/known_limits_v1.md)
- [reports/final_backlog_missing_items.md](reports/final_backlog_missing_items.md)
- [applications/anthropic/measured_evidence_summary.md](applications/anthropic/measured_evidence_summary.md)

## Hala Acik Olan Sinif
Kalan yuksek degerli acik sinif, eksik klasor veya eksik script degil; post-run evidence sinifidir:
- gercek owned training run tamamlama
- trained final weights
- best/latest checkpoint proof
- checkpoint-bound benchmark outputs
- trained demo bundle
- guclu arti olarak trained export/device measurements

Bu kanit olusana kadar ana model claim-grade benchmark anlatimina gecmez.

## Quick Start
### Baseline Kurulum
```bash
bash scripts/bootstrap_venv.sh
bash scripts/verify_all.sh
```

### Readiness Kontrolleri
```bash
bash zero_touch_start.sh --check-only
bash zero_touch_start.sh --plan-only
```

### Kanonik Baslatma Yolu
```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

### Final Closure Chain
```bash
bash scripts/verify_all.sh
bash scripts/final_one_shot.sh
bash scripts/verify_all.sh
```

### Minimal Config Anchor
```toml
use_torch_compile = False
```

## Detayli Dokumantasyon
En kisa public ozet icin:
- [README_SUMMARY.md](README_SUMMARY.md)
- [README_SUMMARY_TR.md](README_SUMMARY_TR.md)

Operasyonel giris ve truth docs icin:
- [START_HERE.md](START_HERE.md)
- [docs/CHESS_ONEFILE_MASTER_TRUTH_TR.md](docs/CHESS_ONEFILE_MASTER_TRUTH_TR.md)
- [docs/PROJECT_MASTER_TRUTH_TR.md](docs/PROJECT_MASTER_TRUTH_TR.md)
- [reports/final_truth_matrix.md](reports/final_truth_matrix.md)
- [reports/known_limits_v1.md](reports/known_limits_v1.md)
- [reports/final_backlog_missing_items.md](reports/final_backlog_missing_items.md)

Daha derin teknik ve basvuru-materyali icin:
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- [SDK_GUIDE.md](SDK_GUIDE.md)
- [USAGE_GUIDE.md](USAGE_GUIDE.md)
- [applications/anthropic/README.md](applications/anthropic/README.md)

## Lisans
Bu repo proprietary ve confidential'dir. Hukuki sinir icin [LICENSE](LICENSE) ve [LICENSE_TR](LICENSE_TR) dosyalarina bak.

### Kanonik Yerleşim (Build 30 V2)
```text
mertformer-titan-core/  # proje kökü (git ls-files envanteri)
├── .github/  # dizin
│   ├── workflows/  # dizin
│   │   └── ci.yml  # YAML yapılandırma dosyası
│   └── CODEOWNERS  # artefakt
├── ablations/  # dizin
│   ├── bitlinear_off/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── dense_only/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── no_liquid/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── no_moe/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── results.md  # dokümantasyon/rapor dosyası
│   └── results_TR.md  # Türkçe doküman karşılığı
├── adr/  # dizin
│   ├── ADR-0001-source-of-truth-and-claim-boundary.md  # dokümantasyon/rapor dosyası
│   ├── ADR-0002-change-control-and-closure-governance.md  # dokümantasyon/rapor dosyası
│   └── ADR-0003-chess-oneclick-delivery-runtime-contract.md  # dokümantasyon/rapor dosyası
├── applications/  # dizin
│   └── anthropic/  # dizin
│       ├── README.md  # ana dokümantasyon (EN)
│       ├── interview_prep.md  # dokümantasyon/rapor dosyası
│       ├── measured_evidence_summary.md  # dokümantasyon/rapor dosyası
│       ├── performance_engineer_fallback.md  # dokümantasyon/rapor dosyası
│       ├── project_summary.md  # dokümantasyon/rapor dosyası
│       ├── science_of_scaling_cv_seed.md  # dokümantasyon/rapor dosyası
│       ├── strongest_stories.md  # dokümantasyon/rapor dosyası
│       ├── tokens_variant_notes.md  # dokümantasyon/rapor dosyası
│       └── why_anthropic_science_of_scaling.md  # dokümantasyon/rapor dosyası
├── apps/  # dizin
│   └── chess_gui/  # dizin
│       ├── checkpoints/  # dizin
│       │   └── README.md  # ana dokümantasyon (EN)
│       ├── logs/  # dizin
│       │   └── README.md  # ana dokümantasyon (EN)
│       ├── .gitignore  # git ignore politikası
│       ├── README.md  # ana dokümantasyon (EN)
│       ├── launch_mertformer_chess_gui.command  # artefakt
│       └── play_mertformer_chess_web.py  # Python modülü/scripti (play mertformer chess web için modül)
├── artifacts/  # dizin
│   └── mertformer_release.zip.sha256  # artefakt sağlama toplamı
├── assets/  # dizin
│   ├── sources/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── header.png  # medya varlığı
│   ├── snake_demo_preview.gif  # medya varlığı
│   ├── snake_demo_proof.mp4  # medya varlığı
│   └── synaptic_map.png  # medya varlığı
├── checklists/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   ├── chess_4060_24h.md  # dokümantasyon/rapor dosyası
│   ├── chess_4060_24h_TR.md  # Türkçe doküman karşılığı
│   ├── chess_4060_24h_all_on_experimental.md  # dokümantasyon/rapor dosyası
│   └── chess_4060_24h_all_on_experimental_TR.md  # Türkçe doküman karşılığı
├── config/  # dizin
│   ├── export/  # dizin
│   │   └── onnx_mobile.yaml  # YAML yapılandırma dosyası
│   ├── model/  # dizin
│   │   ├── mertformer_max_arch.yaml  # YAML yapılandırma dosyası
│   │   ├── mertformer_moe.yaml  # YAML yapılandırma dosyası
│   │   └── mertformer_small.yaml  # YAML yapılandırma dosyası
│   ├── train/  # dizin
│   │   ├── finetune.yaml  # YAML yapılandırma dosyası
│   │   └── pretrain.yaml  # YAML yapılandırma dosyası
│   ├── __init__.py  # Python modülü/scripti (config paket başlatıcısı ve dışa aktarmalar)
│   ├── base.yaml  # YAML yapılandırma dosyası
│   └── config.py  # Python modülü/scripti (çalışma zamanı konfigürasyon modeli ve doğrulama yardımcıları)
├── configs/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   └── chess_onefile_profile_contract.md  # dokümantasyon/rapor dosyası
├── datasets/  # dizin
│   ├── INTERNAL_POLICY.md  # dokümantasyon/rapor dosyası
│   ├── INTERNAL_POLICY_TR.md  # Türkçe doküman karşılığı
│   ├── LICENSES.md  # dokümantasyon/rapor dosyası
│   ├── LICENSES_TR.md  # Türkçe doküman karşılığı
│   ├── README.md  # ana dokümantasyon (EN)
│   ├── README_TR.md  # Türkçe doküman karşılığı
│   ├── SOURCES.md  # dokümantasyon/rapor dosyası
│   ├── SOURCES_TR.md  # Türkçe doküman karşılığı
│   ├── filters.yaml  # YAML yapılandırma dosyası
│   ├── golden_assertions.jsonl  # JSONL veri/log artefaktı
│   ├── golden_samples.jsonl  # JSONL veri/log artefaktı
│   ├── hashes.json  # JSON veri artefaktı
│   ├── inventory.json  # JSON veri artefaktı
│   ├── inventory.md  # dokümantasyon/rapor dosyası
│   ├── inventory_TR.md  # Türkçe doküman karşılığı
│   └── validation.jsonl  # JSONL veri/log artefaktı
├── docs/  # dizin
│   ├── CHAIN_MAP.md  # dokümantasyon/rapor dosyası
│   ├── CHAIN_MAP_TR.md  # Türkçe doküman karşılığı
│   ├── CHESS_ONEFILE_MASTER_TRUTH.md  # dokümantasyon/rapor dosyası
│   ├── CHESS_ONEFILE_MASTER_TRUTH_TR.md  # Türkçe doküman karşılığı
│   ├── PROJECT_MASTER_TRUTH.md  # dokümantasyon/rapor dosyası
│   └── PROJECT_MASTER_TRUTH_TR.md  # Türkçe doküman karşılığı
├── economics/  # dizin
│   ├── cost_model.md  # dokümantasyon/rapor dosyası
│   ├── cost_model_TR.md  # Türkçe doküman karşılığı
│   ├── efficiency_report.md  # dokümantasyon/rapor dosyası
│   ├── efficiency_report_TR.md  # Türkçe doküman karşılığı
│   └── flops_estimator.py  # Python modülü/scripti (flops estimator için modül)
├── eval/  # dizin
│   ├── agentic_suite.py  # Python modülü/scripti (agentic suite için değerlendirme rutini)
│   ├── generalization_suite.py  # Python modülü/scripti (generalization suite için değerlendirme rutini)
│   ├── golden.py  # Python modülü/scripti (golden için değerlendirme rutini)
│   ├── gsm8k.py  # Python modülü/scripti (gsm8k için değerlendirme rutini)
│   ├── humaneval.py  # Python modülü/scripti (humaneval için değerlendirme rutini)
│   └── report_builder.py  # Python modülü/scripti (report builder için değerlendirme rutini)
├── evidence/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   └── chess_evidence_contract.md  # dokümantasyon/rapor dosyası
├── experiments/  # dizin
│   └── exp_001_baseline/  # dizin
│       ├── config.yaml  # YAML yapılandırma dosyası
│       ├── metrics.json  # JSON veri artefaktı
│       ├── notes.md  # dokümantasyon/rapor dosyası
│       └── notes_TR.md  # Türkçe doküman karşılığı
├── interfaces/  # dizin
│   ├── backlog_item_v1.schema.json  # JSON şema artefaktı
│   ├── closure_57_matrix_v1.schema.json  # JSON şema artefaktı
│   ├── inference_contract.md  # dokümantasyon/rapor dosyası
│   ├── inference_contract_TR.md  # Türkçe doküman karşılığı
│   ├── kpi_report_v1.schema.json  # JSON şema artefaktı
│   ├── pilot_report_v1.schema.json  # JSON şema artefaktı
│   ├── run_manifest_v1.schema.json  # JSON şema artefaktı
│   ├── tokenizer_spec.json  # JSON veri artefaktı
│   └── workspace_hygiene_manifest_v1.schema.json  # JSON şema artefaktı
├── knowledge/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   └── chess_onefile_glossary.md  # dokümantasyon/rapor dosyası
├── layers/  # dizin
│   ├── __init__.py  # Python modülü/scripti (layers paket başlatıcısı ve dışa aktarmalar)
│   ├── bitlinear.py  # Python modülü/scripti (BitLinear düşük-bit linear katman implementasyonu)
│   ├── bitnet_patch.py  # Python modülü/scripti (BitNet kuantizasyon patch ve runtime kancaları)
│   ├── cognitive_extensions.py  # Python modülü/scripti (opsiyonel bilişsel genişletme blokları)
│   ├── ffn.py  # Python modülü/scripti (feed-forward ağ blokları (dense ve sparse yollar))
│   ├── lifelong_safety.py  # Python modülü/scripti (yaşam boyu güvenlik koruma katmanı)
│   ├── liquid.py  # Python modülü/scripti (liquid sinir dinamik katmanları)
│   ├── mertformer_block.py  # Python modülü/scripti (çekirdek transformer blok bileşimi)
│   ├── mla.py  # Python modülü/scripti (çok başlı latent attention implementasyonu)
│   ├── moe.py  # Python modülü/scripti (mixture-of-experts yönlendirme ve uzman çalıştırma)
│   ├── qinn.py  # Python modülü/scripti (QINN deneysel regülasyon katmanı (feature-flag))
│   └── world_model_head.py  # Python modülü/scripti (dünya-modeli yardımcı çıktı kafası)
├── limits/  # dizin
│   ├── scaling_breakpoints.md  # dokümantasyon/rapor dosyası
│   ├── scaling_breakpoints_TR.md  # Türkçe doküman karşılığı
│   └── stress_curves.png  # medya varlığı
├── logs/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   └── README_TR.md  # Türkçe doküman karşılığı
├── mertformer_sdk/  # dizin
│   ├── kernels/  # dizin
│   │   ├── cpp/  # dizin
│   │   │   ├── __init__.py  # Python modülü/scripti (cpp paket başlatıcısı ve dışa aktarmalar)
│   │   │   ├── bitnet_cpu.cpp  # C++ kaynak dosyası
│   │   │   └── loader.py  # Python modülü/scripti (loader için SDK bileşeni)
│   │   ├── metal/  # dizin
│   │   │   ├── __init__.py  # Python modülü/scripti (metal paket başlatıcısı ve dışa aktarmalar)
│   │   │   └── engine.py  # Python modülü/scripti (engine için SDK bileşeni)
│   │   ├── npu/  # dizin
│   │   │   ├── __init__.py  # Python modülü/scripti (npu paket başlatıcısı ve dışa aktarmalar)
│   │   │   └── engine.py  # Python modülü/scripti (engine için SDK bileşeni)
│   │   ├── vulkan/  # dizin
│   │   │   ├── __init__.py  # Python modülü/scripti (vulkan paket başlatıcısı ve dışa aktarmalar)
│   │   │   └── engine.py  # Python modülü/scripti (engine için SDK bileşeni)
│   │   ├── __init__.py  # Python modülü/scripti (kernels paket başlatıcısı ve dışa aktarmalar)
│   │   ├── dispatcher.py  # Python modülü/scripti (dispatcher için SDK bileşeni)
│   │   ├── onnx_custom_op.py  # Python modülü/scripti (onnx custom op için SDK bileşeni)
│   │   └── triton_ternary.py  # Python modülü/scripti (triton ternary için SDK bileşeni)
│   ├── utils/  # dizin
│   │   ├── __init__.py  # Python modülü/scripti (utils paket başlatıcısı ve dışa aktarmalar)
│   │   ├── bitpack.py  # Python modülü/scripti (bitpack için SDK bileşeni)
│   │   └── onnx_meta.py  # Python modülü/scripti (onnx meta için SDK bileşeni)
│   ├── __init__.py  # Python modülü/scripti (mertformer_sdk paket başlatıcısı ve dışa aktarmalar)
│   ├── api.py  # Python modülü/scripti (api için SDK bileşeni)
│   ├── cli.py  # Python modülü/scripti (cli için SDK bileşeni)
│   ├── export.py  # Python modülü/scripti (export için SDK bileşeni)
│   ├── kpi.py  # Python modülü/scripti (kpi için SDK bileşeni)
│   └── pilot.py  # Python modülü/scripti (pilot için SDK bileşeni)
├── model/  # dizin
│   ├── __init__.py  # Python modülü/scripti (model paket başlatıcısı ve dışa aktarmalar)
│   └── transformers.py  # Python modülü/scripti (MertFormer model montajı ve ileri geçiş grafiği)
├── orchestrator/  # dizin
│   ├── __init__.py  # Python modülü/scripti (orchestrator paket başlatıcısı ve dışa aktarmalar)
│   ├── agent_registry.py  # Python modülü/scripti (agent registry için orkestratör runtime bileşeni)
│   ├── alignment_contracts.py  # Python modülü/scripti (alignment contracts için orkestratör runtime bileşeni)
│   ├── audio_sense.py  # Python modülü/scripti (audio sense için orkestratör runtime bileşeni)
│   ├── cognitive.py  # Python modülü/scripti (cognitive için orkestratör runtime bileşeni)
│   ├── cognitive_loop.py  # Python modülü/scripti (cognitive loop için orkestratör runtime bileşeni)
│   ├── compute_orchestrator.py  # Python modülü/scripti (compute orchestrator için orkestratör runtime bileşeni)
│   ├── core.py  # Python modülü/scripti (core için orkestratör runtime bileşeni)
│   ├── distillation_manager.py  # Python modülü/scripti (distillation manager için orkestratör runtime bileşeni)
│   ├── experience_store.py  # Python modülü/scripti (experience store için orkestratör runtime bileşeni)
│   ├── failure_budget.py  # Python modülü/scripti (failure budget için orkestratör runtime bileşeni)
│   ├── governance.py  # Python modülü/scripti (governance için orkestratör runtime bileşeni)
│   ├── hardware.py  # Python modülü/scripti (hardware için orkestratör runtime bileşeni)
│   ├── memory.py  # Python modülü/scripti (memory için orkestratör runtime bileşeni)
│   ├── paths.py  # Python modülü/scripti (paths için orkestratör runtime bileşeni)
│   ├── planner.py  # Python modülü/scripti (planner için orkestratör runtime bileşeni)
│   ├── reasoning_engine.py  # Python modülü/scripti (reasoning engine için orkestratör runtime bileşeni)
│   ├── self_audit.py  # Python modülü/scripti (self audit için orkestratör runtime bileşeni)
│   ├── self_improvement_guard.py  # Python modülü/scripti (self improvement guard için orkestratör runtime bileşeni)
│   ├── sense_engine.py  # Python modülü/scripti (sense engine için orkestratör runtime bileşeni)
│   ├── swarm_runtime.py  # Python modülü/scripti (swarm runtime için orkestratör runtime bileşeni)
│   ├── telemetry.py  # Python modülü/scripti (telemetry için orkestratör runtime bileşeni)
│   ├── tool_executor.py  # Python modülü/scripti (tool executor için orkestratör runtime bileşeni)
│   ├── tool_registry.py  # Python modülü/scripti (tool registry için orkestratör runtime bileşeni)
│   ├── verifier.py  # Python modülü/scripti (verifier için orkestratör runtime bileşeni)
│   └── web_sense.py  # Python modülü/scripti (web sense için orkestratör runtime bileşeni)
├── policy/  # dizin
│   └── allow_deny_policy.yaml  # YAML yapılandırma dosyası
├── postmortems/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   ├── README_TR.md  # Türkçe doküman karşılığı
│   ├── _template.md  # dokümantasyon/rapor dosyası
│   ├── _template_TR.md  # Türkçe doküman karşılığı
│   ├── example_001.md  # dokümantasyon/rapor dosyası
│   └── example_001_TR.md  # Türkçe doküman karşılığı
├── prompts/  # dizin
│   ├── changelog.md  # dokümantasyon/rapor dosyası
│   ├── changelog_TR.md  # Türkçe doküman karşılığı
│   └── system_v1.txt  # metin artefaktı
├── registry/  # dizin
│   └── mertformer_v0.1.json  # JSON veri artefaktı
├── releases/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   └── chess_release_contract.md  # dokümantasyon/rapor dosyası
├── reports/  # dizin
│   ├── benchmarks/  # dizin
│   │   ├── linkedin_sweetspot/  # dizin
│   │   │   ├── README.md  # ana dokümantasyon (EN)
│   │   │   ├── README_TR.md  # Türkçe doküman karşılığı
│   │   │   ├── run_20260318_144125_artifact_index.json  # JSON veri artefaktı
│   │   │   ├── run_20260318_144125_compare.csv  # CSV veri artefaktı
│   │   │   ├── run_20260318_144125_compare.json  # JSON veri artefaktı
│   │   │   ├── run_20260318_144125_compare.md  # dokümantasyon/rapor dosyası
│   │   │   ├── run_20260318_144125_health.txt  # metin artefaktı
│   │   │   ├── run_20260318_144125_run_log.jsonl  # JSONL veri/log artefaktı
│   │   │   ├── run_20260318_144125_step_metrics.csv  # CSV veri artefaktı
│   │   │   ├── run_20260318_144125_summary.json  # JSON veri artefaktı
│   │   │   └── zip_manifest.json  # JSON veri artefaktı
│   │   ├── math_fastproof/  # dizin
│   │   │   ├── README.md  # ana dokümantasyon (EN)
│   │   │   ├── README_TR.md  # Türkçe doküman karşılığı
│   │   │   ├── run_20260315_050133_artifact_index.json  # JSON veri artefaktı
│   │   │   ├── run_20260315_050133_compare.csv  # CSV veri artefaktı
│   │   │   ├── run_20260315_050133_compare.json  # JSON veri artefaktı
│   │   │   ├── run_20260315_050133_compare.md  # dokümantasyon/rapor dosyası
│   │   │   ├── run_20260315_050133_health.txt  # metin artefaktı
│   │   │   ├── run_20260315_050133_run_log.jsonl  # JSONL veri/log artefaktı
│   │   │   ├── run_20260315_050133_step_metrics.csv  # CSV veri artefaktı
│   │   │   ├── run_20260315_050133_summary.json  # JSON veri artefaktı
│   │   │   └── zip_manifest.json  # JSON veri artefaktı
│   │   ├── text_understanding/  # dizin
│   │   │   ├── README.md  # ana dokümantasyon (EN)
│   │   │   ├── README_TR.md  # Türkçe doküman karşılığı
│   │   │   ├── run_20260315_180151_artifact_index.json  # JSON veri artefaktı
│   │   │   ├── run_20260315_180151_compare.csv  # CSV veri artefaktı
│   │   │   ├── run_20260315_180151_compare.json  # JSON veri artefaktı
│   │   │   ├── run_20260315_180151_compare.md  # dokümantasyon/rapor dosyası
│   │   │   ├── run_20260315_180151_health.txt  # metin artefaktı
│   │   │   ├── run_20260315_180151_run_log.jsonl  # JSONL veri/log artefaktı
│   │   │   └── run_20260315_180151_summary.json  # JSON veri artefaktı
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   ├── README_TR.md  # Türkçe doküman karşılığı
│   │   ├── agentic_suite_build30.json  # JSON veri artefaktı
│   │   ├── generalization_suite_build30.json  # JSON veri artefaktı
│   │   ├── internal_smoke_summary.json  # JSON veri artefaktı
│   │   ├── kaggle_compare_build30.csv  # CSV veri artefaktı
│   │   ├── kaggle_compare_build30.json  # JSON veri artefaktı
│   │   ├── kaggle_compare_build30.md  # dokümantasyon/rapor dosyası
│   │   ├── smoke_train_metrics.json  # JSON veri artefaktı
│   │   ├── summary.json  # JSON veri artefaktı
│   │   └── summary.md  # dokümantasyon/rapor dosyası
│   ├── commercial_handover/  # dizin
│   │   ├── contract_terms_checklist.md  # dokümantasyon/rapor dosyası
│   │   ├── contract_terms_checklist_TR.md  # Türkçe doküman karşılığı
│   │   ├── handover_scope.md  # dokümantasyon/rapor dosyası
│   │   ├── handover_scope_TR.md  # Türkçe doküman karşılığı
│   │   ├── known_issues.md  # dokümantasyon/rapor dosyası
│   │   ├── known_issues_TR.md  # Türkçe doküman karşılığı
│   │   ├── ownership_and_role.md  # dokümantasyon/rapor dosyası
│   │   ├── ownership_and_role_TR.md  # Türkçe doküman karşılığı
│   │   ├── sla_kpi_90_180.md  # dokümantasyon/rapor dosyası
│   │   └── sla_kpi_90_180_TR.md  # Türkçe doküman karşılığı
│   ├── pilots/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── snapshots/  # dizin
│   │   └── 2026-02-24/  # dizin
│   │       ├── claim_matrix_v2_2026-02-24.json  # JSON veri artefaktı
│   │       ├── commercial_scenarios_v1_2026-02-24.json  # JSON veri artefaktı
│   │       ├── evidence_snapshot_2026-02-24.json  # JSON veri artefaktı
│   │       ├── mertformer_master_decision_report_TR_2026-02-24.md  # dokümantasyon/rapor dosyası
│   │       ├── readiness_scorecard_v1_2026-02-24.json  # JSON veri artefaktı
│   │       ├── report_interface_schema_v1.json  # JSON şema artefaktı
│   │       └── web_validation_sources_2026-02-24.md  # dokümantasyon/rapor dosyası
│   ├── adr_index.md  # dokümantasyon/rapor dosyası
│   ├── architecture_honesty_audit.md  # dokümantasyon/rapor dosyası
│   ├── artifacts_zip_denylist_audit.json  # JSON veri artefaktı
│   ├── asset_stack.md  # dokümantasyon/rapor dosyası
│   ├── asset_stack_TR.md  # Türkçe doküman karşılığı
│   ├── automation_boundary_policy.md  # dokümantasyon/rapor dosyası
│   ├── backlog_operating_contract.md  # dokümantasyon/rapor dosyası
│   ├── backup_restore_report.json  # JSON veri artefaktı
│   ├── bench_cpp_report.json  # JSON veri artefaktı
│   ├── bench_metal_report.json  # JSON veri artefaktı
│   ├── bench_npu_report.json  # JSON veri artefaktı
│   ├── bench_vulkan_report.json  # JSON veri artefaktı
│   ├── bench_zero_copy_report.json  # JSON veri artefaktı
│   ├── benchmark_compare_report.json  # JSON veri artefaktı
│   ├── benchmark_compare_report.md  # dokümantasyon/rapor dosyası
│   ├── benchmark_contract.md  # dokümantasyon/rapor dosyası
│   ├── canonical_entrypoint.md  # dokümantasyon/rapor dosyası
│   ├── cfc_moe_tolerance_report.json  # JSON veri artefaktı
│   ├── change_control_sop.md  # dokümantasyon/rapor dosyası
│   ├── checkpoint_contract.md  # dokümantasyon/rapor dosyası
│   ├── checkpoint_hash_manifest.json  # JSON veri artefaktı
│   ├── checkpoint_restore_report.json  # JSON veri artefaktı
│   ├── chess_gui_onefile_sync_report.json  # JSON veri artefaktı
│   ├── chess_gui_onefile_sync_report.md  # dokümantasyon/rapor dosyası
│   ├── chess_onefile_extension_report.json  # JSON veri artefaktı
│   ├── chess_onefile_extension_report.md  # dokümantasyon/rapor dosyası
│   ├── chess_proof_teaching_case_study.md  # dokümantasyon/rapor dosyası
│   ├── chess_teaching_contract_report.json  # JSON veri artefaktı
│   ├── chess_teaching_contract_report.md  # dokümantasyon/rapor dosyası
│   ├── chess_training_readiness_report.json  # JSON veri artefaktı
│   ├── chess_training_readiness_report.md  # dokümantasyon/rapor dosyası
│   ├── claim_number_audit.json  # JSON veri artefaktı
│   ├── claim_registry.json  # JSON veri artefaktı
│   ├── cleanroom_verification.md  # dokümantasyon/rapor dosyası
│   ├── cleanroom_verification_TR.md  # Türkçe doküman karşılığı
│   ├── cleanup_scoped_closure_junk_report.json  # JSON veri artefaktı
│   ├── cli_smoke_log.md  # dokümantasyon/rapor dosyası
│   ├── cli_smoke_log_TR.md  # Türkçe doküman karşılığı
│   ├── closure_57_matrix.json  # JSON veri artefaktı
│   ├── closure_57_matrix.md  # dokümantasyon/rapor dosyası
│   ├── closure_57_matrix_TR.md  # Türkçe doküman karşılığı
│   ├── closure_report_build30_v2.md  # dokümantasyon/rapor dosyası
│   ├── closure_risk_register.md  # dokümantasyon/rapor dosyası
│   ├── cloud_readiness_report.md  # dokümantasyon/rapor dosyası
│   ├── code_truth_contract.md  # dokümantasyon/rapor dosyası
│   ├── code_truth_delta_audit.json  # JSON veri artefaktı
│   ├── code_truth_delta_audit.md  # dokümantasyon/rapor dosyası
│   ├── codex_deep_audit_DE.md  # dokümantasyon/rapor dosyası
│   ├── codex_deep_audit_DE_TR.md  # Türkçe doküman karşılığı
│   ├── codex_deep_audit_EN.md  # dokümantasyon/rapor dosyası
│   ├── codex_deep_audit_EN_TR.md  # Türkçe doküman karşılığı
│   ├── codex_deep_audit_TR.md  # Türkçe doküman karşılığı
│   ├── commercial_handover_pack.md  # dokümantasyon/rapor dosyası
│   ├── contamination_report_build30.md  # dokümantasyon/rapor dosyası
│   ├── customer_ready_definition.md  # dokümantasyon/rapor dosyası
│   ├── data_pipeline_contract.md  # dokümantasyon/rapor dosyası
│   ├── data_pipeline_provenance.json  # JSON veri artefaktı
│   ├── data_pipeline_token_probe.json  # JSON veri artefaktı
│   ├── dataset_health.md  # dokümantasyon/rapor dosyası
│   ├── dataset_health_TR.md  # Türkçe doküman karşılığı
│   ├── dataset_health_final.md  # dokümantasyon/rapor dosyası
│   ├── dataset_lineage_final.json  # JSON veri artefaktı
│   ├── dealroom_reference.json  # JSON veri artefaktı
│   ├── demo_bundle.md  # dokümantasyon/rapor dosyası
│   ├── demo_bundle_manifest.json  # JSON veri artefaktı
│   ├── deprecated_surface_report.md  # dokümantasyon/rapor dosyası
│   ├── determinism_report.json  # JSON veri artefaktı
│   ├── differential_backend_report.json  # JSON veri artefaktı
│   ├── doc_alignment_report.json  # JSON veri artefaktı
│   ├── doc_alignment_report.md  # dokümantasyon/rapor dosyası
│   ├── doc_ownership_matrix.md  # dokümantasyon/rapor dosyası
│   ├── docs_dedup_canonical_list.md  # dokümantasyon/rapor dosyası
│   ├── docs_packages_hash_manifest.json  # JSON veri artefaktı
│   ├── drone_sitl_demo.md  # dokümantasyon/rapor dosyası
│   ├── drone_sitl_demo_TR.md  # Türkçe doküman karşılığı
│   ├── dry_run_report.json  # JSON veri artefaktı
│   ├── dry_run_report.md  # dokümantasyon/rapor dosyası
│   ├── duplicate_source_of_truth_report.md  # dokümantasyon/rapor dosyası
│   ├── duplicate_zip_guard_report.json  # JSON veri artefaktı
│   ├── edge_readiness_plan.md  # dokümantasyon/rapor dosyası
│   ├── efficiency_convergence_analysis.md  # dokümantasyon/rapor dosyası
│   ├── efficiency_convergence_analysis_TR.md  # Türkçe doküman karşılığı
│   ├── energy_baseline.json  # JSON veri artefaktı
│   ├── entrypoint_deprecation_map.md  # dokümantasyon/rapor dosyası
│   ├── execution_trace.json  # JSON veri artefaktı
│   ├── exit_code_standard.md  # dokümantasyon/rapor dosyası
│   ├── expected_artifacts_list.md  # dokümantasyon/rapor dosyası
│   ├── export_validation_report.json  # JSON veri artefaktı
│   ├── external_readability_checklist.md  # dokümantasyon/rapor dosyası
│   ├── fallback_policy_report.json  # JSON veri artefaktı
│   ├── feature_flag_governance.md  # dokümantasyon/rapor dosyası
│   ├── file_state_inventory.json  # JSON veri artefaktı
│   ├── final_artifact_manifest.json  # JSON veri artefaktı
│   ├── final_backlog_classification.json  # JSON veri artefaktı
│   ├── final_backlog_classification.md  # dokümantasyon/rapor dosyası
│   ├── final_backlog_coverage_diff.md  # dokümantasyon/rapor dosyası
│   ├── final_backlog_missing_items.md  # dokümantasyon/rapor dosyası
│   ├── final_checksum_manifest.json  # JSON veri artefaktı
│   ├── final_commands.md  # dokümantasyon/rapor dosyası
│   ├── final_evidence_pack.md  # dokümantasyon/rapor dosyası
│   ├── final_freeze_manifest.json  # JSON veri artefaktı
│   ├── final_freeze_manifest.md  # dokümantasyon/rapor dosyası
│   ├── final_master_plan_freeze.md  # dokümantasyon/rapor dosyası
│   ├── final_orchestrator_status.json  # JSON veri artefaktı
│   ├── final_orchestrator_status.md  # dokümantasyon/rapor dosyası
│   ├── final_repo_audit.md  # dokümantasyon/rapor dosyası
│   ├── final_sync_matrix.md  # dokümantasyon/rapor dosyası
│   ├── final_sync_matrix_TR.md  # Türkçe doküman karşılığı
│   ├── final_truth_constitution.md  # dokümantasyon/rapor dosyası
│   ├── final_truth_matrix.md  # dokümantasyon/rapor dosyası
│   ├── folder_drift_report.json  # JSON veri artefaktı
│   ├── folder_structure_policy.md  # dokümantasyon/rapor dosyası
│   ├── founders_hub_application.md  # dokümantasyon/rapor dosyası
│   ├── founders_hub_application_TR.md  # Türkçe doküman karşılığı
│   ├── github_policy_report.json  # JSON veri artefaktı
│   ├── go_nogo_signoff_onepager.md  # dokümantasyon/rapor dosyası
│   ├── go_nogo_signoff_onepager_TR.md  # Türkçe doküman karşılığı
│   ├── go_status_matrix.md  # dokümantasyon/rapor dosyası
│   ├── go_status_matrix_TR.md  # Türkçe doküman karşılığı
│   ├── gtm_master_plan.md  # dokümantasyon/rapor dosyası
│   ├── hardening_bundle_summary.json  # JSON veri artefaktı
│   ├── investable_definition.md  # dokümantasyon/rapor dosyası
│   ├── investor_deck.pptx  # artefakt
│   ├── investor_deck_TR.pptx  # artefakt
│   ├── ip_licensing_split.md  # dokümantasyon/rapor dosyası
│   ├── ip_licensing_split_TR.md  # Türkçe doküman karşılığı
│   ├── kernel_fuzz_report.json  # JSON veri artefaktı
│   ├── known_limits_v1.md  # dokümantasyon/rapor dosyası
│   ├── kpi_contract_build30.md  # dokümantasyon/rapor dosyası
│   ├── kpi_pack_v1.md  # dokümantasyon/rapor dosyası
│   ├── kpi_pack_v1_TR.md  # Türkçe doküman karşılığı
│   ├── kpi_report_v1.json  # JSON veri artefaktı
│   ├── latency_baseline.json  # JSON veri artefaktı
│   ├── legal_cleanroom_signoff_internal.md  # dokümantasyon/rapor dosyası
│   ├── legal_ip_pack.md  # dokümantasyon/rapor dosyası
│   ├── license_gate_report.json  # JSON veri artefaktı
│   ├── linkcheck_report.json  # JSON veri artefaktı
│   ├── local_50step_proof_report.json  # JSON veri artefaktı
│   ├── logger_contract.md  # dokümantasyon/rapor dosyası
│   ├── logits_integrity_report.md  # dokümantasyon/rapor dosyası
│   ├── master_closure_matrix.json  # JSON veri artefaktı
│   ├── master_closure_matrix.md  # dokümantasyon/rapor dosyası
│   ├── master_operating_plan.md  # dokümantasyon/rapor dosyası
│   ├── md_lint_report.json  # JSON veri artefaktı
│   ├── model_health.md  # dokümantasyon/rapor dosyası
│   ├── model_health_TR.md  # Türkçe doküman karşılığı
│   ├── model_health_final.md  # dokümantasyon/rapor dosyası
│   ├── offline_assistant_case_study.md  # dokümantasyon/rapor dosyası
│   ├── one_command_full_sop.log  # metin/log artefaktı (tek komut uçtan uca SOP ham logu; her çalıştırmada üzerine yazılır)
│   ├── one_command_full_sop_summary.md  # dokümantasyon/rapor dosyası (tek komut uçtan uca SOP özeti; her çalıştırmada üzerine yazılır)
│   ├── one_pager.md  # dokümantasyon/rapor dosyası
│   ├── one_pager_TR.md  # Türkçe doküman karşılığı
│   ├── owner_matrix.md  # dokümantasyon/rapor dosyası
│   ├── ownership_proof_bundle.json  # JSON veri artefaktı
│   ├── package_smoke_report.json  # JSON veri artefaktı
│   ├── package_validation_report.md  # dokümantasyon/rapor dosyası
│   ├── param_accounting_report.md  # dokümantasyon/rapor dosyası
│   ├── phase2_carryover.md  # dokümantasyon/rapor dosyası
│   ├── pilot_acceptance_signoff.md  # dokümantasyon/rapor dosyası
│   ├── pilot_acceptance_signoff_TR.md  # Türkçe doküman karşılığı
│   ├── pilot_offer_packages.md  # dokümantasyon/rapor dosyası
│   ├── pilot_offer_packages_TR.md  # Türkçe doküman karşılığı
│   ├── pilot_readiness_kit.md  # dokümantasyon/rapor dosyası
│   ├── pilot_readiness_kit_TR.md  # Türkçe doküman karşılığı
│   ├── plot_contract.md  # dokümantasyon/rapor dosyası
│   ├── poc_protocol.md  # dokümantasyon/rapor dosyası
│   ├── poc_protocol_TR.md  # Türkçe doküman karşılığı
│   ├── post_45k_decision_tree.md  # dokümantasyon/rapor dosyası
│   ├── post_train_automation_contract.md  # dokümantasyon/rapor dosyası
│   ├── post_train_autorun_status.json  # JSON veri artefaktı
│   ├── post_train_autorun_status.md  # dokümantasyon/rapor dosyası
│   ├── post_train_state_machine.md  # dokümantasyon/rapor dosyası
│   ├── presentation_readiness_final.md  # dokümantasyon/rapor dosyası
│   ├── proje_zip_rebuild_manifest_v2.json  # JSON veri artefaktı
│   ├── proje_zip_rebuild_manifest_v2.md  # dokümantasyon/rapor dosyası
│   ├── quality_gate_matrix.md  # dokümantasyon/rapor dosyası
│   ├── ram_guard_report.json  # JSON veri artefaktı
│   ├── release_closure_lock_report.json  # JSON veri artefaktı
│   ├── release_closure_note.md  # dokümantasyon/rapor dosyası
│   ├── release_snapshot.md  # dokümantasyon/rapor dosyası
│   ├── release_snapshot_TR.md  # Türkçe doküman karşılığı
│   ├── rented_machine_bringup.md  # dokümantasyon/rapor dosyası
│   ├── repo_closure_scorecard.json  # JSON veri artefaktı
│   ├── repo_closure_scorecard.md  # dokümantasyon/rapor dosyası
│   ├── repo_directory_contract.md  # dokümantasyon/rapor dosyası
│   ├── repo_external_handoff.md  # dokümantasyon/rapor dosyası
│   ├── report_accuracy_audit.md  # dokümantasyon/rapor dosyası
│   ├── report_accuracy_audit_TR.md  # Türkçe doküman karşılığı
│   ├── report_truth_matrix.md  # dokümantasyon/rapor dosyası
│   ├── repro_build_report.json  # JSON veri artefaktı
│   ├── resume_compat_report.json  # JSON veri artefaktı
│   ├── review_checklist.md  # dokümantasyon/rapor dosyası
│   ├── review_checklist_TR.md  # Türkçe doküman karşılığı
│   ├── run_contract.md  # dokümantasyon/rapor dosyası
│   ├── runbook_validation_report.json  # JSON veri artefaktı
│   ├── sales_funnel_90d.md  # dokümantasyon/rapor dosyası
│   ├── sales_funnel_90d_TR.md  # Türkçe doküman karşılığı
│   ├── sanitizer_report.json  # JSON veri artefaktı
│   ├── sbom.cdx.json  # JSON veri artefaktı
│   ├── scoped_external_intake_matrix.json  # JSON veri artefaktı
│   ├── scoped_external_intake_matrix.md  # dokümantasyon/rapor dosyası
│   ├── security_compliance.md  # dokümantasyon/rapor dosyası
│   ├── security_compliance_TR.md  # Türkçe doküman karşılığı
│   ├── smoke_run_report.json  # JSON veri artefaktı
│   ├── snapshot_manifest_dealroom.json  # JSON veri artefaktı
│   ├── snapshot_manifest_main.json  # JSON veri artefaktı
│   ├── source_of_truth_map.md  # dokümantasyon/rapor dosyası
│   ├── stale_script_report.md  # dokümantasyon/rapor dosyası
│   ├── start_gate_operator_decision.json  # JSON veri artefaktı
│   ├── start_gate_operator_decision.md  # dokümantasyon/rapor dosyası
│   ├── start_gate_report.json  # JSON veri artefaktı
│   ├── startup_selfcheck_report.json  # JSON veri artefaktı
│   ├── static_analysis_report.json  # JSON veri artefaktı
│   ├── strategic_value.md  # dokümantasyon/rapor dosyası
│   ├── strategic_value_TR.md  # Türkçe doküman karşılığı
│   ├── support_maintenance_policy.md  # dokümantasyon/rapor dosyası
│   ├── surface_lifecycle_matrix.md  # dokümantasyon/rapor dosyası
│   ├── system_hardware.md  # dokümantasyon/rapor dosyası
│   ├── system_hardware_TR.md  # Türkçe doküman karşılığı
│   ├── system_memory_policy.md  # dokümantasyon/rapor dosyası
│   ├── system_stats.jsonl  # JSONL veri/log artefaktı
│   ├── systems_performance_case_study.md  # dokümantasyon/rapor dosyası
│   ├── target_machine_handoff_manifest.json  # JSON veri artefaktı
│   ├── target_machine_handoff_manifest.md  # dokümantasyon/rapor dosyası
│   ├── teacher_decision_record.md  # dokümantasyon/rapor dosyası
│   ├── teacher_output_license_assessment.md  # dokümantasyon/rapor dosyası
│   ├── technical_snapshot.md  # dokümantasyon/rapor dosyası
│   ├── technical_snapshot_TR.md  # Türkçe doküman karşılığı
│   ├── test_verification_matrix.md  # dokümantasyon/rapor dosyası
│   ├── thermal_baseline.json  # JSON veri artefaktı
│   ├── tokenizer_sync_final_report.md  # dokümantasyon/rapor dosyası
│   ├── train_readiness_decision.json  # JSON veri artefaktı
│   ├── train_readiness_decision.md  # dokümantasyon/rapor dosyası
│   ├── training_readiness_manifest.json  # JSON veri artefaktı
│   ├── unicode_path_guard_report.json  # JSON veri artefaktı
│   ├── update_first_policy.md  # dokümantasyon/rapor dosyası
│   ├── verified_matrix.md  # dokümantasyon/rapor dosyası
│   ├── verified_matrix_TR.md  # Türkçe doküman karşılığı
│   ├── workspace_hygiene_manifest.json  # JSON veri artefaktı
│   ├── workspace_hygiene_manifest.md  # dokümantasyon/rapor dosyası
│   ├── xla_smoke_report.json  # JSON veri artefaktı
│   ├── zip_audit_artifacts.json  # JSON veri artefaktı
│   └── zip_audit_packages.json  # JSON veri artefaktı
├── repro/  # dizin
│   ├── accelerate_default.yaml  # YAML yapılandırma dosyası
│   ├── cuda.lock  # artefakt
│   ├── env.lock  # artefakt
│   ├── pip_freeze.txt  # metin artefaktı
│   ├── python.md  # dokümantasyon/rapor dosyası
│   ├── python_TR.md  # Türkçe doküman karşılığı
│   ├── seed_policy.md  # dokümantasyon/rapor dosyası
│   └── seed_policy_TR.md  # Türkçe doküman karşılığı
├── runbooks/  # dizin
│   ├── README.md  # ana dokümantasyon (EN)
│   ├── chess_4060_24h.md  # dokümantasyon/rapor dosyası
│   ├── chess_4060_24h_TR.md  # Türkçe doküman karşılığı
│   ├── chess_4060_24h_all_on_experimental.md  # dokümantasyon/rapor dosyası
│   └── chess_4060_24h_all_on_experimental_TR.md  # Türkçe doküman karşılığı
├── scripts/  # dizin
│   ├── reports/  # dizin
│   │   ├── model_health.md  # dokümantasyon/rapor dosyası
│   │   └── model_health_TR.md  # Türkçe doküman karşılığı
│   ├── runs/  # dizin
│   │   └── preflight/  # dizin
│   │       └── config_snapshot.json  # JSON veri artefaktı
│   ├── tools/  # dizin
│   │   ├── claim_number_audit.py  # Python modülü/scripti (claim number audit için otomasyon scripti)
│   │   └── denylist_scan_zip.py  # Python modülü/scripti (denylist scan zip için otomasyon scripti)
│   ├── README.md  # ana dokümantasyon (EN)
│   ├── README_TR.md  # Türkçe doküman karşılığı
│   ├── __init__.py  # Python modülü/scripti (scripts paket başlatıcısı ve dışa aktarmalar)
│   ├── apply_github_policy.sh  # kabuk otomasyon scripti
│   ├── benchmarks_internal.py  # Python modülü/scripti (benchmarks internal için otomasyon scripti)
│   ├── bitnet_kernel_benchmark_standalone.py  # Python modülü/scripti (bitnet kernel benchmark standalone için otomasyon scripti)
│   ├── bootstrap_venv.sh  # kabuk otomasyon scripti
│   ├── build_artifacts_release_zip.sh  # kabuk otomasyon scripti
│   ├── build_chess_5080_windows_delivery.py  # Python modülü/scripti (build chess 5080 windows delivery için otomasyon scripti)
│   ├── build_chess_onefile_extension_report.py  # Python modülü/scripti (build chess onefile extension report için otomasyon scripti)
│   ├── build_chess_teaching_contract_report.py  # Python modülü/scripti (build chess teaching contract report için otomasyon scripti)
│   ├── build_chess_training_readiness_report.py  # Python modülü/scripti (build chess training readiness report için otomasyon scripti)
│   ├── build_closure_governance_pack.py  # Python modülü/scripti (build closure governance pack için otomasyon scripti)
│   ├── build_code_truth_audit.py  # Python modülü/scripti (build code truth audit için otomasyon scripti)
│   ├── build_investor_deck.py  # Python modülü/scripti (build investor deck için otomasyon scripti)
│   ├── build_master_closure_matrix.py  # Python modülü/scripti (build master closure matrix için otomasyon scripti)
│   ├── build_max_closure_handoff.py  # Python modülü/scripti (build max closure handoff için otomasyon scripti)
│   ├── build_offline_closure_pack.py  # Python modülü/scripti (build offline closure pack için otomasyon scripti)
│   ├── build_scoped_external_intake_matrix.py  # Python modülü/scripti (build scoped external intake matrix için otomasyon scripti)
│   ├── build_summary_pdf.py  # Python modülü/scripti (build summary pdf için otomasyon scripti)
│   ├── build_target_machine_handoff_bundle.py  # Python modülü/scripti (build target machine handoff bundle için otomasyon scripti)
│   ├── build_train_readiness_contract.py  # Python modülü/scripti (build train readiness contract için otomasyon scripti)
│   ├── build_validation_set.py  # Python modülü/scripti (build validation set için otomasyon scripti)
│   ├── build_workspace_hygiene_manifest.py  # Python modülü/scripti (build workspace hygiene manifest için otomasyon scripti)
│   ├── cfc_moe_tolerance_check.py  # Python modülü/scripti (cfc moe tolerance check için otomasyon scripti)
│   ├── chat.py  # Python modülü/scripti (chat için otomasyon scripti)
│   ├── check_57_matrix.py  # Python modülü/scripti (check 57 matrix için otomasyon scripti)
│   ├── check_doc_claim_consistency.py  # Python modülü/scripti (check doc claim consistency için otomasyon scripti)
│   ├── check_tokenizer_sync.py  # Python modülü/scripti (check tokenizer sync için otomasyon scripti)
│   ├── check_translation_pointer_policy.py  # Python modülü/scripti (check translation pointer policy için otomasyon scripti)
│   ├── checkpoint_restore_drill.py  # Python modülü/scripti (checkpoint restore drill için otomasyon scripti)
│   ├── chess_5080_onefile.py  # Python modülü/scripti (chess 5080 onefile için otomasyon scripti)
│   ├── chess_onefile_contract.py  # Python modülü/scripti (chess onefile contract için otomasyon scripti)
│   ├── clean_runtime_artifacts.sh  # kabuk otomasyon scripti
│   ├── cleanroom_verify.sh  # kabuk otomasyon scripti
│   ├── cleanup_scoped_closure_junk.py  # Python modülü/scripti (cleanup scoped closure junk için otomasyon scripti)
│   ├── data_pipeline.py  # Python modülü/scripti (data pipeline için otomasyon scripti)
│   ├── dealroom_sync.py  # Python modülü/scripti (dealroom sync için otomasyon scripti)
│   ├── docs_inventory.py  # Python modülü/scripti (markdown envanteri ve klasör politika raporlayıcısı)
│   ├── download_tr_tokenizer.py  # Python modülü/scripti (download tr tokenizer için otomasyon scripti)
│   ├── drone_sitl_demo.py  # Python modülü/scripti (drone sitl demo için otomasyon scripti)
│   ├── duplicate_zip_guard.py  # Python modülü/scripti (duplicate zip guard için otomasyon scripti)
│   ├── eval.py  # Python modülü/scripti (eval için otomasyon scripti)
│   ├── export_chess_5080_share.py  # Python modülü/scripti (export chess 5080 share için otomasyon scripti)
│   ├── extract_dataset_refs.py  # Python modülü/scripti (extract dataset refs için otomasyon scripti)
│   ├── failure_budget_drill.py  # Python modülü/scripti (failure budget drill için otomasyon scripti)
│   ├── final_one_shot.sh  # kabuk otomasyon scripti
│   ├── final_orchestrator.py  # Python modülü/scripti (final orchestrator için otomasyon scripti)
│   ├── generate_bench_reports.py  # Python modülü/scripti (generate bench reports için otomasyon scripti)
│   ├── generate_energy_baselines.py  # Python modülü/scripti (generate energy baselines için otomasyon scripti)
│   ├── generate_sbom.py  # Python modülü/scripti (generate sbom için otomasyon scripti)
│   ├── golden_eval.py  # Python modülü/scripti (golden eval için otomasyon scripti)
│   ├── golden_score.py  # Python modülü/scripti (golden score için otomasyon scripti)
│   ├── hardening_bundle.py  # Python modülü/scripti (hardening bundle için otomasyon scripti)
│   ├── hash_manifest_to_json.py  # Python modülü/scripti (hash manifest to json için otomasyon scripti)
│   ├── kaggle_onefile_demo_build30.py  # Python modülü/scripti (kaggle onefile demo build30 için otomasyon scripti)
│   ├── kaggle_onefile_demo_build30_colab_math_fastproof.py  # Python modülü/scripti (kaggle onefile demo build30 colab math fastproof için otomasyon scripti)
│   ├── kaggle_onefile_demo_build30_text_understanding.py  # Python modülü/scripti (kaggle onefile demo build30 text understanding için otomasyon scripti)
│   ├── kaggle_train_compare_build30.py  # Python modülü/scripti (kaggle train compare build30 için otomasyon scripti)
│   ├── linkcheck_gate.py  # Python modülü/scripti (linkcheck gate için otomasyon scripti)
│   ├── logbook_build.py  # Python modülü/scripti (logbook build için otomasyon scripti)
│   ├── mac_simulation.py  # Python modülü/scripti (mac simulation için otomasyon scripti)
│   ├── mathfp_interactive_chat.py  # Python modülü/scripti (mathfp interactive chat için otomasyon scripti)
│   ├── md_build30_sweep.py  # Python modülü/scripti (md build30 sweep için otomasyon scripti)
│   ├── md_integrity_check.py  # Python modülü/scripti (md integrity check için otomasyon scripti)
│   ├── md_quality_gate.py  # Python modülü/scripti (md quality gate için otomasyon scripti)
│   ├── mini_titan_poc.py  # Python modülü/scripti (mini titan poc için otomasyon scripti)
│   ├── mobile_export.py  # Python modülü/scripti (mobile export için otomasyon scripti)
│   ├── nan_kill_test.py  # Python modülü/scripti (nan kill test için otomasyon scripti)
│   ├── offline_4060_demo_train.py  # Python modülü/scripti (offline 4060 demo train için otomasyon scripti)
│   ├── one_command_full_sop.sh  # kabuk otomasyon scripti
│   ├── operator_mode_gate.py  # Python modülü/scripti (operator mode gate için otomasyon scripti)
│   ├── overfit_gate.py  # Python modülü/scripti (overfit gate için otomasyon scripti)
│   ├── plot_training_log.py  # Python modülü/scripti (plot training log için otomasyon scripti)
│   ├── post_train_autorun.py  # Python modülü/scripti (post train autorun için otomasyon scripti)
│   ├── ram_guard.py  # Python modülü/scripti (ram guard için otomasyon scripti)
│   ├── record_dataset_hashes.py  # Python modülü/scripti (record dataset hashes için otomasyon scripti)
│   ├── release_build30.sh  # kabuk otomasyon scripti
│   ├── release_closure_lock.sh  # kabuk otomasyon scripti
│   ├── repro_build_check.py  # Python modülü/scripti (repro build check için otomasyon scripti)
│   ├── resume_compat_check.py  # Python modülü/scripti (resume compat check için otomasyon scripti)
│   ├── run_and_clean_pycache.py  # Python modülü/scripti (komut çalıştırma + garanti pycache temizliği; venv cache temizliği için --include-venv-caches kullan)
│   ├── scaling_audit_math.py  # Python modülü/scripti (scaling audit math için otomasyon scripti)
│   ├── secret_scan.py  # Python modülü/scripti (secret scan için otomasyon scripti)
│   ├── smart_runner.py  # Python modülü/scripti (smart runner için otomasyon scripti)
│   ├── smoke_train_benchmark.py  # Python modülü/scripti (smoke train benchmark için otomasyon scripti)
│   ├── start_gate.py  # Python modülü/scripti (start gate için otomasyon scripti)
│   ├── sync_chess_gui_onefile.py  # Python modülü/scripti (sync chess gui onefile için otomasyon scripti)
│   ├── sync_manifest.py  # Python modülü/scripti (release manifest ve proje-yapısı senkron üreticisi)
│   ├── test_onnx_export.py  # Python modülü/scripti (test onnx export için otomasyon scripti)
│   ├── titan_onnx_stress_test.py  # Python modülü/scripti (titan onnx stress test için otomasyon scripti)
│   ├── titan_preflight.py  # Python modülü/scripti (titan preflight için otomasyon scripti)
│   ├── train_smoke.py  # Python modülü/scripti (train smoke için otomasyon scripti)
│   ├── train_tpu_turbo.py  # Python modülü/scripti (train tpu turbo için otomasyon scripti)
│   ├── unicode_path_guard.py  # Python modülü/scripti (unicode path guard için otomasyon scripti)
│   ├── update_investor_deck.py  # Python modülü/scripti (update investor deck için otomasyon scripti)
│   ├── update_system_hardware.py  # Python modülü/scripti (update system hardware için otomasyon scripti)
│   ├── verify_all.sh  # kabuk otomasyon scripti
│   ├── verify_datasets.py  # Python modülü/scripti (verify datasets için otomasyon scripti)
│   ├── verify_onnx_local.py  # Python modülü/scripti (verify onnx local için otomasyon scripti)
│   ├── version_checker.py  # Python modülü/scripti (version checker için otomasyon scripti)
│   ├── write_cuda_lock.py  # Python modülü/scripti (write cuda lock için otomasyon scripti)
│   ├── xray.py  # Python modülü/scripti (xray için otomasyon scripti)
│   └── zip_denylist_audit.py  # Python modülü/scripti (zip denylist audit için otomasyon scripti)
├── telemetry/  # dizin
│   └── metrics_schema.json  # JSON şema artefaktı
├── tests/  # dizin
│   ├── test_57_matrix_gate.py  # Python modülü/scripti (57 matrix gate için otomatik test modülü)
│   ├── test_agi_cognitive.py  # Python modülü/scripti (agi cognitive için otomatik test modülü)
│   ├── test_architecture_integrity.py  # Python modülü/scripti (architecture integrity için otomatik test modülü)
│   ├── test_build_chess_5080_windows_delivery.py  # Python modülü/scripti (build chess 5080 windows delivery için otomatik test modülü)
│   ├── test_build_chess_onefile_extension_report.py  # Python modülü/scripti (build chess onefile extension report için otomatik test modülü)
│   ├── test_build_chess_teaching_contract_report.py  # Python modülü/scripti (build chess teaching contract report için otomatik test modülü)
│   ├── test_build_chess_training_readiness_report.py  # Python modülü/scripti (build chess training readiness report için otomatik test modülü)
│   ├── test_build_closure_governance_pack.py  # Python modülü/scripti (build closure governance pack için otomatik test modülü)
│   ├── test_build_code_truth_audit.py  # Python modülü/scripti (build code truth audit için otomatik test modülü)
│   ├── test_build_max_closure_handoff.py  # Python modülü/scripti (build max closure handoff için otomatik test modülü)
│   ├── test_build_target_machine_handoff_bundle.py  # Python modülü/scripti (build target machine handoff bundle için otomatik test modülü)
│   ├── test_build_workspace_hygiene_manifest.py  # Python modülü/scripti (build workspace hygiene manifest için otomatik test modülü)
│   ├── test_check_doc_claim_consistency.py  # Python modülü/scripti (check doc claim consistency için otomatik test modülü)
│   ├── test_chess_5080_onefile.py  # Python modülü/scripti (chess 5080 onefile için otomatik test modülü)
│   ├── test_chess_gui_contract.py  # Python modülü/scripti (chess gui contract için otomatik test modülü)
│   ├── test_chess_onefile_curated_suites.py  # Python modülü/scripti (chess onefile curated suites için otomatik test modülü)
│   ├── test_cognitive_extensions.py  # Python modülü/scripti (cognitive extensions için otomatik test modülü)
│   ├── test_comprehensive.py  # Python modülü/scripti (comprehensive için otomatik test modülü)
│   ├── test_continual_adapter.py  # Python modülü/scripti (continual adapter için otomatik test modülü)
│   ├── test_cpp_kernel_loader.py  # Python modülü/scripti (cpp kernel loader için otomatik test modülü)
│   ├── test_dispatcher_extended.py  # Python modülü/scripti (dispatcher extended için otomatik test modülü)
│   ├── test_drone_sitl_demo.py  # Python modülü/scripti (drone sitl demo için otomatik test modülü)
│   ├── test_eval_suites.py  # Python modülü/scripti (eval suites için otomatik test modülü)
│   ├── test_export_chess_5080_share.py  # Python modülü/scripti (export chess 5080 share için otomatik test modülü)
│   ├── test_export_metadata.py  # Python modülü/scripti (export metadata için otomatik test modülü)
│   ├── test_final_orchestrator_cli.py  # Python modülü/scripti (final orchestrator cli için otomatik test modülü)
│   ├── test_kaggle_compare_script.py  # Python modülü/scripti (kaggle compare script için otomatik test modülü)
│   ├── test_kaggle_onefile_colab_math_fastproof.py  # Python modülü/scripti (kaggle onefile colab math fastproof için otomatik test modülü)
│   ├── test_kaggle_onefile_compile_guard.py  # Python modülü/scripti (kaggle onefile compile guard için otomatik test modülü)
│   ├── test_kaggle_onefile_config.py  # Python modülü/scripti (kaggle onefile config için otomatik test modülü)
│   ├── test_kaggle_onefile_feature_coverage.py  # Python modülü/scripti (kaggle onefile feature coverage için otomatik test modülü)
│   ├── test_kaggle_onefile_zero_shot_unseen.py  # Python modülü/scripti (kaggle onefile zero shot unseen için otomatik test modülü)
│   ├── test_kernel_dispatcher.py  # Python modülü/scripti (kernel dispatcher için otomatik test modülü)
│   ├── test_kernel_equivalence.py  # Python modülü/scripti (kernel equivalence için otomatik test modülü)
│   ├── test_kpi_report_cli.py  # Python modülü/scripti (kpi report cli için otomatik test modülü)
│   ├── test_lifelong_safety.py  # Python modülü/scripti (lifelong safety için otomatik test modülü)
│   ├── test_liquid_safeguard.py  # Python modülü/scripti (liquid safeguard için otomatik test modülü)
│   ├── test_mla_regressions.py  # Python modülü/scripti (mla regressions için otomatik test modülü)
│   ├── test_model.py  # Python modülü/scripti (model için otomatik test modülü)
│   ├── test_onnx_custom_op_contract.py  # Python modülü/scripti (onnx custom op contract için otomatik test modülü)
│   ├── test_onnx_export_path.py  # Python modülü/scripti (onnx export path için otomatik test modülü)
│   ├── test_onnx_metadata_hook.py  # Python modülü/scripti (onnx metadata hook için otomatik test modülü)
│   ├── test_orchestrator_swarm_runtime.py  # Python modülü/scripti (orchestrator swarm runtime için otomatik test modülü)
│   ├── test_post_train_autorun_cli.py  # Python modülü/scripti (post train autorun cli için otomatik test modülü)
│   ├── test_scoped_external_tools.py  # Python modülü/scripti (scoped external tools için otomatik test modülü)
│   ├── test_sdk_api.py  # Python modülü/scripti (sdk api için otomatik test modülü)
│   ├── test_sdk_pilot_cli.py  # Python modülü/scripti (sdk pilot cli için otomatik test modülü)
│   ├── test_start_gate.py  # Python modülü/scripti (start gate için otomatik test modülü)
│   ├── test_sync_chess_gui_onefile.py  # Python modülü/scripti (sync chess gui onefile için otomatik test modülü)
│   ├── test_titan_preflight_contract.py  # Python modülü/scripti (titan preflight contract için otomatik test modülü)
│   ├── test_train_loop_sanity.py  # Python modülü/scripti (train loop sanity için otomatik test modülü)
│   ├── test_triad_omega_api.py  # Python modülü/scripti (triad omega api için otomatik test modülü)
│   └── test_world_model_head.py  # Python modülü/scripti (world model head için otomatik test modülü)
├── tokenizer/  # dizin
│   ├── tr/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── drift_report.md  # dokümantasyon/rapor dosyası
│   ├── drift_report_TR.md  # Türkçe doküman karşılığı
│   ├── stats.md  # dokümantasyon/rapor dosyası
│   ├── stats_TR.md  # Türkçe doküman karşılığı
│   └── tokenizer.json  # JSON veri artefaktı
├── tools/  # dizin
│   ├── contracts/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── sandbox/  # dizin
│   │   ├── README.md  # ana dokümantasyon (EN)
│   │   └── README_TR.md  # Türkçe doküman karşılığı
│   ├── abuse_tests.md  # dokümantasyon/rapor dosyası
│   └── abuse_tests_TR.md  # Türkçe doküman karşılığı
├── train/  # dizin
│   ├── __init__.py  # Python modülü/scripti (train paket başlatıcısı ve dışa aktarmalar)
│   ├── continual_adapter.py  # Python modülü/scripti (eğitim için continual learning adaptör yolu)
│   └── train.py  # Python modülü/scripti (ana eğitim döngüsü giriş noktası)
├── training_dynamics/  # dizin
│   ├── cold_vs_warm.md  # dokümantasyon/rapor dosyası
│   └── cold_vs_warm_TR.md  # Türkçe doküman karşılığı
├── utils/  # dizin
│   ├── __init__.py  # Python modülü/scripti (utils paket başlatıcısı ve dışa aktarmalar)
│   ├── dataset_registry.py  # Python modülü/scripti (dataset registry için modül)
│   ├── liquid_safeguard.py  # Python modülü/scripti (liquid safeguard için modül)
│   ├── logger.py  # Python modülü/scripti (logger için modül)
│   └── safety.py  # Python modülü/scripti (safety için modül)
├── .gitignore  # git ignore politikası
├── AGENTS.md  # dokümantasyon/rapor dosyası
├── CHANGELOG.md  # dokümantasyon/rapor dosyası
├── CHANGELOG_TR.md  # Türkçe doküman karşılığı
├── CHESS_5080_POC_INTERNAL.md  # dokümantasyon/rapor dosyası
├── CHESS_5080_POC_INTERNAL_TR.md  # Türkçe doküman karşılığı
├── CITATION.cff  # atıf metaverisi
├── CONTRIBUTING.md  # dokümantasyon/rapor dosyası
├── CONTRIBUTING_TR.md  # Türkçe doküman karşılığı
├── DECISIONS.md  # dokümantasyon/rapor dosyası
├── DECISIONS_TR.md  # Türkçe doküman karşılığı
├── Dockerfile  # container build baseline
├── IMPLEMENTATION_PLAN.md  # dokümantasyon/rapor dosyası
├── IMPLEMENTATION_PLAN_TR.md  # Türkçe doküman karşılığı
├── INTERNAL_AGI_GAP.md  # dokümantasyon/rapor dosyası
├── INTERNAL_AGI_GAP_TR.md  # Türkçe doküman karşılığı
├── LICENSE  # lisans koşulları (EN)
├── LICENSE_TR  # lisans koşulları (TR)
├── MISSION.md  # dokümantasyon/rapor dosyası
├── MISSION_TR.md  # Türkçe doküman karşılığı
├── MODEL_CARD.md  # dokümantasyon/rapor dosyası
├── MODEL_CARD_TR.md  # Türkçe doküman karşılığı
├── MODEL_LICENSE.md  # dokümantasyon/rapor dosyası
├── MODEL_LICENSE_TR.md  # Türkçe doküman karşılığı
├── OFFLINE_4060_DEMO.md  # dokümantasyon/rapor dosyası
├── PITCH.md  # dokümantasyon/rapor dosyası
├── PITCH_TR.md  # Türkçe doküman karşılığı
├── README.md  # ana dokümantasyon (EN)
├── README_CHECKLIST.md  # dokümantasyon/rapor dosyası
├── README_CHECKLIST_TR.md  # Türkçe doküman karşılığı
├── README_SUMMARY.md  # dokümantasyon/rapor dosyası
├── README_SUMMARY.pdf  # artefakt
├── README_SUMMARY_TR.md  # Türkçe doküman karşılığı
├── README_SUMMARY_TR.pdf  # artefakt
├── README_TR.md  # Türkçe doküman karşılığı
├── SDK_GUIDE.md  # dokümantasyon/rapor dosyası
├── SDK_GUIDE_TR.md  # Türkçe doküman karşılığı
├── SECURITY.md  # dokümantasyon/rapor dosyası
├── SECURITY_TR.md  # Türkçe doküman karşılığı
├── START_HERE.md  # dokümantasyon/rapor dosyası
├── TASK.md  # dokümantasyon/rapor dosyası
├── TASK_TR.md  # Türkçe doküman karşılığı
├── TECHNICAL_REPORT.md  # dokümantasyon/rapor dosyası
├── TECHNICAL_REPORT_TR.md  # Türkçe doküman karşılığı
├── TRAINING_PLAN.md  # dokümantasyon/rapor dosyası
├── TRAINING_PLAN_TR.md  # Türkçe doküman karşılığı
├── TROUBLESHOOTING.md  # dokümantasyon/rapor dosyası
├── TROUBLESHOOTING_TR.md  # Türkçe doküman karşılığı
├── USAGE_GUIDE.md  # dokümantasyon/rapor dosyası
├── USAGE_GUIDE_TR.md  # Türkçe doküman karşılığı
├── USE_POLICY.md  # dokümantasyon/rapor dosyası
├── USE_POLICY_TR.md  # Türkçe doküman karşılığı
├── V2_BACKLOG_SEED.md  # dokümantasyon/rapor dosyası
├── WHITE_PAPER_LIQUIDROUTER.md  # dokümantasyon/rapor dosyası
├── WHITE_PAPER_LIQUIDROUTER_TR.md  # Türkçe doküman karşılığı
├── pyproject.toml  # proje metaverisi
├── requirements.txt  # metin artefaktı
├── run.sh  # kabuk otomasyon scripti
├── snake_demo.py  # Python modülü/scripti (snake demo için modül)
└── zero_touch_start.sh  # kabuk otomasyon scripti
```
