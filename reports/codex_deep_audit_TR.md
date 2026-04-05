# Codex Derin Denetim — MertFormer Titan (v1.0 Build 30)
**Repo:** `.`
**Denetim Tarihi (yerel):** 2026-02-06
**Denetim Tipi:** Kod + Dokümantasyon + Çalıştirmali Doğrulama (offline-first)

## Kısa Özet (6-10 satır)
Bu repo; “mobile-first / NPU hedefli” bir LLM mimarisi için BitLinear (düşük-bit ağırlık simülasyonu), MLA etiketli GQA attention, MoE + LiquidRouter routing ve Liquid/CfC dinamik katmanlarini bir araya getiren kapsamlı bir Ar-Ge + engineering PoC çalışmasi. Mimarinin ve eğitim iskeletinin çalıştigi doğrulandi: secret scan PASS, preflight PASS, operator-mode gate PASS, pytest PASS (`139 passed, 3 skipped`) ve `run.sh --test` offline-first PASS. Dataset kaynak/lisans envanteri artık kodla hizalı (inventory + LICENSES + snapshot/hash registry mevcut; `datasets/hashes.json` pinlenmis revision + manifest fingerprint içeriyor); bununla birlikte `bigcode/the-stack-v2` gibi gated/karma lisanslı kaynaklar kurumsal eğitimde hukuki onay süreci gerektirir. Dokümantasyon geniş ama performans/NPU hız/enerji gibi rakamlar şu an “hedef/iddia” seviyesinde (reprodusibl checkpoint + benchmark raporu yok). Seviye: **Engineering PoC / Ar-Ge (Pre-Training)**; “review-ready” (mühendislik incelemesi ve eğitime başlamak için) ama “production-ready” değil (eğitim + benchmark + cihaz profili eksik). Git geçmişi tek author gösteriyor; en olası ekip: 1 kişi (belirsizlik payı: tool/yardımcı katkılar Git’te gorunmeyebilir).

---

## 1) Bağlam ve Kapsam
Amaç: Projeyi tarafsız, kanıta dayalı ve üçüncü kişilerin de doğrulayabileceği şekilde değerlendirmek (mimari, kod kalitesi, pipeline, doğrulama); pazarlama iddialarını “doğru kabul etmeden” ayrıştırmak.
Kapsam dışı: Tam eğitim koşmak (günler/haftalar) veya gerçek cihaz/benchmark validasyonu (reprodusibl checkpoint + ölçüm yok).

**Etiketleme Kuralları (Şeffaflık):**
- **Verified (Code):** kodda doğrudan var ve izlenebilir
- **Verified (Run):** bu ortamda çalıştirildi, sonucu görüldü (exit status / test sonucu)
- **Claim (Docs):** dokümanda iddia var ama kod/run ile kanıtlı değil
- **Assumption:** mantıklı çıkarım ama kesin kanıt değil

---

## 2) Repo Snapshot (Metrikler)
### 2.1 Doğrulama Baseline (Verified (Run))
- Host: MacBook Air (Apple Silicon M4), 16 GB RAM, macOS 26.2 (arm64) (`reports/system_hardware.md`)
- Python (baseline): **3.11.14** (`.titan-venv/bin/python -V`)
- Varsayılan: **offline-first** (`TITAN_OFFLINE=1`)
- Tek komut doğrulama: `bash scripts/verify_all.sh`

### 2.2 İçerik (Verified (Run))
Metrikler `git ls-files` üzerinden (yalnızca tracked; yerel artefaktlar hariç):
- Tracked dosya toplam: **254**
- Markdown: **126**, Python: **89**, JSON: **8**, JSONL: **2**, YAML: **9**, YML: **1**, TOML: **1**, Shell: **3**, TXT: **3**, Other: **12**
- `scripts/*.py`: **35**
- `tests/*.py`: **8**

Metin satırları (tracked, kabaca uzanti bazında; binary dosyalar hariç):
- Python: **14,610** satır
- Markdown: **7,009** satır

En büyük tracked dosyalar (örnek; Verified (Run)):
- `assets/synaptic_map.png` (~0.93 MB)
- `assets/header.png` (~0.86 MB)
- `README.md` / `README_TR.md` (~78 KB)
- `train/train.py` (~68 KB)

### 2.3 Git Analizi (Verified (Run))
- Commit sayısı: **103**
- Git’e göre author sayısı: **1** (`git shortlog -sne HEAD`)
- Görünen commit aralığı: **2026-02-02** ile **2026-02-06**

Yorum (Assumption):
- Tek author + tutarlı “yazım stili” nedeniyle en olası: 1 ana geliştirici (tools/assistant kullanımi Git’te gorunmeyebilir).

---

## 3) Mimari Genel Bakış (Ne var?)
### 3.1 Modül Haritası (Verified (Code))
- **Konfig:** `config/config.py` (global `cfg`, overlay’ler, validasyon)
- **Model:** `model/transformers.py` (embedding, blocks, KV-cache, generate)
- **Katmanlar:** `layers/`
  - `bitlinear.py`: aktivasyon quant + ternary weight quant (STE) + opsiyonel Triton kernel
  - `mla.py`: attention + RoPE + KV-cache + GQA repeat logic
  - `moe.py`: MoE dispatch + LiquidRouter (stateful) + aux loss + collapse handling
  - `liquid.py`: CfC/LiquidCell + (opsiyonel) JIT yol + residual/norm
  - `qinn.py`: opsiyonel unitary layer (Cayley transform)
  - `mertformer_block.py`: block kompozisyonu (Norm -> MLA etiketli GQA -> opsiyonel Liquid -> FFN/MoE -> opsiyonel QINN)
- **Eğitim:** `train/train.py` (Accelerate, curriculum, offline/online distillation, checkpoint, export)
- **Ops/Scripts:** `run.sh`, `scripts/bootstrap_venv.sh`, `scripts/verify_all.sh`, `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `scripts/overfit_gate.py`, `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`
- **Dataset Uyum:** `scripts/extract_dataset_refs.py` (inventory), `scripts/record_dataset_hashes.py` (snapshot/hash registry), `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`
- **SDK/CLI:** `mertformer_sdk/` (API + CLI wrapper)
- **Orchestrator (opsiyonel):** `orchestrator/` (Memory/RAG/Web/Audio/SenseEngine; bazı optional dependency’ler)

### 3.2 Çekirdek Fikirler — Durum (kısa)
- BitNet/ternary weights: **Verified (Code)** ama *on-the-fly simülasyon* (bitpacking ile gerçek bellek kazancı “default” değil)
- MoE + LiquidRouter: **Verified (Code)**
- Liquid/CfC dynamics: **Verified (Code)**
- KV-cache + generate: **Verified (Code)**
- “Mobile/NPU performans rakamları”: **Claim (Docs)** (reprodusibl benchmark/device profile yok)

---

## 4) Build/Run Pipeline (Nasıl çalışir?)
### 4.1 `run.sh` (Verified (Code))
Yüksek seviye akış:
1. Öncelik `.titan-venv/bin/python` (yoksa opsiyonel bootstrap: `scripts/bootstrap_venv.sh`)
2. `.env` yükler (secret’lar ciktiya yazılmaz; offline-first default)
3. `scripts/version_checker.py` (yerel tutarlılık)
4. WandB login sadece `TITAN_OFFLINE=0` ve `TITAN_WANDB=1` iken
5. `scripts/titan_preflight.py` çalıştirir (offline: HF bağlantı kontrolü SKIP)
6. `--test/--verify`: preflight’ten sonra çıkar
7. Normal mod: eğitim pipeline’i **TITAN_OFFLINE=1** iken default olarak kapalı (safety gate)

### 4.2 Ops Notu: Venv Relocation (Verified (Code)+Assumption)
Repo’daki `/.titan-venv` taşınmış/relocated görünüyor olabilir (bazı venv CLI’lari shebang problemi yaşayabilir).
`python -m pip` / `python -m wandb` ile `run.sh` daha sağlam; ama `.titan-venv/bin/*` altındaki direkt CLI çalıştirmalari yine de bozulabilir (Assumption; kurulum yoluna bağlı).

---

## 5) Çalıştirmali Doğrulama (Run Sonuçları)
### 5.1 Sonuç Tablosu (Verified (Run))
| Adım | Komut | Sonuç | Not |
| --- | --- | --- | --- |
| Secret Scan (tracked) | `./.titan-venv/bin/python scripts/secret_scan.py` | **PASS (Exit 0)** | Tracked dosyalarda secret pattern yok |
| Unit testler | `./.titan-venv/bin/python -m pytest -q` | **PASS** | `139 passed, 3 skipped` |
| Preflight (offline) | `TITAN_OFFLINE=1 ./.titan-venv/bin/python scripts/titan_preflight.py` | **PASS (Exit 0)** | HF/WandB bağlantı kontrolleri offline default’ta SKIP; token değeri çıkmaz |
| Operator Gate (safe, offline) | `./.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | **PASS (Exit 0)** | Overfit gate PASS (loss düşüyor), Golden Samples PASS, Benchmarks “ready” |
| `run.sh --test` offline-first | `TITAN_OFFLINE=1 bash run.sh --test` | **PASS (Exit 0)** | External login/download yok; preflight’ten sonra çıkar |

### 5.2 Pytest Uyarıları (Verified (Run))
- `torch.jit.script` DeprecationWarning (Torch): `layers/liquid.py` içindeki JIT yolu
- Torch ONNX export: `dynamic_axes`/dynamo exporter uyarılari (fail değil)

---

## 6) Doküman İddiaları vs Kod Gerçeği (Örnekler)
| Başlık | Claim (Docs) | Kanıt | Durum |
| --- | --- | --- | --- |
| “Pre-Training / Unverified” | Evet | `README.md`, `MODEL_CARD.md` | Verified (Docs) |
| 18 layer / Titan config | Evet | `config/config.py` + model `cfg.num_layers` kadar block kuruyor | Verified (Code) |
| BitNet 1.58-bit “weights” | Evet | `layers/bitlinear.py` (forward’da ternary quant) | Verified (Code) (simülasyon) |
| MoE (8 expert, top-2) | Evet | `config/config.py`, `layers/moe.py` | Verified (Code) |
| Offline distillation | Evet | `train/train.py`, `orchestrator/distillation_manager.py` | Verified (Code) |
| Dataset lineage/lisanslar “tam” | dolaylı | `scripts/extract_dataset_refs.py` → `datasets/inventory*` + `datasets/SOURCES*.md` + `datasets/LICENSES*.md` + `datasets/hashes.json` | **Verified (Code)+Verified (Run)** (registry mevcut) |
| NPU hız/enerji rakamları | Evet | repo’da ölçüm/bmark raporu yok | Claim (Docs) |

---

## 7) Bulgular (Tarafsız, Öncelikli)
### P0 — Eğitim sonrası kanıtlar eksik (Checkpoint/Benchmark/Cihaz ölçümleri)
**Gözlem (Verified (Code)):** Repo bilerek “Pre-Training”; reprodusibl training checkpoint’i ve benchmark output’u yok.
**Risk:** performans/NPU/enerji rakamları hedef olarak kalır; teknik değerlendirme pipeline-merkezli olur.
**Öneri:** hedef donanımda ilk eğitimi koş + `scripts/benchmarks_internal.py` çıktılarını `reports/benchmarks/` altına ekle; README’deki hedefleri “Verified”e çevir.

### P0 — Uyum süreci (gated / karma lisanslı kaynaklar)
**Gözlem (Verified (Code)):** `datasets/LICENSES*.md` ve `datasets/hashes.json` mevcut; ancak `bigcode/the-stack-v2` gated ve karma upstream lisanslı.
**Risk:** kurumsal/denetimli eğitim için hukuki/uyum onayı gerekir.
**Öneri:** ic policy + sign-off ile belgelemek veya daha kolay lisanslı veri kaynaklariyla devam etmek.

### P1 — Platform: `torch.jit.script` Deprecation (Torch)
**Gözlem (Verified (Run)):** test koşularında uyarılar var; JIT uzun vadede “legacy”.
**Etkisi:** orta vadede migrasyon ihtiyacı (örneğin `torch.compile` / `torch.export`).
**Öneri:** JIT yolunu opsiyonel tut; yerine geçiş için roadmap yaz.

### P1 — Konfig/Dayanıklılık: GQA/KV-head hatalı ayarlara karşı “hard fail” (şu an korumalı)
**Gözlem (Verified (Code)+Verified (Run)):** Guard olmadan `num_kv_heads > num_heads` invalid shape üretebilir.
**Durum:** **Denetimde düzeltildi**: `layers/mla.py` guard + test fixture `cfg.num_kv_heads` patchli -> pytest yeşil.
**Öneri:** bu validasyonu ek olarak `config/config.py` tarafında da merkezi hale getir (SSOT).

### P1 — Secret hijyeni: “loglarda token yok” bir süreç kapısı (kısmen çözüldü)
**Gözlem (Verified (Code)):** preflight artık token fragmanı loglamiyor (redacted).
**Durum:** **Denetimde düzeltildi** (kod + eski log/doc snippet’leri redacted).
**Öneri:** CI/Operator gate’te secret scanner: `logs/` + `README*` + `reports/`.

### P2 — Preflight: network check “uzun süreli transfer” baslatmamali
**Gözlem (Verified (Run)):** Streaming-sample download (HF backend’e göre) arkaplanda transfer tetikleyip process’in “ALL GREEN”ten sonra bile bitmemesine yol açabilir.
**Durum:** **Denetimde düzeltildi**: default check metadata bazlı; streaming-sample opt-in (`TITAN_PREFLIGHT_STREAM_SAMPLE=1`).
**Öneri:** preflight’i bilerek hafif tut; network check’ler için timeout uygula.

### P2 — Import side-effect / Global state
**Gözlem (Verified (Code)):** global `cfg` ve bazı import side-effect’ler (auto-tuning/print).
**Etkisi:** SDK/test/orchestrator entegrasyonunda sürpriz davranışlar.
**Öneri:** side-effect’leri `main()`/açık init fonksiyonları arkasına taşı; config’i her run için immutable snapshot yap.

---

## 8) Olgunluk (Kategori)
**Siniflama:** **Engineering PoC / Ar-Ge (Pre-Training), review-ready**
Gerekçe (Verified (Code)+Verified (Run)):
- **Artışı:** mimari bloklar + training skeleton + operator gate + SDK var; offline-first doğrulama pipeline’i yeşil.
- **Eksisi:** reprodusibl training checkpoint/benchmark raporu yok; hedef rakamlar eğitim/benchmark olmadan hedef olarak kalır.

Özet: Teknik yapilabilirlik + pipeline demo + mühendislik incelemesi için güçlü; eğitim+benchmark olmadan “production-ready” değil.

---

## 9) Ekip Tahmini (Kaç kişi?)
**Kanıt (Verified (Run)):** Git geçmişi **1 author** gösteriyor (103 commit).
**En olası:** **1 kişi** ana geliştirici.
**Alternatif (Assumption):** 1 ana geliştirici + ara ara reviewer/tool (Git’te gorunmeyebilir).
**Güvenli cümle:** “En az 1, çok büyük olasılıkla 1”.

---

## 10) Net Öneriler (2 hafta / 2 ay)
### 10.1 2 haftada (P0/P1)
- Hedef donanımda ilk eğitim koşusunu yap (pinlenmis datasetler: `datasets/hashes.json`).
- Benchmark çıktılarını `reports/benchmarks/` altına al; README hedef -> Verified güncelle.
- Gated/karma lisanslı kaynaklar için uyum sign-off’u dokümante et (veya kaynak değiştir).

### 10.2 2 ayda (Pilot hazırlığı)
- Tekrarlanabilir eğitim koşuları (resume/restore drill) + “run manifest” + sabit seed.
- Cihaz profili (NPU/CPU) + enerji/latency ölçüm protokolü.
- Dış review checklist'ine göre son seviye değerlendirme (`reports/review_checklist.md`).

---

## Ek A — En önemli giriş noktalar (entry-points)
- Preflight: `scripts/titan_preflight.py`
- Operator Gate: `scripts/operator_mode_gate.py`
- Training: `train/train.py` (Accelerate ile)
- Data Pipeline: `scripts/data_pipeline.py` (büyük, network-heavy)
- SDK CLI: `mertformer_sdk/cli.py` (entry: `mertformer`)

## Ek B — Lokal artefakt notu
Bazı büyük dosyalar (`*.zip`, `tokenizer/tr/*`) `.gitignore` ile ignore ediliyor.
Test/Export sırasında büyük artefaktlar oluşabilir (örneğin `*.onnx.data`).
