# Codex Derin Denetim — MertFormer Titan (v1.0 Build 27)
**Repo:** `/Users/mertyunlu/Desktop/NİHAİ`  
**Denetim Tarihi (lokal):** 2026-02-06  
**Denetim Tipi:** Kod + Dokumantasyon + Calistirmali Dogrulama (offline-first)

## Kisa Ozet (6-10 satir)
Bu repo; “mobile-first / NPU hedefli” bir LLM mimarisi icin BitLinear (dusuk-bit agirlik simulasyonu), MLA attention, MoE + LiquidRouter routing ve Liquid/CfC dinamik katmanlarini bir araya getiren kapsamli bir Ar-Ge + engineering PoC calismasi. Mimarinin ve egitim iskeletinin calistigi dogrulandi: secret scan PASS, preflight PASS, operator-mode gate PASS, pytest PASS (`21 passed, 4 skipped`) ve `run.sh --test` offline-first PASS. Dataset kaynak/lisans envanteri artik kodla hizali (inventory + LICENSES + snapshot/hash registry mevcut; `datasets/hashes.json` pinlenmis revision + manifest fingerprint iceriyor); bununla birlikte `bigcode/the-stack-v2` gibi gated/karma lisansli kaynaklar kurumsal egitimde hukuki onay sureci gerektirir. Dokumantasyon genis ama performans/NPU hiz/enerji gibi rakamlar su an “hedef/iddia” seviyesinde (reprodusibl checkpoint + benchmark raporu yok). Seviye: **Engineering PoC / Ar-Ge (Pre-Training)**; “review-ready” (muhendislik incelemesi ve egitime baslamak icin) ama “production-ready” degil (egitim + benchmark + cihaz profili eksik). Git gecmisi tek author gosteriyor; en olasi ekip: 1 kisi (belirsizlik payi: tool/yardimci katkilar Git’te gorunmeyebilir).

---

## 1) Baglam ve Kapsam
Amac: Projeyi tarafsiz, kanita dayali ve ucuncu kisilerin de dogrulayabilecegi sekilde degerlendirmek (mimari, kod kalitesi, pipeline, dogrulama); pazarlama iddialarini “dogru kabul etmeden” ayristirmak.  
Kapsam disi: Tam egitim kosmak (gunler/haftalar) veya gercek cihaz/benchmark validasyonu (reprodusibl checkpoint + olcum yok).

**Etiketleme Kurallari (Seffaflik):**
- **Verified (Code):** kodda dogrudan var ve izlenebilir
- **Verified (Run):** bu ortamda calistirildi, sonucu goruldu (exit status / test sonucu)
- **Claim (Docs):** dokumanda iddia var ama kod/run ile kanitli degil
- **Assumption:** mantikli cikarim ama kesin kanit degil

---

## 2) Repo Snapshot (Metrikler)
### 2.1 Dogrulama Baseline (Verified (Run))
- Host: MacBook Air (Apple Silicon M4), 16 GB RAM, macOS 26.2 (arm64) (`reports/system_hardware.md`)
- Python (baseline): **3.11.14** (`.titan-venv/bin/python -V`)
- Varsayilan: **offline-first** (`TITAN_OFFLINE=1`)
- Tek komut dogrulama: `bash scripts/verify_all.sh`

### 2.2 Icerik (Verified (Run))
Metrikler `git ls-files` uzerinden (yalnizca tracked; yerel artefaktlar haric):
- Tracked dosya toplam: **254**
- Markdown: **126**, Python: **89**, JSON: **8**, JSONL: **2**, YAML: **9**, YML: **1**, TOML: **1**, Shell: **3**, TXT: **3**, Other: **12**
- `scripts/*.py`: **35**
- `tests/*.py`: **8**

Metin satirlari (tracked, kabaca uzanti bazinda; binary dosyalar haric):
- Python: **14,610** satir
- Markdown: **7,009** satir

En buyuk tracked dosyalar (ornek; Verified (Run)):
- `assets/synaptic_map.png` (~0.93 MB)
- `assets/header.png` (~0.86 MB)
- `README.md` / `README_TR.md` (~78 KB)
- `train/train.py` (~68 KB)

### 2.3 Git Analizi (Verified (Run))
- Commit sayisi: **103**
- Git’e gore author sayisi: **1** (`git shortlog -sne HEAD`)
- Gorunen commit araligi: **2026-02-02** ile **2026-02-06**

Yorum (Assumption):
- Tek author + tutarli “yazim stili” nedeniyle en olasi: 1 ana gelistirici (tools/assistant kullanimi Git’te gorunmeyebilir).

---

## 3) Mimari Genel Bakis (Ne var?)
### 3.1 Modul Haritasi (Verified (Code))
- **Konfig:** `config/config.py` (global `cfg`, overlay’ler, validasyon)
- **Model:** `model/transformers.py` (embedding, blocks, KV-cache, generate)
- **Katmanlar:** `layers/`  
  - `bitlinear.py`: aktivasyon quant + ternary weight quant (STE) + opsiyonel Triton kernel
  - `mla.py`: attention + RoPE + KV-cache + GQA repeat logic
  - `moe.py`: MoE dispatch + LiquidRouter (stateful) + aux loss + collapse handling
  - `liquid.py`: CfC/LiquidCell + (opsiyonel) JIT yol + residual/norm
  - `qinn.py`: opsiyonel unitary layer (Cayley transform)
  - `mertformer_block.py`: block kompozisyonu (Norm -> MLA -> opsiyonel Liquid -> FFN/MoE -> opsiyonel QINN)
- **Egitim:** `train/train.py` (Accelerate, curriculum, offline/online distillation, checkpoint, export)
- **Ops/Scripts:** `run.sh`, `scripts/bootstrap_venv.sh`, `scripts/verify_all.sh`, `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `scripts/overfit_gate.py`, `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`
- **Dataset Uyum:** `scripts/extract_dataset_refs.py` (inventory), `scripts/record_dataset_hashes.py` (snapshot/hash registry), `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`
- **SDK/CLI:** `mertformer_sdk/` (API + CLI wrapper)
- **Orchestrator (opsiyonel):** `orchestrator/` (Memory/RAG/Web/Audio/SenseEngine; bazi optional dependency’ler)

### 3.2 Cekirdek Fikirler — Durum (kisa)
- BitNet/ternary weights: **Verified (Code)** ama *on-the-fly simulasyon* (bitpacking ile gercek bellek kazanci “default” degil)
- MoE + LiquidRouter: **Verified (Code)**
- Liquid/CfC dynamics: **Verified (Code)**
- KV-cache + generate: **Verified (Code)**
- “Mobile/NPU performans rakamlari”: **Claim (Docs)** (reprodusibl benchmark/device profile yok)

---

## 4) Build/Run Pipeline (Nasil calisir?)
### 4.1 `run.sh` (Verified (Code))
Yuksek seviye akis:
1. Oncelik `.titan-venv/bin/python` (yoksa opsiyonel bootstrap: `scripts/bootstrap_venv.sh`)
2. `.env` yukler (secret’lar ciktiya yazilmaz; offline-first default)
3. `scripts/version_checker.py` (lokal tutarlilik)
4. WandB login sadece `TITAN_OFFLINE=0` ve `TITAN_WANDB=1` iken
5. `scripts/titan_preflight.py` calistirir (offline: HF baglanti kontrolu SKIP)
6. `--test/--verify`: preflight’ten sonra cikar
7. Normal mod: egitim pipeline’i **TITAN_OFFLINE=1** iken default olarak kapali (safety gate)

### 4.2 Ops Notu: Venv Relocation (Verified (Code)+Assumption)
Repo’daki `/.titan-venv` tasinmis/relocated gorunuyor olabilir (bazi venv CLI’lari shebang problemi yasayabilir).  
`python -m pip` / `python -m wandb` ile `run.sh` daha saglam; ama `.titan-venv/bin/*` altindaki direkt CLI calistirmalari yine de bozulabilir (Assumption; kurulum yoluna bagli).

---

## 5) Calistirmali Dogrulama (Run Sonuclari)
### 5.1 Sonuc Tablosu (Verified (Run))
| Adim | Komut | Sonuc | Not |
| --- | --- | --- | --- |
| Secret Scan (tracked) | `./.titan-venv/bin/python scripts/secret_scan.py` | **PASS (Exit 0)** | Tracked dosyalarda secret pattern yok |
| Unit testler | `./.titan-venv/bin/python -m pytest -q` | **PASS** | `21 passed, 4 skipped` |
| Preflight (offline) | `TITAN_OFFLINE=1 ./.titan-venv/bin/python scripts/titan_preflight.py` | **PASS (Exit 0)** | HF/WandB baglanti kontrolleri offline default’ta SKIP; token degeri cikmaz |
| Operator Gate (safe, offline) | `./.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | **PASS (Exit 0)** | Overfit gate PASS (loss dusuyor), Golden Samples PASS, Benchmarks “ready” |
| `run.sh --test` offline-first | `TITAN_OFFLINE=1 bash run.sh --test` | **PASS (Exit 0)** | External login/download yok; preflight’ten sonra cikar |

### 5.2 Pytest Uyarilari (Verified (Run))
- `torch.jit.script` DeprecationWarning (Torch): `layers/liquid.py` icindeki JIT yolu
- Torch ONNX export: `dynamic_axes`/dynamo exporter uyarilari (fail degil)

---

## 6) Dokuman Iddialari vs Kod Gercegi (Ornekler)
| Baslik | Claim (Docs) | Kanit | Durum |
| --- | --- | --- | --- |
| “Pre-Training / Unverified” | Evet | `README.md`, `MODEL_CARD.md` | Verified (Docs) |
| 18 layer / Titan config | Evet | `config/config.py` + model `cfg.num_layers` kadar block kuruyor | Verified (Code) |
| BitNet 1.58-bit “weights” | Evet | `layers/bitlinear.py` (forward’da ternary quant) | Verified (Code) (simulasyon) |
| MoE (8 expert, top-2) | Evet | `config/config.py`, `layers/moe.py` | Verified (Code) |
| Offline distillation | Evet | `train/train.py`, `orchestrator/distillation_manager.py` | Verified (Code) |
| Dataset lineage/lisanslar “tam” | dolayli | `scripts/extract_dataset_refs.py` → `datasets/inventory*` + `datasets/SOURCES*.md` + `datasets/LICENSES*.md` + `datasets/hashes.json` | **Verified (Code)+Verified (Run)** (registry mevcut) |
| NPU hiz/enerji rakamlari | Evet | repo’da olcum/bmark raporu yok | Claim (Docs) |

---

## 7) Bulgular (Tarafsiz, Oncelikli)
### P0 — Egitim sonrasi kanitlar eksik (Checkpoint/Benchmark/Cihaz olcumleri)
**Gozlem (Verified (Code)):** Repo bilerek “Pre-Training”; reprodusibl training checkpoint’i ve benchmark output’u yok.  
**Risk:** performans/NPU/enerji rakamlari hedef olarak kalir; teknik degerlendirme pipeline-merkezli olur.  
**Oneri:** hedef donanimda ilk egitimi kos + `scripts/benchmarks_internal.py` ciktilarini `reports/benchmarks/` altina ekle; README’deki hedefleri “Verified”e cevir.

### P0 — Uyum sureci (gated / karma lisansli kaynaklar)
**Gozlem (Verified (Code)):** `datasets/LICENSES*.md` ve `datasets/hashes.json` mevcut; ancak `bigcode/the-stack-v2` gated ve karma upstream lisansli.  
**Risk:** kurumsal/denetimli egitim icin hukuki/uyum onayi gerekir.  
**Oneri:** ic policy + sign-off ile belgelemek veya daha kolay lisansli veri kaynaklariyla devam etmek.

### P1 — Platform: `torch.jit.script` Deprecation (Torch)
**Gozlem (Verified (Run)):** test kosularinda uyarilar var; JIT uzun vadede “legacy”.  
**Etkisi:** orta vadede migrasyon ihtiyaci (ornegin `torch.compile` / `torch.export`).  
**Oneri:** JIT yolunu opsiyonel tut; yerine gecis icin roadmap yaz.

### P1 — Konfig/Dayaniklilik: GQA/KV-head hatali ayarlara karsi “hard fail” (su an korumali)
**Gozlem (Verified (Code)+Verified (Run)):** Guard olmadan `num_kv_heads > num_heads` invalid shape uretebilir.  
**Durum:** **Denetimde duzeltildi**: `layers/mla.py` guard + test fixture `cfg.num_kv_heads` patchli -> pytest yesil.  
**Oneri:** bu validasyonu ek olarak `config/config.py` tarafinda da merkezi hale getir (SSOT).

### P1 — Secret hijyeni: “loglarda token yok” bir surec kapisi (kismen cozuldu)
**Gozlem (Verified (Code)):** preflight artik token fragmani loglamiyor (redacted).  
**Durum:** **Denetimde duzeltildi** (kod + eski log/doc snippet’leri redacted).  
**Oneri:** CI/Operator gate’te secret scanner: `logs/` + `README*` + `reports/`.

### P2 — Preflight: network check “uzun sureli transfer” baslatmamali
**Gozlem (Verified (Run)):** Streaming-sample download (HF backend’e gore) arkaplanda transfer tetikleyip process’in “ALL GREEN”ten sonra bile bitmemesine yol acabilir.  
**Durum:** **Denetimde duzeltildi**: default check metadata bazli; streaming-sample opt-in (`TITAN_PREFLIGHT_STREAM_SAMPLE=1`).  
**Oneri:** preflight’i bilerek hafif tut; network check’ler icin timeout uygula.

### P2 — Import side-effect / Global state
**Gozlem (Verified (Code)):** global `cfg` ve bazi import side-effect’ler (auto-tuning/print).  
**Etkisi:** SDK/test/orchestrator entegrasyonunda surpriz davranislar.  
**Oneri:** side-effect’leri `main()`/acik init fonksiyonlari arkasina tasi; config’i her run icin immutable snapshot yap.

---

## 8) Olgunluk (Kategori)
**Siniflama:** **Engineering PoC / Ar-Ge (Pre-Training), review-ready**  
Gerekce (Verified (Code)+Verified (Run)):
- **Artisi:** mimari bloklar + training skeleton + operator gate + SDK var; offline-first dogrulama pipeline’i yesil.
- **Eksisi:** reprodusibl training checkpoint/benchmark raporu yok; hedef rakamlar egitim/benchmark olmadan hedef olarak kalir.

Ozet: Teknik yapilabilirlik + pipeline demo + muhendislik incelemesi icin guclu; egitim+benchmark olmadan “production-ready” degil.

---

## 9) Ekip Tahmini (Kac kisi?)
**Kanit (Verified (Run)):** Git gecmisi **1 author** gosteriyor (103 commit).  
**En olasi:** **1 kisi** ana gelistirici.  
**Alternatif (Assumption):** 1 ana gelistirici + ara ara reviewer/tool (Git’te gorunmeyebilir).  
**Guvenli cumle:** “En az 1, cok buyuk olasilikla 1”.

---

## 10) Net Oneriler (2 hafta / 2 ay)
### 10.1 2 haftada (P0/P1)
- Hedef donanimda ilk egitim kosusunu yap (pinlenmis datasetler: `datasets/hashes.json`).
- Benchmark ciktilarini `reports/benchmarks/` altina al; README hedef -> Verified guncelle.
- Gated/karma lisansli kaynaklar icin uyum sign-off’u dokumante et (veya kaynak degistir).

### 10.2 2 ayda (Pilot hazirligi)
- Tekrarlanabilir egitim kosulari (resume/restore drill) + “run manifest” + sabit seed.
- Cihaz profili (NPU/CPU) + enerji/latency olcum protokolu.
- Dis review checklistine gore son seviye degerlendirme (`reports/review_checklist.md`).

---

## Ek A — En onemli giris noktalar (entry-points)
- Preflight: `scripts/titan_preflight.py`
- Operator Gate: `scripts/operator_mode_gate.py`
- Training: `train/train.py` (Accelerate ile)
- Data Pipeline: `scripts/data_pipeline.py` (buyuk, network-heavy)
- SDK CLI: `mertformer_sdk/cli.py` (entry: `mertformer`)

## Ek B — Lokal artefakt notu
Bazi buyuk dosyalar (`*.zip`, `tokenizer/tr/*`) `.gitignore` ile ignore ediliyor.  
Test/Export sirasinda buyuk artefaktlar olusabilir (ornegin `*.onnx.data`).
