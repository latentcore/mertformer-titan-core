# Codex Derin Denetim — MertFormer Titan (v1.0 Build 27)
**Repo:** `/Users/mertyunlu/Desktop/NİHAİ`  
**Denetim Tarihi (lokal):** 2026-02-06  
**Denetim Tipi:** Kod + Dokumantasyon + Calistirmali Dogrulama (Preflight/Operator Gate/Pytest)

## Kisa Ozet (6-10 satir)
Bu repo; “mobile-first / NPU hedefli” bir LLM mimarisi icin BitLinear (dusuk-bit agirlik simulasyonu), MLA attention, MoE + LiquidRouter routing ve Liquid/CfC dinamik katmanlari bir araya getiren kapsamli bir Ar-Ge + engineering PoC calismasi. Mimari ve egitim iskeleti gercek: preflight PASS, operator-mode gate PASS, pytest PASS (21 passed, 4 skipped). Dokumantasyon genis ama performans/NPU hiz/enerji gibi rakamlar su an “hedef/iddia” seviyesinde (cihaz profili + gercek egitim/ckpt yok). En buyuk teknik riskler: (1) dataset kaynak/lisans envanteri kodda referanslanan tum kaynaklarla uyumlu degil, (2) Python 3.14+’ta `torch.jit.script` uyarisi (gelecekte kirilma riski), (3) global `cfg` ve import-side-effect’ler (test/SDK entegrasyonunda surpriz). Seviye olarak: **Engineering PoC / Ar-Ge (Pre-Training)**; pilot/urun icin snapshot’li veri, benchmark ve ops guvenlik standardizasyonu gerekli. Git gecmisi tek author gosteriyor; en olasi ekip: 1 kisi (belirsizlik payi: tool/yardimci katkilar görünmeyebilir). Not: Denetim sirasinda bazi stabilite/guvenlik duzeltmeleri uygulandi (secret redaction, GQA guard, preflight’in daha deterministik bitmesi).

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
### 2.1 Ortam (Verified (Run))
- Host: MacBook Air (Apple Silicon M4), 16 GB RAM, macOS 26.2 (arm64) (`reports/system_hardware.md`)
- Python: 3.14.0
- Torch: 2.10.0 (MPS var, CUDA yok)

### 2.2 Icerik (Verified (Run))
Tarama kapsamindan haric (kaba): `.git/`, `.titan-venv/`, `.lint-venv/`, `.pytest_cache/`, `__pycache__/`.
- Toplam dosya: **266**
- Markdown: **113**, Python: **82**, JSON: **15**, JSONL: **18**, YAML: **8**, TOML: **1**, Shell: **1**
- Test dosyasi: `tests/` altinda **8**
- Script dosyasi: `scripts/` altinda **30**

Metin satirlari (kabaca, uzanti bazinda; `tokenizer/tr/tokenizer.json` gibi buyuk veri dosyalari dahil olabilir):
- Python: **13,259** satir
- Markdown: **5,851** satir

En buyuk dosyalar (ornek, lokal; Verified (Run)):
- `test_export.onnx.data` (~43.6 MB, untracked run artefakti)
- `MertFormer_Titan_OnyxStorm_v1.0_B27_Locked.zip` (~3.7 MB, `.gitignore` ile ignore)
- `tokenizer/tr/tokenizer.json` (~3.3 MB, `.gitignore` ile ignore)
- `assets/synaptic_map.png` (~0.9 MB)

### 2.3 Git Analizi (Verified (Run))
- Git’e gore author sayisi: **1** (91 commit)
- Gorunen commit araligi: **2026-02-02** ile **2026-02-05**

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
- **Ops/Scripts:** `run.sh`, `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `scripts/overfit_gate.py`, `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`
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
1. `.env` yukler (HF/WandB secrets)
2. Dependency kurar (via `python -m pip`)
3. Accelerate auto-config
4. `scripts/titan_preflight.py` calistirir
5. Normal mod: Operator gate (full) + Smart runner (Data -> Distill -> Train)
6. `--test`: preflight’ten sonra cikar

### 4.2 Ops Notu: Venv Relocation (Verified (Code)+Assumption)
Repo’daki `/.titan-venv` tasinmis/relocated gorunuyor olabilir (bazi venv CLI’lari shebang problemi yasayabilir).  
`python -m pip` / `python -m wandb` ile `run.sh` daha saglam; ama `.titan-venv/bin/*` altindaki direkt CLI calistirmalari yine de bozulabilir (Assumption; kurulum yoluna bagli).

---

## 5) Calistirmali Dogrulama (Run Sonuclari)
### 5.1 Sonuc Tablosu (Verified (Run))
| Adim | Komut | Sonuc | Not |
| --- | --- | --- | --- |
| Preflight | `./.titan-venv/bin/python scripts/titan_preflight.py` | **PASS (Exit 0)** | Secrets redacted; denetimde streaming-sample yerine metadata bazli check’e gecildi (once: muhtemel “hang”) |
| Operator Gate (safe) | `./.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | **PASS (Exit 0)** | Overfit gate PASS (loss dusuyor), Golden Samples PASS, Benchmarks “ready” |
| Unit testler | `./.titan-venv/bin/python -m pytest -q` | **PASS** | `21 passed, 4 skipped` |

### 5.2 Pytest Uyarilari (Verified (Run))
- Python 3.14+: `torch.jit.script` DeprecationWarning (ileride kirilabilir; `layers/liquid.py`)
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
| Dataset lineage/lisanslar “tam” | dolayli | `scripts/data_pipeline.py` daha fazla dataset referansliyor; `datasets/SOURCES.md` eksik | **Verified (Code) Bulgu** |
| NPU hiz/enerji rakamlari | Evet | repo’da olcum/bmark raporu yok | Claim (Docs) |

---

## 7) Bulgular (Tarafsiz, Oncelikli)
### P0 — Uyum/Provenance: Dataset envanteri pipeline ile eslesmiyor
**Gozlem (Verified (Code)):** `scripts/data_pipeline.py` u.a. `bigcode/the-stack-v2`, `HuggingFaceFW/fineweb-edu`, `OpenAssistant/oasst_top1_2023-08-25`, `glaiveai/glaive-function-calling-v2`, `TFLai/Turkish-Alpaca`, `turkish-nlp-suite/InstrucTurca`, `HuggingFaceTB/cosmopedia`, `TIGER-Lab/MathInstruct` gibi kaynaklara referans veriyor.  
**Ama:** `datasets/SOURCES.md`/`datasets/LICENSES.md` sadece bir kismini listeliyor.  
**Risk:** lisans/kullanim ve reprodusibilite aciklari (snapshot/hash/license audit tamam degil).  
**Oneri:** SOURCES/LICENSES’i pipeline’dan otomatik uretilir hale getir veya manuel tamamlama yap; her release icin hash-snapshot dokumante et.

### P1 — Platform riski: Python 3.14+’ta `torch.jit.script`
**Gozlem (Verified (Run)):** DeprecationWarning: JIT bozulabilir.  
**Etkisi:** gelecekteki Python/Torch update’leri preflight/test/training’i kirabilir.  
**Oneri:** Python versiyonunu pinle (<=3.13) veya JIT yolunu orta vadede `torch.compile`/`torch.export`’a tasi.

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
**Siniflama:** **Engineering PoC / Ar-Ge (Pre-Training)**  
Gerekce (Verified (Code)+Verified (Run)):
- **Artisi:** mimari bloklar + training skeleton + operator gate + SDK var; preflight/operator gate/pytest calisiyor.
- **Eksisi:** reprodusibl training checkpoint/benchmark raporu yok; dataset compliance eksik; platform riski (Python 3.14 JIT).

Ozet: Teknik yapilabilirlik ve pipeline demo’su icin guclu; “pilot/production ready” degil.

---

## 9) Ekip Tahmini (Kac kisi?)
**Kanit (Verified (Run)):** Git gecmisi **1 author** gosteriyor (91 commit).  
**En olasi:** **1 kisi** ana gelistirici.  
**Alternatif (Assumption):** 1 ana gelistirici + ara ara reviewer/tool (Git’te gorunmeyebilir).  
**Guvenli cumle:** “En az 1, cok buyuk olasilikla 1”.

---

## 10) Net Oneriler (2 hafta / 2 ay)
### 10.1 2 haftada (P0/P1)
- Dataset compliance: SOURCES/LICENSES’i tum pipeline kaynaklariyla tam hizala + snapshot/hash sureci tanimla.
- Platform pinning: Python/Torch versiyonlarini pinle; JIT yolunu planla.
- Ops: `run.sh` test modunda dis login’ler (WandB) olmadan calisabilsin; artefaktlari (`*.onnx.data`) temiz/ignore et.

### 10.2 2 ayda (Pilot hazirligi)
- Stage bazli dataset snapshot + hash + lisans delilleri.
- Minimal “Tiny Titan” checkpoint + benchmark raporu (HumanEval/MBPP/GSM8K) uctan uca.
- Threat model + “no secrets in logs” gate + guvenli log retention politikasi.

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
