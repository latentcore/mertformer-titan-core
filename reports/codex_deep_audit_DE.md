# Codex Deep Audit — MertFormer Titan (v1.0 Build 27)
**Repo:** `/Users/mertyunlu/Desktop/NİHAİ`  
**Audit-Datum (lokal):** 2026-02-06  
**Audit-Typ:** Code + Dokumentation + Run-Verifikation (Preflight/Operator Gate/Pytest)

## Kisa TR Ozet (6-10 satir)
Bu repo; “mobile-first / NPU hedefli” bir LLM mimarisi icin BitLinear (dusuk-bit agirlik simülasyonu), MLA attention, MoE + LiquidRouter routing ve Liquid/CfC dinamik katmanlari bir araya getiren kapsamli bir Ar-Ge + engineering PoC calismasi. Mimari ve egitim iskeleti gercek: preflight PASS, operator-mode gate PASS, pytest PASS (21 passed, 4 skipped). Dokumantasyon genis ama performans/NPU hiz/enerji gibi rakamlar su an “hedef/iddia” seviyesinde (cihaz profili + gercek egitim/ckpt yok). En buyuk teknik riskler: (1) dataset kaynak/lisans envanteri kodda referanslanan tum kaynaklarla uyumlu degil, (2) Python 3.14+’ta `torch.jit.script` uyarisi (gelecekte kirilma riski), (3) global `cfg` ve import-side-effect’ler (test/SDK entegrasyonunda surpriz). Seviye olarak: **Engineering PoC / Ar-Ge (Pre-Training)**; pilot/urun icin snapshot’li veri, benchmark ve ops guvenlik standardizasyonu gerekli. Git gecmisi tek author gosteriyor; en olasi ekip: 1 kisi (belirsizlik payi: tool/yardimci katkilar görünmeyebilir). Not: Audit sirasinda bazi stabilite/guvenlik duzeltmeleri uygulandi (secret redaction, GQA guard, preflight’in daha deterministik bitmesi).

---

## 1) Kontext & Umfang
Ziel: Das Projekt neutral, belegbar und fuer Dritte nachvollziehbar bewerten (Architektur, Code-Qualitaet, Pipeline, Verifikation), ohne Marketingannahmen zu uebernehmen.  
Nicht-Ziel: Volltraining (Tage/Wochen) oder echte Device-/Benchmark-Validierung (dafuer fehlen reproduzierbare Checkpoints + Messungen).

**Labeling-Regeln (Transparenz):**
- **Verified (Code):** direkt im Code implementiert und nachvollziehbar
- **Verified (Run):** in dieser Umgebung ausgefuehrt (Exit-Status / Testresultat)
- **Claim (Docs):** in Dokumenten behauptet, aber nicht durch Code/Run belegt
- **Assumption:** plausibel abgeleitet, aber nicht beweisbar

---

## 2) Repository Snapshot (Metriken)
### 2.1 Umgebung (Verified (Run))
- Host: MacBook Air (Apple Silicon M4), 16 GB RAM, macOS 26.2 (arm64) (`reports/system_hardware.md`)
- Python: 3.14.0
- Torch: 2.10.0 (MPS verfuegbar, CUDA nicht verfuegbar)

### 2.2 Inhalte (Verified (Run))
Erfasst (ohne `.git/`, `.titan-venv/`, `.lint-venv/`, `.pytest_cache/`, `__pycache__/`):
- Dateien gesamt: **266**
- Markdown: **113**, Python: **82**, JSON: **15**, JSONL: **18**, YAML: **8**, TOML: **1**, Shell: **1**
- Tests: **8** Dateien unter `tests/`
- Skripte: **30** Dateien unter `scripts/`

Text-Zeilen (grob, nach Endung; inkl. Daten-Dateien wie `tokenizer/tr/tokenizer.json`):
- Python: **13,259** Zeilen
- Markdown: **5,851** Zeilen

Groesste Dateien (Auszug, lokal; Verified (Run)):
- `test_export.onnx.data` (~43.6 MB, untracked Run-Artefakt)
- `MertFormer_Titan_OnyxStorm_v1.0_B27_Locked.zip` (~3.7 MB, per `.gitignore` ignoriert)
- `tokenizer/tr/tokenizer.json` (~3.3 MB, per `.gitignore` ignoriert)
- `assets/synaptic_map.png` (~0.9 MB)

### 2.3 Git-Analyse (Verified (Run))
- Autor:innen laut Git-Historie: **1** (91 Commits)
- Sichtbarer Zeitraum: **2026-02-02** bis **2026-02-05**

Interpretation (Assumption):
- Sehr wahrscheinlich 1 Hauptentwickler (ggf. mit Tools/Assistenten), da Single-Author-Historie + konsistente Struktur.

---

## 3) Architektur-Ueberblick (Was ist implementiert?)
### 3.1 Modulkarte (Verified (Code))
- **Konfiguration:** `config/config.py` (Global `cfg`, Overlays, Validierung)
- **Modell:** `model/transformers.py` (Embedding, Blocks, KV-Cache, Generate)
- **Layer:** `layers/`  
  - `bitlinear.py`: Aktivations-Quant + ternary Weight-Quant (STE) + optional Triton-Kernel
  - `mla.py`: Attention + RoPE + KV-Cache + GQA Repeat-Logic
  - `moe.py`: MoE Dispatch + LiquidRouter (stateful) + Aux Loss + Collapse Handling
  - `liquid.py`: CfC/LiquidCell + (optional) JIT-Pfad + Residual/Norm
  - `qinn.py`: optionaler unitary Layer (Cayley Transform)
  - `mertformer_block.py`: Block-Komposition (Norm -> MLA -> optional Liquid -> FFN/MoE -> optional QINN)
- **Training:** `train/train.py` (Accelerate, Curriculum, Offline/Online Distillation, Checkpoints, Export)
- **Scripts/Ops:** `run.sh`, `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `scripts/overfit_gate.py`, `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`
- **SDK/CLI:** `mertformer_sdk/` (API + CLI Wrapper)
- **Orchestrator (optional):** `orchestrator/` (Memory/RAG/Web/Audio/SenseEngine; teils optional deps)

### 3.2 Kernideen — Statusbewertung (kurz)
- BitNet/ternary Weights: **Verified (Code)** als *On-the-fly Simulation* (keine echte Speicherkompression ohne Bitpacking-Pfad)
- MoE + LiquidRouter: **Verified (Code)**
- Liquid/CfC Dynamics: **Verified (Code)**
- KV-Cache + Generate: **Verified (Code)**
- “Mobile/NPU Performance Zahlen”: **Claim (Docs)** (keine reproduzierbaren Benchmarks/Device-Profile im Repo)

---

## 4) Build/Run-Pipeline (Wie wird es gestartet?)
### 4.1 `run.sh` (Verified (Code))
High-Level Ablauf:
1. Laedt `.env` (HF/WandB Secrets)
2. Installiert Dependencies (via `python -m pip`)
3. Auto-configure Accelerate
4. Fuehrt `scripts/titan_preflight.py` aus
5. Normalmodus: Operator Gate (full) + Smart Runner (Data -> Distill -> Train)
6. `--test`: Exit nach Preflight

### 4.2 Ops-Hinweis: Venv-Relocation (Verified (Code)+Assumption)
Die Repo-`/.titan-venv` wirkt verschoben/relocated (einige venv-CLIs koennen Shebang-Probleme haben).  
Mit `python -m pip` / `python -m wandb` ist `run.sh` robuster, aber direkte CLI-Aufrufe aus `.titan-venv/bin/*` koennen trotzdem brechen (Assumption; je nach Installationspfad).

---

## 5) Verifikation (Run-Ergebnisse)
### 5.1 Ergebnis-Tabelle (Verified (Run))
| Schritt | Command | Ergebnis | Notizen |
| --- | --- | --- | --- |
| Preflight | `./.titan-venv/bin/python scripts/titan_preflight.py` | **PASS (Exit 0)** | Secrets redacted; waehrend Audit von Streaming-Sample auf metadata-basierten Check umgestellt (vorher: moeglicher “hang”) |
| Operator Gate (safe) | `./.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | **PASS (Exit 0)** | Overfit-Gate passt (Loss faellt), Golden Samples PASS, Benchmarks “ready” |
| Unit Tests | `./.titan-venv/bin/python -m pytest -q` | **PASS** | `21 passed, 4 skipped` |

### 5.2 Pytest-Warnungen (Verified (Run))
- Python 3.14+: `torch.jit.script` DeprecationWarning (JIT kann brechen; betrifft `layers/liquid.py`)
- Torch ONNX Export: Warnungen bzgl. `dynamic_axes` / dynamo-Exporter (kein Fail)

---

## 6) Dokument-Claims vs Code-Realitaet (Auszug)
| Thema | Claim (Docs) | Evidenz | Status |
| --- | --- | --- | --- |
| “Pre-Training / Unverified” | Ja | `README.md`, `MODEL_CARD.md` | Verified (Docs) |
| 18 Layer / Titan Config | Ja | `config/config.py` + Modell baut `cfg.num_layers` Blocks | Verified (Code) |
| BitNet 1.58-bit “weights” | Ja | `layers/bitlinear.py` (ternary quant im forward) | Verified (Code) als Simulation |
| MoE (8 Experten, top-2) | Ja | `config/config.py`, `layers/moe.py` | Verified (Code) |
| Offline Distillation (precomputed logits) | Ja | `train/train.py`, `orchestrator/distillation_manager.py` | Verified (Code) |
| Dataset Lineage/Lizenzen “vollstaendig” | implizit | `scripts/data_pipeline.py` referenziert deutlich mehr Datasets als `datasets/SOURCES.md` listet | **Verified (Code) Finding** |
| NPU Speed / Energy Zahlen | Ja | keine Messungen/Benchmarks im Repo | Claim (Docs) |

---

## 7) Findings (Neutral, priorisiert)
### P0 — Compliance/Provenance: Dataset-Inventar ist nicht deckungsgleich zur Pipeline
**Beobachtung (Verified (Code)):** `scripts/data_pipeline.py` referenziert u.a. `bigcode/the-stack-v2`, `HuggingFaceFW/fineweb-edu`, `OpenAssistant/oasst_top1_2023-08-25`, `glaiveai/glaive-function-calling-v2`, `TFLai/Turkish-Alpaca`, `turkish-nlp-suite/InstrucTurca`, `HuggingFaceTB/cosmopedia`, `TIGER-Lab/MathInstruct`.  
**Aber:** `datasets/SOURCES.md`/`datasets/LICENSES.md` enthalten nur einen Teil.  
**Risiko:** Lizenz-/Nutzungs- und Reproduzierbarkeitsluecken (Snapshot/Hash/License-Audit ist unvollstaendig).  
**Empfehlung:** SOURCES/LICENSES aus Pipeline automatisiert generieren oder manuell angleichen; pro Release Hash-Snapshots dokumentieren.

### P1 — Plattform-Risiko: `torch.jit.script` unter Python 3.14+
**Beobachtung (Verified (Run)):** DeprecationWarning: JIT kann brechen.  
**Impact:** zukuenftige Python/Torch Updates koennen preflight/tests/training brechen.  
**Empfehlung:** Python-Version pinnen (<=3.13) oder JIT-Pfad mittelfristig auf `torch.compile`/`torch.export` migrieren.

### P1 — Konfig/Robustheit: GQA/KV-Head Validierung muss hart failen (jetzt abgesichert)
**Beobachtung (Verified (Code)+Verified (Run)):** Ohne Guards kann `num_kv_heads > num_heads` zu invaliden Shapes fuehren.  
**Status:** **Remediated waehrend Audit**: Guards in `layers/mla.py` + Test-Fixture patcht `cfg.num_kv_heads` → Pytest gruen.  
**Empfehlung:** Validierung ggf. zusaetzlich zentral in `config/config.py` verankern (Single Source of Truth).

### P1 — Secret Hygiene: “kein Token in Logs” ist ein Prozess-Gate (teilweise behoben)
**Beobachtung (Verified (Code)):** Preflight loggt jetzt keine Token-Fragmente mehr (redacted).  
**Status:** **Remediated waehrend Audit** (Code + alte Log-Snippets/Docs redacted).  
**Empfehlung:** Zusaetzlich CI/Operator-Gate: Secret-Scanner auf `logs/` + `README*` + `reports/`.

### P2 — Preflight: Netzwerkcheck darf keine “Long-Running Transfers” starten
**Beobachtung (Verified (Run)):** Ein Streaming-Sample Download kann (je nach HF-Backend) Hintergrundtransfers triggern, wodurch der Preflight-Prozess trotz “ALL GREEN” nicht sauber beendet.  
**Status:** **Remediated waehrend Audit**: Default-Check ist nun metadata-basiert; Streaming-Sample ist opt-in (`TITAN_PREFLIGHT_STREAM_SAMPLE=1`).  
**Empfehlung:** Preflight bewusst “leichtgewichtig” halten; Timeouts fuer Netzwerkchecks setzen.

### P2 — Import-Side-Effects / Global State
**Beobachtung (Verified (Code)):** Globales `cfg` und teils Side-Effects beim Import (Auto-Tuning/Prints).  
**Impact:** SDK/Test/Orchestrator-Integration kann “zufaellig” Verhalten aendern.  
**Empfehlung:** Side-Effects hinter `main()`/explizite Init-Funktionen; config als immutable Snapshot pro Run.

---

## 8) Reifegrad-Einstufung (Kategorie)
**Einstufung:** **Engineering PoC / R&D (Pre-Training)**  
Begruendung (Verified (Code)+Verified (Run)):
- **Plus:** Architekturbausteine + Trainings-Skeleton + Operator Gates + SDK sind implementiert; Preflight/Operator Gate/Pytest laufen.
- **Minus:** Kein reproduzierbarer Trainings-Checkpoint/Benchmark-Report im Repo; Dataset compliance unvollstaendig; Plattformrisiko (Python 3.14 JIT).

Kurz: Sehr stark als technische Machbarkeits- und Pipeline-Demo, aber (noch) nicht “pilot/production ready”.

---

## 9) Team-Estimate (Wie viele Personen?)
**Evidence (Verified (Run)):** Git-Historie zeigt **1 Autor** (91 Commits).  
**Wahrscheinlich:** **1 Person** als Hauptentwickler.  
**Alternative (Assumption):** 1 Kernentwickler + gelegentliche Reviewer/Tools (nicht sichtbar in Git).  
**Sichere Aussage:** “Mindestens 1, sehr wahrscheinlich 1”.

---

## 10) Konkrete Empfehlungen (2 Wochen / 2 Monate)
### 10.1 In 2 Wochen (P0/P1)
- Dataset-Compliance: SOURCES/LICENSES erweitern (alle Pipeline-Quellen) + Snapshot/Hash Prozess definieren.
- Plattform-Pinning: Python/Torch Versionen pinnen; JIT-Pfad planen.
- Ops: `run.sh` test-mode ohne externe Logins (WandB) moeglich machen; Artefakte (`*.onnx.data`) sauber ignorieren/aufräumen.

### 10.2 In 2 Monaten (Pilot-Vorbereitung)
- Reproduzierbare Dataset-Snapshots + Hashes (pro Stage) + Lizenzbelege.
- Minimaler “Tiny Titan” Checkpoint + Benchmark-Report (HumanEval/MBPP/GSM8K) end-to-end.
- Threat Model + “no secrets in logs” Gate, inkl. sichere Log-Retention.

---

## Anhang A — Wichtigste Entry-Points
- Preflight: `scripts/titan_preflight.py`
- Operator Gate: `scripts/operator_mode_gate.py`
- Training: `train/train.py` (via Accelerate)
- Data Pipeline: `scripts/data_pipeline.py` (gross, cloud/network heavy)
- SDK CLI: `mertformer_sdk/cli.py` (Entry: `mertformer`)

## Anhang B — Hinweis zu lokalen Artefakten
Einige grosse Dateien (`*.zip`, `tokenizer/tr/*`) sind per `.gitignore` ignoriert.  
Waerend Tests/Exports koennen grosse Artefakte entstehen (z.B. `*.onnx.data`).
