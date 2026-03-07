# Codex Deep Audit — MertFormer Titan (v1.0 Build 30)
**Repo:** `.`
**Audit-Datum (lokal):** 2026-02-06
**Audit-Typ:** Code + Dokumentation + Run-Verifikation (offline-first)

## Kisa TR Ozet (6-10 satir)
Bu repo; “mobile-first / NPU hedefli” bir LLM mimarisi icin BitLinear (dusuk-bit agirlik simulasyonu), MLA attention, MoE + LiquidRouter routing ve Liquid/CfC dinamik katmanlarini bir araya getiren kapsamli bir Ar-Ge + engineering PoC calismasi. Mimarinin ve egitim iskeletinin calistigi dogrulandi: secret scan PASS, preflight PASS, operator-mode gate PASS, pytest PASS (`21 passed, 4 skipped`) ve `run.sh --test` offline-first PASS. Dataset kaynak/lisans envanteri artik kodla hizali (inventory + LICENSES + snapshot/hash registry mevcut; `datasets/hashes.json` pinlenmis revision + manifest fingerprint iceriyor); bununla birlikte `bigcode/the-stack-v2` gibi gated/karma lisansli kaynaklar kurumsal egitimde hukuki onay sureci gerektirir. Dokumantasyon genis ama performans/NPU hiz/enerji gibi rakamlar su an “hedef/iddia” seviyesinde (reprodusibl checkpoint + benchmark raporu yok). Seviye: **Engineering PoC / Ar-Ge (Pre-Training)**; “review-ready” (muhendislik incelemesi ve egitime baslamak icin) ama “production-ready” degil (egitim + benchmark + cihaz profili eksik). Git gecmisi tek author gosteriyor; en olasi ekip: 1 kisi (belirsizlik payi: tool/yardimci katkilar Git’te gorunmeyebilir).

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
### 2.1 Verification Baseline (Verified (Run))
- Host: MacBook Air (Apple Silicon M4), 16 GB RAM, macOS 26.2 (arm64) (`reports/system_hardware.md`)
- Python (baseline): **3.11.14** (`.titan-venv/bin/python -V`)
- Default: **offline-first** (`TITAN_OFFLINE=1`)
- Single-command verification: `bash scripts/verify_all.sh`

### 2.2 Tracked Contents (Verified (Run))
Metriken basieren auf `git ls-files` (nur tracked; keine lokalen Artefakte):
- Tracked files total: **254**
- Markdown: **126**, Python: **89**, JSON: **8**, JSONL: **2**, YAML: **9**, YML: **1**, TOML: **1**, Shell: **3**, TXT: **3**, Other: **12**
- `scripts/*.py`: **35**
- `tests/*.py`: **8**

Text-Zeilen (tracked, grob nach Endung; Binaerdateien ignoriert):
- Python: **14,610** Zeilen
- Markdown: **7,009** Zeilen

Groesste tracked Dateien (Auszug, lokal; Verified (Run)):
- `assets/synaptic_map.png` (~0.93 MB)
- `assets/header.png` (~0.86 MB)
- `README.md` / `README_TR.md` (~78 KB)
- `train/train.py` (~68 KB)

### 2.3 Git-Analyse (Verified (Run))
- Commits: **103**
- Autor:innen laut Git-Historie: **1** (`git shortlog -sne HEAD`)
- Sichtbarer Zeitraum: **2026-02-02** bis **2026-02-06**

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
- **Scripts/Ops:** `run.sh`, `scripts/bootstrap_venv.sh`, `scripts/verify_all.sh`, `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `scripts/overfit_gate.py`, `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`
- **Dataset Compliance:** `scripts/extract_dataset_refs.py` (Inventory), `scripts/record_dataset_hashes.py` (Snapshot/Hash Registry), `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`
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
1. Waehlt bevorzugt `.titan-venv/bin/python` (sonst optionales Bootstrap via `scripts/bootstrap_venv.sh`)
2. Laedt `.env` (Secrets werden nicht ausgegeben; offline-first Default)
3. `scripts/version_checker.py` (lokale Konsistenz)
4. Optional: WandB Login nur wenn `TITAN_OFFLINE=0` und `TITAN_WANDB=1`
5. Fuehrt `scripts/titan_preflight.py` aus (offline: keine HF-Connectivity)
6. `--test/--verify`: Exit nach Preflight
7. Normalmodus: Training-Pipeline ist **deaktiviert** solange `TITAN_OFFLINE=1` (Safety Gate)

### 4.2 Ops-Hinweis: Venv-Relocation (Verified (Code)+Assumption)
Die Repo-`/.titan-venv` wirkt verschoben/relocated (einige venv-CLIs koennen Shebang-Probleme haben).
Mit `python -m pip` / `python -m wandb` ist `run.sh` robuster, aber direkte CLI-Aufrufe aus `.titan-venv/bin/*` koennen trotzdem brechen (Assumption; je nach Installationspfad).

---

## 5) Verifikation (Run-Ergebnisse)
### 5.1 Ergebnis-Tabelle (Verified (Run))
| Schritt | Command | Ergebnis | Notizen |
| --- | --- | --- | --- |
| Secret Scan (tracked) | `./.titan-venv/bin/python scripts/secret_scan.py` | **PASS (Exit 0)** | Keine Secret-Patterns in tracked Dateien |
| Unit Tests | `./.titan-venv/bin/python -m pytest -q` | **PASS** | `21 passed, 4 skipped` |
| Preflight (offline) | `TITAN_OFFLINE=1 ./.titan-venv/bin/python scripts/titan_preflight.py` | **PASS (Exit 0)** | HF/WandB Connectivity wird im Offline-Default uebersprungen; keine Token-Ausgabe |
| Operator Gate (safe, offline) | `./.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | **PASS (Exit 0)** | Overfit-Gate PASS (Loss faellt), Golden Samples PASS, Benchmarks “ready” |
| `run.sh --test` offline-first | `TITAN_OFFLINE=1 bash run.sh --test` | **PASS (Exit 0)** | Keine externen Logins/Downloads; Exit nach Preflight |

### 5.2 Pytest-Warnungen (Verified (Run))
- `torch.jit.script` DeprecationWarning (Torch): betrifft JIT-Pfad in `layers/liquid.py`
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
| Dataset Lineage/Lizenzen “vollstaendig” | implizit | `scripts/extract_dataset_refs.py` → `datasets/inventory*` + `datasets/SOURCES*.md` + `datasets/LICENSES*.md` + `datasets/hashes.json` | **Verified (Code)+Verified (Run)** (Registry vorhanden) |
| NPU Speed / Energy Zahlen | Ja | keine Messungen/Benchmarks im Repo | Claim (Docs) |

---

## 7) Findings (Neutral, priorisiert)
### P0 — Post-Training Evidenz fehlt (Checkpoints/Benchmarks/Device-Profiling)
**Beobachtung (Verified (Code)):** Repo ist bewusst “Pre-Training”; es gibt keine reproduzierbaren Trainings-Checkpoints oder Benchmark-Outputs.
**Risiko:** Performance-/NPU-/Energy-Zahlen bleiben Targets; technische Bewertung ist pipeline-zentriert.
**Empfehlung:** Erste Trainings-Session auf Zielhardware + `scripts/benchmarks_internal.py` Outputs unter `reports/benchmarks/` ablegen; danach README Targets → Verified umstellen.

### P0 — Compliance Prozess (gated / mixed-license Quellen)
**Beobachtung (Verified (Code)):** `datasets/LICENSES*.md` und `datasets/hashes.json` sind vorhanden; `bigcode/the-stack-v2` ist jedoch gated und hat gemischte Upstream-Lizenzen.
**Risiko:** Kuratorische/legale Freigabe ist fuer kuratierte/denetimli Trainingslaeufe erforderlich.
**Empfehlung:** Interne Freigabe dokumentieren (Policy + Sign-off) oder alternative, einfacher lizenzierbare Datenquellen pinnen.

### P1 — Plattform: `torch.jit.script` Deprecation (Torch)
**Beobachtung (Verified (Run)):** Warnungen im Testlauf; JIT ist langfristig “legacy”.
**Impact:** mittelfristig Migrationsaufwand (z.B. `torch.compile` / `torch.export`).
**Empfehlung:** JIT-Pfad optional halten und Roadmap fuer Ersatz definieren.

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
**Einstufung:** **Engineering PoC / R&D (Pre-Training), review-ready**
Begruendung (Verified (Code)+Verified (Run)):
- **Plus:** Architekturbausteine + Trainings-Skeleton + Operator Gates + SDK sind implementiert; offline-first Verify-Pipeline ist gruen.
- **Minus:** Kein reproduzierbarer Trainings-Checkpoint/Benchmark-Report im Repo; Targets bleiben Targets bis Training/Benchmarks vorliegen.

Kurz: Stark als technische Machbarkeits- und Pipeline-Demo sowie fuer Engineering Review; nicht “production ready” ohne Training + Benchmarks.

---

## 9) Team-Estimate (Wie viele Personen?)
**Evidence (Verified (Run)):** Git-Historie zeigt **1 Autor** (103 Commits).
**Wahrscheinlich:** **1 Person** als Hauptentwickler.
**Alternative (Assumption):** 1 Kernentwickler + gelegentliche Reviewer/Tools (nicht sichtbar in Git).
**Sichere Aussage:** “Mindestens 1, sehr wahrscheinlich 1”.

---

## 10) Konkrete Empfehlungen (2 Wochen / 2 Monate)
### 10.1 In 2 Wochen (P0/P1)
- Erste Trainings-Session auf Zielhardware durchfuehren (mit pinnten Datasets aus `datasets/hashes.json`).
- Benchmark-Outputs erzeugen und unter `reports/benchmarks/` ablegen; README Targets -> Verified aktualisieren.
- Compliance Sign-off fuer gated/mixed-license Quellen dokumentieren (oder Datenquelle anpassen).

### 10.2 In 2 Monaten (Pilot-Vorbereitung)
- Wiederholbare Trainingslaeufe (Resume/Restore Drill) mit “run manifest” + feste Seeds.
- Device-Profiling (NPU/CPU) + Energy/Latency Messprotokoll.
- Reifegrad-Check gegen externe Review-Checklist (`reports/review_checklist.md`).

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
