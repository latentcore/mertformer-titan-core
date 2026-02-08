# Codex Deep Audit — MertFormer Titan (v1.0 Build 27)
**Repo:** `.`  
**Audit date (local):** 2026-02-06  
**Audit type:** Code + Docs + Run Verification (offline-first)

## Executive Summary (6-10 lines)
This repository is an R&D + engineering PoC for a “mobile-first / NPU-targeted” LLM stack that combines BitLinear (low-bit weight simulation), MLA attention, MoE + LiquidRouter routing, and Liquid/CfC dynamics. Core engineering readiness is verified: tracked-file secret scan PASS, preflight PASS, operator-mode gate PASS, pytest PASS (`21 passed, 4 skipped`), and `run.sh --test` PASS in offline-first mode. Dataset compliance artifacts are now present and aligned with code references (dataset inventory + SOURCES/LICENSES + pinned snapshot/hash registry in `datasets/hashes.json`); however, using gated/mixed-license sources (notably `bigcode/the-stack-v2`) still requires a corporate legal/compliance sign-off. Documentation is extensive, but performance/NPU speed/energy numbers remain targets until a real training run produces checkpoints and benchmark outputs. Maturity level: **Engineering PoC / R&D (Pre-Training), review-ready** (for engineering inspection and to start training), but not “production-ready” without post-training evidence. Git history shows a single author; the most likely team size is one primary developer (with normal uncertainty: tooling/assistants are not visible in Git).

---

## 1) Context & Scope
Goal: provide a neutral, evidence-based, third-party-readable evaluation (architecture, code quality, pipelines, verification), without assuming marketing claims are true.  
Out of scope: full pre-training (days/weeks) and real device benchmarking (requires trained checkpoints + measurement infrastructure).

**Labeling rules (transparency):**
- **Verified (Code):** directly present in code
- **Verified (Run):** executed in the baseline environment (command output observed)
- **Claim (Docs):** stated in documents but not backed by code/run evidence
- **Assumption:** plausible inference, but not provable from the repo

---

## 2) Repository Snapshot (Metrics)
### 2.1 Verification Baseline (Verified (Run))
- Host: MacBook Air (Apple Silicon M4), 16 GB RAM, macOS 26.2 (arm64) (`reports/system_hardware.md`)
- Python (baseline): **3.11.14** (`.titan-venv/bin/python -V`)
- Default: **offline-first** (`TITAN_OFFLINE=1`)
- Single-command verification: `bash scripts/verify_all.sh`

### 2.2 Tracked Contents (Verified (Run))
Metrics below are based on `git ls-files` (tracked files only; excludes local artifacts):
- Tracked files total: **254**
- Markdown: **126**, Python: **89**, JSON: **8**, JSONL: **2**, YAML: **9**, YML: **1**, TOML: **1**, Shell: **3**, TXT: **3**, Other: **12**
- `scripts/*.py`: **35**
- `tests/*.py`: **8**

Text line counts (tracked, approximate; binary files excluded):
- Python: **14,610** lines
- Markdown: **7,009** lines

Largest tracked files (sample; Verified (Run)):
- `assets/synaptic_map.png` (~0.93 MB)
- `assets/header.png` (~0.86 MB)
- `README.md` / `README_TR.md` (~78 KB)
- `train/train.py` (~68 KB)

### 2.3 Git Signals (Verified (Run))
- Commits: **103**
- Authors (Git history): **1** (`git shortlog -sne HEAD`)
- Visible date range: **2026-02-02** to **2026-02-06**

Interpretation (Assumption):
- Very likely a single primary developer (potentially with tools/assistants), given single-author history and coherent repo structure.

---

## 3) Architecture Overview (What is implemented?)
### 3.1 Module Map (Verified (Code))
- **Config:** `config/config.py` (global `cfg`, overlays, validation; also prints at import)
- **Model:** `model/transformers.py` (embeddings, blocks, KV-cache, generation)
- **Layers:** `layers/`
  - `bitlinear.py`: activation quant + ternary weight quant (STE) + optional Triton kernel path
  - `mla.py`: attention + RoPE + KV-cache + GQA repeat logic
  - `moe.py`: MoE dispatch + LiquidRouter + aux loss + collapse handling
  - `liquid.py`: CfC/LiquidCell + optional JIT path
  - `qinn.py`: optional unitary layer (Cayley transform)
  - `mertformer_block.py`: block composition (Norm -> MLA -> optional Liquid -> FFN/MoE -> optional QINN)
- **Training:** `train/train.py` (Accelerate, curriculum, offline distillation, checkpoints, export)
- **Ops / Verification:** `run.sh`, `scripts/bootstrap_venv.sh`, `scripts/verify_all.sh`, `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`
- **Dataset compliance:** `scripts/extract_dataset_refs.py` (inventory), `scripts/record_dataset_hashes.py` (snapshot/hash registry), `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`
- **SDK/CLI:** `mertformer_sdk/` (API + CLI wrapper)
- **Orchestrator (optional):** `orchestrator/` (Memory/RAG/Web/Audio/SenseEngine; optional deps)

### 3.2 Core Ideas — Status (short)
- BitNet/ternary weights: **Verified (Code)** as on-the-fly simulation (no real memory compression without bitpacking path)
- MoE + LiquidRouter: **Verified (Code)**
- Liquid/CfC dynamics: **Verified (Code)**
- KV-cache + generation: **Verified (Code)**
- “Mobile/NPU performance numbers”: **Claim (Docs)** (no reproducible trained checkpoints + benchmark outputs in repo)

---

## 4) Build/Run Pipeline (How it runs)
### 4.1 `run.sh` (Verified (Code))
High-level behavior:
1. Prefers `.titan-venv/bin/python` (optional bootstrap via `scripts/bootstrap_venv.sh`)
2. Loads `.env` (secrets are not printed; default mode is offline-first)
3. Runs `scripts/version_checker.py` (local consistency)
4. Logs into WandB only if `TITAN_OFFLINE=0` and `TITAN_WANDB=1`
5. Runs `scripts/titan_preflight.py` (offline: no HF connectivity checks)
6. `--test/--verify`: exits after preflight
7. Normal mode: training pipeline is **blocked** unless `TITAN_OFFLINE=0` (safety gate)

### 4.2 Venv note (Verified (Code))
The repo standardizes around `.titan-venv/bin/python` and uses module invocations (`python -m ...`) to avoid “relocated venv” shebang issues.

---

## 5) Verification (Run Results)
### 5.1 Results Table (Verified (Run))
| Step | Command | Result | Notes |
| --- | --- | --- | --- |
| Secret scan (tracked) | `./.titan-venv/bin/python scripts/secret_scan.py` | **PASS (Exit 0)** | No secret patterns in tracked files |
| Unit tests | `./.titan-venv/bin/python -m pytest -q` | **PASS** | `21 passed, 4 skipped` |
| Preflight (offline) | `TITAN_OFFLINE=1 ./.titan-venv/bin/python scripts/titan_preflight.py` | **PASS (Exit 0)** | Offline: HF/WandB connectivity skipped; token values not printed |
| Operator gate (safe, offline) | `./.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | **PASS (Exit 0)** | Overfit gate PASS, golden samples PASS, benchmarks “ready” |
| `run.sh --test` offline-first | `TITAN_OFFLINE=1 bash run.sh --test` | **PASS (Exit 0)** | No external logins/downloads; exits after preflight |

### 5.2 Warnings (Verified (Run))
- `torch.jit.script` DeprecationWarning (Torch): affects the JIT path in `layers/liquid.py`
- ONNX export warnings about `dynamic_axes` vs `dynamic_shapes` (non-fatal; test passes)

---

## 6) Docs Claims vs Code Reality (sample)
| Topic | Claim (Docs) | Evidence | Status |
| --- | --- | --- | --- |
| “Pre-Training / Unverified” | Yes | `README.md`, `MODEL_CARD.md` | Verified (Docs) |
| 18 layers / Titan config | Yes | `config/config.py` + model builds `cfg.num_layers` blocks | Verified (Code) |
| BitNet 1.58-bit “weights” | Yes | `layers/bitlinear.py` (ternary quant in forward) | Verified (Code) (simulation) |
| MoE (experts, top-k) | Yes | `config/config.py`, `layers/moe.py` | Verified (Code) |
| Offline distillation | Yes | `train/train.py`, `orchestrator/distillation_manager.py` | Verified (Code) |
| Dataset lineage/licenses “complete” | implicit | `datasets/inventory*` + `datasets/SOURCES*.md` + `datasets/LICENSES*.md` + `datasets/hashes.json` | Verified (Code)+Verified (Run) (registry exists) |
| NPU speed / energy numbers | Yes | No reproducible device measurements/benchmarks in repo | Claim (Docs) |

---

## 7) Findings (Neutral, prioritized)
### P0 — Post-training evidence is missing (checkpoints/benchmarks/device profiling)
**Observation (Verified (Code)):** repo is intentionally “Pre-Training”; no reproducible training checkpoints or benchmark outputs are included.  
**Risk:** performance/NPU/energy numbers remain targets; evaluation is pipeline-centric.  
**Recommendation:** run a first training session on target hardware and commit sanitized benchmark outputs under `reports/benchmarks/`; then convert README targets to verified metrics.

### P0 — Compliance process still required for gated/mixed-license sources
**Observation (Verified (Code)):** dataset docs + snapshot/hash registry exist, but `bigcode/the-stack-v2` is gated and has mixed upstream licenses.  
**Risk:** corporate/regulatory training runs require explicit legal/compliance sign-off.  
**Recommendation:** document internal sign-off (policy + approval) or pin alternative datasets with simpler terms.

### P1 — Platform: `torch.jit.script` deprecation (Torch)
**Observation (Verified (Run)):** warnings exist; JIT is long-term “legacy”.  
**Impact:** medium-term migration cost (e.g., `torch.compile` / `torch.export`).  
**Recommendation:** keep JIT path optional and define a migration roadmap.

### P2 — Import side-effects / global state (`cfg`)
**Observation (Verified (Code)):** `config/config.py` creates global `cfg` and prints at import time.  
**Impact:** SDK/tests/orchestrator integrations can see surprising side-effects.  
**Recommendation:** move side-effects behind explicit init functions; treat config as an immutable snapshot per run.

---

## 8) Maturity / Category
**Category:** **Engineering PoC / R&D (Pre-Training), review-ready**  
Rationale (Verified (Code)+Verified (Run)):
- **Pros:** core architecture blocks + training skeleton + operator gates + SDK exist; offline-first verification pipeline is green.
- **Cons:** no trained checkpoints/benchmark reports; targets remain targets until training/benchmarks exist.

---

## 9) Team Estimate (How many people built this?)
**Evidence (Verified (Run)):** Git history shows **1 author** (103 commits).  
**Most likely:** **1** primary developer.  
**Alternative (Assumption):** 1 core developer + occasional reviewers/tools (not visible in Git).  
**Safe statement:** “At least 1, very likely 1.”

---

## 10) Concrete Next Steps (2 weeks / 2 months)
### 10.1 In 2 weeks (P0/P1)
- Run a first training run on target hardware (using pinned datasets from `datasets/hashes.json`).
- Produce benchmark outputs under `reports/benchmarks/`; update README targets to verified values.
- Complete compliance sign-off for gated/mixed-license sources (or adjust data choices).

### 10.2 In 2 months (pilot preparation)
- Repeatable training runs (resume/restore drill) with a “run manifest” and fixed seeds.
- Device profiling (NPU/CPU) + energy/latency measurement protocol.
- Final check against the external review checklist (`reports/review_checklist.md`).

---

## Appendix A — Key entry points
- Review verify: `scripts/bootstrap_venv.sh`, `scripts/verify_all.sh`
- Preflight: `scripts/titan_preflight.py`
- Operator gate: `scripts/operator_mode_gate.py`
- Training: `train/train.py` (via Accelerate)
- Data pipeline: `scripts/data_pipeline.py` (large; network-heavy by design)
- SDK CLI: `mertformer_sdk/cli.py` (entry: `mertformer`)
