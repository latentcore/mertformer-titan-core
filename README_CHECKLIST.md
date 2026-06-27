# ✅ MertFormer Titan README QA Checklist

## 1️⃣ Project Structure
- [x] **Folder List:** Config, layers, scripts, checkpoints, logs are all listed.
- [x] **Descriptions:** File purposes are clearly defined (e.g., `bitlinear.py` → 1.58-bit Quantization).
- [x] **Completeness:** Matches the actual ZIP file content.

## 2️⃣ Quick Start / Deployment
- [x] **Python Examples:** `TitanChat` example provided.
- [x] **ONNX Export:** `scripts/mobile_export.py` documented.
- [x] **Dependencies:** `requirements.txt` is standard.
- [x] **Run Script:** `run.sh` handles setup and testing.
- [x] **Inference:** Sample usage shown in code blocks.

## 3️⃣ Visuals / Diagrams
- [n/a] **Mermaid:** Architecture flow — removed in the 178KB→4KB README cut; see `ARCHITECTURE.md` / `docs/PROJECT_STRUCTURE.md`.
- [n/a] **ASCII Art:** "MertFormer Titan" banner — removed in the 4KB README cut.

## 4️⃣ Metrics / FAQ / Benchmarks
- [x] **Forensic Data:** Sample PoC hash included; production metrics marked as pending.
- [x] **Projections:** Marked as pre-training estimates or targets where appropriate.
- [n/a] **Comparisons:** Llama-3/Phi-3 table removed in the 178KB→4KB README cut; no such table in the current README.
- [x] **FAQ:** Addressed 1.58-bit quality and Mobile capability.

## 5️⃣ Links / Badges / Spelling
- [x] **Links:** Internal links (anchors) verified.
- [x] **Badges:** License and Status badges present.
- [x] **Spelling:** Checked for typos in TR/EN.

## 6️⃣ Optional / Power-Ups
- [x] **Turkish Vision:** Clearly bulleted and readable in `README_TR.md`.
- [x] **Example I/O:** Chat example provides input/output context.
- [x] **Forensic Seal:** Added secure logging verification section.

---
> **Honesty note:** The `[x]` items below are **manual reviewer judgments**, not automated evidence or pass-gate outputs. Items such as Completeness (line 6), Comparisons (line 22), Links (line 26) and Spelling (line 28) are **not backed by a script/report output** — treat them as "evidence pending" rather than verified gates. `[n/a]` items were intentionally dropped in the 4KB README cut.

**Status:** 🟡 **PRE-TRAINING / DRAFT**
**Verified By:** Antigravity Agent (initial) · refreshed in the Build 30 V2 closure pass
**Date:** 2026-02-05 (initial) · 2026-06-17 (post-4KB-README-cut refresh)
