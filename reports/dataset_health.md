# 🛡️ TITAN DATASET HEALTH REPORT
**Date:** 2026-07-18 (refresh — see method note below)

> ⚠️ **SEMI-MANUAL, NOT ONE-COMMAND AUTO-GENERATED — read before editing or trusting blindly.** There is no dedicated generator script for this exact file (checked repo-wide 2026-07-18; only `datasets/inventory.json`/`.md` are produced by a real tool, `scripts/extract_dataset_refs.py --fetch-metadata`). This table's data is real and tool-sourced (same command, no dataset content downloaded), but the table itself was hand-assembled from that tool's JSON output plus a fixed list of 7 infra datasets checked before that tool existed. **To refresh:** re-run `python3 scripts/extract_dataset_refs.py --fetch-metadata`, then rebuild this table's Status/License columns from the fresh `datasets/inventory.json` — do not hand-flip a status to ✅/🔴 without re-running that command first.

## 📊 Status Summary

> 🟡 **SYSTEM YELLOW:** All 16 curriculum datasets now checked at least once (was 7/16). 2 real access problems found; these need a human decision, not a code fix.

## 📋 Detailed Verification Log

**Refresh method (2026-07-18):** `python3 scripts/extract_dataset_refs.py --fetch-metadata` (Hugging Face Hub API metadata fetch only — no dataset content downloaded, disk-safe, this is the tool `BACKLOG.md` already designated for resolving the license-TBD item). `mlabonne/guanaco-llama2-1k` isn't reachable by this tool (not referenced via `load_dataset(...)` in `scripts/eval/train/orchestrator`, only in `datasets/hashes.json`) — checked manually via its HF dataset page instead.

| Stage | Dataset | Split | Logic | Status | License |
|-------|---------|-------|-------|--------|---------|
| Stage 1 | `bigcode/the-stack-dedup` | train | HTTP Check | ✅ Online (revision unpinned — needs training-machine network, see BACKLOG) | — |
| Stage 1 | `TIGER-Lab/MathInstruct` | train | HTTP Check | ✅ Online | — |
| Stage 1 | `openai/gsm8k` | train | HTTP Check | ✅ Online | — |
| Stage 2 | `HuggingFaceFW/fineweb-edu` | train | HTTP Check | ✅ Online | — |
| Stage 3 | `wikimedia/wikipedia` | train | HTTP Check | ✅ Online | — |
| Stage 3 | `HuggingFaceTB/cosmopedia` | train | HTTP Check | ✅ Online | — |
| Stage 4 | `OpenAssistant/oasst_top1_2023-08-25` | train | HTTP Check | ✅ Online | — |
| Stage 3 | `uonlp/CulturaX` | train | HF metadata fetch (2026-07-18) | ✅ Online, **gated=true** (needs `HF_TOKEN` with accepted terms) | unset |
| Stage 4 | `TFLai/Turkish-Alpaca` | train | HF metadata fetch (2026-07-18) | ✅ Online | `apache-2.0` |
| Stage 4 | `turkish-nlp-suite/InstrucTurca` | train | HF metadata fetch (2026-07-18) | ✅ Online | `cc-by-sa-4.0` |
| Stage 4 | `teknium/OpenHermes-2.5` | train | HF metadata fetch (2026-07-18) | ✅ Online | ⚠️ unset (repo has no license tag) |
| Stage 5 | `glaiveai/glaive-function-calling-v2` | train | HF metadata fetch (2026-07-18) | ✅ Online | `apache-2.0` |
| Stage 5 | `NousResearch/hermes-function-calling-v1` | train (config `func_calling`) | HF Hub API fetch (2026-07-19) | ✅ Online, ungated. Replaces `gorilla-llm/gorilla-openfunctions-v2` + `NousResearch/FC-1k`, both dropped after confirmed live HTTP 401 (2026-07-18). | `apache-2.0` |
| Demo | `codeparrot/github-code` | train | HF metadata fetch (2026-07-18) | ✅ Online | `other` (per-file license inherited from source GitHub repos — genuinely mixed, not a single SPDX tag) |
| Legacy | `mlabonne/guanaco-llama2-1k` | train | Manual HF page check (2026-07-18) | ✅ Online | ⚠️ not visibly tagged in the metadata excerpt (needs a manual look at the full dataset card) |

## Still open after this refresh

- **RESOLVED (2026-07-19):** `gorilla-llm/gorilla-openfunctions-v2` and `NousResearch/FC-1k` (both confirmed dead, live HTTP 401 on 2026-07-18) were dropped from `scripts/data_pipeline.py`'s `STAGE5_SOURCES` and replaced with `NousResearch/hermes-function-calling-v1` (config `func_calling`), verified live+ungated+apache-2.0 via the HF Hub API before landing. `scripts/titan_preflight.py`'s `required_datasets` preflight-check list was also updated (it still referenced the dead `gorilla-llm` ID, which would have made the preflight gate fail on a now-intentionally-dropped source). Ratio: the new source takes the combined 0.25+0.15=0.40 share the two removed sources held, so Stage 5's ratios still sum to 1.0 and the overall 23.59B-token target (computed from `TITAN_MAX_STEPS x batch x seq_len`, independent of this list) is unaffected.
- **`teknium/OpenHermes-2.5` has no license tag on HF** — real, confirmed gap (not a lookup failure), matches what `BACKLOG.md` already flagged.
- **`codeparrot/github-code`'s `other` tag is not a single resolvable license** — it inherits per-file licenses from the original GitHub repos it scraped; treating it as "one license" would be inaccurate. Needs its own compliance note if this dataset is actually used at scale, not a single-line fix.
- **`bigcode/the-stack-dedup` revision/sha256 still unpinned** — explicitly requires network access on the training machine per `datasets/hashes.json`'s own note; not attempted here (would be a throwaway pin from a machine that isn't the training machine).

---
*Refreshed 2026-07-18 via `scripts/extract_dataset_refs.py --fetch-metadata` + one manual check. Previous scan: 2026-01-25 (7/16 checked). Generated by MertFormer Titan Verification Suite v1.0.*
