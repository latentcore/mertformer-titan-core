# Verified vs Target Matrix (Engineering Truth)

> **External review note:** If you are evaluating compute sponsorship, start
> with `reports/outreach_compute_sponsorship_messages.md`. This matrix separates
> verified surfaces from targets and should be used as a claim-boundary
> reference.

This document makes a strict distinction between:
- **Verified (Run)**: confirmed by executing commands locally or in CI
- **Verified (Code)**: confirmed by direct code inspection (but not executed)
- **Target / Claim**: intended outcome; requires training/benchmarks to be true

Legend used below:
- ✅ Verified (Run)
- 🔎 Verified (Code)
- 🎯 Target / Claim (not yet validated)
- ⏭️ SKIP (not applicable in this environment)

## Verification Baseline

- Baseline Python: **3.11** (see `repro/python.md`)
- Default mode: **offline-first** (`TITAN_OFFLINE=1`)
- Single-command verification: `bash scripts/verify_all.sh`
- Last local verify (example): 2026-02-06 (macOS, Python 3.11.14)

## Matrix

| Capability | Status | Evidence |
| --- | --- | --- |
| Tracked-file secret scan gate | ✅ Verified (Run) | `python scripts/secret_scan.py` |
| Unit/integration tests | ✅ Verified (Run) | `python -m pytest -q` |
| Preflight (offline-safe) | ✅ Verified (Run) | `TITAN_OFFLINE=1 python scripts/titan_preflight.py` |
| Operator mode gate (safe, offline) | ✅ Verified (Run) | `python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` |
| `run.sh --test` offline-first (no external login/download) | ✅ Verified (Run) | `TITAN_OFFLINE=1 bash run.sh --test` |
| Dataset IDs inventory from code | ✅ Verified (Run) | `python scripts/extract_dataset_refs.py` → `datasets/inventory*` |
| Dataset sources documented | ✅ Verified (Code) | `datasets/SOURCES*.md` |
| Dataset licenses documented (checklist) | 🔎 Verified (Code) | `datasets/LICENSES*.md` (core training datasets verified; optional/demo entries may remain `TBD` until enabled) |
| Dataset snapshot hashes recorded | ✅ Verified (Run) | `python scripts/record_dataset_hashes.py` → `datasets/hashes.json` |
| Training “tiny smoke” (CPU/MPS) | ✅ Verified (Run) | `python scripts/train_smoke.py --cleanup` |
| Full training run (end-to-end) | 🎯 Target / Claim | Requires training hardware + real data snapshots |
| Benchmarks (HumanEval/MBPP) | 🔎 Verified (Code) | `scripts/benchmarks_internal.py` (SKIP behavior if checkpoint missing) |
| ONNX export correctness | ✅ Verified (Run) | `pytest` includes `scripts/test_onnx_export.py::test_export` |
| CI pipeline | 🔎 Verified (Code) | `.github/workflows/ci.yml` |
| Demo (pygame autoplayer) | 🔎 Verified (Code) | `snake_demo.py` (requires `pip install -e '.[demo]'` or `bootstrap_venv.sh --demo`) |

## Notes / Blockers (Truthful)

- Dataset compliance gate (licenses + snapshot registry) is satisfied for the currently pinned HF revisions (see `datasets/hashes.json`).
- Performance numbers remain **targets** until a full training run and benchmark report exist.
