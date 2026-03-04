![MertFormer Titan Header](assets/header.png)

Language: [English](README_SUMMARY.md) | [Turkce](README_SUMMARY_TR.md)

---

# MertFormer Titan - External Summary (Build 30)

## What This Project Is
MertFormer Titan is a mobile-first, offline-capable AI architecture designed for controlled, auditable, and human-supervised use.  
It combines BitNet (1.58-bit), Liquid dynamics, and MoE routing under a production-first engineering approach.

## Current Status
- **Stage**: Pilot-ready, pre-training baseline (`Build 30`)
- **Positioning**: Proof-of-system for real-world constrained environments
- **Not claimed yet**: Final benchmark superiority and production-grade performance claims (require trained checkpoint evidence)

## Safety and Governance Policy
- Human-in-the-loop is mandatory for operational decisions.
- Audit trail and policy boundaries are mandatory on orchestrator/runtime side.
- Unauthorized surveillance, hidden tracking, and non-consensual intervention are out of scope.
- Security and governance gates must pass before pilot performance claims.

## Verified Local Evidence (Latest Run)
| Gate | Result |
| :--- | :--- |
| `python3 -m pytest -q` | `111 passed, 3 skipped` |
| `.titan-venv/bin/python -m ruff check .` | `All checks passed` |
| `bash scripts/verify_all.sh` | `[verify] OK` |

Closure artifacts:
- `reports/closure_57_matrix.json`
- `reports/closure_57_matrix.md`
- `reports/closure_57_matrix_TR.md`

## Quick Start (External Reviewer)
1. Create/refresh virtual environment:
```bash
bash scripts/bootstrap_venv.sh
```
2. Run full offline verification gate:
```bash
bash scripts/verify_all.sh
```
3. Check training readiness (strict gate):
```bash
bash run.sh --train-ready
```
4. Start training when compute + dataset prerequisites are satisfied:
```bash
TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash run.sh
```

## External Pilot Usage Model
- Run reproducible validation gates in the customer environment.
- Share machine-readable pilot evidence (logs/reports), not unverifiable claims.
- Use NDA/private data-room flow for sensitive technical details.
- Keep measured vs projected results clearly separated.

## Claim Boundary (Important)
- Before a fully trained checkpoint and repeatable benchmark outputs, this repo remains:
  - **Pilot-ready engineering baseline**
  - **NOT ELIGIBLE FOR FINAL BENCHMARK CLAIMS**

## Useful Docs
- Main docs: [README.md](README.md), [README_TR.md](README_TR.md)
- Usage guide: [USAGE_GUIDE.md](USAGE_GUIDE.md), [USAGE_GUIDE_TR.md](USAGE_GUIDE_TR.md)
- SDK guide: [SDK_GUIDE.md](SDK_GUIDE.md), [SDK_GUIDE_TR.md](SDK_GUIDE_TR.md)
- Safety/policy: [SECURITY.md](SECURITY.md), [USE_POLICY.md](USE_POLICY.md)
