# GO Status Matrix (Pilot vs Product Claim)

## Scope
This matrix tracks what is fully completed in-repo and what remains external/operational.

## A) Pilot Delivery Readiness (A1-A20)

| ID | Status | Evidence / Reason |
| --- | --- | --- |
| A1 | ✅ | Repo clean and synced (`git status --short` clean on both Desktop + canonical repository root). |
| A2 | ✅ | `strict_checkpoint=True` default in `mertformer_sdk/api.py`. |
| A3 | ✅ | `mertformer verify` command in `mertformer_sdk/cli.py`. |
| A4 | ✅ | `mertformer pilot-report --out ...` in `mertformer_sdk/cli.py`. |
| A5 | ✅ | Pilot schema exists: `interfaces/pilot_report_v1.schema.json`. |
| A6 | ✅ | Claim gate text: `NOT ELIGIBLE FOR CLAIM` in `scripts/benchmarks_internal.py`. |
| A7 | ✅ | Desktop absolute-path gate is clean (no hardcoded Desktop path in tracked files). |
| A8 | ✅ | Claim policy + pilot checklist are explicit in `README.md`. |
| A9 | ✅ | README config sample aligned (`use_torch_compile = False`) with `config/config.py`. |
| A10 | ✅ | Pilot docs set exists: `reports/pilot_readiness_kit.md`, `reports/pilot_offer_packages.md`, `reports/sales_funnel_90d.md`. |
| A11 | ✅ | Test gate pass: `412 passed, 4 skipped`. |
| A12 | ✅ | Lint gate pass: `ruff check` all green. |
| A13 | ✅ | Full gate pass: `bash scripts/verify_all.sh` -> `[verify] OK`. |
| A14 | ✅ | SDK EN/TR aligned: `SDK_GUIDE.md`, `SDK_GUIDE_TR.md`. |
| A15 | ✅ | Release commits are pushed on `main` (see `git log --oneline -n 1`). |
| A16 | ✅ | Tag/release marker pushed (`v0.1.3-review-fix`) + changelog files present. |
| A17 | ✅ | Clean-room verification completed (`reports/cleanroom_verification.md`). |
| A18 | ✅ | Pilot delivery folder standard exists under `reports/pilots/`. |
| A19 | ❌ | Internal legal cleanroom record exists (`reports/legal_cleanroom_signoff_internal.md`), but external counsel sign-off is still pending. |
| A20 | ❌ | Requires 2 paid pilots or 2 signed LOIs; this is commercial execution outside repo. |

## B) Product/Benchmark Claim Readiness (B1-B10)

| ID | Status | Evidence / Reason |
| --- | --- | --- |
| B1 | ❌ | Needs real staged dataset snapshots + final pinned hashes from production data flow. |
| B2 | ❌ | Needs full pretrain/finetune run on target training hardware. |
| B3 | ❌ | Requires trained production checkpoint artifacts. |
| B4 | ❌ | Requires benchmark outputs generated from trained checkpoints. |
| B5 | ❌ | Requires real-device latency/power measurements. |
| B6 | ✅ | Claim language is separated (measured vs target/estimate) in docs. |
| B7 | ❌ | Third-party reproduction requires an external validator team. |
| B8 | ❌ | Internal teacher/output assessment exists (`reports/teacher_output_license_assessment.md`), but final commercial legal approval remains external pending. |
| B9 | ❌ | Security/pentest report requires dedicated external security assessment scope. |
| B10 | ❌ | SLA/incident/rollback contract package requires business/legal execution. |

## C) Pilot Delivery Bundle (Per-Customer)

| Item | Status | Evidence |
| --- | --- | --- |
| verify_all summary/log | ✅ | `scripts/verify_all.sh` + pilot report payload |
| operator gate JSON summary | ✅ | `scripts/operator_mode_gate.py` output |
| pilot_report_v1 JSON | ✅ | `interfaces/pilot_report_v1.schema.json` |
| offline rerun instructions | ✅ | `USAGE_GUIDE.md` |
| risk/limit note | ✅ | README + benchmark eligibility gate |
| acceptance signoff page | ✅ | `reports/pilot_acceptance_signoff.md` |

## Current Gate Decision
- **Pilot technical readiness:** GO
- **Commercial pilot closure readiness:** Pending A19 + A20
- **Product/benchmark claim readiness:** Pending B1-B5, B7-B10
