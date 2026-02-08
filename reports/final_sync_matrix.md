# Final Sync Matrix (Build 27)

This file is the final consistency ledger for documentation, version labels, and verification runs.

## 1) EN/TR Markdown Pairing

Rule: every tracked `X.md` has `X_TR.md` pair where applicable.

| Check | Result |
| --- | --- |
| Missing pairs | 0 |
| Orphan `_TR.md` files | 1 (`reports/codex_deep_audit_TR.md`, intentional canonical TR audit) |
| Added in this final pass | `reports/codex_deep_audit_EN_TR.md`, `reports/codex_deep_audit_DE_TR.md`, `USAGE_GUIDE_TR.md`, `reports/final_sync_matrix_TR.md` |

## 2) Build Label Consistency

| Scope | Result |
| --- | --- |
| User-visible version references | Build 27 aligned |
| Legacy references in active user-facing scripts | normalized |
| Lineage references | retained only where clearly marked roadmap/history |

## 3) Verification Runs (Release Candidate)

| Command | Status |
| --- | --- |
| `.titan-venv/bin/python -m pytest -q` | PASS |
| `bash scripts/verify_all.sh` | PASS |
| `TITAN_OFFLINE=1 bash run.sh --test` | PASS |
| `.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | PASS |

## 4) Demo Proof

| Artifact | Status |
| --- | --- |
| `assets/snake_demo_proof.mp4` | generated |
| `snake_demo.py` auto-restart + telemetry | verified |
| README links | updated |

## 5) Packaging Gates

| Gate | Status |
| --- | --- |
| Clean zip excludes venv/cache/log/.env | PASS |
| Hamdi package includes extra `AUDIT_MEMO.md` only in package | PASS |
| SHA-256 created for generated zips | PASS |

## 6) Release Identifiers

- Final git SHA (main): `8d29263`
- Clean zip: `./NIHAI_clean_build27.zip`
- Clean zip SHA-256: `bf4ac8a263fb451e3058858307e311409de76999c9bc8ad61d5c85c37d05e3c2`
- Hamdi zip: `./NIHAI_hamdi_clean.zip`
- Hamdi zip SHA-256: `ae8b522415436288306e389ddcec2a34c987f18e5f1c517556d788ad34e07dba`
