# Master Closure Matrix

This matrix combines repo closure obligations with actionable items mined from the desktop TXT export.

- txt_source: `missing`
- total_items: `14`
- this_pass: `12`
- phase-2: `0`
- external: `1`
- rejected-with-reason: `1`

| ID | Source | Phase | Risk | Category | Text | Reason | Acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `repo:001` | `implementation-plan` | `this-pass` | `low` | `closure_flow` | Build and maintain a dependency-ordered master closure matrix | Directly affects the canonical closure flow. | `bash scripts/final_one_shot.sh` |
| `repo:004` | `implementation-plan` | `this-pass` | `medium` | `closure_flow` | Unify run.sh, one_command_full_sop.sh, and final_one_shot.sh into one canonical closure flow | Directly affects the canonical closure flow. | `bash scripts/final_one_shot.sh` |
| `repo:005` | `implementation-plan` | `this-pass` | `medium` | `handoff` | Produce one final TRAIN_ALLOWED or NOT_ALLOWED verdict with reason codes | Relevant to this pass and does not exceed the risk ceiling. | `.titan-venv/bin/python scripts/build_max_closure_handoff.py` |
| `repo:011` | `implementation-plan` | `this-pass` | `low` | `handoff` | Make public-good, auditable, human-benefiting deployment the official framing | Relevant to this pass and does not exceed the risk ceiling. | `.titan-venv/bin/python scripts/build_max_closure_handoff.py` |
| `repo:014` | `implementation-plan` | `this-pass` | `low` | `handoff` | Use atomic thematic commits instead of a single mega-commit | Relevant to this pass and does not exceed the risk ceiling. | `.titan-venv/bin/python scripts/build_max_closure_handoff.py` |
| `repo:003` | `implementation-plan` | `this-pass` | `low` | `training_readiness` | Keep 45K readiness as the primary ship gate for this pass | Directly affects the 45K readiness gate. | `.titan-venv/bin/python scripts/build_train_readiness_contract.py` |
| `repo:009` | `implementation-plan` | `this-pass` | `medium` | `training_readiness` | Keep dual-path readiness for offline-clean and online-teacher flows | Directly affects the 45K readiness gate. | `.titan-venv/bin/python scripts/build_train_readiness_contract.py` |
| `repo:006` | `implementation-plan` | `this-pass` | `low` | `truth_claim` | Expand claim discipline across measured, target, vision, verified, hypothesis, and creative/folklore surfaces | Directly improves closure confidence and documentation truth. | `.titan-venv/bin/python scripts/check_doc_claim_consistency.py` |
| `repo:007` | `implementation-plan` | `this-pass` | `low` | `truth_claim` | Enforce no claim without evidence across final docs and reports | Directly improves closure confidence and documentation truth. | `.titan-venv/bin/python scripts/check_doc_claim_consistency.py` |
| `repo:008` | `implementation-plan` | `this-pass` | `medium` | `truth_claim` | Declare feature freeze, config freeze, dataset freeze, tokenizer freeze, and teacher/logits decision | Directly improves closure confidence and documentation truth. | `.titan-venv/bin/python scripts/check_doc_claim_consistency.py` |
| `repo:010` | `implementation-plan` | `this-pass` | `low` | `truth_claim` | Harden data pipeline provenance, optional source policy, token probe, and revision/hash lineage | Directly improves closure confidence and documentation truth. | `.titan-venv/bin/python scripts/check_doc_claim_consistency.py` |
| `repo:013` | `implementation-plan` | `this-pass` | `low` | `truth_claim` | Generate canonical repo handoff, final commands, risk list, and carryover list | Directly improves closure confidence and documentation truth. | `.titan-venv/bin/python scripts/check_doc_claim_consistency.py` |
| `repo:002` | `implementation-plan` | `external` | `low` | `external` | Classify every TXT and repo backlog item into this-pass, phase-2, external, or rejected-with-reason | Requires outside sign-off, commercial action, or external dependency. | `External sign-off / pilot evidence bundle` |
| `repo:012` | `implementation-plan` | `rejected-with-reason` | `high` | `policy` | Reject harmful autonomy and covert surveillance framing | Conflicts with public-good / high-risk guardrail. | `.titan-venv/bin/python scripts/check_doc_claim_consistency.py` |
