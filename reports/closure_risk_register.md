# Closure Risk Register

## High

- `repo:012` Reject harmful autonomy and covert surveillance framing -> Conflicts with public-good / high-risk guardrail.

## Medium

- `repo:004` Unify run.sh, one_command_full_sop.sh, and final_one_shot.sh into one canonical closure flow -> Directly affects the canonical closure flow.
- `repo:005` Produce one final TRAIN_ALLOWED or NOT_ALLOWED verdict with reason codes -> Relevant to this pass and does not exceed the risk ceiling.
- `repo:009` Keep dual-path readiness for offline-clean and online-teacher flows -> Directly affects the 45K readiness gate.
- `repo:008` Declare feature freeze, config freeze, dataset freeze, tokenizer freeze, and teacher/logits decision -> Directly improves closure confidence and documentation truth.
