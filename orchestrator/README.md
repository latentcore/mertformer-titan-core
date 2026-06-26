# `orchestrator/` — out-of-scope, inert at 45K

Türkçe: [README_TR.md](README_TR.md)

**Scope boundary.** This package is a research/agentic scaffold. It is **out-of-scope for the canonical pre-training / 45K run** and is **inert on the training path**: it is not exercised by `train/train.py` and contributes **no parameters** to the model that the 45K run trains. It is retained for a possible future, separate phase only.

**No capability claim.** Nothing here is benchmark-verified, production-ready, or evidence-eligible. The agentic/AGI-direction surfaces are explicitly listed as out-of-scope pending in the closure matrix (see `reports/closure_57_matrix.md` and the `out_of_scope_pending_ids` rows referenced from the root `README.md`).

**Why it stays in the tree.** Removing it is deferred to post-45K cleanup (Pass 4 is sealed — "no Pass 5"). Until then it is documented here as out-of-scope so reviewers are not misled into reading it as part of the trained model.
