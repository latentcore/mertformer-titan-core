# Chess Proof and Teaching Case Study

## One-Line Summary
The chess lane is a proof-and-teaching surface, not a final strength claim surface.

## Why This Lane Exists
This lane exists to prove three things:
1. benchmark honesty
2. pure-model versus product-mode separation
3. explanation and teaching contract discipline

## Canonical Technical Surface
Primary code and product paths:
- `scripts/chess_5080_onefile.py`
- `scripts/export_chess_5080_share.py`
- `apps/chess_gui/play_mertformer_chess_web.py`
- `apps/chess_gui/README.md`

Primary repo-side evidence:
- `reports/chess_training_readiness_report.md`
- `reports/chess_teaching_contract_report.md`
- `reports/chess_onefile_extension_report.md`
- `docs/CHESS_ONEFILE_MASTER_TRUTH.md`
- `CHESS_5080_POC_INTERNAL.md`

## What Is Actually Closed Repo-Side
1. The chess onefile lane is a real code path, not just a doc concept.
2. The training lane is marked `READY_FOR_TRAINING` at the repo level.
3. The GUI product lane exists and uses the repo-canonical onefile as source of truth.
4. Teaching and explanation contracts are testable and smoke-verified.
5. Extension surfaces for curated positions and synthetic teaching corpus are present.
6. Export and delivery helpers exist without pretending the lane is final-release complete.

## Measured Repo-Side Evidence
- `reports/chess_training_readiness_report.md`: `READY_FOR_TRAINING`, required green checks `7/7`.
- `reports/chess_teaching_contract_report.md`: `all_green=True`, `case_pass=5/5`, `mode_pass=5/5`, `level_monotonic_non_decreasing=True`.
- `reports/chess_onefile_extension_report.md`: `final_status=READY`, `curated_position_count=11`, `curated_training_examples=66`, `synthetic_teaching_records=33`.

## Pure Model vs Product Mode
This repo intentionally separates:
- pure training and verification surfaces in `scripts/chess_5080_onefile.py`
- product-style local GUI inspection in `apps/chess_gui/play_mertformer_chess_web.py`
- delivery and packaging in `scripts/export_chess_5080_share.py`

That separation matters because replay output, GUI polish, and benchmark theater are not automatically strength proof.

## Benchmark Honesty Rules
This lane already encodes benchmark honesty in the code and docs:
- internal Stockfish gauntlet is explicitly marked internal and proxy-like
- replay output is not treated as strength proof
- arena mode is not treated as strength proof
- missing or limited benchmarking keeps rating claims in `no_claim` or `proxy_only` territory

## Why This Lane Is Useful For Anthropic
Anthropic does not need a flashy chess project. It benefits more from seeing that:
- you can keep a benchmark lane honest
- you can separate training truth from product presentation
- you can make explanations contractual rather than purely aesthetic
- you can resist the urge to overclaim when the fun path would be to overclaim

## Interview-Ready Chess Story
If asked why this lane matters, the honest answer is:

"I used chess as a proof-and-teaching environment, not as a fake Elo trophy. The repo has a real training path, a local GUI path, a teaching contract, and benchmark-guardrail language that keeps product mode and strength proof separate. That is useful because it shows eval discipline, explanation discipline, and systems discipline in one contained lane."

## Claim Boundary
This case study does not claim:
- verified external Elo
- production-grade chess strength
- release-grade benchmark closure
- final trained checkpoint truth for the core repo

It claims the repo has a serious chess proof lane with explicit teaching and benchmark contracts.
