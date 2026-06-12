<!-- MertFormer Titan PR template. The repo is claim-disciplined and gate-protected. -->

## What & why
<!-- One or two sentences. Link the issue / list item if any. -->

## Change class
- [ ] Behavior-preserving (refactor / docs / comments / tests)
- [ ] Behavior-changing (training/eval/data/config) — explain the impact below
- [ ] Claim-affecting (touches measured/target/vision numbers) — see checklist

## Checklist (CI mirrors these via `scripts/verify_all.sh`)
- [ ] `pytest -q` green (state the count; do not silently change it — `sync_test_stat_claims.py` propagates it)
- [ ] `ruff check .` clean
- [ ] Scoped `mypy` (`pyproject.toml [tool.mypy]`) Success
- [ ] No new contradictions: `scripts/check_facts_consistency.py` + `scripts/check_doc_claim_consistency.py` pass
- [ ] Canonical numbers unchanged OR updated in `reports/FACTS.json` (param 3,672,982,022; DEFAULT_PARAMS 2.64e9)
- [ ] measured/target/vision labels preserved; dated historical snapshots untouched

## Notes
<!-- Anything reviewers should know: deferred follow-ups, GPU-only paths not validated locally, etc. -->
