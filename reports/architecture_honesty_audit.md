# Architecture Honesty Audit

- generated_utc: `2026-04-14T22:29:15Z`
- claim boundary: `pre-training / not eligible for claim without a trained checkpoint`

## Current Honesty Rules

- Treat `2.64B` as the design target, not the measured runtime total.
- Treat `~3.70B` as the current measured runtime total when factual parameter claims are made.
- Treat the 45K run as the first serious architecture validation run, not the final capability ceiling.
- Do not convert deployment vision or benchmark scaffolding into trained-model claims.
