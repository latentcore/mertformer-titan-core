# Stage-3 Source Mixture Freeze Proposal

**DECIDED (2026-07-19): ratified, not reinvented.** When asked to decide now, investigation found `scripts/data_pipeline.py`'s `STAGE3_SOURCES` already encoded a concrete split: `wikimedia/wikipedia` (`20231101.tr`, ratio 0.015) + `uonlp/CulturaX` (`tr`, ratio 0.040) = 0.055 raw Turkish share, vs. `HuggingFaceTB/cosmopedia` (`stories`, English-synthetic, ratio 0.015) — runtime-normalized against their own sum, this is **~78.6% Turkish / ~21.4% synthetic** within Stage 3. This was a real, already-made choice sitting undocumented in code, not an open blank; rather than overwrite a working ratio with an invented number, this pass formally ratified the existing split as the frozen decision. No code changed. See `DECISIONS.md` "Eight launch-time decisions locked...". The proposal below is preserved as-is; its "what Stage-3 refers to" framing remains accurate.

**Purpose:** BACKLOG I.3 #28 — "Stage-3 kaynak karışımını precompute'tan ÖNCE bilinçle dondur" (consciously freeze the Stage-3 source mixture BEFORE precompute). Teacher-logit precompute (Phase-0, BACKLOG #25) is expensive and effectively one-shot — if the underlying corpus source mixture changes AFTER precompute runs, the precomputed logits no longer correspond to the actual training corpus, silently invalidating the whole precompute pass. This proposal exists so the mixture is deliberately locked before that cost is spent, not discovered to have drifted after.

## What "Stage-3" refers to

Per the repo's existing data-pipeline staging (`scripts/data_pipeline.py`, `datasets/inventory.md`), the corpus construction proceeds through source-selection stages; Stage-3 is the point where the final training-corpus source mixture (which datasets, in what proportion) is fixed before tokenization/precompute.

## Proposed frozen mixture (based on current repo state)

The current canonical corpus source is `bigcode/the-stack-dedup`, already re-pinned by revision + sha256 (Pass-4, commit `21366d6`, per memory/BACKLOG). This proposal is: **freeze the mixture as exactly the currently-pinned `the-stack-dedup` revision, with no additional sources mixed in**, unless a specific, documented reason to diversify is raised and recorded in DECISIONS.md before precompute starts.

## Why "no additional sources" by default

1. Every additional source multiplies dedup/license/consent audit surface (BACKLOG I.3 #22 corpus health scan, #75 data-consent audit) — each new source is its own compliance review, not a free addition.
2. The 23.6B target-token budget (`FACTS.json`) is already computed against the single pinned source's known size/composition; mixing in a second source without recomputing the token-budget math risks the exact `TITAN_STRICT_TOKEN_BUDGET` overshoot scenario the guard (now default-ON, see this session's launch-script fix) is meant to catch.
3. `validate_logit_alignment.py`'s ADR-0005 single-naming-mode guard (fixed this session) assumes shard naming is consistent across ONE precompute run over ONE corpus definition — mixing sources mid-stream is exactly the kind of drift that guard was built to catch, so it's simpler to avoid the scenario than rely on the guard to catch a self-inflicted mixture change.

## Action required before Phase-0 precompute starts

1. Confirm no second source is planned (if one is, resolve BACKLOG #22/#75 for it FIRST).
2. Record the freeze decision in DECISIONS.md with the exact revision/sha256 being frozen (mirroring how the `the-stack-dedup` re-pin itself was recorded).
3. Only then run `scripts/precompute_logits_topk.py` / `precompute_logits_parallel.py`.

**This proposal recommends a default (freeze as-is); it does not unilaterally lock the mixture — the DECISIONS.md entry is the actual freeze action, and that's a deliberate human step.**
