# Dataset Sources (Pre-Training Inventory)

This file lists **known sources referenced by code** so far. Licenses are tracked in
`datasets/LICENSES.md`. Exact snapshots, dates, and hashes must be recorded **before**
any production training run.

## Primary corpora (from `scripts/data_pipeline.py` / preflight)
- `uonlp/CulturaX` (Turkish subset)
  Purpose: large-scale Turkish web corpus
  Dataset card: https://huggingface.co/datasets/uonlp/CulturaX
  Status: referenced in code; snapshot + hash pending
- `wikimedia/wikipedia` (Turkish)
  Purpose: clean encyclopedia text
  Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia
  Status: referenced in code; snapshot + hash pending

## Evaluation / benchmarks (from `scripts/benchmarks_internal.py`, `eval/gsm8k.py`)
- `openai/gsm8k`
  Purpose: math reasoning evaluation
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Status: referenced in code; snapshot + hash pending
- `openai_humaneval`
  Purpose: code generation evaluation
  Dataset card: https://huggingface.co/datasets/openai/openai_humaneval
  Status: referenced in code; snapshot + hash pending
- `mbpp` (sanitized)
  Purpose: code generation evaluation
  Dataset card: https://huggingface.co/datasets/mbpp
  Status: referenced in code; snapshot + hash pending

## Dev sanity datasets (from `scripts/mini_titan_poc.py`)
- `wikitext` (wikitext-2-raw-v1)
  Purpose: tiny fast debug dataset
  Dataset card: https://huggingface.co/datasets/wikitext
  Status: referenced in code; snapshot + hash pending

## Internal / custom
- Stage curriculum sets (stage1–stage5)
  Purpose: curated internal curriculum
  Status: internal; snapshot + hash pending
- Golden samples (internal prompts)
  Purpose: regression checks
  Status: internal; snapshot + hash pending
