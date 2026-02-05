# Dataset Sources (Pre-Training Inventory)

This file lists **known sources referenced by code** so far. Exact snapshots, dates,
and hashes must be filled in **before** any production training run.

## Primary corpora (from `scripts/data_pipeline.py` / preflight)
- `uonlp/CulturaX` (Turkish subset)  
  Purpose: large-scale Turkish web corpus  
  Status: source referenced in code, snapshot TBD
- `wikimedia/wikipedia` (Turkish)  
  Purpose: clean encyclopedia text  
  Status: source referenced in code, snapshot TBD

## Evaluation / benchmarks (from `scripts/benchmarks_internal.py`, `scripts/eval.py`)
- `openai/gsm8k`  
  Purpose: math reasoning evaluation  
  Status: source referenced in code, snapshot TBD
- `openai_humaneval`  
  Purpose: code generation evaluation  
  Status: source referenced in code, snapshot TBD
- `mbpp` (sanitized)  
  Purpose: code generation evaluation  
  Status: source referenced in code, snapshot TBD

## Dev sanity datasets (from `scripts/mini_titan_poc.py`)
- `wikitext` (wikitext-2-raw-v1)  
  Purpose: tiny fast debug dataset  
  Status: source referenced in code, snapshot TBD

## Internal / custom
- Stage curriculum sets (stage1–stage5)  
  Purpose: curated internal curriculum  
  Status: internal, snapshot TBD
- Golden samples (internal prompts)  
  Purpose: regression checks  
  Status: internal, snapshot TBD
