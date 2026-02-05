# Dataset Sources (Pre-Training Inventory)

This file lists **known sources referenced by code** so far. Exact snapshots, dates,
and hashes must be filled in **before** any production training run.

## Primary corpora (from `scripts/data_pipeline.py` / preflight)
- `uonlp/CulturaX` (Turkish subset)  
  Purpose: large-scale Turkish web corpus  
  Dataset card: https://huggingface.co/datasets/uonlp/CulturaX  
  Status: source referenced in code, snapshot TBD
- `wikimedia/wikipedia` (Turkish)  
  Purpose: clean encyclopedia text  
  Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia  
  Status: source referenced in code, snapshot TBD

## Evaluation / benchmarks (from `scripts/benchmarks_internal.py`, `scripts/eval.py`)
- `openai/gsm8k`  
  Purpose: math reasoning evaluation  
  Dataset card: https://huggingface.co/datasets/openai/gsm8k  
  Status: source referenced in code, snapshot TBD
- `openai_humaneval`  
  Purpose: code generation evaluation  
  Dataset card: https://huggingface.co/datasets/openai_humaneval  
  Status: source referenced in code, snapshot TBD
- `mbpp` (sanitized)  
  Purpose: code generation evaluation  
  Dataset card: https://huggingface.co/datasets/mbpp  
  Status: source referenced in code, snapshot TBD

## Dev sanity datasets (from `scripts/mini_titan_poc.py`)
- `wikitext` (wikitext-2-raw-v1)  
  Purpose: tiny fast debug dataset  
  Dataset card: https://huggingface.co/datasets/wikitext  
  Status: source referenced in code, snapshot TBD

## Internal / custom
- Stage curriculum sets (stage1–stage5)  
  Purpose: curated internal curriculum  
  Status: internal, snapshot TBD
- Golden samples (internal prompts)  
  Purpose: regression checks  
  Status: internal, snapshot TBD
