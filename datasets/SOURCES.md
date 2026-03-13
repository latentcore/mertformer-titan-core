# Dataset Sources (Pre-Training Inventory)

This file lists **dataset sources referenced by code** and the intended curriculum stages.
It is a compliance checklist (provenance + license + snapshot), not a training guarantee.

See also:
- `datasets/inventory.md` (auto-extracted references from code)
- `datasets/LICENSES.md` (license table)
- `datasets/hashes.json` (snapshot hashes; must be filled before any real training run)

## Training Curriculum (from `scripts/data_pipeline.py`)

### Stage 1 — Logic / Code + Math (target ratio: 42%)
- `bigcode/the-stack-v2` (train; filtered by language)
  Purpose: large-scale code corpus
  Dataset card: https://huggingface.co/datasets/bigcode/the-stack-v2
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `TIGER-Lab/MathInstruct` (train)
  Purpose: math instruction data
  Dataset card: https://huggingface.co/datasets/TIGER-Lab/MathInstruct
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `openai/gsm8k` (train; subset `main`)
  Purpose: math reasoning (also used for eval)
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`

### Stage 2 — World Knowledge (target ratio: 30%)
- `HuggingFaceFW/fineweb-edu` (train)
  Purpose: educational web corpus
  Dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`

### Stage 3 — Identity & Language (target ratio: 8%)
- `wikimedia/wikipedia` (train; subset `20231101.tr`)
  Purpose: clean Turkish encyclopedia text
  Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `uonlp/CulturaX` (train; subset `tr`)
  Purpose: large-scale Turkish web corpus
  Dataset card: https://huggingface.co/datasets/uonlp/CulturaX
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `HuggingFaceTB/cosmopedia` (train; subset `stories`)
  Purpose: synthetic high-quality text
  Dataset card: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`

### Stage 4 — Soul / Instruction (target ratio: 8%)
- `OpenAssistant/oasst_top1_2023-08-25` (train)
  Purpose: high-quality human dialog / instruction
  Dataset card: https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `TFLai/Turkish-Alpaca` (train)
  Purpose: Turkish instruction following
  Dataset card: https://huggingface.co/datasets/TFLai/Turkish-Alpaca
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `turkish-nlp-suite/InstrucTurca` (train)
  Purpose: Turkish instruction data
  Dataset card: https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `teknium/OpenHermes-2.5` (train; optional)
  Purpose: general instruction set (optional fallback)
  Dataset card: https://huggingface.co/datasets/teknium/OpenHermes-2.5
  Status: referenced in code; verify license in `datasets/LICENSES.md`

### Stage 5 — Tools / Function Calling (target ratio: 12%)
- `glaiveai/glaive-function-calling-v2` (train)
  Purpose: function calling / tool use
  Dataset card: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `gorilla-llm/gorilla-openfunctions-v2` (train; optional)
  Purpose: tool-use diversity (optional/gated)
  Dataset card: https://huggingface.co/datasets/gorilla-llm/gorilla-openfunctions-v2
  Status: referenced in code; verify license in `datasets/LICENSES.md`
- `NousResearch/FC-1k` (train; optional)
  Purpose: lightweight function-calling expansion
  Dataset card: https://huggingface.co/datasets/NousResearch/FC-1k
  Status: referenced in code; verify license in `datasets/LICENSES.md`

## Evaluation / Benchmarks (from `scripts/benchmarks_internal.py`, `eval/gsm8k.py`)
- `openai_humaneval` (test)
  Purpose: code generation evaluation
  Dataset card: https://huggingface.co/datasets/openai/openai_humaneval
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `mbpp` (test; subset `sanitized`)
  Purpose: code generation evaluation
  Dataset card: https://huggingface.co/datasets/mbpp
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`
- `openai/gsm8k` (test; subset `main`)
  Purpose: math reasoning evaluation
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`

## Dev sanity datasets (from `scripts/mini_titan_poc.py`)
- `wikitext` (train; subset `wikitext-2-raw-v1`)
  Purpose: tiny fast debug dataset
  Dataset card: https://huggingface.co/datasets/wikitext
  Status: referenced in code; revision pinned + manifest fingerprint recorded in `datasets/hashes.json`

## Internal / custom
- Stage curriculum sets (stage1–stage5) (local jsonl snapshots)
  Purpose: curated internal curriculum outputs (after download/filtering)
  Status: internal (gitignored); hash locally after generation (see `scripts/record_dataset_hashes.py`)
- Golden samples (internal prompts)
  Purpose: regression checks
  Status: internal (tracked) + SHA256 recorded in `datasets/hashes.json`
