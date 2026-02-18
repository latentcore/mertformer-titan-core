# Datasets Overview

This directory contains the staged curriculum datasets and evaluation sets used by the pipeline.

## Structure
- `stage1/` ... `stage5/` curriculum stages
- `stage4_soul/`, `stage5_tools/` specialized subsets
- `golden_samples.jsonl` internal golden prompts
- `validation.jsonl` validation set

## Claim-Grade Validation Set
For smoke tests, tiny validation is acceptable. For benchmark/claim runs, build a representative set:

```bash
python3 scripts/build_validation_set.py --target-size 1500
python3 scripts/record_dataset_hashes.py
```

Training gate:
- `TITAN_CLAIM_MODE=1` enforces minimum validation size (`cfg.validation_min_samples_claim`).

## Provenance
See `datasets/SOURCES.md` and `datasets/LICENSES.md` for sources and licensing notes.
