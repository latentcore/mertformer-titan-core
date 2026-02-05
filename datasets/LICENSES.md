# Dataset Licenses (Pre-Training Checklist)

All datasets must comply with their original licenses and terms (including gated datasets).
This table is the **single checklist** used before any real training run.

Notes:
- “TBD” entries are **blockers** for a production training run until verified on the dataset card / upstream repository.
- Snapshot hashes live in `datasets/hashes.json` and must be filled before training.

| Dataset | License | Reference URL | Status |
| --- | --- | --- | --- |
| `bigcode/the-stack-v2` | TBD (verify on dataset card; may include multiple terms) | https://huggingface.co/datasets/bigcode/the-stack-v2 | TBD |
| `TIGER-Lab/MathInstruct` | TBD (verify on dataset card) | https://huggingface.co/datasets/TIGER-Lab/MathInstruct | TBD |
| `openai/gsm8k` (`main`) | MIT | https://opensource.org/licenses/MIT | Verified (known) |
| `HuggingFaceFW/fineweb-edu` | TBD (verify on dataset card) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | TBD |
| `wikimedia/wikipedia` (`20231101.tr`) | CC BY-SA 4.0 + GFDL (dual) | https://foundation.wikimedia.org/wiki/Terms_of_Use | Verified (Wikipedia terms) |
| `uonlp/CulturaX` (`tr`) | TBD (verify on dataset card; derived corpora apply) | https://huggingface.co/datasets/uonlp/CulturaX | TBD |
| `HuggingFaceTB/cosmopedia` (`stories`) | TBD (verify on dataset card) | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia | TBD |
| `OpenAssistant/oasst_top1_2023-08-25` | TBD (verify on dataset card) | https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25 | TBD |
| `mlabonne/guanaco-llama2-1k` | TBD (verify on dataset card) | https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k | TBD |
| `TFLai/Turkish-Alpaca` | TBD (verify on dataset card) | https://huggingface.co/datasets/TFLai/Turkish-Alpaca | TBD |
| `turkish-nlp-suite/InstrucTurca` | TBD (verify on dataset card) | https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca | TBD |
| `glaiveai/glaive-function-calling-v2` | TBD (verify on dataset card) | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | TBD |
| `openai_humaneval` | MIT | https://opensource.org/licenses/MIT | Verified (known) |
| `mbpp` (`sanitized`) | CC-BY-4.0 | https://creativecommons.org/licenses/by/4.0/ | Verified (known) |
| `wikitext` (`wikitext-2-raw-v1`) | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (known) |
| Internal stage sets | Internal | Internal policy doc | TBD (write policy) |
| Golden samples | Internal | Internal policy doc | TBD (write policy) |
