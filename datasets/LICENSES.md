# Dataset Licenses (Pre-Training Checklist)

All datasets must comply with their original licenses and terms (including gated datasets).
This table is the **single checklist** used before any real training run.

Notes:
- “TBD” entries are **blockers** for a production training run until verified on the dataset card / upstream repository.
- Snapshot hashes live in `datasets/hashes.json` and must be filled before training.

| Dataset | License | Reference URL | Status |
| --- | --- | --- | --- |
| `bigcode/the-stack-v2` | Other (mixed upstream licenses; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-v2 | Verified (HF gated terms) |
| `TIGER-Lab/MathInstruct` | MIT | https://opensource.org/licenses/MIT | Verified (HF metadata) |
| `openai/gsm8k` (`main`) | MIT | https://opensource.org/licenses/MIT | Verified (known) |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://opendatacommons.org/licenses/by/1-0/ | Verified (HF metadata) |
| `wikimedia/wikipedia` (`20231101.tr`) | CC BY-SA 4.0 + GFDL (dual) | https://foundation.wikimedia.org/wiki/Terms_of_Use | Verified (Wikipedia terms) |
| `uonlp/CulturaX` (`tr`) | ODC-By 1.0 + CC0-1.0 (inherits mC4 + OSCAR) | https://huggingface.co/datasets/uonlp/CulturaX | Verified (dataset card license section) |
| `HuggingFaceTB/cosmopedia` (`stories`) | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `mlabonne/guanaco-llama2-1k` | Apache-2.0 (inherits `timdettmers/openassistant-guanaco`) | https://www.apache.org/licenses/LICENSE-2.0 | Verified (upstream dataset README) |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (HF metadata) |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `openai_humaneval` | MIT | https://opensource.org/licenses/MIT | Verified (known) |
| `mbpp` (`sanitized`) | CC-BY-4.0 | https://creativecommons.org/licenses/by/4.0/ | Verified (known) |
| `wikitext` (`wikitext-2-raw-v1`) | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (known) |
| Internal stage sets | Internal (proprietary) | `datasets/INTERNAL_POLICY.md` | Verified (internal policy) |
| Golden samples | Internal (proprietary) | `datasets/INTERNAL_POLICY.md` | Verified (internal policy) |
