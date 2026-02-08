# Dataset Envanteri (Otomatik)

Bu dosya, kod tabaninda referanslanan dataset kimliklerinin otomatik envanteridir (best-effort).
Lisans/provenans dogrulamasi için `datasets/SOURCES*.md` ve `datasets/LICENSES*.md` dosyalarıni referans alin.

| Dataset | License (best-effort) | HF URL | Refs |
| --- | --- | --- | --- |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 1 |
| `HuggingFaceTB/cosmopedia` | Apache-2.0 | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia | 1 |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25 | 1 |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://huggingface.co/datasets/TFLai/Turkish-Alpaca | 1 |
| `TIGER-Lab/MathInstruct` | MIT | https://huggingface.co/datasets/TIGER-Lab/MathInstruct | 1 |
| `bigcode/the-stack-v2` | Other (mixed upstream licenses; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-v2 | 1 |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | 1 |
| `mbpp` | CC-BY-4.0 | https://huggingface.co/datasets/mbpp | 1 |
| `mlabonne/guanaco-llama2-1k` | Apache-2.0 (inherits `timdettmers/openassistant-guanaco`) | https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k | 1 |
| `openai/gsm8k` | MIT | https://huggingface.co/datasets/openai/gsm8k | 4 |
| `openai_humaneval` | MIT | https://huggingface.co/datasets/openai_humaneval | 1 |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca | 1 |
| `uonlp/CulturaX` | ODC-By 1.0 + CC0-1.0 (inherits mC4 + OSCAR) | https://huggingface.co/datasets/uonlp/CulturaX | 2 |
| `wikimedia/wikipedia` | CC BY-SA 4.0 + GFDL (dual) | https://huggingface.co/datasets/wikimedia/wikipedia | 1 |
| `wikitext` | CC BY-SA 4.0 | https://huggingface.co/datasets/wikitext | 1 |

## Reference Details

### `HuggingFaceFW/fineweb-edu`
- scripts/data_pipeline.py:82 (pipeline_source)

### `HuggingFaceTB/cosmopedia`
- scripts/data_pipeline.py:120 (pipeline_source)

### `OpenAssistant/oasst_top1_2023-08-25`
- scripts/data_pipeline.py:139 (pipeline_source)

### `TFLai/Turkish-Alpaca`
- scripts/data_pipeline.py:157 (pipeline_source)

### `TIGER-Lab/MathInstruct`
- scripts/data_pipeline.py:56 (pipeline_source)

### `bigcode/the-stack-v2`
- scripts/data_pipeline.py:47 (pipeline_source)

### `glaiveai/glaive-function-calling-v2`
- scripts/data_pipeline.py:182 (pipeline_source)

### `mbpp`
- scripts/benchmarks_internal.py:106 (load_dataset_safe)

### `mlabonne/guanaco-llama2-1k`
- scripts/data_pipeline.py:148 (pipeline_source)

### `openai/gsm8k`
- scripts/data_pipeline.py:65 (pipeline_source)
- scripts/eval.py:40 (load_dataset)
- eval/gsm8k.py:66 (load_dataset)
- eval/gsm8k.py:68 (load_dataset)

### `openai_humaneval`
- scripts/benchmarks_internal.py:105 (load_dataset_safe)

### `turkish-nlp-suite/InstrucTurca`
- scripts/data_pipeline.py:166 (pipeline_source)

### `uonlp/CulturaX`
- scripts/data_pipeline.py:110 (pipeline_source)
- scripts/titan_preflight.py:185 (dataset_info)

### `wikimedia/wikipedia`
- scripts/data_pipeline.py:100 (pipeline_source)

### `wikitext`
- scripts/mini_titan_poc.py:367 (load_dataset)

