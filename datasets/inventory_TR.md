# Dataset Envanteri (Otomatik)

Bu dosya, kod tabaninda referanslanan dataset kimliklerinin otomatik envanteridir (best-effort).
Lisans/provenans dogrulamasi icin `datasets/SOURCES*.md` ve `datasets/LICENSES*.md` dosyalarini referans alin.

| Dataset | License (best-effort) | HF URL | Refs |
| --- | --- | --- | --- |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 1 |
| `HuggingFaceTB/cosmopedia` | Apache-2.0 | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia | 1 |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25 | 2 |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://huggingface.co/datasets/TFLai/Turkish-Alpaca | 1 |
| `TIGER-Lab/MathInstruct` | MIT | https://huggingface.co/datasets/TIGER-Lab/MathInstruct | 2 |
| `bigcode/the-stack-v2` | Other (mixed upstream licenses; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-v2 | 2 |
| `codeparrot/github-code` | Unknown | https://huggingface.co/datasets/codeparrot/github-code | 1 |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | 1 |
| `mbpp` | CC-BY-4.0 | https://huggingface.co/datasets/mbpp | 1 |
| `mlabonne/guanaco-llama2-1k` | Apache-2.0 (inherits `timdettmers/openassistant-guanaco`) | https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k | 1 |
| `openai/gsm8k` | MIT | https://huggingface.co/datasets/openai/gsm8k | 5 |
| `openai_humaneval` | MIT | https://huggingface.co/datasets/openai_humaneval | 1 |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca | 2 |
| `uonlp/CulturaX` | ODC-By 1.0 + CC0-1.0 (inherits mC4 + OSCAR) | https://huggingface.co/datasets/uonlp/CulturaX | 3 |
| `wikimedia/wikipedia` | CC BY-SA 4.0 + GFDL (dual) | https://huggingface.co/datasets/wikimedia/wikipedia | 2 |
| `wikitext` | CC BY-SA 4.0 | https://huggingface.co/datasets/wikitext | 1 |

## Reference Details

### `HuggingFaceFW/fineweb-edu`
- scripts/data_pipeline.py:89 (pipeline_source)

### `HuggingFaceTB/cosmopedia`
- scripts/data_pipeline.py:127 (pipeline_source)

### `OpenAssistant/oasst_top1_2023-08-25`
- scripts/data_pipeline.py:146 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2373 (pipeline_source)

### `TFLai/Turkish-Alpaca`
- scripts/data_pipeline.py:164 (pipeline_source)

### `TIGER-Lab/MathInstruct`
- scripts/data_pipeline.py:63 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2343 (pipeline_source)

### `bigcode/the-stack-v2`
- scripts/data_pipeline.py:54 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2361 (pipeline_source)

### `codeparrot/github-code`
- scripts/kaggle_onefile_demo_build30.py:2356 (pipeline_source)

### `glaiveai/glaive-function-calling-v2`
- scripts/data_pipeline.py:189 (pipeline_source)

### `mbpp`
- scripts/benchmarks_internal.py:127 (load_dataset_safe)

### `mlabonne/guanaco-llama2-1k`
- scripts/data_pipeline.py:155 (pipeline_source)

### `openai/gsm8k`
- scripts/data_pipeline.py:72 (pipeline_source)
- scripts/eval.py:40 (load_dataset)
- scripts/kaggle_onefile_demo_build30.py:2342 (pipeline_source)
- eval/gsm8k.py:66 (load_dataset)
- eval/gsm8k.py:68 (load_dataset)

### `openai_humaneval`
- scripts/benchmarks_internal.py:126 (load_dataset_safe)

### `turkish-nlp-suite/InstrucTurca`
- scripts/data_pipeline.py:173 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2374 (pipeline_source)

### `uonlp/CulturaX`
- scripts/data_pipeline.py:117 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2330 (pipeline_source)
- scripts/titan_preflight.py:346 (dataset_info)

### `wikimedia/wikipedia`
- scripts/data_pipeline.py:107 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2329 (pipeline_source)

### `wikitext`
- scripts/mini_titan_poc.py:367 (load_dataset)

