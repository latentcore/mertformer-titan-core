# Dataset Envanteri (Otomatik)

Bu dosya, kod tabanında referanslanan dataset kimliklerinin otomatik envanteridir (best-effort).
Lisans/provenans doğrulaması için `datasets/SOURCES*.md` ve `datasets/LICENSES*.md` dosyalarını referans alın.

| Dataset | License (best-effort) | HF URL | Refs |
| --- | --- | --- | --- |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 1 |
| `HuggingFaceTB/cosmopedia` | Apache-2.0 | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia | 1 |
| `NousResearch/FC-1k` | TBD | https://huggingface.co/datasets/NousResearch/FC-1k | 1 |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25 | 3 |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://huggingface.co/datasets/TFLai/Turkish-Alpaca | 1 |
| `TIGER-Lab/MathInstruct` | MIT | https://huggingface.co/datasets/TIGER-Lab/MathInstruct | 3 |
| `bigcode/the-stack-dedup` | Other (mixed upstream licenses; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-dedup | 3 |
| `codeparrot/github-code` | TBD (demo-only) | https://huggingface.co/datasets/codeparrot/github-code | 2 |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | 1 |
| `gorilla-llm/gorilla-openfunctions-v2` | TBD | https://huggingface.co/datasets/gorilla-llm/gorilla-openfunctions-v2 | 1 |
| `mbpp` | CC-BY-4.0 | https://huggingface.co/datasets/mbpp | 1 |
| `openai/gsm8k` | MIT | https://huggingface.co/datasets/openai/gsm8k | 6 |
| `openai_humaneval` | MIT | https://huggingface.co/datasets/openai_humaneval | 1 |
| `teknium/OpenHermes-2.5` | TBD | https://huggingface.co/datasets/teknium/OpenHermes-2.5 | 1 |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca | 3 |
| `uonlp/CulturaX` | ODC-By 1.0 + CC0-1.0 (inherits mC4 + OSCAR) | https://huggingface.co/datasets/uonlp/CulturaX | 4 |
| `wikimedia/wikipedia` | CC BY-SA 4.0 + GFDL (dual) | https://huggingface.co/datasets/wikimedia/wikipedia | 3 |
| `wikitext` | CC BY-SA 4.0 | https://huggingface.co/datasets/wikitext | 1 |

## Reference Details

### `HuggingFaceFW/fineweb-edu`
- scripts/data_pipeline.py:91 (pipeline_source)

### `HuggingFaceTB/cosmopedia`
- scripts/data_pipeline.py:129 (pipeline_source)

### `NousResearch/FC-1k`
- scripts/data_pipeline.py:209 (pipeline_source)

### `OpenAssistant/oasst_top1_2023-08-25`
- scripts/data_pipeline.py:146 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2404 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2948 (pipeline_source)

### `TFLai/Turkish-Alpaca`
- scripts/data_pipeline.py:155 (pipeline_source)

### `TIGER-Lab/MathInstruct`
- scripts/data_pipeline.py:65 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2374 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2918 (pipeline_source)

### `bigcode/the-stack-dedup`
- scripts/data_pipeline.py:56 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2392 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2936 (pipeline_source)

### `codeparrot/github-code`
- scripts/kaggle_onefile_demo_build30.py:2387 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2931 (pipeline_source)

### `glaiveai/glaive-function-calling-v2`
- scripts/data_pipeline.py:190 (pipeline_source)

### `gorilla-llm/gorilla-openfunctions-v2`
- scripts/data_pipeline.py:199 (pipeline_source)

### `mbpp`
- scripts/benchmarks_internal.py:127 (load_dataset_safe)

### `openai/gsm8k`
- scripts/data_pipeline.py:74 (pipeline_source)
- scripts/eval.py:40 (load_dataset)
- scripts/kaggle_onefile_demo_build30.py:2373 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2917 (pipeline_source)
- eval/gsm8k.py:66 (load_dataset)
- eval/gsm8k.py:68 (load_dataset)

### `openai_humaneval`
- scripts/benchmarks_internal.py:126 (load_dataset_safe)

### `teknium/OpenHermes-2.5`
- scripts/data_pipeline.py:173 (pipeline_source)

### `turkish-nlp-suite/InstrucTurca`
- scripts/data_pipeline.py:164 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2405 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2949 (pipeline_source)

### `uonlp/CulturaX`
- scripts/data_pipeline.py:119 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2361 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2905 (pipeline_source)
- scripts/titan_preflight.py:444 (dataset_info)

### `wikimedia/wikipedia`
- scripts/data_pipeline.py:109 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2360 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:2904 (pipeline_source)

### `wikitext`
- scripts/mini_titan_poc.py:367 (load_dataset)

