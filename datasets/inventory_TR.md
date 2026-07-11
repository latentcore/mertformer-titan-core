# Dataset Envanteri (Otomatik)

Bu dosya, kod tabanında referanslanan dataset kimliklerinin otomatik envanteridir (best-effort).
Lisans/provenans doğrulaması için `datasets/SOURCES*.md` ve `datasets/LICENSES*.md` dosyalarını referans alın.

| Dataset | License (best-effort) | HF URL | Refs |
| --- | --- | --- | --- |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | 2 |
| `HuggingFaceH4/ultrachat_200k` | Unknown | https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | 1 |
| `HuggingFaceTB/cosmopedia` | Apache-2.0 | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia | 1 |
| `HuggingFaceTB/smoltalk` | Unknown | https://huggingface.co/datasets/HuggingFaceTB/smoltalk | 1 |
| `NousResearch/FC-1k` | TBD | https://huggingface.co/datasets/NousResearch/FC-1k | 1 |
| `OpenAssistant/oasst1` | Unknown | https://huggingface.co/datasets/OpenAssistant/oasst1 | 1 |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25 | 4 |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://huggingface.co/datasets/TFLai/Turkish-Alpaca | 1 |
| `TIGER-Lab/MathInstruct` | MIT | https://huggingface.co/datasets/TIGER-Lab/MathInstruct | 4 |
| `bigcode/the-stack-dedup` | Other (mixed upstream licenses; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-dedup | 4 |
| `codeparrot/github-code` | TBD | https://huggingface.co/datasets/codeparrot/github-code | 3 |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | 1 |
| `gorilla-llm/gorilla-openfunctions-v2` | TBD | https://huggingface.co/datasets/gorilla-llm/gorilla-openfunctions-v2 | 1 |
| `mbpp` | CC-BY-4.0 | https://huggingface.co/datasets/mbpp | 1 |
| `openai/gsm8k` | MIT | https://huggingface.co/datasets/openai/gsm8k | 8 |
| `openai_humaneval` | MIT | https://huggingface.co/datasets/openai_humaneval | 1 |
| `teknium/OpenHermes-2.5` | TBD | https://huggingface.co/datasets/teknium/OpenHermes-2.5 | 1 |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca | 4 |
| `uonlp/CulturaX` | ODC-By 1.0 + CC0-1.0 (inherits mC4 + OSCAR) | https://huggingface.co/datasets/uonlp/CulturaX | 5 |
| `wikimedia/wikipedia` | CC BY-SA 4.0 + GFDL (dual) | https://huggingface.co/datasets/wikimedia/wikipedia | 5 |
| `wikitext` | CC BY-SA 4.0 | https://huggingface.co/datasets/wikitext | 1 |
| `{DATASET_ID}` | Unknown | https://huggingface.co/datasets/{DATASET_ID} | 1 |

## Reference Details

### `HuggingFaceFW/fineweb-edu`
- scripts/data_pipeline.py:93 (pipeline_source)
- scripts/mertformer_5080_final_onefile.py:7616 (pipeline_source)

### `HuggingFaceH4/ultrachat_200k`
- scripts/mertformer_5080_final_onefile.py:7641 (pipeline_source)

### `HuggingFaceTB/cosmopedia`
- scripts/data_pipeline.py:131 (pipeline_source)

### `HuggingFaceTB/smoltalk`
- scripts/mertformer_5080_final_onefile.py:7640 (pipeline_source)

### `NousResearch/FC-1k`
- scripts/data_pipeline.py:211 (pipeline_source)

### `OpenAssistant/oasst1`
- scripts/mertformer_5080_final_onefile.py:7642 (pipeline_source)

### `OpenAssistant/oasst_top1_2023-08-25`
- scripts/data_pipeline.py:148 (pipeline_source)
- scripts/kaggle_onecell_t4_build30.py:2732 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2440 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3137 (pipeline_source)

### `TFLai/Turkish-Alpaca`
- scripts/data_pipeline.py:157 (pipeline_source)

### `TIGER-Lab/MathInstruct`
- scripts/data_pipeline.py:67 (pipeline_source)
- scripts/kaggle_onecell_t4_build30.py:2702 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2410 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3107 (pipeline_source)

### `bigcode/the-stack-dedup`
- scripts/data_pipeline.py:58 (pipeline_source)
- scripts/kaggle_onecell_t4_build30.py:2720 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2428 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3125 (pipeline_source)

### `codeparrot/github-code`
- scripts/kaggle_onecell_t4_build30.py:2715 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2423 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3120 (pipeline_source)

### `glaiveai/glaive-function-calling-v2`
- scripts/data_pipeline.py:192 (pipeline_source)

### `gorilla-llm/gorilla-openfunctions-v2`
- scripts/data_pipeline.py:201 (pipeline_source)

### `mbpp`
- scripts/benchmarks_internal.py:151 (load_dataset_safe)

### `openai/gsm8k`
- scripts/data_pipeline.py:76 (pipeline_source)
- scripts/eval.py:58 (load_dataset)
- scripts/kaggle_onecell_t4_build30.py:2701 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2409 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3106 (pipeline_source)
- scripts/mertformer_5080_final_onefile.py:7654 (pipeline_source)
- eval/gsm8k.py:74 (load_dataset)
- eval/gsm8k.py:81 (load_dataset)

### `openai_humaneval`
- scripts/benchmarks_internal.py:150 (load_dataset_safe)

### `teknium/OpenHermes-2.5`
- scripts/data_pipeline.py:175 (pipeline_source)

### `turkish-nlp-suite/InstrucTurca`
- scripts/data_pipeline.py:166 (pipeline_source)
- scripts/kaggle_onecell_t4_build30.py:2733 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2441 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3138 (pipeline_source)

### `uonlp/CulturaX`
- scripts/data_pipeline.py:121 (pipeline_source)
- scripts/kaggle_onecell_t4_build30.py:2689 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2397 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3094 (pipeline_source)
- scripts/titan_preflight.py:896 (dataset_info)

### `wikimedia/wikipedia`
- scripts/data_pipeline.py:111 (pipeline_source)
- scripts/kaggle_onecell_t4_build30.py:2688 (pipeline_source)
- scripts/kaggle_onefile_demo_build30.py:2396 (pipeline_source)
- scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py:3093 (pipeline_source)
- scripts/mertformer_5080_final_onefile.py:7628 (pipeline_source)

### `wikitext`
- scripts/mini_titan_poc.py:388 (load_dataset)

### `{DATASET_ID}`
- scripts/preflight_run.py:1117 (load_dataset)

