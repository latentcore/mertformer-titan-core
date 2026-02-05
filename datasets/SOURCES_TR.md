# Dataset Kaynaklari (Egitim Oncesi Envanter)

Bu dosya, **kodda referanslanan dataset kaynaklarini** ve hedeflenen curriculum stage’lerini listeler.
Bu bir uyum/provenans kontrol listesidir; “egitim garantisi” degildir.

Bkz:
- `datasets/inventory_TR.md` (koddan otomatik cikarilan referanslar)
- `datasets/LICENSES_TR.md` (lisans tablosu)
- `datasets/hashes.json` (snapshot hash’leri; gercek egitim oncesi doldurulmeli)

## Egitim Curriculum’u (`scripts/data_pipeline.py`)

### Stage 1 — Lojik / Kod + Matematik (hedef oran: %45)
- `bigcode/the-stack-v2` (train; dil filtreli)
  Amac: buyuk olcekli kod korpusu
  Dataset card: https://huggingface.co/datasets/bigcode/the-stack-v2
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `TIGER-Lab/MathInstruct` (train)
  Amac: matematik talimat/veri
  Dataset card: https://huggingface.co/datasets/TIGER-Lab/MathInstruct
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `openai/gsm8k` (train; subset `main`)
  Amac: matematik akil yurutme (eval’de de kullaniliyor)
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

### Stage 2 — Dunya Bilgisi (hedef oran: %35)
- `HuggingFaceFW/fineweb-edu` (train)
  Amac: egitici web korpusu
  Dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

### Stage 3 — Kimlik ve Dil (hedef oran: %7)
- `wikimedia/wikipedia` (train; subset `20231101.tr`)
  Amac: temiz Turkce ansiklopedi metni
  Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `uonlp/CulturaX` (train; subset `tr`)
  Amac: buyuk olcekli Turkce web korpusu
  Dataset card: https://huggingface.co/datasets/uonlp/CulturaX
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `HuggingFaceTB/cosmopedia` (train; subset `stories`)
  Amac: sentetik yuksek kalite metin
  Dataset card: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

### Stage 4 — Ruh / Talimat (hedef oran: %3)
- `OpenAssistant/oasst_top1_2023-08-25` (train)
  Amac: yuksek kalite insan diyalogu / talimat
  Dataset card: https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `mlabonne/guanaco-llama2-1k` (train)
  Amac: talimat takibi ornekleri
  Dataset card: https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `TFLai/Turkish-Alpaca` (train)
  Amac: Turkce talimat takibi
  Dataset card: https://huggingface.co/datasets/TFLai/Turkish-Alpaca
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `turkish-nlp-suite/InstrucTurca` (train)
  Amac: Turkce talimat verisi
  Dataset card: https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

### Stage 5 — Arac Kullanimi / Function Calling (hedef oran: %10)
- `glaiveai/glaive-function-calling-v2` (train)
  Amac: function calling / tool use
  Dataset card: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

## Degerlendirme / Benchmark (`scripts/benchmarks_internal.py`, `eval/gsm8k.py`)
- `openai_humaneval` (test)
  Amac: kod uretim degerlendirmesi
  Dataset card: https://huggingface.co/datasets/openai/openai_humaneval
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `mbpp` (test; subset `sanitized`)
  Amac: kod uretim degerlendirmesi
  Dataset card: https://huggingface.co/datasets/mbpp
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli
- `openai/gsm8k` (test; subset `main`)
  Amac: matematik akil yurutme degerlendirmesi
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

## Dev / Hizli Debug (`scripts/mini_titan_poc.py`)
- `wikitext` (train; subset `wikitext-2-raw-v1`)
  Amac: kucuk hizli debug dataseti
  Dataset card: https://huggingface.co/datasets/wikitext
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` icinde kayitli

## Dahili / Ozel
- Stage curriculum ciktilari (stage1–stage5) (lokal jsonl snapshot’lar)
  Amac: indirilen/filtrelenen dahili curriculum ciktilari
  Durum: dahili (gitignored); uretim sonrasi lokal hash alin (bkz: `scripts/record_dataset_hashes.py`)
- Golden samples (dahili promptlar)
  Amac: regresyon kontrolleri
  Durum: dahili (trackli) + SHA256 `datasets/hashes.json` icinde kayitli
