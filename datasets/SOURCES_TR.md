# Dataset Kaynakları (Eğitim Öncesi Envanter)

Bu dosya, **kodda referanslanan dataset kaynaklarını** ve hedeflenen curriculum stage’lerini listeler.
Bu bir uyum/provenans kontrol listesidir; “eğitim garantisi” değildir.

Bkz:
- `datasets/inventory_TR.md` (koddan otomatik çıkarılan referanslar)
- `datasets/LICENSES_TR.md` (lisans tablosu)
- `datasets/hashes.json` (snapshot hash’leri; gerçek eğitim öncesi doldurulmeli)

## Eğitim Curriculum’u (`scripts/data_pipeline.py`)

### Stage 1 — Lojik / Kod + Matematik (hedef oran: %42)
- `bigcode/the-stack-v2` (train; dil filtreli)
  Amaç: büyük ölçekli kod korpusu
  Dataset card: https://huggingface.co/datasets/bigcode/the-stack-v2
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `TIGER-Lab/MathInstruct` (train)
  Amaç: matematik talimat/veri
  Dataset card: https://huggingface.co/datasets/TIGER-Lab/MathInstruct
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `openai/gsm8k` (train; subset `main`)
  Amaç: matematik akıl yürütme (eval’de de kullanılıyor)
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı

### Stage 2 — Dünya Bilgisi (hedef oran: %30)
- `HuggingFaceFW/fineweb-edu` (train)
  Amaç: eğitici web korpusu
  Dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı

### Stage 3 — Kimlik ve Dil (hedef oran: %8)
- `wikimedia/wikipedia` (train; subset `20231101.tr`)
  Amaç: temiz Türkçe ansiklopedi metni
  Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `uonlp/CulturaX` (train; subset `tr`)
  Amaç: büyük ölçekli Türkçe web korpusu
  Dataset card: https://huggingface.co/datasets/uonlp/CulturaX
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `HuggingFaceTB/cosmopedia` (train; subset `stories`)
  Amaç: sentetik yüksek kalite metin
  Dataset card: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı

### Stage 4 — Ruh / Talimat (hedef oran: %8)
- `OpenAssistant/oasst_top1_2023-08-25` (train)
  Amaç: yüksek kalite insan diyaloğu / talimat
  Dataset card: https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `TFLai/Turkish-Alpaca` (train)
  Amaç: Türkçe talimat takibi
  Dataset card: https://huggingface.co/datasets/TFLai/Turkish-Alpaca
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `turkish-nlp-suite/InstrucTurca` (train)
  Amaç: Türkçe talimat verisi
  Dataset card: https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `teknium/OpenHermes-2.5` (train; opsiyonel)
  Amaç: genel instruction seti (opsiyonel fallback)
  Dataset card: https://huggingface.co/datasets/teknium/OpenHermes-2.5
  Durum: kodda referansli; lisans `datasets/LICENSES_TR.md` içinde doğrulanmalı

### Stage 5 — Araç Kullanımi / Function Calling (hedef oran: %12)
- `glaiveai/glaive-function-calling-v2` (train)
  Amaç: function calling / tool use
  Dataset card: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `gorilla-llm/gorilla-openfunctions-v2` (train; opsiyonel)
  Amaç: tool-use çeşitliliği (opsiyonel/gated)
  Dataset card: https://huggingface.co/datasets/gorilla-llm/gorilla-openfunctions-v2
  Durum: kodda referansli; lisans `datasets/LICENSES_TR.md` içinde doğrulanmalı
- `NousResearch/FC-1k` (train; opsiyonel)
  Amaç: hafif function-calling genişletmesi
  Dataset card: https://huggingface.co/datasets/NousResearch/FC-1k
  Durum: kodda referansli; lisans `datasets/LICENSES_TR.md` içinde doğrulanmalı

## Değerlendirme / Benchmark (`scripts/benchmarks_internal.py`, `eval/gsm8k.py`)
- `openai_humaneval` (test)
  Amaç: kod üretim değerlendirmesi
  Dataset card: https://huggingface.co/datasets/openai/openai_humaneval
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `mbpp` (test; subset `sanitized`)
  Amaç: kod üretim değerlendirmesi
  Dataset card: https://huggingface.co/datasets/mbpp
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `openai/gsm8k` (test; subset `main`)
  Amaç: matematik akıl yürütme değerlendirmesi
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı

## Dev / Hızlı Debug (`scripts/mini_titan_poc.py`)
- `wikitext` (train; subset `wikitext-2-raw-v1`)
  Amaç: küçük hızlı debug dataseti
  Dataset card: https://huggingface.co/datasets/wikitext
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı

## Dahili / Özel
- Stage curriculum çıktıları (stage1–stage5) (yerel jsonl snapshot’lar)
  Amaç: indirilen/filtrelenen dahili curriculum çıktıları
  Durum: dahili (gitignored); üretim sonrası yerel hash alın (bkz: `scripts/record_dataset_hashes.py`)
- Golden samples (dahili promptlar)
  Amaç: regresyon kontrolleri
  Durum: dahili (trackli) + SHA256 `datasets/hashes.json` içinde kayıtlı


## Demo / tek-dosya datasetleri (egitim disi)
- `codeparrot/github-code` (demo baseline; sadece onefile scriptlerinde)
  Amac: Kaggle/Colab demo icin legacy kod korpusu
  Dataset card: https://huggingface.co/datasets/codeparrot/github-code
  Durum: onefile demo referansi; **Build30 V2 core egitimde kullanilmaz**

## Legacy / izleme amacli datasetler (Build30 V2 mufredatinda yok)
- `mlabonne/guanaco-llama2-1k` (legacy trace; Build30 V2 pipeline'da yok)
  Amac: onceki deneylerde kullanilan kucuk instruction alt kumesi
  Dataset card: https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k
  Durum: hashes kayitlari traceability icin tutulur; **Build30 V2 core egitimde kullanilmaz**

