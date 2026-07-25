# Dataset Kaynakları (Eğitim Öncesi Envanter)

Bu dosya, **kodda referanslanan dataset kaynaklarını** ve hedeflenen curriculum stage’lerini listeler.
Bu bir uyum/provenans kontrol listesidir; “eğitim garantisi” değildir.

Bkz:
- `datasets/inventory_TR.md` (koddan otomatik çıkarılan referanslar)
- `datasets/LICENSES_TR.md` (lisans tablosu)
- `datasets/hashes.json` (snapshot hash’leri; gerçek eğitim öncesi doldurulmalı)

## Eğitim Curriculum’u (`scripts/data_pipeline.py`)

### Stage 1 — Lojik / Kod + Matematik (hedef oran: %42)
- `bigcode/the-stack-dedup` (train; dil filtreli)
  Amaç: büyük ölçekli kod korpusu
  Dataset card: https://huggingface.co/datasets/bigcode/the-stack-dedup
  Durum: kodda referansli; revision/sha256 pinlendi (2026-07-25) — `datasets/hashes.json` içinde `status: "verified"`, `revision: "17cad72c886a2858e08d4c349a00d6466f54df63"`
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

### Stage 5 — Araç Kullanımı / Function Calling (hedef oran: %12)
- `glaiveai/glaive-function-calling-v2` (train)
  Amaç: function calling / tool use
  Dataset card: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
  Durum: kodda referansli; revision pin + manifest fingerprint `datasets/hashes.json` içinde kayıtlı
- `NousResearch/hermes-function-calling-v1` (train, config `func_calling`; opsiyonel)
  Amaç: tool-use çeşitliliği (`gorilla-llm/gorilla-openfunctions-v2` + `NousResearch/FC-1k`'nin yerine —
  ikisi de 2026-07-13/18 itibarıyla canlı HTTP 401 döndüğü doğrulanmış ölü kaynaklardı). Bu
  değiştirme 2026-07-19'da HF Hub API üzerinden canlı + ungated + apache-2.0-lisanslı olarak
  doğrulandı; bkz. `datasets/LICENSES_TR.md` ve `BACKLOG_TR.md`.
  Dataset card: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
  Durum: kodda referansli; revision pin + manifest fingerprint henüz yok (materyalize edilmedi)

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


## Demo / tek-dosya datasetleri (eğitim dışı)
- `codeparrot/github-code` (demo baseline; sadece onefile scriptlerinde)
  Amaç: Kaggle/Colab demo için legacy kod korpusu
  Dataset card: https://huggingface.co/datasets/codeparrot/github-code
  Durum: onefile demo referansı; **Build30 V2 core eğitimde kullanılmaz**

## Legacy / izleme amaçlı datasetler (Build30 V2 müfredatında yok)
- `mlabonne/guanaco-llama2-1k` (legacy trace; Build30 V2 pipeline'da yok)
  Amaç: önceki deneylerde kullanılan küçük instruction alt kümesi
  Dataset card: https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k
  Durum: hashes kayıtları traceability için tutulur; **Build30 V2 core eğitimde kullanılmaz**

