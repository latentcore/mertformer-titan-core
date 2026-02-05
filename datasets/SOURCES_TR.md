# Dataset Kaynaklari (Egitim Oncesi Envanter)

Bu dosya, su ana kadar kodda referanslanan **bilinen kaynaklari** listeler.
Lisanslar `datasets/LICENSES.md` dosyasinda izlenir. Tam snapshot, tarih ve hash
bilgileri **uretim egitimi oncesi** kaydedilmelidir.

## Ana korpuslar ( `scripts/data_pipeline.py` / preflight )
- `uonlp/CulturaX` (Turkce altkume)
  Amac: buyuk olcekli Turkce web korpusu
  Dataset card: https://huggingface.co/datasets/uonlp/CulturaX
  Durum: kodda referansli; snapshot + hash bekliyor
- `wikimedia/wikipedia` (Turkce)
  Amac: temiz ansiklopedi metni
  Dataset card: https://huggingface.co/datasets/wikimedia/wikipedia
  Durum: kodda referansli; snapshot + hash bekliyor

## Degerlendirme / benchmark ( `scripts/benchmarks_internal.py`, `eval/gsm8k.py` )
- `openai/gsm8k`
  Amac: matematik akil yurutme degerlendirmesi
  Dataset card: https://huggingface.co/datasets/openai/gsm8k
  Durum: kodda referansli; snapshot + hash bekliyor
- `openai_humaneval`
  Amac: kod uretim degerlendirmesi
  Dataset card: https://huggingface.co/datasets/openai/openai_humaneval
  Durum: kodda referansli; snapshot + hash bekliyor
- `mbpp` (sanitized)
  Amac: kod uretim degerlendirmesi
  Dataset card: https://huggingface.co/datasets/mbpp
  Durum: kodda referansli; snapshot + hash bekliyor

## Dev sanity datasetleri ( `scripts/mini_titan_poc.py` )
- `wikitext` (wikitext-2-raw-v1)
  Amac: kucuk hizli debug dataseti
  Dataset card: https://huggingface.co/datasets/wikitext
  Durum: kodda referansli; snapshot + hash bekliyor

## Dahili / ozel
- Stage curriculum setleri (stage1–stage5)
  Amac: kureli dahili curriculum
  Durum: dahili; snapshot + hash bekliyor
- Golden samples (dahili promptlar)
  Amac: regresyon kontrolleri
  Durum: dahili; snapshot + hash bekliyor
