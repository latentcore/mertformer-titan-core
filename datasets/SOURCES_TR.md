# Veri Seti Kaynakları (Egitim Oncesi Envanter)

Bu dosya, **kod tarafinda referanslanan** kaynaklari listeler. Kesin
snapshot, tarih ve hash degerleri **uretim egitimi oncesinde** doldurulmalidir.

## Ana corpuslar ( `scripts/data_pipeline.py` / preflight )
- `uonlp/CulturaX` (Turkce alt kume)  
  Amac: buyuk olcekli Turkce web corpus  
  Durum: kodda referansli, snapshot TBD
- `wikimedia/wikipedia` (Turkce)  
  Amac: temiz ansiklopedi metni  
  Durum: kodda referansli, snapshot TBD

## Degerlendirme / benchmarklar ( `scripts/benchmarks_internal.py`, `scripts/eval.py` )
- `openai/gsm8k`  
  Amac: matematiksel muhakeme  
  Durum: kodda referansli, snapshot TBD
- `openai_humaneval`  
  Amac: kod uretimi degerlendirme  
  Durum: kodda referansli, snapshot TBD
- `mbpp` (sanitized)  
  Amac: kod uretimi degerlendirme  
  Durum: kodda referansli, snapshot TBD

## Gelistirme / hizli deneme ( `scripts/mini_titan_poc.py` )
- `wikitext` (wikitext-2-raw-v1)  
  Amac: kucuk hizli debug veri seti  
  Durum: kodda referansli, snapshot TBD

## Dahili / ozel
- Asama mufredat setleri (stage1–stage5)  
  Amac: dahili mufredat  
  Durum: dahili, snapshot TBD
- Golden samples (dahili promptlar)  
  Amac: regresyon kontrolleri  
  Durum: dahili, snapshot TBD
