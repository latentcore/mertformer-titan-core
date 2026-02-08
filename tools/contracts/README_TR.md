# Tool Sozlesmeleri

Tool cagirilari için giris/cikis sözleşmelerini tanimlar.

## Sozlesme Sablonu
- **name**: tool adi (string)
- **inputs**: JSON obje, schema tanimli
- **outputs**: JSON obje, schema tanimli
- **errors**: olasi hata kodlari listesi

## Örnek
```
name: "benchmark.run"
inputs:
  model_id: string
  samples: int
outputs:
  humaneval: int
  mbpp: int
errors:
  - DATASET_MISSING
  - MODEL_NOT_FOUND
```
