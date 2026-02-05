# Tool Sozlesmeleri

Tool cagirilari icin giris/cikis sozlesmelerini tanimlar.

## Sozlesme Sablonu
- **name**: tool adi (string)
- **inputs**: JSON obje, schema tanimli
- **outputs**: JSON obje, schema tanimli
- **errors**: olasi hata kodlari listesi

## Ornek
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
