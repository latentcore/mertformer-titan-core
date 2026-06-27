# Tool Sözleşmeleri

Tool çağrıları için giriş/çıkış sözleşmelerini tanımlar.

## Sözleşme Şablonu
- **name**: tool adı (string)
- **inputs**: JSON obje, schema tanımlı
- **outputs**: JSON obje, schema tanımlı
- **errors**: olası hata kodları listesi

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
