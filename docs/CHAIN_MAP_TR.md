# Zincir Haritası — Bağlı vs Bağımsız

Bu harita, eğitim kanıt zincirinde **doğrudan bağlı** olanları ve **bağımsız** (ürünleşme/deployment) kanıt akışlarını özetler.

## Bağlı Eğitim Zinciri
```mermaid
flowchart TD
  A["Stage JSONL (datasets/stage*)"] --> B["Eğitim (run.sh → train/train.py)"]
  B --> C["Loglar (logs/*.jsonl)"]
  C --> D["SOP artefaktları (reports + packages/artifacts zip)"]
```

## Önkoşul Kapıları (Eğitim Öncesi)
```mermaid
flowchart LR
  G1["accelerate config"] --> B
  G2["cuda.lock"] --> B
  G3["HF_TOKEN + gated teacher"] --> B
  G4["dataset erişimi"] --> B
```

## Bağımsız Kanıt Akışları
```mermaid
flowchart LR
  X["NPU/Vulkan benchmark"]
  Y["Dockerfile etiketleri"]
  Z["README süre tahminleri"]
  X -.-> D
  Y -.-> D
  Z -.-> D
```

Notlar:
- Noktalı oklar eğitim **bloklayıcısı değildir**; deployment/dokümantasyon kalitesini etkiler.
- Bağlı zincir, **claim‑eligible** eğitim kanıtı üreten tek yoldur.
