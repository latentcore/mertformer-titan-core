# Dataset Lisansları (Eğitim Öncesi Kontrol Listesi)

Tüm datasetler kendi lisansları/terimleri ile uyumlu olmalıdır (gated datasetler dahil).
Bu tablo, gerçek eğitim öncesi **tek kaynaktan kontrol listesi** olarak kullanılır.

Notlar:
- “TBD” olan satırlar, **çekirdek eğitim datasetleri** için production eğitimde **blocker** kabul edilir (dataset card / upstream repo’dan doğrulanmadan eğitime girilmez). Opsiyonel/demo datasetler doğrulanana kadar devre dışı tutulmalıdır.
- Snapshot hash’leri `datasets/hashes.json` içinde tutulur ve eğitim öncesi doldurulmalıdır.

| Dataset | Lisans | Referans URL | Durum |
| --- | --- | --- | --- |
| `bigcode/the-stack-dedup` | Other (karma upstream lisanslar; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-dedup | Verified (HF gated koşulları) |
| `TIGER-Lab/MathInstruct` | MIT | https://opensource.org/licenses/MIT | Verified (HF metadata) |
| `openai/gsm8k` (`main`) | MIT | https://opensource.org/licenses/MIT | Verified (bilinen) |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://opendatacommons.org/licenses/by/1-0/ | Verified (HF metadata) |
| `wikimedia/wikipedia` (`20231101.tr`) | CC BY-SA 4.0 + GFDL (çifte lisans) | https://foundation.wikimedia.org/wiki/Terms_of_Use | Verified (Wikipedia koşulları) |
| `uonlp/CulturaX` (`tr`) | ODC-By 1.0 + CC0-1.0 (mC4 + OSCAR türevi) | https://huggingface.co/datasets/uonlp/CulturaX | Verified (dataset card lisans bölümü) |
| `HuggingFaceTB/cosmopedia` (`stories`) | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (HF metadata) |
| `teknium/OpenHermes-2.5` | TBD | https://huggingface.co/datasets/teknium/OpenHermes-2.5 | Dataset card üzerinden doğrula |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `NousResearch/hermes-function-calling-v1` | Apache-2.0 | https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1 | Doğrulandı (HF Hub API, 2026-07-19); `gorilla-llm/gorilla-openfunctions-v2` + `NousResearch/FC-1k`'nin yerine (ikisi de ölü, canlı HTTP 401 doğrulandı) |
| `codeparrot/github-code` | TBD | https://huggingface.co/datasets/codeparrot/github-code | Demo-only (Build30 V2 core eğitimde değil); dataset card üzerinden doğrula |
| `mlabonne/guanaco-llama2-1k` | TBD | https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k | Legacy/optional; dataset card üzerinden doğrula |
| `openai_humaneval` | MIT | https://opensource.org/licenses/MIT | Verified (bilinen) |
| `mbpp` (`sanitized`) | CC-BY-4.0 | https://creativecommons.org/licenses/by/4.0/ | Verified (bilinen) |
| `wikitext` (`wikitext-2-raw-v1`) | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (bilinen) |
| Dahili stage setleri | Internal (proprietary) | `datasets/INTERNAL_POLICY.md` | Verified (internal policy) |
| Golden samples | Internal (proprietary) | `datasets/INTERNAL_POLICY.md` | Verified (internal policy) |
