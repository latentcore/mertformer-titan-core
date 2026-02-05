# Dataset Lisanslari (Egitim Oncesi Kontrol Listesi)

Tum datasetler kendi lisanslari/terimleri ile uyumlu olmalidir (gated datasetler dahil).
Bu tablo, gercek egitim oncesi **tek kaynaktan kontrol listesi** olarak kullanilir.

Notlar:
- “TBD” olan satirlar, production egitim icin **blocker** kabul edilir (dataset card / upstream repo’dan dogrulanmadan egitime girilmez).
- Snapshot hash’leri `datasets/hashes.json` icinde tutulur ve egitim oncesi doldurulmalidir.

| Dataset | Lisans | Referans URL | Durum |
| --- | --- | --- | --- |
| `bigcode/the-stack-v2` | Other (karma upstream lisanslar; gated Terms of Use) | https://huggingface.co/datasets/bigcode/the-stack-v2 | Verified (HF gated kosullari) |
| `TIGER-Lab/MathInstruct` | MIT | https://opensource.org/licenses/MIT | Verified (HF metadata) |
| `openai/gsm8k` (`main`) | MIT | https://opensource.org/licenses/MIT | Verified (bilinen) |
| `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | https://opendatacommons.org/licenses/by/1-0/ | Verified (HF metadata) |
| `wikimedia/wikipedia` (`20231101.tr`) | CC BY-SA 4.0 + GFDL (cifte lisans) | https://foundation.wikimedia.org/wiki/Terms_of_Use | Verified (Wikipedia kosullari) |
| `uonlp/CulturaX` (`tr`) | ODC-By 1.0 + CC0-1.0 (mC4 + OSCAR turevi) | https://huggingface.co/datasets/uonlp/CulturaX | Verified (dataset card lisans bolumu) |
| `HuggingFaceTB/cosmopedia` (`stories`) | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `OpenAssistant/oasst_top1_2023-08-25` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `mlabonne/guanaco-llama2-1k` | Apache-2.0 (`timdettmers/openassistant-guanaco` turevi) | https://www.apache.org/licenses/LICENSE-2.0 | Verified (upstream dataset README) |
| `TFLai/Turkish-Alpaca` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `turkish-nlp-suite/InstrucTurca` | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (HF metadata) |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | https://www.apache.org/licenses/LICENSE-2.0 | Verified (HF metadata) |
| `openai_humaneval` | MIT | https://opensource.org/licenses/MIT | Verified (bilinen) |
| `mbpp` (`sanitized`) | CC-BY-4.0 | https://creativecommons.org/licenses/by/4.0/ | Verified (bilinen) |
| `wikitext` (`wikitext-2-raw-v1`) | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | Verified (bilinen) |
| Dahili stage setleri | Internal (proprietary) | `datasets/INTERNAL_POLICY.md` | Verified (internal policy) |
| Golden samples | Internal (proprietary) | `datasets/INTERNAL_POLICY.md` | Verified (internal policy) |
