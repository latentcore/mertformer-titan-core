# 🛡️ TITAN VERİ SETİ SAĞLIK RAPORU
**Tarih:** 2026-07-19 (tazelik pass'i — aşağıdaki yöntem notuna bakın)

> ⚠️ **YARI-ELLE, TEK-KOMUTLA OTOMATİK ÜRETİLMİYOR — düzenlemeden/körü körüne güvenmeden önce oku.** Bu dosya için özel bir üretici script yok (repo-geneli 2026-07-18'de kontrol edildi; yalnızca `datasets/inventory.json`/`.md` gerçek bir araçla (`scripts/extract_dataset_refs.py --fetch-metadata`) üretiliyor). Bu tablonun verisi gerçek ve araç-kaynaklıdır (aynı komut, hiçbir dataset içeriği indirilmez), ama tablonun kendisi o aracın JSON çıktısı + bu araçtan önce kontrol edilmiş 7 infra dataset'inin sabit bir listesinden elle bir araya getirildi. **Yenilemek için:** `python3 scripts/extract_dataset_refs.py --fetch-metadata`'yı yeniden çalıştır, sonra bu tablonun Durum/Lisans kolonlarını taze `datasets/inventory.json`'dan yeniden inşa et — o komutu önce çalıştırmadan bir durumu ✅/🔴'ya elle çevirme.

## 📊 Durum Özeti

> 🟡 **SİSTEM SARI:** 16 müfredat dataset'inin tamamı en az bir kez kontrol edildi (önceden 7/16'ydı). 2 gerçek erişim sorunu bulundu ve **çözüldü** (aşağıya bkz.) — kod düzeltmesi değil, insan kararı gerektiren türdendi.

## 📋 Detaylı Doğrulama Günlüğü

**Yenileme yöntemi (2026-07-18, dataset listesi 2026-07-19'da güncellendi):** `python3 scripts/extract_dataset_refs.py --fetch-metadata` (Hugging Face Hub API metadata çekimi — hiçbir dataset içeriği indirilmez, disk-güvenli; bu araç zaten `BACKLOG.md`'nin lisans-TBD maddesini çözmek için işaretlediği araç). `mlabonne/guanaco-llama2-1k` bu araçla erişilemiyor (`scripts/eval/train/orchestrator` içinde `load_dataset(...)` ile referanslanmıyor, yalnız `datasets/hashes.json`'da var) — bunun yerine HF dataset sayfasından elle kontrol edildi.

| Aşama | Veri Seti | Kırılım (Split) | Mantık | Durum | Lisans |
|-------|---------|-------|-------|--------|---------|
| Aşama 1 | `bigcode/the-stack-dedup` | train | HTTP Kontrolü | ✅ Çevrimiçi (revision unpinned — eğitim makinesinde network gerektiriyor, bkz. BACKLOG) | — |
| Aşama 1 | `TIGER-Lab/MathInstruct` | train | HTTP Kontrolü | ✅ Çevrimiçi | — |
| Aşama 1 | `openai/gsm8k` | train | HTTP Kontrolü | ✅ Çevrimiçi | — |
| Aşama 2 | `HuggingFaceFW/fineweb-edu` | train | HTTP Kontrolü | ✅ Çevrimiçi | — |
| Aşama 3 | `wikimedia/wikipedia` | train | HTTP Kontrolü | ✅ Çevrimiçi | — |
| Aşama 3 | `HuggingFaceTB/cosmopedia` | train | HTTP Kontrolü | ✅ Çevrimiçi | — |
| Aşama 4 | `OpenAssistant/oasst_top1_2023-08-25` | train | HTTP Kontrolü | ✅ Çevrimiçi | — |
| Aşama 3 | `uonlp/CulturaX` | train | HF metadata fetch (2026-07-18) | ✅ Çevrimiçi, **gated=true** (kabul edilmiş şartlarla `HF_TOKEN` gerekiyor) | unset |
| Aşama 4 | `TFLai/Turkish-Alpaca` | train | HF metadata fetch (2026-07-18) | ✅ Çevrimiçi | `apache-2.0` |
| Aşama 4 | `turkish-nlp-suite/InstrucTurca` | train | HF metadata fetch (2026-07-18) | ✅ Çevrimiçi | `cc-by-sa-4.0` |
| Aşama 4 | `teknium/OpenHermes-2.5` | train | HF metadata fetch (2026-07-18) | ✅ Çevrimiçi | ⚠️ unset (repo'da lisans etiketi yok) |
| Aşama 5 | `glaiveai/glaive-function-calling-v2` | train | HF metadata fetch (2026-07-18) | ✅ Çevrimiçi | `apache-2.0` |
| Aşama 5 | `NousResearch/hermes-function-calling-v1` | train (config `func_calling`) | HF Hub API fetch (2026-07-19) | ✅ Çevrimiçi, ungated. `gorilla-llm/gorilla-openfunctions-v2` + `NousResearch/FC-1k`'nin yerine — ikisi de canlı HTTP 401 doğrulandıktan sonra (2026-07-18) düşürüldü. | `apache-2.0` |
| Demo | `codeparrot/github-code` | train | HF metadata fetch (2026-07-18) | ✅ Çevrimiçi | `other` (kaynak GitHub repolarından miras alınan dosya-bazlı lisans — gerçekten karışık, tek bir SPDX etiketi değil) |
| Legacy | `mlabonne/guanaco-llama2-1k` | train | Elle HF sayfa kontrolü (2026-07-18) | ✅ Çevrimiçi | ⚠️ metadata özetinde görünür etiketli değil (tam dataset card'a elle bakılmalı) |

## Bu tazelemeden sonra hâlâ açık olanlar

- **ÇÖZÜLDÜ (2026-07-19):** `gorilla-llm/gorilla-openfunctions-v2` ve `NousResearch/FC-1k` (ikisi de doğrulanmış ölü, canlı HTTP 401, 2026-07-18) `scripts/data_pipeline.py`'nin `STAGE5_SOURCES`'ından düşürüldü ve `NousResearch/hermes-function-calling-v1` (config `func_calling`) ile değiştirildi — bu da landing'ten önce HF Hub API üzerinden canlı+ungated+apache-2.0 olarak doğrulandı. `scripts/titan_preflight.py`'nin `required_datasets` preflight-kontrol listesi de güncellendi (hâlâ ölü `gorilla-llm` ID'sini referanslıyordu, bu da bilerek düşürülmüş bir kaynak yüzünden preflight kapısının fail etmesine yol açardı). Oran: yeni kaynak, düşürülen iki kaynağın toplam 0.25+0.15=0.40 payını alıyor, yani Aşama 5'in oranları hâlâ 1.0'a toplanıyor ve genel 23.59B-token hedefi (bu listeden bağımsız, `TITAN_MAX_STEPS x batch x seq_len`'den hesaplanıyor) etkilenmiyor.
- **`teknium/OpenHermes-2.5`'in HF'de lisans etiketi yok** — gerçek, doğrulanmış bir boşluk (bir arama hatası değil), `BACKLOG.md`'nin zaten işaretlediğiyle eşleşiyor.
- **`codeparrot/github-code`'un `other` etiketi tek bir çözülebilir lisans değil** — orijinal GitHub repolarından miras alınan dosya-bazlı lisansları var; "tek bir lisans" gibi davranmak yanlış olur. Bu dataset gerçekten ölçekte kullanılırsa kendi uyum notunu ister, tek satırlık bir düzeltme değil.
- **`bigcode/the-stack-dedup` revision/sha256 hâlâ unpinned** — `datasets/hashes.json`'ın kendi notuna göre eğitim makinesinde network erişimi gerektiriyor; burada denenmedi (eğitim makinesi olmayan bir yerden atılacak bir pin işe yaramaz).

---
*2026-07-18'de `scripts/extract_dataset_refs.py --fetch-metadata` + bir elle kontrolle yenilendi; dataset listesi 2026-07-19'da güncellendi. Önceki tarama: 2026-01-25 (7/16 kontrol edilmişti). MertFormer Titan Doğrulama Paketi v1.0 tarafından oluşturulmuştur.*
