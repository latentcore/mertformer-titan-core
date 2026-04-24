# MertFormer 5080 Artifact Intake Checklist

Created: 2026-04-22
Purpose: 5080 challenge çıktısı geldiğinde panik yapmadan, aynı sırayla doğrulama, smoke test, benchmark ve evidence paketleme yapmak.

## 1. Truth Boundary

Bu dosya sonuç iddiası değildir. Bu dosya yalnızca checkpoint/artifact geldiğinde uygulanacak kabul ve paketleme sırasıdır.

Şu an açık claim:

- 5080 delivery paketi hazır.
- Onefile lane içinde run, checkpoint, benchmark, package ve chat yolları mevcut.
- Final trained `latest.pt`, `best.pt` veya `model_final.pt` repo içinde yok.
- Devrim, kalite, latency veya chatbot iddiası ancak ölçülen artifact sonrası açılır.

## 2. Friend / Remote Machine'den İstenecek Paket

Sadece zip isteme. Mümkünse bütün output klasörü istenir.

İstenenler:

- `runs/`
- `checkpoints/`
- `latest.pt`
- `best.pt`
- `model_final.pt`
- run logları
- manifest dosyaları
- config dosyaları
- tokenizer dosyaları
- varsa evidence zip
- varsa final summary veya measured metrics JSON

Minimum kabul:

- En az bir yüklenebilir checkpoint.
- Checkpoint ile uyumlu config/tokenizer.
- Run logu veya manifest.

## 3. İlk Güvenlik Kontrolü

Artifact klasörü geldiğinde önce şu kontrol yapılır:

```bash
find <artifact_dir> -maxdepth 3 -type f | sort
shasum -a 256 <important_file>
du -sh <artifact_dir>
```

Amaç:

- Dosya gerçekten var mı görmek.
- Boyutlar mantıklı mı anlamak.
- SHA üretip evidence zincirini başlatmak.

## 4. Checkpoint Bütünlüğü

Kontrol edilecek dosya adayları:

- `latest.pt`
- `best.pt`
- `model_final.pt`
- `step_*.pt`

Kabul:

- Checkpoint yüklenebilmeli.
- Model config ile şekil uyuşmalı.
- Tokenizer yoksa chat/demo claim açılmaz.

Red flag:

- Dosya 0 byte.
- Checkpoint yüklenirken shape mismatch.
- Tokenizer/config eksik.
- Log yok ve eğitim adımı doğrulanamıyor.

## 5. Chat Smoke Test

Amaç kalite kanıtı değil, sistemin konuşma yolunun çalıştığını doğrulamaktır.

Test promptları:

Türkçe:

1. `Merhaba, kendini tek cümlede tanıt.`
2. `MertFormer nedir?`
3. `Kısa ve net cevap ver: edge AI neden önemli?`
4. `Bir günlük planı 3 maddeyle yaz.`
5. `Bu cümleyi İngilizceye çevir: Bugün küçük ama gerçek bir çıktı alacağım.`
6. `Basit Python fonksiyonu yaz: iki sayıyı topla.`
7. `Neden uyku performans için önemlidir?`
8. `Bir modelin hızlı olması ne demektir?`
9. `Kısa bir README özeti yaz.`
10. `Bu cevabı abartmadan ve dürüstçe bitir.`

İngilizce:

1. `Explain edge AI in one paragraph.`
2. `Write a tiny Python add function.`
3. `Summarize why low latency matters.`
4. `List three risks of overclaiming model quality.`
5. `Describe this project as provisional training evidence.`

Kayıt:

- Prompt.
- Raw output.
- Output length.
- Tekrar eden token var mı?
- Bariz çökme var mı?
- TPS/latency ölçümü varsa ekle.

## 6. Benchmark ve Ölçüm

Önce küçük smoke benchmark, sonra full benchmark.

Ölçülecekler:

- tokens/sec
- first-token latency
- total latency
- peak VRAM/RAM
- checkpoint load time
- benchmark status
- OOM var mı
- NaN var mı

Claim boundary:

- Benchmark yoksa benchmark var deme.
- OOM olduysa eğitim kanıtını silme; ama final benchmark claim açma.
- Tek cihaz sonucu global sonuç değildir.

## 7. Evidence Paketleme

Paket içine girmesi gerekenler:

- run summary
- manifest
- SHA256
- config
- tokenizer bilgisi
- benchmark output
- smoke test raw output
- known limitations
- truth boundary

Paket içine girmemesi gerekenler:

- Gereksiz dev zip tekrarları.
- Private token veya credential.
- Kanıtlanmamış büyük iddia.

## 8. measured_vs_target Tablosu

En az şu tablo hazırlanır:

| Metric | Target | Measured | Status | Note |
|---|---:|---:|---|---|
| Checkpoint load | yes | TBD | pending | Artifact bekleniyor |
| Chat smoke | yes | TBD | pending | 10 TR + 5 EN |
| TPS | TBD | TBD | pending | Cihazla ölçülecek |
| First-token latency | TBD | TBD | pending | Cihazla ölçülecek |
| Peak memory | TBD | TBD | pending | Cihazla ölçülecek |
| Benchmark suite | pass | TBD | pending | OOM ayrıca yazılır |
| Evidence zip | yes | TBD | pending | SHA256 zorunlu |

## 9. Public Claim Sırası

Sıra:

1. Private evidence.
2. Technical summary.
3. GitHub/HF clean presentation.
4. Short demo video.
5. Targeted outreach.

Doğru claim:

- `Provisional 5080 training/evidence package`
- `Measured smoke test and benchmark results`
- `Low-bit/edge-oriented experimental architecture`

Yanlış claim:

- `Production chatbot`
- `Gemma/OpenAI/Claude seviyesinde kanıtlandı`
- `Benchmark validated` benchmark yoksa
- `Mobile NPU production ready` export/profiling yoksa

## 10. Kapanış Kuralı

Artifact gelene kadar yapılacak iş yeni özellik değil, hazırlıktır.

Artifact geldikten sonra yapılacak iş hype değil, ölçümdür.

Ölçüm geldikten sonra yapılacak iş bağırmak değil, paketlemektir.
