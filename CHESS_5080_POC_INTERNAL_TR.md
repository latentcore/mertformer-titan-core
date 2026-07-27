# Chess 5080 PoC Internal TR

Bu hat private/operator odaklı satranç proof akışıdır.

## Giriş Noktaları
- `bash run.sh --chess-5080-poc`
- `bash zero_touch_start.sh --chess-5080-poc`
- `python scripts/chess_5080_onefile.py`
- `python scripts/export_chess_5080_share.py`

## Amaç
- Filtrelenmiş Lichess verisi ile standalone satranç policy/value modeli eğitmek.
- Legal move masking'i zorunlu tutmak.
- Tek RTX 5080 masaüstünde yaklaşık 1-4 saat içinde bitirmek.
- Log, config, provenance, checkpoint ve SHA256 içeren zip proof paketi üretmek.
- Gömülü curated opening/tactical/endgame/blunder-correction suite ile training split'i güçlendirmek.
- Synthetic Turkish teaching corpus ve curated suite benchmark yüzeylerini aynı onefile çevresinde taşımak.
- Windows delivery hattında tek tık EXE üretmek ve runtime artefactlarini `runtime/` kökü altında toplamak.
- Stockfish anchor yüzeyi gerektiğinde runtime tarafında auto-fetch/cache ile ayağa kalkabilsin.

## Claim Sınırı
- `1600+` garanti değil, hedeftir.
- Sadece ölçülen sonuçlar `verified` olarak yazılabilir.
- Stockfish karşılaştırması yoksa sonuç `target-not-verified` veya `not-run` olarak kalmalıdır.
- Bu hat tek başına ana 45K repo claim yüzeyini yükseltmez.

## Share Export
> **⚠ DÜZELTME (2026-07-27):** bu notta işaretlenen EN/TR uyumsuzluğu düzeltildi — `CHESS_5080_POC_INTERNAL.md` artık aynı Windows build workspace + EXE delivery + Stockfish auto-fetch/cache davranışını taşıyor. Bu blockquote, bulgunun tarihsel kaydı olarak (silinmeden) korunuyor.
- Okunabilir kanonik script repo içinde kalır.
- `scripts/export_chess_5080_share.py` şunları üretir:
  - açık kopya
  - Windows build workspace
  - delivery zip
  - SHA256 dosyası
- Share wrapper sadece başarılı paketlenmiş koşudan sonra kendini silebilir.
- Repo içindeki kanonik kopya asla kendini silmez.
- Final kullanıcı artifacti kaynak `.py` değil, `external_delivery/mertformer_chess_5080.exe` olur.
- Final EXE `delivery_windows_oneclick` profilini ve runtime stockfish cache kökünü otomatik kullanır.

## Repo-Side Hazırlık Yüzeyi
- `reports/chess_training_readiness_report.md`: repo-side training readiness kararı
- `reports/chess_onefile_extension_report.md`: curated suite + synthetic teaching corpus extension doğrusu
- `reports/chess_teaching_contract_report.md`: teaching/Türkçe contract smoke raporu
- `reports/curated_position_manifest.md`: opening/tactical/endgame/blunder-correction bank özeti
- `reports/curated_position_suite_report.md`: eğitilmiş checkpoint varsa curated suite hit/tag raporu
- `reports/synthetic_teaching_corpus.md`: seviyeli Turkish teaching corpus özeti

> **Not (2026-07-27, canlı doğrulandı):** son üç dosya `scripts/build_chess_onefile_extension_report.py`/`scripts/chess_5080_onefile.py` tarafından üretilir ama bu ağaçta şu an mevcut değil (hiç üretilip commit edilmemiş) — repo'nun başka yerindeki "generated locally, version-controlled değil" örüntüsüyle aynı; varsayma, yerelde yeniden üret.
