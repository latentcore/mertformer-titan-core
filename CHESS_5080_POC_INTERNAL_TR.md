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

## Claim Sınırı
- `1600+` garanti değil, hedeftir.
- Sadece ölçülen sonuçlar `verified` olarak yazılabilir.
- Stockfish karşılaştırması yoksa sonuç `target-not-verified` veya `not-run` olarak kalmalıdır.
- Bu hat tek başına ana 45K repo claim yüzeyini yükseltmez.

## Share Export
- Okunabilir kanonik script repo içinde kalır.
- `scripts/export_chess_5080_share.py` şunları üretir:
  - açık kopya
  - obfuscate edilmiş share kopyası
  - delivery zip
  - SHA256 dosyası
- Share wrapper sadece başarılı paketlenmiş koşudan sonra kendini silebilir.
- Repo içindeki kanonik kopya asla kendini silmez.
