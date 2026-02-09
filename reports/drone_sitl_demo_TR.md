# Drone SITL Demo Protokolü (Proof-of-System)

## Kapsam
Bu demo, kısıtlar altında otonom drone sınıfı davranış için yazılım-içinde-döngü (SITL) **kanıt akışıdır**. Gerçek uçuş iddiası değildir.

## Amaç
Tekrar üretilebilir loglarla şunları kanıtlamak:
1. Offline karar döngüsü
2. Güven eşiği ihlalinde deterministik fail-safe fallback
3. Geçici sensör arızası sonrasında toparlanma
4. MertFormer bileşenleri ile AI tabanlı aksiyon önerisi

## Çalıştırıcı
- Script: `scripts/drone_sitl_demo.py`
- Çıktı kökü: `reports/pilots/<pilot_id>/sitl_<timestamp>/`
- Politika motoru (varsayılan): `mertformer_liquidrouter`
  - Aksiyon önerisi için `BitLinear` + `LiquidRouter` kullanır.
  - Güvenlik için fail-safe override katmanı aktif kalır.

## Komut
```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --runs 3 --steps 120
```

Opsiyonel baseline karşılaştırması:

```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --policy-engine baseline
```

## Beklenen Çıktılar
1. `sitl_events.jsonl` (adım bazlı olay logu)
2. `sitl_summary.json` (koşu ve toplam özet)
3. `sitl_report.md` (insan-okur kanıt notu)

## Geçiş Kriterleri
1. Her koşuda en az bir fail-safe fallback tetiklenmeli
2. Enjekte edilen arıza penceresi sonrası toparlanma görülmeli
3. Toplam durum `all_green=true` olmalı
4. Koşu özetinde `policy_engine` alanı bulunmalı

## Notlar
- Bu, pilot kanıtı için deterministik bir simülasyon akışıdır.
- Gerçek UAV entegrasyonu bu kapanış döngüsünde kapsam dışıdır.
