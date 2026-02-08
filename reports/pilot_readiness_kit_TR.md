# Pilot Hazırlık Kiti (B2B)

## Amaç
Bu kit, ücretli bir pilot başlamadan önce zorunlu minimum teknik kanıt paketini tanımlar.

## Kapsam
- Offline-first doğrulama ve operasyonel güvenlik kapıları
- Kurumsal pilot kabulü için teslim yapısı
- Eğitim öncesi aşama için risk ve limit bildirimi

## Zorunlu Teknik Kanıtlar
1. `bash scripts/verify_all.sh` sonucu: PASS
2. `TITAN_OFFLINE=1 bash run.sh --test` sonucu: PASS
3. Adım bazlı durumları içeren operator-mode gate özeti
4. `pilot_report_v1` JSON çıktısı (`mertformer pilot-report --out <path>`)

## Offline Prosedür (Müşteri Ortamı)
1. Ortamı `bash scripts/bootstrap_venv.sh` ile kur
2. `mertformer verify` komutunu çalıştır (yalnızca offline)
3. `mertformer pilot-report --out reports/pilot_report.json` komutunu çalıştır
4. Logları ve raporu pilot kabul paketine ekle

## Kabul Kriterleri (Teknik)
1. Secret scan, pytest, preflight ve operator gate adımlarının tamamı geçer
2. Doğrulama komutlarında ağ bağımlılığı gerekmez
3. Gate çıktıları ile pilot rapor alanları birbiriyle tutarlıdır
4. Takipli dokümanlarda makineye özel mutlak path bulunmaz

## Risk ve Limitler (Eğitim Öncesi)
1. Eğitimli checkpoint olmadan model kalite benchmark’ları iddia için uygun değildir
2. Cihaz gecikme/enerji iddiaları, ölçüm yapılana kadar hedef tahminidir
3. Düşük-bit kernel yolu deneyseldir ve opt-in çalışır
