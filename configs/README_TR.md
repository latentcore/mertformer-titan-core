# Configs

Stabil, isimlendirilmiş yapılandırma sözleşmeleri için kanonik config yüzeyi.

Mevcut kapsam:
- satranç onefile profil sözleşmesi
- profil niyeti; üretilmiş koşu çıktıları değil

Üretilmiş runtime-resolved config'ler, koşuya ait `reports/resolved_config.json` altında kalır.

Sınır:
- Accelerate launch profilleri ve hedef makine yeniden üretilebilirlik dosyaları `repro/` altında durur.
- Örnek: `repro/accelerate_8xgpu.yaml` burada tutulmaz; çünkü model/config sözleşmesini değil, koşu ortamını kontrol eder.
