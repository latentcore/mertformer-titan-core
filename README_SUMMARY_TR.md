![MertFormer Titan Header](assets/header.png)

Dil: [English](README_SUMMARY.md) | [Türkçe](README_SUMMARY_TR.md)

---

# MertFormer Titan - Dış Kullanıcı Özeti (Build 30)

## Bu Proje Nedir?
MertFormer Titan, kontrollü, denetlenebilir ve insan onaylı kullanım için tasarlanmış mobil-öncelikli, offline çalışabilen bir yapay zeka mimarisidir.  
BitNet (1.58-bit), Liquid dinamikleri ve MoE yönlendirmesini üretim-öncelikli bir mühendislik yaklaşımıyla birleştirir.

## Mevcut Durum
- **Aşama**: Pilota hazır eğitim öncesi baseline (`Build 30`)
- **Konumlandırma**: Gerçek dünya kısıtlarında proof-of-system
- **Henüz iddia edilmeyen**: Nihai benchmark üstünlüğü ve production performans iddiaları (eğitimli checkpoint kanıtı gerekir)

## Güvenlik ve Yönetişim Politikası
- Operasyonel kararlarda human-in-the-loop zorunludur.
- Orchestrator/runtime tarafında audit izi ve policy sınırları zorunludur.
- İzinsiz gözetim, gizli takip ve onaysız müdahale kapsam dışıdır.
- Pilot iddialarından önce güvenlik ve governance kapıları geçilmelidir.

## Doğrulanmış Yerel Kanıt (Son Koşu)
| Kapı | Sonuç |
| :--- | :--- |
| `python3 -m pytest -q` | `111 passed, 3 skipped` |
| `.titan-venv/bin/python -m ruff check .` | `All checks passed` |
| `bash scripts/verify_all.sh` | `[verify] OK` |

Closure artefaktları:
- `reports/closure_57_matrix.json`
- `reports/closure_57_matrix.md`
- `reports/closure_57_matrix_TR.md`

## Hızlı Başlangıç (Dış İnceleyici)
1. Sanal ortamı oluştur/güncelle:
```bash
bash scripts/bootstrap_venv.sh
```
2. Tam offline doğrulama kapısını çalıştır:
```bash
bash scripts/verify_all.sh
```
3. Eğitim hazırlık kapısını kontrol et (strict gate):
```bash
bash run.sh --train-ready
```
4. Compute + dataset önkoşulları sağlanınca eğitimi başlat:
```bash
TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash run.sh
```

## Dış Pilot Kullanım Modeli
- Müşteri ortamında tekrarlanabilir doğrulama kapılarını çalıştırın.
- Doğrulanamayan iddia yerine makine-okur log/rapor artefaktlarını paylaşın.
- Hassas teknik detaylar için NDA/private data-room çizgisini koruyun.
- Ölçülmüş sonuçlar ile projeksiyonları net biçimde ayırın.

## İddia Sınırı (Kritik)
- Eğitimli checkpoint ve tekrarlanabilir benchmark çıktıları üretilene kadar bu repo:
  - **Pilota hazır mühendislik baseline’ıdır**
  - **Nihai benchmark iddiası için uygun değildir (`NOT ELIGIBLE FOR CLAIM`)**

## Faydalı Dokümanlar
- Ana dokümanlar: [README.md](README.md), [README_TR.md](README_TR.md)
- Kullanım kılavuzu: [USAGE_GUIDE.md](USAGE_GUIDE.md), [USAGE_GUIDE_TR.md](USAGE_GUIDE_TR.md)
- SDK kılavuzu: [SDK_GUIDE.md](SDK_GUIDE.md), [SDK_GUIDE_TR.md](SDK_GUIDE_TR.md)
- Güvenlik/politika: [SECURITY.md](SECURITY.md), [USE_POLICY.md](USE_POLICY.md)
