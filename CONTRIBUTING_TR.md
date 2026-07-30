# Katkıda Bulunma

Bu depo **Apache License 2.0** ile lisanslanmıştır (bkz. [LICENSE](LICENSE), Türkçe
bilgilendirme: [LICENSE_TR](LICENSE_TR)) ve **dış katkılara açıktır**. Bir katkı
gönderdiğinizde, Apache 2.0'ın 5. maddesi uyarınca katkınızın aynı şartlarla
lisanslandığını kabul etmiş olursunuz.

Her pull request `bash scripts/verify_all.sh` komutunu sıfır regresyonla geçmelidir.
Tam liste için [README_TR.md](README_TR.md) içindeki "Katkılar ve PR Kuralları"
bölümüne bakın.

Geliştiren: Mert Yünlü. Uygulama için AI kod asistanları (Claude Code) kullanıldı; tüm mimari, tasarım kararları ve nihai inceleme yazarın kendisine aittir.

## Dahili İş Akışı
- Branch isimlerinde `feature/` ve `fix/` prefix'leri kullan.
- Tüm claim/metrikleri **pre-training / doğrulama bekliyor** statüsüyle hizalı tut.
- Büyük binary/dataset commit'leme; script ve yeniden üretilebilir config tercih et.
- Kullanıcıya dokunan bir davranış değiştiyse ilgili README/MD dosyasında kısaca belirt.

## Güvenlik
Güvenlik sorunlarını gizli bildir — bkz. [SECURITY.md](SECURITY.md). Public issue açma.
