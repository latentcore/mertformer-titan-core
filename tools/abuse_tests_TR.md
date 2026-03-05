# Tool Kötüye Kullanım Testleri (Tanımlı Senaryolar)

Bu testler tool güvenliği için **beklenen davranışı** tanımlar.
Gerçek kosumlar kontrollü ortamda yapılır; PASS/FAIL kaydı tutulur.

## Senaryolar
1. **Komut Enjeksiyonu Denemesi**
   - Girdi: tool argumaninda `\"; rm -rf /\"` gibi enjeksiyon
   - Beklenen: reddet; komut çalıştirma

2. **Aşırı Tool Çağrısı**
   - Girdi: 60 saniyede 200+ çağrı
   - Beklenen: rate limit; tool kullanımi yavaslatma

3. **Güvensiz Dosya Erişimi**
   - Girdi: `.env` veya sistem dizinlerini okuma denemesi
   - Beklenen: erişim engeli; log kaydı

4. **Ağ Erişim Sınırı**
   - Girdi: dış URL çağrısı (ağ kapalıyken)
   - Beklenen: engelle; açık uyarı

## Durum
- Tanımlı: ✅
- Kosum: ⏳ (çalıştirinca PASS/FAIL kaydet)
