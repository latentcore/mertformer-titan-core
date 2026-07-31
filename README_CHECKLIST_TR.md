# ✅ MertFormer Titan README Kalite Kontrol Listesi

> **⚠ SÜPERSEDED (2026-07-31):** Bu, 2026-02-05 / 2026-06-17 README pass'lerinden kalma,
> tek seferlik ve elle doldurulmuş bir anlık görüntü — repo'nun otomatik kapanış kapılarından
> (`scripts/check_doc_claim_consistency.py`, `scripts/md_quality_gate.py`,
> `scripts/md_integrity_check.py`, hepsi `verify_all.sh` tarafından çalıştırılıyor) önce
> yazılmış. Bu kapılar README doğruluğunun şu anki, canlı, kanıt-destekli kontrolü; bu dosya
> hiçbir kanonik yüzeyden linklenmiyor ve yalnızca 4KB-kesim öncesi manuel incelemenin tarihi
> bir kaydı olarak tutuluyor. O tarihten beri hiçbir şey için (Apache 2.0 relicensing veya
> Windows-portlama pass'i dahil) güncellenmedi — güncel kabul etmeyin.

> DÜRÜSTLÜK NOTU: Aşağıdaki `[x]` işaretleri, eğitim öncesi taslak aşamasında elle yapılan ÖZ-BEYANDIR; otomatik doğrulamadan geçmiş bir GEÇME-KAPISI (pass-gate) DEĞİLDİR. Üretim metrikleri/karşılaştırmalar henüz kanıtlanmadı. Tamlık (satır 8), Karşılaştırmalar (satır 24), Bağlantılar (satır 28) ve Yazım Denetimi (satır 30) gibi maddeler bir betik/rapor çıktısıyla DESTEKLENMEMEKTEDİR — bunları doğrulanmış kapılar değil, "kanıt beklemede" olarak değerlendirin. `[n/a]` maddeleri 4KB README kesiminde bilinçli olarak düşürülmüştür.

## 1️⃣ Proje Yapısı
- [x] **Klasör Listesi:** Config, layers, scripts, checkpoints, logs dosyalarının tamamı listelendi.
- [x] **Açıklamalar:** Dosya amaçları net bir şekilde tanımlandı (Örn: `bitlinear.py` → 1.58-bit Kuantizasyon).
- [x] **Tamlık:** Gerçek dosya içeriği ile birebir eşleşiyor.

## 2️⃣ Hızlı Başlangıç / Yayına Alım
- [x] **Python Örnekleri:** `TitanChat` kullanım örneği sunuldu.
- [x] **ONNX Aktarımı:** `scripts/mobile_export.py` dökümante edildi.
- [x] **Bağımlılıklar:** `requirements.txt` standartlara uygun.
- [x] **Çalıştırma Betiği:** `run.sh` kurulum ve test işlemlerini yönetiyor.
- [x] **Çıkarım (Inference):** Örnek kullanım kod blokları ile gösterildi.

## 3️⃣ Görseller / Diyagramlar
- [n/a] **Mermaid:** Mimari akış — 178KB→4KB README kesiminde kaldırıldı; bkz. `ARCHITECTURE.md` / `docs/PROJECT_STRUCTURE.md`.
- [n/a] **ASCII Sanatı:** "MertFormer Titan" başlığı — 4KB README kesiminde kaldırıldı.

## 4️⃣ Metrikler / SSS / Karşılaştırmalar
- [x] **Adli Veriler:** Örnek PoC hash'i eklendi; üretim metrikleri beklemede.
- [x] **Öngörüler:** Eğitim öncesi tahmin/ hedef olarak işaretlendi.
- [n/a] **Karşılaştırmalar:** Llama-3/Phi-3 tablosu — 4KB README kesiminde kaldırıldı (README_TR.md'de tablo yok).
- [x] **SSS (FAQ):** 1.58-bit kalitesi ve Mobil kapasite konuları ele alındı.

## 5️⃣ Bağlantılar / Rozetler / Yazım Denetimi
- [x] **Bağlantılar:** Dahili bağlantılar (anchor links) doğrulandı.
- [x] **Rozetler:** Lisans ve Durum rozetleri eklendi.
- [x] **Yazım Denetimi:** Türkçe ve İngilizce dillerindeki hatalar kontrol edildi.

## 6️⃣ Opsiyonel / Güçlendiriciler
- [x] **Türkiye Vizyonu:** `README_TR.md` içinde net ve okunabilir şekilde belirtildi.
- [x] **Örnek Girdi/Çıktı:** Sohbet örneği ile girdi/çıktı bağlamı sunuldu.
- [x] **Adli Mühür:** Güvenli günlük doğrulama bölümü eklendi.

---
**Durum:** 🟡 **EĞİTİM ÖNCESİ / TASLAK**
**Doğrulayan:** Antigravity Ajanı (ilk) · Build 30 V2 kapanış pass'inde tazelendi
**Tarih:** 2026-02-05 (ilk) · 2026-06-17 (4KB-README-kesimi sonrası tazeleme)
