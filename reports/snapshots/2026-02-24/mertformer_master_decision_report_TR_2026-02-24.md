# MertFormer Titan — Kanıt-Temelli Teknik + Ticarileşme Karar Raporu

- Rapor tarihi (TR): **24 Şubat 2026**
- Kapsam: Yerel kod/dosya kanıtı + resmi web doğrulaması
- Dil: Ölçülebilir, abartısız, karar odaklı

## 1) Yönetici Özeti

Bu proje **ciddiye alınır**. Çünkü mimari düzeyde gerçek mühendislik derinliği var (BitLinear + Liquid + MoE + MLA + güvenlik kapıları) ve bunu destekleyen kod/doküman altyapısı mevcut.

Ama aynı anda şu da net: sistem **pre-training** aşamasında ve henüz claim-eligible benchmark kanıtı yok. Yani bugün için doğru etiket:

- **Teknik çekirdek güçlü**
- **Ürün performansı henüz doğrulama bekliyor**

## 2) Kanıt Tabanı (Dondurulmuş Snapshot)

Bu raporun baz aldığı canlı snapshot:

- Build durumu: `PRE-TRAINING (UNVERIFIED)`
- Runtime parametre (ölçülen): `3,698,246,156` (~3.70B)
- Closure: `57/57 green`, in-scope pending yok
- Benchmark claim eligibility: `false` (trained checkpoint eksik)
- Test: `108 passed, 4 skipped`

Detay dosyası:
- `reports/evidence_snapshot_2026-02-24.json`

## 3) Kod Gerçeklik Denetimi (Tamamlanan Kapsam)

İncelenen çekirdek dosyalar:

- `layers/*.py` (12 dosya): `bitlinear.py`, `liquid.py`, `moe.py`, `mla.py`, `mertformer_block.py`, `world_model_head.py`, vb.
- `model/transformers.py`
- `train/train.py`
- `orchestrator/*` (özellikle `failure_budget.py`, `governance.py`, `distillation_manager.py`)
- `reports/*` (kpi/closure/accuracy/benchmark dosyaları)

### Ana teknik bulgular

1. BitLinear ternary + STE hattı kodda gerçek.
2. Liquid/CfC zamansal hücre ve clamp guard kodda gerçek.
3. LiquidRouter history-window + causal conv kodda gerçek.
4. MLA hiyerarşik KV mantığı kodda gerçek.
5. Offline-first governance ve failure-budget guard gerçek.
6. Performans/benchmark iddialarının bir kısmı hedef/pending durumda.

Detaylı satır bazlı durumlar:
- `reports/claim_matrix_v2_2026-02-24.json`

## 4) Kritik Çelişki: 2.64B vs 3.70B

### Durum

- `2.64B` ifadesi repoda **tasarım hedefi** olarak geçiyor.
- Canlı model instantiate ölçümü **~3.70B** veriyor.

### Teknik yorum

Bu bir “proje çöp” göstergesi değil; ancak iletişimde çözülmezse güven zedeler.

### Doğru ifade standardı

- **Design target:** 2.64B
- **Current measured runtime:** ~3.70B

Bu ifade her pitch/DM/deck içinde birlikte geçmeli.

## 5) İnternet Doğrulaması (Resmi Kaynaklar)

Resmi kaynak seti:
- `reports/web_validation_sources_2026-02-24.md`

Doğrulanan başlıklar:

- HAVELSAN MAIN resmi ürün sayfası mevcut.
- HAVELSAN Yıldız açık inovasyon duyuruları mevcut.
- ASELSAN girişimcilik kanalı resmi sitede mevcut.
- TUSAŞ HangarPro resmi program sayfası mevcut.
- CMMC için resmi DoD kaynakları mevcut.
- Microsoft Founders Hub / AWS Activate kredi kanalları resmi kaynakta mevcut.

### LinkedIn 360 Brew iddiası

- Resmi LinkedIn dokümantasyonunda bu adla doğrulama bulunamadı.
- Etiket: **UNVERIFIED**

## 6) "2.64B Eğitim Başarısı" Ne Anlama Gelir? (Ölçülebilir)

2.64B eğitim denemesi başarılı biterse şu 5 şeyi kanıtlamış olursun:

1. **Convergence:** Eğitim/validasyon eğrileri kararlı düşüş gösteriyor.
2. **Router stabilitesi:** Uzman çökmesi yok (entropy/load metrikleri kabul aralığında).
3. **Reproducibility:** Aynı ayarla yeniden koşuda benzer sonuç alınabiliyor.
4. **Dış benchmark sinyali:** En az bir dış benchmarkta anlamlı skor üretiliyor.
5. **Device profile yolu:** Gerçek cihazda hız/bellek/termal profil çıkarılabiliyor.

### Önerilen minimum başarı eşiği (pragmatik)

- NaN/inf olmadan uzun koşu ve checkpoint restore başarısı
- Son pencere loss trendinde net iyileşme (düzleşme/bozulma değil)
- MoE yük dağılımında çökme sinyali olmaması
- En az 1 dış benchmarkta random/baseline üstü anlamlı marj
- Cihaz testinde p50/p95 TPS + RAM + sıcaklık ölçüm raporu

## 7) "Devrim" Eşiği (Zorunlu Minimumlar)

"Devrim" demek için şu 4 minimumun birlikte olması gerekir:

1. **Gerçek cihaz ölçümü** (yalnızca teorik değil)
2. **Bağımsız tekrar üretim** (başka ekip/ortamda)
3. **Müşteri probleminde PoC başarısı** (iş sonucu)
4. **Hukuki/lisans temizliği** (ticari risk kapalı)

Bu dörtlü yoksa doğru etiket: **yüksek potansiyel**.

## 8) Şirket İlgisi + Para Potansiyeli (3 Senaryo)

Detay dosyası:
- `reports/commercial_scenarios_v1_2026-02-24.json`

### Senaryo A — Başarısız eğitim
- Olasılık bandı: %30-%45
- Muhtemel anlaşma: danışmanlık/acqui-hire

### Senaryo B — Kısmi başarı
- Olasılık bandı: %40-%50
- Muhtemel anlaşma: ücretli PoC, pilot lisans

### Senaryo C — Güçlü başarı
- Olasılık bandı: %10-%20
- Muhtemel anlaşma: kurumsal lisans, stratejik ortaklık, büyük savunma pilotları

## 9) Nihai Skor (10 Üzerinden)

Skor kartı dosyası:
- `reports/readiness_scorecard_v1_2026-02-24.json`

Özet skorlar:

- Teknik: **8.7/10**
- Üretim Hazırlığı: **5.4/10**
- Pazar Uyum: **8.3/10**
- Ticari Hazırlık: **6.1/10**
- Risk (10 = düşük risk): **5.7/10**
- Ağırlıklı genel skor: **6.84/10**

## 10) Sorularına Net Cevaplar

### Bu proje ciddiye alınır mı?
Evet, teknik derinlik nedeniyle ciddiye alınır.

### 2.64B eğitim başarısı neyi kanıtlar?
Mimari teorisinin pratikte çalıştığını ve ticarileşme kapısının açıldığını kanıtlar.

### Devrim demek için ne eksik?
Gerçek cihaz ölçümü + bağımsız tekrar üretim + müşteri PoC sonucu + lisans temizliği.

### Büyük şirket işbirliği olasılığı nedir?
Var; en güçlü yol joint PoC ve measured-only iletişimdir.

### Para eder mi?
Evet, ama değerleme seviyesini eğitim sonucu ve müşteri doğrulaması belirler.

### 10 üzerinden nihai puan kaç?
Genel karar puanı: **6.84/10 (şu an)**.

## 11) Tek Kritik Şart (Tek Cümle)

**Tek kritik şart: claim-eligible, tekrar üretilebilir bir eğitim koşusundan çıkan ölçülebilir teknik kanıtı (özellikle benchmark + cihaz profili) üretmek.**

## 12) Bu Raporun Ürettiği Arayüzler

1. `claim_matrix`
- Dosya: `reports/claim_matrix_v2_2026-02-24.json`

2. `readiness_scorecard`
- Dosya: `reports/readiness_scorecard_v1_2026-02-24.json`

3. `commercial_scenarios`
- Dosya: `reports/commercial_scenarios_v1_2026-02-24.json`

4. Şema tanımı
- Dosya: `reports/report_interface_schema_v1.json`

## 13) Kaynaklar

### Yerel (kod/doküman)
- `model/transformers.py`
- `train/train.py`
- `layers/*.py`
- `orchestrator/failure_budget.py`
- `orchestrator/governance.py`
- `orchestrator/distillation_manager.py`
- `reports/kpi_report_v1.json`
- `reports/closure_57_matrix.md`
- `reports/report_accuracy_audit.md`
- `reports/benchmarks/*`

### Web (resmi)
- HAVELSAN MAIN: https://www.havelsan.com/en/sectors/simulation-autonomous-and-platform-management-technologies/main-the-enterprise-ai-platform
- HAVELSAN Yıldız: https://www.havelsan.com.tr/haberler/havelsan-yildiz-acik-inovasyon-programinin-galipleri-belli-oldu
- ASELSAN Girişimcilik: https://www.aselsan.com/tr/girisimcilik
- TUSAŞ HangarPro: https://www.tusas.com/hangarpro
- DoD CMMC About: https://dodcio.defense.gov/CMMC/About/
- DoD final rule release: https://www.defense.gov/News/Releases/Release/Article/3962067/dod-releases-final-rule-for-cybersecurity-maturity-model-certification-cmmc-p/
- CMMC resources: https://www.acq.osd.mil/cmmc/assessment-guides.html
- Microsoft Founders Hub: https://www.microsoft.com/en-us/startups/blog/founders-hub-benefits/
- Microsoft partner benefits: https://www.microsoft.com/en-us/startups/blog/trusted-partner-benefits/
- AWS credits: https://aws.amazon.com/startups/credits
- LinkedIn Help (feed/content): https://www.linkedin.com/help/linkedin/answer/a2001384, https://www.linkedin.com/help/linkedin/answer/a705554

