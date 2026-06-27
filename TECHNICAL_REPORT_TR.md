# MertFormer Titan Mimarisinin Teknik Analizi

> **Dış inceleme notu:** Compute sponsorship değerlendirmesi için önce
> `private/commercial/outreach_compute_sponsorship_messages.md` ve
> `reports/ocean_pre45k_h200_20260514_partial_evidence.md` okunmalıdır. Bu rapor
> teknik arka plan ve stratejik çerçevedir; benchmark, production, deployment,
> AGI veya model üstünlüğü iddiası değildir.

**Tarih:** 2026-06-18
**Versiyon:** v1.0 (Build 30 V2)
**Yazar:** Mert Yünlü

---

## 1. Yönetici Özeti
Yapay zeka ekosistemi, bulut tabanlı devasa modellerden cihaz içi (on-device), enerji verimli Küçük Dil Modellerine (SLM) doğru kayıyor. **MertFormer Titan v1.0 (Build 30 V2)** projesi, son derin öğrenme literatüründen dört paradigmayı birleştirir:

1.  **BitNet 1.58-bit Kuantizasyonu** (Verimlilik)
2.  **GQA dikkat bloğu (grouped-query, mevcut implementasyon)** (Bellek)
3.  **Seyrek Uzmanlar Karışımı (MoE)** (Kapasite)
4.  **Liquid Sınır Ağları (LNN)** (Dinamizm)

Bu mimari, Samsung Galaxy S25 ve Snapdragon 8 Elite gibi yeni nesil cihaz-içi donanım platformlarını hedefleyen bir mühendislik prototipidir.

---

## 1.1 V2 Refactor Özet
- Cross-dataset deduplication veri pipeline'ında aktif.
- MoE dispatch paralel gather/scatter modunu destekler.
- LiquidMixer fast path `liquid_fast_path` ile açılabilir.
- Eğitim kapıları varsayılan olarak fixed-step token bütçesine geçti.

## 2. Derin Teknik Mimari Analizi

MertFormer Titan projesinin temel taşı, standart transformatör bloklarının ötesine geçerek donanım farkındalıklı (hardware-aware) bir yapı sunmasıdır. Modelin **ölçülen runtime parametre toplamı ~3.67B'dir (3.672.982.022)** — tüm olgusal iddialarda kullanılan rakam budur — ancak çalışma anındaki yükü (inference cost) klasik modellerden çok daha düşüktür. (Not: 2.64B, mimarinin *tasarım hedefidir*, ölçülen toplam değil; `DECISIONS.md`'ye göre olgusal iddialarda 3.67B ölçülen rakam esas alınır. README ve `ARCHITECTURE.md` bu ayrımı aynı şekilde belirtir.)

### 2.1 BitNet b1.58 ve Ternary Hesaplama Devrimi
Geleneksel modeller 16-bit (BF16) kullanırken, MertFormer Titan **BitNet b1.58** teknolojisini temel alarak ağırlıkları $\{-1, 0, 1\}$ değerlerine indirger.

*   **Bellek Tasarrufu (Tahmini):** %93.75 teorik azalma.
*   **VRAM İhtiyacı (Tahmini):** ~0.65 GB (**2.64B tasarım hedefi** parametre sayısı için hesaplandı, 3.67B ölçülen toplam değil; low-bit inference yolu gerektirir).
*   **Enerji Verimliliği (Ölçülmemiş Hedef):** Ternary matematik optimize kernel'lerde çalışırsa NPU üzerinde 70 kata kadar enerji tasarrufu. Henüz böyle bir optimize kernel yok — mevcut Metal/Vulkan/NPU kod yolları generic bir `F.linear` (`torch`) passthrough'a düşüyor, dolayısıyla bu rakam bir ölçüm değil, yansıtılan bir hedeftir.

$$w_q = \text{clamp}(\text{round}(\frac{w}{\gamma + \epsilon}), -1, 1)$$

*   **Residual Scaling Etkisi (Hedef):** 18 katman boyunca sinyal kararlılığı, 1/√2 (1/sqrt(2)) katsayısı ile korunur; gerçek doğrulama gerekir.

### 2.2 GQA Dikkat Bloğu (Grouped-Query, Mevcut Implementasyon)
`mla.py` modülü, `GQA` sınıfında bir GQA attention çekirdeği uygular. KV başlıkları (`num_kv_heads`) azaltılıp paylaştırılır ve çalışma anında query başlıklarına çoğaltılır. (Sınıf eskiden `MLA` adındaydı; implementasyona uyması için yeniden adlandırıldı — gerçek latent-MLA bottleneck uygulanmamıştır.)

*   **Mevcut mekanizma:** GQA projeksiyonu + KV head çoğaltma (latent down/up bottleneck değil).
*   **Cache verim yolu:** Opsiyonel kısa/uzun hiyerarşik KV cache modu decode anında bellek baskısını düşürebilir (hedef davranış; profile bağlı).
*   **RoPE:** $\theta = 100,000$ ile uzun bağlam desteği.
*   **Doğruluk sınırı:** Tam latent-MLA bottleneck bu sürümde roadmap kalemidir.

### 2.3 Liquid Neural Networks (CfC)
Birkaç katmanda sürekli-zamanlı yinelemeli bir mikser. Biyolojik nöronlardan (C. elegans) esinlenen Kapalı Form Sürekli Zamanlı (CfC) hücreleri, girdiye bağımlı diferansiyel denklemlerle çalışır. (Değer sınırı: 12-seed toy ablation ölçülen doğruluk faydası göstermedi, ~%30 daha yavaş; bkz. [ABLATION.md](ABLATION.md).)

*   **Zaman Algısı:** `tau` (zaman sabiti) parametresi dinamiktir.
*   **Süreklilik:** Tokenlar arası momentumu takip eder.
*   **Uygulama:** JIT derlenmiş döngülerle NPU hızında çalışır (`liquid.py`).

Formül:
$$h(t) = A + (h_{prev} - A) \odot \exp(-\tau \Delta t)$$

### 2.4 LiquidRouter & MoE
`LiquidRouter`, MoE token yönlendirmesi için zamansal Conv tabanlı bir yönlendirici olarak uygulanır (`causal depthwise Conv1d` + rolling state buffer).
Yönlendirme politikası token-choice top-k'dır ve `LiquidMixer/LiquidCell` içindeki CfC yolundan ayrı değerlendirilmelidir.

| Parametre | Değer |
| :--- | :--- |
| Uzman Sayısı | 8 |
| Aktif Uzman (Top-k) | 2 |
| Yönlendirici | `LiquidRouter` (Conv1d + state buffer) |
| MoE uzman ara boyutu (`moe_intermediate`) | 8192 (SwiGLU) |
| Dense-FFN ara boyutu (`intermediate_size`) | 5632 (SwiGLU) |

**LiquidRouter'ın Stratejik Farkı (Claim-Safe):**
*   **Zamansal yönlendirme:** Verinin geliş hızını ve kısa geçmişini (`Fluid Path`) analiz eder; formal üstünlük iddiası içermez.
*   **Causal Conv1d Entegrasyonu:** Uzman seçimi sırasında geçmiş 4 token'lık pencereyi (`history_window`) dikkate alarak stratejik bir zeka sergiler.
*   **Donanım verimliliği (Hedef):** Yönlendirme kararsızlığını azaltmayı ve NPU davranışını iyileştirmeyi hedefler; cihaz profili gerekir.

### 2.5 Katman Taksonomisi (Katman-Katman)
MertFormer Titan'ın 18 katmanlı yığını, role göre:
*   **L0-L2 (Temel):** RMSNorm stabilizasyonu ve BitNet ternary linear'larla taban temsil.
*   **L3-L9 (Orta):** MoE uzman dağıtımı; **L4**'te ilk Liquid/CfC mikser.
*   **L10-L15 (Derin):** **L10**'da ikinci Liquid/CfC mikser; daha derin özellik kompozisyonu.
*   **L16-L17 (Çıkış):** **L16**'da üçüncü Liquid/CfC mikser; LM head üzerinden logits.

---

## 3. Eğitim Stratejisi: Bilgi Damıtma (Distillation)

2.64B modelin, 70B zekasına yaklaşması için "Öğretmen-Öğrenci" yapısı kurulmuştur.

### 3.1 Çevrimdışı Damıtma (Offline Distillation)
`distillation_manager.py` ile Llama-3.3-70B modelinin çıktıları (logits) önceden diske kaydedilir.
*   **Hız (Hedef / tahmin):** online 70B teacher'a kıyasla ~12 kat eğitim hızlanması (offline precompute, adım-başına teacher forward'unu kaldırır); 45K'da henüz ölçülmedi.
*   **Bellek:** Training sırasında teacher modelin VRAM'e yüklenmesine gerek kalmaz.

### 3.2 5 Aşamalı Müfredat (Curriculum)
1.  **Saf Mantık & Kod (%45):** Yapısal düşünme.
2.  **Dünya Bilgisi:** FineWeb-Edu.
3.  **Kimlik & Dil (TR):** Türkçe kültürel adaptasyon.
4.  **Ruh (Soul):** Karakter ve talimat takibi.
5.  **Araç Kullanımı:** API ve fonksiyon çağırma.

### 3.3 Build30 Profil Sözleşmesi (Stable vs Max-Arch)
Build30 kapanışında açık bir çalışma profili sözleşmesi uygulanır:

| Profil | Sözleşme | Aktivasyon |
| :--- | :--- | :--- |
| `stable` (varsayılan) | Tekrarlanabilir eğitim başlangıcı için regresyon-güvenli baseline | `bash run.sh` |
| `max_arch` | İleri mimari bayraklarını overlay ile açar (`mertformer_max_arch.yaml`) | `TITAN_PROFILE=max_arch bash run.sh` |

Her iki profilde readiness-only doğrulaması aynıdır:
```bash
bash run.sh --train-ready
```

QINN, ana eğitim hattında throughput ve yakınsama stabilitesini korumak için varsayılan olarak kapalıdır (`use_qinn=false`).

### 3.4 Kanıt-Öncelikli İddia Politikası
- Doğrulanmış maddeler, gate çıktılarından (pytest/verify/preflight/policy kontrolleri) raporlanır.
- Performans projeksiyonları, eğitimli checkpoint benchmark'ı üretilene kadar açık şekilde simülasyon hedefi olarak etiketlenir.
- Bu kapanış turunda dataset kapsamı sabittir (manifest korunur; major genişleme yapılmaz).

---

## 4. Donanım Hedefi: Samsung S25 & Snapdragon 8 Elite

MertFormer Titan, genel bir yazılım değil, **"NPU-Native"** bir motordur. Not: optimize NPU/Metal/Vulkan kernel'leri henüz uygulanmadı — bu backend'ler şu anda özel low-bit shader'lar yerine generic bir `torch` (`F.linear`) fallback çalıştırıyor. Bu nedenle aşağıdaki veriler, ölçülmüş throughput değil, mimarinin operasyonel karmaşıklığı (OPs) ve bellek ayak izi üzerinden hesaplanmış **Mimari Simülasyon Hedefleridir.**

| Platform | Tahmini Hız (Hedef) | Bellek | Optimizasyon |
| :--- | :---: | :---: | :--- |
| **Samsung S25 (NPU)** | **45 - 107 t/s** | < 2.0 GB | Tam (JIT + BitNet) |
| iPhone 17 Pro | 40 - 80 t/s | < 2.5 GB | Yüksek (CoreML) |
| MacBook Pro (M4) | 110+ t/s | ~3.0 GB | Maksimum (Metal, hedef — henüz optimize Metal kernel yok) |

> [!NOTE]
> Gerçek dünya performans verileri, eğitim tamamlandıktan sonra fiziksel cihaz testleriyle doğrulanacaktır.

---

## 5. Stratejik Bağlam (claim-boundary)

Ticari değerleme, fonlama ve kariyer çerçevesi bu teknik rapordan **çıkarılmıştır** ve özel dealroom
(`mertformer-titan-dealroom-private`) içinde tutulur. Burada kapsam:

*   **Yön:** cihaz-içi, düşük-bit (BitNet b1.58) + sparse-MoE + GQA mimarisi; NPU sınıfı donanımı
    (örn. Samsung S25) hedefler. Cihaz kullanımı, gecikme, enerji ve maliyet rakamları, fiziksel
    hedef-cihaz ölçümü + checkpoint-bound koşu oluşana kadar **target/estimate**'tir.
*   **Pazar bağlamı (nötr):** sektör cihaz-içi NPU ekliyor; bu mimari o donanım sınıfını hedefler.
    Burada pazar-büyüklüğü, değerleme ya da "moat" iddiası yapılmaz.
*   **Dürüst duruş:** benchmark, production, deployment ya da model-üstünlüğü iddiası yok.
    Değerlendirilecek değer, bitmiş bir frontier model değil; mühendislik disiplinidir (düşük-bit
    runtime, eğitim güvenilirliği, kanıt/iddia disiplini).

---

## 7. Adli Doğrulama ve Güvenlik (Forensic Verification)

Modelin güvenilirliği, açık doğrulama ve loglama mekanizmalarıyla desteklenmektedir:
*   **SHA256 Chaining (tasarım):** eğitim, her adımı bir önceki adımın özetiyle mühürleyecek şekilde *tasarlanmıştır* (`TITAN_POC_PROOF.jsonl`); zincir gerçek bir koşuyla üretilir — henüz tamamlanmış bir 45K zinciri yok.
*   **Proof-of-Life:** Benchmark sonuçları, benchmark koşuları tamamlandıktan sonra kriptografik hash'ler ve proof-of-life artifact'larına bağlanacak şekilde tasarlanmıştır.
*   **Z-Loss ve Çökme Koruması:** `z_loss` ve `switch_loss` mekanizmaları ile modelin tek bir uzmana çökmesi (collapse) engellenir.

---

## 8. Sonuç

MertFormer Titan, disiplinli bir kanıt sınırıyla kurulmuş cihaz-içi odaklı bir mimaridir (BitNet + MoE + Liquid/CfC + GQA). Mimari kendi içinde tutarlı ve donanım hedefi somut; ama bileşen değeri ve model kalitesi hâlâ hipotez: henüz eğitilmiş checkpoint yok. Değerlendirilecek şey mühendislik disiplini (low-bit runtime, eğitim güvenilirliği, claim disiplini) — bitmiş bir model değil. Başarı artık operasyonel yürütme ve checkpoint-bound kanıtla belirlenecek.

---

## 9. Doğrulama Yol Haritası (claim-boundary)

Herhangi bir üretim ya da yetenek iddiasından ÖNCE gereken adımlar — aşağıdakilerin hiçbiri henüz tamamlanmadı:
1. **Whitepaper**: `LiquidRouter + GQA + BitNet` tasarımını ve ölçülen ablation sonuçlarını (sonuçsuz çıkan Liquid ablation'ı dahil) belgeleyen teknik makale.
2. **Açık Benchmark**: MMLU, GSM8K ve HumanEval skorlarının bağımsız denetçilerce doğrulanması.
3. **Canlı Demo**: Fiziksel bir Samsung S25 üzerinde 100% cihaz içi (on-device) kod üretimi videosu.

---

## 10. Spekülatif Araştırma Ufukları (kapsam-dışı; uygulanmadı)

Aşağıdakiler **yalnızca uzun-vadeli araştırma yönleridir**. Hiçbiri kanonik eğitim yolunda uygulanmadı, hiçbiri 45K koşusunun ya da eğitilen modelin parçası değil ve hiçbiri yetenek olarak iddia edilmiyor — şeffaflık için listeleniyor, özellik olarak değil:
*   **Kalıcı bağlamsal hafıza** — oturumlar arası kullanıcı/proje bağlamı için vektör tabanlı episodik önbellek (araştırma fikri; inşa edilmedi).
*   **Çıkarım-içi plastisite** — izole Liquid katmanlarında gerçek-zamanlı adaptasyon için Hebbian-tarzı güncellemeler (araştırma fikri; inşa edilmedi).
*   **Uyarlanır kazanç regülasyonu** — sinyal kararlılığı için dinamik katman-bazlı kapılama (araştırma fikri; inşa edilmedi).

---

## 11. Yasal Güvenlik Sınırları

- Dağıtım, policy-bound ve denetlenebilir olmalıdır.
- Operasyonel aksiyonlarda insan onayı zorunludur.
- İzinsiz gözetim/takip ve yetkisiz müdahale açıkça kapsam dışıdır.
- Build kapanışı `Code+Test Green` kriteri ile yapılır; ağır eğitim kanıtları `Evidence Pending` olarak ayrı raporlanır.

## 12. Closure-57 Kapısı

```bash
python3 scripts/check_57_matrix.py
```

Üretilen çıktılar:
- `reports/closure_57_matrix.json`
- `reports/closure_57_matrix.md`
- `reports/closure_57_matrix_TR.md`
