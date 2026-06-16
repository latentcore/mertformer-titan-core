# MertFormer Titan Onyx Storm Mimarisinin Teknik Analizi ve Stratejik Değerlemesi

> **Dış inceleme notu:** Compute sponsorship değerlendirmesi için önce
> `reports/outreach_compute_sponsorship_messages.md` ve
> `reports/ocean_pre45k_h200_20260514_partial_evidence.md` okunmalıdır. Bu rapor
> teknik arka plan ve stratejik çerçevedir; benchmark, production, deployment,
> AGI veya model üstünlüğü iddiası değildir.

**Tarih:** 2026-03-13
**Versiyon:** v1.0 (Build 30 V2)
**Yazar:** MertFormer AI Team

---

## 1. Yönetici Özeti
Yapay zeka ekosistemi, bulut tabanlı devasa modellerden cihaz içi (on-device) çalışan, enerji verimliliği yüksek ve gizlilik odaklı Küçük Dil Modellerine (SLM) doğru evrilmektedir. Bu evrimin en uç noktasında yer alan **MertFormer Titan (Onyx Storm) v1.0 (Build 30 V2)** projesi, modern derin öğrenme literatüründeki en gelişmiş dört paradigmanın stratejik bir sentezidir:

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

MertFormer Titan projesinin temel taşı, standart transformatör bloklarının ötesine geçerek donanım farkındalıklı (hardware-aware) bir yapı sunmasıdır. Model **2.64 milyar parametreye** sahiptir ancak çalışma anındaki yükü (inference cost) klasik modellerden çok daha düşüktür. (Not: **ölçülen runtime parametre toplamı ~3.67B'dir (3.672.982.022)** ve olgusal iddialarda bu rakam esas alınır; 2.64B mimari tasarım hedefidir. README ve `ARCHITECTURE.md` bu ayrımı aynı şekilde belirtir.)

### 2.1 BitNet b1.58 ve Ternary Hesaplama Devrimi
Geleneksel modeller 16-bit (BF16) kullanırken, MertFormer Titan **BitNet b1.58** teknolojisini temel alarak ağırlıkları $\{-1, 0, 1\}$ değerlerine indirger.

*   **Bellek Tasarrufu (Tahmini):** %93.75 teorik azalma.
*   **VRAM İhtiyacı (Tahmini):** ~0.65 GB (2.64B parametre için; low-bit inference yolu gerektirir).
*   **Enerji Verimliliği (Hedef):** Ternary matematik optimize kernel ile çalışırsa NPU üzerinde 70 kat enerji tasarrufu hedefi.

$$w_q = \text{clamp}(\text{round}(\frac{w}{\gamma + \epsilon}), -1, 1)$$

*   **Residual Scaling Etkisi (Hedef):** 18 katman boyunca sinyal kararlılığı, 1/√2 (1/sqrt(2)) katsayısı ile korunur; gerçek doğrulama gerekir.

### 2.2 GQA Dikkat Bloğu (Grouped-Query, Mevcut Implementasyon)
`mla.py` modülü, `GQA` sınıfında bir GQA attention çekirdeği uygular. KV başlıkları (`num_kv_heads`) azaltılıp paylaştırılır ve çalışma anında query başlıklarına çoğaltılır. (Sınıf eskiden `MLA` adındaydı; implementasyona uyması için yeniden adlandırıldı — gerçek latent-MLA bottleneck uygulanmamıştır.)

*   **Mevcut mekanizma:** GQA projeksiyonu + KV head çoğaltma (latent down/up bottleneck değil).
*   **Cache verim yolu:** Opsiyonel kısa/uzun hiyerarşik KV cache modu decode anında bellek baskısını düşürebilir (hedef davranış; profile bağlı).
*   **RoPE:** $\theta = 100,000$ ile uzun bağlam desteği.
*   **Doğruluk sınırı:** Tam latent-MLA bottleneck bu sürümde roadmap kalemidir.

### 2.3 Liquid Neural Networks (CfC)
Projenin "canlı" kalbi. Biyolojik nöronlardan (C. elegans) esinlenen Kapalı Form Sürekli Zamanlı (CfC) hücreleri, girdiye bağımlı diferansiyel denklemlerle çalışır.

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
| Ara Boyut | 5632 (SwiGLU) |

**LiquidRouter'ın Stratejik Farkı (Claim-Safe):**
*   **Zamansal yönlendirme:** Verinin geliş hızını ve kısa geçmişini (`Fluid Path`) analiz eder; formal üstünlük iddiası içermez.
*   **Causal Conv1d Entegrasyonu:** Uzman seçimi sırasında geçmiş 4 token'lık pencereyi (`history_window`) dikkate alarak stratejik bir zeka sergiler.
*   **Donanım verimliliği (Hedef):** Yönlendirme kararsızlığını azaltmayı ve NPU davranışını iyileştirmeyi hedefler; cihaz profili gerekir.

### 2.5 Sinaptik Katman Hiyerarşisi (Layer-by-Layer Taxonomy)
MertFormer Titan'ın 18 katmanlı yapısı, veriyi kademeli bir "bilgeliğe" dönüştürür:
*   **L0-L2 (Foundation):** RMSNorm stabilizasyonu ve BitNet verimliliği ile temel gramer kurulumu.
*   **L3-L9 (Abstraction):** MoE uzman dağıtımı ve ilk **Liquid Teması (L4)** ile verinin soyut kavramlara ve niyet analizine evrilmesi.
*   **L10-L15 (Reasoning):** İkinci **Liquid Teması (L10)** ile güçlenen zamansal hafıza; stratejik karar ve kültürel adaptasyon süreçleri.
*   **L16-L17 (Wisdom & Final):** **Nihai Liquid Mührü (L16)** ile akışkan zekaya dönüşüm ve LM Head üzerinden logits üretimi.

---

## 3. Eğitim Stratejisi: Bilgi Damıtma (Distillation)

2.64B modelin, 70B zekasına yaklaşması için "Öğretmen-Öğrenci" yapısı kurulmuştur.

### 3.1 Çevrimdışı Damıtma (Offline Distillation)
`distillation_manager.py` ile Llama-3.3-70B modelinin çıktıları (logits) önceden diske kaydedilir.
*   **Hız:** Eğitim sırasında 12 kat hızlanma.
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

MertFormer Titan, genel bir yazılım değil, **"NPU-Native"** bir motordur. Aşağıdaki veriler, mimarinin operasyonel karmaşıklığı (OPs) ve bellek ayak izi üzerinden hesaplanmış **Mimari Simülasyon Hedefleridir.**

| Platform | Tahmini Hız (Hedef) | Bellek | Optimizasyon |
| :--- | :---: | :---: | :--- |
| **Samsung S25 (NPU)** | **45 - 107 t/s** | < 2.0 GB | Tam (JIT + BitNet) |
| iPhone 17 Pro | 40 - 80 t/s | < 2.5 GB | Yüksek (CoreML) |
| MacBook Pro (M4) | 110+ t/s | ~3.0 GB | Maksimum (Metal) |

> [!NOTE]
> Gerçek dünya performans verileri, eğitim tamamlandıktan sonra fiziksel cihaz testleriyle doğrulanacaktır.

---

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
*   **SHA256 Chaining:** Eğitimdeki her adım, bir önceki adımın özetiyle mühürlenir (`TITAN_POC_PROOF.jsonl`).
*   **Proof-of-Life:** Benchmark sonuçları, benchmark koşuları tamamlandıktan sonra kriptografik hash'ler ve proof-of-life artifact'larına bağlanacak şekilde tasarlanmıştır.
*   **Z-Loss ve Çökme Koruması:** `z_loss` ve `switch_loss` mekanizmaları ile modelin tek bir uzmana çökmesi (collapse) engellenir.

---

## 8. Sonuç

MertFormer Titan Onyx Storm, basit bir LLM olmanın ötesinde, cihaz içi yapay zeka ekosistemi için tasarlanmış **"yüksek performanslı bir çekirdek (kernel)"** mimarisidir.

**Vizyon:**
> *"Tohumu ektik, şimdi ormanı izleme vakti."*

Mimari tutarlı ve donanım hedefi nettir; pazar ilgisi ise kanıt-temelli outreach ile test edilmelidir. Projenin başarısı, operasyonel yürütme kalitesi ve checkpoint-bound kanıtla belirlenecektir.

---

## 9. Doğrulama Yol Haritası (claim-boundary)

Herhangi bir üretim ya da yetenek iddiasından ÖNCE gereken adımlar — aşağıdakilerin hiçbiri henüz tamamlanmadı:
1. **Whitepaper**: `LiquidRouter + GQA + BitNet` tasarımını ve ölçülen ablation sonuçlarını (sonuçsuz çıkan Liquid ablation'ı dahil) belgeleyen teknik makale.
2. **Açık Benchmark**: MMLU, GSM8K ve HumanEval skorlarının bağımsız denetçilerce doğrulanması.
3. **Canlı Demo**: Fiziksel bir Samsung S25 üzerinde 100% cihaz içi (on-device) kod üretimi videosu.

---

## 10. Gelecek Araştırma Ufukları (v28+)

Yapay ve biyolojik sinirsel verimlilik arasındaki farkı daha da kapatmak için, Titan mimarisinin sonraki iterasyonları şunları keşfedecektir:
*   **Kalıcı Bağlamsal Hafıza**: Ağırlık dengesizliği yaratmadan, modelin kullanıcıya özel kodlama stillerini ve proje geçmişini hatırlamasını sağlayan vektör tabanlı bir "Episodik Önbellek" (Episodic Cache) geliştirilmesi.
*   **Sinaptik Plastisite (Araştırma Yolu)**: Gerçek zamanlı davranışsal adaptasyon için izole edilmiş Liquid katmanları içinde "Hebbian" ilhamlı çıkarım-içi güncellemelerin keşfedilmesi.
*   **Homeostatik Regülasyon**: Derin katmanlarda sinyal kararlılığını ve otonom hassasiyet ayarını sağlamak için dinamik nöro-modülatör kapılama mekanizmaları.
*   **Duygusal Ağırlıklandırma (Nöromodülasyon)**: Belirsizlik durumlarında karar verme sürecini geliştirmek için nörotransmitter güdümlü öncelik değişimlerini (aciliyet, güven) simüle eden "Duygusal Kapılama" düzeneklerinin entegrasyonu.

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
