# Teknik Makale: LiquidRouter Mimarisi
**Seyrek Uzmanlar Karışımı (MoE) İçin Zamansal-Duyarlı Yönlendirme**

## 1. Özet
MertFormer Titan mimarisi, Seyrek Uzmanlar Karışımı (MoE) için Kapalı Form Sürekli-Zamanlı (CfC) sinir ağlarını kullanan yeni bir kapılama mekanizması olan **LiquidRouter**'ı tanıtır. Token'ları birbirinden bağımsız olaylar olarak gören geleneksel "statik" yönlendiricilerin aksine LiquidRouter, zamansal momentumu koruyarak cihaz içi yapay zeka uygulamalarında uzman kullanımını ve yönlendirme kararlılığını önemli ölçüde artırır.

## 2. Problem: Statik MoE Çökmesi
Geleneksel MoE yönlendiricileri, uzman seçmek için basit bir doğrusal projeksiyon ve ardından Softmax kullanır. Bu yaklaşım şu sorunlara yol açar:
- **Uzman Çökmesi (Expert Collapse)**: Geçmiş bağlam eksikliği nedeniyle sadece birkaç uzmanın aşırı kullanılması.
- **Çıkarım Titremesi (Inference Jitter)**: Token'lar arasında uzmanların hızla değişmesi, NPU donanımında önbellek hatalarına ve gecikme artışına neden olur.

## 3. Çözüm: CfC Tabanlı Akışkan Yönlendirme
LiquidRouter, standart kapılama ağını bir **Akışkan Sinir Ağı (LNN)** hücresiyle değiştirir. Yönlendirme kararını sürekli bir diferansiyel denklem olarak modelleyerek sistem şunları kazanır:
- **Zamansal Bağlam**: $x_t$ token'ı için uzman seçimi, $x_{t-1 \dots t-n}$ token'larının gizli durumundan ve momentumundan etkilenir.
- **Yumuşak Geçişler**: CfC'nin "akışkan" doğası, yönlendirme kararlarının mantıklı bir şekilde evrilmesini sağlar ve donanım düzeyinde bağlam değiştirmeyi (context switching) azaltır.

### Matematiksel Temel
Yönlendirme ağırlığı $G(t)$ şu şekilde hesaplanır:
$$G(t) = \sigma(CfC(x_t, h_{t-1}))$$
Burada $CfC$, sinirsel ODE'nin Kapalı Form çözümünü temsil eder ve verinin "akışını" izleyen verimli, donanıma duyarlı hesaplamaya olanak tanır.

## 4. Donanım Sinerjisi
NPU'lar (Snapdragon 8 Elite gibi) üzerinde LiquidRouter enerji tüketimini şu şekilde optimize eder:
1. **Öngörülü Aktivasyon**: Token tam olarak ulaşmadan önce olası uzman yollarını önceden hesaplar.
2. **Azaltılmış Geçiş**: Yeni uzman ağırlıklarının NPU'nun yerel belleğine yüklenmesinin yüksek enerji maliyetini minimize eder.

## 5. Sonuç
LiquidRouter, MertFormer Titan için dilin zamansal doğasına saygı duyan, ampirik olarak kararlı ve cihaz üzerinde çalışan ilk MoE mimarisini sağlayan stratejik bir hendek (moat) oluşturur.
