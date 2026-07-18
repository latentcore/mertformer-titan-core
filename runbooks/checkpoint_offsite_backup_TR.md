# Off-site checkpoint yedekleme — runbook

**Bu neden var:** 2026-05-14 tarihli 2xH200 kısmi koşusu step 1880'e kadar eğitim gördü ve
checkpoint'i **kalıcı olarak kayboldu** — hiçbir off-site kopya yoktu, kiralanan makinenin
depolaması geri alınmadan önce kurtarılamadı. Bu runbook'un varlık sebebi tam olarak bu somut
başarısızlığı önlemek. Hiçbir launch script'i bunu otomatik yapmıyor (2026-07-19'da kontrol
edildi: `scripts/launch_*.sh` içinde hiçbir yerde `rsync`/`rclone`/`aws s3 sync`/`gsutil`
çağrısı yok) — bu, kiralanan makinede eğitim koşusuyla birlikte, ikinci bir terminalden
çalıştırılan elle bir operatör adımı.

## Ne zaman çalıştırılır

`TITAN_SAVE_INTERVAL` (varsayılan `1000` adım, bkz. `scripts/launch_ocean_45k.sh`) `cfg.save_dir`
altına (varsayılan `./checkpoints/mertformer_titan_prod/`) her yeni checkpoint yazdığında.
Koşu bitene kadar bekleme — yalnızca kiralanan makinede var olan bir checkpoint, 2026-05-14
koşusuyla aynı akıbete uğramaktan bir preemption uzaktadır.

## Minimum uygulanabilir prosedür (herhangi bir lane için)

1. **Koşu başlamadan önce**, off-site hedefi belirle ve kiralanan makineden oraya yazabildiğini
   teyit et (önce küçük bir dosyayla test et — ilk gerçek checkpoint geldikten sonra bozuk bir
   kimlik bilgisi keşfetme):
   - Kontrolündeki bir cloud bucket (S3/GCS/R2/Backblaze), **veya**
   - Kiralanan makine ulaşılabilir bir port açıyorsa SSH üzerinden kendi makinene `rsync`/`scp`, **veya**
   - Compute sağlayıcısı compute instance'tan ayrı bir depolama hacmi sunuyorsa, ikinci ve
     bağımsız bir kiralık depolama (böylece bir compute-instance geri-alımı depoyu da silmez).
2. **Koşu canlıyken**, kiralanan makinede ayrı bir terminal/oturumda, yeni checkpoint dosyalarını
   tara ve göründükçe dışarı senkronla. Basit bir döngü (1. adımda seçilen hedefe göre uyarla):
   ```bash
   # Örnek: S3-uyumlu bir bucket'a her 5 dakikada bir senkron.
   while true; do
     aws s3 sync ./checkpoints/mertformer_titan_prod/ s3://<your-bucket>/mertformer_titan_prod/ \
       --exclude "*.tmp"
     sleep 300
   done
   ```
   ```bash
   # Örnek: SSH üzerinden kendi makinene rsync (kiralanan makineDEN çalıştırılır).
   while true; do
     rsync -avz --exclude "*.tmp" ./checkpoints/mertformer_titan_prod/ \
       you@your-home-machine:/path/to/backup/mertformer_titan_prod/
     sleep 300
   done
   ```
3. **Her senkrondan sonra**, en azından `best.pt`/`latest.pt` dosyaları için off-site kopyanın
   dosya boyutu/`sha256sum`'ının makine-üstü kopyayla eşleştiğini nokta-kontrol et — kısmen
   senkronlanmış bir checkpoint, checkpoint'in hiç olmamasından daha kötüdür (dürüst bir "burada
   hiçbir şey yok" yerine sessiz bozulma).
4. **Koşu bittikten sonra** (başarı, preemption veya elle durdurma), son bir tam senkron yap ve
   kiralanan instance'ı bırakmadan/sonlandırmadan önce doğrula. Off-site kopya tamamlandığı teyit
   edilmeden instance'ı sonlandırma.

## Bu runbook'un bilerek yapmadığı şeyler

- Senin için belirli bir cloud sağlayıcı seçmiyor — bu, gerçek 45K koşusu için kullanılacak
  compute lane'ine ve bütçeye bağlı bir maliyet/erişim kararı, o karardan önce burada
  hardcode edilmemeli.
- Bunu `scripts/launch_ocean_45k.sh`/`scripts/launch_8xb300.sh`'a otomatik bir arka-plan adımı
  olarak bağlamıyor. Bunu güvenle yapmak, launch script'inin kendi içinde kimlik-bilgisi
  enjeksiyonu, retry/backoff ve kısmi-yazım tespiti gerektirir — bir runbook değil, gerçek bir
  mühendislik işi. Bu belge, o otomasyon var olana kadar elle köprü (BACKLOG.md'de takip ediliyor).
