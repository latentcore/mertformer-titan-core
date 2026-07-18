# Off-site checkpoint yedekleme — runbook

**Bu neden var:** 2026-05-14 tarihli 2xH200 kısmi koşusu step 1880'e kadar eğitim gördü ve
checkpoint'i **kalıcı olarak kayboldu** — hiçbir off-site kopya yoktu, kiralanan makinenin
depolaması geri alınmadan önce kurtarılamadı. Bu runbook'un varlık sebebi tam olarak bu somut
başarısızlığı önlemek.

**Güncelleme (2026-07-19): otomasyon artık var.** `scripts/offsite_backup_watcher.py`, hem
`scripts/launch_8xb300.sh`'a hem `scripts/launch_ocean_45k.sh`'a bağlandı ve launch anında
`TITAN_OFFSITE_BACKUP_DEST` set edildiğinde otomatik başlıyor (varsayılan kapalı — aksi halde
no-op, aşağıdaki "Otomatik yol"a bakın). Bu runbook'taki elle prosedür, belgeli yedek ve
nokta-kontrol çapraz-referansı olarak (aşağıdaki 3. adım) kalıyor — artık tek seçenek değil ama
gereksiz de değil.

## Otomatik yol (tercih edilen)

1. Hedefi belirle (aşağıdaki elle prosedürle aynı üç seçenek) ve launch'tan önce dışa ver:
   ```bash
   export TITAN_OFFSITE_BACKUP_DEST="s3://your-bucket/mertformer_titan_prod/"
   # ya da: export TITAN_OFFSITE_BACKUP_DEST="you@your-home-machine:/path/to/backup/"
   # ya da: export TITAN_OFFSITE_BACKUP_DEST="gs://your-bucket/mertformer_titan_prod/"
   ```
2. Her zamanki gibi başlat (`bash scripts/launch_8xb300.sh --go` ya da `bash scripts/launch_ocean_45k.sh --go`).
   Launch script watcher'ın başlayıp başlamadığını yazdırır; watcher'ın kendi logu `logs/` altına
   düşer (b300 launcher'da `logs/offsite_backup_watcher.log`, ocean launcher'da
   `logs/launch/offsite_backup_watcher_<RUN_ID>.log`).
3. Watcher her `TITAN_OFFSITE_BACKUP_INTERVAL_SECONDS`'ta (varsayılan 300sn) bir izliyor, en yeni
   checkpoint dosyası `TITAN_OFFSITE_BACKUP_STABILITY_SECONDS` (varsayılan 30sn) içinde
   değiştirilmişse o döngüyü atlıyor (hâlâ yazılıyor olabilecek bir dosyayı senkronlamaya karşı
   best-effort koruma), sonra hedefin şemasına göre otomatik seçilen `aws s3 sync`/`gsutil rsync`/`rsync`
   ile, hata durumunda retry+backoff'lu senkronluyor.
4. `scripts/launch_ocean_45k.sh`'ta watcher, koşu bitince temizce trap-kill ediliyor (o scriptteki
   mevcut `nvidia-smi dmon` telemetri process'iyle aynı örüntü). `scripts/launch_8xb300.sh`'ta
   watcher, o scriptin kendi `exec`'inden ÖNCE başlatılıyor ve otomatik durdurulmuyor (`exec`
   process'in yerini alıyor, sonrasında hiçbir şey temizlik yapamaz) — koşu bitip aşağıdaki elle
   prosedürün 4. adımı (son tam senkron + doğrulama) yapıldıktan sonra elle durdur.

2026-07-19'dan önce hiçbir launch script'i bunu otomatik yapmıyordu (o zaman kontrol edildi:
`scripts/launch_*.sh` içinde hiçbir yerde `rsync`/`rclone`/`aws s3 sync`/`gsutil` çağrısı yoktu) —
aşağıdaki bölüm hâlâ o boşluğu, otomatik yolun kullanılamadığı durumlar (örn. watcher'ın
tanımadığı bir hedef şeması) veya elle nokta-kontrol için yedek olarak belgeliyor.

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

## (Hâlâ) bilerek yapmadığı şeyler

- Senin için belirli bir cloud sağlayıcı seçmiyor — bu, gerçek 45K koşusu için kullanılacak
  compute lane'ine ve bütçeye bağlı bir maliyet/erişim kararı, o karardan önce burada
  hardcode edilmemeli. `scripts/offsite_backup_watcher.py`, `TITAN_OFFSITE_BACKUP_DEST`'in
  ima ettiği şemaya göre yönleniyor (`s3://`, `gs://` ya da rsync-tarzı) — seçim hâlâ senin,
  launch anında o tek env değişkeniyle yapılıyor.
- Kimlik bilgisi yönetimi hâlâ operatörün sorumluluğunda (örn. `~/.aws/credentials`,
  `gcloud auth`, ya da rsync hedefi için zaten kurulmuş bir SSH anahtarı) — watcher standart
  CLI araçlarını çağırıyor ve o ortamda zaten yapılandırılmış hangi kimlik bilgisi varsa onu
  devralıyor; kendisi kimlik bilgisi yönetmiyor/enjekte etmiyor.
