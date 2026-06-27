# Tool Sandbox

Tool kullanımı için güvenli yürütme sınırlarını tanımlar.

## İzinli
- Proje klasöründe salt-okuma erişimi
- Deterministik tool kullanımı (yan etki yok)
- Sınırlı CPU/GPU kullanımı

## Yasak
- Sistem düzeyinde yıkıcı işlemler
- Onaylı env dışında gizli bilgi okuma
- Ağ kapalıyken dış URL çağrısı

## Varsayılanlar
- Tool çağrıları açık ve loglu olur
- Her tool için timeout uygulanır
- Hatalar net neden kodlarıyla verilir
