# Tool Sandbox

Tool kullanımı için güvenli yürütme sınırlarını tanımlar.

## Izinli
- Proje klasorunde salt-okuma erisimi
- Deterministik tool kullanımi (yan etki yok)
- Sinirli CPU/GPU kullanımi

## Yasak
- Sistem duzeyinde yikici islemler
- Onayli env disinda gizli bilgi okuma
- Ag kapaliyken dis URL cagrisi

## Varsayilanlar
- Tool cagrilari acik ve loglu olur
- Her tool için timeout uygulanir
- Hatalar net neden kodlariyla verilir
