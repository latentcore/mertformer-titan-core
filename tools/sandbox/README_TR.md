# Tool Sandbox

Tool kullanımı için güvenli yürütme sınırlarını tanımlar.

## Izinli
- Proje klasorunde salt-okuma erisimi
- Deterministik tool kullanimi (yan etki yok)
- Sinirli CPU/GPU kullanimi

## Yasak
- Sistem duzeyinde yikici islemler
- Onayli env disinda gizli bilgi okuma
- Ag kapaliyken dis URL cagrisi

## Varsayilanlar
- Tool cagrilari acik ve loglu olur
- Her tool icin timeout uygulanir
- Hatalar net neden kodlariyla verilir
