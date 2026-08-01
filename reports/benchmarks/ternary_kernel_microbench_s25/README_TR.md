# Ternary CPU-Kernel Mikrobenchmark'ı — Galaxy S25 (NEON)

**İddia sınıfı: ölçülmüş** · **Tarih: 2026-06-27** · **Cihaz: fiziksel Samsung Galaxy S25 (tek-thread, CPU/NEON)**

## Kapsam sınırı (önce bunu oku)
Bu, performans kanıtı olarak elle yazılmış tek bir ternary matris-çarpım
kernel'inin **bağımsız, tek-işlemlik bir mikrobenchmark'ı**. **DEĞİL**:
- tam-model token/sn,
- bir NPU ölçümü (bu CPU/NEON, S25 NPU'su değil),
- kanonik `bitlinear.py` / `triton_fused_bitlinear.py` yoluna entegre.

Bu, **ilgili, ölçülmüş kernel kanıtı** — ana mimari performansı değil.
Bu reponun kendi sınırlarını (ölçülmüş / hedef / vizyon) ele aldığı şekilde
ele alın.

## Kurulum
- N=256, ITERS=8, seed=1453, ağırlıklar ternary {-1,0,+1}, aktivasyonlar uniform[-1,1]
- Float baseline ve ternary kernel'ler **birebir aynı** N/ITERS/veri/seed kullanıyor
- Zemin doğruluk **double precision**'da; **`-ffast-math` yok**
- Tek thread (S25 `nproc` yolu); sayılar **fiziksel cihazdan**, qemu'dan değil

## Sonuçlar (en iyi config = 4×8)

| Kademe | Süre (ms) | Hızlanma | Doğruluk |
|------|-----------|---------|----------|
| float-naive (referans) | 426.5 | 1.00× | — |
| **NEON FMA exact** | 141.9 | **3.01×** | **bit-exact** (float'a karşı max fark = 0.000e+00) |
| **NEON SDOT turbo** | 51.4 | **8.29×** | **yaklaşık** (~%0.4 rms, int8 aktivasyonlar) |

İki dürüstlük notu:
1. **FMA kademesi bit-exact** — float baseline ile birebir aynı matematik (w ∈ {-1,0,+1}
   için x·w tam), yani hızlanma sıfır hatayla gerçek.
2. **SDOT kademesi yaklaşık** — aktivasyonları int8'e kuantize ediyor ve `vdotq_s32`
   kullanıyor (16 MAC/komut). Hızlı ama **bit-exact değil** (~%0.4 rms). Bu, düşük-bit
   deployment-modu rakamı, öyle raporlanıyor.

## Bunun KANITLAMADIĞI şeyler
Tam-model token/sn, NPU hızı, uçtan-uca gecikme, 3.67B modelin herhangi bir
hızda çalıştığı, ya da production/mobile hazırlığı. Yetenek hâlâ 45K
checkpoint-bağımlı koşuya bağlı.

## Yeniden üretilebilirlik
Kaynak: `ternary_matmul_arm.cpp` (bu klasörde). FMA kademeleri çalışma zamanında
`max|kernel − float_naive| = 0` diye kendi kendini doğruluyor; SDOT double zemin
doğruluğa karşı rms raporluyor. Kernel mantığı hem gcc-aarch64 hem clang ile
cross-compile edildi ve cihaz-üstü ölçümden önce qemu-aarch64 altında koşuldu.

Cihazda derle & çalıştır (CxxDroid):
```
g++ -O3 ternary_matmul_arm.cpp -o tern && ./tern
# opsiyonel ekstra: -O3 -march=armv8.2-a+dotprod
```
