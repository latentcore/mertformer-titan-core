# MertFormer Titan One-Pager (TR)

> **Dış inceleme notu:** Compute sponsorship değerlendiriyorsanız önce
> `private/commercial/outreach_compute_sponsorship_messages.md` okunmalıdır. Bu one-pager
> arka plan materyalidir; benchmark, production, deployment, AGI veya model
> üstünlüğü iddiası olarak okunmamalıdır.

**Version:** Build 30 V2

## Özet
MertFormer Titan, mobil hesaplama maliyetinde GPT-3.5 sınıfı bir verimlilik hedefini amaçlayan, 2.64B tasarım hedefli edge-native bir kodlama modelidir (mevcut ölçülen runtime toplamı: ~3.67B). 1.58-bit kuantizasyon, LiquidRouter MoE ve uzun bağlam dikkatini birleştirir.

## Problem
Kurumsal AI pahalıdır, buluta bağımlıdır ve gizlilik açısından risk taşır. Gecikme, veri egemenliği ve maliyetler regüle sektörlerde benimsemeyi zorlaştırır.

## Çözüm
Gizlilik ve verimlilik odaklı, cihaz içi çalışan mobil-first kodlama modeli. Enerji tasarrufu, düşük gecikme ve veri kontrolünü önceliklendirir.

## Ürün
- Cihaz‑içi inference ile edge-native kodlama modeli
- 1.58‑bit kuantizasyon ile ciddi bellek tasarrufu
- LiquidRouter MoE ile stabil uzman yönlendirme
- Kod tabanları için uzun bağlam desteği

## Farklılaştırıcılar
- Donanım‑farkındalıklı mimari
- Zaman‑bağımlı yönlendirme (LiquidRouter) daha iyi uzman dengesi hedefler; nihai kazançlar trained-checkpoint evidence gerektirir.
- Forensik loglama ve reproducibility gate’leri

## Yol Haritası
- Operator‑mode gate’ler ve failure budget
- Telemetri ile yönlendirilen master training
- HumanEval/MBPP benchmark çıktıları

## Proje Durumu
Bu, tek kişilik, kendi kendini finanse eden bir araştırma ve mühendislik projesidir — bir şirket veya ürün değildir. Şu aşamada ticari bir teklif, lisanslama anlaşması veya kurumsal deployment yoktur. Mimari, güvenlik-guard sistemi ve eğitim pipeline'ı uygulanmış ve test edilmiştir (bkz. `reports/verified_matrix_TR.md` ve `README_TR.md`); büyük ölçekli eğitilmiş-checkpoint kanıtı henüz yoktur çünkü master training run için kaynak ayrılmamıştır.

## İstek
- Mimari, güvenlik-guard tasarımı ve kanıt-disiplini pratiklerinin teknik incelemesi/eleştirisi.
- Master training run için compute sponsorluğu — tam kapsam ve maliyet için `private/commercial/outreach_compute_sponsorship_messages.md`'ye bakın.
- Bu işin ilgili olduğu araştırma veya mühendislik rolleri, stajlar veya işbirliği.


V2 refactor: dedup pipeline, parallel MoE dispatch, CfC fast path, stricter train gates.
