# MertFormer Titan One-Pager (TR)

## Özet
MertFormer Titan, mobil hesaplama maliyetinde GPT-3.5 hedefli bir seviyeyi amaçlayan, 2.64B tasarım hedefli edge-native bir kodlama modelidir (mevcut ölçülen runtime toplamı: ~3.70B). 1.58-bit kuantizasyon, LiquidRouter MoE ve uzun bağlam dikkatini birleştirir.

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
- Zaman‑bağımlı yönlendirme (LiquidRouter)
- Forensik loglama ve reproducibility gate’leri

## Yol Haritası
- Operator‑mode gate’ler ve failure budget
- Telemetri ile yönlendirilen master training
- HumanEval/MBPP benchmark çıktıları

## İş Modeli
Kurumsal on‑device deploy ve özel inference lisanslama.

## İstek
Edge kodlama pilot ortakları ve stratejik dağıtım tasarımı.
