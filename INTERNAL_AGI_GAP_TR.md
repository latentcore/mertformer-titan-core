# Dahili AGI Boşluk Haritası (v1.0 (Build 30))

Bu doküman, **AGI-turu yetenek alanlarını** MertFormer v1.0 (Build 30) durumuyla eslestiren dahili bir gerçeklik kontroludur.
**Kamusal bir iddia değildir** ve dahili yol haritası referansı olarak tutulmalıdır.

Gösterge:
- ✅ Var
- 🟡 Kısmı / altyapı var
- 🔴 Yok / plan aşamasinda

## MertFormer v1.0 (Build 30) vs. AGI Yetenek Haritası

| Alan | AGI Hedefi | MertFormer v1.0 (Build 30) | Kanıt | Boşluk / Risk | Sonraki Adım |
| --- | --- | --- | --- | --- | --- |
| Genel akıl yürütme | Alanlar arası transfer | 🟡 Mimari hazır, eğitim kanıtı yok | README, config | Gerçek run yok | Master Run + bench |
| Uzun süreli bellek | Kalıcı geri çağırma | 🟡 Orchestrator memory var | orchestrator/memory.py | Üretim kanıtı yok | Retrieval demo |
| Grounding | Gerçek dünya etkileşimi | 🔴 Metin ağırlıklı | - | Çevre döngüsü yok | Offline agent demo |
| Planlama | Çok adimli hedef tutarlılığı | 🟡 Orchestrator core var | orchestrator/core.py | Stres testi yok | Task runner demo |
| Self-audit | Çıktı doğrulama | 🔴 Yok | - | Halüsinasyon riski | Verifier head |
| Belirsizlik | Güven kalibrasyonu | 🔴 Yok | - | Güven riski | Uncertainty head |
| Tool-use güvenliği | Güvenli araç kullanımi | 🟡 Sensing modülleri var | orchestrator/*_sense.py | Sandbox/contract yok | Tool contracts |
| MoE adaptivligi | Dinamik uzman dengesi | 🟡 Var ama sabit | layers/moe.py | Adaptif değil | Adaptive MoE |
| Online learning | Güvenli sürekli öğrenme | 🔴 Yok | - | Stabilite/gvenlik riski | Controlled updates |
| Transfer | Hızlı adaptasyon | 🟡 Distill + curriculum | scripts/data_pipeline.py | Gerçek eval yok | Bench outputs |
| Alignment | Güvenli kullanım sınırı | 🟡 Kill switch + gate | scripts/operator_mode_gate.py | Red-team yok | Red-team tests |
| Robustness | Stres altında stabilite | 🟡 Failure budget | orchestrator/failure_budget.py | Scale testi yok | Stress tests |
| Evaluation | Ölçülmüş performans | 🟡 Runner var | scripts/benchmarks_internal.py | Gerçek çıktı yok | HumanEval/MBPP |
| Edge operasyonu | Offline / cihaz içi | 🟡 Hedef var | README, export scripts | Cihaz kanıtı yok | Device demo |
| Verimlilik | Düşük enerji/bellek | 🟡 BitNet sim | layers/bitlinear.py | Kernel yok | Low-bit inference |
| Swarm çalışma | Çok ajanli koordinasyon | 🟡 Hedef mimari | README Ek v5.2 | Uygulama yok | Small swarm demo |
| Self-improvement | Hatadan öğrenme | 🟡 SAGE vizyonu | README Swarm v5.2 | Uygulama yok | Post-mortem loop |
| Ethics / use policy | Açık kullanım sınırı | 🟡 Lisans | LICENSE | Politika eksik | USE_POLICY |
| Data lineage | Kaynak izi | 🟡 Kısmı doküman | datasets/README.md | Tam manifest yok | Dataset manifests |
| Reproducibility | Tam tekrar üretim | 🟡 Sablonlar | repro/* | Gerçek CUDA lock yok | write_cuda_lock |

## Özet (Dahili)
- AGI yakınlığı: **uzak**
- Sistem prototip olgunlugu: **yüksek**
- En kritik kanıt eksiği: **gerçek eğitim + benchmark + demo**
- En kritik yetenek boşlukları: **grounding, self-audit, belirsizlik, güvenilir uzun bellek**

## Politika Notu
Bu doküman dahili kullanıma yöneliktir. AGI iddiası değildir.
