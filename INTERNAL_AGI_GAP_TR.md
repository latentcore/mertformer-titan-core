# Dahili AGI Bosluk Haritasi (v1.0 (Build 27))

Bu dokuman, **AGI-turu yetenek alanlarini** MertFormer v1.0 (Build 27) durumuyla eslestiren dahili bir gerceklik kontroludur.
**Kamusal bir iddia degildir** ve dahili yol haritasi referansi olarak tutulmalidir.

Gosterge:
- ✅ Var
- 🟡 Kismi / altyapi var
- 🔴 Yok / plan asamasinda

## MertFormer v1.0 (Build 27) vs. AGI Yetenek Haritasi

| Alan | AGI Hedefi | MertFormer v1.0 (Build 27) | Kanit | Bosluk / Risk | Sonraki Adim |
| --- | --- | --- | --- | --- | --- |
| Genel akil yurutme | Alanlar arasi transfer | 🟡 Mimari hazir, egitim kaniti yok | README, config | Gercek run yok | Master Run + bench |
| Uzun sureli bellek | Kalici geri cagirma | 🟡 Orchestrator memory var | orchestrator/memory.py | Uretim kaniti yok | Retrieval demo |
| Grounding | Gercek dunya etkilesimi | 🔴 Metin agirlikli | - | Cevre dongusu yok | Offline agent demo |
| Planlama | Cok adimli hedef tutarliligi | 🟡 Orchestrator core var | orchestrator/core.py | Stres testi yok | Task runner demo |
| Self-audit | Cikti dogrulama | 🔴 Yok | - | Halusinasyon riski | Verifier head |
| Belirsizlik | Guven kalibrasyonu | 🔴 Yok | - | Guven riski | Uncertainty head |
| Tool-use guvenligi | Guvenli arac kullanimi | 🟡 Sensing modulleri var | orchestrator/*_sense.py | Sandbox/contract yok | Tool contracts |
| MoE adaptivligi | Dinamik uzman dengesi | 🟡 Var ama sabit | layers/moe.py | Adaptif degil | Adaptive MoE |
| Online learning | Guvenli surekli ogrenme | 🔴 Yok | - | Stabilite/gvenlik riski | Controlled updates |
| Transfer | Hizli adaptasyon | 🟡 Distill + curriculum | scripts/data_pipeline.py | Gercek eval yok | Bench outputs |
| Alignment | Guvenli kullanim siniri | 🟡 Kill switch + gate | scripts/operator_mode_gate.py | Red-team yok | Red-team tests |
| Robustness | Stres altinda stabilite | 🟡 Failure budget | orchestrator/failure_budget.py | Scale testi yok | Stress tests |
| Evaluation | Olculmus performans | 🟡 Runner var | scripts/benchmarks_internal.py | Gercek cikti yok | HumanEval/MBPP |
| Edge operasyonu | Offline / cihaz ici | 🟡 Hedef var | README, export scripts | Cihaz kaniti yok | Device demo |
| Verimlilik | Dusuk enerji/bellek | 🟡 BitNet sim | layers/bitlinear.py | Kernel yok | Low-bit inference |
| Swarm calisma | Cok ajanli koordinasyon | 🟡 Hedef mimari | README Ek v5.2 | Uygulama yok | Small swarm demo |
| Self-improvement | Hatadan ogrenme | 🟡 SAGE vizyonu | README Swarm v5.2 | Uygulama yok | Post-mortem loop |
| Ethics / use policy | Acik kullanim siniri | 🟡 Lisans | LICENSE | Politika eksik | USE_POLICY |
| Data lineage | Kaynak izi | 🟡 Kismi dokuman | datasets/README.md | Tam manifest yok | Dataset manifests |
| Reproducibility | Tam tekrar uretim | 🟡 Sablonlar | repro/* | Gercek CUDA lock yok | write_cuda_lock |

## Ozet (Dahili)
- AGI yakinligi: **uzak**
- Sistem prototip olgunlugu: **yuksek**
- En kritik kanit eksigi: **gercek egitim + benchmark + demo**
- En kritik yetenek bosluklari: **grounding, self-audit, belirsizlik, guvenilir uzun bellek**

## Politika Notu
Bu dokuman dahili kullanima yoneliktir. AGI iddiasi degildir.
