# Dahili AGI Boşluk Haritası (Build 30 V2)

Bu belge dahili çözülememiş-problemler kaydıdır. Kamusal AGI iddiası değildir ve repo readiness durumunu yetenek kanıtına yükseltmez.

## Güncel Truth Sınırı
- Güncel repo-side readiness verdict: `TRAIN_ALLOWED`
- Güncel repo-side reason code: `READY_REMOTE_BOOTSTRAP`
- Güncel önerilen aktif lane: `remote_bootstrap`
- Sıkı yerel lane: `offline_clean`
- Kalan non-winning blocker'lar: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`
- En önemli eksik evidence sınıfı: gerçek owned training run, trained checkpoint'ler, checkpoint-bound benchmark'lar, trained demo bundle ve trained export/device measurements

## Durum Göstergesi
- `implemented_scaffold`: kod yüzeyi var, fakat claim-grade kanıt yok
- `partial_research`: bazı bileşenler var, ama çekirdek problem hâlâ açık
- `open_problem`: bugün ikna edici bir kapanış yok

## Çözülememiş Matematik + Sistem Register'ı

| Problem Sınıfı | Neden Hâlâ Açık | Mevcut Repo Scaffold | Durum | Gerçek İlerleme Sayılacak Şey |
| --- | --- | --- | --- | --- |
| Quadratic veya subquadratic context scaling | Uzun bağlam maliyeti ve bellek büyümesi pratik ölçeklemeyi hâlâ zorluyor. | `layers/mla.py`, long-context konumlandırması, runtime notları | `partial_research` | Uzun bağlamda sürdürülebilir kalite ve sınırlı maliyet ile ölçülmüş eğitim/inference kanıtı |
| Memory wall ve host-device bandwidth | Low-bit ağırlıklar activation, cache ve transfer darboğazlarını tek başına kaldırmıyor. | BitNet tarzı katmanlar, export/runtime notları, benchmark scaffold'ları | `partial_research` | Ölçülmüş bandwidth profili, cache politikası kanıtı ve trained checkpoint üstünde uçtan uca throughput kazancı |
| Low-bit ve sparse training stability | Sparse routing ve low-bit matematik gradient, routing veya convergence tarafını bozabilir. | `layers/bitlinear.py`, `layers/moe.py`, `scripts/titan_preflight.py`, güvenlik korumaları | `partial_research` | Uzun koşu convergence kanıtı ve dense/yüksek hassasiyet baseline'larına karşı ablation |
| Router collapse ve expert load balance | MoE'nin faydası, zaman içinde sağlıklı expert kullanımı gerektirir. | `layers/moe.py`, router health sinyalleri, tolerance check'leri | `partial_research` | Gerçek koşu telemetrisi ile kararlı expert kullanımı ve collapse olmadan kalite kazancı |
| Continual learning ve catastrophic forgetting | Yeni yeteneği öğrenirken eski yeteneği korumak hâlâ çözülememiş alandır. | `train/continual_adapter.py`, feature flag'ler, roadmap dokümanları | `implemented_scaffold` | Ardışık görevlerde eski yeteneğin korunduğunu gösteren ölçülmüş kanıt |
| Calibrated uncertainty ve abstention | Modelin ne zaman “bilmiyorum” demesi gerektiği çözülmüş değil. | Governance ve verification yüzeyleri var, ama calibrated uncertainty katmanı yok | `open_problem` | Confidence calibration benchmark'ları, abstention policy ve ölçülmüş truthfulness artışı |
| Long-horizon credit assignment | Uzun zincirli planlama, yerel next-token tahminden çok daha zordur. | Orchestrator planner, verifier ve runtime scaffold'ları | `implemented_scaffold` | Doğrulanmış görev tamamlama ile kararlı çok adımlı planlama benchmark'ları |
| Causal abstraction ve counterfactual reasoning | Örüntü tamamlama, güvenilir neden-sonuç akıl yürütmenin yerine geçmez. | World-model ve cognitive-extension scaffold'ları | `implemented_scaffold` | Intervention veya counterfactual değerlendirmeli kontrollü causal görevler |
| World modeling ve partial observability | Belirsizlik altında güvenilir latent world state burada çözülmüş değil. | `layers/world_model_head.py`, orchestrator sensing modülleri | `implemented_scaffold` | Interactive veya simüle ortamlarda ölçülmüş prediction kalitesi |
| Tool-grounded planning reliability | Tool use ancak araç çıktıları doğrulanıp güvenli recovery yapılabiliyorsa değerlidir. | `orchestrator/tool_executor.py`, governance, verifier, swarm runtime | `implemented_scaffold` | Safety check, verification ve düşük failure rate ile tool-use benchmark'ları |
| Mechanistic interpretability'den intervention'a geçiş | İç yapıyı okumak yetmez; steering/intervention kanıtı eksik. | Audit ve verifier yüzeyleri, raporlama disiplini | `open_problem` | Kaliteyi bozmadan davranışı öngörülebilir biçimde değiştiren intervention kanıtı |
| Adversarial robustness ve auditability | Güçlü sistemler, prompt saldırıları ve misuse altında da güvenilir ölçüm ister. | Policy'ler, governance dokümanları, failure-budget mantığı, tool-abuse notları | `partial_research` | Bağımsız red-team sonuçları, jailbreak direnci kanıtı ve audit-grade trace'ler |

## Repo Closure Sonrası da Açık Kalan Yetenek Boşlukları
- İnsan seviyesinde novel problem solving
- Dar prompt hileleri olmadan alanlar arası transferable planning
- Uzun süren agent iş yüklerinde memory reliability
- Gerçek görevlere bağlı grounded multimodal understanding
- Baskı, belirsizlik ve adversarial prompting altında robust truthfulness
- Sınırları belirli failure mode'larla safe tool-grounded execution
- Dış inceleme karşısında ayakta kalan auditability standartları

## Bugün Ne Var, Ne Yok

### Anlamlı implemented scaffold'lar
- Memory, planner, verifier, governance, self-audit ve swarm runtime yüzeyleri kodda mevcut.
- Zero-touch training ve post-train orchestration yüzeyleri mevcut.
- Readiness, freeze, manifest ve claim-boundary governance alışılmadık derecede açık.

### Hâlâ evidence olarak var olmayan şeyler
- Gerçek uzun training run
- Trained checkpoint hikâyesi
- Checkpoint-bound benchmark kanıtı
- Trained artifact üstünde ölçülmüş device/runtime kanıtı
- AGI dili için herhangi bir bağımsız dayanak

## Dahili Özet
- AGI yakınlığı: uzak
- Repo-side engineering closure: güçlü
- Daha güçlü claim'lerin önündeki ana engel: klasör sayısı değil evidence eksikliği
- En önemli sonraki adım: owned run + checkpoint-bound ölçüm

## Politika Notu
Bu dosya dahili kullanıma yöneliktir. Yetkinlik iddiası değil, çözülememiş matematik ve sistem problemleri kaydıdır.
