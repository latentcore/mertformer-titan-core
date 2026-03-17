# Bilinen Sorunlar (Ticari Devir)

Bu dosya, devir anında bilinen teknik/operasyonel açıkları risk seviyesiyle listeler.

## Mevcut Durum
- Technical GO: PASS (bkz. `reports/go_nogo_signoff_onepager.md`)
- Commercial Claim GO: NO-GO (external pending)

## Sorunlar
| ID | Sorun | Risk | Etki | Mitigasyon | Sorumlu | Çıkış Kriteri | Kanıt |
|---|---|---|---|---|---|---|---|
| KI-001 | İlk bağımlılık kurulumunda CI diskinin dolması | Medium | `verify_all` iş akışı fail olabilir | CI wheel/cache politikasını sabitle, Python env tekrar kullanımını zorunlu kıl | DevOps | 3 ardışık CI run PASS | `reports/final_repo_audit.md` |
| KI-002 | Dış hukuk danışmanı sign-off bekleniyor | High | Ticari claim/kapanış gecikmesi | Dış hukuk onayı + imzalı ek protokol | Legal | Yazılı legal sign-off | `reports/go_nogo_signoff_onepager.md` |
| KI-003 | Ücretli pilot / LOI kapanışı bekleniyor | High | Gelir başlangıcı ve üretim geçişi gecikir | Pilot kapsamı + kabul kriteri + ödeme planı | BizDev | 1 ücretli pilot veya imzalı LOI | `reports/go_nogo_signoff_onepager.md` |
| KI-004 | Bağımsız güvenlik/uyumluluk sign-off bekleniyor | High | Kurumsal satın alma gecikmesi | 3. parti pentest/compliance raporu | Security | Pentest + compliance raporu | `reports/go_nogo_signoff_onepager.md` |
| KI-005 | Hızlı release sonrası doküman wording drift riski | Low | Satış/teknik anlatımda tutarsızlık | Release sonrası doc lint + claim consistency zorunlu gate | Documentation | Lint/consistency gate PASS | `scripts/check_doc_claim_consistency.py` |
| KI-006 | Çözüldü: `liquid_spike_counter` bağlandı + regresyon testi eklendi | Low | Safeguard telemetry artık aktif (runtime riski yok) | Spike threshold/patience/cooldown takibi ve testleri eklendi | Core Training | ✅ Done (2026-03-15) | `utils/liquid_safeguard.py`, `tests/test_liquid_safeguard.py` |

## Notlar
- Bu dosya canlıdır; her release sonrası güncellenir.
- Devirde alıcıya açık şekilde paylaşılır.
