# Known Issues (Commercial Handover)

Bu dosya, devir anında bilinen teknik/operasyonel açıkları risk seviyesiyle listeler.

## Current Status
- Technical GO: PASS (bkz. `reports/go_nogo_signoff_onepager.md`)
- Commercial Claim GO: NO-GO (external pending)

## Issues
| ID | Issue | Risk | Impact | Mitigation | Owner | Exit Criteria | Evidence |
|---|---|---|---|---|---|---|---|
| KI-001 | CI disk exhaustion during fresh dependency bootstrap | Medium | `verify_all` iş akışı fail olabilir | CI wheel/cache politikasını sabitle, Python env tekrar kullanımını zorunlu kıl | DevOps | 3 ardışık CI run PASS | `reports/final_repo_audit.md` |
| KI-002 | External legal counsel sign-off pending | High | Ticari claim/kapanış gecikmesi | Dış hukuk onayı + imzalı ek protokol | Legal | Yazılı legal sign-off | `reports/go_nogo_signoff_onepager.md` |
| KI-003 | Paid pilot / LOI closure pending | High | Gelir başlangıcı ve üretim geçişi gecikir | Pilot kapsamı + kabul kriteri + ödeme planı | BizDev | 1 ücretli pilot veya imzalı LOI | `reports/go_nogo_signoff_onepager.md` |
| KI-004 | Independent security/compliance sign-off pending | High | Kurumsal satın alma gecikmesi | 3. parti pentest/compliance raporu | Security | Pentest + compliance raporu | `reports/go_nogo_signoff_onepager.md` |
| KI-005 | Old docs wording drift risk after rapid releases | Low | Satış/teknik anlatımda tutarsızlık | Release sonrası doc lint + claim consistency zorunlu gate | Documentation | Lint/consistency gate PASS | `scripts/check_doc_claim_consistency.py` |
| KI-006 | Resolved: `liquid_spike_counter` wired + regression test added | Low | Safeguard telemetry now active (no runtime risk) | Implemented spike threshold/patience/cooldown tracking with tests | Core Training | ✅ Done (2026-03-15) | `utils/liquid_safeguard.py`, `tests/test_liquid_safeguard.py` |

## Notes
- Bu dosya canlıdır; her release sonrası güncellenir.
- Devirde alıcıya açık şekilde paylaşılır.
