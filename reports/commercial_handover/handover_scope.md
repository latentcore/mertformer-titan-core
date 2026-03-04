# Handover Scope (Commercial + Technical)

## In Scope (Transferred)
1. Kaynak kod, testler, benchmark scriptleri ve CI workflow'ları.
2. Release artefaktları ve kanıt zinciri (manifest, checksum, raporlar).
3. Teknik dokümantasyon (README, model kartı, policy, runbook).
4. Operasyonel script seti (`verify_all`, gate kontrolleri, audit scriptleri).
5. Dealroom referans ve ownership/provenance kanıt paketi.

## Out of Scope (Not Transferred in this pass)
1. Gerçek 2.64B tam eğitim run'ının başlatılması.
2. Üçüncü taraf hukuk, pentest, bağımsız compliance raporunun üretimi.
3. Müşteri/üretim altyapısının alıcı tarafında işletim maliyetleri.
4. Yeni araştırma modülleri (multimodal unification, ASIC/HDL, global ingest).

## Acceptance Boundaries
- Technical acceptance: gate'ler yeşil + release artefakt zinciri doğrulanmış.
- Commercial acceptance: sözleşme ve dış onaylar tamamlanmış.

## Transfer Deliverables
- `reports/release_manifest.json`
- `reports/final_repo_audit.md`
- `reports/ownership_proof_bundle.json`
- `reports/go_nogo_signoff_onepager.md`
- `artifacts/mertformer_release.zip` ve `artifacts/mertformer_release.zip.sha256`
