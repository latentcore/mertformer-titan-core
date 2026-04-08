# Chess 4060 24h Kanonik Runbook

## Amaç
Donmuş chess lane'i tek bir RTX 4060 üzerinde ilk gerçek 24 saatlik baseline-supported eğitim koşusu için çalıştırmak.

Profil:
- `strength_4060_24h`

Release boundary:
- donmuş chess release-candidate kapısını karşılayabilen tek 4060 profili budur
- `production_5080` desteklenen taşınabilir baseline olarak kalır, ama kanonik uzun koşu release yolu değildir
- `strength_4060_24h_all_on_experimental` ve `strength_4060_24h_omni_max` research-only kalır

## Kanonik Komut
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h
```

## İsteğe Bağlı Desteklenen Varyantlar
Taşınabilir baseline:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile production_5080 --feature-bundle all_stable_extensions
```

Kanonik profil üzerinde koşu-sonrası analiz stack'i:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h --feature-bundle postrun_analysis_stack
```

## Koşu Öncesi Kapılar
- CUDA görünürlüğünü ve VRAM boşluğunu doğrula.
- Checkpoint, bundle, PGN ve raporlar için disk boşluğunu doğrula.
- Seçilen artifact root altında `logs/` ve `reports/` yazılabildiğini doğrula.
- Koşu öncesi Stockfish benchmark yolu veya fetch politikasını netleştir.
- Tam komutu ve seçilen profili operatör notlarında kaydet.

## Koşu Sonrası Zorunlu Kanıt
Koşu dizininde şu artefaktları bekle:
- `reports/run_summary.json`
- `reports/run_summary.md`
- `reports/model_card.json`
- `reports/eval_card.json`
- `reports/feature_flag_report.json`
- `reports/feature_flag_report.md`
- `reports/mirror_parity_report.json`
- `reports/curated_position_suite_report.json`
- `reports/legal_move_safety.json`
- `reports/raw_vs_masked_policy_metrics.json`
- `reports/observability_report.json`
- `reports/selfplay_report.json`
- `reports/inference_mode_tournament_report.json`
- `reports/replay_buffer_manifest.json`
- `reports/run_status_manifest.json`
- `reports/postrun_analysis_manifest.json`
- `reports/artifact_truth_matrix.json`
- `reports/run_contract.json`
- `reports/release_snapshot.json`
- `reports/evidence_pack_stub.json`
- `reports/final_truth_registry.json`
- `reports/claim_registry.json`
- `reports/known_limits.json`
- `reports/support_matrix.json`
- `reports/release_gate_summary.json`
- `reports/rc_stub.json`
- `reports/golden_stub.json`
- `reports/handoff_pack_manifest.json`
- `reports/operator_handoff_summary.json`
- `logs/run_log.jsonl`

## Operatör Yorum Kuralları
- Repo-side kapanış ölçülmüş güç anlamına gelmez.
- Self-play, tournament, replay ve proxy-rating çıktıları ayrı doğrulanmadıkça internal kalır.
- `support_matrix.json` içinde aktif ve release-candidate uygun profil olarak `strength_4060_24h` görünmelidir.
- `release_gate_summary.json`, deneysel profillerin release-candidate uygun olmadığını korumalıdır.
- Zorunlu artefakt eksikleri closure için hard failure kabul edilmelidir.

## Eskalasyon Rehberi
- Koşu dengesizleşirse yeniden denemeden önce mevcut artifact root'u koru.
- Feature report beklenmeyen override gösterirse koşuyu contract drift say.
- `known_limits.json` veya `remaining_core_blockers.json` kanonik blocker sözlüğü dışında bir blocker gösterirse bunu truth drift kabul et ve bir sonraki koşudan önce repo'yu düzelt.
