# Satranç 4060 24 Saat All-On Experimental Runbook

## Amaç
Tek bir RTX 4060 üzerinde chess onefile hattını 24 saatlik, sınırlandırılmış ve geniş feature yüzeyi açık deneysel eğitim koşusunda çalıştırmak.

Profiller:
- `strength_4060_24h_all_on_experimental`
- `strength_4060_24h_omni_max`

## Önerilen Komutlar
Temel all-on experimental:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental
```

Daha agresif omni-max varyantı:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_omni_max
```

Riskli yüzey kapatma örneği:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_omni_max --disable-features use_qinn,use_world_model_head
```

## Açık Yüzeyler
All-on aile profilleri şunları açmayı hedefler:
- MoE / expert paging / cross-expert sync
- Liquid / liquid adapter / QINN
- flash-attn inference + hierarchical KV cache
- workspace / neuromodulatory gain / latent ODE / Hebbian / neuro-symbolic / world-model / lifelong safety
- gradient checkpointing
- yardımcı satranç head’leri: `phase_head`, `wdl_head`, `legality_head`

## Koşu Sonrası Zorunlu Kanıtlar
Run dizini altında en az şu raporlar beklenir:
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
- `reports/external_repro_stub.json`
- `reports/pilot_stub.json`
- `reports/security_stub.json`
- `reports/legal_stub.json`
- `reports/operator_handbook_stub.json`
- `reports/dr_evidence_stub.json`
- `reports/backup_retention_stub.json`
- `reports/blind_handoff_stub.json`
- `reports/release_notes_stub.json`
- `reports/freeze_manifest_stub.json`
- `reports/changelog_snapshot.json`
- `reports/maintenance_policy_stub.json`
- `logs/run_log.jsonl`

## Operatör Kapıları
Koşu öncesi:
- Checkpoint ve bundle için disk alanını doğrula.
- CUDA cihaz görünürlüğünü doğrula.
- Stockfish benchmark bekleniyorsa `stockfish` yolu veya auto-fetch iznini doğrula.
- Deneysel yüzeyleri komutta veya profil adında açıkça belirt.

Koşu sırasında:
- `logs/run_log.jsonl` içinde `fatal_exception`, `oom_event` ve tekrarlayan `midrun_snapshot_stockfish` hatalarını izle.
- Run kararsızsa ilk olarak `use_qinn` veya `use_world_model_head` yüzeylerini kapat.

Koşu sonrası:
- Replay/demo çıktısını güç kanıtı gibi yorumlama.
- Benchmark çıktısını harici tekrar olmadan iç benchmark olarak tut.
- Feature flag raporunu checkpoint bundle ile birlikte koru; böylece tam açık feature/head kombinasyonu audit edilebilir kalsın.
- Self-play / tournament / replay-buffer raporlarını ayrı benchmark kanıtı gibi değil, koşu-sonrası iç teşhis artefaktı olarak sakla.
