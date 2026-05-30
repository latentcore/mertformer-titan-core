# Test ve Doğrulama Matrisi

| Derinlik | Kanonik Komut | Kapsam |
| --- | --- | --- |
| unit/integration baseline | `python3 -m pytest -q` | repo genelindeki Python test yüzeyi |
| packed projection equivalence | `python3 -m pytest -q tests/test_packed_projection_equivalence.py` | opsiyonel FFN/MoE/MLA packed projection yollarının baseline matematikle eşdeğerliği |
| Liquid eğitim implementasyonu safeguard | `python3 -m pytest -q tests/test_liquid_safeguard.py` | opsiyonel Liquid eğitim implementasyonu varyantlarının baseline davranışa karşı kontrolü |
| final orchestrator launch profili | `python3 -m pytest -q tests/test_final_orchestrator_cli.py` | Accelerate config seçimi ve 1024-first net-OOM-only batch fallback orchestration |
| code-truth audit | `python3 scripts/build_code_truth_audit.py` | maturity etiketleri ve dört sütunlu done kuralı |
| closure governance | `python3 scripts/build_closure_governance_pack.py` | source-of-truth, backlog, known-limits, support, ADR ve scorecard yüzeyleri |
| offline verify ladder | `bash scripts/verify_all.sh` | kanonik repo doğrulaması ve sync yenileme hattı |
| one-command SOP | `bash scripts/one_command_full_sop.sh` | closure doğrulaması, paketleme ve yenileme hattı |
| final closeout | `bash scripts/final_one_shot.sh` | maksimum release-side yenileme ve handoff yüzeyleri |
| chess delivery contract | `python3 -m pytest -q tests/test_chess_5080_onefile.py tests/test_export_chess_5080_share.py tests/test_build_chess_5080_windows_delivery.py` | satranç onefile ve delivery hattı |
| governance contract | `python3 -m pytest -q tests/test_build_code_truth_audit.py tests/test_build_workspace_hygiene_manifest.py tests/test_build_closure_governance_pack.py` | closure ve policy üretimi |
