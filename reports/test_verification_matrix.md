# Test and Verification Matrix

| Depth | Canonical Command | Scope |
| --- | --- | --- |
| unit/integration baseline | `python3 -m pytest -q` | repo-wide Python test surface |
| packed projection equivalence | `python3 -m pytest -q tests/test_packed_projection_equivalence.py` | optional FFN/MoE/MLA packed projection paths vs baseline math |
| Liquid train implementation safeguard | `python3 -m pytest -q tests/test_liquid_safeguard.py` | optional Liquid train implementation variants vs baseline behavior |
| final orchestrator launch profile | `python3 -m pytest -q tests/test_final_orchestrator_cli.py` | accelerate config selection and 1024-first clear-OOM-only batch fallback orchestration |
| code-truth audit | `python3 scripts/build_code_truth_audit.py` | maturity labels and four-column done rule |
| closure governance | `python3 scripts/build_closure_governance_pack.py` | source-of-truth, backlog, known-limits, support, ADR, scorecard |
| offline verify ladder | `bash scripts/verify_all.sh` | canonical repo verification and sync refresh |
| one-command SOP | `bash scripts/one_command_full_sop.sh` | closure validation plus packaging/refresh ladder |
| final closeout | `bash scripts/final_one_shot.sh` | maximum release-side refresh and handoff surfaces |
| chess delivery contract | `python3 -m pytest -q tests/test_chess_5080_onefile.py tests/test_export_chess_5080_share.py tests/test_build_chess_5080_windows_delivery.py` | chess onefile and delivery lane |
| governance contract | `python3 -m pytest -q tests/test_build_code_truth_audit.py tests/test_build_workspace_hygiene_manifest.py tests/test_build_closure_governance_pack.py` | closure and policy generation |
