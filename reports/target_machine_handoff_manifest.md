# Target Machine Handoff Manifest

- next_action: `DO_NOT_RENT_YET_FIX_REPO_BLOCKERS`
- train_allowed: `False`
- decision_reason_code: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE__online_teacher:MISSING_HF_TOKEN`
- recommended_path: `none`
- bundle_path: `target_machine_handoff_bundle.zip`
- bundle_sha256_path: `target_machine_handoff_bundle.zip.sha256`

## Operator Steps

1. Copy or extract this bundle onto the target training machine.
2. Run `bash zero_touch_start.sh --check-only` first.
3. If the target machine start gate remains green, launch the canonical path immediately.
4. Canonical offline-clean launcher: `TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh`
5. Optional online teacher lane only if intentionally chosen: `HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh`
6. Current repo-side recommended_path is `offline_clean`.

## Transfer Files

- `zero_touch_start.sh` · `2934` bytes · sha256=`e444192201d64b1bc6ba161c31f79f3c6b34a00097a77b7930114ba80080315a`
- `run.sh` · `24440` bytes · sha256=`aae67b6d2ea0bc7bed6290dbf60d6daff4ee25ee896f83cb0aee6115dd4e2897`
- `scripts/final_orchestrator.py` · `26068` bytes · sha256=`b205bd90b092625346a4cadd7f54b1e1ef1a927451c6edb4edab4c6218ae0968`
- `scripts/start_gate.py` · `8345` bytes · sha256=`30c44468ec976d451abcf5b9aad127c2f35bdf797e66340882938d165610903e`
- `scripts/build_train_readiness_contract.py` · `5974` bytes · sha256=`3c1a8b90842612a03f8ce1c66a945d971c72f51909000893f9515bb78927a420`
- `reports/train_readiness_decision.json` · `5207` bytes · sha256=`7cbd3a53076f70c8e4f6aa832ef240376cc54e3190ed912934419bcd114d1e66`
- `reports/train_readiness_decision.md` · `828` bytes · sha256=`b64a32ac5fb3b4fb422f74a9b69d12829d89c87fad09a3eeaf22130e8fa47bd9`
- `reports/start_gate_report.json` · `3792` bytes · sha256=`fbe1b5c7eb33b708a402212caffafca9a0af00f126bec78e6434099177129e35`
- `reports/start_gate_operator_decision.json` · `710` bytes · sha256=`37fd924ed268fbf3a15b0dcef47f08c749e2079431254a6a50b0cef5305630e9`
- `reports/start_gate_operator_decision.md` · `613` bytes · sha256=`e0894ba2ce8f6b9353cde2284d7de8d5f2707588c87b52c433fc322e4bdd6684`
- `reports/repo_external_handoff.md` · `1537` bytes · sha256=`763d53fdb2fb0213774da89efc3aa5f323f5740cef769992db8ce7c80861fed7`
