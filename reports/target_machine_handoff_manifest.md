# Target Machine Handoff Manifest

- next_action: `ALLOCATE_TARGET_MACHINE_AND_START`
- train_allowed: `True`
- decision_reason_code: `READY_OFFLINE_CLEAN`
- recommended_path: `offline_clean`
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

- `zero_touch_start.sh` · `888` bytes · sha256=`bbd1741bed1fc714f79faf47d65d14a452a606711259bb225ad654cb83ed1a7e`
- `run.sh` · `24440` bytes · sha256=`aae67b6d2ea0bc7bed6290dbf60d6daff4ee25ee896f83cb0aee6115dd4e2897`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/kaggle_onefile_closure_build30.py` · `42540` bytes · sha256=`5ef3ad7885d0cf540a24a699492c5f35eed337521d2ef007431d2b8c9caa60eb`
- `scripts/kaggle_onecell_t4_build30.py` · `285841` bytes · sha256=`4eb8e3303d0af9db5da9551b3882ead3cdfd6d76c2f22e4f53d240c558b4d1ca`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `24901` bytes · sha256=`bd62e06d4be1f262c5748e1139555e945417577e19e1e872f22fd6c777feb840`
- `scripts/start_gate.py` · `8345` bytes · sha256=`30c44468ec976d451abcf5b9aad127c2f35bdf797e66340882938d165610903e`
- `scripts/build_train_readiness_contract.py` · `5746` bytes · sha256=`0b7a89c5ed289814b9d962713caa901d348ba7dde3dd7f6dfa64794ff77e74bb`
- `reports/train_readiness_decision.json` · `2943` bytes · sha256=`fae486c7dc7efbdf7f86819e5b3eab2ebbfae26e47fdd5a4734ac8a71b3eb282`
- `reports/train_readiness_decision.md` · `553` bytes · sha256=`a442f9f895d0fda9e30b149b22666a1f36db187ce35acca82eb869345dca1e76`
- `reports/start_gate_report.json` · `2725` bytes · sha256=`0d8cce98bb5ebc36e939e2100c807276449df8e7aa6927faf419d375c173c33f`
- `reports/start_gate_operator_decision.json` · `1215` bytes · sha256=`7585a4cd81c4915a68eb83eb62124d35e484a93be141803c563b6f9478973d51`
- `reports/start_gate_operator_decision.md` · `1065` bytes · sha256=`754a9b0096ad49d04d8610655145d6ef5f2448a07bb6f28dd8851d6476e51eae`
- `reports/repo_external_handoff.md` · `1468` bytes · sha256=`a49e73b9cbc95aa881c9b50d463603203d41694cc7e2ef9ceeb530637fe08272`
