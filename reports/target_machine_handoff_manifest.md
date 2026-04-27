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

- `zero_touch_start.sh` · `2900` bytes · sha256=`89f001ac1bcc3fc6b08dc03395b10db40b0a6a45cf021c481c5b4d1317d8d0cb`
- `run.sh` · `24440` bytes · sha256=`aae67b6d2ea0bc7bed6290dbf60d6daff4ee25ee896f83cb0aee6115dd4e2897`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `287066` bytes · sha256=`d7c3d362acfbbd9431e12f2c5b0d2273ac8bab37c080e47b039a19f7b76d1d91`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `24901` bytes · sha256=`bd62e06d4be1f262c5748e1139555e945417577e19e1e872f22fd6c777feb840`
- `scripts/start_gate.py` · `8345` bytes · sha256=`30c44468ec976d451abcf5b9aad127c2f35bdf797e66340882938d165610903e`
- `scripts/build_train_readiness_contract.py` · `5746` bytes · sha256=`0b7a89c5ed289814b9d962713caa901d348ba7dde3dd7f6dfa64794ff77e74bb`
- `reports/train_readiness_decision.json` · `2943` bytes · sha256=`2743b702ed0889d3664fc6252768e36550000ea8182e62454284f7ad4ac4ac51`
- `reports/train_readiness_decision.md` · `553` bytes · sha256=`a442f9f895d0fda9e30b149b22666a1f36db187ce35acca82eb869345dca1e76`
- `reports/start_gate_report.json` · `3688` bytes · sha256=`31a9ca9dcdb89039aa1cfbed26a570c71cb08c6603e2f4eb9e1c41e36882f8e5`
- `reports/start_gate_operator_decision.json` · `1215` bytes · sha256=`8e9dbc0ffac4f64be8d828256d92083483f9d459ce451b5af67479ba4cb75ce0`
- `reports/start_gate_operator_decision.md` · `1065` bytes · sha256=`754a9b0096ad49d04d8610655145d6ef5f2448a07bb6f28dd8851d6476e51eae`
- `reports/repo_external_handoff.md` · `1468` bytes · sha256=`7f8d79198c555964dcb9249c9f3c20d2fafdd878682b21cff49570088b1bfd6f`
