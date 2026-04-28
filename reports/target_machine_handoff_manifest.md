# Target Machine Handoff Manifest

- next_action: `ALLOCATE_TARGET_MACHINE_AND_START`
- train_allowed: `True`
- decision_reason_code: `READY_REMOTE_BOOTSTRAP`
- recommended_path: `remote_bootstrap`
- bundle_path: `target_machine_handoff_bundle.zip`
- bundle_sha256_path: `target_machine_handoff_bundle.zip.sha256`

## Operator Steps

1. Copy or extract this bundle onto the target training machine.
2. Run `bash zero_touch_start.sh --check-only` first.
3. If the target machine start gate remains green, launch the canonical path immediately.
4. Recommended rented-machine launcher: `HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh`
5. If stage JSONL or logits are absent locally, the rented machine may generate them there after runtime credential injection.
6. Current repo-side recommended_path is `remote_bootstrap`.

## Transfer Files

- `zero_touch_start.sh` · `2934` bytes · sha256=`e444192201d64b1bc6ba161c31f79f3c6b34a00097a77b7930114ba80080315a`
- `run.sh` · `24440` bytes · sha256=`aae67b6d2ea0bc7bed6290dbf60d6daff4ee25ee896f83cb0aee6115dd4e2897`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `28336` bytes · sha256=`9df3c94824754f0ca508f316de454e88684b03b68f81d5a373a9dc932e3028a5`
- `scripts/smart_runner.py` · `5300` bytes · sha256=`0bccfd2cb7df961b45e9caa84d6a118e677310ece8aee7af9daf35dd3671a89c`
- `scripts/precompute_logits_topk.py` · `15870` bytes · sha256=`47f8d4d33ba6b0c2c671d7f2a4954c2ce3979e50ee8d5f2fa40bb25d6ec3f012`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `287066` bytes · sha256=`d7c3d362acfbbd9431e12f2c5b0d2273ac8bab37c080e47b039a19f7b76d1d91`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `26632` bytes · sha256=`2feee952233996998678b96b5c93e808cd2f277db1cb7ae7a8ecaec48cbf9ae4`
- `scripts/start_gate.py` · `8927` bytes · sha256=`0c3133f77cbad53a96a02e20249a94b1c708d8b8ac5a52c387bd3e744db58dfe`
- `scripts/build_train_readiness_contract.py` · `6368` bytes · sha256=`c98b2ee83ab0818f3eead58af78c5b940c972be0b740c9ccaf62edf9eecec0ec`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`a3f82fbd1c796cc5aa15d8a91f5e0606da2f6966afc6b6209ba31f7b6456d85d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `5373` bytes · sha256=`a1bc55ed98dd86ceb84f0d0f1df68eefff955b6251176e485c149be36e66e352`
- `reports/start_gate_operator_decision.json` · `1475` bytes · sha256=`51cf86b3689efdc65e6d2de75c60c444e69154e45160a052d1e7565817e7c1e4`
- `reports/start_gate_operator_decision.md` · `1313` bytes · sha256=`38a096468ddeffdf21a9b3cc327a9d2b2e4b2cfc6115d28b3b521b9f7385c33e`
- `reports/repo_external_handoff.md` · `1474` bytes · sha256=`36286cbbccafa1396399c4cda92ab76cf381b95e87daa90c79f32d19d6bbf77d`
