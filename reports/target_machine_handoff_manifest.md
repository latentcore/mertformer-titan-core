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

- `zero_touch_start.sh` · `3427` bytes · sha256=`e8d214ebca66e87053481b659d56b07d7940aae525e02944c0c57ebb4e1ba747`
- `run.sh` · `24446` bytes · sha256=`444a0799dcf03d49cb9e9ccde708fcdf0a9a53d625a4dae85de2679c58d679ae`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `28336` bytes · sha256=`9df3c94824754f0ca508f316de454e88684b03b68f81d5a373a9dc932e3028a5`
- `scripts/smart_runner.py` · `5300` bytes · sha256=`0bccfd2cb7df961b45e9caa84d6a118e677310ece8aee7af9daf35dd3671a89c`
- `scripts/precompute_logits_topk.py` · `19373` bytes · sha256=`44f443323657aa40092bd6011e7ae0ad9a9e9788907eef711691e45c3572fd54`
- `scripts/validate_logit_alignment.py` · `8572` bytes · sha256=`f10dd62f5a456b4a5619d901fca05b02d0ffbc5d7b7e5d64b1c806776e54d0bc`
- `train/packing.py` · `9338` bytes · sha256=`0a4f3f71b9382fc9176b696d2deb6ee32a130289a9fb5e2868788d366b58a5d7`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `287066` bytes · sha256=`d7c3d362acfbbd9431e12f2c5b0d2273ac8bab37c080e47b039a19f7b76d1d91`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31252` bytes · sha256=`d7896a6e4f35087fb0c3a0c6b24c2121e4f34aaafce39ec2c4c16930f321493a`
- `scripts/start_gate.py` · `11963` bytes · sha256=`9b3a2a0209f3a83078e4491f5b97e124ccc737c5269dabb87c37d82ef8bd8e35`
- `scripts/build_train_readiness_contract.py` · `6368` bytes · sha256=`c98b2ee83ab0818f3eead58af78c5b940c972be0b740c9ccaf62edf9eecec0ec`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`d36e74fa475730a05e31414a91ee77fad72532169a9ea2ab9bd0a78adb7ebb95`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `5160` bytes · sha256=`15507af1fd5792fe012f8f21ecb9fcbc0c4af8992a99a987cc92f279d8d65ad8`
- `reports/start_gate_operator_decision.json` · `1646` bytes · sha256=`da5b2eaca85b3c65aede617b862e321132b50f785551ca1e75723a6dd8d73fbf`
- `reports/start_gate_operator_decision.md` · `1473` bytes · sha256=`fbdb203433f5854a894080585a8d1ea6a2eedcfecadfe7d7138aeaecfaff03c7`
- `reports/repo_external_handoff.md` · `1698` bytes · sha256=`1ed71e00d89ac386f79a4a2a71e2244a6072e44388ee4ba0af282c830b63ae75`
