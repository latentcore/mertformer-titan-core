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
- `run.sh` · `24446` bytes · sha256=`444a0799dcf03d49cb9e9ccde708fcdf0a9a53d625a4dae85de2679c58d679ae`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `28336` bytes · sha256=`9df3c94824754f0ca508f316de454e88684b03b68f81d5a373a9dc932e3028a5`
- `scripts/smart_runner.py` · `5300` bytes · sha256=`0bccfd2cb7df961b45e9caa84d6a118e677310ece8aee7af9daf35dd3671a89c`
- `scripts/precompute_logits_topk.py` · `15870` bytes · sha256=`47f8d4d33ba6b0c2c671d7f2a4954c2ce3979e50ee8d5f2fa40bb25d6ec3f012`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `287066` bytes · sha256=`d7c3d362acfbbd9431e12f2c5b0d2273ac8bab37c080e47b039a19f7b76d1d91`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31252` bytes · sha256=`d7896a6e4f35087fb0c3a0c6b24c2121e4f34aaafce39ec2c4c16930f321493a`
- `scripts/start_gate.py` · `11888` bytes · sha256=`b7bf16d559d2304903935db37ceafac59a094addb4a7ca9873758f9a6d0acc05`
- `scripts/build_train_readiness_contract.py` · `6368` bytes · sha256=`c98b2ee83ab0818f3eead58af78c5b940c972be0b740c9ccaf62edf9eecec0ec`
- `reports/train_readiness_decision.json` · `8223` bytes · sha256=`42d437ac58f526923fa962c531cbbb712d5114d38bbfce9dbb369a41d9b0e99d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `6091` bytes · sha256=`cd32077ed9ee4b15f031481344e9bcc883b7e0ac741c3f3e85ce0be4cb40b0ce`
- `reports/start_gate_operator_decision.json` · `1579` bytes · sha256=`c9f5af0fd4abafa065e1604d9133684d92949493b86bb67bede054de4d475039`
- `reports/start_gate_operator_decision.md` · `1412` bytes · sha256=`c8dfee1ed74bbe0f680680f32f454203211a73d1fd2c59755eddb3553e6219c3`
- `reports/repo_external_handoff.md` · `1698` bytes · sha256=`68e8136f95538b187287040500bb5e358ff844f3d6677b6e444be459f9f58c8b`
