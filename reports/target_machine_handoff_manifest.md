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

- `zero_touch_start.sh` · `3980` bytes · sha256=`efe830774713eb3c908b94abdcde981770a09022854cf5992234f588702d0ed3`
- `run.sh` · `24446` bytes · sha256=`444a0799dcf03d49cb9e9ccde708fcdf0a9a53d625a4dae85de2679c58d679ae`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `29059` bytes · sha256=`c4f7a2c0a5b7a8a9a1a3bc864bbd1f0c38e427f48f6f5bffe7019e01f89bc21d`
- `scripts/smart_runner.py` · `5300` bytes · sha256=`0bccfd2cb7df961b45e9caa84d6a118e677310ece8aee7af9daf35dd3671a89c`
- `scripts/precompute_logits_topk.py` · `35015` bytes · sha256=`51da8ff0f617ba8d6d35b7d2e81a6b1ce93b3b2fa3b74df84a6cc2e971651687`
- `scripts/precompute_logits_parallel.py` · `19252` bytes · sha256=`fd5e4d7d46924d913d124b84229797804ec0f215726887f955a62f9a231ea8be`
- `scripts/validate_logit_alignment.py` · `8572` bytes · sha256=`f10dd62f5a456b4a5619d901fca05b02d0ffbc5d7b7e5d64b1c806776e54d0bc`
- `train/packing.py` · `9338` bytes · sha256=`0a4f3f71b9382fc9176b696d2deb6ee32a130289a9fb5e2868788d366b58a5d7`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `287066` bytes · sha256=`d7c3d362acfbbd9431e12f2c5b0d2273ac8bab37c080e47b039a19f7b76d1d91`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31886` bytes · sha256=`1243b483cdb2d6e0987853bc1e271298df75e0c44c5c01dcfbbb2dcb190232cd`
- `scripts/start_gate.py` · `12012` bytes · sha256=`de4822cc4f7eb161bc1dfaf94a652fba56047302d7895146630406c3826b829e`
- `scripts/build_train_readiness_contract.py` · `6568` bytes · sha256=`d76cac65bdfd84f0468c045e3878d652f6276c1ce7753ee4d94c9d4087b29a15`
- `reports/train_readiness_decision.json` · `8223` bytes · sha256=`42d437ac58f526923fa962c531cbbb712d5114d38bbfce9dbb369a41d9b0e99d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `4170` bytes · sha256=`83e2f6eab5d932cd06dcf85b76208d127a057764d762f0341628757c7d360a91`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`57efcabd926a9512153090ae43a7a62ff2821186b65f1975cc0e62d44e7027dc`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1698` bytes · sha256=`5f927a5be1201026b58ee2947ab7c6362345f28fc53f5bf78918c91b0a2d185f`
