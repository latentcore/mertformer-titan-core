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
- `scripts/data_pipeline.py` · `29062` bytes · sha256=`40d4145953330af1a786124efa9ac42d8784ce6fef4b8bed86bb8b69011185ba`
- `scripts/smart_runner.py` · `5300` bytes · sha256=`0bccfd2cb7df961b45e9caa84d6a118e677310ece8aee7af9daf35dd3671a89c`
- `scripts/precompute_logits_topk.py` · `35015` bytes · sha256=`51da8ff0f617ba8d6d35b7d2e81a6b1ce93b3b2fa3b74df84a6cc2e971651687`
- `scripts/precompute_logits_parallel.py` · `19791` bytes · sha256=`b19629dda4ae2a362321f45e27ce9ead7ed760f43bad1d67af3a72295d66f3c2`
- `scripts/validate_logit_alignment.py` · `8572` bytes · sha256=`f10dd62f5a456b4a5619d901fca05b02d0ffbc5d7b7e5d64b1c806776e54d0bc`
- `train/packing.py` · `9338` bytes · sha256=`0a4f3f71b9382fc9176b696d2deb6ee32a130289a9fb5e2868788d366b58a5d7`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `287069` bytes · sha256=`30194a79a64427e06624853f1ca078b08a0e2f50b634db0ff76fd42028b78b1a`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31886` bytes · sha256=`1243b483cdb2d6e0987853bc1e271298df75e0c44c5c01dcfbbb2dcb190232cd`
- `scripts/start_gate.py` · `12012` bytes · sha256=`de4822cc4f7eb161bc1dfaf94a652fba56047302d7895146630406c3826b829e`
- `scripts/build_train_readiness_contract.py` · `6568` bytes · sha256=`d76cac65bdfd84f0468c045e3878d652f6276c1ce7753ee4d94c9d4087b29a15`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`a3f82fbd1c796cc5aa15d8a91f5e0606da2f6966afc6b6209ba31f7b6456d85d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `4131` bytes · sha256=`6b7c4c271d1c85f99d5c394ef0487c4c78f1b5bc1fd4d7b0c53e560eccedaa15`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`e00489e9bac7893a7365d1cb1c95fb0b505b718ed79b3f01e5599683dc9b9f1f`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1698` bytes · sha256=`05f5eef495d672a0e433dbfb642da38e4aca218cb1ac24ad0450c57395eac54d`
