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

- `zero_touch_start.sh` · `4741` bytes · sha256=`8f5c229522a36b0a724364a4797677687caec1502dbb09401098262646a6b39f`
- `run.sh` · `24395` bytes · sha256=`a377f9e00f60b77f7a16ed41699606da34275e9e812852768a6892b27206dd0f`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `29486` bytes · sha256=`46fab096c6ab48971f9e897774890c441cf0330315dff347a810c519698577a5`
- `scripts/smart_runner.py` · `5291` bytes · sha256=`301dda82b0fcb5557c52169c775e0032ec2f5cde4b45aaa511fa52cc57c43218`
- `scripts/precompute_logits_topk.py` · `36445` bytes · sha256=`2957e6a177dd07439644ba034df4e4890d8895394d589e6b7b25cbdbc6e09319`
- `scripts/precompute_logits_parallel.py` · `19957` bytes · sha256=`a393ecfee171627a188421b8fe951d127eac82e1246f7de46863aaad09661b06`
- `scripts/validate_logit_alignment.py` · `8661` bytes · sha256=`bd4a9b5635fa5acadfe2dd2f492a292b901cee76b4ee41ce092377a8c00f49a1`
- `train/packing.py` · `9355` bytes · sha256=`61489756ca0bb7b27aa8a2de14ce7db7da964c5a3e3422f589386128b5767cf6`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `293850` bytes · sha256=`f79ed367c494176b6f8ae08d8e7967e5ec7ec40648dcf268f9f0132cbdb7ef23`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31889` bytes · sha256=`c253f460864766ecc63abdd941609c9a6a6827df1b11124a39d96e3eac01a2e9`
- `scripts/start_gate.py` · `12033` bytes · sha256=`d47190b6cb6394523eb318041aee5368f19684326c99792b56309f751dcc749c`
- `scripts/build_train_readiness_contract.py` · `6830` bytes · sha256=`a5b9d55acb37672a41bd7a6f21fa960f04c152541f9de523f0d0feaa198e8568`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`8719b86fcc4abca78d25a3cdd76ed4ef61007ad0df716f602b0114fb786c9813`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `3754` bytes · sha256=`df58406152d15297622fe46ca2795203ae07fd468a35841e573e679bc78ff6d0`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`4762d98d61a0bba91b6d48a9bf002d9a2d41ec178d396c21f0e82581c395d8a1`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1709` bytes · sha256=`505f8201bf0a2766b180d2e747bbbc10410c033ba4f4c199bf88ea2fafaa0fe5`
