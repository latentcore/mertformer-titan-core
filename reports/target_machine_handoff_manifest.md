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
- `scripts/data_pipeline.py` · `30548` bytes · sha256=`c3ff2e6e428c59685c968a3f8c1899b21abee91bc8a64bde22d0b33fecf3635a`
- `scripts/smart_runner.py` · `5291` bytes · sha256=`301dda82b0fcb5557c52169c775e0032ec2f5cde4b45aaa511fa52cc57c43218`
- `scripts/precompute_logits_topk.py` · `36445` bytes · sha256=`2957e6a177dd07439644ba034df4e4890d8895394d589e6b7b25cbdbc6e09319`
- `scripts/precompute_logits_parallel.py` · `19957` bytes · sha256=`a393ecfee171627a188421b8fe951d127eac82e1246f7de46863aaad09661b06`
- `scripts/validate_logit_alignment.py` · `8833` bytes · sha256=`b16c223b26a3cfbb91fd2ab2ec30c4502345ad684921b4dd11fc62c9f29ee82e`
- `train/packing.py` · `9355` bytes · sha256=`61489756ca0bb7b27aa8a2de14ce7db7da964c5a3e3422f589386128b5767cf6`
- `scripts/kaggle_onefile_closure_build30.py` · `42585` bytes · sha256=`c64d8d47cf15000a098a6b3aa0fb08b56567214867af15b03ae776a72ee901f5`
- `scripts/kaggle_onecell_t4_build30.py` · `293850` bytes · sha256=`f79ed367c494176b6f8ae08d8e7967e5ec7ec40648dcf268f9f0132cbdb7ef23`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31889` bytes · sha256=`c253f460864766ecc63abdd941609c9a6a6827df1b11124a39d96e3eac01a2e9`
- `scripts/start_gate.py` · `12033` bytes · sha256=`d47190b6cb6394523eb318041aee5368f19684326c99792b56309f751dcc749c`
- `scripts/build_train_readiness_contract.py` · `6830` bytes · sha256=`a5b9d55acb37672a41bd7a6f21fa960f04c152541f9de523f0d0feaa198e8568`
- `reports/train_readiness_decision.json` · `8223` bytes · sha256=`42d437ac58f526923fa962c531cbbb712d5114d38bbfce9dbb369a41d9b0e99d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `5361` bytes · sha256=`b2e683b9d272aa342434bc72263579fb706d6fc13384ddb8ff70acf4818b7071`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`15435288bb7c67612bfdcb1b88f4d2ad4da3de8c4ea09481a82d3302f25fb9df`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1709` bytes · sha256=`3ddb30ed486ddee06d499322a7aab8ca7fea7c7742e5576f7baf346ce72b10f4`
