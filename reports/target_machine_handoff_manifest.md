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
- `scripts/data_pipeline.py` · `30575` bytes · sha256=`faa59873e87edb609ca5bbffb1cdf481c34561e773f1d70e789757f093020a0a`
- `scripts/smart_runner.py` · `5318` bytes · sha256=`5fadc80a36222075707e2687b5e9c4e6ef7f5a7c10f7d1e0e3614c0d02faa58d`
- `scripts/precompute_logits_topk.py` · `38586` bytes · sha256=`c5b1f66a92e3d205b30c5901badad3d5117ed6e90c4a56f2e29291a9a091f97b`
- `scripts/precompute_logits_parallel.py` · `20496` bytes · sha256=`1a926b590ba57ec5f7f65037e520d04d27bad516ab9cc2e5da0d8a6f43286cc8`
- `scripts/validate_logit_alignment.py` · `8936` bytes · sha256=`226d34efc4fb537d4e1ef6eac466c4f43128e20ab2e45489838eee0ece5852c3`
- `train/packing.py` · `9355` bytes · sha256=`61489756ca0bb7b27aa8a2de14ce7db7da964c5a3e3422f589386128b5767cf6`
- `scripts/kaggle_onefile_closure_build30.py` · `43690` bytes · sha256=`a23eedee31b165b73d91cbb313605dff2c79b9cafe0632867f3ce77f4d1f050b`
- `scripts/kaggle_onecell_t4_build30.py` · `293850` bytes · sha256=`f79ed367c494176b6f8ae08d8e7967e5ec7ec40648dcf268f9f0132cbdb7ef23`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31958` bytes · sha256=`6229aae2c7e5f41b02f726e95b57b54e8b45e4ea4f2e7ad8971797d72f5321c1`
- `scripts/start_gate.py` · `12033` bytes · sha256=`d47190b6cb6394523eb318041aee5368f19684326c99792b56309f751dcc749c`
- `scripts/build_train_readiness_contract.py` · `6830` bytes · sha256=`a5b9d55acb37672a41bd7a6f21fa960f04c152541f9de523f0d0feaa198e8568`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`8719b86fcc4abca78d25a3cdd76ed4ef61007ad0df716f602b0114fb786c9813`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `4861` bytes · sha256=`e43cba9ca151ddbde5bbe0d74aacd933f2cb8dd93de54285e089d0d6f98c8526`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`b5c7f91cfd870a7c6a0030f15d5040ea078cf09e47ea0adb16038373369d9e4b`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1709` bytes · sha256=`b6fcc9849b30d188d4636a1f339849ab37005d98b9c07d449887f7aad27c5a24`
