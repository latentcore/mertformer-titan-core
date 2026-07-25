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
- `scripts/data_pipeline.py` · `31739` bytes · sha256=`34b7d19ecbc1b551458a36476cc2497ac82a1b1d6d0b3b26f9592966a49bee3b`
- `scripts/smart_runner.py` · `5318` bytes · sha256=`5fadc80a36222075707e2687b5e9c4e6ef7f5a7c10f7d1e0e3614c0d02faa58d`
- `scripts/precompute_logits_topk.py` · `39387` bytes · sha256=`32a17764c6da93054e893fd8f31837b28dd4987495975374fda982d25a2d1afd`
- `scripts/precompute_logits_parallel.py` · `20496` bytes · sha256=`1a926b590ba57ec5f7f65037e520d04d27bad516ab9cc2e5da0d8a6f43286cc8`
- `scripts/validate_logit_alignment.py` · `8936` bytes · sha256=`226d34efc4fb537d4e1ef6eac466c4f43128e20ab2e45489838eee0ece5852c3`
- `train/packing.py` · `9355` bytes · sha256=`61489756ca0bb7b27aa8a2de14ce7db7da964c5a3e3422f589386128b5767cf6`
- `scripts/kaggle_onefile_closure_build30.py` · `43690` bytes · sha256=`a23eedee31b165b73d91cbb313605dff2c79b9cafe0632867f3ce77f4d1f050b`
- `scripts/kaggle_onecell_t4_build30.py` · `293850` bytes · sha256=`f79ed367c494176b6f8ae08d8e7967e5ec7ec40648dcf268f9f0132cbdb7ef23`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31958` bytes · sha256=`6229aae2c7e5f41b02f726e95b57b54e8b45e4ea4f2e7ad8971797d72f5321c1`
- `scripts/start_gate.py` · `12033` bytes · sha256=`d47190b6cb6394523eb318041aee5368f19684326c99792b56309f751dcc749c`
- `scripts/build_train_readiness_contract.py` · `6830` bytes · sha256=`a5b9d55acb37672a41bd7a6f21fa960f04c152541f9de523f0d0feaa198e8568`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`a3f82fbd1c796cc5aa15d8a91f5e0606da2f6966afc6b6209ba31f7b6456d85d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `3378` bytes · sha256=`d7cfe25511ef64d3ea93516c47b21bd644c728c93190512052a020ddeaf89e35`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`01265cb1d8a51fcd7bf916a202783908aa8f4293df2806bc9119d807a5bfee05`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1709` bytes · sha256=`64693db9c6a50e0284604f16e8e5846b516d432cab91a4dbc6185f724e6e842a`
