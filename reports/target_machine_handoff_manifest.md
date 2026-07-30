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
- `run.sh` · `24431` bytes · sha256=`ced539190a246ccd24f2b39e756ce265f11cb131bc7952c62b43b0b79492a5e6`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `31738` bytes · sha256=`18d074412002ab3d336472dfc8ac05f1d70b336c821798f41f0c6dca1f88e8b6`
- `scripts/smart_runner.py` · `5317` bytes · sha256=`3e2789e59939558a3ea6b4151db754a0b4267cc1032d9fd31402bd910a72d3c3`
- `scripts/precompute_logits_topk.py` · `42473` bytes · sha256=`1c278f1cb80a62b959a26ccf8f463ef9b339426ae4c93908b34b60e7fba536a7`
- `scripts/precompute_logits_parallel.py` · `20496` bytes · sha256=`1a926b590ba57ec5f7f65037e520d04d27bad516ab9cc2e5da0d8a6f43286cc8`
- `scripts/validate_logit_alignment.py` · `11929` bytes · sha256=`2ad4bc2a5fe7a071af4465503a780aeba6e8383598253eec73ccd7ab8379ea96`
- `train/packing.py` · `10975` bytes · sha256=`f3d11fb437fad88b5ec38e4fbd3449eb8e420fcabdf83d0c2d749bc0538c4372`
- `scripts/kaggle_onefile_closure_build30.py` · `43690` bytes · sha256=`a23eedee31b165b73d91cbb313605dff2c79b9cafe0632867f3ce77f4d1f050b`
- `scripts/kaggle_onecell_t4_build30.py` · `293850` bytes · sha256=`f79ed367c494176b6f8ae08d8e7967e5ec7ec40648dcf268f9f0132cbdb7ef23`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `33794` bytes · sha256=`9ad7ae7970a0fb9d82b42ba41a830d52c643c6e9702d6da716b416732b690524`
- `scripts/start_gate.py` · `12033` bytes · sha256=`d47190b6cb6394523eb318041aee5368f19684326c99792b56309f751dcc749c`
- `scripts/build_train_readiness_contract.py` · `6830` bytes · sha256=`a5b9d55acb37672a41bd7a6f21fa960f04c152541f9de523f0d0feaa198e8568`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`60d22d56aecc9b202907759562444cca205723f341bed4754110dd9050849fee`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `4038` bytes · sha256=`02949f9e6eb37f068059d1700fe7ccf4dc9ff4dede71bfefabe71e0dece7ff04`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`8e9720964067edb9fd0649601b9ebc7978f0eb3796c451e10b3c0ebd41383014`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1709` bytes · sha256=`a3ed4a20a1718a0509f88e9c888403fa358450533801ab2987fbccd4ffe9c61e`
