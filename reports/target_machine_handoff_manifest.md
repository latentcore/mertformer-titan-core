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
- `run.sh` · `24395` bytes · sha256=`da8846ef7af6799ec40536d5e3ac86fa021c824cb4908e67d31c25366b22a8b6`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/data_pipeline.py` · `29240` bytes · sha256=`b3fb2c2689bf2ad67a97d4ff53bb863837b0bb954f287a3d139d29a2e130648d`
- `scripts/smart_runner.py` · `5288` bytes · sha256=`5955a7b0dd1772e45d639e1d56e68bb79d04c28a652946d9120abc03b276942a`
- `scripts/precompute_logits_topk.py` · `36349` bytes · sha256=`adca1edd79cc2cc2c49911230d0c80bac30e876acf91d3f5244eaec38593eb95`
- `scripts/precompute_logits_parallel.py` · `19884` bytes · sha256=`1b10b34592e31720b858c53503c257bf96fd03d6469f8889109071fdad324ef4`
- `scripts/validate_logit_alignment.py` · `8594` bytes · sha256=`21b8fc9e8f70227ea63d26645e88aa91e60832fe28e68869a0bf0bebdf09320a`
- `train/packing.py` · `9355` bytes · sha256=`61489756ca0bb7b27aa8a2de14ce7db7da964c5a3e3422f589386128b5767cf6`
- `scripts/kaggle_onefile_closure_build30.py` · `42570` bytes · sha256=`10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff`
- `scripts/kaggle_onecell_t4_build30.py` · `293850` bytes · sha256=`f79ed367c494176b6f8ae08d8e7967e5ec7ec40648dcf268f9f0132cbdb7ef23`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `31889` bytes · sha256=`c253f460864766ecc63abdd941609c9a6a6827df1b11124a39d96e3eac01a2e9`
- `scripts/start_gate.py` · `12033` bytes · sha256=`d47190b6cb6394523eb318041aee5368f19684326c99792b56309f751dcc749c`
- `scripts/build_train_readiness_contract.py` · `6830` bytes · sha256=`a5b9d55acb37672a41bd7a6f21fa960f04c152541f9de523f0d0feaa198e8568`
- `reports/train_readiness_decision.json` · `8224` bytes · sha256=`a3f82fbd1c796cc5aa15d8a91f5e0606da2f6966afc6b6209ba31f7b6456d85d`
- `reports/train_readiness_decision.md` · `1005` bytes · sha256=`6984cb6e576f3db3fdd7ba0726d5aad94b1d0f17d7935afec80a9225df2d7cd7`
- `reports/start_gate_report.json` · `5804` bytes · sha256=`73a2b77bda52230178333ef1a9836128c1406c14724382a45d17cf00e847e6c7`
- `reports/start_gate_operator_decision.json` · `1691` bytes · sha256=`ea7cb69d59ef9f03866c7b619b30b63c60d40fc9a58e5bbdd60f665f2f931020`
- `reports/start_gate_operator_decision.md` · `1515` bytes · sha256=`9b766f60c680fe78b0dd566b8bbbb730585a0b567b9933ab61317f89af558361`
- `reports/repo_external_handoff.md` · `1709` bytes · sha256=`84e627b863ca9030143844953c5ef0237558b6a43988490ba6728f297c6ac9d2`
