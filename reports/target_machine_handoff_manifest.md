# Target Machine Handoff Manifest

- next_action: `ALLOCATE_TARGET_MACHINE_AND_START`
- train_allowed: `True`
- decision_reason_code: `READY_OFFLINE_CLEAN`
- recommended_path: `offline_clean`
- bundle_path: `target_machine_handoff_bundle.zip`
- bundle_sha256_path: `target_machine_handoff_bundle.zip.sha256`

## Operator Steps

1. Copy or extract this bundle onto the target training machine.
2. Run `bash zero_touch_start.sh --check-only` first.
3. If the target machine start gate remains green, launch the canonical path immediately.
4. Canonical offline-clean launcher: `TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh`
5. Optional online teacher lane only if intentionally chosen: `HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh`
6. Current repo-side recommended_path is `offline_clean`.

## Transfer Files

- `zero_touch_start.sh` · `888` bytes · sha256=`bbd1741bed1fc714f79faf47d65d14a452a606711259bb225ad654cb83ed1a7e`
- `run.sh` · `24440` bytes · sha256=`aae67b6d2ea0bc7bed6290dbf60d6daff4ee25ee896f83cb0aee6115dd4e2897`
- `launch_mertformer_kaggle_closure.command` · `790` bytes · sha256=`6f3d756beb9bc3f5a4a00921fe71378783e852435921a170b1d625da605ffc0b`
- `scripts/kaggle_onefile_closure_build30.py` · `40564` bytes · sha256=`40a9abd72020ebeb3d51e46f4656d434d8b765cda2d2340e49e2eae8ea4200da`
- `scripts/macos_keepawake.sh` · `1440` bytes · sha256=`462f7d4e50b6231ebbec4a9d296760212eafc30b3b7e9f55897e08ddbff2b6a9`
- `scripts/final_orchestrator.py` · `24901` bytes · sha256=`bd62e06d4be1f262c5748e1139555e945417577e19e1e872f22fd6c777feb840`
- `scripts/start_gate.py` · `8297` bytes · sha256=`729faad7f598becc926f11e58f1f7db592940dd13c1e8743d129bfd504ea4d55`
- `scripts/build_train_readiness_contract.py` · `5746` bytes · sha256=`0b7a89c5ed289814b9d962713caa901d348ba7dde3dd7f6dfa64794ff77e74bb`
- `reports/train_readiness_decision.json` · `2943` bytes · sha256=`fae486c7dc7efbdf7f86819e5b3eab2ebbfae26e47fdd5a4734ac8a71b3eb282`
- `reports/train_readiness_decision.md` · `553` bytes · sha256=`a442f9f895d0fda9e30b149b22666a1f36db187ce35acca82eb869345dca1e76`
- `reports/start_gate_report.json` · `2381` bytes · sha256=`0b7db7d123d027ade18562c48ffb7decbe73bf22dec0d368cd9cbc9fadab5322`
- `reports/start_gate_operator_decision.json` · `1171` bytes · sha256=`3ef029c69e98bc8c4afc4d23576ba7ed554dea6662ef568a0168a9ded2e41469`
- `reports/start_gate_operator_decision.md` · `1024` bytes · sha256=`680ad5a5865ee4168ca83def5330fef2322c35641229864c2b169fe91cc6978b`
- `reports/repo_external_handoff.md` · `1468` bytes · sha256=`4b718a62a74cf6185a2de97aaf9df77d145702d83e42df24607ea726886dc273`
