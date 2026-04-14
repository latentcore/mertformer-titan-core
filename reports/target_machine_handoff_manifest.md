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

- `zero_touch_start.sh` · `854` bytes · sha256=`964e874d2dfcbc00c2de0ca2321125ccb3afeefdcde7cc994fa36a01b905be4a`
- `run.sh` · `23997` bytes · sha256=`5a66b3f4db9fefb11b76332eb39b39693b230a638f2af4bf4a748cd874f57470`
- `scripts/final_orchestrator.py` · `24901` bytes · sha256=`bd62e06d4be1f262c5748e1139555e945417577e19e1e872f22fd6c777feb840`
- `scripts/start_gate.py` · `8154` bytes · sha256=`bafd1afb61b5f8d02e153cd822ef5b6f3a89f988876fd849df68b803d91c5e55`
- `scripts/build_train_readiness_contract.py` · `5746` bytes · sha256=`0b7a89c5ed289814b9d962713caa901d348ba7dde3dd7f6dfa64794ff77e74bb`
- `reports/train_readiness_decision.json` · `2943` bytes · sha256=`fae486c7dc7efbdf7f86819e5b3eab2ebbfae26e47fdd5a4734ac8a71b3eb282`
- `reports/train_readiness_decision.md` · `553` bytes · sha256=`a442f9f895d0fda9e30b149b22666a1f36db187ce35acca82eb869345dca1e76`
- `reports/start_gate_report.json` · `5894` bytes · sha256=`4d08fd4850f823d40416648c5bccb6e4b0ce93d3f2d2435570a7f6b920f7a077`
- `reports/start_gate_operator_decision.json` · `1040` bytes · sha256=`e9a3ce1cfe1c9956905eebc7cec4b533376bc5eeed99d183c15925f1cb915020`
- `reports/start_gate_operator_decision.md` · `902` bytes · sha256=`c898ab513e027883fcb2cf03a94bd270438334007fff3ccc4d82529d7d624d85`
- `reports/repo_external_handoff.md` · `1468` bytes · sha256=`bf1a31aee3283b6c036aafeab50392ec47054f772897ede316c88db66d18025a`
