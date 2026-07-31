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

- `zero_touch_start.sh` · `4851` bytes · sha256=`e2bc5abdbd463087c640f8ad5b1c517290064aac0ac2a06b629010c9c1d7b0e9`
- `run.sh` · `25096` bytes · sha256=`ab5f9e98e210720990d6e98a3447de0d2f7bbdc6e1755d6209d3f52aeb057302`
- `launch_mertformer_kaggle_closure.command` · `815` bytes · sha256=`8e2d0130849885e1e6c8a184c4ce1fa8d5900154ccc15d5f942610fdbcae18a4`
- `scripts/data_pipeline.py` · `32543` bytes · sha256=`5589cd0691dd859b847ad52b70433b6ac12af0e518ce8cd123e592877c67b992`
- `scripts/smart_runner.py` · `5454` bytes · sha256=`99d0ab8dbe0c580c57769bf4fe18d83c8b651b6fdc682f017233dfffb6391343`
- `scripts/precompute_logits_topk.py` · `43494` bytes · sha256=`4a1821812b69b5c8fcc705f6cd7b239e2669413d645930c47bd2cefb1c1fe483`
- `scripts/precompute_logits_parallel.py` · `21005` bytes · sha256=`cade1dfc9076b4c98264f995769c74e74edad725e4c70f97b68658ac6722fc19`
- `scripts/validate_logit_alignment.py` · `12222` bytes · sha256=`be6774bccac64df11574707cc62033fc85ad9dffd702c5417b059f3b252248e6`
- `train/packing.py` · `11226` bytes · sha256=`a4ef2f7f1ac5012e193a022f6cd6f59d08b11af86846a12c1c830e8fe1490154`
- `scripts/kaggle_onefile_closure_build30.py` · `44837` bytes · sha256=`100e45e47386414fb36a3b9d5ede51aaece2f61c92cdbdafefd1ad59bcd30d88`
- `scripts/kaggle_onecell_t4_build30.py` · `300951` bytes · sha256=`ffbb305a62c26ac39d118c91c90505d544d1fb77c82833b75313d084627ff568`
- `scripts/macos_keepawake.sh` · `1515` bytes · sha256=`18b88f8eeb65ef8f1b0d77864ff3a3aa3b8d24ce78c1be3749367a7eacbac915`
- `scripts/final_orchestrator.py` · `34701` bytes · sha256=`d96c4f050cd01a4ea01d6551720ae4d623c34b77f819938e7cf85f062e3207f4`
- `scripts/start_gate.py` · `12403` bytes · sha256=`f3f52f630a5a7f358770cb333f787dc22ef04e89c13834219b99c649517de6fc`
- `scripts/build_train_readiness_contract.py` · `7069` bytes · sha256=`2f9186aff91b0fc649d342f1ec2885d802a25217009967ec90496fd45e2d1f96`
- `reports/train_readiness_decision.json` · `5313` bytes · sha256=`6e0d59a558a81c3c92af5c3c5ca79cbc88f15b9c37650007a2cf894fcb2820e7`
- `reports/train_readiness_decision.md` · `970` bytes · sha256=`e0319f2c3d83ea0476c9ac96c1d17b173215a6555d6bca5c01c8e50214b4eff1`
- `reports/start_gate_report.json` · `5606` bytes · sha256=`1ab98eda8561ed71817c8024e75574fc6d1b27c1f628b937ffac8585914f0fd9`
- `reports/start_gate_operator_decision.json` · `1698` bytes · sha256=`ead69707a63416690fcb59fe1a80f589dde1590339ef0657138b43bda1092039`
- `reports/start_gate_operator_decision.md` · `1522` bytes · sha256=`ed11173136117192c3737da2dd02960be5076ae5b71f86b28a1d23c830c3b31e`
- `reports/repo_external_handoff.md` · `1753` bytes · sha256=`cf1a4be4c51b2342a1a0d50105a82925962c04bc8c5dbbb30308ba6d2f7b0221`
