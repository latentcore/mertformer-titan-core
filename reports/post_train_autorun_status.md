# Post-Train Autorun Status

- schema: `post_train_autorun_status_v1`
- mode: `plan-only`
- status: `planned`
- generated_utc: `2026-04-21T19:29:19.188336+00:00`
- checkpoint: `none`

## Steps

| Step | Status | Return Code | Notes |
| --- | --- | --- | --- |
| `checkpoint_resolution` | `planned` | `n/a` | Resolve best/latest checkpoint for post-train tasks. |
| `benchmarks_internal` | `planned` | `n/a` | Run internal benchmark suite on the resolved checkpoint. |
| `golden_eval` | `planned` | `n/a` | Run golden sample evaluation on the resolved checkpoint. |
| `demo_bundle_manifest` | `planned` | `n/a` | Build demo bundle manifest and summary. |
| `mobile_export` | `planned` | `n/a` | Refresh ONNX/mobile export artifacts. |
| `logbook_build` | `planned` | `n/a` | Append run information to the unified logbook. |
| `readme_sync` | `planned` | `n/a` | Refresh manifest/doc sync and claim alignment. |
| `closure_governance_pack` | `planned` | `n/a` | Refresh grouped closure and truth reports. |
| `release_build30` | `planned` | `n/a` | Refresh release zip package and release snapshot. |
| `artifact_release_zip` | `planned` | `n/a` | Refresh tracked artifact release zip. |
| `evidence_pack` | `planned` | `n/a` | Write the current evidence pack summary. |
