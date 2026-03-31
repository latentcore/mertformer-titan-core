# Post-Train State Machine

| Order | Step | Kind | Checkpoint Required | Purpose |
| --- | --- | --- | --- | --- |
| 1 | `checkpoint_resolution` | `internal` | `true` | Resolve best/latest checkpoint for post-train tasks. |
| 2 | `benchmarks_internal` | `command` | `true` | Run internal benchmark suite on the resolved checkpoint. |
| 3 | `golden_eval` | `command` | `true` | Run golden sample evaluation on the resolved checkpoint. |
| 4 | `demo_bundle_manifest` | `internal` | `false` | Build demo bundle manifest and summary. |
| 5 | `mobile_export` | `command` | `false` | Refresh ONNX/mobile export artifacts. |
| 6 | `logbook_build` | `command` | `false` | Append run information to the unified logbook. |
| 7 | `readme_sync` | `command` | `false` | Refresh manifest/doc sync and claim alignment. |
| 8 | `closure_governance_pack` | `command` | `false` | Refresh grouped closure and truth reports. |
| 9 | `release_build30` | `command` | `false` | Refresh release zip package and release snapshot. |
| 10 | `artifact_release_zip` | `command` | `false` | Refresh tracked artifact release zip. |
| 11 | `evidence_pack` | `internal` | `false` | Write the current evidence pack summary. |
