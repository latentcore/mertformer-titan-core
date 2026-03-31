# Source-of-Truth Map

Current authority map for the working tree. Historical reports remain useful context, but they do not override the current closure constitution or the current readiness verdict.

| Path | Role | Audience | Update Mode | Authority | Notes |
| --- | --- | --- | --- | --- | --- |
| `AGENTS.md` | project-closure constitution for contributors and coding agents | contributors | manual | authoritative | Highest-precedence closure rules, claim boundaries, and acceptance criteria. |
| `README.md` | canonical public EN overview and command index | public | manual+artifact-sync | authoritative | Measured/target/vision language must match current reports. |
| `README_TR.md` | canonical public TR overview and command index | public | manual+artifact-sync | authoritative | Must remain aligned with README.md for closure-critical statements. |
| `MISSION.md` | product sentence, scope boundary, and no-claim-without-evidence policy | public | manual | authoritative | Defines current pass mission and long-horizon boundary. |
| `MISSION_TR.md` | TR counterpart for mission and claim boundary | public | manual | authoritative | Must remain aligned with MISSION.md. |
| `MODEL_CARD.md` | canonical model facts and evidence boundary | public | manual+artifact-sync | authoritative | Claims beyond measured artifacts are not allowed. |
| `MODEL_CARD_TR.md` | TR counterpart for model facts and evidence boundary | public | manual+artifact-sync | authoritative | Must remain aligned with MODEL_CARD.md. |
| `USE_POLICY.md` | product use policy and creative or folklore boundary | public | manual | authoritative | High-risk and non-evidence uses are restricted here. |
| `USE_POLICY_TR.md` | TR counterpart for product use policy | public | manual | authoritative | Must remain aligned with USE_POLICY.md. |
| `SECURITY.md` | security posture and primary ship gate language | public | manual | authoritative | 45K readiness remains the primary ship gate for this pass. |
| `SECURITY_TR.md` | TR counterpart for security posture | public | manual | authoritative | Must remain aligned with SECURITY.md. |
| `USAGE_GUIDE.md` | operator usage guide for canonical commands | operators | manual | authoritative | Must point to the same canonical entrypoints as README and TRAINING_PLAN. |
| `TRAINING_PLAN.md` | training workflow, sequencing, and operator expectations | operators | manual | authoritative | Must stay aligned with USAGE_GUIDE and README for the official run flow. |
| `run.sh` | top-level repo command entrypoint | operators | manual | authoritative | Front door for tests, readiness checks, and closure entrypoints. |
| `zero_touch_start.sh` | canonical 45K zero-touch launcher | operators | manual | authoritative | Canonical preflight -> train/resume -> post-train closeout launcher for the 45K path. |
| `scripts/final_orchestrator.py` | canonical train-end orchestrator with run lock and mode contract | operators | manual | authoritative | Owns the runtime contract, train launch policy, and post-train delegation. |
| `scripts/post_train_autorun.py` | canonical post-train state machine | operators | manual | authoritative | Owns benchmark/demo/export/readme/evidence post-train sequencing. |
| `scripts/verify_all.sh` | canonical verification gate | operators | manual | authoritative | Offline-first validation and report refresh chain. |
| `scripts/one_command_full_sop.sh` | one-command closure validation flow | operators | manual | authoritative | Runs verification, packaging, and closure-side refresh tasks. |
| `scripts/final_one_shot.sh` | maximum closeout and release refresh flow | operators | manual | authoritative | Refreshes release-side artifacts after one-command SOP succeeds. |
| `reports/train_readiness_decision.md` | human-readable train allowed or not allowed verdict | operators | generated | authoritative | Current repo truth for 45K readiness. |
| `reports/train_readiness_decision.json` | machine-readable train allowed or not allowed verdict | automation | generated | authoritative | Current repo truth for 45K readiness. |
| `reports/training_readiness_manifest.json` | readiness manifest snapshot for current working tree | automation | generated | authoritative | Must preserve exact blocker reason codes. |
| `reports/master_closure_matrix.md` | raw closure matrix from repo plus desktop TXT | operators | generated | authoritative | Backlog coverage source, not completion truth by itself. |
| `reports/master_closure_matrix.json` | machine-readable raw closure matrix | automation | generated | authoritative | Backlog coverage source, not completion truth by itself. |
| `reports/final_backlog_classification.md` | current grouped backlog classification with status and timing | operators | generated | authoritative | Current grouped truth layer above the raw matrix. |
| `reports/final_backlog_classification.json` | machine-readable grouped backlog classification | automation | generated | authoritative | Used for status accounting and handoff. |
| `reports/source_of_truth_map.md` | current source-of-truth map for the working tree | operators | generated | authoritative | Defines which docs are authoritative versus supporting or historical. |
| `reports/doc_ownership_matrix.md` | doc ownership and refresh matrix | operators | generated | authoritative | Clarifies audience, update mode, and authority level. |
| `reports/final_truth_constitution.md` | claim, evidence, and release-truth constitution | operators | generated | authoritative | Closure writing rules and truth boundaries. |
| `reports/run_contract.md` | canonical runtime contract for the 45K launcher | operators | generated | authoritative | Defines mode flags, resume policy, and launch rules for the canonical orchestrator. |
| `reports/expected_artifacts_list.md` | expected artifact set for check-only, train, and post-train flows | operators | generated | authoritative | Exact artifact expectations for the canonical zero-touch flow. |
| `reports/exit_code_standard.md` | exit code contract for the canonical orchestrator | operators | generated | authoritative | Maps lock, readiness, training, and post-train failures to stable codes. |
| `reports/post_train_automation_contract.md` | canonical post-train automation contract | operators | generated | authoritative | Documents mode behavior and failure policy for the post-train state machine. |
| `reports/post_train_state_machine.md` | ordered post-train state machine reference | operators | generated | authoritative | Ordered sequence for benchmark/demo/export/readme/evidence closeout. |
| `reports/final_truth_matrix.md` | current claim-to-evidence truth matrix | operators | generated | authoritative | Maps risky or important claims to current evidence. |
| `reports/claim_registry.json` | machine-readable claim registry for closure-critical statements | automation | generated | authoritative | Tracks evidence, missing proof, and claim mode. |
| `reports/repo_external_handoff.md` | canonical repo-internal handoff summary with optional desktop copy status | operators | generated | supporting | The repo copy is canonical; a Desktop copy is best-effort when available. |
| `reports/final_repo_audit.md` | current working-tree repo audit and stale-surface summary | operators | generated | authoritative | Current repo audit generated by the offline closure pack. |
| `reports/go_status_matrix.md` | historical pilot and product claim matrix | operators | historical | supporting | Useful context, but current readiness truth comes from current train readiness outputs. |
