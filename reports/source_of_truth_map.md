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
| `reports/code_truth_contract.md` | code-truth gate, maturity labels, and four-column evidence contract | operators | generated | authoritative | Defines when a closure item is truly done instead of only documented. |
| `reports/surface_lifecycle_matrix.md` | frozen, maintained, and living surface policy | operators | generated | authoritative | Keeps frozen rules separate from maintained and living implementation surfaces. |
| `reports/final_master_plan_freeze.md` | repo-side frozen master-plan boundary for the current closure pass | operators | generated | authoritative | Defines which closure items are locked now versus deferred to post-run or phase-2 work. |
| `reports/update_first_policy.md` | update-first repository modification policy | contributors | generated | authoritative | Requires existing working paths to be audited before replacement or duplication. |
| `reports/repo_directory_contract.md` | canonical repo directory and generated-artifact contract | contributors | generated | authoritative | Defines where durable docs, reports, scripts, schemas, and artifacts belong. |
| `reports/automation_boundary_policy.md` | automation-versus-human decision boundary | operators | generated | authoritative | Keeps strategic, legal, security, and claim publication decisions outside blind automation. |
| `reports/change_control_sop.md` | change-control and closure-safe SOP ladder | operators | generated | authoritative | Defines request, implementation, verification, sync, evidence, and rollback expectations. |
| `reports/system_memory_policy.md` | written system-memory policy for long-running work | operators | generated | authoritative | Defines docs, ADRs, manifests, and reports as durable memory instead of chat context alone. |
| `reports/backlog_operating_contract.md` | canonical backlog state and archival operating contract | operators | generated | authoritative | Standardizes todo/in-progress/blocked/completed/archived handling for closure-critical work. |
| `reports/known_limits_v1.md` | known-limits and explicit non-claims summary for the current pass | public | generated | authoritative | Separates measured facts from absent post-run evidence and research-only claims. |
| `reports/support_maintenance_policy.md` | support, maintenance, and update cadence policy | operators | generated | authoritative | Explains frozen vs maintained vs living surfaces and ties support expectations to current scope. |
| `reports/quality_gate_matrix.md` | quality-gate, KPI, and release-readiness matrix | operators | generated | authoritative | Maps closure-critical lanes to minimum commands, reports, and gate expectations. |
| `reports/test_verification_matrix.md` | unit, integration, smoke, and release verification matrix | operators | generated | authoritative | Documents canonical verification depth for the current repo surfaces. |
| `reports/adr_index.md` | active ADR registry and governance index | operators | generated | authoritative | Indexes the current ADR chain that governs source-of-truth, change-control, and delivery decisions. |
| `reports/repo_closure_scorecard.md` | repo-side closure scorecard for the current pass | operators | generated | authoritative | Tracks the repo-side closure count separately from post-run or external evidence work. |
| `reports/repo_closure_scorecard.json` | machine-readable repo-side closure scorecard | automation | generated | authoritative | Machine-readable mirror of the repo-side closure scorecard. |
| `reports/code_truth_delta_audit.md` | human-readable code-truth delta audit | operators | generated | authoritative | Lists maturity, evidence shape, and marker scan hits for critical repo surfaces. |
| `reports/code_truth_delta_audit.json` | machine-readable code-truth delta audit | automation | generated | authoritative | Machine-readable mirror of the code-truth delta audit. |
| `reports/workspace_hygiene_manifest.md` | human-readable quarantine-first workspace hygiene manifest | operators | generated | authoritative | Documents keep vs quarantine vs ignore decisions before any destructive cleanup. |
| `reports/workspace_hygiene_manifest.json` | machine-readable quarantine-first workspace hygiene manifest | automation | generated | authoritative | Machine-readable workspace hygiene manifest for audit-only or apply-quarantine mode. |
| `interfaces/backlog_item_v1.schema.json` | canonical backlog item schema for code-truth closure tracking | automation | manual | authoritative | Defines the lane/risk/maturity/source-of-truth/evidence fields for canonical backlog entries. |
| `interfaces/workspace_hygiene_manifest_v1.schema.json` | canonical workspace hygiene manifest schema | automation | manual | authoritative | Defines quarantine-first workspace hygiene entries and decision states. |
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
| `adr/ADR-0001-source-of-truth-and-claim-boundary.md` | architecture decision record: ADR-0001 Source of Truth and Claim Boundary | operators | manual | authoritative | Decision record; changes require a superseding ADR or explicit governance exception. |
| `adr/ADR-0002-change-control-and-closure-governance.md` | architecture decision record: ADR-0002 Change Control and Closure Governance | operators | manual | authoritative | Decision record; changes require a superseding ADR or explicit governance exception. |
| `adr/ADR-0003-chess-oneclick-delivery-runtime-contract.md` | architecture decision record: ADR-0003 Chess One-Click Delivery Runtime Contract | operators | manual | authoritative | Decision record; changes require a superseding ADR or explicit governance exception. |
