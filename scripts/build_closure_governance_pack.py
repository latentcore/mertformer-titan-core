#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ADR_DIR = ROOT / "adr"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def file_list(paths: list[Path]) -> list[str]:
    return [rel(path) for path in paths if path.exists()]


def inline_paths(paths: list[str]) -> str:
    if not paths:
        return "None"
    return ", ".join(f"`{path}`" for path in paths)


SOURCE_DOCS = [
    {
        "path": "AGENTS.md",
        "role": "project-closure constitution for contributors and coding agents",
        "audience": "contributors",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Highest-precedence closure rules, claim boundaries, and acceptance criteria.",
    },
    {
        "path": "README.md",
        "role": "canonical public EN overview and command index",
        "audience": "public",
        "update_mode": "manual+artifact-sync",
        "authority": "authoritative",
        "notes": "Measured/target/vision language must match current reports.",
    },
    {
        "path": "README_TR.md",
        "role": "canonical public TR overview and command index",
        "audience": "public",
        "update_mode": "manual+artifact-sync",
        "authority": "authoritative",
        "notes": "Must remain aligned with README.md for closure-critical statements.",
    },
    {
        "path": "MISSION.md",
        "role": "product sentence, scope boundary, and no-claim-without-evidence policy",
        "audience": "public",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Defines current pass mission and long-horizon boundary.",
    },
    {
        "path": "MISSION_TR.md",
        "role": "TR counterpart for mission and claim boundary",
        "audience": "public",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Must remain aligned with MISSION.md.",
    },
    {
        "path": "MODEL_CARD.md",
        "role": "canonical model facts and evidence boundary",
        "audience": "public",
        "update_mode": "manual+artifact-sync",
        "authority": "authoritative",
        "notes": "Claims beyond measured artifacts are not allowed.",
    },
    {
        "path": "MODEL_CARD_TR.md",
        "role": "TR counterpart for model facts and evidence boundary",
        "audience": "public",
        "update_mode": "manual+artifact-sync",
        "authority": "authoritative",
        "notes": "Must remain aligned with MODEL_CARD.md.",
    },
    {
        "path": "USE_POLICY.md",
        "role": "product use policy and creative or folklore boundary",
        "audience": "public",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "High-risk and non-evidence uses are restricted here.",
    },
    {
        "path": "USE_POLICY_TR.md",
        "role": "TR counterpart for product use policy",
        "audience": "public",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Must remain aligned with USE_POLICY.md.",
    },
    {
        "path": "SECURITY.md",
        "role": "security posture and primary ship gate language",
        "audience": "public",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "45K readiness remains the primary ship gate for this pass.",
    },
    {
        "path": "SECURITY_TR.md",
        "role": "TR counterpart for security posture",
        "audience": "public",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Must remain aligned with SECURITY.md.",
    },
    {
        "path": "USAGE_GUIDE.md",
        "role": "operator usage guide for canonical commands",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Must point to the same canonical entrypoints as README and TRAINING_PLAN.",
    },
    {
        "path": "TRAINING_PLAN.md",
        "role": "training workflow, sequencing, and operator expectations",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Must stay aligned with USAGE_GUIDE and README for the official run flow.",
    },
    {
        "path": "run.sh",
        "role": "top-level repo command entrypoint",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Front door for tests, readiness checks, and closure entrypoints.",
    },
    {
        "path": "zero_touch_start.sh",
        "role": "canonical 45K zero-touch launcher",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Canonical preflight -> train/resume -> post-train closeout launcher for the 45K path.",
    },
    {
        "path": "scripts/final_orchestrator.py",
        "role": "canonical train-end orchestrator with run lock and mode contract",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Owns the runtime contract, train launch policy, and post-train delegation.",
    },
    {
        "path": "scripts/post_train_autorun.py",
        "role": "canonical post-train state machine",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Owns benchmark/demo/export/readme/evidence post-train sequencing.",
    },
    {
        "path": "scripts/verify_all.sh",
        "role": "canonical verification gate",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Offline-first validation and report refresh chain.",
    },
    {
        "path": "scripts/one_command_full_sop.sh",
        "role": "one-command closure validation flow",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Runs verification, packaging, and closure-side refresh tasks.",
    },
    {
        "path": "scripts/final_one_shot.sh",
        "role": "maximum closeout and release refresh flow",
        "audience": "operators",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Refreshes release-side artifacts after one-command SOP succeeds.",
    },
    {
        "path": "reports/train_readiness_decision.md",
        "role": "human-readable train allowed or not allowed verdict",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Current repo truth for 45K readiness.",
    },
    {
        "path": "reports/train_readiness_decision.json",
        "role": "machine-readable train allowed or not allowed verdict",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Current repo truth for 45K readiness.",
    },
    {
        "path": "reports/training_readiness_manifest.json",
        "role": "readiness manifest snapshot for current working tree",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Must preserve exact blocker reason codes.",
    },
    {
        "path": "reports/master_closure_matrix.md",
        "role": "raw closure matrix from repo plus desktop TXT",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Backlog coverage source, not completion truth by itself.",
    },
    {
        "path": "reports/master_closure_matrix.json",
        "role": "machine-readable raw closure matrix",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Backlog coverage source, not completion truth by itself.",
    },
    {
        "path": "reports/final_backlog_classification.md",
        "role": "current grouped backlog classification with status and timing",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Current grouped truth layer above the raw matrix.",
    },
    {
        "path": "reports/final_backlog_classification.json",
        "role": "machine-readable grouped backlog classification",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Used for status accounting and handoff.",
    },
    {
        "path": "reports/source_of_truth_map.md",
        "role": "current source-of-truth map for the working tree",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines which docs are authoritative versus supporting or historical.",
    },
    {
        "path": "reports/doc_ownership_matrix.md",
        "role": "doc ownership and refresh matrix",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Clarifies audience, update mode, and authority level.",
    },
    {
        "path": "reports/final_truth_constitution.md",
        "role": "claim, evidence, and release-truth constitution",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Closure writing rules and truth boundaries.",
    },
    {
        "path": "reports/code_truth_contract.md",
        "role": "code-truth gate, maturity labels, and four-column evidence contract",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines when a closure item is truly done instead of only documented.",
    },
    {
        "path": "reports/surface_lifecycle_matrix.md",
        "role": "frozen, maintained, and living surface policy",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Keeps frozen rules separate from maintained and living implementation surfaces.",
    },
    {
        "path": "reports/final_master_plan_freeze.md",
        "role": "repo-side frozen master-plan boundary for the current closure pass",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines which closure items are locked now versus deferred to post-run or phase-2 work.",
    },
    {
        "path": "reports/update_first_policy.md",
        "role": "update-first repository modification policy",
        "audience": "contributors",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Requires existing working paths to be audited before replacement or duplication.",
    },
    {
        "path": "reports/repo_directory_contract.md",
        "role": "canonical repo directory and generated-artifact contract",
        "audience": "contributors",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines where durable docs, reports, scripts, schemas, and artifacts belong.",
    },
    {
        "path": "reports/automation_boundary_policy.md",
        "role": "automation-versus-human decision boundary",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Keeps strategic, legal, security, and claim publication decisions outside blind automation.",
    },
    {
        "path": "reports/change_control_sop.md",
        "role": "change-control and closure-safe SOP ladder",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines request, implementation, verification, sync, evidence, and rollback expectations.",
    },
    {
        "path": "reports/system_memory_policy.md",
        "role": "written system-memory policy for long-running work",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines docs, ADRs, manifests, and reports as durable memory instead of chat context alone.",
    },
    {
        "path": "reports/backlog_operating_contract.md",
        "role": "canonical backlog state and archival operating contract",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Standardizes todo/in-progress/blocked/completed/archived handling for closure-critical work.",
    },
    {
        "path": "reports/known_limits_v1.md",
        "role": "known-limits and explicit non-claims summary for the current pass",
        "audience": "public",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Separates measured facts from absent post-run evidence and research-only claims.",
    },
    {
        "path": "reports/support_maintenance_policy.md",
        "role": "support, maintenance, and update cadence policy",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Explains frozen vs maintained vs living surfaces and ties support expectations to current scope.",
    },
    {
        "path": "reports/quality_gate_matrix.md",
        "role": "quality-gate, KPI, and release-readiness matrix",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Maps closure-critical lanes to minimum commands, reports, and gate expectations.",
    },
    {
        "path": "reports/test_verification_matrix.md",
        "role": "unit, integration, smoke, and release verification matrix",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Documents canonical verification depth for the current repo surfaces.",
    },
    {
        "path": "reports/adr_index.md",
        "role": "active ADR registry and governance index",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Indexes the current ADR chain that governs source-of-truth, change-control, and delivery decisions.",
    },
    {
        "path": "reports/repo_closure_scorecard.md",
        "role": "repo-side closure scorecard for the current pass",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Tracks the repo-side closure count separately from post-run or external evidence work.",
    },
    {
        "path": "reports/repo_closure_scorecard.json",
        "role": "machine-readable repo-side closure scorecard",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Machine-readable mirror of the repo-side closure scorecard.",
    },
    {
        "path": "reports/code_truth_delta_audit.md",
        "role": "human-readable code-truth delta audit",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Lists maturity, evidence shape, and marker scan hits for critical repo surfaces.",
    },
    {
        "path": "reports/code_truth_delta_audit.json",
        "role": "machine-readable code-truth delta audit",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Machine-readable mirror of the code-truth delta audit.",
    },
    {
        "path": "reports/workspace_hygiene_manifest.md",
        "role": "human-readable quarantine-first workspace hygiene manifest",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Documents keep vs quarantine vs ignore decisions before any destructive cleanup.",
    },
    {
        "path": "reports/workspace_hygiene_manifest.json",
        "role": "machine-readable quarantine-first workspace hygiene manifest",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Machine-readable workspace hygiene manifest for audit-only or apply-quarantine mode.",
    },
    {
        "path": "interfaces/backlog_item_v1.schema.json",
        "role": "canonical backlog item schema for code-truth closure tracking",
        "audience": "automation",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Defines the lane/risk/maturity/source-of-truth/evidence fields for canonical backlog entries.",
    },
    {
        "path": "interfaces/workspace_hygiene_manifest_v1.schema.json",
        "role": "canonical workspace hygiene manifest schema",
        "audience": "automation",
        "update_mode": "manual",
        "authority": "authoritative",
        "notes": "Defines quarantine-first workspace hygiene entries and decision states.",
    },
    {
        "path": "reports/run_contract.md",
        "role": "canonical runtime contract for the 45K launcher",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Defines mode flags, resume policy, and launch rules for the canonical orchestrator.",
    },
    {
        "path": "reports/expected_artifacts_list.md",
        "role": "expected artifact set for check-only, train, and post-train flows",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Exact artifact expectations for the canonical zero-touch flow.",
    },
    {
        "path": "reports/exit_code_standard.md",
        "role": "exit code contract for the canonical orchestrator",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Maps lock, readiness, training, and post-train failures to stable codes.",
    },
    {
        "path": "reports/post_train_automation_contract.md",
        "role": "canonical post-train automation contract",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Documents mode behavior and failure policy for the post-train state machine.",
    },
    {
        "path": "reports/post_train_state_machine.md",
        "role": "ordered post-train state machine reference",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Ordered sequence for benchmark/demo/export/readme/evidence closeout.",
    },
    {
        "path": "reports/final_truth_matrix.md",
        "role": "current claim-to-evidence truth matrix",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Maps risky or important claims to current evidence.",
    },
    {
        "path": "reports/claim_registry.json",
        "role": "machine-readable claim registry for closure-critical statements",
        "audience": "automation",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Tracks evidence, missing proof, and claim mode.",
    },
    {
        "path": "reports/repo_external_handoff.md",
        "role": "canonical repo-internal handoff summary with optional desktop copy status",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "supporting",
        "notes": "The repo copy is canonical; a Desktop copy is best-effort when available.",
    },
    {
        "path": "reports/final_repo_audit.md",
        "role": "current working-tree repo audit and stale-surface summary",
        "audience": "operators",
        "update_mode": "generated",
        "authority": "authoritative",
        "notes": "Current repo audit generated by the offline closure pack.",
    },
    {
        "path": "reports/go_status_matrix.md",
        "role": "historical pilot and product claim matrix",
        "audience": "operators",
        "update_mode": "historical",
        "authority": "supporting",
        "notes": "Useful context, but current readiness truth comes from current train readiness outputs.",
    },
]


def load_adr_entries() -> list[dict]:
    entries: list[dict] = []
    for path in sorted(ADR_DIR.glob("ADR-*.md")):
        first_heading = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                first_heading = line.removeprefix("# ").strip()
                break
        entries.append(
            {
                "path": rel(path),
                "role": f"architecture decision record: {first_heading}",
                "audience": "operators",
                "update_mode": "manual",
                "authority": "authoritative",
                "notes": "Decision record; changes require a superseding ADR or explicit governance exception.",
            }
        )
    return entries


def current_source_docs() -> list[dict]:
    return [*SOURCE_DOCS, *load_adr_entries()]

ZERO_TOUCH_REQUIRED_PATHS = [
    "zero_touch_start.sh",
    "scripts/final_orchestrator.py",
    "scripts/post_train_autorun.py",
    "reports/run_contract.md",
    "reports/expected_artifacts_list.md",
    "reports/exit_code_standard.md",
    "reports/post_train_automation_contract.md",
    "reports/post_train_state_machine.md",
    "interfaces/run_manifest_v1.schema.json",
]


CLAIMS = [
    {
        "claim_id": "claim.repo_45k_ready_now",
        "claim": "The repo is genuinely main-run-ready right now.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "reports/train_readiness_decision.md",
            "reports/training_readiness_manifest.json",
            "reports/start_gate_report.json",
        ],
        "still_missing": "Repo-side readiness is green on the offline-clean lane. Real trained outputs remain post-run evidence, and the online teacher lane still requires optional gated credentials.",
    },
    {
        "claim_id": "claim.first_serious_validation_run_positioning",
        "claim": "The 45K run is positioned as the first serious architecture validation run, not the final ceiling.",
        "mode": "policy",
        "status": "verified",
        "evidence": [
            "MISSION.md",
            "MODEL_CARD.md",
            "README.md",
            "reports/final_freeze_manifest.md",
        ],
        "still_missing": "This framing still needs to survive future doc edits and the real owned training-run outputs.",
    },
    {
        "claim_id": "claim.no_claim_without_evidence",
        "claim": "No claim without evidence is a hard repository rule.",
        "mode": "policy",
        "status": "verified",
        "evidence": [
            "AGENTS.md",
            "MISSION.md",
            "prompts/system_v1.txt",
            "scripts/check_doc_claim_consistency.py",
        ],
        "still_missing": "Claim registry enforcement is still document and gate based, not a universal CI policy.",
    },
    {
        "claim_id": "claim.training_readiness_reason_codes",
        "claim": "Training readiness produces exact reason-coded blocker outputs.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "scripts/build_train_readiness_contract.py",
            "reports/train_readiness_decision.json",
            "reports/training_readiness_manifest.json",
        ],
        "still_missing": "The repo-side verdict is TRAIN_ALLOWED via offline_clean; the remaining exact reason-coded blocker is the optional online_teacher lane without HF_TOKEN.",
    },
    {
        "claim_id": "claim.canonical_closure_flow_exists",
        "claim": "A canonical closure flow exists for verification and release refresh.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "run.sh",
            "scripts/verify_all.sh",
            "scripts/one_command_full_sop.sh",
            "scripts/final_one_shot.sh",
            "reports/one_command_full_sop_summary.md",
        ],
        "still_missing": "Real trained outputs still require the actual owned training run, but the closure flow itself is implemented.",
    },
    {
        "claim_id": "claim.true_zero_touch_train_end",
        "claim": "A zero-touch train-end orchestrator exists for preflight -> train/resume -> benchmark -> plot -> demo -> README update -> evidence pack.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "zero_touch_start.sh",
            "scripts/final_orchestrator.py",
            "scripts/post_train_autorun.py",
            "reports/run_contract.md",
            "reports/expected_artifacts_list.md",
            "reports/exit_code_standard.md",
            "reports/post_train_automation_contract.md",
            "reports/post_train_state_machine.md",
            "interfaces/run_manifest_v1.schema.json",
        ],
        "still_missing": "The orchestration layer exists now; only the real trained outputs and post-run evidence remain external to the current working tree.",
    },
    {
        "claim_id": "claim.trained_weights_exist",
        "claim": "Trained final weights, best checkpoint, and latest checkpoint exist and are verified.",
        "mode": "measured",
        "status": "external_dependency",
        "evidence": [
            "reports/train_readiness_decision.md",
            "reports/final_backlog_missing_items.md",
        ],
        "still_missing": "Requires the real owned training run plus checkpoint verification artifacts.",
    },
    {
        "claim_id": "claim.benchmark_claims_exist",
        "claim": "Official benchmark outputs exist for the trained checkpoint.",
        "mode": "measured",
        "status": "external_dependency",
        "evidence": [
            "reports/benchmarks/README.md",
            "reports/final_backlog_missing_items.md",
        ],
        "still_missing": "Requires the real owned training run, benchmark execution, and checkpoint-bound manifests.",
    },
    {
        "claim_id": "claim.demo_bundle_exists",
        "claim": "A real trained-model demo bundle exists and is tied to measured artifacts.",
        "mode": "measured",
        "status": "external_dependency",
        "evidence": [
            "reports/final_backlog_missing_items.md",
        ],
        "still_missing": "Requires trained model outputs, demo manifests, and evidence links after the real run.",
    },
    {
        "claim_id": "claim.edge_mobile_measured",
        "claim": "Edge or mobile deployment claims are backed by measured trained-model evidence.",
        "mode": "measured",
        "status": "external_dependency",
        "evidence": [
            "README.md",
            "reports/final_backlog_missing_items.md",
        ],
        "still_missing": "Requires trained checkpoint export validation and device measurements.",
    },
    {
        "claim_id": "claim.param_count_boundary",
        "claim": "The 2.64B design target versus ~3.70B measured runtime total boundary is explicit.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "README.md",
            "MODEL_CARD.md",
            "reports/param_accounting_report.md",
        ],
        "still_missing": "A dedicated parameter accounting report is still missing in this working tree.",
    },
    {
        "claim_id": "claim.repo_external_handoff",
        "claim": "Closure leaves a canonical repo handoff and may refresh an optional Desktop copy.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "reports/repo_external_handoff.md",
        ],
        "still_missing": "Real 45K outputs still remain external to the current working tree.",
    },
    {
        "claim_id": "claim.docs_only_done_forbidden",
        "claim": "Closure-critical work is not done if it only exists as prose without code path, command, verification, and artifact evidence.",
        "mode": "policy",
        "status": "verified",
        "evidence": [
            "AGENTS.md",
            "reports/code_truth_contract.md",
            "scripts/build_code_truth_audit.py",
        ],
        "still_missing": "This rule is now explicit in the repo, but humans still need to honor it during future edits and release decisions.",
    },
    {
        "claim_id": "claim.kernel_maturity_labels_explicit",
        "claim": "Critical kernel and chess proof surfaces carry explicit maturity labels instead of being narrated as uniformly production-depth.",
        "mode": "measured",
        "status": "verified",
        "evidence": [
            "reports/code_truth_delta_audit.md",
            "scripts/build_code_truth_audit.py",
            "reports/surface_lifecycle_matrix.md",
        ],
        "still_missing": "Optimized-production claims still require measured backend-specific evidence rather than label upgrades by prose.",
    },
    {
        "claim_id": "claim.workspace_hygiene_quarantine_first",
        "claim": "Workspace hygiene is governed by a quarantine-first manifest instead of direct destructive cleanup.",
        "mode": "policy",
        "status": "verified",
        "evidence": [
            "reports/workspace_hygiene_manifest.md",
            "reports/workspace_hygiene_manifest.json",
            "interfaces/workspace_hygiene_manifest_v1.schema.json",
            "scripts/build_workspace_hygiene_manifest.py",
        ],
        "still_missing": "Any destructive cleanup still requires a reviewed manifest and explicit operator intent.",
    },
]


BACKLOG_GROUPS = [
    {
        "group_id": "governance_backlog",
        "title": "Canonical backlog classification and governance pack",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["closure_flow", "truth_claim"],
        "evidence": [
            "AGENTS.md",
            "reports/final_backlog_classification.md",
            "reports/final_backlog_classification.json",
            "reports/final_backlog_coverage_diff.md",
            "reports/final_backlog_missing_items.md",
        ],
        "what_is_done": "The working tree now has a canonical grouped backlog view instead of relying only on the raw 2140-item matrix.",
        "still_missing": "This grouped truth layer does not replace the raw matrix or the future checkpoint-bound evidence from the real owned training run.",
    },
    {
        "group_id": "source_of_truth_regime",
        "title": "Source-of-truth map, document ownership, and closure constitution",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["truth_claim", "policy"],
        "evidence": [
            "AGENTS.md",
            "reports/source_of_truth_map.md",
            "reports/doc_ownership_matrix.md",
            "reports/final_truth_constitution.md",
            "reports/canonical_entrypoint.md",
            "reports/entrypoint_deprecation_map.md",
        ],
        "what_is_done": "Current authoritative docs, generated reports, and historical snapshots are separated.",
        "still_missing": "Future release snapshots must continue to honor this authority order.",
    },
    {
        "group_id": "claim_registry_truth_matrix",
        "title": "Claim registry and final truth matrix",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["truth_claim"],
        "evidence": [
            "reports/claim_registry.json",
            "reports/final_truth_matrix.md",
            "scripts/check_doc_claim_consistency.py",
        ],
        "what_is_done": "Closure-critical claims now map to evidence, mode, and missing proof instead of living only as prose.",
        "still_missing": "Real-run claims remain pending until the real owned training run produces their artifacts.",
    },
    {
        "group_id": "code_truth_gate",
        "title": "Code-truth gate, maturity labels, and four-column done rule",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["truth_claim", "closure_flow"],
        "evidence": [
            "reports/code_truth_contract.md",
            "reports/code_truth_delta_audit.md",
            "reports/code_truth_delta_audit.json",
            "scripts/build_code_truth_audit.py",
            "interfaces/backlog_item_v1.schema.json",
        ],
        "what_is_done": "The repo now defines done-ness through code path, canonical command, verification, and artifact evidence, and it labels critical technical surfaces by maturity instead of flattening them into one production story.",
        "still_missing": "Real optimized-production claims still require measured artifacts and backend-specific proof.",
    },
    {
        "group_id": "surface_lifecycle_regime",
        "title": "Frozen, maintained, and living surface policy",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["policy"],
        "evidence": [
            "reports/surface_lifecycle_matrix.md",
            "interfaces/backlog_item_v1.schema.json",
        ],
        "what_is_done": "Frozen governance surfaces are now separated from maintained verification surfaces and living implementation or product surfaces.",
        "still_missing": "Lane-by-lane lifecycle assignment still needs to stay current as the repo evolves.",
    },
    {
        "group_id": "workspace_hygiene_regime",
        "title": "Workspace hygiene manifest and quarantine-first cleanup policy",
        "status": "DONE_NOW",
        "timing_bucket": "required before destructive cleanup",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": [],
        "evidence": [
            "reports/workspace_hygiene_manifest.md",
            "reports/workspace_hygiene_manifest.json",
            "interfaces/workspace_hygiene_manifest_v1.schema.json",
            "scripts/build_workspace_hygiene_manifest.py",
        ],
        "what_is_done": "Workspace hygiene now emits written keep/quarantine/ignore decisions before any destructive action.",
        "still_missing": "Any actual quarantine apply step still requires reviewed operator intent.",
    },
    {
        "group_id": "closure_entrypoints",
        "title": "Canonical closure commands and current entrypoint mapping",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["closure_flow", "handoff"],
        "evidence": [
            "run.sh",
            "scripts/verify_all.sh",
            "scripts/one_command_full_sop.sh",
            "scripts/final_one_shot.sh",
            "reports/final_commands.md",
            "reports/canonical_entrypoint.md",
            "reports/entrypoint_deprecation_map.md",
        ],
        "what_is_done": "The current verification and closeout chain is documented as one authoritative command ladder.",
        "still_missing": "Real-run evidence remains separate from the command ladder and will appear only after the actual owned training run.",
    },
    {
        "group_id": "readiness_gate",
        "title": "Exact train-readiness contract and blocker reason codes",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": True,
        "mapped_categories": ["training_readiness"],
        "evidence": [
            "scripts/build_train_readiness_contract.py",
            "scripts/titan_preflight.py",
            "reports/train_readiness_decision.md",
            "reports/train_readiness_decision.json",
            "reports/training_readiness_manifest.json",
        ],
        "what_is_done": "The repo emits a deterministic TRAIN_ALLOWED or NOT_ALLOWED decision with exact reason codes.",
        "still_missing": "Repo-side readiness is green via offline_clean. The remaining exact blocker is the optional online_teacher lane without HF_TOKEN.",
    },
    {
        "group_id": "data_contract",
        "title": "Data pipeline contract, provenance, and tokenizer or teacher policy hardening",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": True,
        "mapped_categories": ["data_contract", "training_readiness"],
        "evidence": [
            "config/config.py",
            "scripts/data_pipeline.py",
            "datasets/hashes.json",
        ],
        "what_is_done": "The code path now carries optional-source handling, token-probe settings, and provenance-aware data controls.",
        "still_missing": "Claim-grade corpus lineage, large-scale provenance, and the real owned training-run consumption journal remain post-run evidence.",
    },
    {
        "group_id": "handoff_and_freeze",
        "title": "Freeze manifests and canonical handoff surfaces",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["handoff"],
        "evidence": [
            "reports/final_freeze_manifest.md",
            "reports/final_freeze_manifest.json",
            "reports/repo_external_handoff.md",
            "reports/closure_risk_register.md",
        ],
        "what_is_done": "Freeze and handoff surfaces exist and refresh with the closure flow.",
        "still_missing": "They still need the real trained outputs to become final release evidence.",
    },
    {
        "group_id": "zero_touch_train_end",
        "title": "Zero-touch train-end orchestration and post-train autorun",
        "status": "DONE_NOW",
        "timing_bucket": "required before a real main run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": [],
        "evidence": [
            "zero_touch_start.sh",
            "scripts/final_orchestrator.py",
            "scripts/post_train_autorun.py",
            "reports/run_contract.md",
            "reports/expected_artifacts_list.md",
            "reports/exit_code_standard.md",
            "reports/post_train_automation_contract.md",
            "reports/post_train_state_machine.md",
            "interfaces/run_manifest_v1.schema.json",
        ],
        "what_is_done": "A real zero-touch train-end launcher, run-lock contract, mode flags, and post-train state machine now exist in the working tree.",
        "still_missing": "The remaining missing items are the external real-run outputs, not the orchestration layer itself.",
    },
    {
        "group_id": "real_run_outputs",
        "title": "Trained outputs, benchmark evidence, demos, and final release artifacts",
        "status": "EXTERNAL_DEPENDENCY",
        "timing_bucket": "required before application readiness",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["external"],
        "evidence": [
            "reports/final_backlog_missing_items.md",
            "reports/train_readiness_decision.md",
        ],
        "what_is_done": "The repo now names these artifacts explicitly as post-run evidence rather than pretending they already exist.",
        "still_missing": "A real owned training run, trained final weights, best and latest checkpoints, benchmark summaries, demo bundle, and checkpoint-bound evidence pack are still missing.",
    },
    {
        "group_id": "phase2_compute_and_scaleup",
        "title": "XLA or TPU, cloud expansion, and long-horizon scale-up work",
        "status": "PHASE2",
        "timing_bucket": "not required for application readiness",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["phase2"],
        "evidence": [
            "reports/phase2_carryover.md",
        ],
        "what_is_done": "The raw backlog already distinguishes deferred scale-up work from the closure-critical path.",
        "still_missing": "Actual implementation and proof remain future work.",
    },
    {
        "group_id": "commercial_legal_gtm_band",
        "title": "Company, legal, GTM, investor, and operational mega-band",
        "status": "PHASE2",
        "timing_bucket": "not required for application readiness",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": [],
        "evidence": [
            "reports/final_backlog_missing_items.md",
        ],
        "what_is_done": "This band is now explicitly called out instead of being silently mixed into technical closure truth.",
        "still_missing": "Customer, legal, procurement, pricing, SLA, GTM, and company-governance execution packages remain open.",
    },
]


MANUAL_OPEN_ITEMS = {
    "required before a real main run": [
        "target-machine training hardware allocation and transfer",
        "optional online_teacher:MISSING_HF_TOKEN if the gated teacher lane is intentionally chosen",
    ],
    "required before application readiness": [
        "real owned training run completion",
        "trained final weights",
        "best checkpoint proof",
        "latest checkpoint proof",
        "benchmark summary tied to the trained checkpoint",
        "demo bundle tied to the trained checkpoint",
        "checkpoint-bound evidence pack",
    ],
    "strong plus, not hard blocker for application readiness": [
        "trained-model export and edge or mobile measurements",
        "full release bundle polish beyond the core evidence pack",
    ],
    "not required for application readiness": [
        "XLA or TPU smoke and scale-up lanes",
        "cloud expansion and rented-machine industrialization beyond the main closure path",
        "company or legal or GTM or investor or operations mega-band",
    ],
}


def zero_touch_ready() -> bool:
    return all((ROOT / path).exists() for path in ZERO_TOUCH_REQUIRED_PATHS)


def repo_side_train_allowed(readiness: dict) -> bool:
    return readiness.get("final_status") == "TRAIN_ALLOWED"


def current_claims(readiness: dict) -> list[dict]:
    ready = zero_touch_ready()
    train_allowed = repo_side_train_allowed(readiness)
    blockers = list(readiness.get("blockers", []))
    claims = [dict(claim) for claim in CLAIMS]
    for claim in claims:
        if claim["claim_id"] == "claim.repo_45k_ready_now":
            claim["status"] = "verified" if train_allowed else "blocked"
            if train_allowed:
                claim["still_missing"] = "Repo-side readiness is green on the offline-clean lane. Real trained outputs remain post-run evidence, and the online teacher lane still requires optional gated credentials."
            else:
                claim["still_missing"] = f"Exact blockers remain: {', '.join(blockers) or 'unknown'}."
        if claim["claim_id"] == "claim.training_readiness_reason_codes":
            if train_allowed:
                claim["still_missing"] = "The repo-side verdict is TRAIN_ALLOWED via offline_clean; the remaining exact reason-coded blocker is the optional online_teacher lane without HF_TOKEN."
            else:
                claim["still_missing"] = "Exact blockers must still be cleared before TRAIN_ALLOWED can be granted."
        if claim["claim_id"] == "claim.canonical_closure_flow_exists" and ready:
            claim["evidence"] = [
                "zero_touch_start.sh",
                "scripts/final_orchestrator.py",
                "scripts/post_train_autorun.py",
                "scripts/verify_all.sh",
                "scripts/one_command_full_sop.sh",
                "scripts/final_one_shot.sh",
                "reports/run_contract.md",
                "reports/post_train_automation_contract.md",
            ]
            claim["still_missing"] = "Real trained outputs still require the actual owned training run, but the canonical closure flow now includes the zero-touch launcher and post-train state machine."
        if claim["claim_id"] == "claim.true_zero_touch_train_end":
            if ready:
                claim["status"] = "verified"
                claim["evidence"] = [
                    "zero_touch_start.sh",
                    "scripts/final_orchestrator.py",
                    "scripts/post_train_autorun.py",
                    "reports/run_contract.md",
                    "reports/expected_artifacts_list.md",
                    "reports/exit_code_standard.md",
                    "reports/post_train_automation_contract.md",
                    "reports/post_train_state_machine.md",
                    "interfaces/run_manifest_v1.schema.json",
                ]
                claim["still_missing"] = "The orchestrator exists now; only the trained outputs and post-run evidence remain external to the current working tree."
            else:
                claim["evidence"] = [
                    "reports/final_commands.md",
                    "reports/canonical_entrypoint.md",
                ]
    return claims


def current_backlog_groups(readiness: dict) -> list[dict]:
    ready = zero_touch_ready()
    train_allowed = repo_side_train_allowed(readiness)
    groups = [dict(group) for group in BACKLOG_GROUPS]
    for group in groups:
        if group["group_id"] == "closure_entrypoints" and ready:
            group["evidence"] = [
                "zero_touch_start.sh",
                "scripts/final_orchestrator.py",
                "scripts/post_train_autorun.py",
                "run.sh",
                "scripts/verify_all.sh",
                "scripts/one_command_full_sop.sh",
                "scripts/final_one_shot.sh",
                "reports/final_commands.md",
                "reports/canonical_entrypoint.md",
                "reports/entrypoint_deprecation_map.md",
            ]
            group["what_is_done"] = "The repo now has a canonical zero-touch main-run launcher plus the existing verification and release ladders."
            group["still_missing"] = "Real-run evidence remains separate from the command ladder and will appear only after the actual owned training run."
        if group["group_id"] == "readiness_gate" and train_allowed:
            group["blocks_45k_readiness"] = False
            group["still_missing"] = "Repo-side readiness is green via offline_clean. The remaining exact blocker is the optional online_teacher lane without HF_TOKEN."
        if group["group_id"] == "data_contract" and train_allowed:
            group["blocks_45k_readiness"] = False
            group["what_is_done"] = "The stage1..stage5 JSONL files exist, the offline tokenizer cache is accepted, and the offline-clean data path is green."
            group["still_missing"] = "Claim-grade corpus lineage, large-scale provenance, and the real owned training-run consumption journal remain post-run evidence."
        if group["group_id"] == "zero_touch_train_end" and ready:
            group["status"] = "DONE_NOW"
            group["blocks_45k_readiness"] = False
            group["evidence"] = [
                "zero_touch_start.sh",
                "scripts/final_orchestrator.py",
                "scripts/post_train_autorun.py",
                "reports/run_contract.md",
                "reports/expected_artifacts_list.md",
                "reports/exit_code_standard.md",
                "reports/post_train_automation_contract.md",
                "reports/post_train_state_machine.md",
                "interfaces/run_manifest_v1.schema.json",
            ]
            group["what_is_done"] = "A real zero-touch train-end launcher, run-lock contract, mode flags, and post-train state machine now exist in the working tree."
            group["still_missing"] = "The remaining missing items are the external real-run outputs, not the orchestration layer itself."
    return groups


def current_manual_open_items(readiness: dict) -> dict[str, list[str]]:
    ready = zero_touch_ready()
    blockers = list(readiness.get("blockers", []))
    payload = {bucket: list(items) for bucket, items in MANUAL_OPEN_ITEMS.items()}
    if ready:
        completed_items = {
            "zero-touch start launcher or equivalent canonical train-end orchestrator",
            "post-train autorun state machine with success or fail or partial-run branching",
            "explicit run lock, exit code, and failure-policy contract for the real train path",
        }
        payload["required before a real main run"] = [
            item for item in payload["required before a real main run"] if item not in completed_items
        ]
    if "online_teacher:MISSING_HF_TOKEN" not in blockers:
        payload["required before a real main run"] = [
            item for item in payload["required before a real main run"] if "online_teacher:MISSING_HF_TOKEN" not in item
        ]
    return payload


def build_source_of_truth_map() -> str:
    lines = [
        "# Source-of-Truth Map",
        "",
        "Current authority map for the working tree. Historical reports remain useful context, but they do not override the current closure constitution or the current readiness verdict.",
        "",
        "| Path | Role | Audience | Update Mode | Authority | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in current_source_docs():
        lines.append(
            f"| `{entry['path']}` | {entry['role']} | {entry['audience']} | {entry['update_mode']} | {entry['authority']} | {entry['notes']} |"
        )
    return "\n".join(lines)


def build_doc_ownership_matrix() -> str:
    lines = [
        "# Document Ownership Matrix",
        "",
        "Operational ownership map for closure-critical surfaces.",
        "",
        "| Path | Owner Role | Refresh Trigger | Authority |",
        "| --- | --- | --- | --- |",
    ]
    owner_rules = {
        "manual": ("repo maintainer", "when the policy, command ladder, or product wording changes"),
        "manual+artifact-sync": ("repo maintainer + artifact refresh flow", "after report or benchmark or release artifact changes"),
        "generated": ("automation", "every verify or final refresh run"),
        "historical": ("archive", "never overwrite without an explicit archival update"),
    }
    for entry in current_source_docs():
        owner, trigger = owner_rules.get(entry["update_mode"], ("repo maintainer", "manual"))
        lines.append(
            f"| `{entry['path']}` | {owner} | {trigger} | {entry['authority']} |"
        )
    return "\n".join(lines)


def build_truth_constitution(readiness: dict) -> str:
    ready = zero_touch_ready()
    runtime_gate = "- `bash zero_touch_start.sh --check-only`" if ready else "- `bash zero_touch_start.sh --check-only` (pending implementation)"
    train_allowed = repo_side_train_allowed(readiness)
    blockers = list(readiness.get("blockers", []))
    if train_allowed:
        readiness_rule = "The current repo-side readiness verdict is `TRAIN_ALLOWED` via the offline-clean lane. The remaining exact blocker is the optional `online_teacher:MISSING_HF_TOKEN` lane when gated teacher access is intentionally requested."
    else:
        blocker_lines = "\n".join(f"- `{blocker}`" for blocker in blockers or ["unknown"])
        readiness_rule = f"The repo is not genuinely main-run-ready while either of the current blockers remains active:\n{blocker_lines}"
    return dedent(
        """
        # Final Truth Constitution

        ## Current Pass Objective
        - Close the repository for the real owned training run and application-facing evidence path.
        - Prefer verifiable outputs over speculative redesign.
        - Keep the repo honest about what is implemented now versus what only becomes true after the real run.

        ## Claim Modes
        - `measured`: current artifact-backed fact.
        - `target`: planned or estimated behavior, not yet verified.
        - `vision`: long-range direction outside current evidence scope.
        - `policy`: repository rule or restriction.

        ## Status Modes
        - `DONE_NOW`: implemented now with exact repo evidence.
        - `PREPARED_FOR_POSTRUN`: infrastructure exists, but final proof appears only after the real owned training run.
        - `PHASE2`: explicitly deferred and not required for application readiness.
        - `OUT_OF_SCOPE`: not part of the current closure mandate.
        - `EXTERNAL_DEPENDENCY`: blocked by external data, credentials, compute, or the real run itself.

        ## Hard Rules
        - No claim without evidence.
        - Do not say `main-run-ready` unless the current readiness report says `TRAIN_ALLOWED`.
        - A closure-critical item is only `done` when code path, canonical command, verification, and artifact/report evidence all exist together.
        - Docs-only closure is forbidden.
        - Do not convert scaffolds, placeholders, historical snapshots, or plans into completed work.
        - Do not use historical audit files as current truth unless the current source-of-truth files explicitly point back to them.
        - Keep measured vs target vs vision language explicit in README, model card, policy files, and prompts.

        ## Code-Truth Maturity Labels
        - `reference_safe`: correctness-first reference or scaffold path that is safe for parity/debug use, not for production-depth speed claims.
        - `tested_fallback`: deterministic or bounded implementation with test coverage, but not a release-grade performance claim surface.
        - `optimized_production`: measured and release-grade optimized path backed by claim-grade evidence.

        ## Surface Lifecycle Classes
        - `frozen`: rules, schemas, naming, source-of-truth order, and release-truth constraints.
        - `maintained`: verification gates, manifests, and reproducibility or handoff surfaces that must stay current but not churn without reason.
        - `living`: training, benchmark, kernel, export, product, chess, packaging, security, legal, and pilot implementation surfaces.

        ## Research-Lane Rule
        - `3000+ Elo`, `20 ms/move`, `10000x speedup`, AGI/ASI language, and long-context moonshots remain research lanes unless measured evidence explicitly upgrades them.

        ## Release-Truth Gates
        - `bash scripts/verify_all.sh`
        - `bash scripts/one_command_full_sop.sh`
        - `bash scripts/final_one_shot.sh`
        - `python3 scripts/build_train_readiness_contract.py --allow-not-ready`
        - `python3 scripts/build_closure_governance_pack.py`
        {runtime_gate}

        ## Main-Run Readiness Rule
        {readiness_rule}

        ## Post-Run Rule
        Trained weights, checkpoints, benchmark summaries, demo bundle, checkpoint-bound evidence pack, and measured deployment outputs are not current facts until the real owned training run produces them.
        """
    ).format(runtime_gate=runtime_gate, readiness_rule=readiness_rule).strip()


def build_code_truth_contract() -> str:
    return dedent(
        """
        # Code-Truth Contract

        ## Purpose
        - Keep closure truth anchored to executable repo reality instead of prose alone.
        - Prevent scaffold, fallback, or experimental paths from being narrated as uniformly production-depth.

        ## Done Rule
        A closure-critical item is only done when all four columns are present:
        - `code_path`
        - `canonical_command`
        - `verification`
        - `artifact`

        ## Maturity Labels
        - `reference_safe`: safe correctness or parity reference; not a production speed claim.
        - `tested_fallback`: tested bounded implementation or deterministic fallback; safe to rely on for correctness, not enough for release-grade performance claims.
        - `optimized_production`: measured and release-grade optimized surface with claim-grade evidence.

        ## Practical Rules
        - Docs-only closure is forbidden.
        - `scaffold`, `fallback`, `placeholder`, or `experimental` surfaces must remain explicitly labeled.
        - README, model card, release truth, and backlog wording must map back to code-truth evidence.
        - The code-truth audit is a guardrail, not a substitute for the real owned training run or the real chess product benchmarks.
        """
    ).strip()


def build_surface_lifecycle_matrix() -> str:
    lines = [
        "# Surface Lifecycle Matrix",
        "",
        "Canonical lifecycle classes for the current closure pass.",
        "",
        "| Surface Family | Lifecycle Class | Change Barrier | Notes |",
        "| --- | --- | --- | --- |",
        "| `AGENTS.md`, source-of-truth maps, truth constitution, schemas | `frozen` | high | Frozen surfaces define governance, schemas, naming, and release-truth constraints. |",
        "| Verification gates, manifests, handoff packs, readiness contracts | `maintained` | medium | These must stay current and reproducible, but should not churn without measured reason. |",
        "| Training, benchmark, kernel, chess, export, packaging, product, security, legal, pilot lanes | `living` | controlled | These surfaces are expected to change as implementation and measured evidence evolve. |",
        "| Workspace hygiene reports and quarantine manifests | `maintained` | medium | Hygiene policy is stable, but item-level decisions must refresh as the workspace changes. |",
        "| Research moonshots (`3000+ Elo`, `20 ms/move`, `10000x speedup`, AGI/ASI) | `living` | high external proof bar | Research lanes stay outside V1 release truth until independently measured. |",
    ]
    return "\n".join(lines)


def build_master_plan_freeze(readiness: dict) -> str:
    final_status = readiness.get("final_status", "UNKNOWN")
    recommended_path = readiness.get("recommended_path") or "none"
    blockers = readiness.get("blockers", [])
    blocker_text = ", ".join(blockers) if blockers else "none"
    return dedent(
        f"""
        # Final Master Plan Freeze

        ## Purpose
        - Freeze the repo-side closure frame for the current pass.
        - Prevent reopening solved governance surfaces without a real blocker, compliance break, or a measured post-run need.

        ## Current Frozen Scope
        - Source-of-truth order
        - Claim boundary and evidence policy
        - Repo directory contract
        - Frozen/maintained/living lifecycle split
        - Change-control and update-first policy
        - Backlog operating contract
        - Known-limits and support-maintenance framing
        - ADR chain and verification matrix

        ## Current Readiness Snapshot
        - final_status: `{final_status}`
        - recommended_path: `{recommended_path}`
        - blockers: `{blocker_text}`

        ## Freeze Rule
        - Repo-side governance closure is now execution-first.
        - New ideas do not reopen the frozen frame; they move to post-run evidence work, phase-2, or research lanes unless a real blocker appears.
        - The real run, benchmarks, release artifacts, legal sign-off, and external reproduction remain outside this repo-side freeze.
        """
    ).strip()


def build_update_first_policy() -> str:
    return dedent(
        """
        # Update-First Policy

        ## Core Rule
        - Audit what already exists before adding a replacement.
        - If the current path is correct, keep it.
        - If it is incomplete, extend it.
        - If it is broken, repair it.
        - If drift exists, synchronize it.

        ## Required Classification Before Change
        - active path
        - legacy path
        - no-touch surface
        - high-risk surface
        - closure surface

        ## Forbidden Shortcuts
        - blind rewrite without reading the current implementation
        - duplicate entrypoint for the same responsibility
        - key refactor without a measured or maintainability reason
        - replacing a canonical path only because a new one feels cleaner

        ## Enforcement
        - `scripts/verify_all.sh`
        - `scripts/build_code_truth_audit.py`
        - `scripts/build_closure_governance_pack.py`
        """
    ).strip()


def build_repo_directory_contract() -> str:
    return dedent(
        """
        # Repo Directory Contract

        ## Canonical Durable Areas
        - `docs/`: official structural or explanatory documentation
        - `reports/`: generated or curated closure, benchmark, readiness, and truth artifacts
        - `artifacts/`: handoff, package, and delivery bundles
        - `interfaces/`: canonical schemas and machine-readable contracts
        - `scripts/`: real entrypoints and automation helpers
        - `tests/`: verification surfaces
        - `datasets/`: source notes, hashes, and repo-local validation inputs
        - `adr/`: architecture decision records
        - `runbooks/`, `checklists/`, `benchmarks/`, `configs/`, `knowledge/`: reserved canonical homes when used

        ## Generated-Content Rule
        - Generated reports belong under `reports/`.
        - Generated packages belong under `artifacts/` or purpose-built runtime/output roots.
        - Temporary, cache, and debug clutter must not become durable repo truth.

        ## Enforcement
        - `python3 scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json`
        """
    ).strip()


def build_automation_boundary_policy() -> str:
    return dedent(
        """
        # Automation Boundary Policy

        ## Automation Owns
        - verification chains
        - report refresh
        - manifest and hash refresh
        - benchmark harness execution
        - package or handoff bundle assembly
        - scorecard and truth-matrix generation

        ## Human Review Still Owns
        - strategy changes
        - legal interpretation
        - security sign-off
        - product positioning
        - claim publication beyond current measured truth
        - golden release approval
        - freeze-breach approval

        ## Hard Guardrails
        - automation must not silently turn a plan into a verified claim
        - automation must not merge the meaning of proof mode and product mode
        - automation must not hide blockers; it must emit exact reason-coded state
        """
    ).strip()


def build_change_control_sop() -> str:
    return dedent(
        """
        # Change Control SOP

        ## Required Ladder
        1. define scope and affected surfaces
        2. classify risk and freeze impact
        3. inspect current implementation first
        4. implement the smallest change that closes the gap
        5. run targeted tests
        6. run the canonical verification ladder
        7. refresh manifests and generated governance artifacts
        8. refresh docs and truth surfaces
        9. verify no drift remains
        10. capture artifacts and reports
        11. commit with scoped message
        12. push only after the verified tree is clean

        ## Exceptions
        - Hotfixes may shrink scope, but they do not skip verification or truth-sync.
        - Freeze-breach changes require an ADR or an explicit governance note.
        """
    ).strip()


def build_system_memory_policy() -> str:
    return dedent(
        """
        # System Memory Policy

        ## Durable Memory Order
        1. `AGENTS.md`
        2. source-of-truth and truth-constitution reports
        3. ADR chain
        4. runbooks and SOP-aligned operator docs
        5. manifests, hashes, and provenance artifacts
        6. closure reports and scorecards
        7. backlog classification and missing-items reports

        ## Rules
        - Critical state must be written into repo memory, not left in chat context alone.
        - Current state must be recoverable from docs plus manifests plus reports.
        - Resume safety depends on written artifacts, not recollection.
        """
    ).strip()


def build_backlog_operating_contract() -> str:
    return dedent(
        """
        # Backlog Operating Contract

        ## Canonical States
        - `todo`
        - `in_progress`
        - `blocked`
        - `completed`
        - `archived`

        ## Rules
        - Completed work is closed, not deleted.
        - Blocked work keeps the blocker reason.
        - Research-only or post-run work must not masquerade as current closure work.
        - The raw master closure matrix is coverage input; the grouped backlog classification is current closure truth.

        ## Canonical Files
        - `reports/master_closure_matrix.md`
        - `reports/master_closure_matrix.json`
        - `reports/final_backlog_classification.md`
        - `reports/final_backlog_classification.json`
        - `reports/final_backlog_missing_items.md`
        """
    ).strip()


def build_known_limits(readiness: dict) -> str:
    final_status = readiness.get("final_status", "UNKNOWN")
    blockers = ", ".join(readiness.get("blockers", [])) or "none"
    return dedent(
        f"""
        # Known Limits v1

        ## Current Measured Truth
        - Repo-side verification, truth-sync, and governance surfaces are active.
        - Chess onefile delivery, runtime containment, and Stockfish auto-fetch are implemented.
        - Repo-side training readiness is currently `{final_status}` with blockers `{blockers}`.
        - Exact `45K` remains the preferred main-run target, but application readiness is gated by a real owned training run plus checkpoint-bound evidence rather than the exact step count alone.
        - Costly large-scale compute is not a personal-funding requirement; truthful verified evidence is the actual gate.

        ## Not Yet Measured
        - trained final weights from a real owned training run
        - best/latest checkpoint proof from the real owned training run
        - claim-grade benchmark outputs tied to trained checkpoints
        - final checkpoint-bound evidence pack
        - trained-model export or edge/mobile measurement (strong plus, not a hard blocker)

        ## Chess-Specific Limit
        - Internal proxy strength and readiness surfaces exist, but real strength claims still require post-run benchmark evidence.

        ## Research-Lane Limit
        - `3000+ Elo`, `20 ms/move`, `10000x speedup`, and similar moonshots remain non-release research claims unless separately measured.
        """
    ).strip()


def build_support_maintenance_policy() -> str:
    return dedent(
        """
        # Support and Maintenance Policy

        ## Surface Classes
        - `frozen`: change only with explicit governance exception
        - `maintained`: refresh when verification, manifests, or handoff truth changes
        - `living`: implementation surfaces that can evolve under normal change control

        ## Update Cadence
        - verification and truth reports: every verify or final refresh run
        - README and public positioning: when measured truth changes
        - handoff or package surfaces: when delivery/runtime behavior changes
        - ADR chain: when a closure-critical decision changes

        ## Support Rule
        - Repo-side closure support does not equal post-run production support.
        - Commercial SLA expectations remain separate and are referenced through `reports/commercial_handover/sla_kpi_90_180.md`.
        """
    ).strip()


def build_quality_gate_matrix() -> str:
    lines = [
        "# Quality Gate Matrix",
        "",
        "| Lane | Minimum Gate | Evidence Surface |",
        "| --- | --- | --- |",
        "| repo closure | `bash scripts/verify_all.sh` | `reports/release_manifest.json`, `reports/policy_sync_report.json` |",
        "| train readiness | `python3 scripts/build_train_readiness_contract.py --allow-not-ready` | `reports/train_readiness_decision.json` |",
        "| closure governance | `python3 scripts/build_closure_governance_pack.py` | `reports/final_truth_matrix.md`, `reports/repo_closure_scorecard.md` |",
        "| code truth | `python3 scripts/build_code_truth_audit.py` | `reports/code_truth_delta_audit.md` |",
        "| chess onefile readiness | `python3 scripts/build_chess_training_readiness_report.py` | `reports/chess_training_readiness_report.md` |",
        "| windows delivery | `python3 scripts/export_chess_5080_share.py` plus delivery tests | `reports/target_machine_handoff_manifest.md` |",
        "| max closeout | `bash scripts/final_one_shot.sh` | release and handoff refresh artifacts |",
        "",
        "## KPI and SLA Reference",
        "- `reports/kpi_pack_v1.md`",
        "- `reports/commercial_handover/sla_kpi_90_180.md`",
    ]
    return "\n".join(lines)


def build_test_verification_matrix() -> str:
    lines = [
        "# Test and Verification Matrix",
        "",
        "| Depth | Canonical Command | Scope |",
        "| --- | --- | --- |",
        "| unit/integration baseline | `python3 -m pytest -q` | repo-wide Python test surface |",
        "| code-truth audit | `python3 scripts/build_code_truth_audit.py` | maturity labels and four-column done rule |",
        "| closure governance | `python3 scripts/build_closure_governance_pack.py` | source-of-truth, backlog, known-limits, support, ADR, scorecard |",
        "| offline verify ladder | `bash scripts/verify_all.sh` | canonical repo verification and sync refresh |",
        "| one-command SOP | `bash scripts/one_command_full_sop.sh` | closure validation plus packaging/refresh ladder |",
        "| final closeout | `bash scripts/final_one_shot.sh` | maximum release-side refresh and handoff surfaces |",
        "| chess delivery contract | `python3 -m pytest -q tests/test_chess_5080_onefile.py tests/test_export_chess_5080_share.py tests/test_build_chess_5080_windows_delivery.py` | chess onefile and delivery lane |",
        "| governance contract | `python3 -m pytest -q tests/test_build_code_truth_audit.py tests/test_build_workspace_hygiene_manifest.py tests/test_build_closure_governance_pack.py` | closure and policy generation |",
    ]
    return "\n".join(lines)


def build_adr_index() -> str:
    lines = [
        "# ADR Index",
        "",
        "Current active architecture and governance decisions for the closure pass.",
        "",
        "| ADR | Title | Role |",
        "| --- | --- | --- |",
    ]
    adr_paths = sorted(ADR_DIR.glob("ADR-*.md"))
    if not adr_paths:
        lines.append("| none | no active ADR files | add ADRs before freezing new governance decisions |")
        return "\n".join(lines)

    for path in adr_paths:
        title = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line.removeprefix("# ").strip()
                break
        lines.append(
            f"| `{rel(path)}` | {title} | closure-critical decision record |"
        )
    return "\n".join(lines)


REPO_CLOSURE_SCORECARD_ITEMS = [
    {
        "item_id": "observability_provenance_manifest",
        "title": "Observability, provenance, and manifest chain",
        "evidence": ["reports/release_manifest.json", "reports/train_readiness_decision.json", "reports/final_truth_matrix.md"],
    },
    {
        "item_id": "drift_sync_verify",
        "title": "Drift, sync, and verify discipline",
        "evidence": ["scripts/verify_all.sh", "reports/policy_sync_report.json", "reports/project_structure_sync_report.json"],
    },
    {
        "item_id": "closure_reporting",
        "title": "Closure reporting standard",
        "evidence": ["reports/release_closure_note.md", "reports/final_backlog_classification.md", "reports/final_backlog_missing_items.md"],
    },
    {
        "item_id": "cpu_reference_fallback",
        "title": "CPU reference and fallback surface",
        "evidence": ["mertformer_sdk/kernels/cpp/bitnet_cpu.cpp", "tests/test_cpp_kernel_loader.py", "reports/code_truth_delta_audit.md"],
    },
    {
        "item_id": "windows_oneclick_delivery",
        "title": "Windows one-click delivery",
        "evidence": ["scripts/build_chess_5080_windows_delivery.py", "scripts/export_chess_5080_share.py", "tests/test_build_chess_5080_windows_delivery.py"],
    },
    {
        "item_id": "stockfish_runtime_contract",
        "title": "Stockfish auto-fetch and runtime cache",
        "evidence": ["scripts/chess_5080_onefile.py", "tests/test_chess_5080_onefile.py", "reports/chess_training_readiness_report.md"],
    },
    {
        "item_id": "chess_repo_closure",
        "title": "Chess onefile repo-side closure",
        "evidence": ["reports/chess_training_readiness_report.md", "reports/chess_onefile_extension_report.md", "reports/chess_teaching_contract_report.md"],
    },
    {
        "item_id": "claim_safe_boundary",
        "title": "Claim-safe internal versus external truth boundary",
        "evidence": ["reports/final_truth_matrix.md", "reports/code_truth_contract.md", "MODEL_CARD.md"],
    },
    {
        "item_id": "runtime_artifact_containment",
        "title": "Runtime artifact containment and desktop spam reduction",
        "evidence": ["scripts/chess_5080_onefile.py", "scripts/build_chess_5080_windows_delivery.py", "reports/target_machine_handoff_manifest.md"],
    },
    {
        "item_id": "post_change_sync_hygiene",
        "title": "Post-change sync and release hygiene",
        "evidence": ["reports/release_snapshot.md", "reports/final_sync_matrix.md", "reports/one_command_full_sop_summary.md"],
    },
    {
        "item_id": "git_hygiene_remote_sync",
        "title": "Git hygiene and remote sync discipline",
        "evidence": ["AGENTS.md", "README.md", "reports/release_snapshot.md"],
    },
    {
        "item_id": "master_plan_freeze",
        "title": "Frozen master-plan boundary",
        "evidence": ["reports/final_master_plan_freeze.md", "reports/source_of_truth_map.md"],
    },
    {
        "item_id": "update_first_policy",
        "title": "Update-first modification policy",
        "evidence": ["reports/update_first_policy.md", "reports/code_truth_contract.md"],
    },
    {
        "item_id": "repo_directory_contract",
        "title": "Repo directory contract",
        "evidence": ["reports/repo_directory_contract.md", "docs/PROJECT_STRUCTURE.md"],
    },
    {
        "item_id": "surface_lifecycle_regime",
        "title": "Frozen, maintained, and living lifecycle regime",
        "evidence": ["reports/surface_lifecycle_matrix.md", "reports/final_truth_constitution.md"],
    },
    {
        "item_id": "automation_boundary_policy",
        "title": "Automation boundary policy",
        "evidence": ["reports/automation_boundary_policy.md", "reports/change_control_sop.md"],
    },
    {
        "item_id": "change_control_sop",
        "title": "Change-control SOP",
        "evidence": ["reports/change_control_sop.md", "scripts/verify_all.sh"],
    },
    {
        "item_id": "system_memory_policy",
        "title": "Written system-memory policy",
        "evidence": ["reports/system_memory_policy.md", "reports/source_of_truth_map.md"],
    },
    {
        "item_id": "backlog_operating_contract",
        "title": "Backlog operating contract",
        "evidence": ["reports/backlog_operating_contract.md", "reports/final_backlog_classification.json"],
    },
    {
        "item_id": "known_limits_doc",
        "title": "Known limits document",
        "evidence": ["reports/known_limits_v1.md", "reports/final_truth_matrix.md"],
    },
    {
        "item_id": "support_maintenance_policy",
        "title": "Support and maintenance policy",
        "evidence": ["reports/support_maintenance_policy.md", "reports/commercial_handover/sla_kpi_90_180.md"],
    },
    {
        "item_id": "adr_governance_chain",
        "title": "ADR governance chain",
        "evidence": ["reports/adr_index.md", "adr/ADR-0001-source-of-truth-and-claim-boundary.md"],
    },
    {
        "item_id": "quality_gate_matrix",
        "title": "Quality gate matrix",
        "evidence": ["reports/quality_gate_matrix.md", "reports/kpi_pack_v1.md"],
    },
    {
        "item_id": "test_verification_matrix",
        "title": "Test and verification matrix",
        "evidence": ["reports/test_verification_matrix.md", "tests/test_build_closure_governance_pack.py"],
    },
]


def build_repo_closure_scorecard_payload() -> dict:
    items = []
    for entry in REPO_CLOSURE_SCORECARD_ITEMS:
        evidence = [path for path in entry["evidence"] if (ROOT / path).exists()]
        items.append(
            {
                "item_id": entry["item_id"],
                "title": entry["title"],
                "done": len(evidence) == len(entry["evidence"]),
                "evidence": evidence,
            }
        )
    completed_count = sum(1 for item in items if item["done"])
    return {
        "schema": "repo_closure_scorecard_v1",
        "completed_count": completed_count,
        "target_count": len(REPO_CLOSURE_SCORECARD_ITEMS),
        "all_green": completed_count == len(REPO_CLOSURE_SCORECARD_ITEMS),
        "items": items,
    }


def build_repo_closure_scorecard_markdown(payload: dict) -> str:
    lines = [
        "# Repo Closure Scorecard",
        "",
        f"- completed_count: `{payload['completed_count']}`",
        f"- target_count: `{payload['target_count']}`",
        f"- all_green: `{str(payload['all_green']).lower()}`",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in payload["items"]:
        status = "✅" if item["done"] else "❌"
        lines.append(
            f"| {item['title']} | {status} | {inline_paths(item['evidence'])} |"
        )
    return "\n".join(lines)


def build_claim_registry(readiness: dict) -> list[dict]:
    registry = []
    for claim in current_claims(readiness):
        evidence_paths = [path for path in claim["evidence"] if (ROOT / path).exists()]
        registry.append(
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "mode": claim["mode"],
                "status": claim["status"],
                "evidence": evidence_paths,
                "still_missing": claim["still_missing"],
            }
        )
    return registry


def build_truth_matrix(registry: list[dict]) -> str:
    lines = [
        "# Final Truth Matrix",
        "",
        "Closure-critical claims mapped to current evidence.",
        "",
        "| Claim ID | Claim | Mode | Status | Evidence | Still Missing |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for claim in registry:
        lines.append(
            f"| `{claim['claim_id']}` | {claim['claim']} | `{claim['mode']}` | `{claim['status']}` | {inline_paths(claim['evidence'])} | {claim['still_missing']} |"
        )
    return "\n".join(lines)


def build_canonical_entrypoint() -> str:
    ready = zero_touch_ready()
    zero_touch_status = "active" if ready else "missing"
    zero_touch_note = (
        "Canonical preflight -> train/resume -> post-train closeout launcher with run lock."
        if ready
        else "Still required for the full zero-touch 45K path."
    )
    lines = [
        "# Canonical Entrypoint",
        "",
        "Current command ladder for the working tree.",
        "",
        "| Command | Role | Current Status | Notes |",
        "| --- | --- | --- | --- |",
        f"| `bash zero_touch_start.sh --check-only` | canonical start gate and exact readiness verdict | `{zero_touch_status}` | {zero_touch_note} |",
        f"| `bash zero_touch_start.sh` | canonical train-end 45K launcher | `{zero_touch_status}` | Uses the final orchestrator plus post-train state machine. |",
        "| `bash scripts/verify_all.sh` | canonical verification gate | active | Offline-first verification and report refresh. |",
        "| `bash scripts/one_command_full_sop.sh` | one-command closure validation flow | active | Builds verification, packaging, and report artifacts. |",
        "| `bash scripts/final_one_shot.sh` | max closeout and release refresh | active | Runs one-command SOP first, then release-side extras. |",
        "| `python3 scripts/build_train_readiness_contract.py --allow-not-ready` | exact readiness decision | active | Emits current blocker reason codes. |",
    ]
    return "\n".join(lines)


def build_entrypoint_deprecation_map() -> str:
    lines = [
        "# Entrypoint Deprecation Map",
        "",
        "Supporting or legacy entrypoints that should not replace the canonical ladder.",
        "",
        "| Path | Allowed Role | Replace Canonical? | Notes |",
        "| --- | --- | --- | --- |",
        "| `run.sh` | supporting launcher for test/demo/helper flows | no | Keep for `--test`, `--sitl-demo`, `--cleanroom-verify`, and legacy helper flows; canonical 45K launcher is `zero_touch_start.sh`. |",
        "| `scripts/smart_runner.py` | legacy training helper | no | Superseded by `scripts/final_orchestrator.py` for the canonical 45K path. |",
        "| `scripts/operator_mode_gate.py` | supporting preflight or operator checks | no | Useful support command, not the canonical closure front door. |",
        "| `scripts/release_build30.sh` | release helper | no | Called by the one-command closure flow. |",
        "| `scripts/train_smoke.py` | smoke-only validation | no | Not the official 45K launcher. |",
        "| `scripts/train_tpu_turbo.py` | TPU-specific experiment path | no | Future or phase-2 validation lane. |",
        "| `snake_demo.py` | showcase demo | no | Demonstration surface, not an operational entrypoint. |",
    ]
    return "\n".join(lines)


def build_backlog_classification(matrix: dict, readiness: dict) -> tuple[str, dict]:
    items = matrix["items"]
    category_counts = Counter(item["category"] for item in items)
    phase_counts = Counter(item["phase"] for item in items)

    normalized_groups = []
    for group in current_backlog_groups(readiness):
        raw_count = sum(category_counts.get(category, 0) for category in group["mapped_categories"])
        entry = dict(group)
        entry["raw_item_count"] = raw_count
        entry["evidence"] = [path for path in entry["evidence"] if (ROOT / path).exists()]
        normalized_groups.append(entry)

    lines = [
        "# Final Backlog Classification",
        "",
        "Grouped classification layer for the current working tree. This report uses the raw master closure matrix as coverage input, but it does not confuse phase tagging with completion truth.",
        "",
        f"- raw_matrix_items: `{len(items)}`",
        f"- raw_phase_counts: `{json.dumps(phase_counts, sort_keys=True)}`",
        f"- current_readiness_status: `{readiness.get('final_status', 'UNKNOWN')}`",
        f"- current_readiness_blockers: `{', '.join(readiness.get('blockers', [])) or 'none'}`",
        "",
        "| Group | Status | Timing | Raw Coverage | Plan Covered | Blocks Main Run | Evidence | Still Missing |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in normalized_groups:
        lines.append(
            "| {title} | `{status}` | `{timing}` | `{count}` raw items | `{covered}` | `{blocks}` | {evidence} | {missing} |".format(
                title=group["title"],
                status=group["status"],
                timing=group["timing_bucket"],
                count=group["raw_item_count"],
                covered=str(group["plan_covered"]).lower(),
                blocks=str(group["blocks_45k_readiness"]).lower(),
                evidence=inline_paths(group["evidence"]),
                missing=group["still_missing"],
            )
        )

    payload = {
        "schema": "final_backlog_classification_v1",
        "raw_matrix_summary": matrix.get("summary", {}),
        "raw_category_counts": dict(category_counts),
        "raw_phase_counts": dict(phase_counts),
        "current_readiness_status": readiness.get("final_status"),
        "current_readiness_blockers": readiness.get("blockers", []),
        "groups": normalized_groups,
    }
    return "\n".join(lines), payload


def build_coverage_diff(matrix: dict) -> str:
    items = matrix["items"]
    category_counts = Counter(item["category"] for item in items)
    mapping = defaultdict(list)
    readiness = read_json(REPORTS / "training_readiness_manifest.json")
    groups = current_backlog_groups(readiness)
    for group in groups:
        for category in group["mapped_categories"]:
            mapping[category].append(group["group_id"])

    unmapped = [category for category in sorted(category_counts) if category not in mapping]
    manual_only = [group["group_id"] for group in groups if not group["mapped_categories"]]

    lines = [
        "# Final Backlog Coverage Diff",
        "",
        "This report shows how the raw matrix categories map into the grouped backlog classification layer.",
        "",
        "| Raw Category | Raw Items | Group Mapping |",
        "| --- | --- | --- |",
    ]
    for category, count in sorted(category_counts.items()):
        groups = mapping.get(category, [])
        lines.append(
            f"| `{category}` | `{count}` | {', '.join(f'`{group}`' for group in groups) if groups else 'unmapped'} |"
        )

    lines.extend(
        [
            "",
            "## Manual-only Groups",
            *(f"- `{group}`" for group in manual_only),
            "",
            "## Unmapped Raw Categories",
        ]
    )
    if unmapped:
        lines.extend(f"- `{category}`" for category in unmapped)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Notes",
            "- The raw matrix provides coverage and prioritization, not completion truth by itself.",
            "- Manual-only groups capture zero-touch, post-run, and company/legal/GTM bands that need explicit truth handling beyond the raw category split.",
        ]
    )
    return "\n".join(lines)


def build_missing_items(readiness: dict) -> str:
    manual_open_items = current_manual_open_items(readiness)
    lines = [
        "# Final Backlog Missing Items",
        "",
        "Items still open after the current governance checkpoint.",
        "",
    ]
    blockers = readiness.get("blockers", [])
    if blockers:
        lines.extend([
            "## Current Exact Blockers",
            *(f"- `{blocker}`" for blocker in blockers),
            "",
        ])

    for bucket, items in manual_open_items.items():
        lines.append(f"## {bucket}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    matrix = read_json(REPORTS / "master_closure_matrix.json")
    readiness = read_json(REPORTS / "training_readiness_manifest.json")

    registry = build_claim_registry(readiness)

    write_text(REPORTS / "source_of_truth_map.md", build_source_of_truth_map())
    write_text(REPORTS / "doc_ownership_matrix.md", build_doc_ownership_matrix())
    write_text(REPORTS / "final_truth_constitution.md", build_truth_constitution(readiness))
    write_text(REPORTS / "code_truth_contract.md", build_code_truth_contract())
    write_text(REPORTS / "surface_lifecycle_matrix.md", build_surface_lifecycle_matrix())
    write_text(REPORTS / "final_master_plan_freeze.md", build_master_plan_freeze(readiness))
    write_text(REPORTS / "update_first_policy.md", build_update_first_policy())
    write_text(REPORTS / "repo_directory_contract.md", build_repo_directory_contract())
    write_text(REPORTS / "automation_boundary_policy.md", build_automation_boundary_policy())
    write_text(REPORTS / "change_control_sop.md", build_change_control_sop())
    write_text(REPORTS / "system_memory_policy.md", build_system_memory_policy())
    write_text(REPORTS / "backlog_operating_contract.md", build_backlog_operating_contract())
    write_text(REPORTS / "known_limits_v1.md", build_known_limits(readiness))
    write_text(REPORTS / "support_maintenance_policy.md", build_support_maintenance_policy())
    write_text(REPORTS / "quality_gate_matrix.md", build_quality_gate_matrix())
    write_text(REPORTS / "test_verification_matrix.md", build_test_verification_matrix())
    write_text(REPORTS / "adr_index.md", build_adr_index())
    write_text(REPORTS / "canonical_entrypoint.md", build_canonical_entrypoint())
    write_text(REPORTS / "entrypoint_deprecation_map.md", build_entrypoint_deprecation_map())
    write_text(REPORTS / "final_truth_matrix.md", build_truth_matrix(registry))

    backlog_md, backlog_payload = build_backlog_classification(matrix, readiness)
    write_text(REPORTS / "final_backlog_classification.md", backlog_md)
    write_text(REPORTS / "final_backlog_coverage_diff.md", build_coverage_diff(matrix))
    write_text(REPORTS / "final_backlog_missing_items.md", build_missing_items(readiness))

    scorecard_payload = build_repo_closure_scorecard_payload()
    write_text(REPORTS / "repo_closure_scorecard.md", build_repo_closure_scorecard_markdown(scorecard_payload))

    (REPORTS / "claim_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "final_backlog_classification.json").write_text(
        json.dumps(backlog_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (REPORTS / "repo_closure_scorecard.json").write_text(
        json.dumps(scorecard_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("OK: closure governance pack refreshed")
    print(f" - {rel(REPORTS / 'source_of_truth_map.md')}")
    print(f" - {rel(REPORTS / 'doc_ownership_matrix.md')}")
    print(f" - {rel(REPORTS / 'final_truth_constitution.md')}")
    print(f" - {rel(REPORTS / 'code_truth_contract.md')}")
    print(f" - {rel(REPORTS / 'surface_lifecycle_matrix.md')}")
    print(f" - {rel(REPORTS / 'final_master_plan_freeze.md')}")
    print(f" - {rel(REPORTS / 'update_first_policy.md')}")
    print(f" - {rel(REPORTS / 'repo_directory_contract.md')}")
    print(f" - {rel(REPORTS / 'automation_boundary_policy.md')}")
    print(f" - {rel(REPORTS / 'change_control_sop.md')}")
    print(f" - {rel(REPORTS / 'system_memory_policy.md')}")
    print(f" - {rel(REPORTS / 'backlog_operating_contract.md')}")
    print(f" - {rel(REPORTS / 'known_limits_v1.md')}")
    print(f" - {rel(REPORTS / 'support_maintenance_policy.md')}")
    print(f" - {rel(REPORTS / 'quality_gate_matrix.md')}")
    print(f" - {rel(REPORTS / 'test_verification_matrix.md')}")
    print(f" - {rel(REPORTS / 'adr_index.md')}")
    print(f" - {rel(REPORTS / 'repo_closure_scorecard.md')}")
    print(f" - {rel(REPORTS / 'repo_closure_scorecard.json')}")
    print(f" - {rel(REPORTS / 'canonical_entrypoint.md')}")
    print(f" - {rel(REPORTS / 'entrypoint_deprecation_map.md')}")
    print(f" - {rel(REPORTS / 'claim_registry.json')}")
    print(f" - {rel(REPORTS / 'final_truth_matrix.md')}")
    print(f" - {rel(REPORTS / 'final_backlog_classification.md')}")
    print(f" - {rel(REPORTS / 'final_backlog_classification.json')}")
    print(f" - {rel(REPORTS / 'final_backlog_coverage_diff.md')}")
    print(f" - {rel(REPORTS / 'final_backlog_missing_items.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
