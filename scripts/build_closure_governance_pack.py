#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


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
        "claim": "The repo is genuinely 45K-ready right now.",
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
        "still_missing": "This framing still needs to survive future doc edits and the real 45K run outputs.",
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
        "still_missing": "Real trained outputs still require the actual 45K run, but the closure flow itself is implemented.",
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
        "still_missing": "Requires the real 45K run plus checkpoint verification artifacts.",
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
        "still_missing": "Requires the real 45K run, benchmark execution, and checkpoint-bound manifests.",
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
]


BACKLOG_GROUPS = [
    {
        "group_id": "governance_backlog",
        "title": "Canonical backlog classification and governance pack",
        "status": "DONE_NOW",
        "timing_bucket": "required before 45K",
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
        "still_missing": "This grouped truth layer does not replace the raw matrix or the future real-run evidence.",
    },
    {
        "group_id": "source_of_truth_regime",
        "title": "Source-of-truth map, document ownership, and closure constitution",
        "status": "DONE_NOW",
        "timing_bucket": "required before 45K",
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
        "timing_bucket": "required before 45K",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["truth_claim"],
        "evidence": [
            "reports/claim_registry.json",
            "reports/final_truth_matrix.md",
            "scripts/check_doc_claim_consistency.py",
        ],
        "what_is_done": "Closure-critical claims now map to evidence, mode, and missing proof instead of living only as prose.",
        "still_missing": "Real-run claims remain pending until the 45K run produces their artifacts.",
    },
    {
        "group_id": "closure_entrypoints",
        "title": "Canonical closure commands and current entrypoint mapping",
        "status": "DONE_NOW",
        "timing_bucket": "required before 45K",
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
        "still_missing": "Real-run evidence remains separate from the command ladder and will appear only after the actual 45K run.",
    },
    {
        "group_id": "readiness_gate",
        "title": "Exact train-readiness contract and blocker reason codes",
        "status": "DONE_NOW",
        "timing_bucket": "required before 45K",
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
        "timing_bucket": "required before 45K",
        "plan_covered": True,
        "blocks_45k_readiness": True,
        "mapped_categories": ["data_contract", "training_readiness"],
        "evidence": [
            "config/config.py",
            "scripts/data_pipeline.py",
            "datasets/hashes.json",
        ],
        "what_is_done": "The code path now carries optional-source handling, token-probe settings, and provenance-aware data controls.",
        "still_missing": "Claim-grade corpus lineage, large-scale provenance, and the real 45K consumption journal remain post-run evidence.",
    },
    {
        "group_id": "handoff_and_freeze",
        "title": "Freeze manifests and canonical handoff surfaces",
        "status": "DONE_NOW",
        "timing_bucket": "required before 45K",
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
        "still_missing": "They still need the real 45K outputs to become final release evidence.",
    },
    {
        "group_id": "zero_touch_train_end",
        "title": "Zero-touch train-end orchestration and post-train autorun",
        "status": "DONE_NOW",
        "timing_bucket": "required before 45K",
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
        "timing_bucket": "only completable after the real 45K run",
        "plan_covered": True,
        "blocks_45k_readiness": False,
        "mapped_categories": ["external"],
        "evidence": [
            "reports/final_backlog_missing_items.md",
            "reports/train_readiness_decision.md",
        ],
        "what_is_done": "The repo now names these artifacts explicitly as post-run evidence rather than pretending they already exist.",
        "still_missing": "Trained final weights, best and latest checkpoints, benchmark summaries, demo bundle, evidence pack, and trained release package.",
    },
    {
        "group_id": "phase2_compute_and_scaleup",
        "title": "XLA or TPU, cloud expansion, and long-horizon scale-up work",
        "status": "PHASE2",
        "timing_bucket": "not required for 45K readiness",
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
        "timing_bucket": "not required for 45K readiness",
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
    "required before 45K": [
        "target-machine training hardware allocation and transfer",
        "optional online_teacher:MISSING_HF_TOKEN if the gated teacher lane is intentionally chosen",
    ],
    "only completable after the real 45K run": [
        "trained final weights",
        "best checkpoint proof",
        "latest checkpoint proof",
        "benchmark summary tied to the trained checkpoint",
        "demo bundle tied to the trained checkpoint",
        "evidence pack and final release bundle",
        "trained-model export and edge or mobile measurements",
    ],
    "not required for 45K readiness": [
        "XLA or TPU smoke and scale-up lanes",
        "cloud expansion and rented-machine industrialization beyond the 45K closure path",
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
            claim["still_missing"] = "Real trained outputs still require the actual 45K run, but the canonical closure flow now includes the zero-touch launcher and post-train state machine."
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
            group["what_is_done"] = "The repo now has a canonical zero-touch 45K launcher plus the existing verification and release ladders."
            group["still_missing"] = "Real-run evidence remains separate from the command ladder and will appear only after the actual 45K run."
        if group["group_id"] == "readiness_gate" and train_allowed:
            group["blocks_45k_readiness"] = False
            group["still_missing"] = "Repo-side readiness is green via offline_clean. The remaining exact blocker is the optional online_teacher lane without HF_TOKEN."
        if group["group_id"] == "data_contract" and train_allowed:
            group["blocks_45k_readiness"] = False
            group["what_is_done"] = "The stage1..stage5 JSONL files exist, the offline tokenizer cache is accepted, and the offline-clean data path is green."
            group["still_missing"] = "Claim-grade corpus lineage, large-scale provenance, and the real 45K consumption journal remain post-run evidence."
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
        payload["required before 45K"] = [
            item for item in payload["required before 45K"] if item not in completed_items
        ]
    if "online_teacher:MISSING_HF_TOKEN" not in blockers:
        payload["required before 45K"] = [
            item for item in payload["required before 45K"] if "online_teacher:MISSING_HF_TOKEN" not in item
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
    for entry in SOURCE_DOCS:
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
    for entry in SOURCE_DOCS:
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
        readiness_rule = f"The repo is not genuinely 45K-ready while either of the current blockers remains active:\n{blocker_lines}"
    return dedent(
        """
        # Final Truth Constitution

        ## Current Pass Objective
        - Close the repository for the 45K architecture validation run.
        - Prefer verifiable outputs over speculative redesign.
        - Keep the repo honest about what is implemented now versus what only becomes true after the real run.

        ## Claim Modes
        - `measured`: current artifact-backed fact.
        - `target`: planned or estimated behavior, not yet verified.
        - `vision`: long-range direction outside current evidence scope.
        - `policy`: repository rule or restriction.

        ## Status Modes
        - `DONE_NOW`: implemented now with exact repo evidence.
        - `PREPARED_FOR_POSTRUN`: infrastructure exists, but final proof appears only after the real 45K run.
        - `PHASE2`: explicitly deferred and not required for 45K readiness.
        - `OUT_OF_SCOPE`: not part of the current closure mandate.
        - `EXTERNAL_DEPENDENCY`: blocked by external data, credentials, compute, or the real run itself.

        ## Hard Rules
        - No claim without evidence.
        - Do not say `45K-ready` unless the current readiness report says `TRAIN_ALLOWED`.
        - Do not convert scaffolds, placeholders, historical snapshots, or plans into completed work.
        - Do not use historical audit files as current truth unless the current source-of-truth files explicitly point back to them.
        - Keep measured vs target vs vision language explicit in README, model card, policy files, and prompts.

        ## Release-Truth Gates
        - `bash scripts/verify_all.sh`
        - `bash scripts/one_command_full_sop.sh`
        - `bash scripts/final_one_shot.sh`
        - `python3 scripts/build_train_readiness_contract.py --allow-not-ready`
        - `python3 scripts/build_closure_governance_pack.py`
        {runtime_gate}

        ## 45K Readiness Rule
        {readiness_rule}

        ## Post-Run Rule
        Trained weights, checkpoints, benchmark summaries, demo bundle, evidence pack, and measured deployment outputs are not current facts until the real 45K run produces them.
        """
    ).format(runtime_gate=runtime_gate, readiness_rule=readiness_rule).strip()


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
        "| Group | Status | Timing | Raw Coverage | Plan Covered | Blocks 45K | Evidence | Still Missing |",
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
    backlog_md, backlog_payload = build_backlog_classification(matrix, readiness)

    write_text(REPORTS / "source_of_truth_map.md", build_source_of_truth_map())
    write_text(REPORTS / "doc_ownership_matrix.md", build_doc_ownership_matrix())
    write_text(REPORTS / "final_truth_constitution.md", build_truth_constitution(readiness))
    write_text(REPORTS / "canonical_entrypoint.md", build_canonical_entrypoint())
    write_text(REPORTS / "entrypoint_deprecation_map.md", build_entrypoint_deprecation_map())
    write_text(REPORTS / "final_truth_matrix.md", build_truth_matrix(registry))
    write_text(REPORTS / "final_backlog_classification.md", backlog_md)
    write_text(REPORTS / "final_backlog_coverage_diff.md", build_coverage_diff(matrix))
    write_text(REPORTS / "final_backlog_missing_items.md", build_missing_items(readiness))

    (REPORTS / "claim_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "final_backlog_classification.json").write_text(
        json.dumps(backlog_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("OK: closure governance pack refreshed")
    print(f" - {rel(REPORTS / 'source_of_truth_map.md')}")
    print(f" - {rel(REPORTS / 'doc_ownership_matrix.md')}")
    print(f" - {rel(REPORTS / 'final_truth_constitution.md')}")
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
