## MertFormer Titan (Build 30 V2)

Offline-first, auditable, mission-focused AI infrastructure for controlled local deployment.
Current maturity: **pilot-ready pre-training baseline** (training/benchmark claims pending).

**Build 30 V2 note:** V2 refactor pass adds dedup pipeline, MoE dispatch parallel path, CfC fast path, and stricter train gates. Claims remain pre-training.

### Quick Reader Links
- English summary: [README_SUMMARY.md](README_SUMMARY.md)
- Turkish summary: [README_SUMMARY_TR.md](README_SUMMARY_TR.md)
- Turkish full doc: [README_TR.md](README_TR.md)
- Mission: [MISSION.md](MISSION.md)
- Use policy: [USE_POLICY.md](USE_POLICY.md)
- Security: [SECURITY.md](SECURITY.md)
- Repo-side closure scorecard: [reports/repo_closure_scorecard.md](reports/repo_closure_scorecard.md)
- Known limits: [reports/known_limits_v1.md](reports/known_limits_v1.md)
- Closure freeze and ADR index: [reports/final_master_plan_freeze.md](reports/final_master_plan_freeze.md), [reports/adr_index.md](reports/adr_index.md)

### Official Positioning
- Product sentence: `Turkey-serving, offline-first, edge-native, locally integrable intelligence infrastructure.`
- 45K positioning: first serious architecture validation run, not the final capability ceiling.
- Ship gate: 45K readiness remains the primary gate for this pass.

### Truth and Output Modes
- `measured` / `target` / `vision` are distinct claim labels.
- `verified` / `hypothesis` / `creative_or_folklore` are distinct output modes.
- Default mode is `verified`.
- No claim without evidence.

### Claims Boundary
- This repository is **pre-training / unverified** for production-scale quality claims.
- Release closure artifacts prove process integrity, not final model capability.

## Release-Ready One-Shot Closure

This repository includes a decision-complete, single-pass closure flow for engineering hardening and release evidence.

### Canonical Root
- repository root (current working tree)

### Single Entry Point
- `bash scripts/final_one_shot.sh`
- `bash scripts/one_command_full_sop.sh` is the core validation subflow used by the canonical closure wrapper above.

### Outputs
- `reports/start_gate_report.json`
- `reports/release_manifest.json`
- `reports/project_structure_sync_report.json`
- `reports/hardening_bundle_summary.json`
- `reports/master_closure_matrix.md`
- `reports/repo_closure_scorecard.md`
- `reports/known_limits_v1.md`
- `reports/support_maintenance_policy.md`
- `reports/train_readiness_decision.md`
- `reports/final_freeze_manifest.md`
- `reports/repo_external_handoff.md`
- `artifacts/mertformer_release.zip`
- `artifacts/mertformer_release.zip.sha256`

### Repo-Side Closure Pack
- `reports/repo_closure_scorecard.md` tracks the permanent repo-side closure count for the current pass.
- `reports/final_master_plan_freeze.md` freezes what is closed now versus what is post-run or external.
- `reports/known_limits_v1.md` keeps measured truth separate from absent trained evidence.
- `reports/support_maintenance_policy.md`, `reports/quality_gate_matrix.md`, and `reports/test_verification_matrix.md` define the maintenance and verification contract.
- `reports/adr_index.md` indexes the current architecture and governance decisions.
- `configs/`, `releases/`, `knowledge/`, and `evidence/` now hold stable repo-side contract surfaces for named chess profiles, release expectations, glossary terms, and evidence policy.

### Chess Onefile Closure Artifacts
- `reports/run_status_manifest.json` provides the compact end-state snapshot for a finished chess onefile run.
- `reports/postrun_analysis_manifest.json` summarizes curated suite, stockfish, self-play, tournament, and replay-buffer surfaces.
- `reports/artifact_truth_matrix.json` lists expected run artifacts and whether they actually exist.
- `reports/run_contract.json` freezes the operator-facing run contract and claim boundary for that exact onefile execution.
- `reports/release_snapshot.json` records internal release-surface readiness without overstating external release proof.
- `reports/evidence_pack_stub.json` lists what is present now versus what is still missing for external release-grade evidence.
- `reports/final_truth_registry.json` keeps measured/internal/not-eligible chess claims explicit and auditable.
- `reports/claim_registry.json` maps each chess run claim to classification and evidence.
- `reports/known_limits.json` lists run-specific known limits instead of pretending missing proof is closed.
- `reports/support_matrix.json` records active profile/mode/support status for the exact run.
- `reports/release_gate_summary.json` records internal and external release-gate pass/fail state.
- `reports/rc_stub.json` records internal release-candidate stub state for the exact run.
- `reports/golden_stub.json` records that golden release is still separate and stricter than internal closure.
- `reports/handoff_pack_manifest.json` enumerates the operator-facing handoff bundle for the exact run.
- `reports/operator_handoff_summary.json` summarizes whether the operator handoff surface is internally complete.
- `reports/external_repro_stub.json`, `reports/pilot_stub.json`, `reports/security_stub.json`, and `reports/legal_stub.json` keep external closure gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/operator_handbook_stub.json`, `reports/dr_evidence_stub.json`, `reports/backup_retention_stub.json`, and `reports/blind_handoff_stub.json` keep operator/DR closure gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/release_notes_stub.json`, `reports/freeze_manifest_stub.json`, `reports/changelog_snapshot.json`, and `reports/maintenance_policy_stub.json` keep release-governance gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/export_truth_stub.json`, `reports/device_validation_stub.json`, `reports/packaging_closure_stub.json`, and `reports/installer_validation_stub.json` keep device/export/packaging closure gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/benchmark_raw_outputs_stub.json`, `reports/benchmark_compare_report_stub.json`, `reports/benchmark_summary_stub.json`, and `reports/benchmark_manifest_stub.json` keep benchmark closure gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/training_report_stub.json`, `reports/token_accounting_stub.json`, `reports/compute_accounting_stub.json`, and `reports/cost_report_stub.json` keep training/accounting closure gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/final_weights_truth_stub.json`, `reports/best_checkpoint_truth_stub.json`, `reports/latest_checkpoint_truth_stub.json`, and `reports/trained_artifact_registry_stub.json` keep trained-artifact truth gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/core_complete_decision_stub.json`, `reports/research_continues_stub.json`, `reports/product_maintenance_only_stub.json`, and `reports/closure_decision_record_stub.json` keep management-closure gaps explicit instead of pretending they are solved by internal artifacts.
- `reports/master_closure_table.json`, `reports/remaining_core_blockers.json`, `reports/repo_side_completion_summary.json`, and `reports/readiness_snapshot.json` provide a repo-truth summary layer for what is present versus what still blocks final closure.
- `reports/aggregated_master_table.json`, `reports/real_remaining_core_work.json`, `reports/repo_truth_inventory.json`, and `reports/closure_gap_summary.json` provide a more compact top-level truth layer for turning repo reality into a final master table.
- `reports/project_master_truth_reference.json`, `reports/project_remaining_real_blockers.json`, `reports/truth_docs_index.json`, and `reports/truth_docs_drift_report.json` tie onefile release evidence back to canonical repo truth docs and surface documentation drift explicitly.
- `reports/project_blocker_action_plan.json`, `reports/project_blocker_dependency_graph.json`, `reports/project_execution_sequence.json`, `reports/project_lane_status_board.json`, `reports/project_closure_phase_plan.json`, `reports/project_phase_readiness_scoreboard.json`, `reports/project_owner_accountability_matrix.json`, `reports/project_owner_work_queue.json`, `reports/project_critical_path_report.json`, `reports/project_owner_next_actions_summary.json`, `reports/project_ready_now_board.json`, `reports/project_unlock_impact_report.json`, `reports/project_parallel_workset_report.json`, `reports/project_phase_exit_criteria_report.json`, `reports/project_execution_wave_report.json`, `reports/project_evidence_backlog_report.json`, `reports/project_dependency_bottleneck_report.json`, `reports/project_owner_phase_frontier_report.json`, `reports/project_evidence_criticality_report.json`, `reports/project_phase_transition_matrix.json`, `reports/project_owner_load_report.json`, `reports/project_phase_dependency_pressure_report.json`, `reports/project_owner_bottleneck_alignment_report.json`, `reports/project_evidence_phase_heatmap_report.json`, `reports/project_blocker_risk_register_report.json`, `reports/project_release_prereq_matrix_report.json`, `reports/project_foundation_run_dependency_report.json`, `reports/project_release_path_report.json`, `reports/project_external_closure_cluster_report.json`, `reports/project_owner_evidence_gap_report.json`, `reports/project_release_gate_dependency_report.json`, `reports/project_external_signoff_queue_report.json`, `reports/project_release_evidence_bridge_report.json`, `reports/project_training_run_readiness_report.json`, `reports/project_benchmark_closure_dependency_report.json`, `reports/project_release_decision_queue_report.json`, `reports/generated_truth_consistency_report.json`, and `reports/generated_truth_crosscheck_matrix.json` turn the remaining project blockers into an ordered, phase-aware, readiness-aware, owner-aware, leverage-aware, critical-path-aware, parallel-workset-aware, phase-exit-aware, wave-aware, evidence-aware, bottleneck-aware, owner-frontier-aware, evidence-criticality-aware, phase-transition-aware, owner-load-aware, phase-pressure-aware, owner-bottleneck-aware, evidence-phase-aware, risk-aware, release-prereq-aware, foundation-run-aware, release-path-aware, external-closure-cluster-aware, owner-evidence-gap-aware, release-gate-dependency-aware, external-signoff-queue-aware, release-evidence-bridge-aware, training-run-readiness-aware, benchmark-closure-dependency-aware, release-decision-queue-aware, cross-checked closure plan.
- Canonical docs summary: [docs/CHESS_ONEFILE_MASTER_TRUTH.md](docs/CHESS_ONEFILE_MASTER_TRUTH.md)
- Canonical project summary: [docs/PROJECT_MASTER_TRUTH.md](docs/PROJECT_MASTER_TRUTH.md)

![MertFormer Titan Header](assets/header.png)

<div align="center">
  <a href="README.md">🇬🇧 English</a> | <a href="README_TR.md">🇹🇷 Türkçe</a>
</div>

---

## Lawful Safety Deployment Policy (Build 30 V2)

This repository is designed for lawful, auditable, human-approved deployment.

- Human-in-the-loop is mandatory for operational decisions.
- Audit trail and policy boundaries are mandatory in orchestrator/runtime.
- Unauthorized surveillance, covert tracking, and unapproved intervention are out of scope.
- Security and governance checks must pass before any pilot claim.

## Closure 57 Report

Build 30 V2 includes a machine-checkable closure gate:

```bash
python3 scripts/check_57_matrix.py
mertformer 57-report --out reports/closure_57_matrix.json
```

Outputs:
- `reports/closure_57_matrix.json`
- `reports/closure_57_matrix.md`
- `reports/closure_57_matrix_TR.md`
- Transparency note: Closure-57 is process-green (`57/57`) and currently reports `out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 54, 55, 56, 57]`.
<br />

```
 __  __          _   ______
 |  \/  |        | | |  ____|
 | \  / | ___ _ _| |_| |__ ___  _ __ _ __ ___   ___ _ __
 | |\/| |/ _ \ '__| __|  __/ _ \| '__| '_ ` _ \ / _ \ '__|
 | |  | |  __/ |  | |_| | | (_) | |  | | | | | |  __/ |
 |_|  |_|\___|_|   \__|_|  \___/|_|  |_| |_| |_|\___|_|
      _______ _ _
     |__   __(_) |
        | |   _| |_ __ _ _ __
        | |  | | __/ _` | '_ \
        | |  | | || (_| | | | |
        |_|  |_|\__\__,_|_| |_|

   M  O  B  I  L  E     F  I  R  S  T     E  D  G  E     A  I
```

# 🦅 MertFormer Titan: Autonomous Swarm Architecture
> **Target: near-frontier coding capability at mobile compute cost (pending training/benchmarks).**
> **Development Stage:** Pilot-ready pre-training baseline (`Build 30 V2`, training/benchmark claims pending).

## 🇹🇷 Executive Summary (Non-Technical)
> **For decision makers who are not reading source code**

**What is this project?**
MertFormer is designed as an offline-first AI system that can run on controlled local hardware and continue operating without always-on cloud dependency.

**Why is it strategically relevant?**
1. **Data Control:** Primary design goal is local/offline operation to reduce external data exposure.
2. **Operational Continuity:** The architecture is designed to keep functioning under constrained connectivity.
3. **Language/Domain Adaptation:** Turkish-first documentation and workflow alignment are treated as core requirements.

**In short:** The system is positioned as a disciplined, mission-focused AI infrastructure rather than a generic internet chatbot.

### ✅ Validation Evidence (Latest Local Run)
| Gate | Result |
| :--- | :--- |
| `python3 -m pytest -q` | `191 passed, 3 skipped` |
| `.titan-venv/bin/python -m ruff check .` | `All checks passed` |
| `bash scripts/verify_all.sh` | `[verify] OK` |

## 🚀 Training Readiness (Operational)
**Status:** `TRAIN_ALLOWED`

**Feature highlight:** the canonical 45K path is now `bash zero_touch_start.sh`, which owns the exact readiness verdict, run lock, resume policy, and post-train autorun contract.

This repository is no longer in idea/prototype-only state. The current working tree is repo-side 45K-ready on the canonical `offline_clean` lane; trained outputs are still absent because the real long run has not been executed on the target hardware yet.

### Evidence Snapshot
1. **Core quality gates passed**
   - `pytest` passed (`191 passed, 3 skipped`)
   - `ruff check` passed (`All checks passed`)
   - `verify_all.sh` passed (`[verify] OK`)
2. **Architecture and safety checks passed**
   - Offline preflight completed with all-green status.
   - Operator gate passed (overfit, failure-budget, golden-samples).
3. **Traceable artifacts generated**
   - `logs/preflight/titan_preflight.log`
   - `logs/operator_mode/*.manifest.json`
   - `reports/run_contract.md`
   - `reports/post_train_automation_contract.md`
   - `reports/final_truth_matrix.md`

### Current Exact Boundary
- Canonical repo-side lane: `offline_clean`
- Exact readiness verdict: `TRAIN_ALLOWED` / `READY_OFFLINE_CLEAN`
- Remaining exact blocker: `online_teacher:MISSING_HF_TOKEN` (only for the optional gated teacher lane)

### Final prerequisites before long-run training
- Target hardware allocation (GPU/edge) must be reserved.
- Transfer the repo/package artifacts to the real training machine and rerun the canonical start gate there.
- `HF_TOKEN` is required only if you intentionally choose the online teacher/gated-access path.
- Dataset license/hash workflow must remain compliant.
- Full training run and benchmark outputs will be recorded only after those prerequisites.
- Token budget (V2): default `TITAN_TOKEN_BUDGET_MODE=fixed_steps`, `TITAN_MAX_STEPS=45000`, `TITAN_TARGET_TOKENS_MIN=23.6B`.
- Precomputed logits path: `TITAN_LOGITS_PATH` (default `./datasets/logits/`).
- Default token budget now uses `fixed_steps` (45K). Use `TITAN_TOKEN_BUDGET_MODE=open_ended` only with an explicit target override.
- Accelerate config must match GPU count (set `TITAN_FORCE_ACCELERATE_RECONF=1` to regenerate).
- `cuda.lock` must be created on the target training hardware.

### Minimum Training Hardware (Claim-Safe)
- Dev smoke/preflight: CPU/MPS, 16 GB RAM, 50 GB free disk.
- Master training target: 8x A100 80GB (or equivalent), 1 TB fast SSD, NVLink recommended.

### Canonical readiness gate
```bash
bash zero_touch_start.sh --check-only
```

### Start command on the target training hardware
```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

### Portable Train-Ready Checklist (Zip/Transfer Workflow)
1. Inspect the canonical runtime plan:
```bash
bash zero_touch_start.sh --plan-only
```
2. Validate strict readiness without starting training:
```bash
bash zero_touch_start.sh --check-only
```
3. Required environment variables:
- `HF_TOKEN` (required only for the optional gated teacher + online datasets lane)
- `WANDB_API_KEY` (optional)
- See `.env.example` for the full list.
4. One-command training start after transfer/unzip:
```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```
Optional online teacher lane:
```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```
5. Exact readiness and start-gate reports:
- `reports/train_readiness_decision.json`
- `reports/train_readiness_decision.md`
- `reports/start_gate_report.json`
6. Dataset manifest policy:
- Build30 Final Convergence keeps the current dataset manifest pinned (no major dataset expansion in this lock pass).
7. Legacy helper flows:
- `run.sh` remains available for `--test`, `--sitl-demo`, and `--cleanroom-verify`, but it is no longer the canonical 45K train-end launcher.

| Engineering Status | `Pilot-ready pre-training baseline` |
| :--- | :--- |
| **Training Start Readiness** | ✅ TRAIN_ALLOWED (`READY_OFFLINE_CLEAN`; optional online teacher lane remains blocked without `HF_TOKEN`) |
| **Codebase** | ✅ Implemented (tests + offline preflight passing) |
| **Offline Verification** | ✅ PASS (`bash scripts/verify_all.sh`) |
| **Dataset Compliance** | ✅ Ready for offline-clean (`license/hash workflow active; stage JSONL files exist in the current working tree`) |
| **Full Training Run** | ▶️ Not started yet (`starts with first long-run on allocated hardware`) |
| **Benchmarks** | ⛔ Not eligible for claim without a trained checkpoint (`NOT ELIGIBLE FOR CLAIM`) |

### Parameter Disclosure (Claim Boundary)
- **Design target (Build 30 V2):** `2.64B` parameters.
- **Latest measured runtime total:** `3,698,246,156` parameters (`~3.70B`).
- **Interpretation:** `2.64B` is the architecture/positioning target; `~3.70B` is the current measured runtime total and is the authoritative figure for factual claims.

Engineering truth (strict): see `reports/verified_matrix.md`.

> **MertFormer is a structural efficiency standard that decentralizes enterprise intelligence by minimizing AI inference costs at the device level.**

---

### 💼 Executive Brief
**This section translates the structural-efficiency positioning into operational outcomes for technical and executive decision-makers.**

*   **💰 Targeted ~90% Operational Savings (Estimate)**: Cloud server expenses can be minimized in target deployments. MertFormer aims to reduce processing costs by optimizing energy at the NPU level.
*   **🛡️ Data Sovereignty**: Data is designed to be processed on-device. This is a structural advantage for markets with high security standards, such as defense, law, and finance.
*   **🌍 Scalable Access (Target)**: An autonomous system aiming for GPT-3.5-class capability after training, even in low-bandwidth regions without always-on internet.

*Note: All performance and training-duration figures are pre-training estimates and will be empirically validated after full runs.*

---

### 🏰 The Strategic Moat
**Why MertFormer Titan remains unparalleled:**
1.  **Edge-Native Architecture**: Models from Big Tech are optimized for massive compute on the cloud. Titan's 1.58-bit layers are designed as hardware-aware components from the ground up, creating a clear efficiency gap compared to post-quantized models.
2.  **Liquid Momentum**: The proprietary `LiquidRouter` treats data as a temporal flow (momentum), not just a static input. This mathematical approach positions the system with an advantage that competitors cannot close with hardware power alone.
3.  **Forensic Trust**: Chained training logs and cryptographic outputs are designed to support transparency and compliance verification once real runs are produced.

### 🔒 License Boundary (Quick Note)
- This repository is **proprietary and confidential**.
- Source code, assets, and methods may only be used under explicit written agreement or employment contract with the owner.
- Any third-party disclosure of confidential technical details requires signed NDA terms.
- Full legal terms remain in [`LICENSE`](LICENSE) and [`LICENSE_TR`](LICENSE_TR).

---

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg?style=flat-square)](./LICENSE)
[![Repository: Private](https://img.shields.io/badge/Repository-Private-orange.svg?style=flat-square)](https://github.com/latentcore/mertformer-titan-core)
[![Status: Pre-Training](https://img.shields.io/badge/Status-Pre--Training-yellow.svg?style=flat-square)](https://github.com/latentcore/mertformer-titan-core)
[![Architecture: BitNet 1.58b](https://img.shields.io/badge/Architecture-BitNet%201.58b-orange.svg?style=flat-square)](https://www.microsoft.com/en-us/research/publication/the-era-of-1-bit-llms-all-large-language-models-are-in-1-58-bits/)
[![Reference: BitNet 1-bit](https://img.shields.io/badge/Reference-BitNet%201--bit-lightgrey.svg?style=flat-square)](https://arxiv.org/abs/2310.11453)

## 🏗️ Design Principles
*   **Production-First Mindset**: Built for stability, security, and scalability from Day 1.
*   **Security-Aware Architecture**: Built-in secret management, role-based access, and red-teaming.
*   **Scalable Agent Orchestration**: From 3 agents (Nano) to 45 agents (Omega) based on task complexity.
*   **Observability-Ready**: Full logging, post-mortem analysis, and forensic audit trails.

---

## 📋 Table of Contents

- [Docs Index](#docs-index)
- [Overview](#overview)
- [Real-World Application (Experimental)](#real-world-application-experimental)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [SDK API Quick Start](#sdk-api-quickstart)
- [Troubleshooting](#troubleshooting)
- [Training](#training)
- [Training Strategy (Baseline -> v28)](#training-strategy-baseline-v28)
- [Deployment](#deployment)
- [Integration Targets](#integration-targets)
- [Benchmarks](#benchmarks)
- [Turkish Vision](#turkish-vision)
- [FAQ](#faq)
- [License](#license)
- [Strategic Collaboration](#strategic-collaboration)
- [Scalability Vision](#scalability-vision)
- [Contact](#contact)

---

<a id="docs-index"></a>
## 📚 Docs Index

**Core**
Primary entry docs and checklists.
- [README.md](README.md) — English overview.
- [README_TR.md](README_TR.md) — Turkish overview.
- [CITATION.cff](CITATION.cff) — Citation metadata (Cite this repository).
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines (internal use).
- [CONTRIBUTING_TR.md](CONTRIBUTING_TR.md) — Contribution guidelines (TR).
- [README_CHECKLIST.md](README_CHECKLIST.md) — README audit checklist (EN).
- [README_CHECKLIST_TR.md](README_CHECKLIST_TR.md) — README audit checklist (TR).
- [scripts/README.md](scripts/README.md) — Scripts catalog (EN).
- [scripts/README_TR.md](scripts/README_TR.md) — Scripts catalog (TR).
- [snake_demo.py](snake_demo.py) — Pygame cyberpunk Snake autoplayer (LIVE DEMO).
- [USAGE_GUIDE.md](USAGE_GUIDE.md) — Operational usage guide (EN).
- [USAGE_GUIDE_TR.md](USAGE_GUIDE_TR.md) — Operational usage guide (TR).
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Troubleshooting guide (EN).
- [TROUBLESHOOTING_TR.md](TROUBLESHOOTING_TR.md) — Troubleshooting guide (TR).
- [MODEL_LICENSE.md](MODEL_LICENSE.md) — Model license summary (EN).
- [MODEL_LICENSE_TR.md](MODEL_LICENSE_TR.md) — Model license summary (TR).
- [.env.example](.env.example) — Environment variable template.
- [docs/CHAIN_MAP.md](docs/CHAIN_MAP.md) — Connected vs independent chain map (EN).
- [docs/CHAIN_MAP_TR.md](docs/CHAIN_MAP_TR.md) — Connected vs independent chain map (TR).
- [reports/commercial_handover/known_issues.md](reports/commercial_handover/known_issues.md) — Known issues register for transfer risk visibility.
- [reports/commercial_handover/known_issues_TR.md](reports/commercial_handover/known_issues_TR.md) — Known issues register (TR).
- [reports/commercial_handover/handover_scope.md](reports/commercial_handover/handover_scope.md) — Transfer scope and explicit out-of-scope boundaries.
- [reports/commercial_handover/handover_scope_TR.md](reports/commercial_handover/handover_scope_TR.md) — Transfer scope and out-of-scope boundaries (TR).
- [reports/commercial_handover/ownership_and_role.md](reports/commercial_handover/ownership_and_role.md) — Ownership model and decision rights after transfer.
- [reports/commercial_handover/ownership_and_role_TR.md](reports/commercial_handover/ownership_and_role_TR.md) — Ownership model and decision rights (TR).
- [reports/commercial_handover/sla_kpi_90_180.md](reports/commercial_handover/sla_kpi_90_180.md) — 90/180 day SLA and KPI operating plan.
- [reports/commercial_handover/sla_kpi_90_180_TR.md](reports/commercial_handover/sla_kpi_90_180_TR.md) — 90/180 day SLA and KPI plan (TR).
- [reports/commercial_handover/contract_terms_checklist.md](reports/commercial_handover/contract_terms_checklist.md) — Contract checklist for IP, liability, operations and exit.
- [reports/commercial_handover/contract_terms_checklist_TR.md](reports/commercial_handover/contract_terms_checklist_TR.md) — Contract checklist for IP, liability, operations and exit (TR).

**SDK**
Package + CLI for edge deployments.
- [mertformer_sdk/](mertformer_sdk/) — SDK package (API + CLI + kernels).
- [SDK_GUIDE.md](SDK_GUIDE.md) — SDK quick guide (EN).
- [SDK_GUIDE_TR.md](SDK_GUIDE_TR.md) — SDK quick guide (TR).

**Plans**
Execution roadmaps and operator plans.
- [TASK.md](TASK.md) — Operator Mode task plan (EN).
- [TASK_TR.md](TASK_TR.md) — Operator Mode task plan (TR).
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Implementation plan (EN).
- [IMPLEMENTATION_PLAN_TR.md](IMPLEMENTATION_PLAN_TR.md) — Implementation plan (TR).
- [TRAINING_PLAN.md](TRAINING_PLAN.md) — Training roadmap (EN).
- [TRAINING_PLAN_TR.md](TRAINING_PLAN_TR.md) — Training roadmap (TR).
- [CHANGELOG.md](CHANGELOG.md) — Release changelog (EN).
- [CHANGELOG_TR.md](CHANGELOG_TR.md) — Release changelog (TR).

**Technical**
Deep-dive architecture and research references.
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — Technical deep dive (EN).
- [TECHNICAL_REPORT_TR.md](TECHNICAL_REPORT_TR.md) — Technical deep dive (TR).
- [WHITE_PAPER_LIQUIDROUTER.md](WHITE_PAPER_LIQUIDROUTER.md) — LiquidRouter white paper (EN).
- [WHITE_PAPER_LIQUIDROUTER_TR.md](WHITE_PAPER_LIQUIDROUTER_TR.md) — LiquidRouter white paper (TR).

**Internal**
Internal roadmap and capability gap mapping (non-public).
- [INTERNAL_AGI_GAP.md](INTERNAL_AGI_GAP.md) — Internal AGI gap map (EN).
- [INTERNAL_AGI_GAP_TR.md](INTERNAL_AGI_GAP_TR.md) — Internal AGI gap map (TR).

**Audit & Strategy**
Report accuracy audit and strategic value summary.
- [reports/report_accuracy_audit.md](reports/report_accuracy_audit.md) — Report accuracy audit (EN).
- [reports/report_accuracy_audit_TR.md](reports/report_accuracy_audit_TR.md) — Report accuracy audit (TR).
- [reports/codex_deep_audit_EN.md](reports/codex_deep_audit_EN.md) — Deep engineering audit (EN).
- [reports/codex_deep_audit_DE.md](reports/codex_deep_audit_DE.md) — Deep engineering audit (DE).
- [reports/codex_deep_audit_TR.md](reports/codex_deep_audit_TR.md) — Deep engineering audit (TR).
- [reports/codex_deep_audit_EN_TR.md](reports/codex_deep_audit_EN_TR.md) — EN audit Turkish pointer file (TR, canonical content in `codex_deep_audit_TR.md`).
- [reports/codex_deep_audit_DE_TR.md](reports/codex_deep_audit_DE_TR.md) — DE audit Turkish pointer file (TR, canonical content in `codex_deep_audit_TR.md`).
- DE-language audit files are retained as external review artifacts for German-speaking stakeholders.
- [reports/verified_matrix.md](reports/verified_matrix.md) — Verified vs Target matrix (EN).
- [reports/verified_matrix_TR.md](reports/verified_matrix_TR.md) — Verified vs Target matrix (TR).
- [reports/review_checklist.md](reports/review_checklist.md) — External review checklist (EN).
- [reports/review_checklist_TR.md](reports/review_checklist_TR.md) — External review checklist (TR).
- [reports/release_snapshot.md](reports/release_snapshot.md) — Release snapshot (EN).
- [reports/release_snapshot_TR.md](reports/release_snapshot_TR.md) — Release snapshot (TR).
- [reports/final_sync_matrix.md](reports/final_sync_matrix.md) — Final sync matrix (EN).
- [reports/final_sync_matrix_TR.md](reports/final_sync_matrix_TR.md) — Final sync matrix (TR).
- [reports/go_status_matrix.md](reports/go_status_matrix.md) — GO/NO-GO status matrix (EN).
- [reports/go_status_matrix_TR.md](reports/go_status_matrix_TR.md) — GO/NO-GO status matrix (TR).
- [reports/go_nogo_signoff_onepager.md](reports/go_nogo_signoff_onepager.md) — Technical GO/NO-GO one-pager (EN).
- [reports/go_nogo_signoff_onepager_TR.md](reports/go_nogo_signoff_onepager_TR.md) — Technical GO/NO-GO one-pager (TR).
- [reports/closure_57_matrix.md](reports/closure_57_matrix.md) — Closure 57 matrix (EN).
- [reports/closure_57_matrix_TR.md](reports/closure_57_matrix_TR.md) — Closure 57 matrix (TR).
- [reports/report_truth_matrix.md](reports/report_truth_matrix.md) — Report truth matrix (EN).
- [AGENTS.md](AGENTS.md) — Closure constitution for contributors and coding agents.
- [reports/source_of_truth_map.md](reports/source_of_truth_map.md) — Current source-of-truth authority map.
- [reports/final_backlog_classification.md](reports/final_backlog_classification.md) — Current grouped backlog status accounting.
- [reports/final_truth_matrix.md](reports/final_truth_matrix.md) — Current claim-to-evidence truth matrix.
- [reports/release_closure_note.md](reports/release_closure_note.md) — Release closure note (EN).
- [reports/kpi_pack_v1.md](reports/kpi_pack_v1.md) — KPI pack (EN).
- [reports/kpi_pack_v1_TR.md](reports/kpi_pack_v1_TR.md) — KPI pack (TR).
- [reports/cleanroom_verification.md](reports/cleanroom_verification.md) — Fresh-clone reproducibility evidence (EN).
- [reports/cleanroom_verification_TR.md](reports/cleanroom_verification_TR.md) — Fresh-clone reproducibility evidence (TR).
- [reports/legal_cleanroom_signoff_internal.md](reports/legal_cleanroom_signoff_internal.md) — Internal cleanroom legal sign-off record (EN).
- [reports/teacher_output_license_assessment.md](reports/teacher_output_license_assessment.md) — Teacher/output license internal assessment (EN).
- [reports/contamination_report_build30.md](reports/contamination_report_build30.md) — Build30 contamination report (EN).
- [reports/kpi_contract_build30.md](reports/kpi_contract_build30.md) — Technical KPI contract for GO decision (EN).
- [reports/benchmarks/README.md](reports/benchmarks/README.md) — Benchmark outputs guide (EN).
- [reports/benchmarks/README_TR.md](reports/benchmarks/README_TR.md) — Benchmark outputs guide (TR).
- [reports/benchmarks/smoke_train_metrics.json](reports/benchmarks/smoke_train_metrics.json) — Smoke benchmark metrics snapshot (machine-readable).
- [reports/strategic_value.md](reports/strategic_value.md) — Strategic value summary (EN).
- [reports/strategic_value_TR.md](reports/strategic_value_TR.md) — Strategic value summary (TR).
- [reports/efficiency_convergence_analysis.md](reports/efficiency_convergence_analysis.md) — Convergence analysis (BitNet/Liquid/MoE, forecast, EN).
- [reports/efficiency_convergence_analysis_TR.md](reports/efficiency_convergence_analysis_TR.md) — Convergence analysis (BitNet/Liquid/MoE, forecast, TR).

**Pitch & Assets**
Investor-facing materials and launch assets.
- [PITCH.md](PITCH.md) — Investor pitch (EN).
- [PITCH_TR.md](PITCH_TR.md) — Investor pitch (TR).
- [reports/investor_deck.pptx](reports/investor_deck.pptx) — Investor deck (EN).
- [reports/investor_deck_TR.pptx](reports/investor_deck_TR.pptx) — Investor deck (TR).
- [reports/one_pager.md](reports/one_pager.md) — One-pager (EN).
- [reports/one_pager_TR.md](reports/one_pager_TR.md) — One-pager (TR).
- [reports/technical_snapshot.md](reports/technical_snapshot.md) — Technical snapshot (EN).
- [reports/technical_snapshot_TR.md](reports/technical_snapshot_TR.md) — Technical snapshot (TR).
- [reports/asset_stack.md](reports/asset_stack.md) — Asset index (EN).
- [reports/asset_stack_TR.md](reports/asset_stack_TR.md) — Asset index (TR).
- [assets/snake_demo_proof.mp4](assets/snake_demo_proof.mp4) — 30-second snake demo proof clip.
- [assets/snake_demo_preview.gif](assets/snake_demo_preview.gif) — Embedded snake demo preview (GIF).
- [assets/sources/README.md](assets/sources/README.md) — Editable visual source archive standard (EN).
- [assets/sources/README_TR.md](assets/sources/README_TR.md) — Editable visual source archive standard (TR).

![Snake Demo Preview](assets/snake_demo_preview.gif)

Open full clip: [assets/snake_demo_proof.mp4](assets/snake_demo_proof.mp4)
- [reports/founders_hub_application.md](reports/founders_hub_application.md) — Founders Hub draft (EN).
- [reports/founders_hub_application_TR.md](reports/founders_hub_application_TR.md) — Founders Hub draft (TR).
- [reports/security_compliance.md](reports/security_compliance.md) — Security & compliance brief (EN).
- [reports/security_compliance_TR.md](reports/security_compliance_TR.md) — Security & compliance brief (TR).
- [reports/poc_protocol.md](reports/poc_protocol.md) — Pilot/PoC protocol (EN).
- [reports/poc_protocol_TR.md](reports/poc_protocol_TR.md) — Pilot/PoC protocol (TR).
- [reports/pilot_readiness_kit.md](reports/pilot_readiness_kit.md) — Pilot readiness kit (EN).
- [reports/pilot_readiness_kit_TR.md](reports/pilot_readiness_kit_TR.md) — Pilot readiness kit (TR).
- [reports/pilot_offer_packages.md](reports/pilot_offer_packages.md) — Standard pilot offer packages (EN).
- [reports/pilot_offer_packages_TR.md](reports/pilot_offer_packages_TR.md) — Standard pilot offer packages (TR).
- [reports/sales_funnel_90d.md](reports/sales_funnel_90d.md) — 90-day B2B pilot sales funnel (EN).
- [reports/sales_funnel_90d_TR.md](reports/sales_funnel_90d_TR.md) — 90-day B2B pilot sales funnel (TR).
- [reports/drone_sitl_demo.md](reports/drone_sitl_demo.md) — SITL drone proof protocol (EN).
- [reports/drone_sitl_demo_TR.md](reports/drone_sitl_demo_TR.md) — SITL drone proof protocol (TR).
- [reports/pilots/README.md](reports/pilots/README.md) — Pilot evidence folder standard (EN).
- [reports/pilots/README_TR.md](reports/pilots/README_TR.md) — Pilot evidence folder standard (TR).
- [reports/pilot_acceptance_signoff.md](reports/pilot_acceptance_signoff.md) — Pilot acceptance signature template (EN).
- [reports/pilot_acceptance_signoff_TR.md](reports/pilot_acceptance_signoff_TR.md) — Pilot acceptance signature template (TR).
- [reports/ip_licensing_split.md](reports/ip_licensing_split.md) — Sectoral IP split framework (EN).
- [reports/ip_licensing_split_TR.md](reports/ip_licensing_split_TR.md) — Sectoral IP split framework (TR).
- [reports/dataset_health.md](reports/dataset_health.md) — Dataset health report (EN).
- [reports/dataset_health_TR.md](reports/dataset_health_TR.md) — Dataset health report (TR).
- [reports/model_health.md](reports/model_health.md) — Model health report (EN).
- [reports/model_health_TR.md](reports/model_health_TR.md) — Model health report (TR).
- [reports/system_hardware.md](reports/system_hardware.md) — System hardware report (EN).
- [reports/system_hardware_TR.md](reports/system_hardware_TR.md) — System hardware report (TR).
- [reports/cli_smoke_log.md](reports/cli_smoke_log.md) — CLI smoke log (EN).
- [reports/cli_smoke_log_TR.md](reports/cli_smoke_log_TR.md) — CLI smoke log (TR).


**Ops & Governance**
Security, provenance, reproducibility, and ops notes.
- [MODEL_CARD.md](MODEL_CARD.md) — Model card (EN).
- [MODEL_CARD_TR.md](MODEL_CARD_TR.md) — Model card (TR).
- [USE_POLICY.md](USE_POLICY.md) — Use policy (EN).
- [USE_POLICY_TR.md](USE_POLICY_TR.md) — Use policy (TR).
- [SECURITY.md](SECURITY.md) — Security policy (EN).
- [SECURITY_TR.md](SECURITY_TR.md) — Security policy (TR).
- [DECISIONS.md](DECISIONS.md) — Architecture decisions (EN).
- [DECISIONS_TR.md](DECISIONS_TR.md) — Architecture decisions (TR).
- [datasets/README.md](datasets/README.md) — Dataset overview (EN).
- [datasets/README_TR.md](datasets/README_TR.md) — Dataset overview (TR).
- [datasets/SOURCES.md](datasets/SOURCES.md) — Data sources (EN).
- [datasets/SOURCES_TR.md](datasets/SOURCES_TR.md) — Data sources (TR).
- [datasets/LICENSES.md](datasets/LICENSES.md) — Data licenses (EN).
- [datasets/LICENSES_TR.md](datasets/LICENSES_TR.md) — Data licenses (TR).
- [datasets/inventory.md](datasets/inventory.md) — Dataset inventory (auto, EN).
- [datasets/inventory_TR.md](datasets/inventory_TR.md) — Dataset inventory (auto, TR).
- [datasets/inventory.json](datasets/inventory.json) — Dataset inventory (auto, machine-readable).
- [repro/seed_policy.md](repro/seed_policy.md) — Seed policy (EN).
- [repro/seed_policy_TR.md](repro/seed_policy_TR.md) — Seed policy (TR).
- [repro/python.md](repro/python.md) — Python 3.11 baseline setup (EN).
- [repro/python_TR.md](repro/python_TR.md) — Python 3.11 baseline setup (TR).
- [repro/accelerate_default.yaml](repro/accelerate_default.yaml) — Example accelerate config (local).
- [repro/pip_freeze.txt](repro/pip_freeze.txt) — Environment snapshot (pip freeze).
- [logs/README.md](logs/README.md) — Logs index + unified logbook notes.
- `logs/ALL_LOGS.jsonl` — Unified logbook artifact (gitignored; generated via `.titan-venv/bin/python scripts/logbook_build.py --append`).
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — CI pipeline (pytest + preflight + secret scan).
- [interfaces/inference_contract.md](interfaces/inference_contract.md) — Inference contract (EN).
- [interfaces/inference_contract_TR.md](interfaces/inference_contract_TR.md) — Inference contract (TR).
- [interfaces/pilot_report_v1.schema.json](interfaces/pilot_report_v1.schema.json) — Pilot report JSON schema.
- [economics/cost_model.md](economics/cost_model.md) — Cost model (EN).
- [economics/cost_model_TR.md](economics/cost_model_TR.md) — Cost model (TR).
- [economics/efficiency_report.md](economics/efficiency_report.md) — Efficiency report (EN).
- [economics/efficiency_report_TR.md](economics/efficiency_report_TR.md) — Efficiency report (TR).
- [limits/scaling_breakpoints.md](limits/scaling_breakpoints.md) — Scaling breakpoints (EN).
- [limits/scaling_breakpoints_TR.md](limits/scaling_breakpoints_TR.md) — Scaling breakpoints (TR).
- [postmortems/README.md](postmortems/README.md) — Postmortem guide (EN).
- [postmortems/README_TR.md](postmortems/README_TR.md) — Postmortem guide (TR).
- [postmortems/_template.md](postmortems/_template.md) — Postmortem template (EN).
- [postmortems/_template_TR.md](postmortems/_template_TR.md) — Postmortem template (TR).
- [postmortems/example_001.md](postmortems/example_001.md) — Example postmortem (EN).
- [postmortems/example_001_TR.md](postmortems/example_001_TR.md) — Example postmortem (TR).
- [prompts/changelog.md](prompts/changelog.md) — Prompt change log (EN).
- [prompts/changelog_TR.md](prompts/changelog_TR.md) — Prompt change log (TR).
- [tokenizer/stats.md](tokenizer/stats.md) — Tokenizer stats (EN).
- [tokenizer/stats_TR.md](tokenizer/stats_TR.md) — Tokenizer stats (TR).
- [tokenizer/drift_report.md](tokenizer/drift_report.md) — Tokenizer drift report (EN).
- [tokenizer/drift_report_TR.md](tokenizer/drift_report_TR.md) — Tokenizer drift report (TR).
- [tokenizer/tr/README.md](tokenizer/tr/README.md) — Turkish tokenizer cache note (EN).
- [tokenizer/tr/README_TR.md](tokenizer/tr/README_TR.md) — Turkish tokenizer cache note (TR).
- [ablations/results.md](ablations/results.md) — Ablation results (EN).
- [ablations/results_TR.md](ablations/results_TR.md) — Ablation results (TR).
- [ablations/no_moe/README.md](ablations/no_moe/README.md) — Ablation: no MoE (EN).
- [ablations/no_moe/README_TR.md](ablations/no_moe/README_TR.md) — Ablation: no MoE (TR).
- [ablations/no_liquid/README.md](ablations/no_liquid/README.md) — Ablation: no Liquid (EN).
- [ablations/no_liquid/README_TR.md](ablations/no_liquid/README_TR.md) — Ablation: no Liquid (TR).
- [ablations/dense_only/README.md](ablations/dense_only/README.md) — Ablation: dense-only (EN).
- [ablations/dense_only/README_TR.md](ablations/dense_only/README_TR.md) — Ablation: dense-only (TR).
- [ablations/bitlinear_off/README.md](ablations/bitlinear_off/README.md) — Ablation: BitLinear off (EN).
- [ablations/bitlinear_off/README_TR.md](ablations/bitlinear_off/README_TR.md) — Ablation: BitLinear off (TR).
- [experiments/exp_001_baseline/notes.md](experiments/exp_001_baseline/notes.md) — Experiment notes (EN).
- [experiments/exp_001_baseline/notes_TR.md](experiments/exp_001_baseline/notes_TR.md) — Experiment notes (TR).
- [tools/abuse_tests.md](tools/abuse_tests.md) — Tool abuse tests (EN).
- [tools/abuse_tests_TR.md](tools/abuse_tests_TR.md) — Tool abuse tests (TR).
- [tools/sandbox/README.md](tools/sandbox/README.md) — Tool sandbox guide (EN).
- [tools/sandbox/README_TR.md](tools/sandbox/README_TR.md) — Tool sandbox guide (TR).
- [tools/contracts/README.md](tools/contracts/README.md) — Tool contracts (EN).
- [tools/contracts/README_TR.md](tools/contracts/README_TR.md) — Tool contracts (TR).
- [training_dynamics/cold_vs_warm.md](training_dynamics/cold_vs_warm.md) — Training dynamics notes (EN).
- [training_dynamics/cold_vs_warm_TR.md](training_dynamics/cold_vs_warm_TR.md) — Training dynamics notes (TR).

---

<a id="overview"></a>
## 🎯 Overview

MertFormer Titan is a cutting-edge **2.64B parameter** language model designed for **on-device inference** on mobile platforms. Combining **BitNet 1.58-bit quantization**, **Liquid Neural Networks**, **Sparse Mixture of Experts (MoE)**, and **MLA-labeled GQA attention (current implementation)**, it **targets GPT-3.5 level performance (pre-training target)** while running entirely on a smartphone.

Architecture truth note: `layers/mla.py` class naming is `MLA`, while the current attention core is GQA-based (`num_kv_heads` projection + KV head replication). Full latent-MLA bottleneck remains a roadmap item.

Name expansion:
- **MERT**: **Modular Edge Reasoning Transformer**
- **MertFormer**: **Modular Edge Reasoning Transformer Framework for On-device Modular Execution and Reliability**

### 🔗 Chain Map (Connected vs Independent)
```mermaid
flowchart TD
  A["Stage JSONL (datasets/stage*)"] --> B["Training (run.sh → train/train.py)"]
  B --> C["Logs (logs/*.jsonl)"]
  C --> D["SOP artifacts (reports + packages/artifacts zips)"]
```

See the full map: [`docs/CHAIN_MAP.md`](docs/CHAIN_MAP.md)

### Why MertFormer Titan?

- 🛡️ **Privacy-First**: 100% on-device, zero cloud dependency
- ⚡ **Ultra-Efficient**: theoretical **~20x smaller than FP32** (32-bit → 1.58-bit; requires low-bit inference path)
- 🏭 **Industrial-Grade**: Industry-standard optimizations (Flash Attention 2, torch.compile, NCCL tuning)
- 📱 **Mobile-Optimized**: JIT compilation for Samsung S25 NPU
- 🧪 **Research-Grade**: Novel LiquidRouter architecture (contextual MoE routing)
- 🇹🇷 **Turkish-Ready**: Optimized for Turkish language and culture

<a id="real-world-application-experimental"></a>
### 🚁 Real-World Application (Experimental)

- **Proof-of-system target:** Designed to be validated on autonomous UAV/drone-class platforms under real-world constraints.
- **System focus:** Perception → decision → control chain under constrained hardware, latency, and sensor uncertainty.
- **Safety-first behavior:** Fail-safe guardrails and watchdog-style overrides are expected to force deterministic fallback when risk/confidence thresholds are breached.
- **Positioning:** Engineering validation scope only; this is not presented as a certified or production deployment claim.
- **Current constraint:** Validation throughput is currently limited by access to GPU/edge hardware and controlled field-test resources.
- **Collaboration request:** We welcome collaboration for compute support, controlled test environments, and engineering mentorship to accelerate milestones.

---

<a id="key-features"></a>
## 🔥 Key Features

### 1. **BitNet 1.58-bit Quantization** 🤏
- Ternary weights: `{-1, 0, +1}`
- INT8 activations: `[-127, 127]`
- **theoretical ~20x smaller than FP32** (32-bit → 1.58-bit; requires low-bit inference path)
- Straight-Through Estimator (STE) for gradient flow
- RMS scaling for stability (legacy path integrated into Build 30 V2)

### 2. **LiquidRouter (Temporal Conv Router)** 🌍
- **Implementation truth**: Causal depthwise `Conv1d` + rolling state buffer routes MoE tokens.
- **CfC separation**: Closed-form continuous-time (CfC) cells run in `LiquidMixer/LiquidCell`, not inside `LiquidRouter`.
- **Impact**: **estimated 15-20% better routing quality** vs standard routers (stateless).
- **Temporal Routing**: Decisions are based on **historical context**, preventing expert collapse.
- **Dynamic**: Time-constant adaptation with jitter boost for stability.
- **Academic value**: A new paradigm in conditional computation.

### 3. **MLA-labeled GQA Attention (Current Implementation)** 🧠
- GQA-based KV sharing (`num_heads=16`, `num_kv_heads=8` default profile).
- LLaMA-3 compatible RoPE (interleaved & decoupled)
- Optional hierarchical KV cache path (short/long split) for decode efficiency.
- QK normalization for stability
- Flash Attention 2 integration (+30% speedup)
- Long-context ready (theta=100K)

### 4. **Liquid Neural Networks (CfC)** 💧
- True Closed-Form Continuous-time cells
- Dynamic tau (time-constant) adaptation
- Temporal reasoning capabilities
- JIT-compiled for NPU optimization
- 3-strike safeguard system

### 5. **Sparse Mixture of Experts (MoE) & 🚀 LiquidRouter** 🧩
- 8 experts, top-2 routing
- Routing policy: token-choice top-k.
- **Momentum-Based Routing:** Unlike standard routers, `LiquidRouter` selects experts by looking at the data's arrival speed and temporal momentum (`Fluid Path`), not just the immediate word.
- **Causal Conv1d Integration:** It acts more like "strategic intelligence" than a "traffic controller" by considering the past 4-token window (`history_window`) during expert selection.
- **Hardware Compatibility:** `LiquidRouter`'s sharp selections prevent unnecessary expert triggers, leading to an estimated up to 40% energy savings on the Samsung S25 NPU unit.
- Load balancing + Z-loss + Switch loss
- BitSwiGLU experts (quantized)
- Emergency jitter boost for collapse prevention
- Router health monitoring

### 6. **Advanced Training Pipeline** 🚂
- **Knowledge Distillation**: Llama-3.3-70B → 2.64B design target (80% alpha)
- **4 Core Stages + 1 Tool-Use/API Phase**: Logic → Knowledge → Language → Soul (+ Tool Use/API)
- **WSD Scheduler**: Warmup-Stable-Decay (grokking-optimized)
- **Differential Learning Rates**: Router 1.5x, Body 1.0x
- **Early Stopping**: Patience-based with best checkpoint saving
- **Dynamic Alpha**: Progressive distillation weight adjustment

### 7. **Performance Optimizations (v1.0 (Build 30 V2))** ⚡
- ✅ **Flash Attention 2**: Projected +30% speedup (A100/H100)
- ✅ **Fused RMSNorm**: Projected +10% speedup (torch.compile)
- ✅ **torch.compile (max-autotune)**: Projected +15% speedup
- ✅ **CUDA TF32 + cuDNN**: Projected +10% speedup
- ✅ **Enhanced DataLoader**: Projected +5% speedup (16 workers, prefetch=4)
- ✅ **NCCL Tuning**: Projected +5-10% speedup (multi-GPU, auto-detection)
- **Projected total: 70-80% faster training (estimate).**

### 8. **Safety & Reliability** 🛡️
- ✅ **OOM Recovery**: Auto batch size reduction
- ✅ **NaN/Inf Detection**: Gradient zeroing with retry limit
- ✅ **Disk Space Monitoring**: Prevents checkpoint save failures
- ✅ **GPU Memory Tracking**: Real-time utilization reporting
- ✅ **Gradient Norm Monitoring**: Collapse/explosion detection
- ✅ **Liquid Spike Safeguards**: 3-strike freeze mechanism
- ✅ **Best Checkpoint Saving**: Preserves optimal model state

### 9. **Technological Edge (Build 30 V2 Upgrade)** 🛠️
- **GaLore Integration**: Gradient Low-Rank Projection optimization for memory efficiency on Consumer GPUs (Locked).
- **8-bit AdamW**: Memory-optimized optimizer reduces optimizer state footprint by 75% (Locked).
- **Offline Knowledge Distillation**: Pre-computed Llama-3-70B logits for zero-overhead teacher training (requires precomputed shards; falls back to online teacher if missing).
- **Smart Parallel Orchestration (Hyper-Threading)**: Zero-latency pipeline where data download, distillation, and training happen concurrently.

### QINN Status (Current Build)
- **Default state:** `use_qinn=False` (disabled in Build 30 V2).
- **Why disabled now:** prioritizes training stability, throughput, and edge/NPU compatibility in the primary path.
- **If enabled later:** can be evaluated as an experimental regularization layer, but may add compute overhead and convergence risk.
- **Reference path:** `layers/qinn.py` (kept in codebase for controlled ablation use).

---

<a id="architecture"></a>
## 🏗️ Architecture

### Build 30 V2 Cognitive Extensions (Feature-Flag, default OFF)
These modules are implemented in code and can be enabled from config before training/inference runs:

- `use_hierarchical_kv_cache` -> Hierarchical KV cache short/long split (`layers/mla.py`)
- `use_global_workspace_broadcast` -> Global workspace broadcast (`layers/cognitive_extensions.py`)
- `use_cross_expert_sync_bus` -> Cross-expert synchronization bus (`layers/moe.py`)
- `use_latent_ode_state_channel` -> Continuous latent ODE state channel (`layers/cognitive_extensions.py`)
- `use_neuromodulatory_gain` -> Neuromodulatory gain modulation (`layers/cognitive_extensions.py`)
- `use_hebbian_plasticity` -> Hebbian plasticity trace layer (`layers/cognitive_extensions.py`)
- `use_neuro_symbolic_layer` -> Neuro-symbolic residual bridge (`layers/cognitive_extensions.py`)
- `use_world_model_head` -> Causal world model head side outputs (`layers/world_model_head.py`)
- `use_lifelong_safety_layer` -> Lifelong safety/adaptation guard (`layers/lifelong_safety.py`)
- `use_structural_plasticity` -> Structural plasticity hooks for expert grow/prune policy (`layers/moe.py`)
- `use_continual_adapter` -> Continual learning adapter path in training (`train/continual_adapter.py`)
- `use_expert_paging` -> Inference-first on-demand expert residency (`layers/moe.py`)

Runtime note:
- Defaults are OFF to preserve a stable baseline.
- These are integrated as non-breaking extensions and can be enabled per experiment.

### Advanced Feature Matrix (Stable vs Max-Arch)
`run.sh` now supports a profile contract through `TITAN_PROFILE`.

| Flag | Stable (default) | Max-Arch | Purpose | File |
| --- | --- | --- | --- | --- |
| `use_hierarchical_kv_cache` | `false` | `true` | Short/long KV split for decode efficiency | `layers/mla.py` |
| `use_global_workspace_broadcast` | `false` | `true` | Shared workspace signal across tokens | `layers/cognitive_extensions.py` |
| `use_neuromodulatory_gain` | `false` | `true` | Workspace-driven gain/bias modulation | `layers/cognitive_extensions.py` |
| `use_latent_ode_state_channel` | `false` | `true` | Continuous latent state dynamics | `layers/cognitive_extensions.py` |
| `use_cross_expert_sync_bus` | `false` | `true` | MoE expert synchronization path | `layers/moe.py` |
| `use_structural_plasticity` | `false` | `true` | Expert grow/prune hooks | `layers/moe.py` |
| `use_hebbian_plasticity` | `false` | `true` | Local plasticity trace layer | `layers/cognitive_extensions.py` |
| `use_neuro_symbolic_layer` | `false` | `true` | Rule-conditioned residual bridge | `layers/cognitive_extensions.py` |
| `use_world_model_head` | `false` | `true` | Side-channel causal prediction outputs | `layers/world_model_head.py` |
| `use_lifelong_safety_layer` | `false` | `true` | Drift-aware adaptive safety clamp | `layers/lifelong_safety.py` |
| `use_continual_adapter` | `false` | `true` | Continual replay/drift adapter in training | `train/continual_adapter.py` |
| `use_expert_paging` | `false` | `true` | On-demand expert residency (inference-first) | `layers/moe.py` |
| `use_qinn` | `false` | `false` | Kept off for stability/throughput in Build30 | `layers/qinn.py` |

Profile examples:
```bash
#- Stable baseline (default)
bash run.sh

#- Max architecture profile
TITAN_PROFILE=max_arch bash run.sh

#- Readiness-only gate
bash zero_touch_start.sh --check-only
```

### Chess Onefile Feature-Bundle Path
`/Users/mertyunlu/Desktop/NİHAİ/mertformer-titan-core/scripts/chess_5080_onefile.py` now supports named bundle overlays and explicit per-flag overrides for the mirrored advanced surfaces.

- Bundle CLI: `--feature-bundle <name>`
- Per-flag CLI: `--enable-features flag_a,flag_b` and `--disable-features flag_c`
- Objective bundle for auxiliary chess heads: `objective_stack`
- Post-run analysis bundle: `postrun_analysis_stack`
- Long-run all-on local ablation profiles: `strength_4060_24h_all_on_experimental` and `strength_4060_24h_omni_max`
- New auxiliary heads: `phase_head`, `wdl_head`, `legality_head`
- Per-run bundle evidence: `reports/feature_flag_report.json` and `reports/feature_flag_report.md`
- New post-run chess artifacts: `reports/selfplay_report.json`, `reports/inference_mode_tournament_report.json`, `reports/replay_buffer_manifest.json`
- Operator runbook/checklist path: `runbooks/chess_4060_24h_all_on_experimental.md` and `checklists/chess_4060_24h_all_on_experimental.md`

Example:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental
```

```text
      ╔═══════════════════════════════════════════════════════════════════════════╗
      ║  M E R T F O R M E R   T I T A N   (O N Y X   S T O R M)                  ║
      ║  » ARCHITECTURE BLUEPRINT v1.0 (Build 30 V2) // TARGET: SAMSUNG S25 NPU «    ║
      ╚═══════════════════════════════════════════════════════════════════════════╝
                                            │
      ┌─────────────────────────────────────▼─────────────────────────────────────┐
      │  INPUT EMBEDDINGS [Batch, Seq, 2048]  ⚡  RoPE (Theta=100k, Float32)       │
      └─────────────────────────────────────┬─────────────────────────────────────┘
                                            │ [B, S, 2048]
                  ┌─────────────────────────▼──────────────────────────┐
                  │            R E S I D U A L   S T R E A M           │◄──┐
                  └─────────────────────────┬──────────────────────────┘   │
      ┌─────────────────────────────────────▼─────────────────────────────────────┐
      │  TRANSFORMER BLOCK [Layers 0-17]  (Iterative Process)                     │
      │                                                                           │
      │  ┌──────────────┐    ┌─────────────────────────────────────────────────┐  │
      │  │ RMSNorm (F)  │───►│ [MLA-LABELED GQA] ATTENTION               │  │
      │  └──────────────┘    │ » GQA heads: Q=16, KV=8 (default profile)                      │  │
      │                      │ » Op: Softmax(Q·K^T / √d) · V                   │  │
      │                      │ » H/W: FlashAttn2 Kernel (SRAM Optimized)       │  │
      │                      └────────────────────────┬────────────────────────┘  │
      │    (Add) ─────────────────────────────────────┘                           │
      │      ▼                                                                    │
      │  ┌──────────────┐    ┌─────────────────────────────────────────────────┐  │
      │  │ RMSNorm (F)  │───►│ [ROUTER] LIQUID CONTEXT AWARENESS               │  │
      │  └──────────────┘    │ » In: [B, S, 2048]                              │  │
      │                      │ » Op: CausalConv1d(k=4) + SiLU + Linear         │  │
      │                      └────────────────────────┬────────────────────────┘  │
      │                                               ▼ [B, S, Num_Experts]       │
      │                        ┌───────────────────────────────────────┐          │
      │                        │    TOP-2 DYNAMIC EXPERT SWITCH (Gate) │          │
      │     └─┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬─┘         │
      │       │       │       │       │       │       │       │       │           │
      │       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼           │
      │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
      │    │EXP_0│ │EXP_1│ │EXP_2│ │EXP_3│ │EXP_4│ │EXP_5│ │EXP_6│ │EXP_7│        │
      │    │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│ │(Bit)│        │
      │    └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘        │
      │       │       │       │       │       │       │       │       │           │
      │       └───────┴───────┴───────┼───────┴───────┴───────┴───────┘           │
      │                               ▼ [B, S, 2048]                              │
      │                     (Weighted Summation Σ g(x)·E(x))                      │
      │                              │                                            │
      │  ┌───────────────────────────▼─────────────────────────────────────────┐  │
      │  │ [LIQUID MIXER] (Active only on Layers 4, 10, 16)                    │  │
      │  │ » Core: CfC (Closed-form Continuous) Cells                          │  │
      │  │ » Eq:   x(t) = (-1/τ)x(t) + A·I(t)                                  │  │
      │  │ » Role: Long-term dependency & Temporal Reasoning                   │  │
      │  └───────────────────────────┬─────────────────────────────────────────┘  │
      │                              │                                            │
      │    (Add) <───────────────────┘                                            │
      │      │ [B, S, 2048]                                                       │
      └──────┼────────────────────────────────────────────────────────────────────┘
             │
      ┌──────▼────────────────────────────────────────────────────────────────────┐
      │ [RMSNorm] + [LM HEAD] 1.58-bit Projection ──► OUTPUT LOGITS [B, S, 128k]  │
      └───────────────────────────────────────────────────────────────────────────┘
```

### 🦅 MertFormer Titan: Synaptic Layer Hierarchy

![Synaptic Hierarchy Map](assets/synaptic_map.png)

The journey of data from Layer 0 to 17:

*   **Layer 0 (Input Block):** First stop for vectorized data; basic word relationships are established and signal amplitude is stabilized using `RMSNorm`.
*   **Layer 1 (Grammar Foundation):** Processing the most fundamental building blocks of language; the `MLA`-labeled GQA attention mechanism creates the initial focus map.
*   **Layer 2 (Efficiency Seal):** Simple context between words is established; thanks to the `BitNet 1.58-bit` structure, all weights are processed with the lowest energy in the $\{-1, 0, +1\}$ space.
*   **Layer 3 (Expert Distribution):** Semantic density increases; the `MoE` structure directs data to the most appropriate 2 out of 8 experts.
*   **Layer 4 (First Liquid Contact):** **Critical Threshold.** The first `LiquidMixer` (CfC) kicks in here, instilling the first sense of "temporal flow" and "momentum."
*   **Layer 5 (Fluid Attention):** Data gaining fluidity is filtered by `MLA`-labeled GQA attention in a deeper dimension, strengthening contextual relationships.
*   **Layer 6 (Complex Syntax):** Indirect structures within sentences are resolved; `MoE` experts continue specific analyses.
*   **Layer 7 (Mathematical Stability):** Foundation for logical inferences is laid; the `UnitaryQINN` path remains available only when `use_qinn=true` (Build 30 V2 default: OFF).
*   **Layer 8 (Abstraction):** Data evolves from concrete words to abstract concepts; the hierarchical structure is deepened with `MLA`-labeled GQA attention.
*   **Layer 9 (Intent Analysis):** Decision mechanisms strengthen; the model begins to grasp user intent and the background of the question.
*   **Layer 10 (Second Liquid Contact):** **Critical Threshold.** The second `LiquidMixer` activates here; data's temporal memory and speed are dynamically refreshed during complex reasoning.
*   **Layer 11 (Strategic Decision):** Logic gaining fluidity is converted into strategic response parameters by `MoE` experts.
*   **Layer 12 (High-Level Meaning):** Information approaches the "wisdom" level; the tone, purpose, and target of the sentence become clear at this stage.
*   **Layer 13 (Response Construction):** The skeleton of the generated answer is built; `MLA`-labeled GQA attention focuses on the most critical points of the response.
*   **Layer 14 (Cultural Adaptation):** Technical details and cultural/idiomatic structures are injected into the model at this stage.
*   **Layer 15 (Pre-Final Analysis):** The final major audit and quality control layer before the response takes its final form.
*   **Layer 16 (Final Liquid Seal):** **Critical Threshold.** Final `LiquidMixer` engages; all information is transformed into a final "fluid intelligence" and temporal consistency is sealed before exit.
*   **Layer 17 (Final Block):** Final checks are performed; data processed via `RMSNorm` and `LM Head` is converted into word probabilities (logits) to be presented to the user.
```

```mermaid
graph TD
    subgraph "MertFormer Titan: 18-Layer Synaptic Flow"
        direction TB
        Phase4["💎 WISDOM - Layers 16-17<br/>Final Liquid Seal | Transformation into Language"]
        Phase3["🎭 REASONING - Layers 10-15<br/>Liquid Momentum | Strategic Logic & Cultural Adaptation"]
        Phase2["☁️ ABSTRACTION - Layers 3-9<br/>MoE Expert Distribution | Conceptual Depth & First Liquid Contact (L4)"]
        Phase1["🧱 FOUNDATION - Layers 0-2<br/>BitNet 1.58-bit | Grammar Setup & RMSNorm Stabilization"]

        Phase1 ==> Phase2 ==> Phase3 ==> Phase4
    end

    subgraph "The Engineering Heart of Every Layer (Block)"
        style BlockInner fill:#1a1a1a,stroke:#3fb1e3,stroke-width:2px
        BlockInner[Input] --> Norm1[RMSNorm]
        Norm1 --> MLA["MLA-labeled GQA Attention (Current Implementation)"]
        MLA --> Norm2[RMSNorm]
        Norm2 --> Router{"LiquidRouter (Temporal Selector)"}
        Router -- "Top-2 Experts" --> MoE["BitSwiGLU Experts"]
        Router -- "Dynamic Flow" --> Liquid["Liquid CfC Cell"]
        MoE --> Combine[Combined Signal]
        Liquid --> Combine
        Combine --> FinalNorm[Residual Add]
    end
```

```
MertFormer Titan (2.64B Parameters)
├── Embedding Layer (128256 vocab, Llama-3 tokenizer)
├── 18× Transformer Blocks
│   ├── RMSNorm (fused with torch.compile)
│   ├── MLA-labeled GQA attention (current implementation)
│   │   ├── BitLinear Projections (Q, K, V, O)
│   │   ├── RoPE (theta=100K, long-context ready)
│   │   ├── QK Normalization (stability)
│   │   ├── Flash Attention 2 (training mode)
│   │   └── KV Cache (inference mode)
│   ├── LiquidMixer (layers 4, 10, 16)
│   │   ├── LiquidCell (CfC core)
│   │   ├── Dynamic Tau (time-constant)
│   │   ├── CfC Update Rule
│   │   └── Residual + LayerNorm
│   ├── RMSNorm (fused with torch.compile)
│   └── Sparse MoE / Dense FFN
│       ├── LiquidRouter (Conv1d context-aware)
│       ├── 8× BitSwiGLU Experts
│       ├── Top-2 Routing
│       ├── Aux Loss (load balance + Z-loss + switch)
│       └── Emergency Jitter Boost
└── LM Head (BitLinear projection)
```

**Model Specifications:**
- **Hidden Size**: 2048
- **Intermediate Size**: 5632
- **Num Layers**: 18 (mobile-optimized)
- **Num Heads**: 16
- **Head Dim**: 128
- **Max Seq Length**: 4096 (expandable to 8K-16K)
- **Vocab Size**: 128256 (Llama-3 tokenizer)
- **RoPE Base**: 100,000 (long-context support)

---

<a id="performance"></a>
## 📊 Performance

**Claim policy:** Unless explicitly marked as measured, values in this section are targets/estimates and are not benchmark claim evidence.

### Performance Targets (Projected vs Baseline, Not Measured) — Training Speed (8x A100 80GB)
| Configuration | Time/Step | Throughput | GPU Utilization | VRAM Usage |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 2.0 sec | 64 tok/s | 47% | 38 GB |
| **v1.0 (Build 30 V2) (Optimized)** | **~1.2 sec** (Est.) | **~107 tok/s** (Est.) | **~95%** (Target) | **~76 GB** (Target) |
| **Speedup (Proj.)** | **+67%** | **+67%** | **+102%** | **+100%** |
**Aggregate Throughput Target (Projected): up to 11,000 tok/s.**
This value is a roadmap target for aggregate system capacity under a defined deployment profile; it is **not** a single-device measured benchmark result.
Operational meaning: higher concurrent session capacity, lower unit inference cost under load, and shorter queue times in multi-user scenarios.
*Note: Performance metrics are pre-training estimates based on architecture simulation. BitNet 1.58 inference now includes an optional low-bit kernel path; the Tensor Core path is **experimental** and opt-in (`MERTFORMER_TENSORCORE=1`). Energy/TOPS gains still require real device measurement. Kernel criticism applies to inference only; BitNet training exists as a separate layer, and low-bit inference is explicitly a roadmap item. **Training still runs on standard PyTorch matmul paths; the low-bit kernel does not accelerate training.** Furthermore, the **Residual Scaling Effect** maintains signal stability throughout 18 layers using the 1/√2 (1/sqrt(2)) factor, aiming to keep gradient flow stable even in the deepest layer.*

### Memory Footprint
| Component | FP32 | BF16 | BitNet 1.58 |
| :--- | :---: | :---: | :---: |
| Weights | 10.4 GB | 5.2 GB | **~0.65 GB (estimate)** |
| Optimizer (AdamW) | 41.6 GB | 20.8 GB | **20.8 GB** (distributed) |
| Activations | 40 GB | 20 GB | **12 GB** (w/ checkpointing) |
| **Total (per GPU)** | 92 GB | 46 GB | **33.45 GB** |
| **Total (8 GPUs)** | 736 GB | 368 GB | **267.6 GB** |
*Note: Values in this table are based on architectural comparisons and projections from similar models.*

### Inference (Samsung S25 - Estimated)
- ⏱️ **Latency**: ~50ms/token (NPU-optimized)
- 💾 **Memory**: <2GB RAM
- 🔋 **Power**: <3W (on-device)
- 🏎️ **Throughput**: ~45+ tokens/sec (Targeting 100+ with NPU Kernel Optimization)
*Note: Samsung S25 and Snapdragon 8 Elite NPU values are theoretical inference results based on manufacturer roadmaps and similar NPU architectures. Much higher speeds are possible due to the 1.58-bit architecture overcoming bandwidth bottlenecks.*

### 🔄 Universal Compatibility & System Requirements
Thanks to the BitNet architecture, MertFormer runs not just on flagships, but on **almost any device**:

| Device Class | Example Hardware | Expected Performance | Runtime Mode |
| :--- | :--- | :---: | :---: |
| **Tier 1 (Target)** | S25, iPhone 17, 8 Elite | **~100 tok/s** | **NPU / Neural Engine** |
| **Tier 2 (Modern)** | S23/S24, Pixel 8, iPhone 14 | **~40-60 tok/s** | GPU / DSP |
| **Tier 3 (Entry)** | Galaxy A54, A34 | **~15-25 tok/s** | CPU (Optimized) |
| **Tier 4 (Legacy)** | Samsung M51 (Snapdragon 730G) | **~12 tok/s** | CPU (BitNet) |

**Minimum Requirements:**
- **RAM**: 2GB (Target; uses ~0.65GB VRAM estimate)
- **Storage**: 2GB free space
- **OS**: Android 10+ / iOS 15+ / Windows / macOS / Linux

---

<a id="quick-start"></a>
## 🚀 Quick Start

### Baseline (Review-Ready)
- Python **3.11** (see `repro/python.md`)
- Offline-first runtime is default (`TITAN_OFFLINE=1`): no HF/WandB logins or dataset downloads unless explicitly enabled.

### Installation (Recommended)
Creates/updates `.titan-venv` using Python 3.11 and installs deps + dev tooling:

```bash
bash scripts/bootstrap_venv.sh
```

Optional demo deps (pygame):

```bash
bash scripts/bootstrap_venv.sh --demo
```

### Verify (Offline-First, Single Command)

```bash
bash scripts/verify_all.sh
```

### BitNet Kernel Benchmark (Standalone, Single File)

```bash
python3 scripts/bitnet_kernel_benchmark_standalone.py --shapes 2048x2048x2048,4096x2048x2048
```

This script is intentionally self-contained: kernel code, quantization path, reference path, and benchmark harness are all in one `.py` file.
It is also Jupyter/Colab-safe: extra runtime args (like `-f kernel.json`) are ignored automatically.
For default execution without CLI args, use:

```python
from scripts.bitnet_kernel_benchmark_standalone import run_default
run_default()
```

Performance note: this benchmark runs on a single selected device and does not aggregate multiple GPUs (for example, T4 x2 instances).

SDK-level verification and pilot reporting:

```bash
mertformer verify
mertformer pilot-report --out reports/pilot_report.json
```

<a id="sdk-api-quickstart"></a>
### 🧪 SDK API Quick Start (Developer)

```python
from mertformer_sdk.api import load_model, generate, benchmark

#- Use a trained checkpoint in production/pilot flows.
model, tokenizer, device = load_model(
    ckpt="checkpoints/my_trained.pt",
    strict_checkpoint=True,
)

text = generate(
    model,
    tokenizer,
    prompt="Summarize the verification gates in one paragraph.",
    max_new_tokens=96,
)
print(text)

results = benchmark(
    ckpt="checkpoints/my_trained.pt",
    out_dir="reports/benchmarks",
    samples=5,
    strict_checkpoint=True,
)
print(results)
```

```python
from mertformer_sdk.pilot import run_verify_all, build_pilot_report, write_pilot_report

verify_summary = run_verify_all(offline=True)
report = build_pilot_report(verify_summary=verify_summary)
write_pilot_report("reports/pilot_report_v1.json", report)
```

`strict_checkpoint=False` is available only for controlled random-weight diagnostics.

### LIVE DEMO (Snake Autoplayer)

```bash
bash scripts/bootstrap_venv.sh --demo
.titan-venv/bin/python snake_demo.py
```

### Drone SITL Proof Demo (Offline, No Physical Drone Required)

```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --runs 3 --steps 120
bash run.sh --sitl-demo
```

Artifacts are written to `reports/pilots/<pilot_id>/sitl_<timestamp>/`.
Default policy engine is `mertformer_liquidrouter` (BitLinear + LiquidRouter action proposal with fail-safe override).

Baseline comparison:

```bash
python3 scripts/drone_sitl_demo.py --pilot-id pilot_001 --policy-engine baseline
```

### Clean-Room Verification (Fresh Clone + New Venv)

```bash
bash run.sh --cleanroom-verify
```

### Preflight Only

```bash
TITAN_OFFLINE=1 bash run.sh --test
```

### Full Preflight Log (Latest Snapshot, 2026-03-07)

```text
2026-03-07 21:25:01,761 - [INFO] - ✈️ ============================================================
2026-03-07 21:25:01,761 - [INFO] - ✈️ 🚀 MERTFORMER TITAN - ULTIMATE PREFLIGHT JUDGE 🚀
2026-03-07 21:25:01,761 - [INFO] - ✈️ ============================================================
2026-03-07 21:25:01,761 - [WARNING] - ⚠️ .env file not found, skipping local load.
2026-03-07 21:25:01,761 - [INFO] - ✈️ STEP 1: SECRET SCAN...
2026-03-07 21:25:01,761 - [WARNING] - ⚠️ HF_TOKEN missing (offline mode): OK (online checks will be skipped).
2026-03-07 21:25:01,761 - [WARNING] - ⚠️ WANDB_API_KEY missing (offline mode): OK (WandB checks disabled).
2026-03-07 21:25:01,761 - [INFO] - ✅ Secrets check completed.
2026-03-07 21:25:01,761 - [INFO] - ✈️ STEP 2: ARCHITECTURAL AUDIT...
2026-03-07 21:25:01,761 - [INFO] - ✅ Layer configuration validated: No Liquid/MoE conflicts.
2026-03-07 21:25:01,761 - [INFO] - ✅ MLA Dimensions: Consistent (2048 features).
2026-03-07 21:25:01,761 - [INFO] - ✅ BitNet b1.58 logic: ACTIVE (Locked).
2026-03-07 21:25:01,761 - [INFO] - ✈️ STEP 3: DATA & DISTILLATION TEST...
2026-03-07 21:25:02,483 - [INFO] - ✈️ Offline mode: skipping Hugging Face connectivity checks.
2026-03-07 21:25:02,483 - [INFO] - 🛡️ Teacher Model mocked (Prevented 140GB download).
2026-03-07 21:25:02,483 - [INFO] - ⚙️  Pre-computing logits for preflight...
2026-03-07 21:25:02,722 - [INFO] - ✅ Saved Final Chunk 0: <REPO_ROOT>/temp_preflight_logits/preflight_test_part_0.pt
2026-03-07 21:25:02,723 - [INFO] - ✅ Distillation pipeline: PROVEN (Logits generated/saved).
2026-03-07 21:25:02,723 - [INFO] - ✈️ STEP 4: MOE GURU LEARNING TEST...
2026-03-07 21:25:02,724 - [INFO] - ✈️ 🏗️  CONFIG: Using 'Mini-Titan' (2 Layers, 256 Hidden, forced MoE/Liquid) for RAM safety.
2026-03-07 21:25:02,909 - [INFO] - ✈️ Checking Architectural Gradient Health...
2026-03-07 21:25:02,915 - [INFO] - ✅ MoE Learning: PROVEN (48 expert params receiving gradients).
2026-03-07 21:25:02,915 - [INFO] - ✅ Liquid Dynamics: PROVEN (7 liquid params receiving gradients).
2026-03-07 21:25:02,915 - [INFO] - ✈️ Shared Expert Grad: OK
2026-03-07 21:25:02,915 - [INFO] - ✅ MertFormer forward/backward pass verified.
2026-03-07 21:25:02,916 - [INFO] - ✅ OVERALL SYSTEM STATUS: 100% PROTECTED & READY.
2026-03-07 21:25:02,916 - [INFO] - ✈️ CLEANUP: Removing temporary files...
2026-03-07 21:25:02,916 - [INFO] - ✈️ Removed <REPO_ROOT>/temp_preflight_data
2026-03-07 21:25:02,940 - [INFO] - ✈️ Removed <REPO_ROOT>/temp_preflight_logits
2026-03-07 21:25:02,940 - [INFO] - ✅ CLEANUP: Done.
2026-03-07 21:25:02,940 - [INFO] - ✈️ Preflight Duration: 1.18s
2026-03-07 21:25:02,940 - [INFO] - ✈️ ============================================================
2026-03-07 21:25:02,940 - [INFO] - ✈️ RESULT: 🏆 ALL GREEN
2026-03-07 21:25:02,940 - [INFO] - ✈️ Full Report: <REPO_ROOT>/logs/preflight/titan_preflight.log
2026-03-07 21:25:02,940 - [INFO] - ✈️ ============================================================
```

Historical snapshot (2026-02-10) is preserved in git history for audit continuity.

### Training (Online / Training Hardware)

```bash
#- Explicitly enable online mode + (optional) WandB + installs
TITAN_OFFLINE=0 TITAN_WANDB=1 TITAN_INSTALL=1 bash run.sh
```

Notes:
- Online mode requires `HF_TOKEN`. WandB is optional (set `TITAN_WANDB=0`).
- Dependency installs are opt-in via `TITAN_INSTALL=1` (recommended to install once via bootstrap).

### Operator Mode Gate
Run the single-entry safety and readiness suite (safe mode by default):

```bash
TITAN_OFFLINE=1 .titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl
#- Use --full on training hardware
```

### Operator Mode Checklist (Evidence-Backed)
The following items are implemented and tied to concrete files/logs:

- Phase -1: Safety & Failure Budget
- Auto-Kill NaN Injection: `scripts/nan_kill_test.py`
- Failure Budget (Pivot/Debug trigger): `orchestrator/failure_budget.py`
- Checkpoint Restore Drill: `scripts/checkpoint_restore_drill.py`
- Phase 0: Infrastructure & Reality Gates
- Reproducibility Stamp (git/config/seed/datasets): `scripts/operator_mode_gate.py`, `utils/logger.py`
- Overfit Gate (1MB): `scripts/overfit_gate.py` (safe mode run completed; full 1MB gate runs on training hardware with `--full`)
- Observability (grad norms/router entropy/VRAM): `orchestrator/telemetry.py`
- Golden Sample Eval (50 prompts): `datasets/golden_samples.jsonl`, `scripts/golden_eval.py`
- Phase 1: Telemetry-Driven Execution
- Expected vs Actual tracking scaffold: `orchestrator/telemetry.py`
- Master Training (2.64B design target): execution on training hardware (not run locally)
- Internal Truth Benchmarks (HumanEval/MBPP): `scripts/benchmarks_internal.py`
- Phase 2: Asset Stack
- Snake proof video generator: `.titan-venv/bin/python snake_demo.py --headless --record assets/snake_demo_proof.mp4 --record-seconds 30`
- One-Pager / Technical Snapshot: `reports/one_pager.md`, `reports/technical_snapshot.md`, `PITCH.md`
- Founders Hub Application Draft: `reports/founders_hub_application.md`
- Phase 3: Future Horizons
- White Paper: `WHITE_PAPER_LIQUIDROUTER.md`
- Verification Plan
- Sanity Drills: `scripts/checkpoint_restore_drill.py`, `scripts/failure_budget_drill.py`

Operator-mode artifacts:
- Written under `logs/operator_mode/` (gitignored by default; artifacts are not committed).
- The script prints a JSON summary to stdout; use that as the review attachment.

### 🛡️ Diagnostic Excellence (Pre-Flight)
Run preflight (offline-first):

```bash
TITAN_OFFLINE=1 .titan-venv/bin/python scripts/titan_preflight.py
#- or:
TITAN_OFFLINE=1 bash run.sh --test
```

What it verifies:
- Secrets check (never prints token fragments; offline mode tolerates missing secrets unless `TITAN_PREFLIGHT_REQUIRE_SECRETS=1`)
- Architecture audit (cfg + MLA dims)
- Distillation pipeline dry-run (teacher mocked; temporary logits; cleanup)
- MoE/Liquid gradient health

Artifacts:
- `logs/preflight/titan_preflight.log` (gitignored; generated artifact)

---

### 🧾 Unified Logbook
All logs can be aggregated into a single artifact file: `logs/ALL_LOGS.jsonl` (gitignored).

Build or append with:

```bash
.titan-venv/bin/python scripts/logbook_build.py --append
```

This file includes source metadata for every imported log line and is designed for audit-grade traceability.

---

### 💻 Interactive Terminal Simulation
The following block demonstrates how a MertFormer Agent analyzes and resolves a complex failure:

```bash
[TITAN-ORCHESTRATOR] ⚡ Agent 'Architect' authorized...
[ARCHITECT] 🔍 Analyzing: MLA Layer-4 dimension mismatch.
[ARCHITECT] 💡 Root cause found: GQA Repetition factor mismatch in Mini-Titan config.
[TITAN-SEC] 🛡️ Security Audit: Patch safe. Signature: 0x88AF
[ARCHITECT] 🛠️  Applied Patch: cfg.num_kv_heads = 2
[TITAN-ORCHESTRATOR] ✅ Issue resolved. Preflight status: 🏆 ALL GREEN
```

---

<a id="troubleshooting"></a>
## 🛠️ Troubleshooting

| Symptom | Likely Cause | Action |
| :--- | :--- | :--- |
| `run.sh` refuses to start training | `TITAN_OFFLINE=1` (offline-first default) | Run with `TITAN_OFFLINE=0` and required credentials. |
| `HF_TOKEN is missing` | Online mode active without token | Set `HF_TOKEN` in environment or `.env`, or switch to offline mode. |
| `WANDB_API_KEY missing` warning | `TITAN_WANDB=1` without key | Set `WANDB_API_KEY` or disable WandB with `TITAN_WANDB=0`. |
| `Checkpoint not found` | `strict_checkpoint=True` with missing file | Use a valid checkpoint path; only use `strict_checkpoint=False` for controlled diagnostics. |
| `verify_all.sh` fails | One or more quality gates failed | Re-run `python3 -m pytest -q`, `ruff check`, and `bash scripts/verify_all.sh`; inspect `logs/preflight/titan_preflight.log`. |
| Unexpected low-bit/Tensor Core behavior | Experimental kernel path enabled | Treat `MERTFORMER_LOWBIT_KERNEL=1` and `MERTFORMER_TENSORCORE=1` as opt-in experiments and keep baseline path for production gating. |

---

<a id="training"></a>
## 🎓 Training

### Training Configuration

**File**: [`config/config.py`](config/config.py)

Key hyperparameters:
```python
#- Model Architecture
hidden_size = 2048
num_layers = 18
num_heads = 16
intermediate_size = 5632

#- Training
learning_rate = 1.5e-3
max_steps = 45000
warmup_steps = 3000
batch_size = 128  # Global (auto-configured per GPU)
grad_clip = 2.0

#- Distillation
teacher_model = "meta-llama/Llama-3.3-70B-Instruct"
distill_alpha = 0.8  # Dynamic (0.8 → 0.15)
teacher_temp = 1.0

#- Optimizations
use_torch_compile = False
torch_compile_mode = "max-autotune"
use_gradient_checkpointing = True

#- Safety
early_stop_patience = 5
liquid_warmup_steps = 10000
liquid_spike_threshold = 5.0
```

### Environment Variables (Operational Controls)

| Variable | Default | Scope | Purpose |
| :--- | :--- | :--- | :--- |
| `TITAN_OFFLINE` | `1` | `run.sh` / runtime | Offline-first mode; blocks online-dependent steps unless set to `0`. |
| `TITAN_WANDB` | Auto (`0` offline, `1` online) | `run.sh` / tracking | Enables or disables WandB flow depending on mode. |
| `TITAN_INSTALL` | `0` | `run.sh` | Installs dependencies when explicitly set to `1`. |
| `TITAN_PYTHON` | unset | launcher | Forces a specific Python interpreter path. |
| `TITAN_BOOTSTRAP` | `1` | launcher | Auto-bootstraps `.titan-venv` if local venv is missing. |
| `MERTFORMER_TENSORCORE` | unset/`0` | kernel path | Experimental Tensor Core low-bit path (opt-in). |
| `MERTFORMER_LOWBIT_KERNEL` | unset/`0` | kernel path | Enables experimental low-bit inference kernel path (opt-in). |
| `HF_TOKEN` | unset | online ops | Required for authenticated online dataset/model operations. |
| `WANDB_API_KEY` | unset | tracking | Required only when WandB is enabled in online mode. |

### Curriculum Learning (4 Core Stages + 1 Tool-Use/API Phase)

| Stage | Steps | Focus | Dataset Size |
| :--- | :---: | :--- | :--- |
| **1. Logic & Reasoning** | 0-42% | Math, coding, logic | 42% of corpus |
| **2. World Knowledge** | 42-72% | Facts, history, science | 30% of corpus |
| **3. Language (TR)** | 72-80% | Grammar, fluency, culture | 8% of corpus |
| **4. Soul (Identity)** | 80-88% | Personality, instruction | 8% of corpus |
| **5. Tool Use** | 88-100% | Function calling, APIs | 12% of corpus |

**Total Tokens**: ~23.6 Billion (high-quality, KD-focused)
*Note: Distillation boosts effective learning per token, but it does not increase raw token count.*

This curriculum order and token budget are designed to be **sufficient for a strong general foundation**.
For **niche or proprietary domains**, we recommend **targeted fine-tuning** on domain data to maximize specialization.

<a id="training-strategy-baseline-v28"></a>
### Training Strategy (Baseline -> v28, Claim-Safe)

No emergency architecture change is required before baseline training.
However, the following items are high-impact quality multipliers for the first tuning cycle.

**Recommended v28 tuning items (post-baseline evidence):**
1. Increase Stage 4 (`Soul/Identity`) influence when identity drift is observed (ratio increase or controlled oversampling).
2. Add a model-specific self-identity dataset to reinforce role, boundaries, and mission tone.
3. Keep SFT as baseline and move DPO/RLHF into a post-SFT alignment track.
4. Increase effective token budget (samples and/or epochs) if convergence indicates under-training.
5. Inject small custom tool/orchestrator examples into Stage 5 for better tool-use grounding.

**Execution order (operational):**
1. Run baseline training unchanged:
   `cd \"$(git rev-parse --show-toplevel)\" && TITAN_OFFLINE=0 TITAN_INSTALL=1 bash run.sh`
2. Produce first checkpoint + first benchmark evidence (reference baseline).
3. Apply the v28 tuning bundle in one controlled pass.
4. Compare baseline vs v28 with A/B evaluation and keep the measured winner.

All points above are claim-safe and conditional until empirically validated.

### Monitoring

Training metrics are logged to:
- 📈 **WandB**: Real-time dashboards (loss, grad norm, MoE health, etc.)
- 📄 **CSV**: `logs/run_*.csv` (artifact; gitignored)
- 📋 **JSONL**: `logs/run_*.jsonl` (artifact; gitignored)
- 🧾 **Unified logbook**: `logs/ALL_LOGS.jsonl` (artifact; gitignored)
- 💻 **Console**: Step-by-step progress

Note: By policy, `logs/` contains **artifacts only** and is not committed (except `logs/README.md`).

---

<a id="deployment"></a>
## 📱 Deployment

### ONNX Export

```bash
python scripts/mobile_export.py
```

Generates:
- `checkpoints/mertformer_titan_prod/titan_s25_fp32.onnx` (Dynamic axes)
- `checkpoints/mertformer_titan_prod/titan_s25_int8_quantized.onnx`
- Optimized for Samsung S25 NPU
- INT8 quantization ready

### Inference

Use a trained checkpoint for deployment/runtime validation:

```python
from mertformer_sdk.api import load_model, generate

model, tokenizer, device = load_model(
    ckpt="checkpoints/my_trained.pt",
    strict_checkpoint=True,
)

response = generate(
    model,
    tokenizer,
    prompt="What is the meaning of life?",
    max_new_tokens=256,
    temperature=0.7,
)
print(response)
```

**Context limits**: default input limit is **4096 tokens** (`cfg.max_seq_len`). Output length is caller-defined; `scripts/chat.py` defaults to `--max_tokens=128`, and `scripts/benchmarks_internal.py` defaults to `--max-new-tokens=256`.

---

<a id="integration-targets"></a>
## 🔌 Integration Targets

This section lists realistic integration paths with explicit current status.

### Available (Current Repository Capability)

| Target | Integration Method | Status | Primary Paths |
| :--- | :--- | :---: | :--- |
| Local Offline Ops | CLI + gate execution (`verify`, `pilot-report`, `verify_all.sh`) | ✅ Available | `mertformer_sdk/cli.py`, `scripts/verify_all.sh`, `scripts/operator_mode_gate.py` |
| Python App Embedding | SDK import + direct API usage | ✅ Available | `mertformer_sdk/api.py`, `SDK_GUIDE.md` |
| Edge Export Pipeline | ONNX export path for edge/mobile deployment flow | ✅ Available | `scripts/mobile_export.py`, `mertformer_sdk/export.py`, `config/export/onnx_mobile.yaml` |
| Pilot Evidence Delivery | Report + logs + schema-based pilot bundle | ✅ Available | `reports/pilots/`, `interfaces/pilot_report_v1.schema.json` |
| SITL Demonstration Flow | Deterministic drone SITL proof protocol | ✅ Available (Demo) | `scripts/drone_sitl_demo.py`, `reports/drone_sitl_demo.md` |

### Planned / Optional (Not Claimed as Complete)

| Target | Scope | Status | Note |
| :--- | :--- | :---: | :--- |
| Fine-tuning | Domain specialization after base readiness | 🟡 Planned / Optional | Claim-ready quality requires real domain data + compute + validation runs. |
| Coordinated Multi-Agent Runtime | Swarm/role-based orchestrated workflows | 🟡 Planned / Target Architecture | Implemented modules are partial/experimental under `orchestrator/`; production orchestration requires additional validation. |
| Expanded On-Prem Connectors | Environment-specific enterprise adapters | 🟡 Optional | Implement only per customer integration contract and security policy. |

---

<a id="benchmarks"></a>
## 🏆 Benchmarks
**Status: Pre-Training Projection (Not Eligible for Claim)**
*All metrics below are targets/estimates and require empirical validation after a full training run with a real checkpoint.*

### Benchmark Placeholders (To Be Measured)
| Benchmark | Status | Notes |
| :--- | :---: | :--- |
| linkedin_sweetspot (35K steps) | loss: 0.8368, exact_match: 2.5%, division: 27.3% | run_20260318_144125 — loss gate ✅ speed gate ✅ |
| HumanEval | TBD | Will be reported after a trained checkpoint. |

### Comparison with Similar Models

| Model | Parameters | Quantization | Mobile-Ready | On-Device | Turkish Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MertFormer Titan** | **2.64B** | **1.58-bit** | ✅ | ✅ | ✅ |
| Llama-3.2-3B | 3.0B | BF16 | ❌ | ❌ | Partial |
| Phi-3-mini | 3.8B | FP16 | ❌ | ❌ | ❌ |
| Gemma-2B | 2.0B | BF16 | ❌ | ❌ | ❌ |

### Performance Metrics (Estimated)

| Task | MertFormer Titan | Llama-3.2-3B | Phi-3-mini |
| :--- | :---: | :---: | :---: |
| **MMLU** | ~55% | 63% | 69% |
| **HellaSwag** | ~70% | 72% | 75% |
| **TruthfulQA** | ~45% | 50% | 55% |
| **Turkish NLU** | **~65%** | 45% | 30% |

*Note: Benchmarks will be updated after training completion*

---

<a id="turkish-vision"></a>
## 🇹🇷 Turkish Vision & Digital Sovereignty

### Why It Matters for Turkish Digital Sovereignty

MertFormer Titan is a critical step toward Türkiye's digital sovereignty. Today, AI runs on cloud servers owned by a few big companies (OpenAI, Google, Meta), which means your data is handled by them.

**The MertFormer Titan difference:**
- ✅ **100% On-Device**: Data is processed on-device; no cloud dependency for inference.
- ✅ **Turkish Optimization**: Training focus for Turkish language and culture.
- ✅ **National Technology**: Built locally with governance and licensing defined in `LICENSE`.
- ✅ **Independence**: Designed to operate without always-on internet.

### Vision: Digital Independence

> "Tohumu ektik, şimdi ormanı izleme vakti."
> "We planted the seed; now it's time to watch the forest."

MertFormer Titan is not just an AI model; it is a digital sovereignty manifesto:

1. **Data Sovereignty**: Citizen data can remain on-device and within local jurisdiction.
2. **Technology Independence**: Reduced reliance on foreign cloud providers.
3. **Cultural Preservation**: Turkish language and culture represented in AI.
4. **Economic Efficiency**: Lower cloud costs through edge inference.

### Turkish Corpus (Post-Build-27 Roadmap)

Planned Turkish data sources:
- **Wikipedia TR**: ~500K articles
- **Turkish News**: News archives
- **Literature**: Turkish literary classics
- **Social Media**: Filtered Turkish social corpus
- **Government**: Public official documents

**Target**: 30%+ Turkish performance uplift after training.

---

<a id="faq"></a>
## ❓ FAQ

### Q: Why 2.64B parameters? Could it be larger?

**A**: 2.64B is the **current design target (Build 30 V2) for mobile-class deployment**:
- Fits Samsung S25-class memory budgets (12GB RAM)
- ~0.65GB weights with BitNet (estimate)
- Strong speed/quality balance
- Larger models (7B+) are typically slower on mobile

### Q: Does BitNet 1.58-bit quantization hurt quality?

**A**: Quality impact is **task-dependent** and should be validated with full benchmarks:
- Mitigated with Knowledge Distillation
- Learns from a Llama-3.3-70B teacher
- Backed by production-focused low-bit research direction (Microsoft Research)

### Q: Why is Flash Attention 2 used only in training?

**A**: **KV cache compatibility constraints**:
- Flash Attention 2 currently does not match this inference KV-cache path
- Inference uses standard attention path
- Inference impact remains limited because the serving path is already optimized

### Q: What does NCCL tuning provide?

**A**: **Multi-GPU communication optimization**:
- Faster inter-GPU data transfer
- P2P paths activate when NVLink is available
- ~5-10% speedup potential on 8x GPU configurations

### Q: How long does training take?

**A**: On **8x A100 80GB** (projection, **not a benchmark claim**):
- Baseline: ~25 hours (45K steps x 2 sec/step)
- v1.0 (Build 30 V2) optimized: **estimate only** (45K steps; wall time depends on throughput and hardware; measured after run)
- ~10 hours estimated savings

### Q: Does it really run on Samsung S25?

**A**: **Theoretically yes** under the current roadmap:
- ONNX export path is ready
- NPU optimization is planned
- Real-device validation is a Post-Build-27 roadmap item
- Real-device performance measurements are still pending

### Q: Is the low-bit kernel production-ready?

**A**: It is an **experimental reference kernel** (correctness-first):
- BitNet training path remains a separate, existing layer
- Low-bit inference path is **opt-in**
- Tensor Core path is **experimental** (`MERTFORMER_TENSORCORE=1`)
- No speed/energy claim is made without real profiling/measurement

### Q: Is there a Turkish tokenizer?

**A**: **Opt-in** (disabled by default):
- `use_tr_tokenizer=false` (default)
- Downloadable via `scripts/download_tr_tokenizer.py`
- A risk-controlled PoC is recommended for distillation compatibility

### Q: What is the difference between Pilot-ready and Claim-ready?

**A**:
- **Pilot-ready** means gates, safety flow, and operational docs are in place for controlled demonstrations.
- **Claim-ready** requires a trained checkpoint plus reproducible benchmark evidence and measurement logs.

### Q: Which environment variables are mandatory in offline vs online mode?

**A**:
- **Offline mode (`TITAN_OFFLINE=1`)**: no external credential is mandatory for baseline verification flow.
- **Online mode (`TITAN_OFFLINE=0`)**: `HF_TOKEN` is required for authenticated online dataset/model operations.
- `WANDB_API_KEY` is required only if `TITAN_WANDB=1`.

---

### SOP Output Artifacts

- `reports/one_command_full_sop_summary.md` — Single-document summary of the end-to-end one-command SOP run.
- `reports/one_command_full_sop.log` — Raw full execution log for the same run.
- Note: both files are refreshed/overwritten on each new full SOP run.

### Latest SOP Snapshot (Copied from `reports/one_command_full_sop_summary.md`)

- Source of truth: `reports/one_command_full_sop_summary.md` (refreshed on each full SOP run)
- Included checks: `pytest`, `md_quality_all`, `linkcheck_all`, `unicode_path_guard`, `duplicate_zip_guard`, `clean_runtime_artifacts_check`, `zip_denylist_audit`, `secret_scan`
- Latest package hash is recorded in the summary file under `release_zip_sha256`
- Raw log: `reports/one_command_full_sop.log`

<a id="project-structure"></a>
## 📂 Project Structure

### Repository Control Map

- `Core System`: `config/`, `layers/`, `model/`, `train/`, `utils/`
- `SDK & Runtime`: `mertformer_sdk/`, `scripts/`, `run.sh`
- `Data & Evidence`: `datasets/`, `reports/`, `logs/`, `interfaces/`
- `Research & Extensions`: `ablations/`, `experiments/`, `orchestrator/`, `economics/`, `limits/`

### Canonical Layout (Build 30 V2)

```text
mertformer-titan-core/  # project root (git ls-files inventory)
├── .github/  # directory
│   ├── workflows/  # directory
│   │   └── ci.yml  # YAML configuration file
│   └── CODEOWNERS  # artifact
├── ablations/  # directory
│   ├── bitlinear_off/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── dense_only/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── no_liquid/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── no_moe/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── results.md  # documentation/report file
│   └── results_TR.md  # Turkish document counterpart
├── apps/  # directory
│   └── chess_gui/  # directory
│       ├── checkpoints/  # directory
│       │   └── README.md  # primary documentation (EN)
│       ├── logs/  # directory
│       │   └── README.md  # primary documentation (EN)
│       ├── .gitignore  # git ignore policy
│       ├── README.md  # primary documentation (EN)
│       ├── launch_mertformer_chess_gui.command  # artifact
│       └── play_mertformer_chess_web.py  # Python module/script (module for play mertformer chess web)
├── artifacts/  # directory
│   └── mertformer_release.zip.sha256  # artifact checksum
├── assets/  # directory
│   ├── sources/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── header.png  # media asset
│   ├── snake_demo_preview.gif  # media asset
│   ├── snake_demo_proof.mp4  # media asset
│   └── synaptic_map.png  # media asset
├── config/  # directory
│   ├── export/  # directory
│   │   └── onnx_mobile.yaml  # YAML configuration file
│   ├── model/  # directory
│   │   ├── mertformer_max_arch.yaml  # YAML configuration file
│   │   ├── mertformer_moe.yaml  # YAML configuration file
│   │   └── mertformer_small.yaml  # YAML configuration file
│   ├── train/  # directory
│   │   ├── finetune.yaml  # YAML configuration file
│   │   └── pretrain.yaml  # YAML configuration file
│   ├── __init__.py  # Python module/script (config package initializer and exports)
│   ├── base.yaml  # YAML configuration file
│   └── config.py  # Python module/script (runtime configuration model and validation helpers)
├── datasets/  # directory
│   ├── INTERNAL_POLICY.md  # documentation/report file
│   ├── INTERNAL_POLICY_TR.md  # Turkish document counterpart
│   ├── LICENSES.md  # documentation/report file
│   ├── LICENSES_TR.md  # Turkish document counterpart
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── SOURCES.md  # documentation/report file
│   ├── SOURCES_TR.md  # Turkish document counterpart
│   ├── filters.yaml  # YAML configuration file
│   ├── golden_assertions.jsonl  # JSONL data/log artifact
│   ├── golden_samples.jsonl  # JSONL data/log artifact
│   ├── hashes.json  # JSON data artifact
│   ├── inventory.json  # JSON data artifact
│   ├── inventory.md  # documentation/report file
│   ├── inventory_TR.md  # Turkish document counterpart
│   └── validation.jsonl  # JSONL data/log artifact
├── docs/  # directory
│   ├── CHAIN_MAP.md  # documentation/report file
│   └── CHAIN_MAP_TR.md  # Turkish document counterpart
├── economics/  # directory
│   ├── cost_model.md  # documentation/report file
│   ├── cost_model_TR.md  # Turkish document counterpart
│   ├── efficiency_report.md  # documentation/report file
│   ├── efficiency_report_TR.md  # Turkish document counterpart
│   └── flops_estimator.py  # Python module/script (module for flops estimator)
├── eval/  # directory
│   ├── agentic_suite.py  # Python module/script (evaluation routine for agentic suite)
│   ├── generalization_suite.py  # Python module/script (evaluation routine for generalization suite)
│   ├── golden.py  # Python module/script (evaluation routine for golden)
│   ├── gsm8k.py  # Python module/script (evaluation routine for gsm8k)
│   ├── humaneval.py  # Python module/script (evaluation routine for humaneval)
│   └── report_builder.py  # Python module/script (evaluation routine for report builder)
├── experiments/  # directory
│   └── exp_001_baseline/  # directory
│       ├── config.yaml  # YAML configuration file
│       ├── metrics.json  # JSON data artifact
│       ├── notes.md  # documentation/report file
│       └── notes_TR.md  # Turkish document counterpart
├── interfaces/  # directory
│   ├── backlog_item_v1.schema.json  # JSON schema artifact
│   ├── closure_57_matrix_v1.schema.json  # JSON schema artifact
│   ├── inference_contract.md  # documentation/report file
│   ├── inference_contract_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.schema.json  # JSON schema artifact
│   ├── pilot_report_v1.schema.json  # JSON schema artifact
│   ├── run_manifest_v1.schema.json  # JSON schema artifact
│   ├── tokenizer_spec.json  # JSON data artifact
│   └── workspace_hygiene_manifest_v1.schema.json  # JSON schema artifact
├── layers/  # directory
│   ├── __init__.py  # Python module/script (layers package initializer and exports)
│   ├── bitlinear.py  # Python module/script (BitLinear low-bit linear layer implementation)
│   ├── bitnet_patch.py  # Python module/script (BitNet quantization patch and runtime hooks)
│   ├── cognitive_extensions.py  # Python module/script (optional cognitive extension blocks)
│   ├── ffn.py  # Python module/script (feed-forward network blocks (dense and sparse paths))
│   ├── lifelong_safety.py  # Python module/script (lifelong safety guard layer)
│   ├── liquid.py  # Python module/script (liquid neural dynamics layers)
│   ├── mertformer_block.py  # Python module/script (core transformer block composition)
│   ├── mla.py  # Python module/script (multi-head latent attention implementation)
│   ├── moe.py  # Python module/script (mixture-of-experts routing and expert execution)
│   ├── qinn.py  # Python module/script (QINN experimental regulation layer (feature-flag))
│   └── world_model_head.py  # Python module/script (world-model auxiliary head)
├── limits/  # directory
│   ├── scaling_breakpoints.md  # documentation/report file
│   ├── scaling_breakpoints_TR.md  # Turkish document counterpart
│   └── stress_curves.png  # media asset
├── logs/  # directory
│   ├── README.md  # primary documentation (EN)
│   └── README_TR.md  # Turkish document counterpart
├── mertformer_sdk/  # directory
│   ├── kernels/  # directory
│   │   ├── cpp/  # directory
│   │   │   ├── __init__.py  # Python module/script (cpp package initializer and exports)
│   │   │   ├── bitnet_cpu.cpp  # C++ source file
│   │   │   └── loader.py  # Python module/script (SDK component for loader)
│   │   ├── metal/  # directory
│   │   │   ├── __init__.py  # Python module/script (metal package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── npu/  # directory
│   │   │   ├── __init__.py  # Python module/script (npu package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── vulkan/  # directory
│   │   │   ├── __init__.py  # Python module/script (vulkan package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── __init__.py  # Python module/script (kernels package initializer and exports)
│   │   ├── dispatcher.py  # Python module/script (SDK component for dispatcher)
│   │   ├── onnx_custom_op.py  # Python module/script (SDK component for onnx custom op)
│   │   └── triton_ternary.py  # Python module/script (SDK component for triton ternary)
│   ├── utils/  # directory
│   │   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   │   ├── bitpack.py  # Python module/script (SDK component for bitpack)
│   │   └── onnx_meta.py  # Python module/script (SDK component for onnx meta)
│   ├── __init__.py  # Python module/script (mertformer_sdk package initializer and exports)
│   ├── api.py  # Python module/script (SDK component for api)
│   ├── cli.py  # Python module/script (SDK component for cli)
│   ├── export.py  # Python module/script (SDK component for export)
│   ├── kpi.py  # Python module/script (SDK component for kpi)
│   └── pilot.py  # Python module/script (SDK component for pilot)
├── model/  # directory
│   ├── __init__.py  # Python module/script (model package initializer and exports)
│   └── transformers.py  # Python module/script (MertFormer model assembly and forward graph)
├── orchestrator/  # directory
│   ├── __init__.py  # Python module/script (orchestrator package initializer and exports)
│   ├── agent_registry.py  # Python module/script (orchestrator runtime component for agent registry)
│   ├── alignment_contracts.py  # Python module/script (orchestrator runtime component for alignment contracts)
│   ├── audio_sense.py  # Python module/script (orchestrator runtime component for audio sense)
│   ├── cognitive.py  # Python module/script (orchestrator runtime component for cognitive)
│   ├── cognitive_loop.py  # Python module/script (orchestrator runtime component for cognitive loop)
│   ├── compute_orchestrator.py  # Python module/script (orchestrator runtime component for compute orchestrator)
│   ├── core.py  # Python module/script (orchestrator runtime component for core)
│   ├── distillation_manager.py  # Python module/script (orchestrator runtime component for distillation manager)
│   ├── experience_store.py  # Python module/script (orchestrator runtime component for experience store)
│   ├── failure_budget.py  # Python module/script (orchestrator runtime component for failure budget)
│   ├── governance.py  # Python module/script (orchestrator runtime component for governance)
│   ├── hardware.py  # Python module/script (orchestrator runtime component for hardware)
│   ├── memory.py  # Python module/script (orchestrator runtime component for memory)
│   ├── paths.py  # Python module/script (orchestrator runtime component for paths)
│   ├── planner.py  # Python module/script (orchestrator runtime component for planner)
│   ├── reasoning_engine.py  # Python module/script (orchestrator runtime component for reasoning engine)
│   ├── self_audit.py  # Python module/script (orchestrator runtime component for self audit)
│   ├── self_improvement_guard.py  # Python module/script (orchestrator runtime component for self improvement guard)
│   ├── sense_engine.py  # Python module/script (orchestrator runtime component for sense engine)
│   ├── swarm_runtime.py  # Python module/script (orchestrator runtime component for swarm runtime)
│   ├── telemetry.py  # Python module/script (orchestrator runtime component for telemetry)
│   ├── tool_executor.py  # Python module/script (orchestrator runtime component for tool executor)
│   ├── tool_registry.py  # Python module/script (orchestrator runtime component for tool registry)
│   ├── verifier.py  # Python module/script (orchestrator runtime component for verifier)
│   └── web_sense.py  # Python module/script (orchestrator runtime component for web sense)
├── policy/  # directory
│   └── allow_deny_policy.yaml  # YAML configuration file
├── postmortems/  # directory
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── _template.md  # documentation/report file
│   ├── _template_TR.md  # Turkish document counterpart
│   ├── example_001.md  # documentation/report file
│   └── example_001_TR.md  # Turkish document counterpart
├── prompts/  # directory
│   ├── changelog.md  # documentation/report file
│   ├── changelog_TR.md  # Turkish document counterpart
│   └── system_v1.txt  # text artifact
├── registry/  # directory
│   └── mertformer_v0.1.json  # JSON data artifact
├── reports/  # directory
│   ├── benchmarks/  # directory
│   │   ├── linkedin_sweetspot/  # directory
│   │   │   ├── README.md  # primary documentation (EN)
│   │   │   ├── README_TR.md  # Turkish document counterpart
│   │   │   ├── run_20260318_144125_artifact_index.json  # JSON data artifact
│   │   │   ├── run_20260318_144125_compare.csv  # CSV data artifact
│   │   │   ├── run_20260318_144125_compare.json  # JSON data artifact
│   │   │   ├── run_20260318_144125_compare.md  # documentation/report file
│   │   │   ├── run_20260318_144125_health.txt  # text artifact
│   │   │   ├── run_20260318_144125_run_log.jsonl  # JSONL data/log artifact
│   │   │   ├── run_20260318_144125_step_metrics.csv  # CSV data artifact
│   │   │   ├── run_20260318_144125_summary.json  # JSON data artifact
│   │   │   └── zip_manifest.json  # JSON data artifact
│   │   ├── math_fastproof/  # directory
│   │   │   ├── README.md  # primary documentation (EN)
│   │   │   ├── README_TR.md  # Turkish document counterpart
│   │   │   ├── run_20260315_050133_artifact_index.json  # JSON data artifact
│   │   │   ├── run_20260315_050133_compare.csv  # CSV data artifact
│   │   │   ├── run_20260315_050133_compare.json  # JSON data artifact
│   │   │   ├── run_20260315_050133_compare.md  # documentation/report file
│   │   │   ├── run_20260315_050133_health.txt  # text artifact
│   │   │   ├── run_20260315_050133_run_log.jsonl  # JSONL data/log artifact
│   │   │   ├── run_20260315_050133_step_metrics.csv  # CSV data artifact
│   │   │   ├── run_20260315_050133_summary.json  # JSON data artifact
│   │   │   └── zip_manifest.json  # JSON data artifact
│   │   ├── text_understanding/  # directory
│   │   │   ├── README.md  # primary documentation (EN)
│   │   │   ├── README_TR.md  # Turkish document counterpart
│   │   │   ├── run_20260315_180151_artifact_index.json  # JSON data artifact
│   │   │   ├── run_20260315_180151_compare.csv  # CSV data artifact
│   │   │   ├── run_20260315_180151_compare.json  # JSON data artifact
│   │   │   ├── run_20260315_180151_compare.md  # documentation/report file
│   │   │   ├── run_20260315_180151_health.txt  # text artifact
│   │   │   ├── run_20260315_180151_run_log.jsonl  # JSONL data/log artifact
│   │   │   └── run_20260315_180151_summary.json  # JSON data artifact
│   │   ├── README.md  # primary documentation (EN)
│   │   ├── README_TR.md  # Turkish document counterpart
│   │   ├── agentic_suite_build30.json  # JSON data artifact
│   │   ├── generalization_suite_build30.json  # JSON data artifact
│   │   ├── internal_smoke_summary.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.csv  # CSV data artifact
│   │   ├── kaggle_compare_build30.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.md  # documentation/report file
│   │   ├── smoke_train_metrics.json  # JSON data artifact
│   │   ├── summary.json  # JSON data artifact
│   │   └── summary.md  # documentation/report file
│   ├── commercial_handover/  # directory
│   │   ├── contract_terms_checklist.md  # documentation/report file
│   │   ├── contract_terms_checklist_TR.md  # Turkish document counterpart
│   │   ├── handover_scope.md  # documentation/report file
│   │   ├── handover_scope_TR.md  # Turkish document counterpart
│   │   ├── known_issues.md  # documentation/report file
│   │   ├── known_issues_TR.md  # Turkish document counterpart
│   │   ├── ownership_and_role.md  # documentation/report file
│   │   ├── ownership_and_role_TR.md  # Turkish document counterpart
│   │   ├── sla_kpi_90_180.md  # documentation/report file
│   │   └── sla_kpi_90_180_TR.md  # Turkish document counterpart
│   ├── pilots/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── snapshots/  # directory
│   │   └── 2026-02-24/  # directory
│   │       ├── claim_matrix_v2_2026-02-24.json  # JSON data artifact
│   │       ├── commercial_scenarios_v1_2026-02-24.json  # JSON data artifact
│   │       ├── evidence_snapshot_2026-02-24.json  # JSON data artifact
│   │       ├── mertformer_master_decision_report_TR_2026-02-24.md  # documentation/report file
│   │       ├── readiness_scorecard_v1_2026-02-24.json  # JSON data artifact
│   │       ├── report_interface_schema_v1.json  # JSON schema artifact
│   │       └── web_validation_sources_2026-02-24.md  # documentation/report file
│   ├── architecture_honesty_audit.md  # documentation/report file
│   ├── artifacts_zip_denylist_audit.json  # JSON data artifact
│   ├── asset_stack.md  # documentation/report file
│   ├── asset_stack_TR.md  # Turkish document counterpart
│   ├── backup_restore_report.json  # JSON data artifact
│   ├── bench_cpp_report.json  # JSON data artifact
│   ├── bench_metal_report.json  # JSON data artifact
│   ├── bench_npu_report.json  # JSON data artifact
│   ├── bench_vulkan_report.json  # JSON data artifact
│   ├── bench_zero_copy_report.json  # JSON data artifact
│   ├── benchmark_compare_report.json  # JSON data artifact
│   ├── benchmark_compare_report.md  # documentation/report file
│   ├── benchmark_contract.md  # documentation/report file
│   ├── canonical_entrypoint.md  # documentation/report file
│   ├── cfc_moe_tolerance_report.json  # JSON data artifact
│   ├── checkpoint_contract.md  # documentation/report file
│   ├── checkpoint_hash_manifest.json  # JSON data artifact
│   ├── checkpoint_restore_report.json  # JSON data artifact
│   ├── chess_gui_onefile_sync_report.json  # JSON data artifact
│   ├── chess_gui_onefile_sync_report.md  # documentation/report file
│   ├── chess_onefile_extension_report.json  # JSON data artifact
│   ├── chess_onefile_extension_report.md  # documentation/report file
│   ├── chess_teaching_contract_report.json  # JSON data artifact
│   ├── chess_teaching_contract_report.md  # documentation/report file
│   ├── chess_training_readiness_report.json  # JSON data artifact
│   ├── chess_training_readiness_report.md  # documentation/report file
│   ├── claim_number_audit.json  # JSON data artifact
│   ├── claim_registry.json  # JSON data artifact
│   ├── cleanroom_verification.md  # documentation/report file
│   ├── cleanroom_verification_TR.md  # Turkish document counterpart
│   ├── cleanup_scoped_closure_junk_report.json  # JSON data artifact
│   ├── cli_smoke_log.md  # documentation/report file
│   ├── cli_smoke_log_TR.md  # Turkish document counterpart
│   ├── closure_57_matrix.json  # JSON data artifact
│   ├── closure_57_matrix.md  # documentation/report file
│   ├── closure_57_matrix_TR.md  # Turkish document counterpart
│   ├── closure_report_build30_v2.md  # documentation/report file
│   ├── closure_risk_register.md  # documentation/report file
│   ├── cloud_readiness_report.md  # documentation/report file
│   ├── code_truth_contract.md  # documentation/report file
│   ├── code_truth_delta_audit.json  # JSON data artifact
│   ├── code_truth_delta_audit.md  # documentation/report file
│   ├── codex_deep_audit_DE.md  # documentation/report file
│   ├── codex_deep_audit_DE_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_EN.md  # documentation/report file
│   ├── codex_deep_audit_EN_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_TR.md  # Turkish document counterpart
│   ├── commercial_handover_pack.md  # documentation/report file
│   ├── contamination_report_build30.md  # documentation/report file
│   ├── customer_ready_definition.md  # documentation/report file
│   ├── data_pipeline_contract.md  # documentation/report file
│   ├── data_pipeline_provenance.json  # JSON data artifact
│   ├── data_pipeline_token_probe.json  # JSON data artifact
│   ├── dataset_health.md  # documentation/report file
│   ├── dataset_health_TR.md  # Turkish document counterpart
│   ├── dataset_health_final.md  # documentation/report file
│   ├── dataset_lineage_final.json  # JSON data artifact
│   ├── dealroom_reference.json  # JSON data artifact
│   ├── demo_bundle.md  # documentation/report file
│   ├── demo_bundle_manifest.json  # JSON data artifact
│   ├── deprecated_surface_report.md  # documentation/report file
│   ├── determinism_report.json  # JSON data artifact
│   ├── differential_backend_report.json  # JSON data artifact
│   ├── doc_alignment_report.json  # JSON data artifact
│   ├── doc_alignment_report.md  # documentation/report file
│   ├── doc_ownership_matrix.md  # documentation/report file
│   ├── docs_dedup_canonical_list.md  # documentation/report file
│   ├── docs_packages_hash_manifest.json  # JSON data artifact
│   ├── drone_sitl_demo.md  # documentation/report file
│   ├── drone_sitl_demo_TR.md  # Turkish document counterpart
│   ├── dry_run_report.json  # JSON data artifact
│   ├── dry_run_report.md  # documentation/report file
│   ├── duplicate_source_of_truth_report.md  # documentation/report file
│   ├── duplicate_zip_guard_report.json  # JSON data artifact
│   ├── edge_readiness_plan.md  # documentation/report file
│   ├── efficiency_convergence_analysis.md  # documentation/report file
│   ├── efficiency_convergence_analysis_TR.md  # Turkish document counterpart
│   ├── energy_baseline.json  # JSON data artifact
│   ├── entrypoint_deprecation_map.md  # documentation/report file
│   ├── execution_trace.json  # JSON data artifact
│   ├── exit_code_standard.md  # documentation/report file
│   ├── expected_artifacts_list.md  # documentation/report file
│   ├── export_validation_report.json  # JSON data artifact
│   ├── fallback_policy_report.json  # JSON data artifact
│   ├── feature_flag_governance.md  # documentation/report file
│   ├── file_state_inventory.json  # JSON data artifact
│   ├── final_artifact_manifest.json  # JSON data artifact
│   ├── final_backlog_classification.json  # JSON data artifact
│   ├── final_backlog_classification.md  # documentation/report file
│   ├── final_backlog_coverage_diff.md  # documentation/report file
│   ├── final_backlog_missing_items.md  # documentation/report file
│   ├── final_checksum_manifest.json  # JSON data artifact
│   ├── final_commands.md  # documentation/report file
│   ├── final_evidence_pack.md  # documentation/report file
│   ├── final_freeze_manifest.json  # JSON data artifact
│   ├── final_freeze_manifest.md  # documentation/report file
│   ├── final_orchestrator_status.json  # JSON data artifact
│   ├── final_orchestrator_status.md  # documentation/report file
│   ├── final_repo_audit.md  # documentation/report file
│   ├── final_sync_matrix.md  # documentation/report file
│   ├── final_sync_matrix_TR.md  # Turkish document counterpart
│   ├── final_truth_constitution.md  # documentation/report file
│   ├── final_truth_matrix.md  # documentation/report file
│   ├── folder_drift_report.json  # JSON data artifact
│   ├── folder_structure_policy.md  # documentation/report file
│   ├── founders_hub_application.md  # documentation/report file
│   ├── founders_hub_application_TR.md  # Turkish document counterpart
│   ├── github_policy_report.json  # JSON data artifact
│   ├── go_nogo_signoff_onepager.md  # documentation/report file
│   ├── go_nogo_signoff_onepager_TR.md  # Turkish document counterpart
│   ├── go_status_matrix.md  # documentation/report file
│   ├── go_status_matrix_TR.md  # Turkish document counterpart
│   ├── gtm_master_plan.md  # documentation/report file
│   ├── hardening_bundle_summary.json  # JSON data artifact
│   ├── investable_definition.md  # documentation/report file
│   ├── investor_deck.pptx  # artifact
│   ├── investor_deck_TR.pptx  # artifact
│   ├── ip_licensing_split.md  # documentation/report file
│   ├── ip_licensing_split_TR.md  # Turkish document counterpart
│   ├── kernel_fuzz_report.json  # JSON data artifact
│   ├── kpi_contract_build30.md  # documentation/report file
│   ├── kpi_pack_v1.md  # documentation/report file
│   ├── kpi_pack_v1_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.json  # JSON data artifact
│   ├── latency_baseline.json  # JSON data artifact
│   ├── legal_cleanroom_signoff_internal.md  # documentation/report file
│   ├── legal_ip_pack.md  # documentation/report file
│   ├── license_gate_report.json  # JSON data artifact
│   ├── linkcheck_report.json  # JSON data artifact
│   ├── local_50step_proof_report.json  # JSON data artifact
│   ├── logger_contract.md  # documentation/report file
│   ├── logits_integrity_report.md  # documentation/report file
│   ├── master_closure_matrix.json  # JSON data artifact
│   ├── master_closure_matrix.md  # documentation/report file
│   ├── master_operating_plan.md  # documentation/report file
│   ├── md_lint_report.json  # JSON data artifact
│   ├── model_health.md  # documentation/report file
│   ├── model_health_TR.md  # Turkish document counterpart
│   ├── model_health_final.md  # documentation/report file
│   ├── one_command_full_sop.log  # text/log artifact (single-command end-to-end SOP raw log; overwritten each run)
│   ├── one_command_full_sop_summary.md  # documentation/report file (single-command end-to-end SOP summary; overwritten each run)
│   ├── one_pager.md  # documentation/report file
│   ├── one_pager_TR.md  # Turkish document counterpart
│   ├── owner_matrix.md  # documentation/report file
│   ├── ownership_proof_bundle.json  # JSON data artifact
│   ├── package_smoke_report.json  # JSON data artifact
│   ├── package_validation_report.md  # documentation/report file
│   ├── param_accounting_report.md  # documentation/report file
│   ├── phase2_carryover.md  # documentation/report file
│   ├── pilot_acceptance_signoff.md  # documentation/report file
│   ├── pilot_acceptance_signoff_TR.md  # Turkish document counterpart
│   ├── pilot_offer_packages.md  # documentation/report file
│   ├── pilot_offer_packages_TR.md  # Turkish document counterpart
│   ├── pilot_readiness_kit.md  # documentation/report file
│   ├── pilot_readiness_kit_TR.md  # Turkish document counterpart
│   ├── plot_contract.md  # documentation/report file
│   ├── poc_protocol.md  # documentation/report file
│   ├── poc_protocol_TR.md  # Turkish document counterpart
│   ├── post_45k_decision_tree.md  # documentation/report file
│   ├── post_train_automation_contract.md  # documentation/report file
│   ├── post_train_autorun_status.json  # JSON data artifact
│   ├── post_train_autorun_status.md  # documentation/report file
│   ├── post_train_state_machine.md  # documentation/report file
│   ├── presentation_readiness_final.md  # documentation/report file
│   ├── proje_zip_rebuild_manifest_v2.json  # JSON data artifact
│   ├── proje_zip_rebuild_manifest_v2.md  # documentation/report file
│   ├── ram_guard_report.json  # JSON data artifact
│   ├── release_closure_lock_report.json  # JSON data artifact
│   ├── release_closure_note.md  # documentation/report file
│   ├── release_snapshot.md  # documentation/report file
│   ├── release_snapshot_TR.md  # Turkish document counterpart
│   ├── rented_machine_bringup.md  # documentation/report file
│   ├── repo_external_handoff.md  # documentation/report file
│   ├── report_accuracy_audit.md  # documentation/report file
│   ├── report_accuracy_audit_TR.md  # Turkish document counterpart
│   ├── report_truth_matrix.md  # documentation/report file
│   ├── repro_build_report.json  # JSON data artifact
│   ├── resume_compat_report.json  # JSON data artifact
│   ├── review_checklist.md  # documentation/report file
│   ├── review_checklist_TR.md  # Turkish document counterpart
│   ├── run_contract.md  # documentation/report file
│   ├── runbook_validation_report.json  # JSON data artifact
│   ├── sales_funnel_90d.md  # documentation/report file
│   ├── sales_funnel_90d_TR.md  # Turkish document counterpart
│   ├── sanitizer_report.json  # JSON data artifact
│   ├── sbom.cdx.json  # JSON data artifact
│   ├── scoped_external_intake_matrix.json  # JSON data artifact
│   ├── scoped_external_intake_matrix.md  # documentation/report file
│   ├── security_compliance.md  # documentation/report file
│   ├── security_compliance_TR.md  # Turkish document counterpart
│   ├── smoke_run_report.json  # JSON data artifact
│   ├── snapshot_manifest_dealroom.json  # JSON data artifact
│   ├── snapshot_manifest_main.json  # JSON data artifact
│   ├── source_of_truth_map.md  # documentation/report file
│   ├── stale_script_report.md  # documentation/report file
│   ├── start_gate_operator_decision.json  # JSON data artifact
│   ├── start_gate_operator_decision.md  # documentation/report file
│   ├── start_gate_report.json  # JSON data artifact
│   ├── startup_selfcheck_report.json  # JSON data artifact
│   ├── static_analysis_report.json  # JSON data artifact
│   ├── strategic_value.md  # documentation/report file
│   ├── strategic_value_TR.md  # Turkish document counterpart
│   ├── surface_lifecycle_matrix.md  # documentation/report file
│   ├── system_hardware.md  # documentation/report file
│   ├── system_hardware_TR.md  # Turkish document counterpart
│   ├── system_stats.jsonl  # JSONL data/log artifact
│   ├── target_machine_handoff_manifest.json  # JSON data artifact
│   ├── target_machine_handoff_manifest.md  # documentation/report file
│   ├── teacher_decision_record.md  # documentation/report file
│   ├── teacher_output_license_assessment.md  # documentation/report file
│   ├── technical_snapshot.md  # documentation/report file
│   ├── technical_snapshot_TR.md  # Turkish document counterpart
│   ├── thermal_baseline.json  # JSON data artifact
│   ├── tokenizer_sync_final_report.md  # documentation/report file
│   ├── train_readiness_decision.json  # JSON data artifact
│   ├── train_readiness_decision.md  # documentation/report file
│   ├── training_readiness_manifest.json  # JSON data artifact
│   ├── unicode_path_guard_report.json  # JSON data artifact
│   ├── verified_matrix.md  # documentation/report file
│   ├── verified_matrix_TR.md  # Turkish document counterpart
│   ├── workspace_hygiene_manifest.json  # JSON data artifact
│   ├── workspace_hygiene_manifest.md  # documentation/report file
│   ├── xla_smoke_report.json  # JSON data artifact
│   ├── zip_audit_artifacts.json  # JSON data artifact
│   └── zip_audit_packages.json  # JSON data artifact
├── repro/  # directory
│   ├── accelerate_default.yaml  # YAML configuration file
│   ├── cuda.lock  # artifact
│   ├── env.lock  # artifact
│   ├── pip_freeze.txt  # text artifact
│   ├── python.md  # documentation/report file
│   ├── python_TR.md  # Turkish document counterpart
│   ├── seed_policy.md  # documentation/report file
│   └── seed_policy_TR.md  # Turkish document counterpart
├── scripts/  # directory
│   ├── reports/  # directory
│   │   ├── model_health.md  # documentation/report file
│   │   └── model_health_TR.md  # Turkish document counterpart
│   ├── runs/  # directory
│   │   └── preflight/  # directory
│   │       └── config_snapshot.json  # JSON data artifact
│   ├── tools/  # directory
│   │   ├── claim_number_audit.py  # Python module/script (automation script for claim number audit)
│   │   └── denylist_scan_zip.py  # Python module/script (automation script for denylist scan zip)
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── __init__.py  # Python module/script (scripts package initializer and exports)
│   ├── apply_github_policy.sh  # shell automation script
│   ├── benchmarks_internal.py  # Python module/script (automation script for benchmarks internal)
│   ├── bitnet_kernel_benchmark_standalone.py  # Python module/script (automation script for bitnet kernel benchmark standalone)
│   ├── bootstrap_venv.sh  # shell automation script
│   ├── build_artifacts_release_zip.sh  # shell automation script
│   ├── build_chess_5080_windows_delivery.py  # Python module/script (automation script for build chess 5080 windows delivery)
│   ├── build_chess_onefile_extension_report.py  # Python module/script (automation script for build chess onefile extension report)
│   ├── build_chess_teaching_contract_report.py  # Python module/script (automation script for build chess teaching contract report)
│   ├── build_chess_training_readiness_report.py  # Python module/script (automation script for build chess training readiness report)
│   ├── build_closure_governance_pack.py  # Python module/script (automation script for build closure governance pack)
│   ├── build_code_truth_audit.py  # Python module/script (automation script for build code truth audit)
│   ├── build_investor_deck.py  # Python module/script (automation script for build investor deck)
│   ├── build_master_closure_matrix.py  # Python module/script (automation script for build master closure matrix)
│   ├── build_max_closure_handoff.py  # Python module/script (automation script for build max closure handoff)
│   ├── build_offline_closure_pack.py  # Python module/script (automation script for build offline closure pack)
│   ├── build_scoped_external_intake_matrix.py  # Python module/script (automation script for build scoped external intake matrix)
│   ├── build_summary_pdf.py  # Python module/script (automation script for build summary pdf)
│   ├── build_target_machine_handoff_bundle.py  # Python module/script (automation script for build target machine handoff bundle)
│   ├── build_train_readiness_contract.py  # Python module/script (automation script for build train readiness contract)
│   ├── build_validation_set.py  # Python module/script (automation script for build validation set)
│   ├── build_workspace_hygiene_manifest.py  # Python module/script (automation script for build workspace hygiene manifest)
│   ├── cfc_moe_tolerance_check.py  # Python module/script (automation script for cfc moe tolerance check)
│   ├── chat.py  # Python module/script (automation script for chat)
│   ├── check_57_matrix.py  # Python module/script (automation script for check 57 matrix)
│   ├── check_doc_claim_consistency.py  # Python module/script (automation script for check doc claim consistency)
│   ├── check_tokenizer_sync.py  # Python module/script (automation script for check tokenizer sync)
│   ├── check_translation_pointer_policy.py  # Python module/script (automation script for check translation pointer policy)
│   ├── checkpoint_restore_drill.py  # Python module/script (automation script for checkpoint restore drill)
│   ├── chess_5080_onefile.py  # Python module/script (automation script for chess 5080 onefile)
│   ├── clean_runtime_artifacts.sh  # shell automation script
│   ├── cleanroom_verify.sh  # shell automation script
│   ├── cleanup_scoped_closure_junk.py  # Python module/script (automation script for cleanup scoped closure junk)
│   ├── data_pipeline.py  # Python module/script (automation script for data pipeline)
│   ├── dealroom_sync.py  # Python module/script (automation script for dealroom sync)
│   ├── docs_inventory.py  # Python module/script (markdown inventory and folder policy reporter)
│   ├── download_tr_tokenizer.py  # Python module/script (automation script for download tr tokenizer)
│   ├── drone_sitl_demo.py  # Python module/script (automation script for drone sitl demo)
│   ├── duplicate_zip_guard.py  # Python module/script (automation script for duplicate zip guard)
│   ├── eval.py  # Python module/script (automation script for eval)
│   ├── export_chess_5080_share.py  # Python module/script (automation script for export chess 5080 share)
│   ├── extract_dataset_refs.py  # Python module/script (automation script for extract dataset refs)
│   ├── failure_budget_drill.py  # Python module/script (automation script for failure budget drill)
│   ├── final_one_shot.sh  # shell automation script
│   ├── final_orchestrator.py  # Python module/script (automation script for final orchestrator)
│   ├── generate_bench_reports.py  # Python module/script (automation script for generate bench reports)
│   ├── generate_energy_baselines.py  # Python module/script (automation script for generate energy baselines)
│   ├── generate_sbom.py  # Python module/script (automation script for generate sbom)
│   ├── golden_eval.py  # Python module/script (automation script for golden eval)
│   ├── golden_score.py  # Python module/script (automation script for golden score)
│   ├── hardening_bundle.py  # Python module/script (automation script for hardening bundle)
│   ├── hash_manifest_to_json.py  # Python module/script (automation script for hash manifest to json)
│   ├── kaggle_onefile_demo_build30.py  # Python module/script (automation script for kaggle onefile demo build30)
│   ├── kaggle_onefile_demo_build30_colab_math_fastproof.py  # Python module/script (automation script for kaggle onefile demo build30 colab math fastproof)
│   ├── kaggle_onefile_demo_build30_text_understanding.py  # Python module/script (automation script for kaggle onefile demo build30 text understanding)
│   ├── kaggle_train_compare_build30.py  # Python module/script (automation script for kaggle train compare build30)
│   ├── linkcheck_gate.py  # Python module/script (automation script for linkcheck gate)
│   ├── logbook_build.py  # Python module/script (automation script for logbook build)
│   ├── mac_simulation.py  # Python module/script (automation script for mac simulation)
│   ├── mathfp_interactive_chat.py  # Python module/script (automation script for mathfp interactive chat)
│   ├── md_build30_sweep.py  # Python module/script (automation script for md build30 sweep)
│   ├── md_integrity_check.py  # Python module/script (automation script for md integrity check)
│   ├── md_quality_gate.py  # Python module/script (automation script for md quality gate)
│   ├── mini_titan_poc.py  # Python module/script (automation script for mini titan poc)
│   ├── mobile_export.py  # Python module/script (automation script for mobile export)
│   ├── nan_kill_test.py  # Python module/script (automation script for nan kill test)
│   ├── offline_4060_demo_train.py  # Python module/script (automation script for offline 4060 demo train)
│   ├── one_command_full_sop.sh  # shell automation script
│   ├── operator_mode_gate.py  # Python module/script (automation script for operator mode gate)
│   ├── overfit_gate.py  # Python module/script (automation script for overfit gate)
│   ├── plot_training_log.py  # Python module/script (automation script for plot training log)
│   ├── post_train_autorun.py  # Python module/script (automation script for post train autorun)
│   ├── ram_guard.py  # Python module/script (automation script for ram guard)
│   ├── record_dataset_hashes.py  # Python module/script (automation script for record dataset hashes)
│   ├── release_build30.sh  # shell automation script
│   ├── release_closure_lock.sh  # shell automation script
│   ├── repro_build_check.py  # Python module/script (automation script for repro build check)
│   ├── resume_compat_check.py  # Python module/script (automation script for resume compat check)
│   ├── run_and_clean_pycache.py  # Python module/script (run command + guaranteed post-run cache sweep; add --include-venv-caches for venv cache cleanup)
│   ├── scaling_audit_math.py  # Python module/script (automation script for scaling audit math)
│   ├── secret_scan.py  # Python module/script (automation script for secret scan)
│   ├── smart_runner.py  # Python module/script (automation script for smart runner)
│   ├── smoke_train_benchmark.py  # Python module/script (automation script for smoke train benchmark)
│   ├── start_gate.py  # Python module/script (automation script for start gate)
│   ├── sync_chess_gui_onefile.py  # Python module/script (automation script for sync chess gui onefile)
│   ├── sync_manifest.py  # Python module/script (release manifest and project-structure sync generator)
│   ├── test_onnx_export.py  # Python module/script (automation script for test onnx export)
│   ├── titan_onnx_stress_test.py  # Python module/script (automation script for titan onnx stress test)
│   ├── titan_preflight.py  # Python module/script (automation script for titan preflight)
│   ├── train_smoke.py  # Python module/script (automation script for train smoke)
│   ├── train_tpu_turbo.py  # Python module/script (automation script for train tpu turbo)
│   ├── unicode_path_guard.py  # Python module/script (automation script for unicode path guard)
│   ├── update_investor_deck.py  # Python module/script (automation script for update investor deck)
│   ├── update_system_hardware.py  # Python module/script (automation script for update system hardware)
│   ├── verify_all.sh  # shell automation script
│   ├── verify_datasets.py  # Python module/script (automation script for verify datasets)
│   ├── verify_onnx_local.py  # Python module/script (automation script for verify onnx local)
│   ├── version_checker.py  # Python module/script (automation script for version checker)
│   ├── write_cuda_lock.py  # Python module/script (automation script for write cuda lock)
│   ├── xray.py  # Python module/script (automation script for xray)
│   └── zip_denylist_audit.py  # Python module/script (automation script for zip denylist audit)
├── telemetry/  # directory
│   └── metrics_schema.json  # JSON schema artifact
├── tests/  # directory
│   ├── test_57_matrix_gate.py  # Python module/script (automated test module for 57 matrix gate)
│   ├── test_agi_cognitive.py  # Python module/script (automated test module for agi cognitive)
│   ├── test_architecture_integrity.py  # Python module/script (automated test module for architecture integrity)
│   ├── test_build_chess_5080_windows_delivery.py  # Python module/script (automated test module for build chess 5080 windows delivery)
│   ├── test_build_chess_onefile_extension_report.py  # Python module/script (automated test module for build chess onefile extension report)
│   ├── test_build_chess_teaching_contract_report.py  # Python module/script (automated test module for build chess teaching contract report)
│   ├── test_build_chess_training_readiness_report.py  # Python module/script (automated test module for build chess training readiness report)
│   ├── test_build_code_truth_audit.py  # Python module/script (automated test module for build code truth audit)
│   ├── test_build_max_closure_handoff.py  # Python module/script (automated test module for build max closure handoff)
│   ├── test_build_target_machine_handoff_bundle.py  # Python module/script (automated test module for build target machine handoff bundle)
│   ├── test_build_workspace_hygiene_manifest.py  # Python module/script (automated test module for build workspace hygiene manifest)
│   ├── test_check_doc_claim_consistency.py  # Python module/script (automated test module for check doc claim consistency)
│   ├── test_chess_5080_onefile.py  # Python module/script (automated test module for chess 5080 onefile)
│   ├── test_chess_gui_contract.py  # Python module/script (automated test module for chess gui contract)
│   ├── test_chess_onefile_curated_suites.py  # Python module/script (automated test module for chess onefile curated suites)
│   ├── test_cognitive_extensions.py  # Python module/script (automated test module for cognitive extensions)
│   ├── test_comprehensive.py  # Python module/script (automated test module for comprehensive)
│   ├── test_continual_adapter.py  # Python module/script (automated test module for continual adapter)
│   ├── test_cpp_kernel_loader.py  # Python module/script (automated test module for cpp kernel loader)
│   ├── test_dispatcher_extended.py  # Python module/script (automated test module for dispatcher extended)
│   ├── test_drone_sitl_demo.py  # Python module/script (automated test module for drone sitl demo)
│   ├── test_eval_suites.py  # Python module/script (automated test module for eval suites)
│   ├── test_export_chess_5080_share.py  # Python module/script (automated test module for export chess 5080 share)
│   ├── test_export_metadata.py  # Python module/script (automated test module for export metadata)
│   ├── test_final_orchestrator_cli.py  # Python module/script (automated test module for final orchestrator cli)
│   ├── test_kaggle_compare_script.py  # Python module/script (automated test module for kaggle compare script)
│   ├── test_kaggle_onefile_colab_math_fastproof.py  # Python module/script (automated test module for kaggle onefile colab math fastproof)
│   ├── test_kaggle_onefile_compile_guard.py  # Python module/script (automated test module for kaggle onefile compile guard)
│   ├── test_kaggle_onefile_config.py  # Python module/script (automated test module for kaggle onefile config)
│   ├── test_kaggle_onefile_feature_coverage.py  # Python module/script (automated test module for kaggle onefile feature coverage)
│   ├── test_kaggle_onefile_zero_shot_unseen.py  # Python module/script (automated test module for kaggle onefile zero shot unseen)
│   ├── test_kernel_dispatcher.py  # Python module/script (automated test module for kernel dispatcher)
│   ├── test_kernel_equivalence.py  # Python module/script (automated test module for kernel equivalence)
│   ├── test_kpi_report_cli.py  # Python module/script (automated test module for kpi report cli)
│   ├── test_lifelong_safety.py  # Python module/script (automated test module for lifelong safety)
│   ├── test_liquid_safeguard.py  # Python module/script (automated test module for liquid safeguard)
│   ├── test_mla_regressions.py  # Python module/script (automated test module for mla regressions)
│   ├── test_model.py  # Python module/script (automated test module for model)
│   ├── test_onnx_custom_op_contract.py  # Python module/script (automated test module for onnx custom op contract)
│   ├── test_onnx_export_path.py  # Python module/script (automated test module for onnx export path)
│   ├── test_onnx_metadata_hook.py  # Python module/script (automated test module for onnx metadata hook)
│   ├── test_orchestrator_swarm_runtime.py  # Python module/script (automated test module for orchestrator swarm runtime)
│   ├── test_post_train_autorun_cli.py  # Python module/script (automated test module for post train autorun cli)
│   ├── test_scoped_external_tools.py  # Python module/script (automated test module for scoped external tools)
│   ├── test_sdk_api.py  # Python module/script (automated test module for sdk api)
│   ├── test_sdk_pilot_cli.py  # Python module/script (automated test module for sdk pilot cli)
│   ├── test_start_gate.py  # Python module/script (automated test module for start gate)
│   ├── test_sync_chess_gui_onefile.py  # Python module/script (automated test module for sync chess gui onefile)
│   ├── test_titan_preflight_contract.py  # Python module/script (automated test module for titan preflight contract)
│   ├── test_train_loop_sanity.py  # Python module/script (automated test module for train loop sanity)
│   ├── test_triad_omega_api.py  # Python module/script (automated test module for triad omega api)
│   └── test_world_model_head.py  # Python module/script (automated test module for world model head)
├── tokenizer/  # directory
│   ├── tr/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── drift_report.md  # documentation/report file
│   ├── drift_report_TR.md  # Turkish document counterpart
│   ├── stats.md  # documentation/report file
│   ├── stats_TR.md  # Turkish document counterpart
│   └── tokenizer.json  # JSON data artifact
├── tools/  # directory
│   ├── contracts/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── sandbox/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── abuse_tests.md  # documentation/report file
│   └── abuse_tests_TR.md  # Turkish document counterpart
├── train/  # directory
│   ├── __init__.py  # Python module/script (train package initializer and exports)
│   ├── continual_adapter.py  # Python module/script (continual learning adapter path for training)
│   └── train.py  # Python module/script (main training loop entrypoint)
├── training_dynamics/  # directory
│   ├── cold_vs_warm.md  # documentation/report file
│   └── cold_vs_warm_TR.md  # Turkish document counterpart
├── utils/  # directory
│   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   ├── dataset_registry.py  # Python module/script (module for dataset registry)
│   ├── liquid_safeguard.py  # Python module/script (module for liquid safeguard)
│   ├── logger.py  # Python module/script (module for logger)
│   └── safety.py  # Python module/script (module for safety)
├── .gitignore  # git ignore policy
├── AGENTS.md  # documentation/report file
├── CHANGELOG.md  # documentation/report file
├── CHANGELOG_TR.md  # Turkish document counterpart
├── CHESS_5080_POC_INTERNAL.md  # documentation/report file
├── CHESS_5080_POC_INTERNAL_TR.md  # Turkish document counterpart
├── CITATION.cff  # citation metadata
├── CONTRIBUTING.md  # documentation/report file
├── CONTRIBUTING_TR.md  # Turkish document counterpart
├── DECISIONS.md  # documentation/report file
├── DECISIONS_TR.md  # Turkish document counterpart
├── Dockerfile  # container build baseline
├── IMPLEMENTATION_PLAN.md  # documentation/report file
├── IMPLEMENTATION_PLAN_TR.md  # Turkish document counterpart
├── INTERNAL_AGI_GAP.md  # documentation/report file
├── INTERNAL_AGI_GAP_TR.md  # Turkish document counterpart
├── LICENSE  # license terms (EN)
├── LICENSE_TR  # license terms (TR)
├── MISSION.md  # documentation/report file
├── MISSION_TR.md  # Turkish document counterpart
├── MODEL_CARD.md  # documentation/report file
├── MODEL_CARD_TR.md  # Turkish document counterpart
├── MODEL_LICENSE.md  # documentation/report file
├── MODEL_LICENSE_TR.md  # Turkish document counterpart
├── OFFLINE_4060_DEMO.md  # documentation/report file
├── PITCH.md  # documentation/report file
├── PITCH_TR.md  # Turkish document counterpart
├── README.md  # primary documentation (EN)
├── README_CHECKLIST.md  # documentation/report file
├── README_CHECKLIST_TR.md  # Turkish document counterpart
├── README_SUMMARY.md  # documentation/report file
├── README_SUMMARY.pdf  # artifact
├── README_SUMMARY_TR.md  # Turkish document counterpart
├── README_SUMMARY_TR.pdf  # artifact
├── README_TR.md  # Turkish document counterpart
├── SDK_GUIDE.md  # documentation/report file
├── SDK_GUIDE_TR.md  # Turkish document counterpart
├── SECURITY.md  # documentation/report file
├── SECURITY_TR.md  # Turkish document counterpart
├── TASK.md  # documentation/report file
├── TASK_TR.md  # Turkish document counterpart
├── TECHNICAL_REPORT.md  # documentation/report file
├── TECHNICAL_REPORT_TR.md  # Turkish document counterpart
├── TRAINING_PLAN.md  # documentation/report file
├── TRAINING_PLAN_TR.md  # Turkish document counterpart
├── TROUBLESHOOTING.md  # documentation/report file
├── TROUBLESHOOTING_TR.md  # Turkish document counterpart
├── USAGE_GUIDE.md  # documentation/report file
├── USAGE_GUIDE_TR.md  # Turkish document counterpart
├── USE_POLICY.md  # documentation/report file
├── USE_POLICY_TR.md  # Turkish document counterpart
├── V2_BACKLOG_SEED.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER_TR.md  # Turkish document counterpart
├── pyproject.toml  # project metadata
├── requirements.txt  # text artifact
├── run.sh  # shell automation script
├── snake_demo.py  # Python module/script (module for snake demo)
└── zero_touch_start.sh  # shell automation script
```

### Clickable Path Map

- `Core System`: [config/](config/), [layers/](layers/), [model/](model/), [train/](train/), [utils/](utils/)
- `SDK & Runtime`: [mertformer_sdk/](mertformer_sdk/), [scripts/](scripts/), [run.sh](run.sh)
- `Data & Evidence`: [datasets/](datasets/), [reports/](reports/), [logs/](logs/), [interfaces/](interfaces/)
- `Research & Extensions`: [ablations/](ablations/), [experiments/](experiments/), [orchestrator/](orchestrator/), [economics/](economics/), [limits/](limits/)
- `Primary Docs`: [README.md](README.md), [README_TR.md](README_TR.md), [USAGE_GUIDE.md](USAGE_GUIDE.md), [SDK_GUIDE.md](SDK_GUIDE.md)

### Maintenance Rule

- The control map above is the source of truth for navigation.
- The canonical layout is generated from tracked files (`git ls-files`) and updated during release closure.
- Detailed point-in-time inventories remain in `reports/final_sync_matrix.md` and `reports/release_snapshot.md`.
- Only existing paths should be referenced in README links.

---

<a id="license"></a>
## 📄 License

This project is **confidential and proprietary**. All rights are reserved by the **MertFormer AI Team**. Unauthorized copying, modification, or distribution is strictly prohibited. See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

- **Meta AI**: Llama-3.3-70B teacher model & tokenizer
- **Microsoft Research**: BitNet quantization research
- **Liquid AI**: Liquid Neural Networks (CfC) inspiration
- **Research Positioning**: MertFormer explores an orthogonal path to temporal intelligence by integrating liquid dynamics into MoE routing.
- **DeepSeek**: MLA literature inspiration (current implementation in this repo is MLA-labeled GQA)
- **HazyResearch / Stanford (Tri Dao et al.)**: Flash Attention 2
- **PyTorch**: Core training and inference framework
- **Triton**: Experimental low-bit kernel research
- **ONNX / ONNX Runtime**: Export and verification tooling
- **SentencePiece**: Tokenization tooling
- **Weights & Biases (WandB)**: Experiment tracking
- **NVIDIA**: Apex optimizations, NCCL
- **Hugging Face**: Transformers, Accelerate, Datasets libraries
- **Turkish AI Community**: Support and feedback

---

<a id="strategic-collaboration"></a>
## 🤝 Strategic Collaboration

MertFormer accepts collaboration in a controlled, evidence-first format.

**Can be shared (controlled scope):**
- Architecture and validation documents under `reports/`
- Pilot evidence bundles (`verify_all`, operator gates, `pilot_report_v1`)
- Integration requirements and deployment constraints

**Not shared without explicit legal controls:**
- Raw source redistribution rights
- Checkpoints/weights and secret-bearing artifacts
- Internal security procedures beyond approved scope

All commercial/partner engagement follows written agreement terms and confidentiality controls consistent with `LICENSE`.

---

## 📧 Contact

**Project**: MertFormer Titan (Onyx Storm)
**Version**: v1.0 (Build 30 V2, Pre-Training Baseline)
**Status**: 🟡 Pilot-Ready (training & benchmark claims pending)
**Developed in Türkiye**

---

## ✅ Sales-Ready Checklist (B2B Pilot Mode)

For paid pilots in pre-training stage, the minimum acceptance set is:

- `bash scripts/verify_all.sh` must pass in offline mode.
- Operator mode gate logs must be attached to pilot delivery.
- `mertformer pilot-report --out <json>` must be delivered as `pilot_report_v1`.
- If trained checkpoint is missing, benchmark status must stay `NOT ELIGIBLE FOR CLAIM`.
- Customer-side offline execution (`mertformer verify`) must be demonstrated.
- Commercial closure target: 2 paid pilot contracts or signed PoC LOI.

---

## 🛡️ Strategic Transparency & Roadmap

### Final Scope & Intent
This project intentionally concludes at the proof-of-system level. The goal is to demonstrate a complete and working autonomous reasoning stack under real-world constraints, not to claim a production-ready or certified platform. Architecture boundaries, safety behavior, real-time constraints, and failure modes are treated as first-class engineering concerns. Large-scale deployment, certification, and long-horizon field validation are explicitly out of scope for this release by design.

### ⚠️ Technical Risk Factors
*   **Performance Projection**: Mobile NPU metrics (<50ms/token) are currently architecture-modeled and will be empirically validated post-training.
*   **Deployment Kernel**: 1.58-bit ternary execution on mobile may require custom kernel optimization beyond standard ONNX runtimes for peak speed.
*   **MoE Stability**: `LiquidRouter` is a novel research contribution; its exact edge over classical routers will be benchmarked during full-scale training.

### 🗺️ Validation Roadmap
- [x] **Phase 0**: Architecture Simulation & Mathematical Verification
- [ ] **Phase 1**: Training Convergence & Distillation Health Check
- [ ] **Phase 2**: Multi-Domain Benchmarking (GSM8K, HumanEval, MMLU)
- [ ] **Phase 3**: Physical Device Performance Profiling (S25/M4)
- [ ] **Phase 4**: Low-Level Kernel Optimization (C++ / ENN / QNN, optional post-training track)

<a id="scalability-vision"></a>
### 📈 Scalability Vision (Claim-Safe)
Build 30 V2 is intentionally centered on **2.64B** validation and reproducible evidence gates.
Future **13B / 70B / 256B** exploration is treated as a conditional research track and is evaluated only after:
- trained-checkpoint evidence on 2.64B,
- reproducible benchmark outputs,
- hardware/cost feasibility review,
- security and compliance boundary checks.

### 🚫 What MertFormer Titan Is NOT
*   **Not a General Chatbot**: Optimized specifically for code orchestration and structural reasoning.
*   **Not a Cloud-Scale Infrastructure Competitor**: Designed for private, local execution rather than massive web-scale serving via data centers.
*   **Not a Legacy Transformer**: This is a non-standard synthesis of CfC, MLA-labeled GQA attention, and BitNet layers.

---

## 📜 Citation

```bibtex
@software{mertformer_titan_2026,
  author = {MertFormer AI Team},
  title = {MertFormer Titan: 1.58-bit Mobile-First LLM},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/latentcore/mertformer-titan-core}}
}
```

---

<div align="center">

**🚀 Built for the Future of On-Device AI 🚀**

*"The best AI is the one that respects your privacy."*

**"We planted the seed; now it's time to watch the forest."**

</div>

## Build30 V1 Final Closure Addendum (2026-02-28)

The one-file companion at `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py` was upgraded to a closure contract release track:

- Payload schemas upgraded: `build30_colab_math_fastproof_payload_v2`, `build30_colab_math_fastproof_compare_v2`, `kaggle_onefile_deep_build30_v6`.
- Strict run/config governance: schema validation, required-key fail-fast, unknown-key reject, override/source trace.
- Compile stall mitigation: compile default OFF, timeout fallback, compile/cudagraph guard telemetry.
- Evidence extensions: ownership proof, runtime fingerprint, redacted env snapshot, reproducible command string.
- Evaluation extensions: unseen-range zero-shot exact-match reporting and interpretability artifacts.
- Feature tracking: exhaustive `feature_coverage_matrix` with per-feature IDs and completeness percent.
