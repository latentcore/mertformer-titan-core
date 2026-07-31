# GOVERNANCE — policies, contracts, change control

Single index for the repo's governance surfaces (previously scattered across `reports/`).
Turkish: [GOVERNANCE_TR.md](GOVERNANCE_TR.md). Deliberate decisions: [DECISIONS.md](DECISIONS.md).

## Change control
- Commits scoped + reason-coded; CI gate name `verify` (see [REPRODUCE.md](REPRODUCE.md)).
- No-destruction policy: superseded/bulk content is **moved** (`archive/`, `private/`) or **gitignored** (generated), never hard-deleted — git history is the recovery surface.
- Behavior-changing fixes are deferred until after the first 45K run and documented in [DECISIONS.md](DECISIONS.md) / [BACKLOG.md](BACKLOG.md) so the run is not confounded.

## Repo policies (full text under `reports/`)
- `reports/automation_boundary_policy.md` — what automation may/may not do.
- `reports/change_control_sop.md` — change-control SOP.
- `reports/update_first_policy.md` — update-existing-before-adding-new.
- `reports/system_memory_policy.md` — durable-memory policy.
- `reports/folder_structure_policy.md` + `reports/repo_directory_contract.md` — layout rules.
- `reports/backlog_operating_contract.md` — backlog operating rules.
- `reports/code_truth_contract.md` — code-vs-doc truth alignment.
- `reports/support_maintenance_policy.md` — support/maintenance boundary.

## Contracts (machine-checked surfaces, full text under `reports/`)
- `run_contract.md`, `checkpoint_contract.md`, `data_pipeline_contract.md`, `logger_contract.md`, `plot_contract.md`, `benchmark_contract.md`, `kpi_contract_build30.md`, `post_train_automation_contract.md`.

## Privacy / visibility
- The main repo and the separate dealroom repo are **PRIVATE**. The only public surface is the README-derived Gist.
- **Updated (2026-07-30, Apache 2.0 relicensing pass):** README.md/README_TR.md now carry a
  short, deliberate "Hiring & Commercial Opportunities" section (role interest + a real
  contact address) as part of preparing the repo for public release — see [DECISIONS.md](DECISIONS.md).
  Deeper investor/pitch/outreach material (decks, one-pagers, sponsorship messaging) stays
  out of the technical repo, in `private/commercial/` and the separate dealroom repo.
