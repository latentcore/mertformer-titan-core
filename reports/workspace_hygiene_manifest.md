# Workspace Hygiene Manifest

Quarantine-first workspace hygiene report. No repo-tracked path should be moved or deleted by this lane.

- mode: `audit_only`
- workspace_root: `<WORKSPACE_ROOT>`
- repo_root: `<REPO_ROOT>`
- quarantine_root: `<QUARANTINE_ROOT>`

## Decisions

| Path | Classification | Decision | Reason | Restore Target |
| --- | --- | --- | --- | --- |
| `<WORKSPACE_ROOT>/desktop.ini` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/desktop.ini` |
| `<WORKSPACE_ROOT>/MERT_YÜNLÜ_NİHAİ_HAYAT_SENTEZİ_v3_2026-07-06.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MERT_YÜNLÜ_NİHAİ_HAYAT_SENTEZİ_v3_2026-07-06.md` |
| `<REPO_ROOT>` | `project_repo` | `keep` | Active main repository root; never quarantine or delete directly from workspace hygiene. | `<REPO_ROOT>` |
| `<WORKSPACE_ROOT>/MertFormer_Build30_Max_Closure_Handoff.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MertFormer_Build30_Max_Closure_Handoff.md` |
| `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_050915` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_050915` |
| `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_050915.zip` | `archive_or_dump` | `quarantine_first` | Top-level archive or dump outside the main repo. Review and quarantine before any permanent cleanup. | `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_050915.zip` |
| `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_050915.zip.sha256` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_050915.zip.sha256` |
| `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_052426` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_052426` |
| `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_052426.zip` | `archive_or_dump` | `quarantine_first` | Top-level archive or dump outside the main repo. Review and quarantine before any permanent cleanup. | `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_052426.zip` |
| `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_052426.zip.sha256` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MertFormer_Chess_5080_Delivery_20260731_052426.zip.sha256` |
| `<WORKSPACE_ROOT>/MertFormer_Kapanış_Release_TruthSync_Master_Protokolu_2026-05-12.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MertFormer_Kapanış_Release_TruthSync_Master_Protokolu_2026-05-12.md` |
| `<WORKSPACE_ROOT>/MERTFORMER_PRE45K_POST45K_TEK_LISTE_2026-07-25.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MERTFORMER_PRE45K_POST45K_TEK_LISTE_2026-07-25.md` |
| `<WORKSPACE_ROOT>/mertformer_titan_bagimsiz_denetim_2026-07-27.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/mertformer_titan_bagimsiz_denetim_2026-07-27.md` |
| `<WORKSPACE_ROOT>/MERTFORMER_TITAN_TEK_NIHAI_REFERANS_2026-07-08.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/MERTFORMER_TITAN_TEK_NIHAI_REFERANS_2026-07-08.md` |

## Quarantine Moves
- none
