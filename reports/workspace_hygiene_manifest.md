# Workspace Hygiene Manifest

Quarantine-first workspace hygiene report. No repo-tracked path should be moved or deleted by this lane.

- mode: `audit_only`
- workspace_root: `<WORKSPACE_ROOT>`
- repo_root: `<REPO_ROOT>`
- quarantine_root: `<QUARANTINE_ROOT>`

## Decisions

| Path | Classification | Decision | Reason | Restore Target |
| --- | --- | --- | --- | --- |
| `<WORKSPACE_ROOT>/.DS_Store` | `desktop_metadata` | `ignore` | macOS metadata file. Ignore by default unless a human wants cosmetic cleanup. | `<WORKSPACE_ROOT>/.DS_Store` |
| `<WORKSPACE_ROOT>/.idea` | `workspace_metadata` | `keep` | IDE workspace metadata outside the repo root. Keep unless a human explicitly retires the workspace. | `<WORKSPACE_ROOT>/.idea` |
| `<WORKSPACE_ROOT>/.pass_backups` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/.pass_backups` |
| `<WORKSPACE_ROOT>/Applications` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/Applications` |
| `<WORKSPACE_ROOT>/Claude Reis` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/Claude Reis` |
| `<WORKSPACE_ROOT>/Documents` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/Documents` |
| `<WORKSPACE_ROOT>/İndirilenler` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/İndirilenler` |
| `<REPO_ROOT>` | `project_repo` | `keep` | Active main repository root; never quarantine or delete directly from workspace hygiene. | `<REPO_ROOT>` |
| `<WORKSPACE_ROOT>/Samsung Smart Switch Yedekleri (A34)` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/Samsung Smart Switch Yedekleri (A34)` |
| `<WORKSPACE_ROOT>/Sistem_ve_Ayar_Yedekleri` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/Sistem_ve_Ayar_Yedekleri` |
| `<WORKSPACE_ROOT>/Uygulamalar` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/Uygulamalar` |

## Quarantine Moves
- none
