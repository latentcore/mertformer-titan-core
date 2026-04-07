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
| `<WORKSPACE_ROOT>/.ruff_cache` | `workspace_cache` | `quarantine_first` | Rebuildable lint cache outside the repo root. Safe candidate for quarantine-first handling. | `<WORKSPACE_ROOT>/.ruff_cache` |
| `<REPO_ROOT>` | `project_repo` | `keep` | Active main repository root; never quarantine or delete directly from workspace hygiene. | `<REPO_ROOT>` |

## Quarantine Moves
- none
