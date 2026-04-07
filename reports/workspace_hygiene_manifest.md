# Workspace Hygiene Manifest

Quarantine-first workspace hygiene report. No repo-tracked path should be moved or deleted by this lane.

- mode: `audit_only`
- workspace_root: `/Users/mertyunlu/Desktop/NİHAİ`
- repo_root: `/Users/mertyunlu/Desktop/NİHAİ/mertformer-titan-core`
- quarantine_root: `/Users/mertyunlu/Desktop/NİHAİ/workspace_quarantine`

## Decisions

| Path | Classification | Decision | Reason | Restore Target |
| --- | --- | --- | --- | --- |
| `/Users/mertyunlu/Desktop/NİHAİ/.DS_Store` | `desktop_metadata` | `ignore` | macOS metadata file. Ignore by default unless a human wants cosmetic cleanup. | `/Users/mertyunlu/Desktop/NİHAİ/.DS_Store` |
| `/Users/mertyunlu/Desktop/NİHAİ/.idea` | `workspace_metadata` | `keep` | IDE workspace metadata outside the repo root. Keep unless a human explicitly retires the workspace. | `/Users/mertyunlu/Desktop/NİHAİ/.idea` |
| `/Users/mertyunlu/Desktop/NİHAİ/.ruff_cache` | `workspace_cache` | `quarantine_first` | Rebuildable lint cache outside the repo root. Safe candidate for quarantine-first handling. | `/Users/mertyunlu/Desktop/NİHAİ/.ruff_cache` |
| `/Users/mertyunlu/Desktop/NİHAİ/mertformer-titan-core` | `project_repo` | `keep` | Active main repository root; never quarantine or delete directly from workspace hygiene. | `/Users/mertyunlu/Desktop/NİHAİ/mertformer-titan-core` |

## Quarantine Moves
- none
