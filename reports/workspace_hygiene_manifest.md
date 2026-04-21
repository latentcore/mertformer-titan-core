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
| `<WORKSPACE_ROOT>/adb_pull` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/adb_pull` |
| `<WORKSPACE_ROOT>/anthropic_closure_private` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/anthropic_closure_private` |
| `<WORKSPACE_ROOT>/codex_chat_extract.py` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/codex_chat_extract.py` |
| `<WORKSPACE_ROOT>/codex_chats_found.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/codex_chats_found.md` |
| `<WORKSPACE_ROOT>/current_codex_chat_transcript.md` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/current_codex_chat_transcript.md` |
| `<REPO_ROOT>` | `project_repo` | `keep` | Active main repository root; never quarantine or delete directly from workspace hygiene. | `<REPO_ROOT>` |
| `<WORKSPACE_ROOT>/mertformer_5080_runner_for_friend_20260420` | `external_workspace_entry` | `quarantine_first` | Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first. | `<WORKSPACE_ROOT>/mertformer_5080_runner_for_friend_20260420` |
| `<WORKSPACE_ROOT>/mertformer_5080_runner_for_friend_20260420.zip` | `archive_or_dump` | `quarantine_first` | Top-level archive or dump outside the main repo. Review and quarantine before any permanent cleanup. | `<WORKSPACE_ROOT>/mertformer_5080_runner_for_friend_20260420.zip` |
| `<WORKSPACE_ROOT>/mertformer_5080_runner_for_friend_20260420.zip.sha256` | `external_workspace_file` | `quarantine_first` | Top-level file outside the active repo root. Quarantine first instead of deleting in place. | `<WORKSPACE_ROOT>/mertformer_5080_runner_for_friend_20260420.zip.sha256` |

## Quarantine Moves
- none
