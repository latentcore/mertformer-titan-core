#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    md_files = sorted([p for p in ROOT.rglob('*.md') if '.git' not in p.parts and '.titan-venv' not in p.parts])

    by_name = defaultdict(list)
    for p in md_files:
        by_name[p.name.lower()].append(str(p.relative_to(ROOT)))

    dedup_groups = {k: v for k, v in by_name.items() if len(v) > 1}

    canonical = ROOT / 'reports' / 'docs_dedup_canonical_list.md'
    canonical.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Docs Canonical List', '', 'Generated automatically.', '']
    for p in md_files:
        lines.append(f'- `{p.relative_to(ROOT)}`')
    lines.append('')
    lines.append('## Duplicate Name Groups')
    for k, group in sorted(dedup_groups.items()):
        lines.append(f'- `{k}`')
        for g in group:
            lines.append(f'  - `{g}`')
    canonical.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    policy_md = ROOT / 'reports' / 'folder_structure_policy.md'
    policy_md.write_text(
        '# Folder Structure Policy\n\n'
        '- Keep generated artifacts under `artifacts/` and `reports/`.\n'
        '- Keep source code under `config/`, `layers/`, `model/`, `train/`, `scripts/`, `mertformer_sdk/`.\n'
        '- Keep policy files under `policy/`.\n'
        '- Keep docs index in `docs/PROJECT_STRUCTURE.md`.\n',
        encoding='utf-8',
    )

    drift = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'md_file_count': len(md_files),
        'duplicate_name_group_count': len(dedup_groups),
        'duplicate_name_groups': dedup_groups,
        'ok': True,
    }
    (ROOT / 'reports' / 'folder_drift_report.json').write_text(json.dumps(drift, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'ok': True, 'md_files': len(md_files), 'dup_groups': len(dedup_groups)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
