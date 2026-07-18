#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# Directories whose .md files are vendored/cache noise, not real repo content -- including
# them in the "duplicate markdown" scan produces false-positive groups (e.g. .lint-venv's
# vendored pip LICENSE.md, .pytest_cache's auto-generated README.md). Added 2026-07-19 after
# an audit found these accounted for 2 of the ~27 flagged "duplicate" groups.
_EXCLUDED_DIR_PARTS = {'.git', '.titan-venv', '.lint-venv', '.pytest_cache', '.mypy_cache', '.ruff_cache'}


def main() -> int:
    md_files = sorted([p for p in ROOT.rglob('*.md') if not (_EXCLUDED_DIR_PARTS & set(p.parts))])

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
        '> Cross-reference (added 2026-07-19): this is a short-form summary of the same rule\n'
        '> `reports/repo_directory_contract.md` states in full, including its enforcement command. That\n'
        '> file is canonical; this one exists for a quick-glance summary only.\n\n'
        '- Keep generated artifacts under `artifacts/` and `reports/`.\n'
        '- Keep source code under `config/`, `layers/`, `model/`, `train/`, `scripts/`, `mertformer_sdk/`.\n'
        '- Keep policy files under `policy/`.\n'
        '- Keep docs index in `docs/PROJECT_STRUCTURE.md`.\n',
        encoding='utf-8',
    )

    # 'ok' artik gercek duruma baglandi: duplicate doc isim grubu yoksa True.
    # Cikis kodu/akis kasitli olarak BOZULMUYOR (bu adim frozen pipeline'da
    # run_step ile cagriliyor); 'ok' bayragi yalnizca durust raporlama icin.
    # NOT: 'ok' burada bir gecme-kapisi DEGIL; downstream'in fail etmesi
    # gerekiyorsa duplicate_name_group_count alanini kullanmali.
    ok = len(dedup_groups) == 0
    drift = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'md_file_count': len(md_files),
        'duplicate_name_group_count': len(dedup_groups),
        'duplicate_name_groups': dedup_groups,
        'ok': ok,
    }
    (ROOT / 'reports' / 'folder_drift_report.json').write_text(json.dumps(drift, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'ok': ok, 'md_files': len(md_files), 'dup_groups': len(dedup_groups)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
