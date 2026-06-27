#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEALROOM = ROOT / 'mertformer-titan-dealroom-private'


def cmd(cwd: Path, *args: str) -> str:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        # Komut basarisiz oldu; geriye bos string donuyoruz (rapor o alani 'bos'
        # gosterir) ama hatanin sessizce yutulmamasi için stderr'e uyari yaziyoruz.
        sys.stderr.write(
            f"dealroom_sync: command failed (rc={p.returncode}) in {cwd}: "
            f"{' '.join(args)}: {p.stderr.strip()}\n"
        )
        return ''
    return p.stdout.strip()


def main() -> int:
    reports = ROOT / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    payload = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'dealroom_path': str(DEALROOM.relative_to(ROOT)),
        'exists': DEALROOM.exists(),
        'git': {
            'branch': cmd(DEALROOM, 'git', 'branch', '--show-current') if DEALROOM.exists() else '',
            'commit': cmd(DEALROOM, 'git', 'rev-parse', 'HEAD') if DEALROOM.exists() else '',
            'remote': cmd(DEALROOM, 'git', 'remote', '-v') if DEALROOM.exists() else '',
        },
        'ok': DEALROOM.exists(),
    }

    (reports / 'dealroom_reference.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    ownership = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'main_repo_commit': cmd(ROOT, 'git', 'rev-parse', 'HEAD'),
        'dealroom_commit': payload['git']['commit'],
        'main_repo_remote': cmd(ROOT, 'git', 'remote', '-v'),
        'dealroom_remote': payload['git']['remote'],
        'ok': bool(payload['git']['commit']),
    }
    (reports / 'ownership_proof_bundle.json').write_text(json.dumps(ownership, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'ok': payload['ok']}))
    return 0 if payload['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
