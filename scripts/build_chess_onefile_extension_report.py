#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_JSON = ROOT / 'reports' / 'chess_onefile_extension_report.json'
REPORT_MD = ROOT / 'reports' / 'chess_onefile_extension_report.md'
REQUIRED_SUITES = {'opening', 'tactical', 'endgame', 'blunder_correction'}


def build_payload() -> dict[str, Any]:
    from scripts import chess_5080_onefile as onefile

    manifest = onefile.build_curated_position_manifest(onefile.RUN_CONFIG)
    _, training_manifest = onefile.build_curated_training_examples(onefile.RUN_CONFIG)
    corpus = onefile.build_synthetic_teaching_corpus(onefile.RUN_CONFIG)
    suites_present = set(manifest.get('suite_counts', {}).keys())
    missing_suites = sorted(REQUIRED_SUITES - suites_present)
    all_green = (
        manifest.get('enabled', False)
        and training_manifest.get('examples_total', 0) > 0
        and corpus.get('record_count', 0) > 0
        and not missing_suites
    )
    return {
        'schema': 'chess_onefile_extension_report_v1',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'final_status': 'READY' if all_green else 'NOT_READY',
        'summary': {
            'all_green': all_green,
            'required_suites': sorted(REQUIRED_SUITES),
            'missing_suites': missing_suites,
            'curated_position_count': int(manifest.get('position_count', 0)),
            'curated_training_examples': int(training_manifest.get('examples_total', 0)),
            'synthetic_teaching_records': int(corpus.get('record_count', 0)),
        },
        'curated_position_manifest': manifest,
        'training_augmentation_manifest': training_manifest,
        'synthetic_teaching_corpus': {
            'enabled': bool(corpus.get('enabled', False)),
            'record_count': int(corpus.get('record_count', 0)),
            'suite_counts': dict(corpus.get('suite_counts', {})),
            'level_counts': dict(corpus.get('level_counts', {})),
        },
        'notes': [
            'This report validates repo-side onefile chess extension surfaces only.',
            'It does not claim a measured Elo outcome or a completed training run.',
            'Curated suites are intended for repeatable augmentation and contract-faithfulness checks.',
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Chess Onefile Extension Report',
        '',
        f"- final_status: `{payload['final_status']}`",
        f"- curated_position_count: `{payload['summary']['curated_position_count']}`",
        f"- curated_training_examples: `{payload['summary']['curated_training_examples']}`",
        f"- synthetic_teaching_records: `{payload['summary']['synthetic_teaching_records']}`",
        f"- missing_suites: `{', '.join(payload['summary']['missing_suites']) or 'none'}`",
        '',
        '## Notes',
        '',
    ]
    lines.extend(f'- {note}' for note in payload['notes'])
    return '\n'.join(lines) + '\n'


def main() -> int:
    payload = build_payload()
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    REPORT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({'final_status': payload['final_status'], 'all_green': payload['summary']['all_green']}, ensure_ascii=False))
    return 0 if payload['summary']['all_green'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
