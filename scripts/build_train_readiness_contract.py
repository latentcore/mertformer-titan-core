#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / 'logs' / 'preflight'
REPORT_JSON = ROOT / 'reports' / 'train_readiness_decision.json'
REPORT_MD = ROOT / 'reports' / 'train_readiness_decision.md'
LEGACY_MANIFEST = ROOT / 'reports' / 'training_readiness_manifest.json'
LEGACY_LOG = LOG_DIR / 'train_ready_status.json'

PROFILES = [
    {
        'name': 'offline_clean',
        'profile': 'strict_offline_training_readiness',
        'env': {
            'TITAN_OFFLINE': '1',
            'TITAN_REQUIRE_GATED_TEACHER': '1',
            'TITAN_USE_PRECOMPUTED_LOGITS': '1',
            # [tier-2 BLOCKER] offline_clean is teacher-tokenizer KD; precompute refuses
            # TR. Forcing TR=1 here makes preflight's logit-alignment check report
            # TOKENIZER_IDENTITY_DRIFT and turns the healthy lane red.
            'TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL': '0',
            'TITAN_PREFLIGHT_REQUIRE_STAGE_JSONL': '1',
        },
    },
    {
        'name': 'remote_bootstrap',
        'profile': 'runtime_injected_training_readiness',
        'env': {
            'TITAN_OFFLINE': '0',
            'TITAN_PREFLIGHT_REQUIRE_STAGE_JSONL': '0',
        },
    },
    {
        'name': 'online_teacher',
        'profile': 'strict_online_training_readiness',
        'env': {
            'TITAN_OFFLINE': '0',
        },
    },
]

READY_CODES = {
    'offline_clean': 'READY_OFFLINE_CLEAN',
    'remote_bootstrap': 'READY_REMOTE_BOOTSTRAP',
    'online_teacher': 'READY_ONLINE_TEACHER',
}

PREFERRED_PATH_ORDER = ('offline_clean', 'remote_bootstrap', 'online_teacher')


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def sanitize_text(text: str) -> str:
    cleaned = text.replace(str(ROOT), '<repo>')
    desktop = Path.home() / 'Desktop'
    cleaned = cleaned.replace(str(desktop), '<desktop>')
    return cleaned


def sanitize_value(value):
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_value(key): sanitize_value(item) for key, item in value.items()}
    return value


def run_profile(py: str, entry: dict) -> dict:
    env = os.environ.copy()
    env.update(entry['env'])
    proc = subprocess.run(
        [py, str(ROOT / 'scripts' / 'titan_preflight.py'), '--profile', entry['profile']],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    profile_report = LOG_DIR / f"train_ready_status.{entry['profile']}.json"
    payload = load_json(profile_report)
    payload.update(
        {
            'path_name': entry['name'],
            'profile': entry['profile'],
            'exit_code': proc.returncode,
            'stdout_tail': sanitize_text(proc.stdout[-4000:]),
            'stderr_tail': sanitize_text(proc.stderr[-4000:]),
        }
    )
    return payload


def choose_decision(results: list[dict]) -> tuple[str, str, str | None, list[str]]:
    blockers = [f"{r.get('path_name')}:{r.get('reason_code', 'UNKNOWN')}" for r in results if r.get('status') != 'PASS']
    by_name = {row.get('path_name'): row for row in results}
    for name in PREFERRED_PATH_ORDER:
        row = by_name.get(name)
        if row and row.get('status') == 'PASS':
            return 'TRAIN_ALLOWED', READY_CODES[name], name, blockers
    return 'NOT_ALLOWED', '__'.join(blockers) if blockers else 'NO_READY_PATH', None, blockers


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        '# Train Readiness Decision',
        '',
        f"- final_status: `{payload['final_status']}`",
        f"- decision_reason_code: `{payload['decision_reason_code']}`",
        f"- recommended_path: `{payload.get('recommended_path') or 'none'}`",
        f"- guardrail: `{payload['guardrail']}`",
        '',
        '## Paths',
        '',
    ]
    for row in payload['paths']:
        lines.extend(
            [
                f"### {row['path_name']}",
                f"- profile: `{row['profile']}`",
                f"- status: `{row.get('status', 'UNKNOWN')}`",
                f"- reason_code: `{row.get('reason_code', 'UNKNOWN')}`",
                f"- exit_code: `{row.get('exit_code')}`",
                '',
            ]
        )
    lines.extend(['## Blockers', ''])
    if payload['blockers']:
        lines.extend(f'- `{x}`' for x in payload['blockers'])
    else:
        lines.append('- none')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Build dual-path training readiness contract.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--allow-not-ready', action='store_true', help='Return exit code 0 even when final_status=NOT_ALLOWED.')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    results = [run_profile(args.python, row) for row in PROFILES]
    final_status, decision_reason_code, recommended_path, blockers = choose_decision(results)
    payload = {
        'schema': 'train_readiness_decision_v1',
        'final_status': final_status,
        'decision_reason_code': decision_reason_code,
        'recommended_path': recommended_path,
        'guardrail': 'At least one readiness path must pass cleanly before TRAIN_ALLOWED is granted; offline_clean stays strict precomputed KD, while remote_bootstrap is valid only when the rented-machine bootstrap flow can inject credentials and fetch datasets at runtime.',
        'paths': results,
        'blockers': blockers,
    }

    sanitized_payload = sanitize_value(payload)
    REPORT_JSON.write_text(json.dumps(sanitized_payload, indent=2), encoding='utf-8')
    LEGACY_MANIFEST.write_text(json.dumps(sanitized_payload, indent=2), encoding='utf-8')
    LEGACY_LOG.write_text(json.dumps(sanitized_payload, indent=2), encoding='utf-8')
    write_markdown(REPORT_MD, sanitized_payload)
    print(json.dumps({'final_status': final_status, 'decision_reason_code': decision_reason_code}, ensure_ascii=False))

    if final_status != 'TRAIN_ALLOWED' and not args.allow_not_ready:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
