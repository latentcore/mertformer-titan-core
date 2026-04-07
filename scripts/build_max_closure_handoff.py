#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'
DESKTOP = Path.home() / 'Desktop'
FREEZE_JSON = REPORTS / 'final_freeze_manifest.json'
FREEZE_MD = REPORTS / 'final_freeze_manifest.md'
COMMANDS_MD = REPORTS / 'final_commands.md'
HANDOFF_MD = REPORTS / 'repo_external_handoff.md'
DESKTOP_HANDOFF = DESKTOP / 'MertFormer_Build30_Max_Closure_Handoff.md'


def desktop_handoff_mode() -> str:
    return os.environ.get('TITAN_DESKTOP_HANDOFF_MODE', 'auto').strip().lower() or 'auto'


def desktop_handoff_display_path() -> str:
    return f"<DESKTOP_PATH>/{DESKTOP_HANDOFF.name}"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def build_freeze_manifest(readiness: dict) -> dict:
    return {
        'schema': 'final_freeze_manifest_v1',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'official_product_sentence': 'Türkiye’ye fayda sağlayacak, offline-first, edge-native, yerli ve entegre edilebilir zeka altyapısı.',
        'risk_ceiling': 'Medium Refine',
        'guardrail': 'If any task increases risk to 45K readiness, reproducibility, or closure confidence, demote it to phase-2.',
        'freeze': {
            'feature_freeze': 'locked_for_45k_pass',
            'config_freeze': 'locked_for_45k_pass',
            'dataset_freeze': 'locked_for_45k_pass',
            'tokenizer_freeze': 'locked_for_45k_pass',
            'teacher_logits_decision': 'dual_path_contract',
        },
        'training_positioning': '45K is the first serious architecture validation run, not the final capability ceiling.',
        'readiness_final_status': readiness.get('final_status', 'UNKNOWN'),
        'recommended_path': readiness.get('recommended_path'),
    }


def write_freeze_docs(payload: dict) -> None:
    FREEZE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    md = [
        '# Final Freeze Manifest',
        '',
        f"- official_product_sentence: {payload['official_product_sentence']}",
        f"- risk_ceiling: `{payload['risk_ceiling']}`",
        f"- readiness_final_status: `{payload['readiness_final_status']}`",
        f"- recommended_path: `{payload.get('recommended_path') or 'none'}`",
        '',
        '## Freeze State',
        '',
    ]
    for key, value in payload['freeze'].items():
        md.append(f'- `{key}`: `{value}`')
    md.extend(['', '## Guardrail', '', f"- {payload['guardrail']}", '', '## Positioning', '', f"- {payload['training_positioning']}"])
    FREEZE_MD.write_text('\n'.join(md) + '\n', encoding='utf-8')


def write_commands(readiness: dict) -> None:
    lines = [
        '# Final Commands',
        '',
        '## Canonical 45K Start Gate',
        '',
        '```bash',
        'bash zero_touch_start.sh --check-only',
        '```',
        '',
        '## Canonical 45K Launcher',
        '',
        '```bash',
        'TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh',
        '```',
        '',
        'Optional online teacher lane:',
        '',
        '```bash',
        'HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh',
        '```',
        '',
        '## Canonical Closure',
        '',
        '```bash',
        'bash scripts/final_one_shot.sh',
        '```',
        '',
        '## Refresh Backlog and Readiness Reports',
        '',
        '```bash',
        'python3 scripts/build_master_closure_matrix.py',
        'python3 scripts/build_train_readiness_contract.py --allow-not-ready',
        'python3 scripts/start_gate.py --skip-verify-all --allow-not-ready',
        'python3 scripts/build_target_machine_handoff_bundle.py',
        'python3 scripts/build_max_closure_handoff.py',
        '```',
        '',
        f"Current readiness status: `{readiness.get('final_status', 'UNKNOWN')}`",
        f"Recommended path: `{readiness.get('recommended_path') or 'none'}`",
    ]
    COMMANDS_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_desktop_copy(body: str) -> dict:
    mode = desktop_handoff_mode()
    payload = {
        'mode': mode,
        'path': str(DESKTOP_HANDOFF),
        'display_path': desktop_handoff_display_path(),
        'status': 'skipped',
        'reason': '',
    }
    if mode in {'0', 'off', 'false', 'disabled', 'skip'}:
        payload['reason'] = 'disabled by TITAN_DESKTOP_HANDOFF_MODE'
        return payload

    should_force = mode in {'1', 'on', 'true', 'force'}
    if not DESKTOP.is_dir() and not should_force:
        payload['reason'] = 'desktop directory unavailable'
        return payload

    DESKTOP.mkdir(parents=True, exist_ok=True)
    DESKTOP_HANDOFF.write_text(body, encoding='utf-8')
    payload['status'] = 'written'
    payload['reason'] = 'desktop copy refreshed'
    return payload


def build_handoff_body(readiness: dict, matrix: dict, freeze: dict, desktop_copy: dict) -> str:
    summary = matrix.get('summary', {})
    lines = [
        '# MertFormer Build 30 Max Closure Handoff',
        '',
        f"- generated_utc: `{freeze['generated_utc']}`",
        f"- product_sentence: {freeze['official_product_sentence']}",
        f"- canonical_closure_entrypoint: `bash scripts/final_one_shot.sh`",
        f"- train_readiness_status: `{readiness.get('final_status', 'UNKNOWN')}`",
        f"- train_readiness_reason: `{readiness.get('decision_reason_code', 'UNKNOWN')}`",
        f"- recommended_path: `{readiness.get('recommended_path') or 'none'}`",
        f"- desktop_copy_status: `{desktop_copy['status']}`",
        f"- desktop_copy_path: `{desktop_copy['display_path']}`",
        f"- desktop_copy_reason: `{desktop_copy['reason'] or 'none'}`",
        '',
        '## Closure Matrix Summary',
        '',
        f"- total_items: `{summary.get('total_items', 0)}`",
        f"- this_pass: `{summary.get('this_pass', 0)}`",
        f"- phase_2: `{summary.get('phase_2', 0)}`",
        f"- external: `{summary.get('external', 0)}`",
        f"- rejected_with_reason: `{summary.get('rejected_with_reason', 0)}`",
        '',
        '## Key Evidence Files',
        '',
        '- `reports/master_closure_matrix.md`',
        '- `reports/phase2_carryover.md`',
        '- `reports/train_readiness_decision.md`',
        '- `reports/start_gate_operator_decision.md`',
        '- `reports/target_machine_handoff_manifest.md`',
        '- `reports/final_freeze_manifest.md`',
        '- `reports/one_command_full_sop_summary.md`',
        '- `artifacts/target_machine_handoff_bundle.zip`',
        '',
        '## Guardrail',
        '',
        f"- {freeze['guardrail']}",
        '',
        '## Notes',
        '',
        '- The repo-internal handoff is canonical; the desktop copy is best-effort and optional.',
        '- Txt backlog is captured, classified, and never silently dropped.',
        '- 45K readiness remains the primary ship gate for this pass.',
        '- Any item that threatens 45K readiness is intentionally carried to phase-2.',
    ]
    return '\n'.join(lines) + '\n'


def write_handoff(body: str) -> None:
    HANDOFF_MD.write_text(body, encoding='utf-8')


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    matrix = load_json(REPORTS / 'master_closure_matrix.json')
    readiness = load_json(REPORTS / 'train_readiness_decision.json')
    freeze = build_freeze_manifest(readiness)
    write_freeze_docs(freeze)
    write_commands(readiness)
    desktop_copy = {
        'mode': desktop_handoff_mode(),
        'path': str(DESKTOP_HANDOFF),
        'display_path': desktop_handoff_display_path(),
        'status': 'pending',
        'reason': 'desktop copy not attempted yet',
    }
    body = build_handoff_body(readiness, matrix, freeze, desktop_copy)
    desktop_copy = write_desktop_copy(body)
    body = build_handoff_body(readiness, matrix, freeze, desktop_copy)
    write_handoff(body)
    if desktop_copy['status'] == 'written':
        DESKTOP_HANDOFF.write_text(body, encoding='utf-8')
    print(json.dumps({'repo_handoff': str(HANDOFF_MD), 'desktop_copy': desktop_copy}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
