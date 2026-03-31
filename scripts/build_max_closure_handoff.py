#!/usr/bin/env python3
from __future__ import annotations

import json
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
        'python3 scripts/build_max_closure_handoff.py',
        '```',
        '',
        f"Current readiness status: `{readiness.get('final_status', 'UNKNOWN')}`",
        f"Recommended path: `{readiness.get('recommended_path') or 'none'}`",
    ]
    COMMANDS_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_handoff(readiness: dict, matrix: dict, freeze: dict) -> None:
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
        '- `reports/final_freeze_manifest.md`',
        '- `reports/one_command_full_sop_summary.md`',
        '',
        '## Guardrail',
        '',
        f"- {freeze['guardrail']}",
        '',
        '## Notes',
        '',
        '- Txt backlog is captured, classified, and never silently dropped.',
        '- 45K readiness remains the primary ship gate for this pass.',
        '- Any item that threatens 45K readiness is intentionally carried to phase-2.',
    ]
    body = '\n'.join(lines) + '\n'
    HANDOFF_MD.write_text(body, encoding='utf-8')
    DESKTOP_HANDOFF.write_text(body, encoding='utf-8')


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    matrix = load_json(REPORTS / 'master_closure_matrix.json')
    readiness = load_json(REPORTS / 'train_readiness_decision.json')
    freeze = build_freeze_manifest(readiness)
    write_freeze_docs(freeze)
    write_commands(readiness)
    write_handoff(readiness, matrix, freeze)
    print(str(DESKTOP_HANDOFF))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
