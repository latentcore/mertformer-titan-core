#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESKTOP = Path.home() / 'Desktop'

ACTION_KEYWORDS = {
    '45k', 'closure', 'sop', 'truth', 'claim', 'evidence', 'readiness', 'train allowed', 'train-ready',
    'benchmark', 'regression', 'freeze', 'feature freeze', 'config freeze', 'dataset freeze',
    'teacher', 'logits', 'tokenizer', 'distillation', 'curriculum', 'offline-first', 'edge-native',
    'public-good', 'auditable', 'human', 'high-risk', 'handoff', 'phase-2', 'phase 2', 'reason_code',
    'verification', 'master matrix', 'no claim without evidence', 'creative mode', 'truth mode',
    'folklore', 'provenance', 'manifest', 'optional source', 'token probe', 'data pipeline',
    'carryover', 'train-ready', 'atomic commit', 'sop', 'one-command'
}
PHASE2_KEYWORDS = {
    'agi', 'asi', 'multimodal', 'turboquant', 'latency pack', 'kernel fusion', 'zero-copy',
    'mobile/npu özel hızlı yol', 'world model', 'continual learning', 'retrieval', 'open refactor',
    'full c++ rewrite', '500b', '1t moe', 'bigger-token', '70b', '100b', 'phase-2', 'phase 2'
}
EXTERNAL_KEYWORDS = {
    'legal', 'lawyer', 'counsel', 'teknofest', 'pilot contract', 'loi', 'external', '3. parti',
    'third-party', 'pentest', 'savunma sanayii', 'investor', 'müşteri', 'customer', 'procurement',
    'hibe', 'teşvik', 'council'
}
REJECT_KEYWORDS = {
    'hack yapan', 'hack yapanı', 'covert surveillance', 'unauthorized surveillance', 'offensive cyber',
    'otonom saldırı', 'zarar verme', 'harmful autonomy'
}

DEPENDENCY_BY_CATEGORY = {
    'closure_flow': 'run.sh -> one_command_full_sop.sh -> final_one_shot.sh',
    'truth_claim': 'claim gates -> docs/policy/prompt surfaces',
    'freeze_contract': 'freeze manifest -> readiness decision -> training start',
    'training_readiness': 'titan_preflight -> readiness contract -> train-ready verdict',
    'data_contract': 'data_pipeline -> datasets/* -> training contract',
    'policy': 'USE_POLICY/SECURITY/MODEL_CARD/prompt surfaces',
    'handoff': 'closure artifacts -> reports -> desktop handoff',
    'external': 'repo artifacts -> external sign-off / pilot / legal',
    'phase2': 'phase-2 carryover after 45K gate'
}
ACCEPTANCE_BY_CATEGORY = {
    'closure_flow': 'bash scripts/final_one_shot.sh',
    'truth_claim': '.titan-venv/bin/python scripts/check_doc_claim_consistency.py',
    'freeze_contract': '.titan-venv/bin/python scripts/build_max_closure_handoff.py',
    'training_readiness': '.titan-venv/bin/python scripts/build_train_readiness_contract.py',
    'data_contract': '.titan-venv/bin/python scripts/data_pipeline.py --target-samples <n>',
    'policy': '.titan-venv/bin/python scripts/check_doc_claim_consistency.py',
    'handoff': '.titan-venv/bin/python scripts/build_max_closure_handoff.py',
    'external': 'External sign-off / pilot evidence bundle',
    'phase2': 'Explicit carryover listing in reports/phase2_carryover.md'
}

REPO_ITEMS = [
    'Build and maintain a dependency-ordered master closure matrix',
    'Classify every TXT and repo backlog item into this-pass, phase-2, external, or rejected-with-reason',
    'Keep 45K readiness as the primary ship gate for this pass',
    'Unify run.sh, one_command_full_sop.sh, and final_one_shot.sh into one canonical closure flow',
    'Produce one final TRAIN_ALLOWED or NOT_ALLOWED verdict with reason codes',
    'Expand claim discipline across measured, target, vision, verified, hypothesis, and creative/folklore surfaces',
    'Enforce no claim without evidence across final docs and reports',
    'Declare feature freeze, config freeze, dataset freeze, tokenizer freeze, and teacher/logits decision',
    'Keep dual-path readiness for offline-clean and online-teacher flows',
    'Harden data pipeline provenance, optional source policy, token probe, and revision/hash lineage',
    'Make public-good, auditable, human-benefiting deployment the official framing',
    'Reject harmful autonomy and covert surveillance framing',
    'Generate repo-external handoff, final commands, risk list, and phase-2 carryover list',
    'Use atomic thematic commits instead of a single mega-commit',
]


@dataclass
class Item:
    item_id: str
    source: str
    source_ref: str
    text: str
    phase: str
    risk: str
    category: str
    dependency: str
    evidence: str
    acceptance: str
    reason: str


def _norm(text: str) -> str:
    out = text.replace('\xa0', ' ')
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def discover_txt(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    candidates = sorted(DEFAULT_DESKTOP.glob('ChatGPT-Claude Code*.txt'))
    return candidates[0] if candidates else None


def classify_phase(text: str) -> tuple[str, str, str]:
    lower = text.lower()
    if any(k in lower for k in REJECT_KEYWORDS):
        return 'rejected-with-reason', 'policy', 'Conflicts with public-good / high-risk guardrail.'
    if any(k in lower for k in EXTERNAL_KEYWORDS):
        return 'external', 'external', 'Requires outside sign-off, commercial action, or external dependency.'
    if any(k in lower for k in PHASE2_KEYWORDS):
        return 'phase-2', 'phase2', 'Guardrail: risks 45K readiness or exceeds Medium Refine ceiling.'
    if any(k in lower for k in ('freeze', 'claim', 'evidence', 'truth', 'creative', 'folklore', 'policy', 'handoff')):
        return 'this-pass', 'truth_claim', 'Directly improves closure confidence and documentation truth.'
    if any(k in lower for k in ('teacher', 'logits', 'tokenizer', 'readiness', 'train allowed', 'train-ready')):
        return 'this-pass', 'training_readiness', 'Directly affects the 45K readiness gate.'
    if any(k in lower for k in ('data pipeline', 'dataset', 'curriculum', 'token probe', 'optional source', 'provenance')):
        return 'this-pass', 'data_contract', 'Directly affects the training data contract.'
    if any(k in lower for k in ('sop', 'closure', 'one-command', 'master matrix', 'run.sh', 'final_one_shot')):
        return 'this-pass', 'closure_flow', 'Directly affects the canonical closure flow.'
    return 'this-pass', 'handoff', 'Relevant to this pass and does not exceed the risk ceiling.'


def classify_risk(text: str, phase: str) -> str:
    lower = text.lower()
    if phase in {'phase-2', 'rejected-with-reason'}:
        return 'high'
    if any(k in lower for k in ('teacher', 'logits', 'tokenizer', 'dataset', 'run.sh', 'train', 'preflight', 'freeze')):
        return 'medium'
    return 'low'


def extract_txt_items(txt_path: Path) -> list[Item]:
    text = txt_path.read_text(encoding='utf-8', errors='ignore')
    seen: set[str] = set()
    items: list[Item] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = _norm(raw)
        if not line or len(line) < 24:
            continue
        if line.startswith(('User:', 'Created:', 'Updated:', 'Exported:', 'Link:')):
            continue
        if line.startswith(('## Prompt:', '## Response:')):
            continue
        if line.startswith(('```', '│', '├', '└', 'import ', 'def ', 'class ')):
            continue
        lower = line.lower()
        if not any(k in lower for k in ACTION_KEYWORDS):
            continue
        if len(line) > 320:
            line = line[:317] + '...'
        dedupe_key = lower
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        phase, category, reason = classify_phase(line)
        risk = classify_risk(line, phase)
        items.append(
            Item(
                item_id=f'txt:{idx}',
                source='txt',
                source_ref=f'{txt_path.name}:{idx}',
                text=line,
                phase=phase,
                risk=risk,
                category=category,
                dependency=DEPENDENCY_BY_CATEGORY[category],
                evidence='TXT export + repo-truth reconciliation required.',
                acceptance=ACCEPTANCE_BY_CATEGORY[category],
                reason=reason,
            )
        )
    return items


def repo_items() -> Iterable[Item]:
    for idx, text in enumerate(REPO_ITEMS, start=1):
        phase, category, reason = classify_phase(text)
        risk = classify_risk(text, phase)
        yield Item(
            item_id=f'repo:{idx:03d}',
            source='repo',
            source_ref='implementation-plan',
            text=text,
            phase=phase,
            risk=risk,
            category=category,
            dependency=DEPENDENCY_BY_CATEGORY[category],
            evidence='Plan item mapped to repo closure surfaces.',
            acceptance=ACCEPTANCE_BY_CATEGORY[category],
            reason=reason,
        )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_markdown(path: Path, items: list[Item], summary: Counter, txt_path: Path | None) -> None:
    lines = [
        '# Master Closure Matrix',
        '',
        'This matrix combines repo closure obligations with actionable items mined from the desktop TXT export.',
        '',
        f'- txt_source: `{txt_path}`' if txt_path else '- txt_source: `missing`',
        f"- total_items: `{sum(summary.values())}`",
        f"- this_pass: `{summary['this-pass']}`",
        f"- phase-2: `{summary['phase-2']}`",
        f"- external: `{summary['external']}`",
        f"- rejected-with-reason: `{summary['rejected-with-reason']}`",
        '',
        '| ID | Source | Phase | Risk | Category | Text | Reason | Acceptance |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for item in items:
        text = item.text.replace('|', '\\|')
        reason = item.reason.replace('|', '\\|')
        acceptance = item.acceptance.replace('|', '\\|')
        lines.append(
            f'| `{item.item_id}` | `{item.source_ref}` | `{item.phase}` | `{item.risk}` | `{item.category}` | {text} | {reason} | `{acceptance}` |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_phase2(path: Path, items: list[Item]) -> None:
    phase2 = [x for x in items if x.phase == 'phase-2']
    rejected = [x for x in items if x.phase == 'rejected-with-reason']
    external = [x for x in items if x.phase == 'external']
    lines = [
        '# Phase-2 Carryover',
        '',
        'Items listed here were not dropped. They were intentionally demoted by the 45K guardrail, held for external follow-up, or rejected on policy grounds.',
        '',
        '## Phase-2',
    ]
    if phase2:
        for item in phase2:
            lines.append(f'- `{item.item_id}` {item.text}  ')
            lines.append(f'  reason: {item.reason}')
    else:
        lines.append('- none')
    lines.extend(['', '## External', ''])
    if external:
        for item in external:
            lines.append(f'- `{item.item_id}` {item.text}  ')
            lines.append(f'  reason: {item.reason}')
    else:
        lines.append('- none')
    lines.extend(['', '## Rejected with Reason', ''])
    if rejected:
        for item in rejected:
            lines.append(f'- `{item.item_id}` {item.text}  ')
            lines.append(f'  reason: {item.reason}')
    else:
        lines.append('- none')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_risk_register(path: Path, items: list[Item]) -> None:
    high = [x for x in items if x.risk == 'high']
    medium = [x for x in items if x.risk == 'medium']
    lines = ['# Closure Risk Register', '', '## High', '']
    if high:
        for item in high:
            lines.append(f'- `{item.item_id}` {item.text} -> {item.reason}')
    else:
        lines.append('- none')
    lines.extend(['', '## Medium', ''])
    if medium:
        for item in medium[:80]:
            lines.append(f'- `{item.item_id}` {item.text} -> {item.reason}')
    else:
        lines.append('- none')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Build master closure matrix from repo + desktop TXT backlog.')
    parser.add_argument('--txt', default='', help='Explicit TXT path override.')
    parser.add_argument('--out-json', default='reports/master_closure_matrix.json')
    parser.add_argument('--out-md', default='reports/master_closure_matrix.md')
    parser.add_argument('--phase2-md', default='reports/phase2_carryover.md')
    parser.add_argument('--risk-md', default='reports/closure_risk_register.md')
    args = parser.parse_args()

    txt_path = discover_txt(args.txt or None)
    items = list(repo_items())
    if txt_path is not None:
        items.extend(extract_txt_items(txt_path))

    items.sort(key=lambda x: (x.phase != 'this-pass', x.category, x.item_id))
    summary = Counter(item.phase for item in items)
    payload = {
        'schema': 'master_closure_matrix_v1',
        'txt_source': str(txt_path) if txt_path else None,
        'summary': {
            'total_items': len(items),
            'this_pass': summary['this-pass'],
            'phase_2': summary['phase-2'],
            'external': summary['external'],
            'rejected_with_reason': summary['rejected-with-reason'],
        },
        'guardrail': 'If any task increases risk to 45K readiness, reproducibility, or closure confidence, demote it to phase-2.',
        'items': [item.__dict__ for item in items],
    }

    write_json(ROOT / args.out_json, payload)
    write_markdown(ROOT / args.out_md, items, summary, txt_path)
    write_phase2(ROOT / args.phase2_md, items)
    write_risk_register(ROOT / args.risk_md, items)
    print(json.dumps(payload['summary'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
