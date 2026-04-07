#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_JSON = ROOT / 'reports' / 'chess_teaching_contract_report.json'
REPORT_MD = ROOT / 'reports' / 'chess_teaching_contract_report.md'

import scripts.chess_5080_onefile as onefile


@dataclass(frozen=True)
class ContractCase:
    case_id: str
    title: str
    fen: str
    move: str
    expected_tags: tuple[str, ...]
    mode: str = 'teach'
    teaching_level: str = 'club'
    value: float = 0.34
    confidence_score: float = 0.61
    confidence_gap: float = 0.17
    confidence_tier: str = 'medium'


CASES: tuple[ContractCase, ...] = (
    ContractCase(
        case_id='center_control',
        title='Opening central pawn push',
        fen=onefile.chess.STARTING_FEN,
        move='e2e4',
        expected_tags=('center_control',),
    ),
    ContractCase(
        case_id='development',
        title='Knight development from the home square',
        fen=onefile.chess.STARTING_FEN,
        move='g1f3',
        expected_tags=('development',),
    ),
    ContractCase(
        case_id='capture_check',
        title='Capture that also gives check',
        fen='rnbqkbnr/pppp1ppp/8/4p2Q/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1',
        move='h5f7',
        expected_tags=('capture', 'check'),
        value=0.91,
        confidence_score=0.83,
        confidence_gap=0.44,
        confidence_tier='high',
    ),
    ContractCase(
        case_id='castle',
        title='Kingside castling move',
        fen='r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 3',
        move='e1g1',
        expected_tags=('castle',),
        mode='analyze',
        teaching_level='advanced',
        value=0.08,
        confidence_score=0.42,
        confidence_gap=0.08,
        confidence_tier='medium',
    ),
    ContractCase(
        case_id='promotion',
        title='Promotion move',
        fen='7k/P7/7K/8/8/8/8/8 w - - 0 1',
        move='a7a8q',
        expected_tags=('promotion',),
        mode='turkish_teach',
        teaching_level='basic',
        value=0.97,
        confidence_score=0.91,
        confidence_gap=0.52,
        confidence_tier='high',
    ),
)

LEVEL_SEQUENCE = ('basic', 'club', 'advanced')
MODE_SEQUENCE = ('play', 'teach', 'analyze', 'turkish_teach', 'benchmark')


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def make_trace(case: ContractCase) -> Dict[str, Any]:
    return {
        'move': case.move,
        'value': case.value,
        'latency_ms': 0.91,
        'masked_topk': [case.move, 'd2d4', 'g1f3'],
        'confidence': {
            'score': case.confidence_score,
            'gap': case.confidence_gap,
            'tier': case.confidence_tier,
        },
    }


def evaluate_case(case: ContractCase) -> Dict[str, Any]:
    board = onefile.chess.Board(case.fen)
    trace = make_trace(case)
    contract = onefile.build_chess_response_contract(
        board,
        trace,
        mode=case.mode,
        teaching_level=case.teaching_level,
    )
    problems: List[str] = []
    for tag in case.expected_tags:
        if tag not in contract['teaching_tags']:
            problems.append(f'missing_expected_tag:{tag}')
    if contract['best_move'] != case.move:
        problems.append('best_move_mismatch')
    if not contract['principal_variation'] or contract['principal_variation'][0] != case.move:
        problems.append('principal_variation_mismatch')
    if contract['mode'] != onefile.normalize_chess_response_mode(case.mode):
        problems.append('mode_not_normalized')
    if contract['teaching_level'] != onefile.normalize_teaching_level(case.teaching_level):
        problems.append('teaching_level_not_normalized')
    if contract['confidence']['tier'] not in {'low', 'medium', 'high'}:
        problems.append('confidence_tier_invalid')
    if contract['best_move_san'] not in contract['explanation_tr_short']:
        problems.append('short_missing_san')
    if contract['best_move'] not in contract['explanation_tr_long']:
        problems.append('long_missing_uci')
    if contract['evaluation']['phrase_tr'] not in contract['explanation_tr_short']:
        problems.append('short_missing_eval_phrase')
    if 'principal variation search-derinliği değil' not in contract['explanation_tr_long']:
        problems.append('long_missing_pv_disclaimer')
    all_reasons = onefile.build_teaching_reasons_tr(contract['teaching_tags'], contract['teaching_level'])
    if all_reasons and not any(reason in contract['explanation_tr_long'] for reason in all_reasons):
        problems.append('long_missing_any_reason')
    return {
        'case_id': case.case_id,
        'title': case.title,
        'fen': case.fen,
        'mode': contract['mode'],
        'teaching_level': contract['teaching_level'],
        'move': contract['best_move'],
        'san': contract['best_move_san'],
        'teaching_tags': contract['teaching_tags'],
        'confidence': contract['confidence'],
        'short': contract['explanation_tr_short'],
        'long': contract['explanation_tr_long'],
        'status': 'pass' if not problems else 'fail',
        'problems': problems,
    }


def evaluate_levels() -> List[Dict[str, Any]]:
    tags = ['development', 'center_control', 'activity', 'queen_pressure']
    counts = []
    previous = 0
    monotonic = True
    for level in LEVEL_SEQUENCE:
        reasons = onefile.build_teaching_reasons_tr(tags, level)
        count = len(reasons)
        monotonic = monotonic and count >= previous
        previous = count
        counts.append({'level': level, 'reason_count': count, 'reasons': reasons})
    return counts + [{'monotonic_non_decreasing': monotonic}]


def evaluate_modes() -> List[Dict[str, Any]]:
    board = onefile.chess.Board()
    trace = make_trace(CASES[0])
    result = []
    for mode in MODE_SEQUENCE:
        contract = onefile.build_chess_response_contract(board, trace, mode=mode, teaching_level='club')
        result.append(
            {
                'mode': mode,
                'normalized_mode': contract['mode'],
                'short_prefix': contract['explanation_tr_short'].split(':', 1)[0],
                'status': 'pass' if contract['mode'] == mode else 'fail',
            }
        )
    return result


def build_payload() -> Dict[str, Any]:
    case_results = [evaluate_case(case) for case in CASES]
    level_results = evaluate_levels()
    mode_results = evaluate_modes()
    case_pass = sum(1 for item in case_results if item['status'] == 'pass')
    mode_pass = sum(1 for item in mode_results if item['status'] == 'pass')
    level_monotonic = bool(level_results[-1]['monotonic_non_decreasing'])
    return {
        'schema': 'chess_teaching_contract_report_v1',
        'generated_utc': utc_now(),
        'contract_version': onefile.CHESS_RESPONSE_CONTRACT_VERSION,
        'summary': {
            'case_total': len(case_results),
            'case_pass': case_pass,
            'mode_total': len(mode_results),
            'mode_pass': mode_pass,
            'level_monotonic_non_decreasing': level_monotonic,
            'all_green': case_pass == len(case_results) and mode_pass == len(mode_results) and level_monotonic,
        },
        'case_results': case_results,
        'mode_results': mode_results,
        'level_results': level_results,
    }


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        '# Chess Teaching Contract Report',
        '',
        f"- generated_utc: `{payload['generated_utc']}`",
        f"- contract_version: `{payload['contract_version']}`",
        f"- all_green: `{payload['summary']['all_green']}`",
        f"- case_pass: `{payload['summary']['case_pass']}/{payload['summary']['case_total']}`",
        f"- mode_pass: `{payload['summary']['mode_pass']}/{payload['summary']['mode_total']}`",
        f"- level_monotonic_non_decreasing: `{payload['summary']['level_monotonic_non_decreasing']}`",
        '',
        '## Case Results',
        '',
        '| Case | Status | Mode | Level | Move | Tags | Problems |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for item in payload['case_results']:
        problems = ', '.join(item['problems']) if item['problems'] else '—'
        lines.append(
            f"| `{item['case_id']}` | `{item['status']}` | `{item['mode']}` | `{item['teaching_level']}` | `{item['move']}` | `{', '.join(item['teaching_tags'])}` | {problems} |"
        )
    lines.extend([
        '',
        '## Mode Results',
        '',
        '| Mode | Status | Short Prefix |',
        '| --- | --- | --- |',
    ])
    for item in payload['mode_results']:
        lines.append(f"| `{item['mode']}` | `{item['status']}` | `{item['short_prefix']}` |")
    lines.extend([
        '',
        '## Level Results',
        '',
        '| Level | Reason Count | Reasons |',
        '| --- | --- | --- |',
    ])
    for item in payload['level_results'][:-1]:
        lines.append(f"| `{item['level']}` | `{item['reason_count']}` | `{'; '.join(item['reasons'])}` |")
    lines.extend([
        '',
        f"- level_monotonic_non_decreasing: `{payload['level_results'][-1]['monotonic_non_decreasing']}`",
        '- This is a local contract and explanation-faithfulness smoke layer. It does not replace trained-model benchmark evidence.',
    ])
    return '\n'.join(lines)


def main() -> int:
    payload = build_payload()
    write_text(REPORT_JSON, json.dumps(payload, indent=2, ensure_ascii=False))
    write_text(REPORT_MD, build_markdown(payload))
    print(json.dumps(payload['summary'], ensure_ascii=False))
    return 0 if payload['summary']['all_green'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
