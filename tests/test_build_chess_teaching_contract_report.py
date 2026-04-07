from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / 'scripts' / 'build_chess_teaching_contract_report.py'
    spec = importlib.util.spec_from_file_location('build_chess_teaching_contract_report_module', script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_payload_is_all_green() -> None:
    module = _load_module()
    payload = module.build_payload()
    assert payload['summary']['all_green'] is True
    assert payload['summary']['case_pass'] == payload['summary']['case_total']
    assert payload['summary']['mode_pass'] == payload['summary']['mode_total']
    assert payload['summary']['level_monotonic_non_decreasing'] is True


def test_main_writes_reports(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, 'REPORT_JSON', tmp_path / 'reports' / 'chess_teaching_contract_report.json')
    monkeypatch.setattr(module, 'REPORT_MD', tmp_path / 'reports' / 'chess_teaching_contract_report.md')

    rc = module.main()
    payload = json.loads(module.REPORT_JSON.read_text(encoding='utf-8'))
    markdown = module.REPORT_MD.read_text(encoding='utf-8')

    assert rc == 0
    assert payload['schema'] == 'chess_teaching_contract_report_v1'
    assert payload['summary']['all_green'] is True
    assert '# Chess Teaching Contract Report' in markdown
    assert 'local contract and explanation-faithfulness smoke layer' in markdown
