from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.check_bandit_security_scan as bandit_gate


def test_exclude_dirs_uses_the_dot_slash_relative_form() -> None:
    """Regression test: TWO earlier forms were tried and silently excluded
    nothing from bandit's -r . walk (a bare '.titan-venv' with no prefix, and
    an absolute path rooted at ROOT) -- both let bandit scan third-party
    site-packages and surface THEIR HIGH findings as if they were this repo's
    own code, only caught by inspecting this script's actual live runs. Only
    the `./`-relative form (matching how bandit's own walk reports paths)
    actually excludes anything -- see test below for the real, run-bandit-for-
    real proof, not just a string-shape assertion (a string-shape check alone
    is exactly what let the absolute-path regression slip through once already)."""
    for entry in bandit_gate.EXCLUDE_DIRS.split(","):
        assert entry.startswith("./"), f"exclude entry {entry!r} must start with './'"
        assert not Path(entry).is_absolute(), f"exclude entry {entry!r} must be relative, not absolute"


def test_bandit_actually_excludes_the_venv_directories_end_to_end() -> None:
    """Real, run-bandit-for-real regression test (not a string-shape check --
    see the docstring above for why a shape-only check already missed this bug
    class twice). Runs the actual scanner and asserts zero findings land inside
    .titan-venv/.lint-venv, which would only happen if the exclude flag is
    truly working against bandit's real path-matching, not just plausible-
    looking on paper."""
    report = bandit_gate.run_bandit()
    results = report.get("results", [])
    assert results, "sanity check: the real repo should produce SOME findings (it does, ~2500+)"
    venv_hits = [
        r for r in results
        if ".titan-venv" in r.get("filename", "") or ".lint-venv" in r.get("filename", "")
    ]
    assert venv_hits == [], f"exclude is not working -- {len(venv_hits)} finding(s) leaked from venv dirs"


def test_main_passes_when_no_findings_meet_the_threshold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bandit_gate, "run_bandit", lambda: {"results": []})
    exit_code = bandit_gate.main([])
    assert exit_code == 0
    assert "OK" in capsys.readouterr().out


def test_main_fails_on_a_high_severity_high_confidence_finding_by_default(monkeypatch, capsys) -> None:
    fake_results = {
        "results": [
            {
                "filename": "x.py", "line_number": 1, "issue_severity": "HIGH",
                "issue_confidence": "HIGH", "test_id": "B001", "issue_text": "bad",
            },
        ]
    }
    monkeypatch.setattr(bandit_gate, "run_bandit", lambda: fake_results)
    exit_code = bandit_gate.main([])
    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_does_not_fail_on_high_severity_low_confidence_by_default(monkeypatch, capsys) -> None:
    """Matches .github/workflows/ci.yml's `bandit -lll -iii` policy (HIGH severity AND
    HIGH confidence, not severity alone) -- a local run should not be stricter than CI."""
    fake_results = {
        "results": [
            {
                "filename": "x.py", "line_number": 1, "issue_severity": "HIGH",
                "issue_confidence": "LOW", "test_id": "B001", "issue_text": "bad",
            },
        ]
    }
    monkeypatch.setattr(bandit_gate, "run_bandit", lambda: fake_results)
    exit_code = bandit_gate.main([])
    assert exit_code == 0


def test_main_does_not_fail_on_medium_by_default(monkeypatch, capsys) -> None:
    fake_results = {
        "results": [
            {
                "filename": "x.py", "line_number": 1, "issue_severity": "MEDIUM",
                "issue_confidence": "HIGH", "test_id": "B002", "issue_text": "meh",
            },
        ]
    }
    monkeypatch.setattr(bandit_gate, "run_bandit", lambda: fake_results)
    exit_code = bandit_gate.main([])
    assert exit_code == 0


def test_main_fails_on_medium_when_fail_on_medium_is_passed(monkeypatch, capsys) -> None:
    fake_results = {
        "results": [
            {
                "filename": "x.py", "line_number": 1, "issue_severity": "MEDIUM",
                "issue_confidence": "HIGH", "test_id": "B002", "issue_text": "meh",
            },
        ]
    }
    monkeypatch.setattr(bandit_gate, "run_bandit", lambda: fake_results)
    exit_code = bandit_gate.main(["--fail-on", "medium"])
    assert exit_code == 1
