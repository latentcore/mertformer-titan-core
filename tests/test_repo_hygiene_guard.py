"""Tests for scripts/repo_hygiene_guard.py (the bare-except + build-fossil regression gate).

NOTE: the fossil string literals below are assembled with ``+`` (e.g. "BUILD" + "27")
on purpose — writing any old build/version fossil as one contiguous token directly
here would make this very test file trip the guard's own scan of the tree.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "repo_hygiene_guard", ROOT / "scripts" / "repo_hygiene_guard.py"
)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def test_bare_except_regex_matches_only_bare():
    assert guard._BARE_EXCEPT.match("except:")
    assert guard._BARE_EXCEPT.match("    except :")
    assert not guard._BARE_EXCEPT.match("except Exception:")
    assert not guard._BARE_EXCEPT.match("    except ValueError as e:")


def test_build_fossil_regex_matches_old_but_not_current():
    assert guard._BUILD_FOSSIL.search("BUILD" + "27")
    assert guard._BUILD_FOSSIL.search("Build" + " 28")
    assert guard._BUILD_FOSSIL.search("V27" + ".0")
    assert not guard._BUILD_FOSSIL.search("BUILD30")
    assert not guard._BUILD_FOSSIL.search("Build 30 V2")


def test_fossil_allowlist():
    assert guard._is_allowlisted_for_fossil("CHANGELOG.md")
    assert guard._is_allowlisted_for_fossil("a/docs/x.md")
    assert not guard._is_allowlisted_for_fossil("train/train.py")


def test_scan_current_tree_is_clean():
    # The whole point of the guard: the live tree stays free of both regressions.
    result = guard.scan()
    assert result["ok"], (
        f"repo hygiene regressions found: bare_except={result['bare_except']}, "
        f"build_fossils={result['build_fossils']}"
    )
