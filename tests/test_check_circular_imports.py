from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.check_circular_imports as checker


def test_find_cycles_detects_a_two_node_cycle() -> None:
    graph = {"a": {"b"}, "b": {"a"}, "c": set()}
    cycles = checker.find_cycles(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b"}


def test_find_cycles_returns_empty_for_a_dag() -> None:
    graph = {"a": {"b"}, "b": {"c"}, "c": set()}
    assert checker.find_cycles(graph) == []


def test_find_cycles_ignores_self_edges_pointed_at_missing_nodes() -> None:
    graph = {"a": {"nonexistent"}}
    assert checker.find_cycles(graph) == []


def test_main_soft_fails_by_default_when_cycles_exist(monkeypatch, capsys) -> None:
    monkeypatch.setattr(checker, "build_graph", lambda: {"x": {"y"}, "y": {"x"}})
    exit_code = checker.main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "WARN" in out


def test_main_hard_fails_with_strict_when_cycles_exist(monkeypatch, capsys) -> None:
    monkeypatch.setattr(checker, "build_graph", lambda: {"x": {"y"}, "y": {"x"}})
    exit_code = checker.main(["--strict"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL" in out


def test_real_repo_packages_all_import_without_error() -> None:
    import importlib

    for pkg in ("layers", "mertformer_sdk", "orchestrator", "scripts", "model", "train"):
        importlib.import_module(pkg)  # raises if there were a REAL circular-import failure
