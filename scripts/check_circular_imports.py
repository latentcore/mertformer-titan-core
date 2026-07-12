#!/usr/bin/env python3
"""
Circular-import prevention gate.

[2026-07-12] BACKLOG I.7 #88: "Circular-import onleme kapisi". Static
import-graph analysis: parses every tracked repo-internal .py file's
top-level `import X` / `from X import Y` statements (X resolved to a
first-party module -- stdlib/third-party imports are not part of this
repo's own dependency graph and are skipped), builds a directed graph, and
reports any cycle via DFS. No import is actually executed (fast, safe,
catches cycles even in code paths that aren't exercised by the current test
suite).

Usage:
    python scripts/check_circular_imports.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent

FIRST_PARTY_PACKAGES = {
    "layers", "config", "orchestrator", "scripts", "train", "model", "utils",
    "eval", "interfaces", "mertformer_sdk", "adr",
}


def _tracked_python_files() -> List[Path]:
    out = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, encoding="utf-8")
    return [ROOT / line for line in out.splitlines() if line.strip()]


def _module_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _first_party_root(dotted: str) -> str | None:
    top = dotted.split(".", 1)[0]
    return top if top in FIRST_PARTY_PACKAGES else None


def _imports_of(path: Path) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _first_party_root(alias.name)
                if root:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import within the same package -- not a cross-package edge
            if node.module:
                root = _first_party_root(node.module)
                if root:
                    found.add(root)
    return found


def build_graph() -> Dict[str, Set[str]]:
    """Package-level graph (coarser than per-module, but that's what matters for
    catching e.g. layers <-> orchestrator style cycles; per-module would need
    full import resolution, which risks false positives from TYPE_CHECKING-only
    imports and conditional imports this scanner intentionally stays clear of)."""
    graph: Dict[str, Set[str]] = {pkg: set() for pkg in FIRST_PARTY_PACKAGES}
    for path in _tracked_python_files():
        rel = path.relative_to(ROOT)
        if not rel.parts or rel.parts[0] not in FIRST_PARTY_PACKAGES:
            continue
        if "tests" in rel.parts or ".titan-venv" in rel.parts or ".lint-venv" in rel.parts:
            continue
        own_pkg = rel.parts[0]
        for target in _imports_of(path):
            if target != own_pkg:
                graph[own_pkg].add(target)
    return graph


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    cycles: List[List[str]] = []
    visited: Set[str] = set()
    stack: List[str] = []
    on_stack: Set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(graph.get(node, set())):
            if neighbor not in graph:
                continue
            if neighbor in on_stack:
                cycle_start = stack.index(neighbor)
                cycle = stack[cycle_start:] + [neighbor]
                if sorted(cycle) not in [sorted(c) for c in cycles]:
                    cycles.append(cycle)
            elif neighbor not in visited:
                dfs(neighbor)
        stack.pop()
        on_stack.discard(node)

    for node in sorted(graph):
        if node not in visited:
            dfs(node)
    return cycles


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Static package-level circular-import scan.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Hard-fail (exit 1) instead of warn (exit 0). Off by default: package-level "
        "granularity conflates a leaf-only import (e.g. `from mertformer_sdk import "
        "__version__`, which never triggers mertformer_sdk's own heavier submodules) "
        "with a full dependency edge, so a reported cycle is not automatically a real "
        "runtime ImportError -- verify manually before treating one as a bug.",
    )
    args = parser.parse_args(argv)

    graph = build_graph()
    cycles = find_cycles(graph)

    if cycles:
        print(f"{'FAIL' if args.strict else 'WARN'}: {len(cycles)} package-level import cycle(s) found:")
        for cycle in cycles:
            print(f"  - {' -> '.join(cycle)}")
        print(
            "\nThese are package-level (not module-level) edges -- a cycle here does not "
            "necessarily mean an actual circular ImportError. Verify manually (e.g. `python3 "
            "-c 'import <pkg>'` for each involved package) before treating as a real bug."
        )
        return 1 if args.strict else 0

    print(f"OK: no package-level circular imports found across {len(graph)} first-party packages.")
    for pkg in sorted(graph):
        deps = sorted(graph[pkg])
        if deps:
            print(f"  {pkg} -> {', '.join(deps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
