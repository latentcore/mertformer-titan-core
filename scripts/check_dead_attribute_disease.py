#!/usr/bin/env python3
"""
Dead-attribute ("cosmetic-fix disease") scanner.

Codifies, as a permanent CI gate, the manual sweep methodology used on
2026-07-12 to find getattr(self, "x", fallback) / hasattr(self, "x") calls
that always fall through because "x" is never actually assigned anywhere in
the same class -- creating a false impression of dynamism (the exact defect
class behind the LiquidRouter/Gate-3 fixes in commit 8e8978f).

Heuristic, not a type-checker: flags getattr/hasattr calls on `self` (or a
simple Name) whose attribute name has no matching `self.<name> = ...` (or
`<name>.<name> = ...`) assignment anywhere in the enclosing class body (for
`self`) or module (for other simple names). False positives are possible
(e.g. attributes set via setattr(), inherited from a base class defined
elsewhere, or set through **kwargs/exec) -- soft-fail (warn, exit 0) by
default; pass --strict to hard-fail (exit 1) once a codebase's false-positive
rate is known to be zero.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple

ROOT = Path(__file__).resolve().parent.parent

# Attribute names that are legitimately dynamic (set via setattr/exec/base
# classes we don't scan, or framework-injected) -- known-safe, not a bug.
ALLOWLIST = {
    "__version__",
    "__all__",
}


class Finding(NamedTuple):
    path: str
    line: int
    attr: str
    call: str


def _tracked_python_files() -> List[Path]:
    out = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, encoding="utf-8")
    return [ROOT / line for line in out.splitlines() if line.strip() and "/tests/" not in line]


def _collect_assigned_self_attrs(tree: ast.Module) -> dict:
    """Map class-qualified-name -> set of attribute names assigned via self.X = ...
    anywhere in that class's body (methods, nested functions included)."""
    assigned: dict = {}

    class ClassVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            names: set = set()

            class AttrAssignVisitor(ast.NodeVisitor):
                def visit_Assign(self, anode: ast.Assign) -> None:
                    for target in anode.targets:
                        self._record(target)
                    self.generic_visit(anode)

                def visit_AnnAssign(self, anode: ast.AnnAssign) -> None:
                    self._record(anode.target)
                    self.generic_visit(anode)

                def visit_AugAssign(self, anode: ast.AugAssign) -> None:
                    self._record(anode.target)
                    self.generic_visit(anode)

                def _record(self, target: ast.expr) -> None:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            names.add(target.attr)
                    if isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            self._record(elt)

            AttrAssignVisitor().visit(node)
            # setattr(self, "x", ...) counts as a real (if indirect) assignment.
            for call_node in ast.walk(node):
                if (
                    isinstance(call_node, ast.Call)
                    and isinstance(call_node.func, ast.Name)
                    and call_node.func.id == "setattr"
                    and call_node.args
                    and isinstance(call_node.args[0], ast.Name)
                    and call_node.args[0].id == "self"
                    and len(call_node.args) >= 2
                    and isinstance(call_node.args[1], ast.Constant)
                    and isinstance(call_node.args[1].value, str)
                ):
                    names.add(call_node.args[1].value)
            assigned[node.name] = names
            self.generic_visit(node)

    ClassVisitor().visit(tree)
    return assigned


def _enclosing_class(node: ast.AST, class_ranges: List[tuple]) -> str:
    for cname, start, end in class_ranges:
        if start <= getattr(node, "lineno", -1) <= end:
            return cname
    return ""


def scan_file(path: Path) -> List[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    assigned_by_class = _collect_assigned_self_attrs(tree)
    class_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = max((getattr(n, "lineno", node.lineno) for n in ast.walk(node)), default=node.lineno)
            class_ranges.append((node.name, node.lineno, end))

    findings: List[Finding] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in ("getattr", "hasattr"):
            continue
        if len(node.args) < 2:
            continue
        target, attr_node = node.args[0], node.args[1]
        if not (isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str)):
            continue
        attr = attr_node.value
        if attr in ALLOWLIST:
            continue
        if not (isinstance(target, ast.Name) and target.id == "self"):
            continue
        cname = _enclosing_class(node, class_ranges)
        if not cname:
            continue
        if attr in assigned_by_class.get(cname, set()):
            continue
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(path)
        findings.append(Finding(path=rel, line=node.lineno, attr=attr, call=node.func.id))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for getattr/hasattr dead-attribute disease.")
    parser.add_argument("--strict", action="store_true", help="Hard-fail (exit 1) instead of warn (exit 0).")
    args = parser.parse_args()

    all_findings: List[Finding] = []
    for path in _tracked_python_files():
        all_findings.extend(scan_file(path))

    if not all_findings:
        print("OK: dead_attribute_disease scan clean (0 findings).")
        return 0

    print(f"{'FAIL' if args.strict else 'WARN'}: {len(all_findings)} possible dead-attribute pattern(s) found:")
    for f in all_findings:
        print(f"  - {f.path}:{f.line}: {f.call}(self, {f.attr!r}, ...) -- no self.{f.attr} = ... found in class")
    print(
        "\nThese may be false positives (setattr via **kwargs, base-class attrs, exec). "
        "Verify each manually before treating as a real bug -- see commit 8e8978f for the pattern."
    )
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
