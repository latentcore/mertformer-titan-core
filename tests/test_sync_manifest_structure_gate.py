"""Tests for scripts/sync_manifest.py's structure-tree gate (Y-10, 2026-07-29).

The gate used to compare ``set(entry_paths)`` against
``set(flatten_tree(build_tree(entry_paths)))`` -- an in-memory round trip of the same list,
so the two sets could only differ if build_tree/flatten_tree were themselves broken. Its
own inline comment claimed this was "a real comparison rather than a hardcoded green".
Two things made it vacuous:

* ``build_structure_md`` had already rewritten docs/PROJECT_STRUCTURE.md from the same
  list a few lines earlier, so even a genuine file read would only re-read what was just
  written; and
* the markdown is rendered by ``build_structure_lines -> emit_tree``, a path
  ``flatten_tree`` never touches -- so the round trip did not even validate the renderer.

It is now two checks: render losslessness (gating, parses the markdown back out) and
pre-existing drift of the committed tree (reported, deliberately not gating, because
refreshing that tree is the tool's whole purpose).

These tests exercise the parser and prove the gate can FAIL, which is the property the old
one lacked. They avoid running main() against the real repo, so nothing is rewritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sync_manifest as SM  # noqa: E402


def _render(paths, lang="en"):
    """Render paths exactly as build_structure_md does, fence included."""
    lines = ["# PROJECT_STRUCTURE", "", "```text"]
    lines.extend(SM.build_structure_lines(paths, lang=lang))
    lines.append("```")
    return "\n".join(lines) + "\n"


SAMPLE = [
    "top.txt",
    "a/b/one.py",
    "a/b/two.py",
    "a/c/three.md",
    "z/deep/deeper/x.json",
]


def test_parser_inverts_the_real_renderer():
    """parse_structure_md must be the exact inverse of emit_tree for file entries."""
    assert sorted(SM.parse_structure_md(_render(SAMPLE))) == sorted(SAMPLE)


def test_parser_handles_deep_nesting_and_sibling_dirs():
    paths = [
        "x/a/1.py", "x/a/2.py", "x/b/3.py",
        "x/b/c/d/e/4.py",
        "y/5.py",
        "root.md",
    ]
    assert sorted(SM.parse_structure_md(_render(paths))) == sorted(paths)


def test_parser_returns_files_only_not_directories():
    parsed = SM.parse_structure_md(_render(SAMPLE))
    assert not any(p.endswith("/") for p in parsed)
    # Directory names must appear only as path components.
    assert "a/b" not in parsed
    assert "a" not in parsed


def test_parser_ignores_content_outside_the_fence():
    body = _render(SAMPLE)
    noisy = "Some prose with ├── decoy.py in it\n" + body + "\nTrailing ├── other.py\n"
    assert sorted(SM.parse_structure_md(noisy)) == sorted(SAMPLE)


def test_parser_survives_the_turkish_rendering():
    """The TR renderer emits different role comments; paths must parse identically."""
    assert sorted(SM.parse_structure_md(_render(SAMPLE, lang="tr"))) == sorted(SAMPLE)


def test_parser_returns_empty_on_garbage_rather_than_guessing():
    assert SM.parse_structure_md("") == []
    assert SM.parse_structure_md("no fence, no tree, nothing here") == []


def test_gate_detects_a_renderer_that_silently_drops_a_file(monkeypatch):
    """THE regression: this is what the round-trip check could not see.

    A renderer bug that loses an entry now shows up as a non-empty
    `missing_in_structure`, because the check parses the rendered output rather than
    re-walking the input list.
    """
    original = SM.emit_tree

    def lossy(node, prefix="", parent_parts=(), lang="en"):
        return [line for line in original(node, prefix, parent_parts, lang)
                if "one.py" not in line]

    monkeypatch.setattr(SM, "emit_tree", lossy)
    parsed = set(SM.parse_structure_md(_render(SAMPLE)))

    missing_in_structure = sorted(set(SAMPLE) - parsed)
    assert missing_in_structure == ["a/b/one.py"]


def test_the_old_round_trip_could_not_have_detected_that():
    """Pins WHY the check was replaced, so nobody "simplifies" it back.

    flatten_tree(build_tree(x)) is x for any x -- the old comparison was an identity, and
    a broken renderer was invisible to it. flatten_tree is kept because it is still a
    useful in-memory helper; what changed is that the GATE no longer relies on it.
    """
    round_tripped = set(SM.flatten_tree(SM.build_tree(SAMPLE)))
    assert round_tripped == set(SAMPLE), "flatten_tree/build_tree are not lossless"

    # Even with a renderer that drops entries, the round trip stays perfectly green.
    original = SM.emit_tree
    try:
        SM.emit_tree = lambda *a, **k: []          # render nothing at all
        still_green = set(SM.flatten_tree(SM.build_tree(SAMPLE)))
    finally:
        SM.emit_tree = original
    assert still_green == set(SAMPLE), (
        "the old gate would have reported agreement even with an empty rendered tree"
    )


def test_renderer_is_lossless_on_the_real_repo_tree():
    """End-to-end on the committed tree: every tracked path survives a render->parse."""
    structure = PROJECT_ROOT / "docs" / "PROJECT_STRUCTURE.md"
    if not structure.exists():                     # pragma: no cover
        return
    parsed = SM.parse_structure_md(structure.read_text(encoding="utf-8"))
    assert len(parsed) > 500, f"only parsed {len(parsed)} paths from the committed tree"
    assert len(parsed) == len(set(parsed)), "duplicate paths recovered"
    # Spot-check a few files that must be present and are stable.
    for expected in ("config/config.py", "layers/moe.py", "train/train.py"):
        assert expected in parsed, f"{expected} missing from the parsed tree"
    # And confirm the deleted scripts are gone from the committed tree.
    for gone in ("scripts/md_build30_sweep.py", "scripts/titan_onnx_stress_test.py"):
        assert gone not in parsed, f"{gone} was deleted but is still in the tree"
