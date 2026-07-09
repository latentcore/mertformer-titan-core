"""Tests for the strict config-overlay loader (config/config.py).

[2026-07-09] The overlay loader used to fail OPEN: a missing/misspelled overlay
file, malformed YAML, or an unknown key was silently ignored, so a run could start
on canonical defaults with zero warning (BACKLOG "config overlay silent no-op" —
the same silent-wrong-result class the repo fixed elsewhere in Track 2). These
tests pin the hardened fail-CLOSED behavior: an EXPLICITLY requested overlay that
is missing/malformed/non-mapping, or an unknown key, must RAISE — while the
no-overlay path and non-required loads stay exactly as before.

The last test is the load-bearing invariant: the unknown-key raise is only safe
because every shipped overlay uses real config fields. If a future overlay adds a
typo'd key, that test fails here instead of silently no-op'ing at runtime.
"""
from pathlib import Path

import pytest

import config.config as C


def test_load_yaml_missing_not_required_returns_empty(tmp_path):
    # Non-required load keeps the old tolerant behavior byte-for-byte.
    assert C._load_yaml(tmp_path / "nope.yaml") == {}


def test_load_yaml_missing_required_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        C._load_yaml(tmp_path / "nope.yaml", required=True)


def test_load_yaml_malformed_required_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        C._load_yaml(bad, required=True)


def test_load_yaml_non_mapping_required_raises(tmp_path):
    lst = tmp_path / "list.yaml"
    lst.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        C._load_yaml(lst, required=True)


def test_load_yaml_malformed_not_required_stays_tolerant(tmp_path):
    # Any OTHER (non-overlay) caller that does not pass required keeps fail-open.
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n", encoding="utf-8")
    assert C._load_yaml(bad) == {}


def test_load_yaml_empty_file_is_empty_dict(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("# only a comment\n", encoding="utf-8")
    assert C._load_yaml(empty, required=True) == {}


def test_apply_overrides_unknown_key_raises():
    cfg = C.MertFormerConfig()
    with pytest.raises(AttributeError):
        C._apply_overrides(cfg, {"definitely_not_a_real_field_xyz": 1})


def test_apply_overrides_known_key_applies():
    cfg = C.MertFormerConfig()
    C._apply_overrides(cfg, {"benchmark_profile": "full"})
    assert cfg.benchmark_profile == "full"


def test_shipped_overlays_have_only_known_keys():
    # Load-bearing invariant: the unknown-key raise is safe ONLY because every
    # shipped overlay uses real MertFormerConfig fields. Guard it here.
    import yaml

    cfg = C.MertFormerConfig()
    cdir = Path(C.__file__).resolve().parent
    for rel in [
        "base.yaml",
        "model/mertformer_max_arch.yaml",
        "model/mertformer_moe.yaml",
        "model/mertformer_small.yaml",
        "model/mertformer_pilot_stabilization.yaml",
    ]:
        p = cdir / rel
        if not p.exists():
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        unknown = [k for k in data if not hasattr(cfg, k)]
        assert not unknown, f"{rel} has unknown keys: {unknown}"
