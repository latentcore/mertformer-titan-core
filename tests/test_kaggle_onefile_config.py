from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile


def test_run_config_schema_v2_defaults_ok():
    cfg = onefile.resolve_runtime_config(dict(onefile.RUN_CONFIG))
    report = cfg.get("run_config_schema_report", {})
    assert report.get("schema") == "run_config_schema_v2"
    assert bool(report.get("ok", False)) is True
    assert report.get("missing_required", []) == []
    assert report.get("unknown_keys", []) == []


def test_run_config_unknown_key_rejected_in_strict_mode():
    cfg = dict(onefile.RUN_CONFIG)
    cfg["unknown_universal_key"] = 1
    with pytest.raises(ValueError):
        _ = onefile.resolve_runtime_config(cfg)


def test_required_core_keys_present():
    required = set(onefile.RUN_CONFIG_REQUIRED_KEYS)
    resolved = onefile.resolve_runtime_config(dict(onefile.RUN_CONFIG))
    assert required.issubset(set(resolved.keys()))


def test_defaults_include_compile_and_determinism_contract():
    assert str(onefile.RUN_CONFIG.get("compile_policy", "")) == "off"
    assert bool(onefile.RUN_CONFIG.get("compile_fallback_on_timeout", False)) is True
    assert bool(onefile.RUN_CONFIG.get("determinism_strict", False)) is True
    assert bool(onefile.RUN_CONFIG.get("warn_nondeterministic_ops", False)) is True
