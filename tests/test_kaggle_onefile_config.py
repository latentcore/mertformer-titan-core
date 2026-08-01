from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile


# [2026-08-01] out_dir/artifact_root must be tmp_path-scoped in the dict passed INTO
# resolve_runtime_config(), not after: RUN_CONFIG's default out_dir ("/content/mertformer_outputs")
# is a Colab-mirroring convention that, off Kaggle/Colab, falls back to a real
# ~/Downloads/content/mertformer_outputs and mkdir()s it as a side effect of the writability
# probe in resolve_writable_dir() -- the same class of bug already fixed in
# tests/test_kaggle_onefile_colab_math_fastproof.py for one test in that file; this file's
# three resolve_runtime_config() call sites were never covered by that fix and independently
# left the same stray empty directory behind on every pytest run (any OS, not Windows-specific).


def test_run_config_schema_v2_defaults_ok(tmp_path):
    base_cfg = dict(onefile.RUN_CONFIG)
    base_cfg["out_dir"] = str(tmp_path / "out")
    base_cfg["artifact_root"] = str(tmp_path / "out")
    cfg = onefile.resolve_runtime_config(base_cfg)
    report = cfg.get("run_config_schema_report", {})
    assert report.get("schema") == "run_config_schema_v2"
    assert bool(report.get("ok", False)) is True
    assert report.get("missing_required", []) == []
    assert report.get("unknown_keys", []) == []


def test_run_config_unknown_key_rejected_in_strict_mode(tmp_path):
    cfg = dict(onefile.RUN_CONFIG)
    cfg["out_dir"] = str(tmp_path / "out")
    cfg["artifact_root"] = str(tmp_path / "out")
    cfg["unknown_universal_key"] = 1
    with pytest.raises(ValueError):
        _ = onefile.resolve_runtime_config(cfg)


def test_required_core_keys_present(tmp_path):
    required = set(onefile.RUN_CONFIG_REQUIRED_KEYS)
    base_cfg = dict(onefile.RUN_CONFIG)
    base_cfg["out_dir"] = str(tmp_path / "out")
    base_cfg["artifact_root"] = str(tmp_path / "out")
    resolved = onefile.resolve_runtime_config(base_cfg)
    assert required.issubset(set(resolved.keys()))


def test_defaults_include_compile_and_determinism_contract():
    assert str(onefile.RUN_CONFIG.get("compile_policy", "")) == "off"
    assert bool(onefile.RUN_CONFIG.get("compile_fallback_on_timeout", False)) is True
    assert bool(onefile.RUN_CONFIG.get("determinism_strict", False)) is True
    assert bool(onefile.RUN_CONFIG.get("warn_nondeterministic_ops", False)) is True
