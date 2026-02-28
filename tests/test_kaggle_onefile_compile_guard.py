from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile


def test_compile_policy_default_off_sets_env_guard(monkeypatch):
    cfg = dict(onefile.RUN_CONFIG)
    cfg["compile_policy"] = "off"
    cfg["compile_timeout_sec"] = 17.5
    rep = onefile.apply_runtime_acceleration_policy(cfg)
    assert bool(rep.get("enabled", True)) is False
    assert str(rep.get("policy", "")) == "off"
    assert os.environ.get("MERTFORMER_ONEFILE_BITNET_COMPILE") == "0"
    assert os.environ.get("MERTFORMER_ONEFILE_COMPILE_TIMEOUT_SEC") == "17.5000"


def test_compile_policy_safe_enables_compile_flag():
    cfg = dict(onefile.RUN_CONFIG)
    cfg["compile_policy"] = "safe"
    rep = onefile.apply_runtime_acceleration_policy(cfg)
    assert bool(rep.get("enabled", False)) is True
    assert str(rep.get("policy", "")) == "safe"
    assert os.environ.get("MERTFORMER_ONEFILE_BITNET_COMPILE") == "1"


def test_compile_guard_snapshot_has_required_fields():
    snap = onefile.get_compile_guard_snapshot()
    for k in (
        "enabled",
        "policy",
        "attempted",
        "compiled",
        "fallback_reason",
        "compile_elapsed_sec",
        "compile_timeout_sec",
        "cudagraph_enabled",
        "cudagraph_static_shapes_only",
    ):
        assert k in snap
