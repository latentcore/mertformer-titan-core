from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30 as onefile


def test_defaults_are_portable_non_strict():
    assert bool(onefile.RUN_CONFIG.get("strict_data", True)) is False
    assert bool(onefile.RUN_CONFIG.get("require_code_stage_data", True)) is False
    assert bool(onefile.RUN_CONFIG.get("allow_degraded_data", False)) is True


def test_resolve_runtime_config_falls_back_from_unwritable_absolute_paths():
    cfg = dict(onefile.RUN_CONFIG)
    cfg["out_dir"] = "/dev/null/mertformer_outputs"
    cfg["artifact_root"] = "/dev/null/mertformer_outputs"
    cfg["checkpoint_dir"] = "/dev/null/mertformer_outputs/checkpoints/kaggle_onefile_build30"

    resolved = onefile.resolve_runtime_config(cfg)
    assert str(resolved["artifact_root"]).startswith(str(Path.cwd()))
    assert str(resolved["checkpoint_dir"]).startswith(str(Path.cwd()))
    assert str(resolved["out_dir"]).startswith(str(Path.cwd()))


def test_preflight_marks_disallowed_degraded_mode(monkeypatch):
    def _always_fail_probe(*args, **kwargs):
        return False, "timeout"

    monkeypatch.setattr(onefile, "_quick_hf_dataset_probe", _always_fail_probe)
    cfg = onefile.resolve_runtime_config(dict(onefile.RUN_CONFIG))
    cfg["strict_data"] = False
    cfg["allow_degraded_data"] = False
    cfg["require_code_stage_data"] = False

    report = onefile.run_data_preflight(cfg)
    assert report["degraded_data_mode"] is True
    assert "degraded_data_not_allowed" in report["reason_codes"]
