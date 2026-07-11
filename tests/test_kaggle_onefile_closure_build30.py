from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    script_path = ROOT / "scripts" / "kaggle_onefile_closure_build30.py"
    spec = importlib.util.spec_from_file_location("kaggle_onefile_closure_build30_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_choose_profile_auto_t4x2_prefers_dual_t4():
    module = _load_module()
    runtime = module.RuntimeMeta(
        kaggle=True,
        colab=False,
        gpu_count=2,
        gpu_names=("Tesla T4", "Tesla T4"),
        gpu_label="GPU T4 x2",
        device="cuda",
    )
    assert module.choose_profile("auto", runtime) == "t4x2_dist"


def test_choose_profile_auto_single_t4_prefers_onecell_lane():
    module = _load_module()
    runtime = module.RuntimeMeta(
        kaggle=True,
        colab=False,
        gpu_count=1,
        gpu_names=("Tesla T4",),
        gpu_label="Tesla T4",
        device="cuda",
    )
    assert module.choose_profile("auto", runtime) == "onecell_t4_sweetspot"


def test_choose_profile_auto_p100_prefers_safe_path():
    module = _load_module()
    runtime = module.RuntimeMeta(
        kaggle=True,
        colab=False,
        gpu_count=1,
        gpu_names=("Tesla P100-PCIE-16GB",),
        gpu_label="GPU P100",
        device="cuda",
    )
    assert module.choose_profile("auto", runtime) == "p100_safe"


def test_p100_safe_max_steps_does_not_collide_with_canonical_45k():
    """[2026-07-11] p100_safe.overrides.max_steps was literally 45000 -- an exact
    numeric collision with the canonical 45K training run, despite this profile
    running a completely different (much smaller) batch_size/seq_len. A report that
    only surfaces the step count could misread a P100 probe as the real 45K run."""
    module = _load_module()
    assert module.PROFILE_SPECS["p100_safe"]["overrides"]["max_steps"] != 45000


def test_maybe_refresh_repo_posttrain_returns_status_schema_when_checkpoint_missing():
    """[2026-07-11] The checkpoint-None branch used to return learning_rate/max_steps/
    warmup_ratio hyperparameter fields instead of the ok/return_code/stdout_tail status
    fields every other run_command()-shaped result carries. Pin the real schema."""
    module = _load_module()
    result = module.maybe_refresh_repo_posttrain(None)
    assert result["cmd"] == "<skipped>"
    assert result["ok"] is False
    assert result["return_code"] != 0
    assert "stdout_tail" in result
    assert "stderr_tail" in result
    assert "learning_rate" not in result
    assert "max_steps" not in result
    assert "warmup_ratio" not in result


def test_choose_profile_auto_unknown_falls_back_to_sweetspot():
    module = _load_module()
    runtime = module.RuntimeMeta(
        kaggle=False,
        colab=False,
        gpu_count=0,
        gpu_names=(),
        gpu_label="none",
        device="cpu",
    )
    assert module.choose_profile("auto", runtime) == "sweetspot"


def test_verify_mode_writes_contract_payload(tmp_path: Path):
    report_out = tmp_path / "verify.json"
    result = __import__("subprocess").run(
        [
            "python3",
            "scripts/kaggle_onefile_closure_build30.py",
            "--mode",
            "verify",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "verify"
    assert payload["status"] == "completed"
    assert "checks" in payload
    assert payload["checks"]["legacy_build30_exists"] is True
    assert payload["checks"]["legacy_onecell_t4_exists"] is True
    assert payload["checks"]["legacy_fastproof_exists"] is True


def test_package_only_without_checkpoint_emits_warning_and_nonzero(tmp_path: Path):
    run_dir = tmp_path / "runs" / "canon_case"
    run_dir.mkdir(parents=True)
    (run_dir / "train_step_metrics.csv").write_text("step,train_loss\n1,4.0\n2,3.5\n", encoding="utf-8")
    (run_dir / "notes.txt").write_text("hello", encoding="utf-8")
    report_out = tmp_path / "package.json"
    result = __import__("subprocess").run(
        [
            "python3",
            "scripts/kaggle_onefile_closure_build30.py",
            "--mode",
            "package-only",
            "--artifact-root",
            str(tmp_path / "runs"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--run-dir",
            str(run_dir),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "package-only"
    assert payload["status"] == "warning"
    assert payload["checkpoint"] is None
    assert Path(payload["bundle_path"]).exists()
    assert Path(payload["artifact_index"]).exists()
    assert Path(payload["sha256_manifest"]).exists()


def test_find_checkpoint_prefers_manifest_entries(tmp_path: Path):
    module = _load_module()
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    latest = ckpt_dir / "latest.pt"
    best = ckpt_dir / "best.pt"
    latest.write_text("latest", encoding="utf-8")
    best.write_text("best", encoding="utf-8")
    (ckpt_dir / "manifest.json").write_text(
        json.dumps({"latest": str(latest), "best": str(best)}),
        encoding="utf-8",
    )
    layout = module.Layout(
        artifact_root=tmp_path,
        checkpoint_dir=ckpt_dir,
        run_id="x",
        run_dir=tmp_path / "run",
        auxiliary_dir=tmp_path / "run" / "aux",
        closure_dir=tmp_path / "run" / "closure",
        report_out=tmp_path / "report.json",
        canonical_summary_path=tmp_path / "run" / "closure" / "summary.json",
        canonical_summary_md_path=tmp_path / "run" / "closure" / "summary.md",
        first100_snapshot_path=tmp_path / "run" / "closure" / "first100.json",
        canonical_artifact_index_path=tmp_path / "run" / "closure" / "artifact_index.json",
        canonical_sha256_manifest_path=tmp_path / "run" / "closure" / "sha256.txt",
        canonical_package_manifest_path=tmp_path / "run" / "closure" / "package.json",
        canonical_bundle_path=tmp_path / "run" / "closure" / "bundle.zip",
    )
    resolved = module.find_checkpoint("", layout)
    assert resolved == latest


def test_first100_snapshot_parses_loss_csv(tmp_path: Path):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "sample_step_metrics.csv").write_text(
        "step,train_loss\n1,5.0\n2,4.0\n3,3.0\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "snapshot.json"
    payload = module.build_first100_snapshot(run_dir, out_path)
    assert payload["row_count"] == 3
    assert payload["first_loss"] == 5.0
    assert payload["last_loss"] == 3.0
    assert payload["loss_drop"] == 2.0
    assert out_path.exists()
