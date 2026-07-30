"""The training-outputs bundle must ship the run's dashboard (2026-07-30).

Before this, a finished 45K run produced NO visualisation at all:

* ``scripts/plot_training_log.py`` was wired into exactly one place --
  ``scripts/one_command_full_sop.sh`` -- behind ``SOP_PLOT_TRAINING_LOG``, off by default;
* and even there it ran AFTER the ``training_outputs_bundle`` step, so a freshly rendered
  dashboard could never reach the zip (at best the previous run's stale image did);
* ``scripts/launch_ocean_45k.sh`` -- the canonical 45K lane -- called
  ``build_training_outputs_bundle.py`` directly and never invoked the SOP at all;
* ``scripts/launch_8xb300.sh`` ``exec``s ``zero_touch_start.sh`` and so has no post-run
  hook whatsoever.

The guarantee is therefore attached to the BUNDLE, which every lane passes through, rather
than to any one launcher. These tests pin that, and pin that it stays best-effort: a
charting failure must never cost the operator the outputs of a run that burned GPU hours.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_training_outputs_bundle as BUNDLE  # noqa: E402


def _write_run_log(logs_dir: Path, n_steps: int = 40) -> Path:
    """A minimal run log shaped like utils/logger.RunLogger output."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "run_test.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for step in range(1, n_steps + 1):
            handle.write(json.dumps({
                "type": "step", "step": step,
                "loss": 6.0 - step * 0.01, "ce": 4.0, "kd": 2.0,
                "aux_loss": 0.01, "router_entropy": 2.0, "router_max_load": 0.16,
                "collapse_detected": False, "grad_norm": 0.8, "lr": 3e-4,
                "tokens_per_sec": 11000.0, "tokens_seen": step * 1024,
                "capacity_overflow_ratio": 0.02,
            }) + "\n")
    return path


def _install_plotter(root: Path) -> None:
    """Copy the real plot_training_log.py into a sandbox project root."""
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "plot_training_log.py").write_bytes(
        (PROJECT_ROOT / "scripts" / "plot_training_log.py").read_bytes()
    )


def test_bundle_renders_the_dashboard_when_missing(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib
    _write_run_log(tmp_path / "logs")
    _install_plotter(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()

    status = BUNDLE.ensure_training_dashboard(tmp_path, reports)

    assert status["attempted"] is True
    assert status["rendered"] is True, status
    dashboard = reports / BUNDLE.DASHBOARD_NAME
    assert dashboard.exists() and dashboard.stat().st_size > 1000
    assert dashboard.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_dashboard_lands_inside_the_bundle_zip(tmp_path):
    """The whole point: the image must be IN the archive, not merely on disk."""
    pytest.importorskip("matplotlib")
    _write_run_log(tmp_path / "logs")
    (tmp_path / "reports").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "scripts").mkdir()
    for name in ("plot_training_log.py", "build_training_outputs_bundle.py"):
        (tmp_path / "scripts" / name).write_bytes(
            (PROJECT_ROOT / "scripts" / name).read_bytes()
        )

    proc = subprocess.run(
        [sys.executable, str(tmp_path / "scripts" / "build_training_outputs_bundle.py"),
         "--project-root", str(tmp_path),
         "--reports-dir", str(tmp_path / "reports"),
         "--artifacts-dir", str(tmp_path / "artifacts")],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr

    bundle_zip = tmp_path / "artifacts" / BUNDLE.BUNDLE_NAME
    assert bundle_zip.exists()
    with zipfile.ZipFile(bundle_zip) as archive:
        names = archive.namelist()
    assert f"reports/{BUNDLE.DASHBOARD_NAME}" in names, names


def test_rendering_is_skipped_when_already_current(tmp_path):
    """Don't re-render on every bundle: an image newer than the log is kept."""
    log_path = _write_run_log(tmp_path / "logs")
    reports = tmp_path / "reports"
    reports.mkdir()
    dashboard = reports / BUNDLE.DASHBOARD_NAME
    dashboard.write_bytes(b"pretend-png")
    import os
    newer = log_path.stat().st_mtime + 10
    os.utime(dashboard, (newer, newer))

    status = BUNDLE.ensure_training_dashboard(tmp_path, reports)

    assert status["attempted"] is False
    assert status["reason"] == "already_current"
    assert dashboard.read_bytes() == b"pretend-png"


def test_no_run_log_is_reported_not_fatal(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    status = BUNDLE.ensure_training_dashboard(tmp_path, reports)
    assert status["rendered"] is False
    assert status["reason"] == "no_run_log_found"


def test_opt_out_env_var_is_honoured(tmp_path, monkeypatch):
    _write_run_log(tmp_path / "logs")
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("TITAN_BUNDLE_SKIP_PLOT", "1")

    status = BUNDLE.ensure_training_dashboard(tmp_path, reports)

    assert status["attempted"] is False
    assert status["reason"] == "disabled_by_TITAN_BUNDLE_SKIP_PLOT"
    assert not (reports / BUNDLE.DASHBOARD_NAME).exists()


def test_a_broken_plotter_does_not_break_the_bundle(tmp_path):
    """Best-effort contract: charting must never cost a run its outputs."""
    _write_run_log(tmp_path / "logs")
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "plot_training_log.py").write_text(
        "import sys; sys.exit(3)", encoding="utf-8"
    )

    status = BUNDLE.ensure_training_dashboard(tmp_path, reports)

    assert status["rendered"] is False
    assert "no_image_produced" in status["reason"]


def test_missing_plotter_is_reported_not_raised(tmp_path):
    _write_run_log(tmp_path / "logs")
    reports = tmp_path / "reports"
    reports.mkdir()
    status = BUNDLE.ensure_training_dashboard(tmp_path, reports)
    assert status["reason"] == "plot_training_log.py_missing"


def test_sop_renders_the_dashboard_before_bundling():
    """Ordering guard on scripts/one_command_full_sop.sh.

    The plot step used to sit AFTER training_outputs_bundle, which made a fresh dashboard
    structurally unable to reach the zip. Pin the order in the script source.
    """
    sop = (PROJECT_ROOT / "scripts" / "one_command_full_sop.sh").read_text(encoding="utf-8")
    plot_at = sop.find("plot_training_log")
    bundle_at = sop.find('run_step "training_outputs_bundle"')
    assert plot_at != -1 and bundle_at != -1
    assert plot_at < bundle_at, (
        "plot_training_log must run BEFORE training_outputs_bundle, otherwise the "
        "freshly rendered dashboard cannot be included in the zip"
    )


def test_ocean_45k_launcher_renders_the_dashboard():
    """The canonical 45K lane must produce a visualisation at all."""
    launcher = (PROJECT_ROOT / "scripts" / "launch_ocean_45k.sh").read_text(encoding="utf-8")
    assert "scripts/plot_training_log.py" in launcher, (
        "launch_ocean_45k.sh does not render a training dashboard; a finished 45K run "
        "would ship an outputs bundle with no charts"
    )
    # Match the INVOCATIONS ("scripts/<name>"), not the launcher's own test-list mention
    # of tests/test_build_training_outputs_bundle.py, which appears earlier in the file.
    plot_at = launcher.find("scripts/plot_training_log.py")
    bundle_at = launcher.find("scripts/build_training_outputs_bundle.py")
    assert bundle_at != -1
    assert plot_at < bundle_at, "dashboard must be rendered before the bundle is built"
