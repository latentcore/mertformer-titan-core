from __future__ import annotations

from pathlib import Path

from orchestrator import telemetry
from utils.logger import DEFAULT_STEP_CSV_FIELDS, RunLogger


class _BrokenPsutil:
    @staticmethod
    def cpu_percent(interval=None):  # pragma: no cover - exercised through failure handling
        raise RuntimeError("cpu probe unavailable")

    @staticmethod
    def virtual_memory():  # pragma: no cover - exercised through failure handling
        raise RuntimeError("ram probe unavailable")


def test_system_snapshot_keeps_schema_when_optional_probes_are_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telemetry, "psutil", None)
    monkeypatch.setattr(telemetry, "torch", None)
    monkeypatch.setattr(telemetry.shutil, "which", lambda _: None)

    snapshot = telemetry.system_snapshot(tmp_path)

    assert tuple(snapshot.keys()) == telemetry.SYSTEM_SNAPSHOT_FIELDS
    assert isinstance(snapshot["timestamp_utc"], str)
    assert snapshot["cpu_percent"] is None
    assert snapshot["gpu_util_percent"] is None
    assert snapshot["disk_free_gb"] is not None


def test_system_snapshot_tolerates_probe_failures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telemetry, "psutil", _BrokenPsutil())
    monkeypatch.setattr(telemetry, "torch", None)
    monkeypatch.setattr(telemetry.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(telemetry.shutil, "disk_usage", lambda _: (_ for _ in ()).throw(OSError("disk unavailable")))

    snapshot = telemetry.system_snapshot(tmp_path)

    assert isinstance(snapshot["timestamp_utc"], str)
    assert snapshot["cpu_percent"] is None
    assert snapshot["ram_used_gb"] is None
    assert snapshot["gpu_power_w"] is None
    assert snapshot["disk_total_gb"] is None


def test_run_logger_default_csv_contract_handles_sparse_telemetry(tmp_path: Path) -> None:
    logger = RunLogger(
        cfg={"profile": "pytest"},
        log_dir=tmp_path,
        run_name="telemetry_contract",
        also_csv=True,
    )

    logger.log_step(
        {
            "step": 12,
            "loss": 1.25,
            "kd": 0.5,
            "tok_s": 321.0,
            "interval_sec": 8.0,
            "step_wall_sec": 2.0,
            "cpu_percent": 55.0,
            "disk_free_gb": 128.0,
            "gpu_power_w": 210.0,
            "gpu_mem_used_gb_smi": 23.5,
            "moe_max_load": 0.42,
        }
    )
    logger.finalize()

    csv_path = tmp_path / "telemetry_contract.csv"
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == DEFAULT_STEP_CSV_FIELDS

    header = rows[0].split(",")
    values = rows[1].split(",")
    row = dict(zip(header, values))
    assert row["step"] == "12"
    assert row["interval_sec"] == "8"
    assert row["gpu_power_w"] == "210"
    assert row["gpu_mem_used_gb_smi"] == "23.5"
    assert row["gpu_util_percent"] == ""
