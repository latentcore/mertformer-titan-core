from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orchestrator import telemetry
from utils.logger import DEFAULT_STEP_CSV_FIELDS, RunLogger


def _verify_run_logger_chain(jsonl_path: Path) -> str:
    prev = hashlib.sha256(b"").hexdigest()
    final_hash = prev
    for expected_n, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        chain = record["_chain"]
        assert chain["prev"] == prev
        assert chain["n"] == expected_n

        actual_hash = chain["hash"]
        record_for_hash = dict(record)
        chain_for_hash = dict(chain)
        del chain_for_hash["hash"]
        record_for_hash["_chain"] = chain_for_hash
        canonical = json.dumps(record_for_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        h = hashlib.sha256()
        h.update(prev.encode("utf-8"))
        h.update((canonical + "\n").encode("utf-8"))
        assert h.hexdigest() == actual_hash
        prev = actual_hash
        final_hash = actual_hash
    return final_hash


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


def test_run_logger_hash_chain_is_verifiable_and_manifest_matches(tmp_path: Path) -> None:
    logger = RunLogger(
        cfg={"profile": "pytest", "tokens_seen": 123},
        log_dir=tmp_path,
        run_name="hash_contract",
    )

    logger.log_meta()
    logger.log_event("probe", {"status": "ok"})
    manifest = logger.finalize("completed", extra={"tokens_seen": 456})

    jsonl_path = tmp_path / "hash_contract.jsonl"
    final_hash = _verify_run_logger_chain(jsonl_path)
    assert manifest["final_chain_hash"] == final_hash

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["type"] == "final"
    assert records[-1]["pre_final_chain_hash"] == records[-2]["_chain"]["hash"]


def test_run_logger_redacts_secrets_without_redacting_token_counts(tmp_path: Path) -> None:
    fake_hf = "hf" + "_" + "ABCDEFGHIJKLMNOP"
    fake_sk = "sk" + "-" + "ABCDEFGHIJKLMNOPQRST"
    fake_wandb = "wandb" + "_" + "ABCDEFGHIJKLMNOP"
    logger = RunLogger(
        cfg={
            "profile": "pytest",
            "HF_TOKEN": fake_hf,
            "tokens_seen": 123,
            "nested": {"api_key": fake_sk},
        },
        log_dir=tmp_path,
        run_name="redaction_contract",
        also_csv=True,
    )

    logger.log_meta(extra={"password": "plain-password", "tokens_seen": 456})
    logger.log_event(
        "secret_event",
        {
            "message": f"{fake_hf} and {fake_sk} should not leak",
            "auth_token": "plain-auth-token",
            "tokens_seen": 789,
        },
    )
    logger.log_step({"step": 1, "loss": 1.0, "mem": fake_wandb})
    manifest = logger.finalize("completed", extra={"private_key": "plain-private-key", "tokens_seen": 999})

    jsonl_text = (tmp_path / "redaction_contract.jsonl").read_text(encoding="utf-8")
    csv_text = (tmp_path / "redaction_contract.csv").read_text(encoding="utf-8")
    manifest_text = (tmp_path / "redaction_contract.manifest.json").read_text(encoding="utf-8")
    combined = jsonl_text + csv_text + manifest_text + json.dumps(manifest, ensure_ascii=False)

    assert "hf_" not in combined
    assert "sk-" not in combined
    assert "wandb_" not in combined
    assert "plain-password" not in combined
    assert "plain-auth-token" not in combined
    assert "plain-private-key" not in combined
    assert '"tokens_seen":123' in jsonl_text
    assert '"tokens_seen":789' in jsonl_text
    assert '"tokens_seen": 999' in manifest_text
