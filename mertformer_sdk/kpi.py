"""KPI report builder for pilot and release readiness."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.swarm_runtime import SwarmRuntime

from .pilot import build_pilot_report, run_verify_all


@dataclass
class KPIItem:
    key: str
    label: str
    value: float
    target: float
    status: str
    note: str


def _status(value: float, target: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "pass" if value >= target else "warn"
    return "pass" if value <= target else "warn"


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_onnx_smoke(project_root: Path) -> bool:
    cmd = [
        ".titan-venv/bin/python",
        "-m",
        "pytest",
        "-q",
        "scripts/test_onnx_export.py::test_export",
    ]
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, check=False)
    return result.returncode == 0


def collect_kpis(
    *,
    project_root: Path,
    run_verify: bool = True,
    run_onnx_check: bool = False,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    verify_summary = run_verify_all(project_root=project_root, offline=True) if run_verify else {
        "status": "skipped",
        "secret_scan_pass": False,
        "pytest_pass": False,
        "preflight_pass": False,
        "operator_gate_pass": False,
    }
    pilot = build_pilot_report(project_root=project_root, verify_summary=verify_summary)

    smoke_metrics = _load_json(project_root / "reports" / "benchmarks" / "smoke_train_metrics.json")
    kaggle_metrics = _load_json(project_root / "reports" / "benchmarks" / "kaggle_compare_build30.json")

    release_files = [
        project_root / "packages" / "MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip",
        project_root / "packages" / "MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age",
    ]
    release_total = float(len(release_files))

    swarm = SwarmRuntime()
    swarm_report = swarm.run("build30 omega readiness verification", mode="omega")
    omega_ready = 1.0 if len(swarm_report.selected_agents) == 45 and swarm_report.governance.get("allowed") else 0.0

    onnx_ok = 1.0
    onnx_note = "onnx smoke skipped"
    if run_onnx_check:
        onnx_ok = 1.0 if _run_onnx_smoke(project_root) else 0.0
        onnx_note = "onnx smoke executed"

    smoke_elapsed = float(smoke_metrics.get("elapsed_sec", 0.0)) if smoke_metrics else 0.0
    kaggle_delta = 0.0
    if kaggle_metrics:
        a = float(kaggle_metrics.get("mertformer", {}).get("final_loss", 0.0))
        b = float(kaggle_metrics.get("vanilla", {}).get("final_loss", 0.0))
        kaggle_delta = b - a

    checks = [
        KPIItem("kpi01_verify_all", "verify_all PASS", 1.0 if verify_summary.get("status") == "pass" else 0.0, 1.0, _status(1.0 if verify_summary.get("status") == "pass" else 0.0, 1.0), "offline gate"),
        KPIItem("kpi02_secret_scan", "secret scan", 1.0 if verify_summary.get("secret_scan_pass") else 0.0, 1.0, _status(1.0 if verify_summary.get("secret_scan_pass") else 0.0, 1.0), "tracked files"),
        KPIItem("kpi03_pytest", "pytest pass", 1.0 if verify_summary.get("pytest_pass") else 0.0, 1.0, _status(1.0 if verify_summary.get("pytest_pass") else 0.0, 1.0), "test gate"),
        KPIItem("kpi04_preflight", "preflight pass", 1.0 if verify_summary.get("preflight_pass") else 0.0, 1.0, _status(1.0 if verify_summary.get("preflight_pass") else 0.0, 1.0), "system drill"),
        KPIItem("kpi05_operator_gate", "operator gate pass", 1.0 if verify_summary.get("operator_gate_pass") else 0.0, 1.0, _status(1.0 if verify_summary.get("operator_gate_pass") else 0.0, 1.0), "nan/restore/failure-budget"),
        KPIItem("kpi06_pilot_schema", "pilot schema output", 1.0 if pilot.get("schema") == "pilot_report_v1" else 0.0, 1.0, _status(1.0 if pilot.get("schema") == "pilot_report_v1" else 0.0, 1.0), "contract"),
        KPIItem("kpi07_release_artifacts", "release artifacts present", float(sum(1 for p in release_files if p.exists())) / release_total, 1.0, _status(float(sum(1 for p in release_files if p.exists())) / release_total, 1.0), "zip+locked age"),
        KPIItem("kpi08_swarm_omega", "omega profile ready", omega_ready, 1.0, _status(omega_ready, 1.0), "45 agents"),
        KPIItem("kpi09_onnx_smoke", "onnx smoke", onnx_ok, 1.0, _status(onnx_ok, 1.0), onnx_note),
        KPIItem("kpi10_smoke_metrics", "smoke benchmark availability", 1.0 if smoke_metrics else 0.0, 1.0, _status(1.0 if smoke_metrics else 0.0, 1.0), f"elapsed={smoke_elapsed:.2f}s" if smoke_metrics else "missing"),
        KPIItem("kpi11_kaggle_compare", "kaggle compare availability", 1.0 if kaggle_metrics else 0.0, 1.0, _status(1.0 if kaggle_metrics else 0.0, 1.0), f"loss_delta(vanilla-mertformer)={kaggle_delta:.4f}" if kaggle_metrics else "missing"),
        KPIItem("kpi12_claim_eligibility", "benchmark claim eligibility", 1.0 if pilot.get("benchmark_eligibility", {}).get("eligible_for_claim") else 0.0, 1.0, _status(1.0 if pilot.get("benchmark_eligibility", {}).get("eligible_for_claim") else 0.0, 1.0), "checkpoint + full gates"),
    ]

    pass_count = sum(1 for item in checks if item.status == "pass")
    readiness = pass_count / float(len(checks))
    pilot_quality = {
        "score": readiness,
        "status": "ready" if readiness >= 0.8 else "needs_work",
        "gates_passed": pass_count,
        "gates_total": len(checks),
    }

    return {
        "schema": "kpi_report_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kpi_version": "1.0",
        "pass_count": pass_count,
        "total_count": len(checks),
        "readiness_score": readiness,
        "pilot_quality": pilot_quality,
        "checks": [asdict(item) for item in checks],
        "verify_summary": verify_summary,
        "pilot_report": pilot,
    }


def write_kpi_report(path: str | Path, payload: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
