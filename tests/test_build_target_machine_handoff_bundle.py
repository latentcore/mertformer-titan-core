from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / "scripts" / "build_target_machine_handoff_bundle.py"
    spec = importlib.util.spec_from_file_location("build_target_machine_handoff_bundle_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_target_machine_handoff_bundle_outputs_zip_and_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    root = tmp_path / "repo"
    reports = root / "reports"
    artifacts = root / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    for rel, body in {
        "zero_touch_start.sh": "#!/usr/bin/env bash\necho ok\n",
        "run.sh": "#!/usr/bin/env bash\necho run\n",
        "scripts/final_orchestrator.py": "print('plan')\n",
        "reports/train_readiness_decision.json": json.dumps({"final_status": "TRAIN_ALLOWED"}),
        "reports/train_readiness_decision.md": "# ready\n",
        "reports/start_gate_report.json": json.dumps({"ok": True}),
        "reports/start_gate_operator_decision.json": json.dumps(
            {
                "next_action": "ALLOCATE_TARGET_MACHINE_AND_START",
                "train_allowed": True,
                "decision_reason_code": "READY_OFFLINE_CLEAN",
                "recommended_path": "offline_clean",
                "required_transfer_files": [
                    "zero_touch_start.sh",
                    "run.sh",
                    "scripts/final_orchestrator.py",
                    "reports/train_readiness_decision.json",
                ],
            }
        ),
        "reports/start_gate_operator_decision.md": "# decision\n",
        "reports/repo_external_handoff.md": "# handoff\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "REPORTS", reports)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)
    monkeypatch.setattr(module, "MANIFEST_JSON", reports / "target_machine_handoff_manifest.json")
    monkeypatch.setattr(module, "MANIFEST_MD", reports / "target_machine_handoff_manifest.md")
    monkeypatch.setattr(module, "BUNDLE_ZIP", artifacts / "target_machine_handoff_bundle.zip")
    monkeypatch.setattr(module, "BUNDLE_SHA256", artifacts / "target_machine_handoff_bundle.zip.sha256")
    monkeypatch.setattr(module, "TRANSFER_FILE_CANDIDATES", ["zero_touch_start.sh"])

    rc = module.main()

    assert rc == 0
    assert module.MANIFEST_JSON.exists()
    assert module.MANIFEST_MD.exists()
    assert module.BUNDLE_ZIP.exists()
    assert module.BUNDLE_SHA256.exists()

    manifest = json.loads(module.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["next_action"] == "ALLOCATE_TARGET_MACHINE_AND_START"
    assert manifest["recommended_path"] == "offline_clean"
    assert len(manifest["transfer_files"]) == 4

    with zipfile.ZipFile(module.BUNDLE_ZIP) as zf:
        names = set(zf.namelist())
    assert "zero_touch_start.sh" in names
    assert "run.sh" in names
    assert "scripts/final_orchestrator.py" in names
    assert "reports/train_readiness_decision.json" in names
    assert "TARGET_MACHINE_README.md" in names
    assert "reports/target_machine_handoff_manifest.json" in names
    assert "reports/target_machine_handoff_manifest.md" in names
