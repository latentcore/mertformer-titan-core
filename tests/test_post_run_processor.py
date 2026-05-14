from __future__ import annotations

import json
from pathlib import Path
import zipfile

from scripts.post_run_processor import process_ocean_output


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_post_run_processor_accepts_final_evidence_pack(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    dest = tmp_path / "processed"
    (outputs / "logs").mkdir(parents=True)
    (outputs / "eval").mkdir()
    (outputs / "checkpoints" / "final_math_h200").mkdir(parents=True)

    _write_json(
        outputs / "proof_decision.json",
        {
            "verdict": "OPERATIONAL_PASS",
            "reason_codes": [],
            "learning_signal_pass": False,
            "capability_claim_eligible": False,
        },
    )
    (outputs / "proof_decision.md").write_text("# Proof Decision\n", encoding="utf-8")
    _write_json(outputs / "final_math_h200_manifest.json", {"proof_version": "test"})
    _write_json(outputs / "final_math_h200_report.json", {"proof_version": "test"})
    (outputs / "eval" / "eval_history.csv").write_text(
        "step,exact_accuracy\n1,0.0\n5,0.1\n",
        encoding="utf-8",
    )
    (outputs / "logs" / "final_math_h200.jsonl").write_text(
        "\n".join(
            [
                '{"event":"training_start"}',
                '{"event":"step","step":1}',
                '{"event":"checkpoint_save","step":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (outputs / "checkpoints" / "final_math_h200" / "last.pt").write_bytes(b"last")
    (outputs / "checkpoints" / "final_math_h200" / "final.pt").write_bytes(b"final")
    (outputs / "github_visibility_audit_20260514.md").write_text("# GitHub Audit\n", encoding="utf-8")
    (outputs / "sha256.txt").write_text(
        "\n".join(
            [
                "0" * 64 + "  proof_decision.json",
                "1" * 64 + "  proof_decision.md",
                "2" * 64 + "  final_math_h200_manifest.json",
                "3" * 64 + "  final_math_h200_report.json",
                "4" * 64 + "  sha256.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(outputs / "mertformer_final_evidence_proof_outputs.zip", "w") as zf:
        zf.writestr("proof_decision.json", "{}")

    summary = process_ocean_output(outputs, dest)

    assert summary["post_run_fail_codes"] == []
    assert summary["claim_boundary"]["operational_evidence_supported"] is True
    assert summary["claim_boundary"]["capability_claim_supported"] is False
    assert (dest / "post_run_summary.json").exists()
    assert (dest / "post_run_claim_report.md").exists()
    assert (dest / "evidence_pack_manifest.json").exists()


def test_post_run_processor_reports_missing_step_progress(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    dest = tmp_path / "processed"
    outputs.mkdir()
    _write_json(outputs / "proof_decision.json", {"verdict": "OPERATIONAL_PASS"})

    summary = process_ocean_output(outputs, dest)

    assert "NO_STEP_PROGRESS" in summary["post_run_fail_codes"]
    assert summary["claim_boundary"]["operational_evidence_supported"] is False
