from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.check_benchmark_regression_gate as gate


def test_compare_flags_a_regression_on_a_lower_is_better_metric() -> None:
    current = {"held_out_ppl": 15.0}
    baseline = {"held_out_ppl": 10.0}
    regressions = gate.compare(current, baseline, tolerance=0.10)
    assert len(regressions) == 1
    assert "held_out_ppl" in regressions[0]


def test_compare_passes_within_tolerance() -> None:
    current = {"held_out_ppl": 10.5}
    baseline = {"held_out_ppl": 10.0}
    assert gate.compare(current, baseline, tolerance=0.10) == []


def test_compare_flags_a_regression_on_a_higher_is_better_metric() -> None:
    current = {"adversarial_similarity": 0.5}
    baseline = {"adversarial_similarity": 0.9}
    regressions = gate.compare(current, baseline, tolerance=0.10)
    assert len(regressions) == 1
    assert "adversarial_similarity" in regressions[0]


def test_compare_skips_metrics_missing_from_either_side() -> None:
    current = {"held_out_ppl": 100.0}
    baseline = {}
    assert gate.compare(current, baseline, tolerance=0.10) == []


def test_main_skips_gracefully_without_a_checkpoint(tmp_path: Path, capsys) -> None:
    missing = str(tmp_path / "does_not_exist.pt")
    out_path = tmp_path / "out.json"
    exit_code = gate.main(["--checkpoint", missing, "--out", str(out_path)])
    assert exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["status"] == "SKIPPED"
    assert written["reason_code"] == "NO_CHECKPOINT"


def test_main_skips_gracefully_without_a_baseline(tmp_path: Path) -> None:
    ckpt = tmp_path / "real.pt"
    ckpt.write_bytes(b"fake checkpoint")
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    out_path = tmp_path / "out.json"
    baseline_path = tmp_path / "no_baseline_here.json"

    exit_code = gate.main(
        [
            "--checkpoint", str(ckpt),
            "--benchmarks-dir", str(benchmarks_dir),
            "--baseline", str(baseline_path),
            "--out", str(out_path),
        ]
    )
    assert exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["status"] == "SKIPPED"
    assert written["reason_code"] == "NO_BASELINE"


def test_main_update_baseline_writes_current_metrics(tmp_path: Path) -> None:
    ckpt = tmp_path / "real.pt"
    ckpt.write_bytes(b"fake checkpoint")
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    (benchmarks_dir / "held_out_ppl_summary.json").write_text(json.dumps({"ppl": 12.3}), encoding="utf-8")
    out_path = tmp_path / "out.json"
    baseline_path = tmp_path / "baseline.json"

    exit_code = gate.main(
        [
            "--checkpoint", str(ckpt),
            "--benchmarks-dir", str(benchmarks_dir),
            "--baseline", str(baseline_path),
            "--out", str(out_path),
            "--update-baseline",
        ]
    )
    assert exit_code == 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["held_out_ppl"] == 12.3
