from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.check_overlay_validity as checker

OVERLAY_DIR = ROOT / "config" / "model"


def test_check_overlay_passes_on_a_real_valid_overlay() -> None:
    ok, stdout, stderr = checker.check_overlay("mertformer_small.yaml")
    assert ok, f"expected mertformer_small.yaml to validate cleanly; stderr={stderr}"
    assert "OVERLAY_OK" in stdout


def test_check_overlay_fails_on_a_colliding_liquid_moe_overlay() -> None:
    broken_name = "_test_only_colliding_overlay.yaml"
    broken_path = OVERLAY_DIR / broken_name
    broken_path.write_text(
        "num_layers: 6\n"
        "use_moe: true\n"
        "moe_every_n_layers: 3\n"
        "use_liquid: true\n"
        "liquid_layers_idx: [2, 5]\n",
        encoding="utf-8",
    )
    try:
        ok, stdout, stderr = checker.check_overlay(broken_name)
        assert not ok, "colliding Liquid/MoE overlay should fail validate_layer_config()"
        assert "overlap" in stderr.lower() or "ValueError" in stderr
    finally:
        broken_path.unlink(missing_ok=True)


def test_main_reports_ok_for_all_real_repo_overlays(capsys) -> None:
    exit_code = checker.main()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "OK: all" in out
