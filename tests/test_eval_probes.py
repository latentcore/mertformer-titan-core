"""Tests for the 2026-07-12 eval/*_probe.py harnesses.

Model-dependent "measured" paths were manually smoke-tested end-to-end
against a real tiny MertFormer() (MERTFORMER_MODEL_CONFIG=mertformer_small.yaml,
TITAN_USE_TR_TOKENIZER=1, --allow-random-weights) during development -- see
each probe's own module docstring for the methodology those runs exercised.
This suite covers what must stay fast, offline, and network-free in the
regular pytest run: the NO_CHECKPOINT graceful-skip contract (shared by all
six probes) and each probe's pure computational logic in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval._probe_common as common
import eval.adversarial_prompt_robustness as adversarial
import eval.bias_fairness_probe as bias
import eval.calibration_ece as calibration
import eval.hallucination_rate_probe as hallucination
import eval.membership_inference_probe as membership
import eval.toxicity_probe as toxicity

ALL_PROBE_MODULES = [calibration, adversarial, bias, toxicity, hallucination, membership]


def test_resolve_checkpoint_or_none_returns_none_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pt"
    assert common.resolve_checkpoint_or_none(str(missing)) is None


def test_resolve_checkpoint_or_none_returns_path_for_existing_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "real.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    resolved = common.resolve_checkpoint_or_none(str(ckpt))
    assert resolved == ckpt


def test_no_checkpoint_summary_has_skipped_status_and_reason_code() -> None:
    summary = common.no_checkpoint_summary("demo_schema_v1", "missing.pt", "boundary text")
    assert summary["status"] == "SKIPPED"
    assert summary["reason_code"] == "NO_CHECKPOINT"
    assert summary["schema"] == "demo_schema_v1"
    assert summary["claim_boundary"] == "boundary text"


def test_all_six_probes_skip_gracefully_without_a_checkpoint(tmp_path: Path, capsys) -> None:
    missing_ckpt = str(tmp_path / "does_not_exist.pt")
    out_dir = tmp_path / "out"
    for module in ALL_PROBE_MODULES:
        out_path = out_dir / f"{module.__name__.rsplit('.', 1)[-1]}.json"
        exit_code = module.main(["--checkpoint", missing_ckpt, "--out", str(out_path)])
        assert exit_code == 0, f"{module.__name__} should exit 0 on missing checkpoint"
        assert out_path.exists(), f"{module.__name__} should still write a SKIPPED summary"
        import json

        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written["status"] == "SKIPPED"
        assert written["reason_code"] == "NO_CHECKPOINT"


def test_adversarial_jaccard_identical_lists_is_one() -> None:
    assert adversarial._jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_adversarial_jaccard_disjoint_lists_is_zero() -> None:
    assert adversarial._jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_adversarial_jaccard_partial_overlap() -> None:
    sim = adversarial._jaccard(["a", "b", "c"], ["b", "c", "d"])
    assert sim == 2 / 4


def test_adversarial_perturb_produces_distinct_variants() -> None:
    variants = adversarial._perturb("The capital of France is")
    assert len(variants) == 3
    assert all(isinstance(v, str) and v for v in variants)


def test_membership_attack_accuracy_is_perfect_when_losses_are_cleanly_separated() -> None:
    member_losses = [0.1, 0.2, 0.3]
    nonmember_losses = [5.0, 6.0, 7.0]
    accuracy = membership._attack_accuracy(member_losses, nonmember_losses)
    assert accuracy == 1.0


def test_membership_attack_accuracy_is_near_chance_on_the_same_distribution() -> None:
    member_losses = [1.0, 2.0, 3.0, 4.0]
    nonmember_losses = [1.0, 2.0, 3.0, 4.0]
    accuracy = membership._attack_accuracy(member_losses, nonmember_losses)
    assert 0.0 <= accuracy <= 1.0


def test_membership_attack_accuracy_handles_empty_input() -> None:
    assert membership._attack_accuracy([], []) == 0.5


def test_all_probes_share_the_same_claim_boundary_pattern() -> None:
    for module in ALL_PROBE_MODULES:
        assert "CLAIM_BOUNDARY" in dir(module)
        boundary = module.CLAIM_BOUNDARY
        assert "not a capability" in boundary.lower() or "not a full" in boundary.lower()
