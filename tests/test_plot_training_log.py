"""Regression tests for scripts/plot_training_log.py's log parser.

[2026-07-29] The dashboard could not read the logs the trainer actually produces.
Three independent layers were broken, and because `logs/*.jsonl` only exists after a
real run, scripts/one_command_full_sop.sh reported the step as "skipped" and nobody
noticed:

  1. SHAPE. ``utils/logger.py::RunLogger.log_step`` writes step metrics FLAT at the
     top level of each record; only ``log_event`` nests its payload under ``"data"``.
     ``parse_log`` read ``entry["data"]`` exclusively, so every step record resolved
     to ``{}``, ``step`` came back ``None``, and the parser skipped all of them --
     then ``main()`` printed "No training steps found" and exited 1.
  2. NAMES. ``train/train.py`` emits the MoE telemetry as ``moe_load_entropy`` /
     ``moe_max_load`` / ``moe_capacity_overflow`` and the distillation/aux terms as
     ``distill`` / ``aux``, while the dashboard panels were written against
     ``router_entropy`` / ``router_max_load`` / ``capacity_overflow_ratio`` / ``kd`` /
     ``aux_loss``. Even with the shape fixed, the MoE Health panel stayed empty.
  3. DUPLICATES. ``train/train.py`` calls ``logger.log_step()`` twice per logged
     optimizer step (the compact ``metrics`` dict, then the richer ``log_data`` dict),
     both tagged ``type="step"`` with the same step number, so a naive append produced
     two half-populated points per step instead of one complete point.

These tests pin all three against records shaped exactly like the real logger output
(including the ``_chain`` field RunLogger appends), so a future schema drift on either
side fails here instead of silently blanking the dashboard after a paid training run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Load the script by path: it is a CLI under scripts/, not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "plot_training_log", PROJECT_ROOT / "scripts" / "plot_training_log.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _write_log(tmp_path: Path, records: list[dict]) -> str:
    path = tmp_path / "run_test.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path)


def _flat_metrics_record(step: int) -> dict:
    """Mirror of train/train.py's FIRST log_step call (the compact `metrics` dict)."""
    return {
        "type": "step",
        "timestamp_utc": "2026-07-29T00:00:00Z",
        "loss": 2.5,
        "ce": 2.4,
        "distill": 0.10,
        "aux": 0.01,
        "lr": 3e-4,
        "grad_norm": 1.2,
        "alpha": 0.8,
        "stage": 1,
        "step": step,
        "global_step": step,
        "_chain": {"prev": "0" * 64, "hash": "a" * 64, "n": step},
    }


def _flat_log_data_record(step: int) -> dict:
    """Mirror of train/train.py's SECOND log_step call (the rich `log_data` dict)."""
    return {
        "type": "step",
        "timestamp_utc": "2026-07-29T00:00:00Z",
        "step": step,
        "curriculum_stage": 1,
        "loss": 2.5,
        "ce": 2.4,
        "kd": 0.10,
        "aux": 0.01,
        "tok_s": 1234.0,
        "lr": 3e-4,
        "grad_norm": 1.2,
        "max_grad_norm": 9.0,
        "moe_max_load": 0.31,
        "moe_avg_std": 0.02,
        "moe_load_entropy": 0.72,
        "moe_capacity_overflow": 0.004,
        "_chain": {"prev": "a" * 64, "hash": "b" * 64, "n": step + 1},
    }


def test_flat_step_records_are_parsed(mod, tmp_path):
    """Layer 1: a FLAT RunLogger step record must be seen (was silently skipped)."""
    path = _write_log(tmp_path, [_flat_metrics_record(10)])
    steps, _evals, _cfg = mod.parse_log(path)
    assert steps["step"] == [10]
    assert steps["loss"] == [2.5]


def test_moe_field_aliases_are_mapped(mod, tmp_path):
    """Layer 2: train.py's moe_*/distill/aux spellings feed the dashboard keys."""
    path = _write_log(tmp_path, [_flat_log_data_record(10)])
    steps, _evals, _cfg = mod.parse_log(path)
    assert steps["router_entropy"] == [0.72]          # <- moe_load_entropy
    assert steps["router_max_load"] == [0.31]         # <- moe_max_load
    assert steps["capacity_overflow_ratio"] == [0.004]  # <- moe_capacity_overflow
    assert steps["kd"] == [0.10]
    assert steps["aux_loss"] == [0.01]                # <- aux
    assert steps["tok_s"] == [1234.0]


def test_two_log_step_calls_per_step_are_merged(mod, tmp_path):
    """Layer 3: one merged point per step carrying the union of both field sets."""
    path = _write_log(
        tmp_path, [_flat_metrics_record(10), _flat_log_data_record(10)]
    )
    steps, _evals, _cfg = mod.parse_log(path)
    assert steps["step"] == [10], "duplicate step records must merge, not append"
    # `distill` came only from the first record, `moe_load_entropy` only from the second.
    assert steps["kd"] == [0.10]
    assert steps["router_entropy"] == [0.72]
    assert steps["tok_s"] == [1234.0]


def test_nested_data_records_still_parse(mod, tmp_path):
    """Back-compat: log_event-style nested payloads must keep working."""
    path = _write_log(
        tmp_path,
        [{"type": "step", "data": {"step": 5, "loss": 3.0, "moe_max_load": 0.5}}],
    )
    steps, _evals, _cfg = mod.parse_log(path)
    assert steps["step"] == [5]
    assert steps["loss"] == [3.0]
    assert steps["router_max_load"] == [0.5]


def test_steps_are_sorted_and_eval_records_collected(mod, tmp_path):
    path = _write_log(
        tmp_path,
        [
            _flat_metrics_record(20),
            _flat_metrics_record(10),
            {"type": "eval", "step": 20, "val_loss": 2.05, "val_ppl": 7.8},
            {"type": "eval", "step": 10, "val_loss": 2.30, "val_ppl_capped": 9.9},
        ],
    )
    steps, evals, _cfg = mod.parse_log(path)
    assert steps["step"] == [10, 20], "steps must come out in ascending order"
    assert evals["step"] == [10, 20]
    assert evals["val_loss"] == [2.30, 2.05]
    assert evals["val_ppl"] == [9.9, 7.8], "val_ppl_capped takes priority over val_ppl"


def test_meta_and_malformed_lines_are_ignored(mod, tmp_path):
    path = tmp_path / "run_mixed.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "meta", "run_id": "r1"}) + "\n")
        handle.write("\n")
        handle.write("{not valid json\n")
        handle.write(json.dumps(_flat_metrics_record(1)) + "\n")
        handle.write(json.dumps({"type": "step", "loss": 1.0}) + "\n")  # no step -> skip
        handle.write(json.dumps({"type": "final", "status": "completed"}) + "\n")
    steps, _evals, _cfg = mod.parse_log(str(path))
    assert steps["step"] == [1]


def test_matplotlib_import_is_lazy(mod):
    """Parsing must not require a plotting backend.

    The module previously did `sys.exit(1)` at import when matplotlib was missing,
    which turned an absent optional dependency into a FAILED ladder step in
    scripts/one_command_full_sop.sh (matplotlib is genuinely absent from a freshly
    bootstrapped .titan-venv even though requirements.txt lists it).
    """
    assert hasattr(mod, "_require_matplotlib")
    assert callable(mod.parse_log)
