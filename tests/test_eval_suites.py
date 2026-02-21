from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from eval import agentic_suite, generalization_suite


def _required_metric_keys() -> set[str]:
    return {"metric", "baseline", "current", "delta", "confidence", "evidence_ref"}


def test_generalization_suite_emits_metric_schema():
    payload = generalization_suite.evaluate_static_stub()
    assert payload["schema"] == "generalization_suite_v1"
    assert "metrics" in payload and payload["metrics"]
    keys = set(payload["metrics"][0].keys())
    assert _required_metric_keys().issubset(keys)


def test_agentic_suite_emits_metric_schema():
    payload = agentic_suite.evaluate_static_stub()
    assert payload["schema"] == "agentic_suite_v1"
    assert "metrics" in payload and payload["metrics"]
    for item in payload["metrics"]:
        assert _required_metric_keys().issubset(set(item.keys()))
