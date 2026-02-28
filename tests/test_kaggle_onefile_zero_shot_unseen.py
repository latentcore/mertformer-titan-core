from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile


_EXPR_RE = re.compile(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*$")


def _parse_prompt(prompt: str):
    m = _EXPR_RE.match(prompt)
    assert m is not None, prompt
    return int(m.group(1)), m.group(2), int(m.group(3))


def test_unseen_split_is_built_and_disjoint():
    cfg = dict(onefile.RUN_CONFIG)
    cfg.update(
        {
            "math_num_train": 128,
            "math_num_val": 32,
            "math_num_test": 32,
            "math_num_unseen": 64,
            "eval_unseen_enabled": True,
            "math_min_value": -200,
            "math_max_value": 200,
            "eval_unseen_min": 500,
            "eval_unseen_max": 900,
            "math_ops": ["+", "-", "*", "/"],
        }
    )
    bundle = onefile.mathfp_build_datasets(cfg)
    unseen = list(bundle.get("unseen", []))
    assert unseen

    seen_keys = set()
    for part in ("train", "val", "test"):
        for row in bundle.get(part, []):
            seen_keys.add((str(row.get("prompt", "")), str(row.get("answer", ""))))

    unseen_keys = {(str(r.get("prompt", "")), str(r.get("answer", ""))) for r in unseen}
    assert seen_keys.isdisjoint(unseen_keys)


def test_unseen_operands_respect_unseen_magnitude_band():
    cfg = dict(onefile.RUN_CONFIG)
    cfg["math_num_unseen"] = 48
    cfg["eval_unseen_enabled"] = True
    cfg["eval_unseen_min"] = 500
    cfg["eval_unseen_max"] = 900
    bundle = onefile.mathfp_build_datasets(cfg)
    unseen = list(bundle.get("unseen", []))
    assert unseen

    low = int(cfg["eval_unseen_min"])
    high = int(cfg["eval_unseen_max"])
    for row in unseen:
        a, _op, b = _parse_prompt(str(row.get("prompt", "")))
        assert low <= abs(int(a)) <= high
        assert low <= abs(int(b)) <= high
