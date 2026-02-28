from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile


def test_feature_coverage_matrix_is_complete():
    matrix = onefile.build_feature_coverage_matrix()
    assert matrix.get("schema") == "feature_coverage_matrix_v1"
    rows = matrix.get("rows", [])
    assert isinstance(rows, list)
    assert rows
    assert all(bool(r.get("implemented", False)) for r in rows)
    assert float(matrix.get("coverage_completeness_percent", 0.0)) == 100.0


def test_feature_coverage_matrix_rows_have_required_fields():
    rows = onefile.build_feature_coverage_matrix().get("rows", [])
    required = {
        "feature_id",
        "group",
        "name",
        "implemented",
        "flag_name",
        "default_state",
        "evidence_field",
        "file_anchor",
    }
    for row in rows:
        assert required.issubset(set(row.keys()))
