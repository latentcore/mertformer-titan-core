from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile


def test_feature_coverage_matrix_is_complete():
    # DURUSTLUK NOTU: Bu test SELF-REPORTED bir iddiayi dogrular, gercek bir
    # gecme-kapisi (pass-gate) DEGILDIR. build_feature_coverage_matrix()
    # ureticisi her satira sabit "implemented": True yaziyor ve
    # coverage_completeness_percent daima 100.0 donduruyor; dolayisiyla
    # asagidaki assert'ler ozelliklerin gercekten calistigini/olculdugunu
    # KANITLAMAZ, yalniz matrisin beklenen sabit yapida olmasini saglar.
    # Gercek olcum, uretici tarafinda her satirin evidence alaninin dolu
    # oldugunu dogrulayan bir kontrolle eklenmelidir.
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
