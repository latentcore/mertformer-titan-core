from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import record_dataset_hashes  # noqa: E402


def test_extract_dataset_refs_ignores_fstring_placeholder_but_keeps_real_call(tmp_path: Path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "fake_preflight.py").write_text(
        'DATASET_ID = "bigcode/the-stack-dedup"\n'
        'msg = f\'load_dataset("{DATASET_ID}", split="{DATASET_SPLIT}")\'\n'
        'real = load_dataset("bigcode/the-stack-dedup")\n',
        encoding="utf-8",
    )

    out_json = tmp_path / "inventory.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_dataset_refs.py"),
            "--root",
            str(tmp_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(tmp_path / "inventory.md"),
            "--out-md-tr",
            str(tmp_path / "inventory_TR.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    dataset_ids = {item["dataset"] for item in payload["items"]}

    assert "{DATASET_ID}" not in dataset_ids
    assert not any("{" in ds or "}" in ds for ds in dataset_ids)
    assert "bigcode/the-stack-dedup" in dataset_ids


def test_pin_datasets_isolates_a_bad_id_and_keeps_pinning_the_rest():
    class FakeApi:
        def dataset_info(self, ds, token=None, files_metadata=False):
            if ds == "{DATASET_ID}":
                raise ValueError("Repo id must use alphanumeric chars: '{DATASET_ID}'")
            return SimpleNamespace(siblings=[], sha="deadbeef", gated=False)

    sources = record_dataset_hashes._pin_datasets(
        ["{DATASET_ID}", "bigcode/the-stack-dedup"],
        FakeApi(),
        token=None,
        generated_at="2026-07-25T00:00:00Z",
    )

    assert sources["{DATASET_ID}"]["status"] == "error"
    assert "ValueError" in sources["{DATASET_ID}"]["error"]

    good = sources["bigcode/the-stack-dedup"]
    assert good["status"] == "verified"
    assert good["revision"] == "deadbeef"
