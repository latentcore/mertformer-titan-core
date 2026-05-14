from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_secret_scan_module():
    script_path = ROOT / "scripts" / "secret_scan.py"
    spec = importlib.util.spec_from_file_location("secret_scan_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_secret_scan_falls_back_to_package_walk_without_git(tmp_path: Path):
    module = _load_secret_scan_module()
    (tmp_path / "README.md").write_text("no secrets here\n", encoding="utf-8")

    mode, paths = module.discover_scan_files(tmp_path)

    assert mode == "package_walk"
    assert [path.name for path in paths] == ["README.md"]


def test_secret_scan_package_walk_detects_and_redacts_secret(tmp_path: Path):
    module = _load_secret_scan_module()
    fake_key = "sk-" + "ABCDEFGHIJKL"
    (tmp_path / "config.txt").write_text(f"OPENAI_API_KEY={fake_key}\n", encoding="utf-8")

    mode, paths = module.discover_scan_files(tmp_path, package_mode=True)
    hits = module.scan_paths(paths, tmp_path)

    assert mode == "package_walk"
    assert len(hits) == 1
    assert hits[0][0] == "openai_key"
    assert "<REDACTED>" in hits[0][3]
    assert fake_key not in hits[0][3]


def test_secret_scan_package_walk_skips_cache_directories(tmp_path: Path):
    module = _load_secret_scan_module()
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    fake_key = "sk-" + "ABCDEFGHIJKL"
    (cache / "cached.py").write_text(f"OPENAI_API_KEY={fake_key}\n", encoding="utf-8")
    (tmp_path / "safe.py").write_text("print('ok')\n", encoding="utf-8")

    mode, paths = module.discover_scan_files(tmp_path, package_mode=True)
    hits = module.scan_paths(paths, tmp_path)

    assert mode == "package_walk"
    assert [path.name for path in paths] == ["safe.py"]
    assert hits == []
