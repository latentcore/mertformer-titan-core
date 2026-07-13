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


def test_secret_scan_detects_explicit_wandb_api_key(tmp_path: Path):
    module = _load_secret_scan_module()
    fake_key = "a" * 40
    (tmp_path / "wandb.env").write_text(f"WANDB_API_KEY={fake_key}\n", encoding="utf-8")

    mode, paths = module.discover_scan_files(tmp_path, package_mode=True)
    hits = module.scan_paths(paths, tmp_path)

    assert mode == "package_walk"
    assert len(hits) == 1
    assert hits[0][0] == "wandb_api_key"
    assert "<REDACTED>" in hits[0][3]
    assert fake_key not in hits[0][3]


def test_secret_scan_detects_contextual_40hex_token(tmp_path: Path):
    module = _load_secret_scan_module()
    fake_key = "b" * 40
    (tmp_path / "secrets.txt").write_text(f"token: '{fake_key}'\n", encoding="utf-8")

    mode, paths = module.discover_scan_files(tmp_path, package_mode=True)
    hits = module.scan_paths(paths, tmp_path)

    assert mode == "package_walk"
    assert len(hits) == 1
    assert hits[0][0] == "hex40_context"
    assert "<REDACTED>" in hits[0][3]
    assert fake_key not in hits[0][3]


def test_discover_home_directory_files_scans_documents_and_downloads(tmp_path: Path):
    module = _load_secret_scan_module()
    docs = tmp_path / "Documents"
    docs.mkdir()
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    other = tmp_path / "Desktop"
    other.mkdir()
    (docs / "chat_export.txt").write_text("token=irrelevant\n", encoding="utf-8")
    (downloads / "notes.md").write_text("nothing here\n", encoding="utf-8")
    (other / "not_scanned.txt").write_text("should not appear\n", encoding="utf-8")

    paths = module.discover_home_directory_files(home=tmp_path)

    assert sorted(p.name for p in paths) == ["chat_export.txt", "notes.md"]


def test_discover_home_directory_files_missing_dirs_returns_empty(tmp_path: Path):
    module = _load_secret_scan_module()
    assert module.discover_home_directory_files(home=tmp_path) == []


def test_discover_home_directory_files_skips_cache_and_dotdirs(tmp_path: Path):
    module = _load_secret_scan_module()
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "real.txt").write_text("fine\n", encoding="utf-8")
    hidden = docs / ".config"
    hidden.mkdir()
    (hidden / "ignored.txt").write_text("fine\n", encoding="utf-8")
    cache = docs / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_text("fine\n", encoding="utf-8")

    paths = module.discover_home_directory_files(home=tmp_path)

    assert [p.name for p in paths] == ["real.txt"]


def test_discover_home_directory_files_survives_unreadable_subdirectory(tmp_path: Path):
    module = _load_secret_scan_module()
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "real.txt").write_text("fine\n", encoding="utf-8")
    locked = docs / "locked"
    locked.mkdir()
    (locked / "inside.txt").write_text("fine\n", encoding="utf-8")
    locked.chmod(0o000)
    try:
        paths = module.discover_home_directory_files(home=tmp_path)
        assert "real.txt" in [p.name for p in paths]
    finally:
        locked.chmod(0o755)


def test_cli_include_home_dirs_flag_scans_and_reports_home_hits(tmp_path, capsys, monkeypatch):
    module = _load_secret_scan_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("no secrets\n", encoding="utf-8")
    home_root = tmp_path / "home"
    docs = home_root / "Documents"
    docs.mkdir(parents=True)
    fake_key = "a" * 40
    (docs / "leak.txt").write_text(f"WANDB_API_KEY={fake_key}\n", encoding="utf-8")

    monkeypatch.setattr(module.Path, "home", lambda: home_root)

    exit_code = module.main(["--root", str(repo_root), "--package-mode", "--include-home-dirs"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "home-directory files" in captured.out
    assert fake_key not in captured.out
    assert "<REDACTED>" in captured.out


def test_cli_without_include_home_dirs_flag_ignores_home(tmp_path, capsys, monkeypatch):
    module = _load_secret_scan_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("no secrets\n", encoding="utf-8")
    home_root = tmp_path / "home"
    docs = home_root / "Documents"
    docs.mkdir(parents=True)
    fake_key = "a" * 40
    (docs / "leak.txt").write_text(f"WANDB_API_KEY={fake_key}\n", encoding="utf-8")

    monkeypatch.setattr(module.Path, "home", lambda: home_root)

    exit_code = module.main(["--root", str(repo_root), "--package-mode"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "home-directory" not in captured.out


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
