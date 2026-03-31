from pathlib import Path

import scripts.titan_preflight as titan_preflight


def test_local_tokenizer_ready_accepts_real_local_cache(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "repo"
    (project_root / "tokenizer").mkdir(parents=True)
    (project_root / "tokenizer" / "tokenizer.json").write_text(
        '{"note":"Tokenizer is loaded at runtime; this file stores metadata only."}\n',
        encoding="utf-8",
    )

    cache_root = project_root / "data" / "tokenizer" / "tr"
    cache_root.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (cache_root / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(titan_preflight, "PROJECT_ROOT", project_root)
    ok, detail = titan_preflight._local_tokenizer_ready()
    assert ok is True
    assert "data/tokenizer/tr" in detail


def test_local_tokenizer_ready_rejects_runtime_only_metadata_without_cache(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "repo"
    (project_root / "tokenizer").mkdir(parents=True)
    (project_root / "tokenizer" / "tokenizer.json").write_text(
        '{"note":"Tokenizer is loaded at runtime; this file stores metadata only."}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(titan_preflight, "PROJECT_ROOT", project_root)
    ok, detail = titan_preflight._local_tokenizer_ready()
    assert ok is False
    assert "runtime-only" in detail
