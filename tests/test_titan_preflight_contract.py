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


def _write_jsonl(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '{"text":"sample"}\n'
    path.write_text(payload * rows, encoding="utf-8")


def _prepare_offline_repo(tmp_path: Path) -> Path:
    project_root = tmp_path / "repo"
    _write_jsonl(project_root / "datasets" / "stage1" / "stage1_data.jsonl", 2)
    _write_jsonl(project_root / "datasets" / "stage2" / "stage2_data.jsonl", 2)
    _write_jsonl(project_root / "datasets" / "stage3" / "stage3_data.jsonl", 2)
    _write_jsonl(project_root / "datasets" / "stage4_soul" / "stage4_data.jsonl", 2)
    _write_jsonl(project_root / "datasets" / "stage5_tools" / "stage5_data.jsonl", 2)
    _write_jsonl(project_root / "datasets" / "validation.jsonl", 2)
    (project_root / "datasets" / "hashes.json").write_text("{}", encoding="utf-8")
    (project_root / "checkpoints" / "mertformer_titan_prod").mkdir(parents=True, exist_ok=True)
    cache_root = project_root / "data" / "tokenizer" / "tr"
    cache_root.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (cache_root / name).write_text("{}", encoding="utf-8")
    (project_root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (project_root / "tokenizer" / "tokenizer.json").write_text(
        '{"note":"Tokenizer is loaded at runtime; this file stores metadata only."}\n',
        encoding="utf-8",
    )
    (project_root / "logs" / "preflight").mkdir(parents=True, exist_ok=True)
    return project_root


def _patch_offline_cfg(monkeypatch, project_root: Path) -> None:
    monkeypatch.setattr(titan_preflight, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(titan_preflight, "LOG_DIR", project_root / "logs" / "preflight")
    monkeypatch.setattr(titan_preflight.cfg, "validation_min_samples_claim", 2)
    monkeypatch.setattr(titan_preflight.cfg, "save_dir", "./checkpoints/mertformer_titan_prod")
    monkeypatch.setattr(titan_preflight.cfg, "precomputed_logits_path", str(project_root / "datasets" / "logits"))
    monkeypatch.setattr(titan_preflight.cfg, "use_precomputed_logits", True)
    monkeypatch.setattr(titan_preflight.cfg, "require_gated_teacher", True)
    monkeypatch.setattr(titan_preflight.cfg, "teacher_model_id", "meta-llama/Llama-3.3-70B-Instruct")


def test_strict_offline_readiness_fails_without_logits_or_actionable_phase0(monkeypatch, tmp_path: Path):
    project_root = _prepare_offline_repo(tmp_path)
    _patch_offline_cfg(monkeypatch, project_root)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("TITAN_PREFLIGHT_STRICT_CUDA_LOCK", "0")

    rc = titan_preflight.strict_offline_training_readiness_profile()
    assert rc == 1
    payload = titan_preflight.json.loads(
        (project_root / "logs" / "preflight" / "train_ready_status.strict_offline_training_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["reason_code"] == "PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE"


def test_strict_offline_readiness_passes_with_complete_logits(monkeypatch, tmp_path: Path):
    project_root = _prepare_offline_repo(tmp_path)
    logits_root = project_root / "datasets" / "logits"
    logits_root.mkdir(parents=True, exist_ok=True)
    for stage in ("stage1", "stage2", "stage3", "stage4", "stage5"):
        (logits_root / f"{stage}_train_part_0.pt").write_bytes(b"stub")
        (logits_root / f"{stage}_train_topk_state.json").write_text('{"done_samples": 2}', encoding="utf-8")

    _patch_offline_cfg(monkeypatch, project_root)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("TITAN_PREFLIGHT_STRICT_CUDA_LOCK", "0")

    rc = titan_preflight.strict_offline_training_readiness_profile()
    assert rc == 0
    payload = titan_preflight.json.loads(
        (project_root / "logs" / "preflight" / "train_ready_status.strict_offline_training_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["reason_code"] == "READY_PRECOMPUTED_LOGITS_COMPLETE"


def test_strict_offline_readiness_passes_with_actionable_phase0(monkeypatch, tmp_path: Path):
    project_root = _prepare_offline_repo(tmp_path)
    _patch_offline_cfg(monkeypatch, project_root)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123456789")
    monkeypatch.setenv("TITAN_PREFLIGHT_STRICT_CUDA_LOCK", "0")

    rc = titan_preflight.strict_offline_training_readiness_profile()
    assert rc == 0
    payload = titan_preflight.json.loads(
        (project_root / "logs" / "preflight" / "train_ready_status.strict_offline_training_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["reason_code"] == "READY_ACTIONABLE_PHASE0_PRECOMPUTE"
