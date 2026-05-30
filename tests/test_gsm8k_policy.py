from types import SimpleNamespace

import pytest

from eval import gsm8k


def test_gsm8k_resolves_latest_checkpoint_fallback(tmp_path) -> None:
    latest = tmp_path / "unit_latest.pt"
    latest.write_bytes(b"checkpoint")

    resolved = gsm8k._resolve_checkpoint_path(
        str(tmp_path / "missing.pt"),
        save_dir=str(tmp_path),
        model_name="unit",
    )

    assert resolved == latest


def test_gsm8k_missing_checkpoint_requires_explicit_random_flag(tmp_path) -> None:
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="--allow-random-weights"):
        gsm8k._require_checkpoint_or_allow_random(missing, allow_random_weights=False)


def test_gsm8k_random_weight_escape_hatch_is_explicit(tmp_path) -> None:
    missing = tmp_path / "missing.pt"

    gsm8k._require_checkpoint_or_allow_random(missing, allow_random_weights=True)


def test_gsm8k_run_generation_checks_checkpoint_before_dataset(monkeypatch, tmp_path) -> None:
    def fake_load_model(ckpt, allow_random_weights=False):
        raise FileNotFoundError("--allow-random-weights")

    monkeypatch.setattr(gsm8k, "_load_model_and_tokenizer", fake_load_model)
    monkeypatch.setattr(
        gsm8k,
        "_load_dataset",
        lambda: pytest.fail("dataset loaded before checkpoint policy"),
    )

    with pytest.raises(FileNotFoundError, match="--allow-random-weights"):
        gsm8k.run_generation(
            tmp_path / "outputs.jsonl",
            max_new_tokens=4,
            samples=1,
            ckpt="missing.pt",
        )


def test_gsm8k_run_generation_forwards_random_weight_policy(monkeypatch, tmp_path) -> None:
    calls = {}

    monkeypatch.setattr(gsm8k, "_load_dataset", lambda: [])

    def fake_load_model(ckpt, allow_random_weights=False):
        calls["ckpt"] = ckpt
        calls["allow_random_weights"] = allow_random_weights
        return SimpleNamespace(), SimpleNamespace(), "cpu"

    monkeypatch.setattr(gsm8k, "_load_model_and_tokenizer", fake_load_model)

    generated = gsm8k.run_generation(
        tmp_path / "outputs.jsonl",
        max_new_tokens=4,
        samples=1,
        ckpt="missing.pt",
        allow_random_weights=True,
    )

    assert generated == 0
    assert calls == {"ckpt": "missing.pt", "allow_random_weights": True}
