from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train import train as train_mod


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(save_dir="unused", model_name="unit")


def test_resume_mismatch_hard_fails_by_default(tmp_path, monkeypatch) -> None:
    model = nn.Linear(2, 2)
    ckpt = tmp_path / "partial.pt"
    torch.save({"model": {"weight": model.weight.detach().clone()}, "step": 7}, ckpt)
    monkeypatch.setenv("TITAN_RESUME_FROM", str(ckpt))
    monkeypatch.delenv("TITAN_RESUME_ALLOW_PARTIAL", raising=False)

    with pytest.raises(RuntimeError, match="Default closure policy requires exact model-state compatibility"):
        train_mod._load_resume_payload(_cfg(), model, is_main_process=False)


def test_resume_partial_override_is_explicit(tmp_path, monkeypatch) -> None:
    model = nn.Linear(2, 2)
    ckpt = tmp_path / "partial.pt"
    torch.save({"model": {"weight": model.weight.detach().clone()}, "step": 7}, ckpt)
    monkeypatch.setenv("TITAN_RESUME_FROM", str(ckpt))
    monkeypatch.setenv("TITAN_RESUME_ALLOW_PARTIAL", "1")

    payload = train_mod._load_resume_payload(_cfg(), model, is_main_process=False)

    assert payload is not None
    assert payload["step"] == 7
    assert payload["missing_keys"] == ["bias"]
    assert payload["unexpected_keys"] == []


def test_resume_orig_mod_prefix_variant_passes(tmp_path, monkeypatch) -> None:
    model = nn.Linear(2, 2)
    ckpt = tmp_path / "orig_mod.pt"
    torch.save(
        {
            "model": {
                "_orig_mod.weight": model.weight.detach().clone(),
                "_orig_mod.bias": model.bias.detach().clone(),
            },
            "step": 11,
        },
        ckpt,
    )
    monkeypatch.setenv("TITAN_RESUME_FROM", str(ckpt))
    monkeypatch.delenv("TITAN_RESUME_ALLOW_PARTIAL", raising=False)

    payload = train_mod._load_resume_payload(_cfg(), model, is_main_process=False)

    assert payload is not None
    assert payload["step"] == 11
    assert payload["missing_keys"] == []
    assert payload["unexpected_keys"] == []
