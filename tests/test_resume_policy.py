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


def test_periodic_checkpoint_persists_best_val_loss(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = SimpleNamespace(save_dir=str(tmp_path), model_name="unit")

    train_mod.save_checkpoint_smart(
        model,
        optimizer,
        scheduler,
        step=3,
        cfg=cfg,
        val_loss=None,
        best_val_loss=0.42,
    )

    state = torch.load(tmp_path / "unit_latest.pt", map_location="cpu")
    assert state["best_val_loss"] == 0.42
    assert "val_loss" not in state


def test_resume_payload_reads_best_val_loss(tmp_path, monkeypatch) -> None:
    model = nn.Linear(2, 2)
    ckpt = tmp_path / "best_val.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 17,
            "val_loss": 9.0,
            "best_val_loss": 1.25,
        },
        ckpt,
    )
    monkeypatch.setenv("TITAN_RESUME_FROM", str(ckpt))
    monkeypatch.delenv("TITAN_RESUME_ALLOW_PARTIAL", raising=False)

    payload = train_mod._load_resume_payload(_cfg(), model, is_main_process=False)

    assert payload is not None
    assert payload["step"] == 17
    assert payload["val_loss"] == 9.0
    assert payload["best_val_loss"] == 1.25
