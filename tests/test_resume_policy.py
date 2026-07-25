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


# ---------------------------------------------------------------------------
# [F3, 2026-07-25] Guard/spike state persistence across resume.
#
# BACKLOG.md "F3 — new guard/spike state not persisted across resume": a crash+resume
# WHILE diverging (TITAN_AUTO_RESUME defaults ON in launch_ocean_45k.sh) used to
# re-baseline the divergence guard's reference to the already-broken loss level,
# desensitizing the brake right when it matters most. save_checkpoint_smart() now
# accepts an optional `guard_state` dict; train()'s init block restores it via
# resume_state.get("guard_state", {}).get(name, <current-default>) — additive and
# backward-compatible (old checkpoints simply lack the key).
# ---------------------------------------------------------------------------
_SAMPLE_GUARD_STATE = {
    "loss_ema": 8.7341,
    "loss_ema_observations": 512,
    "warmup_end_loss_ema": 8.4012,
    "ce_ema": 3.1105,
    "ce_ema_observations": 512,
    "warmup_end_ce_ema": 3.0509,
    "divergence_breaches": 4,
    "liquid_spike_counter": 0,
    "liquid_frozen_until": 0,
}


def test_checkpoint_persists_guard_state_when_provided(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = SimpleNamespace(save_dir=str(tmp_path), model_name="unit")

    train_mod.save_checkpoint_smart(
        model,
        optimizer,
        scheduler,
        step=5,
        cfg=cfg,
        val_loss=None,
        best_val_loss=None,
        guard_state=dict(_SAMPLE_GUARD_STATE),
    )

    state = torch.load(tmp_path / "unit_latest.pt", map_location="cpu")
    assert state["guard_state"] == _SAMPLE_GUARD_STATE


def test_checkpoint_omits_guard_state_key_when_not_provided(tmp_path) -> None:
    """Backward compatibility: callers that don't pass guard_state (or pass the
    default None) must produce a byte-identical state dict to before F3 -- no stray
    empty "guard_state": {} key that would confuse an old reader."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = SimpleNamespace(save_dir=str(tmp_path), model_name="unit")

    train_mod.save_checkpoint_smart(
        model, optimizer, scheduler, step=5, cfg=cfg, val_loss=None, best_val_loss=None,
    )

    state = torch.load(tmp_path / "unit_latest.pt", map_location="cpu")
    assert "guard_state" not in state


def test_resume_payload_round_trips_guard_state(tmp_path, monkeypatch) -> None:
    """End-to-end: save with guard_state -> _load_resume_payload -> the exact dict is
    reachable at payload["state"]["guard_state"], matching how train.py's init-block
    restoration reads it (resume_payload.get("state", {}).get("guard_state"))."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = SimpleNamespace(save_dir=str(tmp_path), model_name="unit")

    train_mod.save_checkpoint_smart(
        model,
        optimizer,
        scheduler,
        step=9,
        cfg=cfg,
        val_loss=None,
        best_val_loss=None,
        guard_state=dict(_SAMPLE_GUARD_STATE),
    )

    monkeypatch.setenv("TITAN_RESUME_FROM", str(tmp_path / "unit_latest.pt"))
    monkeypatch.delenv("TITAN_RESUME_ALLOW_PARTIAL", raising=False)
    payload = train_mod._load_resume_payload(_cfg(), model, is_main_process=False)

    assert payload is not None
    assert payload["state"].get("guard_state") == _SAMPLE_GUARD_STATE


def test_resume_from_old_checkpoint_without_guard_state_is_graceful(tmp_path, monkeypatch) -> None:
    """A checkpoint saved before F3 landed has no "guard_state" key at all. The resume
    path must not crash; the caller-side `.get("guard_state") or {}` pattern (see
    train.py's init block) is the thing under test here at the payload boundary."""
    model = nn.Linear(2, 2)
    ckpt = tmp_path / "pre_f3.pt"
    torch.save({"model": model.state_dict(), "step": 3}, ckpt)
    monkeypatch.setenv("TITAN_RESUME_FROM", str(ckpt))
    monkeypatch.delenv("TITAN_RESUME_ALLOW_PARTIAL", raising=False)

    payload = train_mod._load_resume_payload(_cfg(), model, is_main_process=False)

    assert payload is not None
    assert payload["state"].get("guard_state") is None
    # Mirrors train.py's exact restoration expression -- must not raise.
    resolved = payload["state"].get("guard_state") or {}
    assert resolved == {}


def test_train_py_init_block_restores_guard_state_from_resume_payload() -> None:
    """[F3] Source-scan regression (same discipline as the F1 call-site test above):
    the restoration logic lives inline in train()'s init block, not in an
    independently callable helper. Assert the wiring directly from source."""
    src = train_mod.__file__
    from pathlib import Path

    text = Path(src).read_text(encoding="utf-8")
    assert 'resume_payload.get("state", {}).get("guard_state")' in text
    for name in (
        "loss_ema", "loss_ema_observations", "warmup_end_loss_ema",
        "ce_ema", "ce_ema_observations", "warmup_end_ce_ema",
        "divergence_breaches", "liquid_spike_counter", "liquid_frozen_until",
    ):
        assert f'_resume_guard_state.get("{name}"' in text, f"missing restore for {name}"

    # Every save_checkpoint_smart(...) call site must pass the live snapshot through,
    # or the persistence half of the round trip is dead on arrival.
    assert text.count("guard_state=_current_guard_state()") == 6
