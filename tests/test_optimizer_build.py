"""H3: build_optimizer must honor cfg flags and log the ACTIVE class (config==reality)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from train import train as T  # noqa: E402


def _params():
    body = [torch.nn.Parameter(torch.randn(4, 4))]
    router = [torch.nn.Parameter(torch.randn(4))]
    return body, router


def test_falls_back_to_torch_adamw_when_galore_and_8bit_unavailable(monkeypatch):
    import sys
    # Force BOTH memory-efficient backends unavailable (galore=None AND
    # `import bitsandbytes` raising), regardless of what the env has installed.
    monkeypatch.setattr(T, "galore_torch", None, raising=False)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # -> import raises ImportError
    cfg = SimpleNamespace(use_galore=True, use_8bit_adam=True,
                          learning_rate=1e-4, weight_decay=0.01)
    body, router = _params()
    opt = T.build_optimizer(body, router, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2


def test_plain_adamw_when_flags_off(monkeypatch):
    monkeypatch.setattr(T, "galore_torch", None, raising=False)
    cfg = SimpleNamespace(use_galore=False, use_8bit_adam=False,
                          learning_rate=1e-4, weight_decay=0.01)
    body, router = _params()
    opt = T.build_optimizer(body, router, cfg)
    assert type(opt).__name__ == "AdamW"


def test_galore_used_when_available(monkeypatch):
    captured = {}

    class _FakeGaLore:
        def __init__(self, groups):
            captured["groups"] = groups

    fake_mod = SimpleNamespace(GaLoreAdamW8bit=_FakeGaLore, GaLoreAdamW=_FakeGaLore)
    monkeypatch.setattr(T, "galore_torch", fake_mod, raising=False)
    cfg = SimpleNamespace(use_galore=True, use_8bit_adam=True,
                          learning_rate=1e-4, weight_decay=0.01)
    body, router = _params()
    opt = T.build_optimizer(body, router, cfg)
    assert isinstance(opt, _FakeGaLore)
    # GaLore rank settings flowed into the param groups.
    assert any("rank" in g for g in captured["groups"])
