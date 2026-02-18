from __future__ import annotations

from pathlib import Path

import scripts.test_onnx_export as onnx_smoke


class _FakeExported:
    def __init__(self):
        self.saved = False

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"onnx")
        self.saved = True


def test_onnx_program_save_fallback(monkeypatch, tmp_path: Path):
    fake = _FakeExported()

    def _fake_export(*args, **kwargs):
        return fake

    monkeypatch.setattr(onnx_smoke.torch.onnx, "export", _fake_export)
    out_path = tmp_path / "model.onnx"

    class _Wrapper:
        def __call__(self, x):
            return x

    onnx_smoke._onnx_export_compat(_Wrapper(), onnx_smoke.torch.randint(0, 10, (1, 4)), str(out_path))
    assert out_path.exists()
    assert fake.saved is True
