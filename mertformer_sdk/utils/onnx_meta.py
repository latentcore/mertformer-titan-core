"""ONNX metadata helpers for bitpack hooks."""
from __future__ import annotations

from pathlib import Path


def add_bitpack_metadata(path: str | Path, meta_file: str = "titan_s25_bitpack.json") -> None:
    """Attach bitpack metadata to an ONNX file.

    Skips if onnx is not installed.
    """
    try:
        import onnx
    except Exception:
        return

    model = onnx.load(str(path))
    meta = model.metadata_props
    meta.clear()
    meta.add(key="mertformer.bitpack", value="ternary5in8")
    meta.add(key="mertformer.bitpack_meta", value=meta_file)
    onnx.save(model, str(path))
