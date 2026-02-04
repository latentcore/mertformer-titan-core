"""ONNX export helpers for MertFormer SDK."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def export_onnx(
    ckpt: str | Path = "latest",
    output_dir: Optional[str | Path] = None,
    bitpack: bool = False,
) -> None:
    """Export ONNX with optional bitpack metadata."""
    from scripts.mobile_export import export_production_model

    export_production_model(
        ckpt_override=ckpt,
        output_dir=output_dir,
        bitpack=bitpack,
    )
