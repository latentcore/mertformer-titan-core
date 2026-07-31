"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - LAYERS PACKAGE
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30 V2) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)

Public API surface for the core neural layers. Submodules remain importable
directly (e.g. ``from layers.bitlinear import BitLinear``); these re-exports
only add the package-level convenience surface.
==============================================================================
"""

from layers.bitlinear import BitLinear, activation_quant, weight_quant
from layers.ffn import MertFormerFFN
from layers.liquid import LiquidCell, LiquidMixer
from layers.mertformer_block import MertFormerBlock, RMSNorm
# NOTE: the module file is named `mla` for legacy/import-path reasons, but the
# attention class it exports is GQA (grouped-query attention), NOT latent MLA
# (multi-head latent attention). Low-rank KV bottleneck is intentionally not
# implemented; see layers/mla.py docstring. Module rename is out of scope (sealed
# import path) -- only the exported name (GQA) reflects the real mechanism.
from layers.mla import GQA, RotaryEmbedding, apply_rope_optimized, rotate_half
from layers.moe import MoE, BitSwiGLU, LiquidRouter

__all__ = [
    "BitLinear",
    "BitSwiGLU",
    "LiquidCell",
    "LiquidMixer",
    "GQA",
    "LiquidRouter",
    "MertFormerBlock",
    "MertFormerFFN",
    "MoE",
    "RMSNorm",
    "RotaryEmbedding",
    "activation_quant",
    "apply_rope_optimized",
    "rotate_half",
    "weight_quant",
]
