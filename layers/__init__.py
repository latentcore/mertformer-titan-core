"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - LAYERS PACKAGE
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

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
