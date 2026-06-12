"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

Role: standalone nn.Linear -> BitLinear conversion utility (apply_bitnet). The canonical
model build instantiates BitLinear directly, so the MAIN path does NOT use this module; it
is consumed by the self-contained onefile demo (scripts/mertformer_5080_final_onefile.py) and
is available for ad-hoc conversion of an existing fp checkpoint.
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import torch
import torch.nn as nn
from layers.bitlinear import BitLinear


def _convert_linear_modules(
    module: nn.Module,
    prefix: str = "",
    skip_output_head: bool = True,
    verbose: bool = True,
):
    for name, child in list(module.named_children()):
        # V21.0 FIX: Whitelist check for sensitive layers
        # V25.1 UPDATE: Added 'tau' (generic) to protect ALL True Liquid params (tau_input_w, tau_hidden_w)
        if any(x in name for x in ["router", "tau", "gate", "shared_expert_gate"]):
            if verbose:
                print(f"Skipping sensitive layer (V21.0 FIX): {prefix}.{name} (Keeping FP16/BF16)")
            continue

        child_prefix = f"{prefix}.{name}" if prefix else name

        # Recursive conversion
        _convert_linear_modules(
            child,
            prefix=child_prefix,
            skip_output_head=skip_output_head,
            verbose=verbose,
        )

        if isinstance(child, nn.Linear):

            # Skip output layers like lm_head
            if skip_output_head and name in ("lm_head", "output_head", "classifier"):
                if verbose:
                    print(f"[BitNet] Skip output head: {child_prefix}")
                continue

            in_f = child.in_features
            out_f = child.out_features
            use_bias = child.bias is not None
            device = child.weight.device
            dtype = child.weight.dtype

            new_layer = BitLinear(
                in_features=in_f,
                out_features=out_f,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )

            # Copy weights
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if use_bias:
                    new_layer.bias.copy_(child.bias)

            setattr(module, name, new_layer)

            if verbose:
                print(f"[BitNet] Linear → BitLinear: {child_prefix}")


def apply_bitnet(
    model: nn.Module,
    skip_output_head: bool = True,
    verbose: bool = True,
) -> nn.Module:

    _convert_linear_modules(
        model,
        skip_output_head=skip_output_head,
        verbose=verbose,
    )

    if verbose:
        print("[BitNet] All eligible nn.Linear layers have been converted to BitLinear.")

    return model
