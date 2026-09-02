"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

Role: standalone nn.Linear -> BitLinear conversion utility (apply_bitnet). It is consumed by
the self-contained onefile demo (scripts/mertformer_5080_final_onefile.py) and is available
for ad-hoc conversion of an existing fp checkpoint (e.g. one trained with cfg.use_bitnet=False).

[2026-09-02 fix] This docstring previously claimed "canonical model build instantiates
BitLinear directly, so the MAIN path does NOT use this module" -- true at the time, but it
masked a real bug: layers/ffn.py, layers/mla.py, layers/moe.py and layers/liquid.py called
BitLinear(...) UNCONDITIONALLY, so cfg.use_bitnet (config.py) had NO effect on the model at
all -- the ablations/bitlinear_off ablation was structurally meaningless as a result (both
arms produced identical loss curves; confirmed empirically before this fix). Those four files
now read cfg.use_bitnet for real via layers/bitlinear.py::make_linear; this file (apply_bitnet)
is unchanged, still a separate, valid post-hoc fp->BitLinear conversion utility.
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

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
        # Whitelist check for sensitive layers.
        # 'tau' (generic) protects ALL True Liquid params (tau_input_w, tau_hidden_w).
        # A trailing '_proj' (e.g. 'gate_proj') is a standard linear projection that MUST be
        # quantized — exclude it so the 'gate' substring does not over-match and skip it.
        if any(x in name for x in ["router", "tau", "gate", "shared_expert_gate"]) and not name.endswith("_proj"):
            if verbose:
                print(f"Skipping sensitive layer: {prefix}.{name} (Keeping FP16/BF16)")
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
