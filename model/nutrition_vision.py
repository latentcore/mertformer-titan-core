"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - NUTRITION5K VISION SIDE-EXPERIMENT
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Bounded side experiment (see scripts/train_nutrition5k.py). NOT part of the
canonical 45K text-pretraining run and NOT wired into model/transformers.py.
Reuses the real BitNet/MoE/LiquidMixer/RMSNorm building blocks from the
canonical LM trunk on a non-causal vision (patch) sequence instead of a
token sequence.

Reuse boundary (read before touching this file or the shared layers below):
  - BitLinear (layers/bitlinear.py), MoE (layers/moe.py), MertFormerFFN
    (layers/ffn.py), LiquidMixer (layers/liquid.py) and RMSNorm
    (layers/mertformer_block.py) are imported UNMODIFIED from the canonical
    LM trunk. They operate on a generic (B, T, H) sequence and do not know
    or care whether T indexes tokens or image patches.
  - layers/mla.py's GQA hardcodes causal attention on every code path
    (verified 2026-07-17 by reading the file, not assumed: `causal=True` on
    the flash path, `is_causal=True` on the mask-free SDPA path). That is
    correct for autoregressive text and wrong for a 2D image -- patch (0,0)
    has no legitimate reason to be blind to patch (5,5). Rather than adding
    a causal-toggle to the shared, sealed GQA class (out of scope for a
    bounded side experiment touching the canonical 45K-run trunk), this file
    defines a local VisionAttention that mirrors GQA's structure (BitLinear
    q/k/v/o, QK-RMSNorm, SDPA) but is bidirectional and uses a learned
    absolute position embedding instead of RoPE (images have no natural
    rotary/sequential order; this is the standard ViT choice).
  - MoE's LiquidRouter (layers/moe.py) still applies a small CAUSAL depthwise
    Conv1d over the flattened patch sequence for its "context momentum" term
    (see LiquidRouter's own docstring in moe.py). This is a real, known
    raster-scan directional bias left in on purpose: it is inherent to the
    shared, unmodified MoE module, harmless (zero-padded, not a correctness
    bug), and documented here rather than hidden.
  - MoE and MertFormerFFN take ZERO constructor args -- they read their own
    sizing (hidden_size / num_experts / moe_intermediate / intermediate_size)
    from the GLOBAL config.config.cfg singleton at construction time. The
    caller MUST mutate cfg to the vision-appropriate values BEFORE
    constructing NutritionVisionModel (see
    scripts/train_nutrition5k.py:phase_config). This module asserts
    cfg.hidden_size matches the requested hidden_size so a missed mutation
    fails loudly instead of silently building a shape-mismatched model.
==============================================================================
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.bitlinear import BitLinear
from layers.ffn import MertFormerFFN
from layers.liquid import LiquidMixer
from layers.mertformer_block import RMSNorm
from layers.moe import MoE

NUTRITION_TARGETS: Tuple[str, ...] = ("calories", "mass", "fat", "carb", "protein")


class _QKRMSNorm(nn.Module):
    """Tiny QK-norm, duplicated locally (not imported from layers/mla.py) so this
    module never pulls in GQA's RoPE cache / cfg-coupled construction path."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms.to(x.dtype) * self.weight.to(x.dtype)


class VisionAttention(nn.Module):
    """Bidirectional GQA-style attention over a patch sequence.

    Structurally mirrors layers/mla.py:GQA (BitLinear q/k/v/o projections,
    QK-RMSNorm, SDPA) but with is_causal=False and no RoPE/KV-cache: an image
    is a fixed-size, order-free set of patches, not an incrementally decoded
    stream, so there is nothing here for RoPE or a cache to do.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = BitLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = BitLinear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = _QKRMSNorm(head_dim)
        self.k_norm = _QKRMSNorm(head_dim)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class VisionBlock(nn.Module):
    """One trunk block: bidirectional attention -> optional LiquidMixer -> MoE/dense FFN.

    Mirrors layers/mertformer_block.py:MertFormerBlock's residual/ordering
    structure, minus the KV-cache/incremental-generation plumbing (this
    model does one full forward pass per image, never decodes step-by-step).
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        use_moe_here: bool,
        use_liquid_here: bool,
        attn_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        liquid_fast_path: Optional[bool] = None,
        liquid_train_impl: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Same DeepNorm-style residual scaling as MertFormerBlock.
        self.residual_scale = (2 * num_layers) ** -0.5

        self.attn = VisionAttention(hidden_size, num_heads, num_kv_heads, head_dim, attn_dropout)

        self.is_moe_layer = bool(use_moe_here)
        self.ff = MoE() if self.is_moe_layer else MertFormerFFN()

        # Same explicit fast_path/train_impl threading as
        # layers/mertformer_block.py:MertFormerBlock -- LiquidMixer's own
        # constructor defaults (env-var-based) are NOT read from cfg, so
        # omitting these here would silently ignore cfg.liquid_fast_path /
        # cfg.liquid_train_impl set by scripts/train_nutrition5k.py:phase_config.
        self.liquid = (
            LiquidMixer(hidden_size, fast_path=liquid_fast_path, train_impl=liquid_train_impl)
            if use_liquid_here
            else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        x = x + self.attn(h) * self.residual_scale

        if self.liquid is not None:
            # One-shot forward over the whole patch sequence every time (no
            # decode loop to thread state across), so the recurrence always
            # starts cold: h_init=None.
            x, _ = self.liquid(x, h_init=None, return_state=True)

        h = self.norm2(x)
        if self.is_moe_layer:
            ff_out, aux_loss = self.ff(h)
        else:
            ff_out = self.ff(h)
            aux_loss = h.new_zeros(())
        x = x + ff_out * self.residual_scale

        if aux_loss.ndim > 0:
            aux_loss = aux_loss.sum()
        return x, aux_loss


class NutritionVisionModel(nn.Module):
    """
    Patchifies a fixed-size RGB image, runs it through N reused MertFormer
    trunk blocks (bidirectional attention + MoE/dense FFN + optional
    LiquidMixer), then predicts 5 nutrition targets from the pooled patch
    representation via 5 small independent regression heads -- mirroring the
    Nutrition5k paper's own multi-task head design (Fig. 6 of
    arXiv:2103.03375: shared trunk, per-task FC stack, single scalar output
    per task).

    See this module's top-of-file docstring for the exact reuse boundary
    (what is imported unmodified vs. what is new/local to this file).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        head_dim: int = 32,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        moe_layers: Tuple[int, ...] = (2, 5),
        liquid_layers: Tuple[int, ...] = (3,),
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        head_hidden: int = 128,
        targets: Tuple[str, ...] = NUTRITION_TARGETS,
    ) -> None:
        super().__init__()

        # Local import: keep this module importable even before the caller has
        # mutated the global config singleton (see class docstring / module
        # docstring). MoE/MertFormerFFN read cfg.hidden_size etc. at
        # *construction* time below, not at import time.
        from config.config import cfg as _cfg

        if int(_cfg.hidden_size) != int(hidden_size):
            raise ValueError(
                f"config.config.cfg.hidden_size ({_cfg.hidden_size}) != requested "
                f"hidden_size ({hidden_size}). MoE and MertFormerFFN read cfg.hidden_size "
                "(and cfg.num_experts / cfg.moe_intermediate / cfg.intermediate_size) "
                "from the global config singleton, not from constructor arguments -- "
                "mutate cfg BEFORE constructing NutritionVisionModel. See "
                "scripts/train_nutrition5k.py:phase_config()."
            )
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
        for idx in (*moe_layers, *liquid_layers):
            if not (0 <= idx < num_layers):
                raise ValueError(f"layer index {idx} out of range for num_layers={num_layers}")
        if set(moe_layers) & set(liquid_layers):
            raise ValueError(
                f"moe_layers and liquid_layers overlap on {sorted(set(moe_layers) & set(liquid_layers))} "
                "(same conflict layers/mertformer_block.py guards against for the LM trunk)."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.in_channels = in_channels
        self.targets = tuple(targets)

        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = BitLinear(patch_dim, hidden_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.drop = nn.Dropout(dropout)

        # Read once here (not per-block) and thread through explicitly --
        # same values layers/mertformer_block.py:MertFormerBlock reads via
        # getattr(cfg, "liquid_fast_path", True) / getattr(cfg,
        # "liquid_train_impl", "baseline") for the LM trunk's Liquid layers.
        liquid_fast_path = bool(getattr(_cfg, "liquid_fast_path", True))
        liquid_train_impl = str(getattr(_cfg, "liquid_train_impl", "baseline"))

        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    layer_id=i,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    use_moe_here=(i in moe_layers),
                    use_liquid_here=(i in liquid_layers),
                    attn_dropout=attn_dropout,
                    rms_norm_eps=rms_norm_eps,
                    liquid_fast_path=liquid_fast_path,
                    liquid_train_impl=liquid_train_impl,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Full-precision regression heads: BitLinear body, full-precision output
        # projection -- same precedent as model/transformers.py's tied nn.Linear
        # lm_head (see its own comment on why the final output projection stays
        # full precision rather than ternary).
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_size, head_hidden),
                    nn.SiLU(),
                    nn.Linear(head_hidden, 1),
                )
                for name in self.targets
            }
        )

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, C*patch_size*patch_size)."""
        B, C, H, W = images.shape
        p = self.patch_size
        x = images.unfold(2, p, p).unfold(3, p, p)  # (B, C, gh, gw, p, p)
        x = x.contiguous().view(B, C, self.grid, self.grid, p * p)
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, self.num_patches, C * p * p)
        return x

    def forward(self, images: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            images: (B, 3, image_size, image_size), normalized RGB.
        Returns:
            preds: dict of target_name -> (B,) raw (un-normalized-space) scalar prediction.
            aux_loss: scalar MoE aux loss summed over all MoE layers (0 if none).
        """
        x = self._patchify(images)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.drop(x)

        aux_total = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux

        x = self.final_norm(x)
        pooled = x.mean(dim=1)  # (B, H) -- no [CLS] token, plain mean pool over patches.

        preds = {name: self.heads[name](pooled).squeeze(-1) for name in self.targets}
        return preds, aux_total
