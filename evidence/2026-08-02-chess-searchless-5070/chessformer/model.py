"""ChessFormer: canonical MertFormer trunk + chess-specific heads.

The trunk is the mirrored Build30 stack (``chessformer/arch``). What changes
here is the *head*, which is where the upstream onefile was weakest.

POLICY HEAD
-----------
``scripts/chess_5080_onefile.py`` mean-pooled all 76 tokens into one vector and
fed a single ``Linear(hidden, vocab)``. Averaging destroys the board's spatial
structure before the head ever sees it: the layer that has to answer "which
piece moves where" gets a bag of features with no notion of *where*.

Here the head is factorized over squares, which is the natural structure of a
chess move::

    logit(move) = <from_proj(h[from]), to_proj(h[to])> / sqrt(d)
                + promo_proj(h[from])[promo_slot]
                + move_bias[move]

Every move's score reads the representation of its own origin and destination
square. The public ``MOVE_VOCAB`` (4208 UCI moves) is unchanged, so the
checkpoint/GUI contract still holds.

VALUE HEAD
----------
128 uniform bins over win probability (DeepMind arXiv:2402.04494 used 128 bins
+ cross-entropy) trained with HL-Gauss soft targets (arXiv:2403.03950).
Prediction is the expectation over bin centres, so it stays a scalar win
probability at inference.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .arch import MertFormerBlock, RMSNorm, make_linear
from .board import (
    MOVE_FROM_SQ,
    MOVE_PROMO,
    MOVE_TO_SQ,
    NUM_PROMO_SLOTS,
    VOCAB_SIZE,
    bin_centers,
)
from .config import (
    META_CARDINALITIES,
    NUM_META_TOKENS,
    NUM_SQUARE_TOKENS,
    SEQ_LEN,
    ModelConfig,
)


class FactorizedPolicyHead(nn.Module):
    """Scores every move from its origin- and destination-square tokens."""

    def __init__(self, hidden_size: int, head_dim: int, vocab_size: int, use_bitnet: bool = False) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.vocab_size = int(vocab_size)
        self.from_proj = make_linear(use_bitnet, hidden_size, head_dim)
        self.to_proj = make_linear(use_bitnet, hidden_size, head_dim)
        self.promo_proj = make_linear(use_bitnet, hidden_size, NUM_PROMO_SLOTS)
        self.move_bias = nn.Parameter(torch.zeros(vocab_size))
        self.scale = 1.0 / math.sqrt(float(head_dim))

        # Flat gather indices, precomputed once.
        from_sq = torch.from_numpy(MOVE_FROM_SQ.astype(np.int64))
        to_sq = torch.from_numpy(MOVE_TO_SQ.astype(np.int64))
        promo = torch.from_numpy(MOVE_PROMO.astype(np.int64))
        self.register_buffer("fromto_index", from_sq * NUM_SQUARE_TOKENS + to_sq, persistent=False)
        self.register_buffer("promo_index", from_sq * NUM_PROMO_SLOTS + promo, persistent=False)

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        B = square_tokens.size(0)
        q = self.from_proj(square_tokens)                       # [B, 64, d]
        k = self.to_proj(square_tokens)                         # [B, 64, d]
        fromto = torch.matmul(q, k.transpose(1, 2)) * self.scale  # [B, 64, 64]
        promo = self.promo_proj(square_tokens)                  # [B, 64, 5]

        fromto_flat = fromto.reshape(B, NUM_SQUARE_TOKENS * NUM_SQUARE_TOKENS)
        promo_flat = promo.reshape(B, NUM_SQUARE_TOKENS * NUM_PROMO_SLOTS)

        idx_ft = self.fromto_index.unsqueeze(0).expand(B, -1)
        idx_pr = self.promo_index.unsqueeze(0).expand(B, -1)

        logits = fromto_flat.gather(1, idx_ft) + promo_flat.gather(1, idx_pr)
        return logits + self.move_bias


class PooledPolicyHead(nn.Module):
    """The upstream head, kept behind a flag so the two can be compared."""

    def __init__(self, hidden_size: int, vocab_size: int, use_bitnet: bool = False) -> None:
        super().__init__()
        self.proj = make_linear(use_bitnet, hidden_size, vocab_size, bias=True)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


class ChessFormer(nn.Module):
    """Board encoder with policy / value / WDL / legality heads."""

    def __init__(self, cfg: ModelConfig, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        h = int(cfg.hidden_size)

        # --- embeddings ---
        self.piece_embed = nn.Embedding(13, h)          # 0 = empty, 1..12 = pieces
        self.square_embed = nn.Embedding(NUM_SQUARE_TOKENS, h)
        self.meta_type_embed = nn.Embedding(NUM_META_TOKENS, h)
        self.meta_value_embeds = nn.ModuleList(
            nn.Embedding(card, h) for card in META_CARDINALITIES
        )
        self.embed_dropout = nn.Dropout(float(cfg.dropout))

        # --- trunk ---
        self.blocks = nn.ModuleList(
            MertFormerBlock(layer_id=i, cfg=cfg) for i in range(int(cfg.num_layers))
        )
        self.final_norm = RMSNorm(h, eps=float(cfg.rms_norm_eps))

        # --- heads ---
        self.policy_head_kind = cfg.policy_head
        if cfg.policy_head == "factorized":
            self.policy_head = FactorizedPolicyHead(
                h, int(cfg.policy_head_dim), self.vocab_size, use_bitnet=bool(cfg.use_bitnet)
            )
        else:
            self.policy_head = PooledPolicyHead(h, self.vocab_size, use_bitnet=bool(cfg.use_bitnet))

        self.value_head = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, int(cfg.num_value_bins)),
        )
        self.wdl_head = nn.Linear(h, 3) if cfg.use_wdl_head else None
        self.legality_head = (
            FactorizedPolicyHead(h, max(32, int(cfg.policy_head_dim) // 2), self.vocab_size,
                                 use_bitnet=bool(cfg.use_bitnet))
            if cfg.use_legality_head
            else None
        )

        self.register_buffer(
            "value_bin_centers",
            torch.from_numpy(bin_centers(int(cfg.num_value_bins))),
            persistent=False,
        )
        self._square_index: torch.Tensor
        self.register_buffer(
            "_square_index", torch.arange(NUM_SQUARE_TOKENS, dtype=torch.long), persistent=False
        )
        self._meta_index: torch.Tensor
        self.register_buffer(
            "_meta_index", torch.arange(NUM_META_TOKENS, dtype=torch.long), persistent=False
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- forward ------------------------------------------------------------
    def embed(self, piece_ids: torch.Tensor, meta_ids: torch.Tensor) -> torch.Tensor:
        B = piece_ids.size(0)
        squares = self._square_index.unsqueeze(0).expand(B, -1)
        board = self.piece_embed(piece_ids) + self.square_embed(squares)

        meta_tokens = []
        for slot, embed in enumerate(self.meta_value_embeds):
            values = meta_ids[:, slot]
            type_tok = self.meta_type_embed(self._meta_index[slot].expand(B))
            meta_tokens.append(embed(values) + type_tok)
        meta = torch.stack(meta_tokens, dim=1)

        x = torch.cat([meta, board], dim=1)  # [B, 76, H]
        x = x * (self.cfg.hidden_size ** 0.5)
        return self.embed_dropout(x)

    def forward(
        self,
        piece_ids: torch.Tensor,
        meta_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = self.embed(piece_ids.long(), meta_ids.long())

        aux_loss = x.new_zeros(())
        liquid_state: Optional[torch.Tensor] = None
        use_ckpt = bool(self.cfg.use_gradient_checkpointing and self.training)

        for block in self.blocks:
            if use_ckpt:
                def _run(inp: torch.Tensor, state: Optional[torch.Tensor], _b=block):
                    out, aux, st = _b(inp, liquid_state=state)
                    return out, aux, st

                x, aux, liquid_state = torch.utils.checkpoint.checkpoint(
                    _run, x, liquid_state, use_reentrant=False
                )
            else:
                x, aux, liquid_state = block(x, liquid_state=liquid_state)
            aux_loss = aux_loss + aux

        x = self.final_norm(x)
        square_tokens = x[:, NUM_META_TOKENS:, :]
        cls = x[:, 0, :]

        if self.policy_head_kind == "factorized":
            policy_logits = self.policy_head(square_tokens)
        else:
            policy_logits = self.policy_head(x.mean(dim=1))

        out: Dict[str, torch.Tensor] = {
            "policy_logits": policy_logits,
            "value_logits": self.value_head(cls),
            "aux_loss": aux_loss,
        }
        if self.wdl_head is not None:
            out["wdl_logits"] = self.wdl_head(cls)
        if self.legality_head is not None:
            out["legality_logits"] = self.legality_head(square_tokens)
        return out

    # -- convenience --------------------------------------------------------
    @torch.no_grad()
    def predict_win_prob(self, value_logits: torch.Tensor) -> torch.Tensor:
        """Expectation over bin centres -> scalar win probability in [0, 1]."""
        probs = F.softmax(value_logits.float(), dim=-1)
        return (probs * self.value_bin_centers.to(probs.device)).sum(dim=-1)

    def parameter_report(self) -> Dict[str, object]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        by_group: Dict[str, int] = {}
        for name, param in self.named_parameters():
            group = name.split(".")[0]
            by_group[group] = by_group.get(group, 0) + param.numel()
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "parameters_by_group": dict(sorted(by_group.items(), key=lambda kv: -kv[1])),
            "policy_head": self.policy_head_kind,
            "vocab_size": self.vocab_size,
            "seq_len": SEQ_LEN,
            "attention_mode": self.cfg.attention_mode,
            "config": self.cfg.to_dict(),
        }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Total loss plus per-term metrics, all kept as on-device tensors.

    Nothing here calls ``.item()``: every scalar stays on the GPU and is only
    synced by the training loop every ``metric_sync_every_steps`` steps. The
    upstream loop synced 5+ times per step, which serializes the pipeline.
    """
    policy_logits = outputs["policy_logits"]
    legal_mask = batch["legal_mask"]

    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits.float(), dim=-1)
    # Illegal entries are -inf here and the target is exactly 0 there, so the
    # naive product is 0 * -inf = NaN. Zeroing the masked positions is exact
    # (the target has no mass outside the legal set) and keeps the loss finite.
    log_probs = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))

    # Soft policy target over the legal moves (multi-PV aware). Rows always sum
    # to 1 by construction in the dataset.
    policy_target = batch["policy_target"]
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()

    value_logits = outputs["value_logits"]
    value_target = batch["value_target"]
    value_loss = -(value_target * F.log_softmax(value_logits.float(), dim=-1)).sum(dim=-1).mean()

    total = cfg.policy_loss_coef * policy_loss + cfg.value_loss_coef * value_loss

    metrics: Dict[str, torch.Tensor] = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
    }

    if "wdl_logits" in outputs and "wdl_target" in batch:
        wdl_loss = F.cross_entropy(outputs["wdl_logits"].float(), batch["wdl_target"])
        total = total + cfg.wdl_loss_coef * wdl_loss
        metrics["wdl_loss"] = wdl_loss.detach()
        metrics["wdl_accuracy"] = (
            outputs["wdl_logits"].argmax(dim=-1) == batch["wdl_target"]
        ).float().mean().detach()

    if "legality_logits" in outputs:
        targets = legal_mask.to(outputs["legality_logits"].dtype)
        positives = targets.sum().clamp(min=1.0)
        total_slots = torch.tensor(
            float(targets.numel()), device=targets.device, dtype=targets.dtype
        )
        negatives = (total_slots - positives).clamp(min=1.0)
        pos_weight = (negatives / positives).clamp(max=float(cfg.legality_pos_weight_cap))
        legality_loss = F.binary_cross_entropy_with_logits(
            outputs["legality_logits"].float(), targets.float(), pos_weight=pos_weight.float()
        )
        total = total + cfg.legality_loss_coef * legality_loss
        metrics["legality_loss"] = legality_loss.detach()

    aux = outputs.get("aux_loss")
    if aux is not None and aux.numel() > 0:
        total = total + cfg.moe_aux_loss_coef * aux
        metrics["aux_loss"] = aux.detach()

    # --- accuracy diagnostics (on-device) ---
    best_target = policy_target.argmax(dim=-1)
    masked_top1 = masked_logits.argmax(dim=-1)
    metrics["policy_top1"] = (masked_top1 == best_target).float().mean().detach()

    topk = min(5, masked_logits.size(-1))
    top5 = masked_logits.topk(topk, dim=-1).indices
    metrics["policy_top5"] = (
        (top5 == best_target.unsqueeze(-1)).any(dim=-1).float().mean().detach()
    )
    # Raw (unmasked) legality: does the model pick a legal move without help?
    raw_top1 = policy_logits.argmax(dim=-1)
    metrics["raw_top1_is_legal"] = (
        legal_mask.gather(1, raw_top1.unsqueeze(-1)).squeeze(-1).float().mean().detach()
    )
    metrics["loss"] = total.detach()
    return total, metrics
