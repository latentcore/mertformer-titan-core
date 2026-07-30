"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yunlu
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture (Samsung S25 NPU + NVIDIA RTX 5080 / T4)
Version: v1.0 (Build 30 V2) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from config.config import cfg
from layers.cognitive_extensions import (
    ContinuousLatentODEStateChannel,
    NeuromodulatoryGainLayer,
)
from layers.world_model_head import CausalWorldModelHead
from layers.mertformer_block import MertFormerBlock, RMSNorm


class MertFormer(nn.Module):
    """
    MertFormer Titan - Main Transformer Model.

    Features:
    - Token embedding
    - N MertFormerBlock layers (Attention + FFN / MoE)
    - Final RMSNorm
    - LM Head (optional weight tying)
    - KV Cache support (inference acceleration)

    Forward:
        input_ids (B, T) -> logits (B, T, V), aux_loss (scalar)
    """

    def __init__(self) -> None:
        """MertFormer initializer."""
        super().__init__()
        self.cfg = cfg

        vocab_size = cfg.vocab_size
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        dropout = getattr(cfg, "dropout", 0.0)
        rms_eps = getattr(cfg, "rms_norm_eps", 1e-6)
        tie_weights = getattr(cfg, "tie_weights", True)

        # Token embedding
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [MertFormerBlock(layer_id=i) for i in range(num_layers)]
        )

        # Final norm + LM head
        self.final_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying: Embedding and LM head share the same weights
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight
        
        # Gradient Checkpointing (40% VRAM savings)
        self.use_gradient_checkpointing = getattr(cfg, "use_gradient_checkpointing", False)
        self.use_global_workspace_broadcast = bool(getattr(cfg, "use_global_workspace_broadcast", False))
        self.workspace_blend = float(getattr(cfg, "workspace_blend", 0.7))
        self.latent_ode_channel = (
            ContinuousLatentODEStateChannel(hidden_size)
            if bool(getattr(cfg, "use_latent_ode_state_channel", False))
            else None
        )
        self.neuromod_gain_layer = (
            NeuromodulatoryGainLayer(hidden_size)
            if bool(getattr(cfg, "use_neuromodulatory_gain", False))
            else None
        )
        self.world_model_head = (
            CausalWorldModelHead(
                hidden_size,
                horizon=int(getattr(cfg, "world_model_horizon", 1)),
            )
            if bool(getattr(cfg, "use_world_model_head", False))
            else None
        )
        self._last_world_model_outputs: Optional[dict] = None
        # [2026-07-08] Per-layer LiquidMixer hidden states produced by the most recent
        # forward(). Published as an attribute (same pattern as _last_world_model_outputs)
        # rather than appended to forward()'s return tuple: ~15 call sites across
        # train/, eval/, tests/, mertformer_sdk/ AND the ONNX export wrapper in
        # train/trainer_core.py unpack forward() as a strict 3-tuple, so widening the
        # arity would change the traced export contract. Read via get_last_liquid_states().
        self._present_liquid_states: Optional[List[Optional[torch.Tensor]]] = None
        self.latent_ode_dt = float(getattr(cfg, "latent_ode_dt", 1.0))

        # Weight init (embedding / output projection only): keep PyTorch-default
        # (fan-in-scaled) init for the BitLinear bodies so the ternary RMS-scale
        # stays large enough to propagate signal, but initialize the tied embedding
        # == output projection small so initial logits are near-uniform
        # (start loss ~= ln(vocab)). nn.Embedding's default N(0,1), amplified by the
        # x sqrt(hidden) embedding scaling and shared with lm_head, otherwise made
        # the untrained model wildly overconfident (start loss ~= 121 vs ln(V)).
        _init_std = float(getattr(cfg, "initializer_range", 0.02))
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=_init_std)
        if not tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=_init_std)

    @property
    def vocab_size(self) -> int:
        """Current embedding vocab size."""
        return self.tok_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Align embedding and lm_head to the tokenizer vocabulary
        (closes the 128000/128256 mismatch).

        Preserves existing rows/cols and weight tying. No-op if already aligned.
        """
        old_num = self.tok_embeddings.num_embeddings
        if new_num_tokens == old_num:
            return

        hidden_size = self.tok_embeddings.embedding_dim
        weight = self.tok_embeddings.weight
        device, dtype = weight.device, weight.dtype
        tie_weights = self.lm_head.weight is self.tok_embeddings.weight

        new_emb = nn.Embedding(new_num_tokens, hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            keep = min(old_num, new_num_tokens)
            new_emb.weight[:keep] = weight[:keep]
        self.tok_embeddings = new_emb

        new_head = nn.Linear(hidden_size, new_num_tokens, bias=False, device=device, dtype=dtype)
        if tie_weights:
            new_head.weight = self.tok_embeddings.weight
        else:
            with torch.no_grad():
                new_head.weight[:keep] = self.lm_head.weight[:keep]
        self.lm_head = new_head
        self.cfg.vocab_size = new_num_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        liquid_states: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass - Converts tokens to logits.

        Args:
            input_ids (torch.Tensor): Token IDs [Batch, Seq]
            past_key_values (Optional[List]): Previous layer caches
            use_cache (bool): Return cache?
            liquid_states (Optional[List]): Previous per-layer LiquidMixer hidden states,
                parallel to `past_key_values`. `None` starts each recurrence from zeros.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
                - logits: Token probabilities [Batch, Seq, Vocab]
                - aux_loss: Total aux loss from all MoE layers
                - present_key_values: New caches (if use_cache=True)

        The updated per-layer Liquid states are published on `self._present_liquid_states`
        and read via `get_last_liquid_states()` — the return arity is intentionally kept at
        3 because the ONNX export wrapper and ~15 other call sites unpack exactly 3 values.
        """
        # Reset router state at new sequence start (KV cache determinism)
        if use_cache and past_key_values is None and not self.training:
            self.reset_router_state(batch_size=input_ids.size(0))

        # Embedding
        x = self.tok_embeddings(input_ids)  # (B, T, H)
        
        # Embedding Scaling (GPT/LLaMA standard)
        # Provides stability by scaling embeddings for large hidden_size
        x = x * (self.cfg.hidden_size ** 0.5)
        
        x = self.drop(x)
        workspace_state = x.mean(dim=1) if self.use_global_workspace_broadcast else None
        if self.latent_ode_channel is not None and use_cache and past_key_values is None and not self.training:
            self.latent_ode_channel.reset_state(
                batch_size=input_ids.size(0),
                device=x.device,
                dtype=x.dtype,
            )

        # Aux loss sum (from all MoE layers)
        aux_total = x.new_zeros(())

        # KV Cache management
        present_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )

        # [2026-07-08] Per-layer Liquid recurrent states, threaded exactly like the KV cache.
        present_liquid_states: List[Optional[torch.Tensor]] = []

        # Apply blocks sequentially
        for i, block in enumerate(self.layers):
            # Safe KV retrieval
            past_kv = past_key_values[i] if (past_key_values is not None and i < len(past_key_values)) else None
            # Safe Liquid-state retrieval (parallel to past_kv)
            past_liquid = liquid_states[i] if (liquid_states is not None and i < len(liquid_states)) else None
            if self.latent_ode_channel is not None:
                x = self.latent_ode_channel(x, dt=self.latent_ode_dt)
            
            # Gradient Checkpointing (VRAM savings during training)
            if self.use_gradient_checkpointing and self.training and not use_cache:
                # Checkpointing: don't save activations during forward, recompute in backward
                from torch.utils.checkpoint import checkpoint
                # Checkpoint Hardening (Tensor-only closure).
                # Bind per-iteration `past_kv`/`workspace_state` as default args so
                # the closure captures them BY VALUE at def time. Without this they
                # are captured by reference and the backward recompute would read
                # the loop-final values (stale) when workspace broadcast is enabled.
                # On the canonical path both are None, so this is a numeric no-op.
                def run_checkpointed(
                    x_tensor: torch.Tensor,
                    _past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = past_kv,
                    _workspace: Optional[torch.Tensor] = workspace_state,
                    _liquid_state: Optional[torch.Tensor] = past_liquid,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    x_out, aux_out, _, _ = block(
                        x_tensor,
                        past_key_value=_past_kv,
                        use_cache=False,
                        workspace=_workspace,
                        liquid_state=_liquid_state,
                    )
                    return x_out, aux_out

                # [ADR-0004] In-place MoE telemetry buffer writes during forward
                #     break use_reentrant=False's recompute metadata-equality check
                #     (CheckpointError on first backward). Reentrant mode skips that
                #     check, fixing the blocker without touching the protected MoE
                #     core (minor RNG-on-recompute difference accepted).
                x, aux = checkpoint(
                    run_checkpointed, x, use_reentrant=True
                )
                present_kv = None
                # Gradient checkpointing is training-only (use_cache=False); the Liquid
                # state is not carried out of the recomputed closure.
                present_liquid = None
            else:
                x, aux, present_kv, present_liquid = block(
                    x,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    workspace=workspace_state,
                    liquid_state=past_liquid,
                )

            present_liquid_states.append(present_liquid)
            aux_total = aux_total + aux
            if workspace_state is not None:
                token_summary = x.mean(dim=1)
                blend = min(max(self.workspace_blend, 0.0), 1.0)
                workspace_state = workspace_state * blend + token_summary * (1.0 - blend)
            if self.neuromod_gain_layer is not None:
                x = self.neuromod_gain_layer(x, workspace_state)
            
            if use_cache:
                present_key_values.append(present_kv)  # type: ignore[arg-type, union-attr]  # list is non-None when use_cache; present_kv may be None for checkpointed layers

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if self.world_model_head is not None:
            self._last_world_model_outputs = self.world_model_head(x).to_dict()
        else:
            self._last_world_model_outputs = None

        self._present_liquid_states = present_liquid_states

        return logits, aux_total, present_key_values

    def get_last_world_model_outputs(self) -> Optional[dict]:
        """Return optional world-model side outputs from the latest forward pass."""
        return self._last_world_model_outputs

    def get_last_liquid_states(self) -> Optional[List[Optional[torch.Tensor]]]:
        """
        Per-layer LiquidMixer final hidden states from the latest forward pass.

        Entries are `None` for layers without a LiquidMixer (and for every layer when
        gradient checkpointing recomputed the block). Feed the list straight back into
        `forward(..., liquid_states=...)` to continue the recurrence, exactly as
        `past_key_values` is fed back for the attention KV cache.
        """
        return self._present_liquid_states

    def reset_router_state(self, batch_size: int = 1) -> None:
        """
        Resets the latent-ODE channel state, the LiquidRouter (MoE) state and the
        per-layer LiquidMixer recurrent states (for deterministic KV cache).

        [2026-07-08] Extended (deliberately NOT renamed — the repo's sealed `mla.py`
        precedent: a misleading name bound to state-dict/manifest continuity gets a
        corrected docstring, not a new identifier) to also drop the Liquid hidden states,
        so a fresh generation never inherits the previous sequence's recurrence.
        """
        self._present_liquid_states = None
        if self.latent_ode_channel is not None:
            self.latent_ode_channel.reset_state(
                batch_size=batch_size,
                device=self.tok_embeddings.weight.device,
                dtype=self.tok_embeddings.weight.dtype,
            )
        for block in self.layers:
            if getattr(block, "is_moe_layer", False):
                router = getattr(getattr(block, "ff", None), "router", None)
                if router is None:
                    continue
                # Create clean state matching batch size
                state = torch.zeros(
                    batch_size,
                    router.hidden_size,
                    router.history_window - 1,
                    device=router.inference_state.device,
                    dtype=router.inference_state.dtype,
                )
                router.set_state(state)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation - Accelerated with KV Cache.

        Args:
            input_ids (torch.Tensor): Initial tokens [Batch, Seq]
            max_new_tokens (int): Max new tokens
            temperature (float): Sampling temperature
            top_k (Optional[int]): Top-k sampling
            top_p (Optional[float]): Nucleus sampling
            eos_token_id (Optional[int]): Stop token
        Returns:
            torch.Tensor: Generated token sequence
        """
        # Reset router state for fresh generation (also clears Liquid recurrent states)
        self.reset_router_state(batch_size=input_ids.size(0))
        generated = input_ids
        past_key_values = None
        # [2026-07-08] Liquid/CfC recurrent state, carried across decode steps exactly
        # like past_key_values. Without this every step restarted the recurrence at h=0
        # and the Liquid layers were stateless no-ops during generation.
        liquid_states = None

        for _ in range(max_new_tokens):
            # Process only last token (thanks to KV Cache)
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            # Forward pass with cache
            logits, _, past_key_values = self.forward(
                curr_input,
                past_key_values=past_key_values,
                use_cache=True,
                liquid_states=liquid_states,
            )
            liquid_states = self._present_liquid_states

            # Get logits of last position
            # Guard temperature=0 (greedy) against div-by-zero/inf -> NaN.
            safe_temperature = max(float(temperature), 1e-6)
            next_token_logits = logits[:, -1, :] / safe_temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # EOS check
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
