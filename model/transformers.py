"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30 V2) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

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
    TR: MertFormer Titan - Ana Transformer Modeli.
    EN: MertFormer Titan - Main Transformer Model.

    Özellikler / Features:
    - Token embedding
    - N adet MertFormerBlock (Attention + FFN / MoE)
    - Final RMSNorm
    - LM Head (opsiyonel weight tying)
    - KV Cache desteği (inference hızlandırma) [V21.0]

    Forward:
        input_ids (B, T) -> logits (B, T, V), aux_loss (scalar)
    """

    def __init__(self) -> None:
        """TR: MertFormer başlatıcı. EN: MertFormer initializer."""
        super().__init__()
        self.cfg = cfg

        vocab_size = cfg.vocab_size
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        dropout = getattr(cfg, "dropout", 0.0)
        rms_eps = getattr(cfg, "rms_norm_eps", 1e-6)
        tie_weights = getattr(cfg, "tie_weights", True)

        # TR: Token embedding / EN: Token embedding
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.drop = nn.Dropout(dropout)

        # TR: Transformer blokları / EN: Transformer blocks
        self.layers = nn.ModuleList(
            [MertFormerBlock(layer_id=i) for i in range(num_layers)]
        )

        # TR: Final norm + LM head / EN: Final norm + LM head
        self.final_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # TR: Weight tying: Embedding ve LM head aynı ağırlıkları paylaşır
        # EN: Weight tying: Embedding and LM head share same weights
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight
        
        # TR: V22.0: Gradient Checkpointing (VRAM %40 tasarrufu)
        # EN: V22.0: Gradient Checkpointing (40% VRAM savings)
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
        self.latent_ode_dt = float(getattr(cfg, "latent_ode_dt", 1.0))

    @property
    def vocab_size(self) -> int:
        """TR: Mevcut embedding vocab boyutu. EN: Current embedding vocab size."""
        return self.tok_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """TR: Embedding ve lm_head'i tokenizer'a hizala. EN: Align embedding and
        lm_head to the tokenizer vocabulary (closes the 128000/128256 mismatch).

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        TR: İleri yayılım - Token'ları logit'lere dönüştürür.
        EN: Forward pass - Converts tokens to logits.

        Args:
            input_ids (torch.Tensor): Token ID'leri / Token IDs [Batch, Seq]
            past_key_values (Optional[List]): Önceki tüm layer cache'leri / Previous layer caches
            use_cache (bool): Cache döndürülsün mü? / Return cache?
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
                - logits: Token olasılıkları / Token probabilities [Batch, Seq, Vocab]
                - aux_loss: Tüm MoE katmanlarından gelen toplam aux loss / Total aux loss
                - present_key_values: Yeni cache'ler / New caches (if use_cache=True)
        """
        # TR: Yeni sequence başlangıcında router state sıfırla (KV cache determinism)
        # EN: Reset router state at new sequence start (KV cache determinism)
        if use_cache and past_key_values is None and not self.training:
            self.reset_router_state(batch_size=input_ids.size(0))

        # TR: Embedding / EN: Embedding
        x = self.tok_embeddings(input_ids)  # (B, T, H)
        
        # TR: V23.0: Embedding Scaling (GPT/LLaMA standardı)
        # EN: V23.0: Embedding Scaling (GPT/LLaMA standard)
        # TR: Büyük hidden_size'da embedding'leri ölçekleyerek stabilite sağlar
        # EN: Provides stability by scaling embeddings for large hidden_size
        x = x * (self.cfg.hidden_size ** 0.5)
        
        x = self.drop(x)
        workspace_state = x.mean(dim=1) if self.use_global_workspace_broadcast else None
        if self.latent_ode_channel is not None and use_cache and past_key_values is None and not self.training:
            self.latent_ode_channel.reset_state(
                batch_size=input_ids.size(0),
                device=x.device,
                dtype=x.dtype,
            )

        # TR: Aux loss toplamı (tüm MoE katmanlarından)
        # EN: Aux loss sum (from all MoE layers)
        aux_total = x.new_zeros(())

        # TR: KV Cache yönetimi / EN: KV Cache management
        present_key_values = [] if use_cache else None

        # TR: Blokları sırayla uygula / EN: Apply blocks sequentially
        for i, block in enumerate(self.layers):
            # TR: Güvenli KV alımı / EN: Safe KV retrieval
            past_kv = past_key_values[i] if (past_key_values is not None and i < len(past_key_values)) else None
            if self.latent_ode_channel is not None:
                x = self.latent_ode_channel(x, dt=self.latent_ode_dt)
            
            # TR: V22.0: Gradient Checkpointing (eğitimde VRAM tasarrufu)
            # EN: V22.0: Gradient Checkpointing (VRAM savings during training)
            if self.use_gradient_checkpointing and self.training and not use_cache:
                # TR: Checkpointing: forward sırasında aktivasyonları kaydetme, backward'da yeniden hesapla
                # EN: Checkpointing: don't save activations during forward, recompute in backward
                from torch.utils.checkpoint import checkpoint
                # TR: [V26.4 FIX] Checkpoint Hardening (Sadece Tensor closure)
                # EN: [V26.4 FIX] Checkpoint Hardening (Tensor-only closure)
                def run_checkpointed(x_tensor):
                    x_out, aux_out, _ = block(
                        x_tensor,
                        past_key_value=past_kv,
                        use_cache=False,
                        workspace=workspace_state,
                    )
                    return x_out, aux_out
                
                # TR: [ADR-0004] MoE.forward icindeki in-place telemetry buffer
                #     yazimlari (last_expert_load/collapse_detected vb.) ile
                #     use_reentrant=False'un metadata-esitlik recompute kontrolu
                #     catisip ilk backward'da CheckpointError firlatiyordu.
                #     Reentrant mod bu kontrolu yapmaz; korunan MoE cekirdegine
                #     dokunmadan blocker'i cozer (RNG recompute farki kabul edildi).
                # EN: [ADR-0004] In-place MoE telemetry buffer writes during forward
                #     break use_reentrant=False's recompute metadata-equality check
                #     (CheckpointError on first backward). Reentrant mode skips that
                #     check, fixing the blocker without touching the protected MoE
                #     core (minor RNG-on-recompute difference accepted).
                x, aux = checkpoint(
                    run_checkpointed, x, use_reentrant=True
                )
                present_kv = None
            else:
                x, aux, present_kv = block(
                    x,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    workspace=workspace_state,
                )
            
            aux_total = aux_total + aux
            if workspace_state is not None:
                token_summary = x.mean(dim=1)
                blend = min(max(self.workspace_blend, 0.0), 1.0)
                workspace_state = workspace_state * blend + token_summary * (1.0 - blend)
            if self.neuromod_gain_layer is not None:
                x = self.neuromod_gain_layer(x, workspace_state)
            
            if use_cache:
                present_key_values.append(present_kv)

        # TR: Son norm + LM head / EN: Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if self.world_model_head is not None:
            self._last_world_model_outputs = self.world_model_head(x).to_dict()
        else:
            self._last_world_model_outputs = None

        return logits, aux_total, present_key_values

    def get_last_world_model_outputs(self) -> Optional[dict]:
        """Return optional world-model side outputs from the latest forward pass."""
        return self._last_world_model_outputs

    def reset_router_state(self, batch_size: int = 1) -> None:
        """
        TR: LiquidRouter state'ini sıfırlar (deterministik cache için).
        EN: Resets LiquidRouter state (for deterministic KV cache).
        """
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
                # TR: Batch boyutuna göre temiz state üret
                # EN: Create clean state matching batch size
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
        TR: Autoregressive text generation - KV Cache ile hızlandırılmış.
        EN: Autoregressive text generation - Accelerated with KV Cache.

        Args:
            input_ids (torch.Tensor): Başlangıç token'ları / Initial tokens [Batch, Seq]
            max_new_tokens (int): Üretilecek maksimum token sayısı / Max new tokens
            temperature (float): Sampling sıcaklığı / Sampling temperature
            top_k (Optional[int]): Top-k sampling / Top-k sampling
            top_p (Optional[float]): Nucleus sampling / Nucleus sampling
            eos_token_id (Optional[int]): Durma token'ı / Stop token
        Returns:
            torch.Tensor: Üretilen token dizisi / Generated token sequence
        """
        # Reset router state for fresh generation
        self.reset_router_state(batch_size=input_ids.size(0))
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            # TR: Sadece son token'ı işle (KV Cache sayesinde)
            # EN: Process only last token (thanks to KV Cache)
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            # TR: Cache ile forward pass / EN: Forward pass with cache
            logits, _, past_key_values = self.forward(
                curr_input, past_key_values=past_key_values, use_cache=True
            )

            # TR: Son pozisyonun logit'lerini al
            # EN: Get logits of last position
            # TR: temperature=0 (greedy) bolme-sifir/inf -> NaN'i onle.
            # EN: Guard temperature=0 (greedy) against div-by-zero/inf -> NaN.
            safe_temperature = max(float(temperature), 1e-6)
            next_token_logits = logits[:, -1, :] / safe_temperature

            # TR: Top-k filtreleme / EN: Top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # TR: Top-p (nucleus) filtreleme / EN: Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # TR: Sample / EN: Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # TR: Ekle / EN: Append
            generated = torch.cat([generated, next_token], dim=1)

            # TR: EOS kontrolü / EN: EOS check
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
