"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Licensed under MIT License.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Titan Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from config.config import cfg
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
        # TR: Embedding / EN: Embedding
        x = self.tok_embeddings(input_ids)  # (B, T, H)
        
        # TR: V23.0: Embedding Scaling (GPT/LLaMA standardı)
        # EN: V23.0: Embedding Scaling (GPT/LLaMA standard)
        # TR: Büyük hidden_size'da embedding'leri ölçekleyerek stabilite sağlar
        # EN: Provides stability by scaling embeddings for large hidden_size
        x = x * (self.cfg.hidden_size ** 0.5)
        
        x = self.drop(x)

        # TR: Aux loss toplamı (tüm MoE katmanlarından)
        # EN: Aux loss sum (from all MoE layers)
        aux_total = x.new_zeros(())

        # TR: KV Cache yönetimi / EN: KV Cache management
        present_key_values = [] if use_cache else None

        # TR: Blokları sırayla uygula / EN: Apply blocks sequentially
        for i, block in enumerate(self.layers):
            # TR: Güvenli KV alımı / EN: Safe KV retrieval
            past_kv = past_key_values[i] if (past_key_values is not None and i < len(past_key_values)) else None
            
            # TR: V22.0: Gradient Checkpointing (eğitimde VRAM tasarrufu)
            # EN: V22.0: Gradient Checkpointing (VRAM savings during training)
            if self.use_gradient_checkpointing and self.training and not use_cache:
                # TR: Checkpointing: forward sırasında aktivasyonları kaydetme, backward'da yeniden hesapla
                # EN: Checkpointing: don't save activations during forward, recompute in backward
                from torch.utils.checkpoint import checkpoint
                # TR: [V26.4 FIX] Checkpoint Hardening (Sadece Tensor closure)
                # EN: [V26.4 FIX] Checkpoint Hardening (Tensor-only closure)
                def run_checkpointed(x_tensor):
                    x_out, aux_out, _ = block(x_tensor, past_key_value=past_kv, use_cache=False)
                    return x_out, aux_out
                
                x, aux = checkpoint(
                    run_checkpointed, x, use_reentrant=False
                )
                present_kv = None
            else:
                x, aux, present_kv = block(x, past_key_value=past_kv, use_cache=use_cache)
            
            aux_total = aux_total + aux
            
            if use_cache:
                present_key_values.append(present_kv)

        # TR: Son norm + LM head / EN: Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, aux_total, present_key_values

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
            next_token_logits = logits[:, -1, :] / temperature

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

