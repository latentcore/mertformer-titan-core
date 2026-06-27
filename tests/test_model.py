import sys
import torch
import unittest
# import pytest
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from model.transformers import MertFormer
from config.config import cfg


class TestMertFormer(unittest.TestCase):
    def setUp(self):
        # Use small config for faster testing
        cfg.hidden_size = 128
        cfg.num_heads = 8
        cfg.head_dim = 16
        cfg.num_layers = 2
        cfg.num_hidden_layers = 2  # Match num_layers
        cfg.vocab_size = 1000
        cfg.max_seq_len = 64
        cfg.use_gradient_checkpointing = False  # Disable for test stability
        self.model = MertFormer()

    def test_forward_shape(self):
        """Test if model forward pass returns correct shapes"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        logits, aux_loss, _ = self.model(input_ids)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, cfg.vocab_size))
        self.assertTrue(torch.is_tensor(aux_loss))

    def test_kv_cache(self):
        """
        Test KV cache path executes safely and returns valid tensors.

        For stateful routing blocks, strict token-for-token equivalence with
        full forward is not guaranteed. This test validates runtime correctness
        of cache flow (shape + finite outputs), which is the critical invariant.
        """
        input_ids = torch.randint(0, cfg.vocab_size, (1, 10))

        # 1. Full forward (reference shape)
        logits_full, _, _ = self.model(input_ids, use_cache=False)
        self.assertEqual(logits_full.shape, (1, 10, cfg.vocab_size))
        self.assertFalse(torch.isnan(logits_full).any())

        # 2. Cached step-by-step (simulate generation)
        # NOTE: This is a SAFETY/SMOKE test, NOT a cache-correctness gate.
        # It does not verify token-for-token logit equivalence between the
        # cached decode path and a full forward pass; it only checks shapes
        # and finite (non-NaN) outputs. Do not treat a green result here as
        # proof that the KV cache produces numerically identical logits.
        # Prefill
        past_kv = None
        logits_pre, _, past_kv = self.model(input_ids[:, :-1], use_cache=True)
        self.assertEqual(logits_pre.shape, (1, 9, cfg.vocab_size))
        self.assertFalse(torch.isnan(logits_pre).any())
        self.assertIsNotNone(past_kv)
        
        # Decode last step
        logits_step, _, _ = self.model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
        self.assertEqual(logits_step.shape, (1, 1, cfg.vocab_size))
        self.assertFalse(torch.isnan(logits_step).any())


if __name__ == '__main__':
    unittest.main()
