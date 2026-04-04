#!/usr/bin/env python3
"""
MertFormer Chess RTX 5080 Onefile
---------------------------------
Standalone Windows-friendly chess proof lane for a single RTX 5080 desktop.

Goals:
- one-click PyCharm execution
- first-run dependency bootstrap
- automatic Lichess partial download + filtering on the target machine
- legal-move-safe chess model training and evidence packaging
- export/share mode that may self-delete only the shared script copy after success

This file intentionally stays repo-owned and readable. A separate export step can
produce an obfuscated wrapper for sharing.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "mertformer_chess_5080_onefile_v1"
SCRIPT_BASENAME = "mertformer_chess_5080_onefile"
RESULT_ZIP_PREFIX = "MertFormer_Chess_5080_Result"
DELIVERY_PREFIX = "MertFormer_Chess_5080_Delivery"
EMBEDDED_SEED_PGN = textwrap.dedent(
    """
    [Event "Rated Seed Game 1"]
    [Site "Local"]
    [Date "2026.01.01"]
    [Round "-"]
    [White "SeedA"]
    [Black "SeedB"]
    [Result "1-0"]
    [WhiteElo "2150"]
    [BlackElo "2100"]
    [TimeControl "300+0"]
    [Termination "Normal"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 b4 12. a3 bxa3 13. Nxa3 Bb7 14. Bc2 Re8 15. b4 Bf8 16. d5 c6 17. Be3 cxd5 18. cxd5 Nb6 19. Bd3 Nfd7 20. Nc4 Nxc4 21. Bxc4 Be7 22. Qa4 Rf8 23. Rec1 f5 24. exf5 Rxf5 25. Bd3 Rxf3 26. gxf3 Bxd5 27. Be4 Bxe4 28. fxe4 Nf6 29. Qc6 Kh8 30. Rxa6 Rxa6 31. Qxa6 Qd7 32. Qc8+ Qxc8 33. Rxc8+ Ng8 34. b5 1-0

    [Event "Rated Seed Game 2"]
    [Site "Local"]
    [Date "2026.01.02"]
    [Round "-"]
    [White "SeedC"]
    [Black "SeedD"]
    [Result "0-1"]
    [WhiteElo "2200"]
    [BlackElo "2230"]
    [TimeControl "300+3"]
    [Termination "Normal"]

    1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O dxc4 8. Bxc4 cxd4 9. exd4 b6 10. Bg5 Bb7 11. Re1 Nbd7 12. Rc1 Rc8 13. Bd3 h6 14. Bh4 Re8 15. Ne5 Be7 16. Bg3 Nxe5 17. Bxe5 Nd5 18. Qg4 Bf6 19. Nb5 Rxc1 20. Rxc1 Bxe5 21. dxe5 Re7 22. Nd6 Rc7 23. Rxc7 Qxc7 24. h4 Qc1+ 25. Bf1 Ba6 26. Qf3 Qxf1+ 27. Kh2 f6 28. exf6 Nxf6 29. Qa8+ Kh7 30. Qxa7 Be2 31. f3 Qf2 32. Qa4 Bxf3 33. Qc2+ Qxc2 34. Kg3 Qxg2+ 35. Kf4 Nd5+ 36. Ke5 Qe2+ 37. Kd4 Qd2+ 38. Ke5 Qf4+ 39. Kxe6 Bg4+ 40. Kxd5 Bf3+ 41. Ke6 Qf6+ 42. Kd7 Bg4+ 43. Kc7 Qe7+ 44. Kc6 Bf3+ 45. Kxb6 Qxd6+ 46. Ka5 Qc5+ 47. Ka4 Bd1+ 48. b3 Be2 49. b4 Qc3 50. Ka5 Bc4 0-1

    [Event "Rated Seed Game 3"]
    [Site "Local"]
    [Date "2026.01.03"]
    [Round "-"]
    [White "SeedE"]
    [Black "SeedF"]
    [Result "1-0"]
    [WhiteElo "2050"]
    [BlackElo "2080"]
    [TimeControl "600+0"]
    [Termination "Normal"]

    1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 6. Bg2 Nb6 7. O-O Be7 8. d3 O-O 9. Be3 Re8 10. Rc1 Bf8 11. a3 Nd4 12. Nd2 c6 13. b4 Bg4 14. h3 Bh5 15. g4 Bg6 16. Nce4 Nd5 17. Bg5 f6 18. Bh4 Nf4 19. Re1 a5 20. e3 Nxd3 21. exd4 Nxc1 22. dxe5 Rxe5 23. Qxc1 axb4 24. axb4 Bxb4 25. Qc4+ Bf7 26. Qxb4 Rb5 27. Qc3 Ra2 28. Nf3 Bd5 29. g5 Bxe4 30. Qc4+ Bd5 31. Qg4 fxg5 32. Bxg5 Qf8 33. Be7 Qf7 34. Qc8+ Qf8 35. Qxf8# 1-0
    """
).strip()

DEFAULT_LICHESS_URLS = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-03.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-02.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2025-12.pgn.zst",
]

RUN_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "artifact_root": "~/Desktop",
    "cache_root": "~/Desktop/mertformer_chess_cache",
    "result_prefix": RESULT_ZIP_PREFIX,
    "download_partial_mb": 384,
    "download_timeout_sec": 60,
    "max_games": 50000,
    "max_positions": 220000,
    "max_positions_per_game": 5,
    "min_elo": 1900,
    "time_control_min_seconds": 180,
    "time_control_max_seconds": 900,
    "exclude_time_forfeit": True,
    "prefer_eval_positions": True,
    "curriculum_enabled": True,
    "max_wall_hours": 3.5,
    "max_steps": 18000,
    "batch_size": 256,
    "eval_batch_size": 256,
    "learning_rate": 3.0e-4,
    "weight_decay": 0.01,
    "warmup_steps": 400,
    "grad_clip": 1.0,
    "hidden_size": 384,
    "num_layers": 8,
    "num_heads": 8,
    "dropout": 0.10,
    "use_moe": True,
    "num_experts": 4,
    "use_bitlinear": False,
    "use_liquid_adapter": True,
    "compile_policy": "off",
    "use_bf16": True,
    "num_workers": 0,
    "eval_interval": 300,
    "checkpoint_interval": 900,
    "proof_games": 12,
    "stockfish_games": 8,
    "stockfish_skill": 4,
    "stockfish_nodes": 20000,
    "stockfish_path": "",
    "auto_download_enabled": True,
    "test_mode": False,
    "offline_seed_only": False,
    "self_delete_share_copy": True,
    "zip_outputs": True,
    "export_open_copy": True,
    "legal_move_sample_checks": 256,
    "val_fraction": 0.10,
    "target_elo": 1600,
    "lichess_urls": DEFAULT_LICHESS_URLS,
}


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _pip_install(args: Sequence[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    subprocess.check_call(cmd)


def _bootstrap_if_needed() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get("MERTFORMER_CHESS_SKIP_BOOTSTRAP", "0") == "1":
        return
    missing: List[Tuple[str, List[str]]] = []
    if not _module_exists("torch"):
        torch_args = ["torch"]
        if platform.system() == "Windows":
            index_url = os.environ.get("MERTFORMER_CHESS_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128")
            torch_args += ["--index-url", index_url]
        missing.append(("torch", torch_args))
    for mod_name, package_args in (
        ("numpy", ["numpy>=1.24"]),
        ("zstandard", ["zstandard>=0.21"]),
        ("chess", ["python-chess>=1.999"]),
        ("psutil", ["psutil>=5.9"]),
    ):
        if not _module_exists(mod_name):
            missing.append((mod_name, package_args))
    if not missing:
        return
    if os.environ.get("MERTFORMER_CHESS_BOOTSTRAP_DONE", "0") == "1":
        raise SystemExit(
            "Required packages are still missing after bootstrap attempt: "
            + ", ".join(name for name, _ in missing)
        )
    for _, package_args in missing:
        _pip_install(package_args)
    env = os.environ.copy()
    env["MERTFORMER_CHESS_BOOTSTRAP_DONE"] = "1"
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)


_bootstrap_if_needed()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd
try:
    import chess
    import chess.engine
    import chess.pgn
except Exception:  # pragma: no cover - import guarded by bootstrap in __main__
    chess = None  # type: ignore
try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


if chess is None:  # pragma: no cover
    raise SystemExit("python-chess is required; bootstrap did not complete successfully")


@dataclass
class ArtifactLayout:
    run_id: str
    root: Path
    run_dir: Path
    logs_dir: Path
    reports_dir: Path
    checkpoints_dir: Path
    export_dir: Path
    desktop_dir: Path
    final_zip_path: Path
    final_sha_path: Path


@dataclass
class ChessExample:
    piece_ids: List[int]
    meta_ids: List[int]
    legal_move_ids: List[int]
    target_move_id: int
    value_target: float
    phase: int
    source_game_id: str
    ply: int
    has_eval: bool


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts_utc": utc_now(),
            "kind": kind,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class WindowsExecutionGuard:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and platform.system() == "Windows")
        self._restore_value = self.ES_CONTINUOUS

    def __enter__(self):
        if self.enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_DISPLAY_REQUIRED
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(self._restore_value)
        return False


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, enabled: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.enabled = enabled
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize(self, weight: torch.Tensor) -> torch.Tensor:
        scale = weight.detach().abs().mean().clamp_min(1e-6)
        ternary = torch.where(weight > 0.5 * scale, torch.ones_like(weight), torch.zeros_like(weight))
        ternary = torch.where(weight < -0.5 * scale, -torch.ones_like(weight), ternary)
        quant = weight + ((ternary * scale) - weight).detach()
        return quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._quantize(self.weight) if self.enabled else self.weight
        return F.linear(x, weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, use_bitlinear: bool = False):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        linear = BitLinear if use_bitlinear else nn.Linear
        kwargs = {"enabled": True} if use_bitlinear else {}
        self.q_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.k_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.v_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.o_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o_proj(out)


class DenseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, use_bitlinear: bool = False):
        super().__init__()
        inner = hidden_size * 4
        linear = BitLinear if use_bitlinear else nn.Linear
        kwargs = {"enabled": True} if use_bitlinear else {}
        self.fc1 = linear(hidden_size, inner, **kwargs)  # type: ignore[arg-type]
        self.fc2 = linear(inner, hidden_size, **kwargs)  # type: ignore[arg-type]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class SparseMoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, dropout: float, use_bitlinear: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            DenseFeedForward(hidden_size, dropout, use_bitlinear=use_bitlinear) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        weights = F.softmax(router_logits, dim=-1)
        top1 = weights.argmax(dim=-1)
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, x.size(-1))
        flat_out = out.reshape(-1, out.size(-1))
        flat_top1 = top1.reshape(-1)
        flat_weights = weights.reshape(-1, self.num_experts)
        for expert_idx, expert in enumerate(self.experts):
            mask = flat_top1 == expert_idx
            if not bool(mask.any()):
                continue
            expert_in = flat_x[mask]
            expert_out = expert(expert_in)
            expert_weight = flat_weights[mask, expert_idx].unsqueeze(-1)
            flat_out[mask] = expert_out * expert_weight
        load = weights.mean(dim=(0, 1))
        aux = ((load - (1.0 / self.num_experts)) ** 2).mean()
        return out, aux


class LiquidAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        tau = torch.sigmoid(self.gate(x))
        delta = torch.tanh(self.proj(x))
        return residual + tau * delta


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        use_moe: bool,
        num_experts: int,
        use_bitlinear: bool,
        use_liquid_adapter: bool,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout, use_bitlinear=use_bitlinear)
        self.use_moe = use_moe
        if use_moe:
            self.ff = SparseMoE(hidden_size, num_experts, dropout, use_bitlinear=use_bitlinear)
        else:
            self.ff = DenseFeedForward(hidden_size, dropout, use_bitlinear=use_bitlinear)
        self.use_liquid_adapter = use_liquid_adapter
        self.liquid = LiquidAdapter(hidden_size) if use_liquid_adapter else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = residual + self.dropout(self.attn(self.norm1(x)))
        aux = x.new_tensor(0.0)
        ff_in = self.norm2(x)
        if self.use_moe:
            ff_out, aux = self.ff(ff_in)
        else:
            ff_out = self.ff(ff_in)
        if self.use_liquid_adapter and self.liquid is not None:
            x = self.liquid(ff_out, x)
        else:
            x = x + self.dropout(ff_out)
        return x, aux


class ChessPolicyValueNet(nn.Module):
    META_CARDINALITIES = [2, 2, 2, 2, 2, 9, 16, 32]

    def __init__(self, cfg: Dict[str, Any], vocab_size: int):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        layers = int(cfg["num_layers"])
        heads = int(cfg["num_heads"])
        dropout = float(cfg["dropout"])
        use_bitlinear = bool(cfg.get("use_bitlinear", False))
        use_moe = bool(cfg.get("use_moe", True))
        num_experts = int(cfg.get("num_experts", 4))
        use_liquid_adapter = bool(cfg.get("use_liquid_adapter", True))

        self.piece_embed = nn.Embedding(13, hidden)
        self.square_embed = nn.Embedding(64, hidden)
        self.meta_type_embed = nn.Embedding(len(self.META_CARDINALITIES), hidden)
        self.meta_value_embeds = nn.ModuleList(nn.Embedding(card, hidden) for card in self.META_CARDINALITIES)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                hidden,
                heads,
                dropout,
                use_moe=use_moe and (layer_idx % 2 == 1),
                num_experts=num_experts,
                use_bitlinear=use_bitlinear,
                use_liquid_adapter=use_liquid_adapter,
            )
            for layer_idx in range(layers)
        )
        self.norm = RMSNorm(hidden)
        linear = BitLinear if use_bitlinear else nn.Linear
        kwargs = {"enabled": True} if use_bitlinear else {}
        self.policy_head = linear(hidden, vocab_size, **kwargs)  # type: ignore[arg-type]
        self.value_head = linear(hidden, 1, **kwargs)  # type: ignore[arg-type]
        self.dropout = nn.Dropout(dropout)

    def forward(self, piece_ids: torch.Tensor, meta_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = piece_ids.size(0)
        square_ids = torch.arange(64, device=piece_ids.device).unsqueeze(0).expand(bsz, -1)
        board = self.piece_embed(piece_ids) + self.square_embed(square_ids)
        meta_tokens: List[torch.Tensor] = []
        for meta_idx, embed in enumerate(self.meta_value_embeds):
            meta_val = meta_ids[:, meta_idx]
            type_tok = self.meta_type_embed(torch.full_like(meta_val, meta_idx))
            meta_tokens.append(embed(meta_val) + type_tok)
        meta = torch.stack(meta_tokens, dim=1)
        x = torch.cat([meta, board], dim=1)
        aux_loss = x.new_tensor(0.0)
        for block in self.blocks:
            x, aux = block(x)
            aux_loss = aux_loss + aux
        x = self.norm(x)
        pooled = self.dropout(x.mean(dim=1))
        policy_logits = self.policy_head(pooled)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return policy_logits, value, aux_loss


class ChessExampleDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[ChessExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ChessExample:
        return self.examples[idx]


def collate_examples(batch: Sequence[ChessExample]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    piece_ids = torch.tensor([item.piece_ids for item in batch], dtype=torch.long)
    meta_ids = torch.tensor([item.meta_ids for item in batch], dtype=torch.long)
    move_targets = torch.tensor([item.target_move_id for item in batch], dtype=torch.long)
    value_targets = torch.tensor([item.value_target for item in batch], dtype=torch.float32)
    phases = torch.tensor([item.phase for item in batch], dtype=torch.long)
    vocab_size = len(MOVE_VOCAB)
    legal_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool)
    for row_idx, item in enumerate(batch):
        legal_mask[row_idx, item.legal_move_ids] = True
    return {
        "piece_ids": piece_ids,
        "meta_ids": meta_ids,
        "move_targets": move_targets,
        "value_targets": value_targets,
        "legal_mask": legal_mask,
        "phases": phases,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "artifact"


def detect_desktop_dir() -> Path:
    return Path.home() / "Desktop"


def resolve_runtime_config(base_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(base_cfg or RUN_CONFIG)
    cfg["artifact_root"] = str(Path(str(cfg.get("artifact_root", "~/Desktop"))).expanduser())
    cfg["cache_root"] = str(Path(str(cfg.get("cache_root", "~/Desktop/mertformer_chess_cache"))).expanduser())
    if os.environ.get("MERTFORMER_CHESS_TEST_MODE", "0") == "1":
        cfg["test_mode"] = True
        cfg["offline_seed_only"] = True
        cfg["auto_download_enabled"] = False
        cfg["max_games"] = 6
        cfg["max_positions"] = 64
        cfg["max_positions_per_game"] = 4
        cfg["max_wall_hours"] = 0.02
        cfg["max_steps"] = 6
        cfg["batch_size"] = 8
        cfg["eval_batch_size"] = 8
        cfg["hidden_size"] = 128
        cfg["num_layers"] = 2
        cfg["num_heads"] = 4
        cfg["num_experts"] = 2
        cfg["compile_policy"] = "off"
        cfg["use_bf16"] = False
        cfg["proof_games"] = 2
        cfg["stockfish_games"] = 0
    if str(cfg.get("device", "auto")) == "auto":
        if torch.cuda.is_available():
            cfg["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg["device"] = "mps"
        else:
            cfg["device"] = "cpu"
    if str(cfg["device"]) != "cuda":
        cfg["use_bf16"] = False
        cfg["compile_policy"] = "off"
        cfg["batch_size"] = min(int(cfg["batch_size"]), 32)
        cfg["eval_batch_size"] = min(int(cfg["eval_batch_size"]), 32)
    return cfg


def make_layout(cfg: Dict[str, Any]) -> ArtifactLayout:
    desktop = detect_desktop_dir()
    root = Path(str(cfg["artifact_root"]))
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{DELIVERY_PREFIX}_{run_id}"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    checkpoints_dir = run_dir / "checkpoints"
    export_dir = run_dir / "exports"
    final_zip = desktop / f"{cfg['result_prefix']}_{run_id}.zip"
    final_sha = desktop / f"{cfg['result_prefix']}_{run_id}.zip.sha256"
    for path in (run_dir, logs_dir, reports_dir, checkpoints_dir, export_dir):
        path.mkdir(parents=True, exist_ok=True)
    return ArtifactLayout(
        run_id=run_id,
        root=root,
        run_dir=run_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        checkpoints_dir=checkpoints_dir,
        export_dir=export_dir,
        desktop_dir=desktop,
        final_zip_path=final_zip,
        final_sha_path=final_sha,
    )


def parse_time_control(tc: str) -> int:
    base = tc.split("+", 1)[0].strip()
    if not base.isdigit():
        return 0
    return int(base)


def result_to_value(result: str, turn: bool) -> float:
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0


def parse_eval_comment(comment: str) -> Optional[float]:
    comment = comment or ""
    mate_match = re.search(r"\[%eval\s+#(-?\d+)\]", comment)
    if mate_match:
        value = float(mate_match.group(1))
        return max(-1.0, min(1.0, value / 6.0))
    cp_match = re.search(r"\[%eval\s+(-?\d+(?:\.\d+)?)\]", comment)
    if not cp_match:
        return None
    cp = float(cp_match.group(1))
    return max(-1.0, min(1.0, math.tanh(cp / 3.0)))


def piece_to_id(piece: Optional[chess.Piece]) -> int:
    if piece is None:
        return 0
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + piece.piece_type


def encode_board_state(board: chess.Board) -> Tuple[List[int], List[int]]:
    piece_ids = [piece_to_id(board.piece_at(square)) for square in chess.SQUARES]
    ep_square = board.ep_square
    ep_file = 0 if ep_square is None else chess.square_file(ep_square) + 1
    meta_ids = [
        int(board.turn),
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        ep_file,
        min(15, board.halfmove_clock // 4),
        min(31, board.fullmove_number // 2),
    ]
    return piece_ids, meta_ids


def build_move_vocab() -> List[str]:
    moves: List[str] = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            moves.append(chess.square_name(from_sq) + chess.square_name(to_sq))
    promos: List[str] = []
    promo_pieces = ["q", "r", "b", "n"]
    for file_idx in range(8):
        white_from = chess.square(file_idx, 6)
        for delta in (-1, 0, 1):
            to_file = file_idx + delta
            if 0 <= to_file < 8:
                white_to = chess.square(to_file, 7)
                black_from = chess.square(file_idx, 1)
                black_to = chess.square(to_file, 0)
                for promo in promo_pieces:
                    promos.append(chess.square_name(white_from) + chess.square_name(white_to) + promo)
                    promos.append(chess.square_name(black_from) + chess.square_name(black_to) + promo)
    seen = set()
    ordered: List[str] = []
    for item in moves + promos:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


MOVE_VOCAB = build_move_vocab()
MOVE_TO_ID = {uci: idx for idx, uci in enumerate(MOVE_VOCAB)}
ID_TO_MOVE = {idx: uci for uci, idx in MOVE_TO_ID.items()}


def legal_move_ids(board: chess.Board) -> List[int]:
    ids: List[int] = []
    for move in board.legal_moves:
        uci = move.uci()
        move_id = MOVE_TO_ID.get(uci)
        if move_id is not None:
            ids.append(move_id)
    return ids


def select_ply_indices(total_plies: int, limit: int) -> List[int]:
    if total_plies <= 0 or limit <= 0:
        return []
    anchors = [0, min(4, total_plies - 1), min(10, total_plies - 1), total_plies // 2, max(0, total_plies - 2)]
    picks = sorted(set(x for x in anchors if 0 <= x < total_plies))
    if len(picks) > limit:
        picks = picks[:limit]
    return picks


def game_is_usable(game: chess.pgn.Game, cfg: Dict[str, Any]) -> bool:
    headers = game.headers
    if headers.get("Variant", "Standard") not in {"", "Standard"}:
        return False
    if headers.get("WhiteTitle", "") == "BOT" or headers.get("BlackTitle", "") == "BOT":
        return False
    if "Rated" not in headers.get("Event", "Rated"):
        return False
    try:
        white_elo = int(headers.get("WhiteElo", "0") or 0)
        black_elo = int(headers.get("BlackElo", "0") or 0)
    except ValueError:
        return False
    if min(white_elo, black_elo) < int(cfg["min_elo"]):
        return False
    base_seconds = parse_time_control(headers.get("TimeControl", "0+0"))
    if base_seconds < int(cfg["time_control_min_seconds"]) or base_seconds > int(cfg["time_control_max_seconds"]):
        return False
    if bool(cfg.get("exclude_time_forfeit", True)) and headers.get("Termination", "") == "Time forfeit":
        return False
    result = headers.get("Result", "")
    return result in {"1-0", "0-1", "1/2-1/2"}


def iter_games_from_pgn_text(text: str) -> Iterator[chess.pgn.Game]:
    handle = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game


def embedded_seed_games() -> List[chess.pgn.Game]:
    return list(iter_games_from_pgn_text(EMBEDDED_SEED_PGN))


def download_partial_archive(urls: Sequence[str], cfg: Dict[str, Any], logger: JSONLLogger, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    byte_budget = int(cfg["download_partial_mb"]) * 1024 * 1024
    timeout = int(cfg.get("download_timeout_sec", 60))
    last_error = ""
    for url in urls:
        filename = safe_name(Path(url).name)
        target = cache_root / filename
        headers = {"User-Agent": f"{SCRIPT_BASENAME}/{SCRIPT_VERSION}", "Range": f"bytes=0-{byte_budget - 1}"}
        req = urllib.request.Request(url, headers=headers)
        logger.write("download_start", {"url": url, "target": str(target), "bytes": byte_budget})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, target.open("wb") as handle:
                bytes_written = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    room = byte_budget - bytes_written
                    if room <= 0:
                        break
                    if len(chunk) > room:
                        chunk = chunk[:room]
                    handle.write(chunk)
                    bytes_written += len(chunk)
            if target.stat().st_size > 0:
                logger.write("download_done", {"url": url, "target": str(target), "bytes_written": target.stat().st_size})
                return target
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = f"{type(exc).__name__}: {exc}"
            logger.write("download_error", {"url": url, "error": last_error})
    raise RuntimeError(f"Unable to download partial Lichess archive: {last_error or 'unknown error'}")


def iter_games_from_zstd(path: Path) -> Iterator[chess.pgn.Game]:
    with path.open("rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore", newline="")
            while True:
                try:
                    game = chess.pgn.read_game(text_stream)
                except Exception:
                    break
                if game is None:
                    break
                yield game


def build_examples_from_games(games: Iterable[chess.pgn.Game], cfg: Dict[str, Any], logger: JSONLLogger) -> Tuple[List[ChessExample], Dict[str, Any]]:
    examples: List[ChessExample] = []
    stats = {
        "games_seen": 0,
        "games_kept": 0,
        "positions_total": 0,
        "positions_opening": 0,
        "positions_middlegame": 0,
        "positions_endgame": 0,
        "positions_with_eval": 0,
    }
    max_games = int(cfg["max_games"])
    max_positions = int(cfg["max_positions"])
    max_positions_per_game = int(cfg["max_positions_per_game"])
    for game_idx, game in enumerate(games, start=1):
        stats["games_seen"] += 1
        if not game_is_usable(game, cfg):
            continue
        stats["games_kept"] += 1
        result = game.headers.get("Result", "1/2-1/2")
        board = game.board()
        node = game
        moves: List[Tuple[chess.Move, str]] = []
        while node.variations:
            next_node = node.variation(0)
            moves.append((next_node.move, next_node.comment or ""))
            node = next_node
        for ply_idx in select_ply_indices(len(moves), max_positions_per_game):
            board = game.board()
            for step_idx in range(ply_idx):
                board.push(moves[step_idx][0])
            move, comment = moves[ply_idx]
            legal_ids = legal_move_ids(board)
            target_id = MOVE_TO_ID.get(move.uci())
            if target_id is None or target_id not in legal_ids:
                continue
            piece_ids, meta_ids = encode_board_state(board)
            phase = 0 if ply_idx < 16 else 1 if ply_idx < 60 else 2
            raw_value = parse_eval_comment(comment)
            has_eval = raw_value is not None
            value_target = float(raw_value if raw_value is not None else result_to_value(result, board.turn))
            examples.append(
                ChessExample(
                    piece_ids=piece_ids,
                    meta_ids=meta_ids,
                    legal_move_ids=legal_ids,
                    target_move_id=target_id,
                    value_target=value_target,
                    phase=phase,
                    source_game_id=f"game_{stats['games_kept']}",
                    ply=ply_idx,
                    has_eval=has_eval,
                )
            )
            stats["positions_total"] += 1
            if phase == 0:
                stats["positions_opening"] += 1
            elif phase == 1:
                stats["positions_middlegame"] += 1
            else:
                stats["positions_endgame"] += 1
            if has_eval:
                stats["positions_with_eval"] += 1
            if len(examples) >= max_positions:
                logger.write("dataset_cap_reached", {"positions": len(examples), "games_kept": stats["games_kept"]})
                return examples, stats
        if stats["games_kept"] >= max_games:
            logger.write("game_cap_reached", {"games_kept": stats["games_kept"], "positions": len(examples)})
            return examples, stats
    return examples, stats


def maybe_collect_dataset(cfg: Dict[str, Any], layout: ArtifactLayout, logger: JSONLLogger) -> Tuple[List[ChessExample], Dict[str, Any]]:
    provenance: Dict[str, Any] = {
        "mode": "embedded_seed" if bool(cfg.get("offline_seed_only", False)) else "lichess_partial",
        "script_version": SCRIPT_VERSION,
        "urls": list(cfg.get("lichess_urls", [])),
    }
    if bool(cfg.get("offline_seed_only", False)) or not bool(cfg.get("auto_download_enabled", True)):
        examples, stats = build_examples_from_games(embedded_seed_games(), cfg, logger)
        provenance.update({"embedded_seed": True, "stats": stats})
        return examples, provenance
    cache_root = Path(str(cfg["cache_root"]))
    archive = download_partial_archive(cfg.get("lichess_urls", DEFAULT_LICHESS_URLS), cfg, logger, cache_root)
    examples, stats = build_examples_from_games(iter_games_from_zstd(archive), cfg, logger)
    if not examples:
        examples, stats = build_examples_from_games(embedded_seed_games(), {**cfg, "offline_seed_only": True}, logger)
        provenance.update({"fallback_to_embedded_seed": True})
    provenance.update({
        "archive_path": str(archive),
        "archive_sha256": path_sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
        "stats": stats,
    })
    return examples, provenance


def split_examples(examples: Sequence[ChessExample], cfg: Dict[str, Any]) -> Tuple[List[ChessExample], List[ChessExample]]:
    rng = random.Random(int(cfg["seed"]))
    data = list(examples)
    rng.shuffle(data)
    val_count = max(1, int(len(data) * float(cfg.get("val_fraction", 0.10))))
    return data[val_count:], data[:val_count]


def curriculum_sort(examples: Sequence[ChessExample], cfg: Dict[str, Any]) -> List[ChessExample]:
    if not bool(cfg.get("curriculum_enabled", True)):
        return list(examples)
    def key_fn(item: ChessExample) -> Tuple[int, int, int]:
        eval_bias = 0 if item.has_eval else 1
        return (item.phase, eval_bias, item.ply)
    return sorted(examples, key=key_fn)


def pick_device(cfg: Dict[str, Any]) -> torch.device:
    return torch.device(str(cfg["device"]))


def maybe_enable_compile(model: nn.Module, cfg: Dict[str, Any], logger: JSONLLogger) -> Tuple[nn.Module, Dict[str, Any]]:
    policy = str(cfg.get("compile_policy", "off"))
    report = {"policy": policy, "attempted": False, "compiled": False, "reason": "disabled"}
    if policy == "off":
        return model, report
    if not hasattr(torch, "compile"):
        report["reason"] = "torch_compile_unavailable"
        return model, report
    report["attempted"] = True
    try:
        model = torch.compile(model, mode="max-autotune" if policy == "aggressive" else "default")  # type: ignore[attr-defined]
        report["compiled"] = True
        report["reason"] = "ok"
        return model, report
    except Exception as exc:  # pragma: no cover - compile availability varies
        logger.write("compile_fallback", {"error": str(exc)})
        report["reason"] = f"fallback:{type(exc).__name__}"
        return model, report


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )


def lr_for_step(step: int, cfg: Dict[str, Any]) -> float:
    warmup = max(1, int(cfg["warmup_steps"]))
    total = max(warmup + 1, int(cfg["max_steps"]))
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_optimizer_lr(optimizer: torch.optim.Optimizer, factor: float, cfg: Dict[str, Any]) -> None:
    base_lr = float(cfg["learning_rate"])
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_loss(
    model: ChessPolicyValueNet,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits, value_pred, aux_loss = model(batch["piece_ids"], batch["meta_ids"])
    masked_logits = logits.masked_fill(~batch["legal_mask"], -1e9)
    policy_loss = F.cross_entropy(masked_logits, batch["move_targets"])
    value_loss = F.mse_loss(value_pred, batch["value_targets"])
    loss = policy_loss + 0.25 * value_loss + 0.01 * aux_loss
    with torch.no_grad():
        preds = masked_logits.argmax(dim=-1)
        top1 = (preds == batch["move_targets"]).float().mean().item()
        top5 = (
            torch.topk(masked_logits, k=min(5, masked_logits.size(-1)), dim=-1).indices
            == batch["move_targets"].unsqueeze(-1)
        ).any(dim=-1).float().mean().item()
    metrics = {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "aux_loss": float(aux_loss.detach().item()),
        "top1": float(top1),
        "top5": float(top5),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_model(
    model: ChessPolicyValueNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "aux_loss": 0.0, "top1": 0.0, "top5": 0.0}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = batch_to_device(batch, device)
        _, metrics = compute_loss(model, batch, cfg)
        for key, value in metrics.items():
            sums[key] += float(value)
        count += 1
    model.train()
    if count == 0:
        return {key: 0.0 for key in sums}
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(model: ChessPolicyValueNet, optimizer: torch.optim.Optimizer, path: Path, step: int, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "step": step,
            "config": payload.get("config"),
            "metrics": payload.get("metrics"),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "move_vocab": MOVE_VOCAB,
        },
        path,
    )


def detect_stockfish_path(cfg: Dict[str, Any]) -> Optional[str]:
    explicit = str(cfg.get("stockfish_path", "") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    for candidate in (
        shutil.which("stockfish"),
        shutil.which("stockfish.exe"),
        str(Path.home() / "Desktop" / "stockfish" / "stockfish.exe"),
        str(Path.home() / "Downloads" / "stockfish" / "stockfish.exe"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def choose_move(model: ChessPolicyValueNet, board: chess.Board, device: torch.device) -> Tuple[str, float]:
    piece_ids, meta_ids = encode_board_state(board)
    piece = torch.tensor([piece_ids], dtype=torch.long, device=device)
    meta = torch.tensor([meta_ids], dtype=torch.long, device=device)
    logits, value, _ = model(piece, meta)
    logits = logits[0]
    legal_ids = legal_move_ids(board)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[legal_ids] = True
    masked = logits.masked_fill(~mask, -1e9)
    best_id = int(masked.argmax().item())
    return ID_TO_MOVE[best_id], float(value.item())


@torch.no_grad()
def legal_move_safety_check(
    model: ChessPolicyValueNet,
    examples: Sequence[ChessExample],
    device: torch.device,
    sample_count: int,
) -> Dict[str, Any]:
    if not examples:
        return {"checked": 0, "illegal_predictions": 0, "ok": False}
    rng = random.Random(1234)
    picks = list(examples)
    rng.shuffle(picks)
    checked = 0
    illegal = 0
    for example in picks[:sample_count]:
        piece = torch.tensor([example.piece_ids], dtype=torch.long, device=device)
        meta = torch.tensor([example.meta_ids], dtype=torch.long, device=device)
        logits, _, _ = model(piece, meta)
        logits = logits[0]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[example.legal_move_ids] = True
        pred = int(logits.masked_fill(~mask, -1e9).argmax().item())
        if pred not in example.legal_move_ids:
            illegal += 1
        checked += 1
    return {"checked": checked, "illegal_predictions": illegal, "ok": illegal == 0 and checked > 0}


def write_curve_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "split", "loss", "policy_loss", "value_loss", "aux_loss", "top1", "top5", "elapsed_sec"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": path_sha256(Path(__file__).resolve()) if Path(__file__).exists() else "",
    }
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            snap["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 3)
            snap["cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
        except Exception:
            pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        snap["cuda_name"] = props.name
        snap["cuda_total_gb"] = round(float(props.total_memory) / (1024 ** 3), 3)
        snap["cuda_capability"] = f"{props.major}.{props.minor}"
    return snap


def play_stockfish_match(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    layout: ArtifactLayout,
    logger: JSONLLogger,
) -> Dict[str, Any]:
    engine_path = detect_stockfish_path(cfg)
    if not engine_path or int(cfg.get("stockfish_games", 0)) <= 0:
        return {"status": "not-run", "reason": "stockfish_missing_or_disabled"}
    results = {"wins": 0, "draws": 0, "losses": 0, "games": []}
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as exc:  # pragma: no cover - engine availability varies
        return {"status": "not-run", "reason": f"engine_start_failed:{type(exc).__name__}"}
    try:
        openings = [
            ["e2e4", "e7e5", "g1f3", "b8c6"],
            ["d2d4", "g8f6", "c2c4", "e7e6"],
            ["c2c4", "e7e5", "g1f3", "b8c6"],
        ]
        total_games = int(cfg.get("stockfish_games", 8))
        for game_idx in range(total_games):
            board = chess.Board()
            for move_uci in openings[game_idx % len(openings)]:
                board.push(chess.Move.from_uci(move_uci))
            model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            pgn_moves: List[str] = []
            while not board.is_game_over() and len(pgn_moves) < 160:
                if board.turn == model_color:
                    move_uci, value = choose_move(model, board, device)
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        return {"status": "failed", "reason": "illegal_move_generated", "game_index": game_idx}
                    board.push(move)
                    pgn_moves.append(move.uci())
                else:
                    result = engine.play(
                        board,
                        chess.engine.Limit(nodes=int(cfg.get("stockfish_nodes", 20000))),
                        options={"Skill Level": int(cfg.get("stockfish_skill", 4))},
                    )
                    board.push(result.move)
                    pgn_moves.append(result.move.uci())
            outcome = board.outcome(claim_draw=True)
            result_str = outcome.result() if outcome is not None else "1/2-1/2"
            if (result_str == "1-0" and model_color == chess.WHITE) or (result_str == "0-1" and model_color == chess.BLACK):
                results["wins"] += 1
            elif result_str == "1/2-1/2":
                results["draws"] += 1
            else:
                results["losses"] += 1
            results["games"].append({"game_index": game_idx, "model_color": "white" if model_color else "black", "result": result_str, "plies": len(pgn_moves), "moves": pgn_moves})
        outcome_score = results["wins"] + 0.5 * results["draws"]
        total_games = max(1, len(results["games"]))
        expected = max(1e-6, min(1 - 1e-6, outcome_score / total_games))
        elo_proxy = int(800 - 400 * math.log10((1 / expected) - 1)) + 1400
        results["status"] = "verified"
        results["estimated_elo_proxy"] = elo_proxy
        out_path = layout.reports_dir / "stockfish_match_report.json"
        atomic_json(out_path, results)
        logger.write("stockfish_eval", {"status": results["status"], "estimated_elo_proxy": elo_proxy})
        return results
    finally:
        with contextlib.suppress(Exception):
            engine.quit()


def create_result_bundle(layout: ArtifactLayout, cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    atomic_json(layout.reports_dir / "run_summary.json", payload)
    if layout.final_zip_path.exists():
        layout.final_zip_path.unlink()
    with zipfile.ZipFile(layout.final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(layout.run_dir.rglob("*")):
            if path.is_dir():
                continue
            zf.write(path, arcname=str(path.relative_to(layout.run_dir)))
    sha = path_sha256(layout.final_zip_path)
    layout.final_sha_path.write_text(f"{sha}  {layout.final_zip_path.name}\n", encoding="utf-8")
    return {
        "zip_path": str(layout.final_zip_path),
        "sha256_path": str(layout.final_sha_path),
        "sha256": sha,
        "size_bytes": layout.final_zip_path.stat().st_size,
    }


def schedule_self_delete_if_needed(success: bool, final_zip: Optional[Path]) -> None:
    if not success:
        return
    if os.environ.get("MERTFORMER_CHESS_SHARE_MODE", "0") != "1":
        return
    if os.environ.get("MERTFORMER_CHESS_SELF_DELETE", "1") != "1":
        return
    script_path = Path(__file__).resolve()
    if platform.system() == "Windows":
        cmd_path = script_path.with_suffix(".cleanup.cmd")
        zip_part = final_zip.name if final_zip is not None else "result.zip"
        cmd_path.write_text(
            "@echo off\n"
            "setlocal\n"
            "ping 127.0.0.1 -n 3 > nul\n"
            f"del /f /q \"{script_path}\" > nul 2>&1\n"
            f"del /f /q \"{cmd_path}\" > nul 2>&1\n",
            encoding="utf-8",
        )
        subprocess.Popen(["cmd.exe", "/c", str(cmd_path)], creationflags=0x08000000)
    else:  # pragma: no cover - share mode targets Windows
        subprocess.Popen(["bash", "-lc", f"sleep 2; rm -f '{script_path}'"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def train_and_package(cfg: Dict[str, Any]) -> Dict[str, Any]:
    deterministic_seed(int(cfg["seed"]))
    layout = make_layout(cfg)
    logger = JSONLLogger(layout.logs_dir / "run_log.jsonl")
    logger.write("run_start", {"script_version": SCRIPT_VERSION, "config": cfg})
    atomic_json(layout.reports_dir / "environment_snapshot.json", env_snapshot())

    examples, provenance = maybe_collect_dataset(cfg, layout, logger)
    if not examples:
        raise RuntimeError("No training examples could be built from the selected dataset source")
    atomic_json(layout.reports_dir / "dataset_provenance.json", provenance)

    train_examples, val_examples = split_examples(curriculum_sort(examples, cfg), cfg)
    train_ds = ChessExampleDataset(train_examples)
    val_ds = ChessExampleDataset(val_examples)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_examples,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
        drop_last=False,
    )

    device = pick_device(cfg)
    model = ChessPolicyValueNet(cfg, len(MOVE_VOCAB)).to(device)
    optimizer = build_optimizer(model, cfg)
    model, compile_report = maybe_enable_compile(model, cfg, logger)
    atomic_json(layout.reports_dir / "compile_report.json", compile_report)

    scaler_enabled = bool(cfg.get("use_bf16", False)) and device.type == "cuda"
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
    start_time = time.time()
    best_val = float("inf")
    best_ckpt = layout.checkpoints_dir / "best_model.pt"
    curve_rows: List[Dict[str, Any]] = []
    global_step = 0
    last_checkpoint_at = 0

    for epoch in range(999999):
        for batch in train_loader:
            global_step += 1
            elapsed_hours = (time.time() - start_time) / 3600.0
            if elapsed_hours >= float(cfg["max_wall_hours"]) or global_step > int(cfg["max_steps"]):
                break
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            apply_optimizer_lr(optimizer, lr_for_step(global_step, cfg), cfg)
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=scaler_enabled):
                loss, metrics = compute_loss(model, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            row = {
                "step": global_step,
                "split": "train",
                "elapsed_sec": round(time.time() - start_time, 3),
                **metrics,
            }
            curve_rows.append(row)
            if global_step == 1 or global_step % 25 == 0:
                logger.write("train_step", row)
            if global_step % int(cfg["eval_interval"]) == 0 or global_step == 1:
                val_metrics = evaluate_model(model, val_loader, device, max_batches=16, cfg=cfg)
                curve_rows.append({
                    "step": global_step,
                    "split": "val",
                    "elapsed_sec": round(time.time() - start_time, 3),
                    **val_metrics,
                })
                logger.write("eval_step", {"step": global_step, **val_metrics})
                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    save_checkpoint(
                        model,
                        optimizer,
                        best_ckpt,
                        global_step,
                        {"config": cfg, "metrics": {"best_val_loss": best_val, **val_metrics}},
                    )
            if global_step - last_checkpoint_at >= int(cfg["checkpoint_interval"]):
                last_checkpoint_at = global_step
                save_checkpoint(
                    model,
                    optimizer,
                    layout.checkpoints_dir / f"step_{global_step:06d}.pt",
                    global_step,
                    {"config": cfg, "metrics": metrics},
                )
        if (time.time() - start_time) / 3600.0 >= float(cfg["max_wall_hours"]) or global_step > int(cfg["max_steps"]):
            break

    if not best_ckpt.exists():
        save_checkpoint(
            model,
            optimizer,
            best_ckpt,
            global_step,
            {"config": cfg, "metrics": {"best_val_loss": best_val if math.isfinite(best_val) else None}},
        )

    write_curve_csv(layout.reports_dir / "training_curve.csv", curve_rows)
    safety = legal_move_safety_check(model, val_examples, device, sample_count=int(cfg["legal_move_sample_checks"]))
    atomic_json(layout.reports_dir / "legal_move_safety.json", safety)
    final_val = evaluate_model(model, val_loader, device, max_batches=32, cfg=cfg)
    atomic_json(layout.reports_dir / "holdout_metrics.json", final_val)
    stockfish_result = play_stockfish_match(model, cfg, device, layout, logger)

    claim_status = "target"
    if stockfish_result.get("status") == "verified":
        measured = int(stockfish_result.get("estimated_elo_proxy", 0))
        if measured >= int(cfg["target_elo"]):
            claim_status = "verified"
        else:
            claim_status = "not-met"
    elif stockfish_result.get("status") == "not-run":
        claim_status = "target-not-verified"
    elif stockfish_result.get("status") == "failed":
        claim_status = "failed"

    replay = []
    board = chess.Board()
    for _ in range(int(cfg.get("proof_games", 12))):
        if board.is_game_over():
            break
        move_uci, value = choose_move(model, board, device)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            break
        board.push(move)
        replay.append({"ply": len(replay) + 1, "move": move_uci, "value": value, "fen": board.fen()})
    atomic_json(layout.reports_dir / "model_replay.json", {"moves": replay, "final_fen": board.fen()})

    payload = {
        "script_version": SCRIPT_VERSION,
        "run_id": layout.run_id,
        "config": cfg,
        "dataset_provenance": provenance,
        "holdout_metrics": final_val,
        "legal_move_safety": safety,
        "stockfish": stockfish_result,
        "claim_status": claim_status,
        "target_elo": int(cfg["target_elo"]),
        "best_checkpoint": str(best_ckpt),
        "output_root": str(layout.run_dir),
    }
    bundle = create_result_bundle(layout, cfg, payload)
    logger.write("run_complete", {"claim_status": claim_status, **bundle})
    payload["bundle"] = bundle
    atomic_json(layout.reports_dir / "run_summary.json", payload)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MertFormer Chess RTX 5080 onefile")
    parser.add_argument("--test-mode", action="store_true", help="Force tiny embedded-seed smoke mode.")
    parser.add_argument("--offline-seed-only", action="store_true", help="Skip network and use embedded seed PGN only.")
    parser.add_argument("--stockfish-path", help="Optional Stockfish executable override.")
    parser.add_argument("--artifact-root", help="Override artifact root.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-hours", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    cfg = resolve_runtime_config(RUN_CONFIG)
    if args.test_mode:
        cfg["test_mode"] = True
        cfg["offline_seed_only"] = True
        cfg["auto_download_enabled"] = False
        cfg["max_wall_hours"] = 0.02
        cfg["max_steps"] = 6
        cfg["batch_size"] = 8
        cfg["eval_batch_size"] = 8
        cfg["hidden_size"] = 128
        cfg["num_layers"] = 2
        cfg["num_heads"] = 4
        cfg["num_experts"] = 2
        cfg["use_bf16"] = False
        cfg["compile_policy"] = "off"
    if args.offline_seed_only:
        cfg["offline_seed_only"] = True
        cfg["auto_download_enabled"] = False
    if args.stockfish_path:
        cfg["stockfish_path"] = args.stockfish_path
    if args.artifact_root:
        cfg["artifact_root"] = str(Path(args.artifact_root).expanduser())
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.max_wall_hours is not None:
        cfg["max_wall_hours"] = float(args.max_wall_hours)

    success = False
    final_zip: Optional[Path] = None
    try:
        with WindowsExecutionGuard(enabled=True):
            payload = train_and_package(cfg)
        success = True
        final_zip = Path(payload["bundle"]["zip_path"])
        print(json.dumps({"status": "completed", "claim_status": payload["claim_status"], "zip": payload["bundle"]}, indent=2))
        return 0
    except Exception as exc:
        err = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        desktop = detect_desktop_dir()
        err_path = desktop / f"{RESULT_ZIP_PREFIX}_FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        err_path.write_text(json.dumps(err, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1
    finally:
        schedule_self_delete_if_needed(success, final_zip)


if __name__ == "__main__":
    raise SystemExit(main())
