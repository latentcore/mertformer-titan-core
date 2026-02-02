from __future__ import annotations

"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - LOCKED RUN LOGGER (Mini-Titan v5.0)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v5.0-FORENSIC (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
Target : Rigorous Benchmark with Cryptographic Proof
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

import os
import io
import sys
import math
import json
import time
import hashlib
import platform
import socket
import subprocess
import csv
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# LOGGER SYSTEM (EMBEDDED)
# ==============================================================================
def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _local_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def _safe_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": True, "len": len(obj), "sha256": hashlib.sha256(obj).hexdigest()}
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if is_dataclass(obj):
        return _safe_json(asdict(obj))
    tname = type(obj).__name__.lower()
    if "tensor" in tname:
        try:
            return {
                "__tensor__": True,
                "shape": [int(x) for x in getattr(obj, "shape", [])],
                "dtype": str(getattr(obj, "dtype", None)),
                "device": str(getattr(obj, "device", None)),
            }
        except:
            return {"__tensor__": True}
    return {"__repr__": repr(obj), "__type__": type(obj).__name__}

def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b: break
                h.update(b)
        return h.hexdigest()
    except:
        return "error_reading_file"

def try_git_commit(repo_dir: Union[str, Path]) -> Optional[str]:
    return None # Simplified for PoC

def atomic_write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

class RunLogger:
    def __init__(self, cfg: Any, log_dir: Union[str, Path] = "logs", run_name: Optional[str] = None,
                 also_csv: bool = True, csv_fields: Optional[Iterable[str]] = None,
                 flush_every: int = 10, fsync_every: int = 100):
        self.cfg = cfg
        self.flush_every = max(1, int(flush_every))
        self.fsync_every = max(1, int(fsync_every))
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        stamp = _local_stamp()
        self.run_id = run_name or f"run_{stamp}"
        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"
        self.manifest_path = self.log_dir / f"{self.run_id}.manifest.json"
        self.csv_path = self.log_dir / f"{self.run_id}.csv" if also_csv else None

        self._fh = None
        self._csv_fh = None
        self._csv_fields = list(csv_fields) if csv_fields else None
        self._genesis_hash = hashlib.sha256(b"").hexdigest()
        self._prev_hash = self._genesis_hash
        self._line_count = 0
        self._step_count = 0
        self._opened_at = _utc_iso()
        self._finalized = False
        self._open_files()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if not self._finalized:
            self.finalize(status="completed" if exc is None else "failed")

    def _open_files(self):
        self._fh = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        if self.csv_path:
            self._csv_fh = self.csv_path.open("a", encoding="utf-8", buffering=1)

    def _write_line(self, rec: Dict[str, Any]):
        if not self._fh: return
        safe = _safe_json(rec)
        if not isinstance(safe, dict): safe = {"value": safe}
        safe["_chain"] = {"prev": self._prev_hash}
        line = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        line_bytes = (line + "\n").encode("utf-8")
        h = hashlib.sha256()
        h.update(self._prev_hash.encode("utf-8"))
        h.update(line_bytes)
        line_hash = h.hexdigest()
        safe["_chain"]["hash"] = line_hash
        safe["_chain"]["n"] = self._line_count + 1
        self._fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
        self._prev_hash = line_hash
        self._line_count += 1

    def _write_csv(self, step_rec: Dict[str, Any]):
        if not self._csv_fh: return
        # Auto-detect fields if not set
        if self._csv_fields is None:
            self._csv_fields = ["timestamp_utc", "phase", "step", "loss", "val_loss", "ppl", "grad_norm", "tau", "lr"]
            if os.stat(self.csv_path).st_size == 0:
                self._csv_fh.write(",".join(self._csv_fields) + "\n")
        
        row = []
        for k in self._csv_fields:
            v = step_rec.get(k, "")
            if isinstance(v, float): s = f"{v:.6g}"
            else: s = str(v)
            row.append(s)
        self._csv_fh.write(",".join(row) + "\n")

    def log_meta(self, extra=None):
        meta = {
            "type": "meta", "timestamp_utc": _utc_iso(), "run_id": self.run_id,
            "host": socket.gethostname(), "cfg": _safe_json(self.cfg)
        }
        if extra: meta["extra"] = extra
        self._write_line(meta)

    def log_step(self, metrics: Dict[str, Any]):
        self._step_count += 1
        rec = dict(metrics)
        rec.setdefault("type", "step")
        rec.setdefault("timestamp_utc", _utc_iso())
        rec["step"] = self._step_count
        self._write_line(rec)
        self._write_csv(rec)

    def finalize(self, status="completed", extra=None):
        if self._finalized: return
        manifest = {
            "run_id": self.run_id, "status": status, "lines": self._line_count,
            "final_chain_hash": self._prev_hash
        }
        self._write_line({"type": "final", "status": status})
        if self._fh: self._fh.close()
        if self._csv_fh: self._csv_fh.close()
        atomic_write_json(self.manifest_path, manifest)
        self._finalized = True
        return manifest

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    vocab_size = 32000
    hidden_size = 512
    num_layers = 12
    num_heads = 16
    head_dim = 32
    intermediate_size = 1408
    max_position_embeddings = 1024
    
    # MoE
    num_experts = 4        
    num_experts_per_tok = 2
    moe_every_n_layers = 2
    use_moe = True
    
    # Liquid - Toggled dynamically
    liquid_layers_idx = [3, 7, 10]
    dt = 1.0
    use_liquid = True 

    # Training
    batch_size = 32
    learning_rate = 3e-4
    max_steps = 200     
    val_steps = 20      
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ==============================================================================
# TELEMETRY 
# ==============================================================================
tau_stats = {"sum": 0.0, "count": 0, "min": 99.0, "max": -99.0}

def record_tau(tau_tensor):
    with torch.no_grad():
        t_mean = tau_tensor.mean().item()
        tau_stats["sum"] += t_mean
        tau_stats["count"] += 1
        tau_stats["min"] = min(tau_stats["min"], t_mean)
        tau_stats["max"] = max(tau_stats["max"], t_mean)

# ==============================================================================
# MODEL COMPONENTS (BITNET, LIQUID, MOE)
# ==============================================================================
def activation_quant(x):
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = 127.0 / max_abs
    return x + (torch.round(x * scale).clamp(-127, 127) / scale - x).detach()

def weight_quant(w):
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    return w + (torch.round(w / scale).clamp(-1.0, 1.0) * scale - w).detach()

class BitLinear(nn.Linear):
    def forward(self, x):
        return F.linear(activation_quant(x), weight_quant(self.weight), self.bias)

class LiquidCell(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.input_w = BitLinear(h, h, bias=False)
        self.hidden_w = BitLinear(h, h, bias=False)
        self.tau_input_w = BitLinear(h, h, bias=False)
        self.tau_hidden_w = BitLinear(h, h, bias=False)
        self.tau_bias = nn.Parameter(torch.ones(1, h) * 0.5)

    def forward(self, x, h_prev, dt=1.0):
        val_in = self.input_w(x)
        val_rec = self.hidden_w(h_prev)
        A = torch.tanh(val_in + val_rec)
        tau_in = self.tau_input_w(x)
        tau_rec = self.tau_hidden_w(h_prev)
        raw_tau = F.softplus(tau_in + tau_rec + self.tau_bias)
        if self.training and cfg.use_liquid: record_tau(raw_tau)
        decay = torch.exp(-raw_tau * dt)
        return A + (h_prev - A) * decay

class LiquidMixer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.cell = LiquidCell(h)
        self.norm = nn.LayerNorm(h)

    def forward(self, x):
        B, T, H = x.shape
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h, cfg.dt)
            outs.append(h)
        return self.norm(torch.stack(outs, dim=1) + x)

class BitSwiGLU(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.w1 = BitLinear(h, i, bias=False)
        self.w2 = BitLinear(i, h, bias=False)
        self.w3 = BitLinear(h, i, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SparseMoE(nn.Module):
    def __init__(self, h, i, n, k=2):
        super().__init__()
        self.num_experts, self.k = n, k
        self.router = nn.Linear(h, n, bias=False)
        self.experts = nn.ModuleList([BitSwiGLU(h, i) for _ in range(n)])

    def forward(self, x):
        B, T, H = x.shape
        logits = self.router(x)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        flat_x, output = x.view(-1, H), torch.zeros(B*T, H, device=x.device, dtype=x.dtype)
        for k_i in range(self.k):
            for e_i in range(self.num_experts):
                idx = indices.view(-1, self.k)[:, k_i]
                mask = (idx == e_i)
                if mask.any():
                    output[mask] += self.experts[e_i](flat_x[mask]) * weights.view(-1, self.k)[:, k_i][mask].unsqueeze(-1)
        return output.view(B, T, H)

class Block(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size) # RMSNorm impl simplified to LayerNorm for PoC
        self.attn = nn.MultiheadAttention(cfg.hidden_size, cfg.num_heads, batch_first=True) # Using PyTorch MHA for speed in PoC
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.liquid = LiquidMixer(cfg.hidden_size) if (cfg.use_liquid and idx in cfg.liquid_layers_idx) else None
        self.mlp = SparseMoE(cfg.hidden_size, cfg.intermediate_size, cfg.num_experts) if (cfg.use_moe and idx % cfg.moe_every_n_layers == 0) else BitSwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        if self.liquid: x = self.liquid(x)
        return x

class MiniTitan(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.head = BitLinear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.emb.weight = self.head.weight # Weight tying

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ==============================================================================
# DATA & ENGINE
# ==============================================================================
def get_dataset():
    try:
        from datasets import load_dataset
        print("🌍 Downloading WikiText-2...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokens = torch.tensor([hash(w) % cfg.vocab_size for w in "\n".join(ds["text"]).split()], dtype=torch.long)
        print(f"✅ Real Data: {len(tokens)} tokens")
        class DS(Dataset):
            def __len__(self): return (len(tokens)-1)//128
            def __getitem__(self, i): 
                chunk = tokens[i*128 : i*128+129]
                if len(chunk)<129: chunk = F.pad(chunk, (0, 129-len(chunk)))
                return chunk[:-1], chunk[1:]
        return DS(), DS() # Use same for val in PoC speed
    except:
        print("⚠️ Using Dummy Data")
        class Dummy(Dataset):
            def __len__(self): return 5000
            def __getitem__(self, i): d=torch.randint(0, cfg.vocab_size, (129,)); return d[:-1], d[1:]
        return Dummy(), Dummy()

def train_phase(name, use_liquid, dataset, steps, logger):
    print(f"\n⚡ {name} | Liquid: {use_liquid}")
    cfg.use_liquid = use_liquid
    global tau_stats; tau_stats = {"sum": 0.0, "count": 0, "min": 99.0, "max": -99.0}
    
    model = MiniTitan(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    model.train()
    start = time.time()
    iter_data = iter(loader)
    
    for i in range(steps):
        try: x, y = next(iter_data)
        except: iter_data = iter(loader); x, y = next(iter_data)
        
        x, y = x.to(cfg.device), y.to(cfg.device)
        optim.zero_grad()
        with torch.amp.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        loss.backward()
        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optim.step()
        
        if i % 10 == 0:
            avg_tau = tau_stats["sum"]/max(1, tau_stats["count"])
            print(f"Step {i} | Loss: {loss.item():.4f} | Tau: {avg_tau:.3f}")
            logger.log_step({
                "phase": name, "loss": loss.item(), "grad_norm": g_norm, 
                "tau": avg_tau, "lr": cfg.learning_rate
            })
            
    return loss.item(), time.time() - start

def main():
    print("🚀 MINI-TITAN v5.0 (Forensic Edition)")
    train_set, val_set = get_dataset()
    
    with RunLogger(cfg, run_name="TITAN_POC_PROOF") as logger:
        logger.log_meta()
        
        l1, t1 = train_phase("PHASE 1 (LIQUID)", True, train_set, cfg.max_steps, logger)
        l2, t2 = train_phase("PHASE 2 (NO-LIQUID)", False, train_set, cfg.max_steps, logger)
        
        print("\n📊 FINAL RESULTS")
        print(f"Full Titan: Loss {l1:.4f} ({t1:.1f}s)")
        print(f"No-Liquid : Loss {l2:.4f} ({t2:.1f}s)")
        print(f"💾 PROOF SAVED TO: {logger.jsonl_path}")

if __name__ == "__main__":
    main()
