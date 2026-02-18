"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - ARCHITECTURAL PATHS
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30"
__author__ = "Mert"

import pathlib

# TR: Config import - fallback mekanizmalı
# EN: Config import - with fallback mechanism
try:
    from config.config import cfg
except ImportError:
    class cfg:  # type: ignore
        save_dir = "checkpoints"
        model_name = "mertformer"


class AGIPaths:
    """TR: AGI Orchestrator için dosya ve dizin yolları. / EN: File and directory paths for AGI Orchestrator."""
    
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    THIS_FILE = pathlib.Path(__file__).resolve()

    # TR: Hafıza / EN: Memory
    MEMORY_DIR = ROOT / "agi_memory"
    MEMORY_FILE = MEMORY_DIR / "memory_omni.jsonl"

    # TR: Doküman RAG / EN: Document RAG
    DOC_DIR = ROOT / "agi_docs"

    # TR: Vektör indeks / EN: Vector index
    VECTOR_DIR = ROOT / "agi_vector_index"
    VECTOR_FILE = VECTOR_DIR / "vector_index.json"

    # TR: Checkpoint / EN: Checkpoint
    CHECKPOINT_DIR = ROOT / getattr(cfg, "save_dir", "checkpoints")
    CHECKPOINT_FILE = CHECKPOINT_DIR / f"{getattr(cfg, 'model_name', 'mertformer')}_latest.pt"
    
    @classmethod
    def ensure_dirs(cls):
        """TR: Tüm gerekli dizinleri oluştur. / EN: Create all required directories."""
        cls.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOC_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
