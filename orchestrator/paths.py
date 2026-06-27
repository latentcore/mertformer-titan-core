"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - ARCHITECTURAL PATHS
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

NOTE: This module lives under orchestrator/ and is INERT / OUT-OF-SCOPE for the
45K training path (orchestrator is closed/feature-flagged off in that path).
This file only defines pathlib directory locations; it does not touch any
hardware backend.

Project: Mobile-First LLM Architecture (general; no measured NPU/accelerator
         claim — header previously named "Samsung S25 NPU", which is an
         unverified, context-free hardware claim and unrelated to this file).
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

# NOTE: legacy/fossil version label hand-copied across orchestrator modules
# (memory.py/sense_engine.py carry the same string). Not derived from a single
# source of truth; treat as cosmetic, not an authoritative build version.
__version__ = "1.0-BUILD30-V2"
__author__ = "Mert Yünlü"

import pathlib

# Config import - with fallback mechanism
try:
    from config.config import cfg
except ImportError:
    class cfg:  # type: ignore
        save_dir = "checkpoints"
        model_name = "mertformer"


class AGIPaths:
    """File and directory paths for AGI Orchestrator."""
    
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    THIS_FILE = pathlib.Path(__file__).resolve()

    # Memory
    MEMORY_DIR = ROOT / "agi_memory"
    MEMORY_FILE = MEMORY_DIR / "memory_omni.jsonl"

    # Document RAG
    DOC_DIR = ROOT / "agi_docs"

    # Vector index
    VECTOR_DIR = ROOT / "agi_vector_index"
    VECTOR_FILE = VECTOR_DIR / "vector_index.json"

    # Checkpoint
    CHECKPOINT_DIR = ROOT / getattr(cfg, "save_dir", "checkpoints")
    CHECKPOINT_FILE = CHECKPOINT_DIR / f"{getattr(cfg, 'model_name', 'mertformer')}_latest.pt"
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        cls.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOC_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
