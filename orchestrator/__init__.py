"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI ORCHESTRATOR PACKAGE
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

from .paths import AGIPaths
from .hardware import HardwareSense
from .web_sense import WebSense
from .audio_sense import AudioSense
from .sense_engine import SenseEngine
from .memory import GodMemory, DocChunk, DocIndexer, RAGEngine
from .core import MertFormerOrchestrator, main

__all__ = [
    "AGIPaths",
    "HardwareSense", 
    "WebSense",
    "AudioSense",
    "SenseEngine",
    "GodMemory",
    "DocChunk",
    "DocIndexer",
    "RAGEngine",
    "MertFormerOrchestrator",
]

__version__ = "27.0-FINAL"
