"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI ORCHESTRATOR PACKAGE
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from .paths import AGIPaths
from .hardware import HardwareSense
from .web_sense import WebSense
from .audio_sense import AudioSense
from .sense_engine import SenseEngine
from .memory import GodMemory, DocChunk, DocIndexer, RAGEngine, HierarchicalMemoryContract
from .core import MertFormerOrchestrator, main
from .agent_registry import AgentSpec, ALL_AGENT_SPECS, get_profile_specs
from .swarm_runtime import SwarmRuntime
from .governance import GovernanceGate, GovernancePolicy
from .planner import SwarmPlanner
from .verifier import SwarmVerifier
from .tool_registry import ToolSpec, default_tool_registry
from .self_improvement_guard import SelfImprovementGuard, ImprovementProposal
from .alignment_contracts import AlignmentContracts, AlignmentViolation
from .compute_orchestrator import ComputeNode, ComputeOrchestrator

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
    "HierarchicalMemoryContract",
    "AgentSpec",
    "ALL_AGENT_SPECS",
    "get_profile_specs",
    "SwarmRuntime",
    "GovernanceGate",
    "GovernancePolicy",
    "SwarmPlanner",
    "SwarmVerifier",
    "ToolSpec",
    "default_tool_registry",
    "SelfImprovementGuard",
    "ImprovementProposal",
    "AlignmentContracts",
    "AlignmentViolation",
    "ComputeNode",
    "ComputeOrchestrator",
    "MertFormerOrchestrator",
]

__version__ = "1.0-BUILD30"
