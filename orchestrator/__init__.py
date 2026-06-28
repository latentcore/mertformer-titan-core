"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI ORCHESTRATOR PACKAGE
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

NOTE: Bu paket (orchestrator/) inert / out-of-scope; 45K egitim yolunda
kapalidir. Burada tutulan surum etiketi legacy bir etikettir, kanonik
surum kaynagi degildir.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from .paths import AGIPaths
from .hardware import HardwareSense
from .web_sense import WebSense
from .audio_sense import AudioSense
from .sense_engine import SenseEngine
from .memory import GodMemory, DocChunk, DocIndexer, RAGEngine, HierarchicalMemoryContract
from .core import MertFormerOrchestrator, EpisodeBudget, EpisodeResult, main
from .agent_registry import AgentSpec, ALL_AGENT_SPECS, get_profile_specs
from .swarm_runtime import SwarmRuntime
from .governance import GovernanceGate, GovernancePolicy
from .planner import SwarmPlanner
from .verifier import SwarmVerifier, GateDecision
from .tool_registry import ToolSpec, default_tool_registry
from .self_improvement_guard import SelfImprovementGuard, ImprovementProposal, ApplyResult
from .alignment_contracts import AlignmentContracts, AlignmentViolation
from .compute_orchestrator import ComputeNode, ComputeOrchestrator

# AGI Cognitive Modules
from .reasoning_engine import (
    ReasoningEngine,
    ReasoningResult,
    ThoughtStep,
    ThoughtTree,
    Hypothesis,
    HypothesisSet,
)
from .tool_executor import ToolExecutor, ToolResult
from .self_audit import SelfAuditor, AuditReport
from .experience_store import ExperienceStore, Episode, StrategyPerformance
from .cognitive_loop import CognitiveLoop, CognitiveResult

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
    "GateDecision",
    "ToolSpec",
    "default_tool_registry",
    "SelfImprovementGuard",
    "ImprovementProposal",
    "ApplyResult",
    "AlignmentContracts",
    "AlignmentViolation",
    "ComputeNode",
    "ComputeOrchestrator",
    "MertFormerOrchestrator",
    "EpisodeBudget",
    "EpisodeResult",
    "main",
    # AGI Cognitive Modules
    "ReasoningEngine",
    "ReasoningResult",
    "ThoughtStep",
    "ThoughtTree",
    "Hypothesis",
    "HypothesisSet",
    "ToolExecutor",
    "ToolResult",
    "SelfAuditor",
    "AuditReport",
    "ExperienceStore",
    "Episode",
    "StrategyPerformance",
    "CognitiveLoop",
    "CognitiveResult",
]

# Legacy build etiketi; kanonik surum kaynagi degil (inert orchestrator paketi).
__version__ = "1.0-BUILD30-V2"
