"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI ORCHESTRATOR PACKAGE
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

NOTE: Bu paket (orchestrator/) inert / out-of-scope; 45K egitim yolunda
kapalidir. Burada tutulan surum etiketi legacy bir etikettir, kanonik
surum kaynagi degildir.

EAGER-IMPORT NOTE (2026-07-08, documentation-only):
This package imports ALL 24 of its submodules at package level (below). Therefore
train/train.py's single real orchestrator import
    from orchestrator.distillation_manager import DistillationManager
makes Python execute the module-level code of EVERY file in this package, even though
only that one class is ever used and the package is otherwise inert on the 45K path.

As of this pass, every dependency those submodules touch was checked and is either
properly declared (e.g. networkx in requirements.txt) or guarded by a module-level
try/except ImportError (e.g. sense_engine.py, audio_sense.py, web_sense.py) -- so there
is NO live bug today. The coupling is nonetheless real: any new file added to this
package that imports an undeclared/unguarded dependency will silently break train.py's
ability to even start. Any new file here MUST keep its imports declared-or-guarded,
precisely because this package loads unconditionally the moment anything imports from it.

Making these imports lazy/guarded would be a real behavior change to inert code and is
deliberately out of scope; this note is the correct minimal action.

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
