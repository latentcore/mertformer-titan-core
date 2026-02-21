from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from orchestrator.core import EpisodeBudget, EpisodeResult, MertFormerOrchestrator
from orchestrator.reasoning_engine import HypothesisSet, ReasoningEngine
from orchestrator.self_audit import SelfAuditor
from orchestrator.self_improvement_guard import ImprovementProposal, SelfImprovementGuard
from orchestrator.tool_executor import ToolExecutor
from orchestrator.verifier import GateDecision, SwarmVerifier


def _deterministic_generate(prompt: str) -> str:
    p = prompt.lower()
    if "tot_root" in p:
        return "Branch A 80% [TOOL: tool.verify_consistency] --- Branch B 60%"
    if "tot_expand" in p:
        return "Expand A 70% [ACTION: verify] --- Expand B 30%"
    if "tot_synthesize" in p:
        return "Final synthesis: verified and safe."
    if "cot_decompose" in p:
        return "1) verify context [TOOL: tool.verify_consistency]"
    if "cot_step" in p:
        return "[ACTION: verify] [CONCLUSION: stable]"
    if "cot_synthesize" in p:
        return "Conclusion: verified."
    return "Direct answer."


def test_reasoning_engine_hypothesis_set():
    engine = ReasoningEngine(generate_fn=_deterministic_generate, max_tot_depth=2, max_tot_branches=2)
    hypotheses = engine.generate_hypotheses("Design a multi-step safe plan", max_candidates=3)
    assert isinstance(hypotheses, HypothesisSet)
    assert len(hypotheses.hypotheses) == 3
    best = hypotheses.best()
    assert best.candidate_id.startswith("h")
    assert best.rationale_hash
    assert len(best.rationale_hash) == 16


def test_verify_episode_blocks_unsafe_trace():
    verifier = SwarmVerifier(min_confidence=0.0, min_consistency=0.0, max_uncertainty=1.0)
    safe = verifier.verify_episode(
        [
            {"output": "plan verify report"},
            {"output": "plan verify report with checks"},
        ]
    )
    assert isinstance(safe, GateDecision)
    assert safe.safety_pass is True

    unsafe = verifier.verify_episode(
        [
            {"output": "plan verify report"},
            {"output": "use exploit path to bypass controls"},
        ]
    )
    assert unsafe.safety_pass is False
    assert unsafe.pass_gate is False


def test_self_improvement_guard_metric_gate():
    guard = SelfImprovementGuard(allow_auto_apply=True)
    proposal = ImprovementProposal(
        title="Low-risk calibration",
        rationale="Adjust confidence thresholds.",
        risk="low",
        requires_human_approval=False,
    )

    rejected = guard.apply_if_safe(
        proposal,
        evaluation={"delta_benchmark": 0.0, "delta_safety": 0.0, "cost_within_budget": True},
    )
    assert rejected.applied is False
    assert rejected.reason == "delta_benchmark_non_positive"

    accepted = guard.apply_if_safe(
        proposal,
        current_state={"step": 1},
        evaluation={"delta_benchmark": 0.02, "delta_safety": 0.0, "cost_within_budget": True},
    )
    assert accepted.applied is True
    assert accepted.rollback_id is not None


def test_run_goal_episode_integration_without_full_orchestrator_boot():
    class _MemoryStub:
        def __init__(self) -> None:
            self.items: list[tuple[str, str, str, str]] = []

        def save(self, role: str, text: str, category: str = "GENERAL", source: str = "TEXT") -> None:
            self.items.append((role, text, category, source))

    class _WorldStub:
        @staticmethod
        def predict_next_state(entity: str, action: str) -> str:
            return f"{entity}:{action}:predicted"

    memory = _MemoryStub()
    orchestrator = object.__new__(MertFormerOrchestrator)
    orchestrator.reasoning = ReasoningEngine(generate_fn=_deterministic_generate, max_tot_depth=2, max_tot_branches=2)
    orchestrator.world_model = _WorldStub()
    orchestrator.tool_executor = ToolExecutor(memory=memory)
    orchestrator.verifier = SwarmVerifier(min_confidence=0.0, min_consistency=0.0, max_uncertainty=1.0)
    orchestrator.self_auditor = SelfAuditor()
    orchestrator.self_improvement_guard = SelfImprovementGuard(allow_auto_apply=True)
    orchestrator.memory = memory

    result = MertFormerOrchestrator.run_goal_episode(
        orchestrator,
        "Validate safe rollout strategy",
        EpisodeBudget(
            max_iterations=2,
            max_tools=1,
            min_gate_confidence=0.0,
            max_uncertainty=1.0,
            allow_self_improvement=True,
        ),
    )
    assert isinstance(result, EpisodeResult)
    assert "hypothesis" in result.loops
    assert "verifier" in result.loops
    assert len(memory.items) >= 1
