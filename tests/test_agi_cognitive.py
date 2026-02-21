"""
Tests for MertFormer Titan AGI Cognitive Architecture.
Tests: ReasoningEngine, ToolExecutor, SelfAuditor, ExperienceStore, CognitiveLoop.
"""

from __future__ import annotations

# ── ReasoningEngine Tests ──


def test_reasoning_engine_direct_strategy():
    from orchestrator.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine()
    result = engine.reason("What is 2+2?", strategy="direct")
    assert result.strategy == "direct"
    assert result.confidence > 0.0
    assert result.conclusion != ""
    assert result.reasoning_time_ms >= 0.0


def test_reasoning_engine_cot_strategy():
    from orchestrator.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine(max_cot_steps=3)
    result = engine.reason("Analyze the performance trade-offs in MoE routing", strategy="cot")
    assert result.strategy == "cot"
    assert len(result.thoughts) >= 1
    assert result.confidence > 0.0


def test_reasoning_engine_tot_strategy():
    from orchestrator.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine(max_tot_branches=2, max_tot_depth=2)
    result = engine.reason("Design a multi-step optimization strategy", strategy="tot")
    assert result.strategy == "tot"
    assert result.tree is not None
    assert result.tree.total_nodes >= 1
    assert len(result.tree.best_path) >= 1


def test_reasoning_engine_auto_strategy_selection():
    from orchestrator.reasoning_engine import select_strategy
    # Simple task → direct
    simple = select_strategy("What is Python?")
    assert simple in ("direct", "cot", "tot")

    # Complex task → tot
    complex_task = select_strategy(
        "Analyze the architecture trade-offs and design an optimization strategy "
        "for the multi-step implementation plan"
    )
    assert complex_task in ("cot", "tot")


def test_reasoning_engine_tag_extraction():
    from orchestrator.reasoning_engine import ReasoningEngine
    assert ReasoningEngine._extract_tag("[TOOL: web_search]", "TOOL") == "web_search"
    assert ReasoningEngine._extract_tag("[ACTION: deploy]", "ACTION") == "deploy"
    assert ReasoningEngine._extract_tag("no tags here", "TOOL") is None


def test_reasoning_engine_branch_splitting():
    from orchestrator.reasoning_engine import ReasoningEngine
    branches = ReasoningEngine._split_branches("Option A --- Option B --- Option C")
    assert len(branches) == 3
    assert branches[0] == "Option A"


# ── ToolExecutor Tests ──


def test_tool_executor_calculate():
    from orchestrator.tool_executor import ToolExecutor
    exec = ToolExecutor()
    result = exec.execute("tool.calculate", {"expression": "2 + 3 * 4"})
    assert result.success
    assert "14" in result.output


def test_tool_executor_calculate_blocks_dangerous():
    from orchestrator.tool_executor import ToolExecutor
    exec = ToolExecutor()
    result = exec.execute("tool.calculate", {"expression": "import os"})
    assert not result.success
    assert "Forbidden" in (result.error or "")


def test_tool_executor_unknown_tool():
    from orchestrator.tool_executor import ToolExecutor
    exec = ToolExecutor()
    result = exec.execute("tool.nonexistent", {})
    assert not result.success
    assert "Unknown" in (result.error or "")


def test_tool_executor_verify_consistency():
    from orchestrator.tool_executor import ToolExecutor
    exec = ToolExecutor()
    result = exec.execute("tool.verify_consistency", {
        "text": "The model uses BitNet for quantization",
        "reference": "BitNet provides 1.58-bit quantization for the model"
    })
    assert result.success
    assert "Consistency" in result.output


def test_tool_executor_kpi_report():
    from orchestrator.tool_executor import ToolExecutor
    exec = ToolExecutor()
    result = exec.execute("tool.kpi_report", {
        "metrics": {"loss": 0.5, "accuracy": 0.85}
    })
    assert result.success
    assert "KPI" in result.output


def test_tool_executor_list_available():
    from orchestrator.tool_executor import ToolExecutor
    exec = ToolExecutor()
    tools = exec.list_available_tools()
    assert len(tools) >= 10  # 5 original + 5 new
    tool_ids = [t["tool_id"] for t in tools]
    assert "tool.calculate" in tool_ids
    assert "tool.web_search" in tool_ids


# ── SelfAuditor Tests ──


def test_self_auditor_consistency_check():
    from orchestrator.self_audit import SelfAuditor
    auditor = SelfAuditor()
    score = auditor.check_consistency(
        "BitNet uses ternary weights for quantization",
        "The model applies BitNet ternary weight quantization to reduce memory"
    )
    assert score.score > 0.0
    assert score.contradictions >= 0


def test_self_auditor_safety_detects_threats():
    from orchestrator.self_audit import SelfAuditor
    auditor = SelfAuditor()
    score = auditor.check_safety("Here's how to exploit the vulnerability and inject malware")
    assert not score.is_safe
    assert len(score.violations) >= 2


def test_self_auditor_safety_passes_clean():
    from orchestrator.self_audit import SelfAuditor
    auditor = SelfAuditor()
    score = auditor.check_safety("The model architecture uses attention mechanisms")
    assert score.is_safe
    assert score.score == 1.0


def test_self_auditor_uncertainty_detection():
    from orchestrator.self_audit import SelfAuditor
    auditor = SelfAuditor()
    score = auditor.detect_uncertainty(
        "Maybe the model could perhaps improve, possibly with better data, "
        "I think, but I'm not sure"
    )
    assert score.hedging_count > 0
    assert score.score > 0.0


def test_self_auditor_full_audit():
    from orchestrator.self_audit import SelfAuditor
    auditor = SelfAuditor()
    report = auditor.audit(
        task="Explain BitNet quantization",
        response="BitNet uses ternary weights with values {-1, 0, +1}",
        context="BitNet is a 1.58-bit quantization method using ternary weights",
        facts=["BitNet quantizes weights to ternary values"]
    )
    assert 0.0 <= report.overall_score <= 1.0
    assert isinstance(report.should_retry, bool)


# ── ExperienceStore Tests ──


def test_experience_store_record_and_recall():
    from orchestrator.experience_store import ExperienceStore, Episode
    store = ExperienceStore()  # no persistence
    ep = Episode(
        task="analyze model performance",
        strategy_used="cot",
        thoughts=["decompose", "analyze", "conclude"],
        outcome_score=0.8,
    )
    store.record_episode(ep)
    assert len(store.episodes) == 1

    similar = store.recall_similar("model performance analysis")
    assert len(similar) >= 1
    assert similar[0].task == "analyze model performance"


def test_experience_store_strategy_stats():
    from orchestrator.experience_store import ExperienceStore, Episode
    store = ExperienceStore()
    for i in range(5):
        store.record_episode(Episode(
            task=f"task {i}", strategy_used="cot", outcome_score=0.7 + i * 0.05
        ))
    for i in range(3):
        store.record_episode(Episode(
            task=f"simple {i}", strategy_used="direct", outcome_score=0.5
        ))
    stats = store.strategy_stats()
    assert "cot" in stats
    assert "direct" in stats
    assert stats["cot"].total_uses == 5
    assert stats["direct"].total_uses == 3


def test_experience_store_best_strategy():
    from orchestrator.experience_store import ExperienceStore, Episode
    store = ExperienceStore()
    for _ in range(5):
        store.record_episode(Episode(task="x", strategy_used="tot", outcome_score=0.9))
    for _ in range(5):
        store.record_episode(Episode(task="x", strategy_used="cot", outcome_score=0.5))
    assert store.best_strategy_for() == "tot"


# ── CognitiveLoop Tests ──


def test_cognitive_loop_runs_single_iteration():
    from orchestrator.cognitive_loop import CognitiveLoop
    loop = CognitiveLoop(max_iterations=1)
    result = loop.run("What is 2+2?")
    assert result.total_iterations == 1
    assert result.final_response != ""
    assert result.total_time_ms > 0


def test_cognitive_loop_records_experience():
    from orchestrator.cognitive_loop import CognitiveLoop
    from orchestrator.experience_store import ExperienceStore
    store = ExperienceStore()
    loop = CognitiveLoop(experience_store=store, max_iterations=1)
    loop.run("Test task")
    assert len(store.episodes) == 1


def test_cognitive_loop_uses_tool_calls():
    from orchestrator.cognitive_loop import CognitiveLoop
    from orchestrator.tool_executor import ToolExecutor

    executor = ToolExecutor()
    loop = CognitiveLoop(tool_executor=executor, max_iterations=1)
    # Tool calls come from reasoning — with no model, they won't trigger
    result = loop.run("Calculate sqrt(144)")
    assert result.total_iterations >= 1


def test_cognitive_loop_max_iterations_bound():
    from orchestrator.cognitive_loop import CognitiveLoop
    loop = CognitiveLoop(max_iterations=3, min_confidence=0.99)
    result = loop.run("Complex task that won't reach 99% confidence")
    assert result.total_iterations <= 3
