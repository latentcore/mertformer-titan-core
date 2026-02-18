"""Agent registry for deterministic swarm orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    capabilities: tuple[str, ...]
    policy_tags: tuple[str, ...]
    priority: int


def _cap(*items: str) -> tuple[str, ...]:
    return tuple(items)


def build_agent_specs() -> List[AgentSpec]:
    """Return a fixed 45-agent catalog for nano/mid/omega profiles."""
    specs = [
        AgentSpec("A01", "architect", _cap("architecture", "integration", "risk"), _cap("offline_safe"), 1),
        AgentSpec("A02", "planner", _cap("planning", "decomposition", "routing"), _cap("offline_safe"), 2),
        AgentSpec("A03", "verifier", _cap("verification", "consistency", "uncertainty"), _cap("offline_safe"), 3),
        AgentSpec("A04", "trainer", _cap("training", "stability", "gradients"), _cap("offline_safe"), 4),
        AgentSpec("A05", "moe_specialist", _cap("moe", "routing", "balance"), _cap("offline_safe"), 5),
        AgentSpec("A06", "liquid_specialist", _cap("liquid", "state", "dynamics"), _cap("offline_safe"), 6),
        AgentSpec("A07", "attention_specialist", _cap("mla", "rope", "gqa"), _cap("offline_safe"), 7),
        AgentSpec("A08", "quant_specialist", _cap("bitnet", "quant", "kernel"), _cap("offline_safe"), 8),
        AgentSpec("A09", "export_specialist", _cap("onnx", "coreml", "metadata"), _cap("offline_safe"), 9),
        AgentSpec("A10", "sdk_specialist", _cap("sdk", "cli", "api"), _cap("offline_safe"), 10),
        AgentSpec("A11", "kpi_specialist", _cap("kpi", "report", "pilot"), _cap("offline_safe"), 11),
        AgentSpec("A12", "qa_engineer", _cap("pytest", "regression", "gates"), _cap("offline_safe"), 12),
        AgentSpec("A13", "security_engineer", _cap("security", "policy", "audit"), _cap("offline_safe"), 13),
        AgentSpec("A14", "release_engineer", _cap("release", "checksums", "packaging"), _cap("offline_safe"), 14),
        AgentSpec("A15", "docs_engineer", _cap("docs", "consistency", "localization"), _cap("offline_safe"), 15),
        AgentSpec("A16", "data_engineer", _cap("dataset", "pipeline", "hashing"), _cap("offline_safe"), 16),
        AgentSpec("A17", "benchmark_engineer", _cap("benchmark", "throughput", "latency"), _cap("offline_safe"), 17),
        AgentSpec("A18", "observability_engineer", _cap("telemetry", "metrics", "health"), _cap("offline_safe"), 18),
        AgentSpec("A19", "failure_budget_engineer", _cap("failure_budget", "pivot", "alerts"), _cap("offline_safe"), 19),
        AgentSpec("A20", "governance_engineer", _cap("governance", "boundaries", "compliance"), _cap("offline_safe"), 20),
        AgentSpec("A21", "orchestrator_engineer", _cap("swarm", "runtime", "determinism"), _cap("offline_safe"), 21),
        AgentSpec("A22", "memory_engineer", _cap("memory", "rag", "index"), _cap("offline_safe"), 22),
        AgentSpec("A23", "planner_controller", _cap("planning", "controller", "tools"), _cap("offline_safe"), 23),
        AgentSpec("A24", "critic", _cap("critic", "self_check", "scoring"), _cap("offline_safe"), 24),
        AgentSpec("A25", "compiler_engineer", _cap("compile", "graph", "optim"), _cap("offline_safe"), 25),
        AgentSpec("A26", "ddp_engineer", _cap("ddp", "sync", "distributed"), _cap("offline_safe"), 26),
        AgentSpec("A27", "inference_engineer", _cap("inference", "cache", "decode"), _cap("offline_safe"), 27),
        AgentSpec("A28", "kaggle_engineer", _cap("kaggle", "demo", "compare"), _cap("offline_safe"), 28),
        AgentSpec("A29", "mobile_engineer", _cap("mobile", "npu", "deployment"), _cap("offline_safe"), 29),
        AgentSpec("A30", "test_architect", _cap("test_design", "coverage", "acceptance"), _cap("offline_safe"), 30),
        AgentSpec("A31", "lint_engineer", _cap("lint", "style", "quality"), _cap("offline_safe"), 31),
        AgentSpec("A32", "ci_engineer", _cap("ci", "pipelines", "gates"), _cap("offline_safe"), 32),
        AgentSpec("A33", "ops_engineer", _cap("operations", "runbooks", "incident"), _cap("offline_safe"), 33),
        AgentSpec("A34", "forensics_engineer", _cap("forensics", "sha256", "integrity"), _cap("offline_safe"), 34),
        AgentSpec("A35", "compliance_engineer", _cap("compliance", "controls", "audit"), _cap("offline_safe"), 35),
        AgentSpec("A36", "prompt_engineer", _cap("prompting", "instruction", "evaluation"), _cap("offline_safe"), 36),
        AgentSpec("A37", "localization_engineer", _cap("tr", "en", "de"), _cap("offline_safe"), 37),
        AgentSpec("A38", "sales_engineer", _cap("pilot", "kpi", "signoff"), _cap("offline_safe"), 38),
        AgentSpec("A39", "partner_engineer", _cap("partnership", "nda", "handoff"), _cap("offline_safe"), 39),
        AgentSpec("A40", "economics_engineer", _cap("cost", "roi", "tco"), _cap("offline_safe"), 40),
        AgentSpec("A41", "safety_engineer", _cap("safety", "abuse", "policy"), _cap("offline_safe"), 41),
        AgentSpec("A42", "redteam_engineer", _cap("redteam", "adversarial", "resilience"), _cap("offline_safe"), 42),
        AgentSpec("A43", "knowledge_engineer", _cap("knowledge", "taxonomy", "retrieval"), _cap("offline_safe"), 43),
        AgentSpec("A44", "world_model_engineer", _cap("world_model", "dynamics", "forecast"), _cap("offline_safe"), 44),
        AgentSpec("A45", "chief_operator", _cap("governance", "decision", "approval"), _cap("offline_safe"), 45),
    ]
    return specs


ALL_AGENT_SPECS: tuple[AgentSpec, ...] = tuple(build_agent_specs())


def get_profile_agent_ids(mode: str) -> List[str]:
    """Return deterministic profile membership for a mode."""
    normalized = str(mode).strip().lower()
    if normalized == "nano":
        return [spec.agent_id for spec in ALL_AGENT_SPECS[:3]]
    if normalized == "mid":
        return [spec.agent_id for spec in ALL_AGENT_SPECS[:15]]
    if normalized == "omega":
        return [spec.agent_id for spec in ALL_AGENT_SPECS]
    raise ValueError(f"Unknown swarm mode: {mode}")


def get_profile_specs(mode: str) -> List[AgentSpec]:
    ids = set(get_profile_agent_ids(mode))
    return [spec for spec in ALL_AGENT_SPECS if spec.agent_id in ids]


def index_specs(specs: Iterable[AgentSpec]) -> Dict[str, AgentSpec]:
    return {spec.agent_id: spec for spec in specs}
