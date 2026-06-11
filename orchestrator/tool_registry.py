"""Tool registry contract for planner/controller routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    description: str
    capabilities: tuple[str, ...]
    allowed_offline: bool = True


def default_tool_registry() -> Dict[str, ToolSpec]:
    specs = [
        ToolSpec("tool.search_local_docs", "Search indexed local docs", ("retrieval", "memory", "docs"), True),
        ToolSpec("tool.verify_consistency", "Cross-check response consistency", ("verification", "consistency"), True),
        ToolSpec("tool.kpi_report", "Generate KPI summary", ("kpi", "reporting"), True),
        ToolSpec("tool.swarm_route", "Dispatch role-based swarm execution", ("routing", "planning"), True),
        ToolSpec("tool.eval_generalization", "Run local generalization suite", ("evaluation", "generalization"), True),
        # New AGI tools
        ToolSpec("tool.web_search", "Search the web via DuckDuckGo", ("web", "search", "research"), False),
        ToolSpec("tool.calculate", "Perform safe mathematical calculations", ("math", "calculate", "compute"), True),
        ToolSpec("tool.memorize", "Store information in memory", ("memory", "store", "remember"), True),
        ToolSpec("tool.recall", "Recall information from memory", ("memory", "recall", "retrieve"), True),
        ToolSpec("tool.analyze_image", "Analyze image using CLIP vision", ("vision", "image", "analyze"), True),
    ]
    return {s.tool_id: s for s in specs}


def list_tools_by_capability(capability: str, registry: Dict[str, ToolSpec] | None = None) -> List[ToolSpec]:
    registry = registry or default_tool_registry()
    key = capability.lower().strip()
    return [spec for spec in registry.values() if key in spec.capabilities]

