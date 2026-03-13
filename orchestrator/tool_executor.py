from __future__ import annotations
"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI TOOL EXECUTOR
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

TR: Araç çalıştırma motoru — governance-gated, sandboxed, timeout-enforced.
EN: Tool execution engine — governance-gated, sandboxed, timeout-enforced.
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import math
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tool_registry import ToolSpec, default_tool_registry
from .governance import GovernanceGate, GovernancePolicy


# -----------------------------------------------------------------------------
# TR: VERİ YAPILARI / EN: DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class ToolResult:
    """TR: Araç çalıştırma sonucu. / EN: Tool execution result."""
    tool_id: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    governance_check: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# TR: ARAÇ ÇALIŞTIRMA MOTORU / EN: TOOL EXECUTION ENGINE
# -----------------------------------------------------------------------------

class ToolExecutor:
    """
    TR: Güvenli araç çalıştırma motoru.
    EN: Secure tool execution engine.

    - Her araç çalıştırma öncesinde governance kontrolü yapılır
    - Timeout enforcement (varsayılan 30s)
    - Tüm sonuçlar yapılandırılmış ToolResult olarak döndürülür
    """

    def __init__(
        self,
        governance_gate: Optional[GovernanceGate] = None,
        memory: Optional[Any] = None,
        web_sense: Optional[Any] = None,
        sense_engine: Optional[Any] = None,
        registry: Optional[Dict[str, ToolSpec]] = None,
        max_timeout_s: float = 30.0,
    ) -> None:
        self.governance = governance_gate or GovernanceGate(
            GovernancePolicy(
                offline_only=False,
                allow_network=True,
                allowed_actions=(
                    "analyze", "plan", "verify", "report", "summarize",
                    "search", "calculate", "memorize", "recall", "see",
                ),
            )
        )
        self.memory = memory
        self.web_sense = web_sense
        self.sense_engine = sense_engine
        self.registry = registry or default_tool_registry()
        self.max_timeout_s = float(max_timeout_s)

        # TR: Araç → çalıştırıcı eşlemesi / EN: Tool → executor mapping
        self._executors: Dict[str, Callable[..., ToolResult]] = {
            "tool.search_local_docs": self._exec_search_docs,
            "tool.verify_consistency": self._exec_verify_consistency,
            "tool.kpi_report": self._exec_kpi_report,
            "tool.swarm_route": self._exec_swarm_route,
            "tool.eval_generalization": self._exec_eval_generalization,
            "tool.web_search": self._exec_web_search,
            "tool.calculate": self._exec_calculate,
            "tool.memorize": self._exec_memorize,
            "tool.recall": self._exec_recall,
            "tool.analyze_image": self._exec_analyze_image,
        }

    def execute(self, tool_id: str, params: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        TR: Aracı çalıştır (governance + timeout ile).
        EN: Execute tool (with governance + timeout).
        """
        params = params or {}
        t0 = time.monotonic()

        # TR: Kayıt kontrolü / EN: Registry check
        if tool_id not in self.registry and tool_id not in self._executors:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=f"Unknown tool: {tool_id}",
                execution_time_ms=0.0,
                governance_check=False,
            )

        # TR: Governance kontrolü / EN: Governance check
        tool_spec = self.registry.get(tool_id)
        # TR: Araç tipinden aksiyon çıkar / EN: Derive action from tool type
        _tool_action_map = {
            "tool.search_local_docs": "search",
            "tool.verify_consistency": "verify",
            "tool.kpi_report": "report",
            "tool.swarm_route": "plan",
            "tool.eval_generalization": "analyze",
            "tool.web_search": "search",
            "tool.calculate": "calculate",
            "tool.memorize": "memorize",
            "tool.recall": "recall",
            "tool.analyze_image": "see",
        }
        action_name = _tool_action_map.get(tool_id, "analyze")
        gov = self.governance.evaluate(
            f"Execute tool {tool_id} with params {params}",
            requested_actions=[action_name],
        )
        if not gov.allowed:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=f"Governance blocked: {', '.join(gov.reasons)}",
                execution_time_ms=(time.monotonic() - t0) * 1000.0,
                governance_check=False,
            )

        # TR: Çalıştır / EN: Execute
        executor = self._executors.get(tool_id)
        if executor is None:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=f"No executor registered for {tool_id}",
                execution_time_ms=(time.monotonic() - t0) * 1000.0,
            )

        try:
            result = self._execute_with_timeout(executor, params)
            result.execution_time_ms = (time.monotonic() - t0) * 1000.0
            return result
        except TimeoutError as e:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=str(e),
                execution_time_ms=(time.monotonic() - t0) * 1000.0,
            )
        except Exception as e:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=f"Execution error: {e}",
                execution_time_ms=(time.monotonic() - t0) * 1000.0,
            )

    def _execute_with_timeout(
        self,
        executor: Callable[[Dict[str, Any]], ToolResult],
        params: Dict[str, Any],
    ) -> ToolResult:
        """Run a tool with best-effort timeout enforcement."""
        timeout_s = float(max(0.0, self.max_timeout_s))
        if timeout_s <= 0.0:
            return executor(params)

        # signal-based timeout only works reliably on main thread (POSIX).
        if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
            return executor(params)

        class _ToolDeadline(Exception):
            pass

        def _handler(signum: int, frame: object) -> None:
            raise _ToolDeadline()

        prev_handler = signal.getsignal(signal.SIGALRM)
        prev_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return executor(params)
        except _ToolDeadline as exc:
            raise TimeoutError(f"Execution timeout after {timeout_s:.1f}s") from exc
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
            signal.signal(signal.SIGALRM, prev_handler)
            if prev_timer and (prev_timer[0] > 0.0 or prev_timer[1] > 0.0):
                signal.setitimer(signal.ITIMER_REAL, prev_timer[0], prev_timer[1])

    def list_available_tools(self) -> List[Dict[str, str]]:
        """TR: Kullanılabilir araçları listele. / EN: List available tools."""
        return [
            {
                "tool_id": tid,
                "description": spec.description,
                "capabilities": ", ".join(spec.capabilities),
                "has_executor": tid in self._executors,
            }
            for tid, spec in self.registry.items()
        ]

    # -----------------------------------------------------------------
    # TR: ARAÇ UYGULAMALARI / EN: TOOL IMPLEMENTATIONS
    # -----------------------------------------------------------------

    def _exec_search_docs(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Yerel belgelerde arama. / EN: Search local documents."""
        query = str(params.get("query", ""))
        if not query:
            return ToolResult("tool.search_local_docs", False, "", error="Missing 'query' parameter")

        if self.memory is None:
            return ToolResult("tool.search_local_docs", False, "", error="Memory not available")

        try:
            results = self.memory.recall(query, top_k=int(params.get("top_k", 5)))
            return ToolResult(
                "tool.search_local_docs",
                success=bool(results),
                output=results if results else "No relevant documents found.",
            )
        except Exception as e:
            return ToolResult("tool.search_local_docs", False, "", error=str(e))

    def _exec_verify_consistency(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Tutarlılık doğrulaması. / EN: Consistency verification."""
        text = str(params.get("text", ""))
        reference = str(params.get("reference", ""))

        if not text:
            return ToolResult("tool.verify_consistency", False, "", error="Missing 'text' parameter")

        # TR: Basit tutarlılık kontrolü — kelime örtüşmesi
        # EN: Simple consistency check — word overlap
        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split()) if reference else set()

        if ref_words:
            overlap = len(text_words & ref_words) / max(1, len(text_words | ref_words))
            consistency = f"Consistency score: {overlap:.2f}"
            notes = []
            if overlap < 0.1:
                notes.append("⚠️ Very low overlap — possible hallucination")
            elif overlap > 0.5:
                notes.append("✅ High consistency with reference")
        else:
            consistency = "No reference provided — self-consistency check only"
            overlap = 0.5
            notes = ["ℹ️ Self-consistency analysis performed (no reference)"]

        # TR: Çelişki tespiti / EN: Contradiction detection
        negation_pairs = [
            ("yes", "no"), ("true", "false"), ("always", "never"),
            ("all", "none"), ("increase", "decrease"), ("evet", "hayır"),
            ("doğru", "yanlış"), ("her zaman", "asla"),
        ]
        contradictions = []
        for pos, neg in negation_pairs:
            if pos in text.lower() and neg in text.lower():
                contradictions.append(f"Potential contradiction: '{pos}' vs '{neg}'")

        output_parts = [consistency]
        if notes:
            output_parts.extend(notes)
        if contradictions:
            output_parts.append("--- Contradictions ---")
            output_parts.extend(contradictions)

        return ToolResult(
            "tool.verify_consistency",
            success=True,
            output="\n".join(output_parts),
            metadata={"overlap": overlap, "contradictions": len(contradictions)},
        )

    def _exec_kpi_report(self, params: Dict[str, Any]) -> ToolResult:
        """TR: KPI raporu oluştur. / EN: Generate KPI report."""
        metrics = params.get("metrics", {})
        if not metrics:
            return ToolResult(
                "tool.kpi_report", True,
                output="No metrics provided. Run training first to collect KPIs.",
            )

        lines = ["📊 KPI REPORT", "=" * 40]
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("=" * 40)
        return ToolResult("tool.kpi_report", True, output="\n".join(lines))

    def _exec_swarm_route(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Swarm route tool stub. / EN: Swarm route tool stub."""
        objective = str(params.get("objective", "")).strip()
        output = "Swarm routing delegated to planner."
        if objective:
            output += f" Objective: {objective}"
        return ToolResult("tool.swarm_route", True, output=output)

    def _exec_eval_generalization(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Generalization eval tool stub. / EN: Generalization eval tool stub."""
        metrics = params.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            summary = ", ".join(f"{k}={v}" for k, v in metrics.items())
            return ToolResult("tool.eval_generalization", True, output=f"Eval summary: {summary}")
        return ToolResult(
            "tool.eval_generalization",
            True,
            output="Generalization evaluation entrypoint reached (no metrics payload).",
        )

    def _exec_web_search(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Web araması. / EN: Web search."""
        query = str(params.get("query", ""))
        if not query:
            return ToolResult("tool.web_search", False, "", error="Missing 'query' parameter")

        if self.web_sense is None or not getattr(self.web_sense, "enabled", False):
            return ToolResult(
                "tool.web_search", False, "",
                error="Web sense module not available or disabled",
            )

        try:
            max_results = int(params.get("max_results", 5))
            results = self.web_sense.search(query, max_results=max_results)
            return ToolResult("tool.web_search", success=True, output=results)
        except Exception as e:
            return ToolResult("tool.web_search", False, "", error=str(e))

    def _exec_calculate(self, params: Dict[str, Any]) -> ToolResult:
        """
        TR: Güvenli matematiksel hesaplama.
        EN: Safe mathematical calculation.
        """
        expression = str(params.get("expression", ""))
        if not expression:
            return ToolResult("tool.calculate", False, "", error="Missing 'expression' parameter")

        # TR: Güvenli eval — sadece matematiksel operatörler
        # EN: Safe eval — mathematical operators only
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow,
            # TR: math modülünden güvenli fonksiyonlar
            "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
            "log10": math.log10, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
            "exp": math.exp, "factorial": math.factorial,
        }

        try:
            # TR: Tehlikeli karakter kontrolü / EN: Dangerous character check
            forbidden = ["import", "exec", "eval", "open", "os.", "sys.", "__", "lambda"]
            for f in forbidden:
                if f in expression.lower():
                    return ToolResult(
                        "tool.calculate", False, "",
                        error=f"Forbidden token in expression: {f}",
                    )

            result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
            return ToolResult(
                "tool.calculate", True,
                output=f"{expression} = {result}",
                metadata={"result": result},
            )
        except Exception as e:
            return ToolResult("tool.calculate", False, "", error=f"Calculation error: {e}")

    def _exec_memorize(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Bilgiyi hafızaya kaydet. / EN: Store information in memory."""
        text = str(params.get("text", ""))
        category = str(params.get("category", "GENERAL"))

        if not text:
            return ToolResult("tool.memorize", False, "", error="Missing 'text' parameter")

        if self.memory is None:
            return ToolResult("tool.memorize", False, "", error="Memory not available")

        try:
            self.memory.save("system", text, category=category, source="TOOL")
            return ToolResult(
                "tool.memorize", True,
                output=f"Memorized: {text[:100]}... [category={category}]",
            )
        except Exception as e:
            return ToolResult("tool.memorize", False, "", error=str(e))

    def _exec_recall(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Hafızadan bilgi hatırla. / EN: Recall information from memory."""
        query = str(params.get("query", ""))
        if not query:
            return ToolResult("tool.recall", False, "", error="Missing 'query' parameter")

        if self.memory is None:
            return ToolResult("tool.recall", False, "", error="Memory not available")

        try:
            results = self.memory.recall(query, top_k=int(params.get("top_k", 10)))
            return ToolResult(
                "tool.recall", success=bool(results),
                output=results if results else "No relevant memories found.",
            )
        except Exception as e:
            return ToolResult("tool.recall", False, "", error=str(e))

    def _exec_analyze_image(self, params: Dict[str, Any]) -> ToolResult:
        """TR: Görüntü analizi (CLIP). / EN: Image analysis (CLIP)."""
        image_path = str(params.get("image_path", ""))
        if not image_path:
            return ToolResult("tool.analyze_image", False, "", error="Missing 'image_path' parameter")

        if self.sense_engine is None:
            return ToolResult(
                "tool.analyze_image", False, "",
                error="Sense engine not available",
            )

        try:
            description = self.sense_engine.see(image_path)
            embedding = self.sense_engine.encode_image(image_path)
            return ToolResult(
                "tool.analyze_image", True,
                output=description,
                metadata={"has_embedding": embedding is not None},
            )
        except Exception as e:
            return ToolResult("tool.analyze_image", False, "", error=str(e))
