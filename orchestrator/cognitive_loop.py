from __future__ import annotations
"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI COGNITIVE LOOP
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

TR: AGI Bilişsel Döngü — Algıla → Düşün → Eylem → Yansıt
EN: AGI Cognitive Loop — Perceive → Think → Act → Reflect

Bu modül MertFormer Titan'ın AGI çekirdeğidir. Tüm bilişsel modülleri
(akıl yürütme, araç kullanımı, hafıza, öz denetim, deneyim) tek bir
iteratif döngüde birleştirir.

This module is MertFormer Titan's AGI core. It unifies all cognitive modules
(reasoning, tool use, memory, self-audit, experience) into a single
iterative loop.

SCOPE NOTE: This orchestrator is INERT / out-of-scope for the canonical 45K
training path; it is disabled (feature-flagged off) on that path and is not
exercised during pre-training. Treat as an experimental cognitive shell.
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from .reasoning_engine import ReasoningEngine, ReasoningResult
from .tool_executor import ToolExecutor, ToolResult
from .self_audit import SelfAuditor, AuditReport
from .experience_store import ExperienceStore, Episode


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class PerceptionState:
    """Perception state - input to the loop."""
    task: str
    context: str = ""
    memory_context: str = ""
    world_state: str = ""
    past_experiences: List[Episode] = field(default_factory=list)
    iteration: int = 0
    previous_reflection: str = ""


@dataclass
class ThoughtPlan:
    """Thought plan - reasoning engine output."""
    strategy: str
    conclusion: str
    confidence: float
    action_plan: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    reasoning_result: Optional[ReasoningResult] = None


@dataclass
class ActionResult:
    """Action result."""
    response: str
    tool_results: List[ToolResult] = field(default_factory=list)
    success: bool = True
    action_type: str = "generate"  # "generate", "tool", "delegate"


@dataclass
class ReflectionReport:
    """Reflection report."""
    audit: AuditReport
    should_retry: bool = False
    revision_hints: List[str] = field(default_factory=list)
    learning: str = ""
    outcome_score: float = 0.0


@dataclass
class CognitiveIteration:
    """Record of a single cognitive iteration."""
    iteration: int
    perception: PerceptionState
    thought: ThoughtPlan
    action: ActionResult
    reflection: ReflectionReport
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveResult:
    """Final result of the cognitive loop."""
    task: str
    final_response: str
    total_iterations: int
    strategy_used: str
    confidence: float
    outcome_score: float
    iterations: List[CognitiveIteration] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_time_ms: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"🧠 AGI Cognitive Result",
            f"   Task: {self.task[:80]}...",
            f"   Strategy: {self.strategy_used}",
            f"   Iterations: {self.total_iterations}",
            f"   Confidence: {self.confidence:.0%}",
            f"   Score: {self.outcome_score:.0%}",
            f"   Tools: {', '.join(self.tools_used) if self.tools_used else 'none'}",
            f"   Time: {self.total_time_ms:.0f}ms",
            f"",
            f"   Response: {self.final_response[:300]}...",
        ]
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# COGNITIVE LOOP
# -----------------------------------------------------------------------------

class CognitiveLoop:
    """
    AGI Cognitive Loop - Perceive -> Think -> Act -> Reflect.

    Each iteration:
    1. PERCEIVE: Gather context (memory, senses, world state, experiences)
    2. THINK: Reason about the task (CoT/ToT)
    3. ACT: Execute chosen action (generate, tool use, delegate)
    4. REFLECT: Audit output, learn from outcome
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[str], str]] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        tool_executor: Optional[ToolExecutor] = None,
        self_auditor: Optional[SelfAuditor] = None,
        experience_store: Optional[ExperienceStore] = None,
        memory: Optional[Any] = None,
        world_model: Optional[Any] = None,
        max_iterations: int = 5,
        min_confidence: float = 0.5,
    ) -> None:
        self.generate_fn = generate_fn
        self.reasoning = reasoning_engine or ReasoningEngine(generate_fn=generate_fn)
        self.tools = tool_executor or ToolExecutor()
        self.auditor = self_auditor or SelfAuditor()
        self.experience = experience_store or ExperienceStore()
        self.memory = memory
        self.world_model = world_model
        self.max_iterations = max(1, int(max_iterations))
        self.min_confidence = float(min_confidence)

    def run(self, task: str, max_iterations: Optional[int] = None) -> CognitiveResult:
        """
        Run the cognitive loop.
        """
        t0 = time.monotonic()
        max_iter = max_iterations or self.max_iterations
        iterations: List[CognitiveIteration] = []
        all_tools: List[str] = []
        previous_reflection = ""

        # Get strategy suggestion from experience store
        suggested_strategy = self.experience.best_strategy_for(task)

        print(f"\n🧠 AGI COGNITIVE LOOP BAŞLATILIYOR...")
        print(f"   Görev: {task[:80]}...")
        print(f"   Önerilen strateji: {suggested_strategy}")
        print(f"   Max iterasyon: {max_iter}\n")

        for i in range(max_iter):
            print(f"   ── İterasyon {i + 1}/{max_iter} ──")

            # ============================================================
            # 1. PERCEIVE
            # ============================================================
            perception = self._perceive(task, i, previous_reflection)
            print(f"   👁️ Perception: context={len(perception.context)} chars, "
                  f"memories={len(perception.memory_context)} chars, "
                  f"past_exp={len(perception.past_experiences)}")

            # ============================================================
            # 2. THINK
            # ============================================================
            strategy = suggested_strategy if i == 0 else "cot"
            thought = self._think(perception, strategy)
            print(f"   🧠 Think: strategy={thought.strategy}, "
                  f"confidence={thought.confidence:.0%}, "
                  f"tools={thought.tool_calls}, "
                  f"actions={len(thought.action_plan)}")

            # ============================================================
            # 3. ACT
            # ============================================================
            action = self._act(thought, perception)
            all_tools.extend(
                tr.tool_id for tr in action.tool_results if tr.success
            )
            print(f"   ⚡ Act: type={action.action_type}, "
                  f"success={action.success}, "
                  f"response_len={len(action.response)}")

            # ============================================================
            # 4. REFLECT
            # ============================================================
            reflection = self._reflect(task, thought, action, perception)
            previous_reflection = reflection.learning
            print(f"   🪞 Reflect: score={reflection.outcome_score:.0%}, "
                  f"retry={reflection.should_retry}, "
                  f"hints={len(reflection.revision_hints)}")

            # Record iteration
            iteration = CognitiveIteration(
                iteration=i + 1,
                perception=perception,
                thought=thought,
                action=action,
                reflection=reflection,
            )
            iterations.append(iteration)

            # Good enough -> stop
            if not reflection.should_retry and thought.confidence >= self.min_confidence:
                print(f"   ✅ Sonuca ulaşıldı (confidence={thought.confidence:.0%})\n")
                break

            # Last iteration -> forced stop
            if i == max_iter - 1:
                print(f"   ⚠️ Max iterasyon ulaşıldı\n")
                break

            print(f"   🔄 Yeniden denenecek: {', '.join(reflection.revision_hints[:2])}\n")

        # Final result
        final_iteration = iterations[-1] if iterations else None
        final_response = final_iteration.action.response if final_iteration else ""
        confidence = final_iteration.thought.confidence if final_iteration else 0.0
        outcome = final_iteration.reflection.outcome_score if final_iteration else 0.0
        strategy_used = iterations[0].thought.strategy if iterations else "direct"

        result = CognitiveResult(
            task=task,
            final_response=final_response,
            total_iterations=len(iterations),
            strategy_used=strategy_used,
            confidence=confidence,
            outcome_score=outcome,
            iterations=iterations,
            tools_used=list(set(all_tools)),
            total_time_ms=(time.monotonic() - t0) * 1000.0,
        )

        # Record experience
        self._record_experience(result)

        print(result.to_summary())
        return result

    # -----------------------------------------------------------------
    # 1. PERCEIVE
    # -----------------------------------------------------------------
    def _perceive(
        self,
        task: str,
        iteration: int,
        previous_reflection: str,
    ) -> PerceptionState:
        """
        Gather context - memory, world state, experiences.
        """
        # Memory context
        memory_context = ""
        if self.memory is not None:
            try:
                memory_context = self.memory.build_context_block(task)
            except Exception:
                logger.warning(
                    "memory.build_context_block failed; falling back to recall",
                    exc_info=True,
                )
                try:
                    memory_context = self.memory.recall(task, top_k=5)
                except Exception:
                    logger.warning(
                        "memory.recall fallback also failed; using empty context",
                        exc_info=True,
                    )
                    memory_context = ""

        # World state
        world_state = ""
        if self.world_model is not None:
            try:
                relevant_facts = self.world_model.recall_relevant(task)
                if relevant_facts:
                    world_state = "Known facts:\n" + "\n".join(
                        f"- {f}" for f in relevant_facts
                    )
            except Exception:
                logger.warning(
                    "world_model.recall_relevant failed; skipping world state",
                    exc_info=True,
                )

        # Experiences
        past = self.experience.recall_similar(task, top_k=3)

        # Build full context
        context_parts: List[str] = []
        if memory_context:
            context_parts.append(memory_context)
        if world_state:
            context_parts.append(world_state)
        if past:
            exp_lines = ["[PAST EXPERIENCES]"]
            for ep in past:
                exp_lines.append(
                    f"- Task: {ep.task[:80]} → Strategy: {ep.strategy_used}, "
                    f"Score: {ep.outcome_score:.0%}"
                )
            exp_lines.append("[/PAST EXPERIENCES]")
            context_parts.append("\n".join(exp_lines))
        if previous_reflection:
            context_parts.append(
                f"[PREVIOUS REFLECTION]\n{previous_reflection}\n[/PREVIOUS REFLECTION]"
            )

        return PerceptionState(
            task=task,
            context="\n\n".join(context_parts),
            memory_context=memory_context,
            world_state=world_state,
            past_experiences=past,
            iteration=iteration,
            previous_reflection=previous_reflection,
        )

    # -----------------------------------------------------------------
    # 2. THINK
    # -----------------------------------------------------------------
    def _think(
        self,
        perception: PerceptionState,
        strategy: str = "auto",
    ) -> ThoughtPlan:
        """
        Think with the reasoning engine.
        """
        result = self.reasoning.reason(
            task=perception.task,
            context=perception.context,
            strategy=strategy,
        )

        return ThoughtPlan(
            strategy=result.strategy,
            conclusion=result.conclusion,
            confidence=result.confidence,
            action_plan=result.action_plan,
            tool_calls=result.tool_calls,
            reasoning_result=result,
        )

    # -----------------------------------------------------------------
    # 3. ACT
    # -----------------------------------------------------------------
    def _act(
        self,
        plan: ThoughtPlan,
        perception: PerceptionState,
    ) -> ActionResult:
        """
        Execute the plan - call tools or generate response.
        """
        tool_results: List[ToolResult] = []
        tool_outputs: List[str] = []

        # Tool calls
        for tool_id in plan.tool_calls:
            # Try to extract tool parameters from task
            params = self._infer_tool_params(tool_id, perception.task)
            result = self.tools.execute(tool_id, params)
            tool_results.append(result)
            if result.success:
                tool_outputs.append(f"[{tool_id}] {result.output}")
            else:
                tool_outputs.append(f"[{tool_id}] ERROR: {result.error}")

        # Tool-enriched response
        if tool_outputs and self.generate_fn is not None:
            enrichment = "\n".join(tool_outputs)
            enriched_prompt = (
                f"Görev: {perception.task}\n\n"
                f"Düşünce: {plan.conclusion}\n\n"
                f"Araç Sonuçları:\n{enrichment}\n\n"
                "Bu bilgileri kullanarak kapsamlı bir yanıt oluştur."
            )
            try:
                response = self.generate_fn(enriched_prompt)
            except Exception:
                logger.warning(
                    "generate_fn failed; falling back to plan.conclusion",
                    exc_info=True,
                )
                response = plan.conclusion
        else:
            response = plan.conclusion

        action_type = "tool" if tool_results else "generate"
        if tool_results:
            action_success = all(tr.success for tr in tool_results) and bool(response.strip())
        else:
            action_success = bool(response.strip())

        return ActionResult(
            response=response,
            tool_results=tool_results,
            success=action_success,
            action_type=action_type,
        )

    def _infer_tool_params(self, tool_id: str, task: str) -> Dict[str, Any]:
        """Infer tool parameters from task."""
        params: Dict[str, Any] = {}

        if tool_id in ("tool.search_local_docs", "tool.web_search", "tool.recall"):
            params["query"] = task
        elif tool_id == "tool.calculate":
            # Try to find mathematical expression
            import re
            match = re.search(r'[\d\+\-\*\/\(\)\.\s]+', task)
            if match:
                params["expression"] = match.group().strip()
            else:
                params["expression"] = task
        elif tool_id == "tool.memorize":
            params["text"] = task
        elif tool_id == "tool.verify_consistency":
            params["text"] = task
        elif tool_id == "tool.analyze_image":
            # Try to find file path
            import re
            match = re.search(r'(/[\w./\-_]+\.\w+)', task)
            if match:
                params["image_path"] = match.group()

        return params

    # -----------------------------------------------------------------
    # 4. REFLECT
    # -----------------------------------------------------------------
    def _reflect(
        self,
        task: str,
        thought: ThoughtPlan,
        action: ActionResult,
        perception: PerceptionState,
    ) -> ReflectionReport:
        """
        Audit the result and reflect.
        """
        # Self audit
        facts = []
        if perception.world_state:
            facts = [
                line.lstrip("- ") for line in perception.world_state.split("\n")
                if line.strip().startswith("-")
            ]

        audit = self.auditor.audit(
            task=task,
            response=action.response,
            context=perception.context,
            facts=facts,
        )

        # Heuristik outcome skoru (ölçülmüş bir başarı metriği DEĞİL):
        # sabit ağırlıklı (0.4/0.4/0.1/0.1) confidence/audit/success/retry
        # kombinasyonu. 'score' adı yalnızca bu kestirimsel birleşimi ifade eder;
        # downstream (experience_store, run() durdurma koşulu, to_summary)
        # bunu kanıtlanmış başarı oranı gibi yorumlamamalıdır.
        outcome_score = (
            0.4 * thought.confidence
            + 0.4 * audit.overall_score
            + 0.1 * (1.0 if action.success else 0.0)
            + 0.1 * (1.0 if not audit.should_retry else 0.0)
        )

        # Learning notes
        learning_parts: List[str] = []
        if audit.revision_hints:
            learning_parts.append(
                f"Revision needed: {'; '.join(audit.revision_hints[:3])}"
            )
        if thought.strategy:
            learning_parts.append(f"Strategy '{thought.strategy}' used")
        if action.tool_results:
            successful_tools = [
                tr.tool_id for tr in action.tool_results if tr.success
            ]
            failed_tools = [
                tr.tool_id for tr in action.tool_results if not tr.success
            ]
            if successful_tools:
                learning_parts.append(f"Successful tools: {successful_tools}")
            if failed_tools:
                learning_parts.append(f"Failed tools: {failed_tools}")

        learning = "; ".join(learning_parts) if learning_parts else "No specific learnings"

        return ReflectionReport(
            audit=audit,
            should_retry=audit.should_retry,
            revision_hints=audit.revision_hints,
            learning=learning,
            outcome_score=max(0.0, min(1.0, outcome_score)),
        )

    # -----------------------------------------------------------------
    # EXPERIENCE RECORDING
    # -----------------------------------------------------------------
    def _record_experience(self, result: CognitiveResult) -> None:
        """Record result to experience store."""
        thoughts = []
        actions = []
        for iteration in result.iterations:
            thoughts.append(iteration.thought.conclusion[:200])
            if iteration.action.response:
                actions.append(iteration.action.action_type)

        episode = Episode(
            task=result.task,
            strategy_used=result.strategy_used,
            thoughts=thoughts,
            actions=actions,
            tools_used=result.tools_used,
            outcome_score=result.outcome_score,
            audit_score=(
                result.iterations[-1].reflection.audit.overall_score
                if result.iterations else 0.0
            ),
            reflection=(
                result.iterations[-1].reflection.learning
                if result.iterations else ""
            ),
            iterations=result.total_iterations,
        )

        self.experience.record_episode(episode)
