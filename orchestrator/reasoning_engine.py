from __future__ import annotations
"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI REASONING ENGINE
-------------------------------------------------------------------------------
Copyright 2026 Mert Yunlu
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

TR: Zincir Düşünce (CoT) ve Düşünce Ağacı (ToT) tabanlı çok stratejili akıl yürütme motoru.
EN: Chain-of-Thought (CoT) and Tree-of-Thought (ToT) based multi-strategy reasoning engine.

NOTE (scope): inert / out-of-scope; 45K eğitim yolunda kapalı (feature-flag).
Bu orchestrator modülü çekirdek eğitim hattının parçası değildir; deneysel/
spekülatif akıl yürütme katmanıdır.
"""

# NOTE: legacy/fossil version tag, hand-maintained and not the single source of
# truth; prefer the central package version. Kept only for backward compatibility.
from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import time
import math
import hashlib
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class ThoughtStep:
    """A single reasoning step."""
    step_number: int
    thought: str
    confidence: float  # 0.0 - 1.0; heuristik PROXY (kalibre olasılık DEĞİL)
    action_proposal: Optional[str] = None
    tool_suggestion: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    node_id: int
    depth: int
    thought: str
    confidence: float
    children: List["ThoughtNode"] = field(default_factory=list)
    action_proposal: Optional[str] = None
    pruned: bool = False


@dataclass
class ThoughtTree:
    """Thought tree structure."""
    root: ThoughtNode
    best_path: List[ThoughtNode] = field(default_factory=list)
    exploration_count: int = 0
    total_nodes: int = 0


@dataclass
class ReasoningResult:
    """Reasoning result."""
    strategy: str  # "cot", "tot", "direct"
    conclusion: str
    confidence: float
    thoughts: List[ThoughtStep] = field(default_factory=list)
    tree: Optional[ThoughtTree] = None
    action_plan: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    reasoning_time_ms: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Candidate hypothesis."""
    candidate_id: str
    strategy: str
    conclusion: str
    confidence: float
    rationale_hash: str
    action_plan: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class HypothesisSet:
    """Multi-hypothesis bundle."""
    task: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    selected_index: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def best(self) -> Hypothesis:
        if not self.hypotheses:
            return Hypothesis(
                candidate_id="h0",
                strategy="direct",
                conclusion="",
                confidence=0.0,
                rationale_hash="",
            )
        idx = min(max(0, int(self.selected_index)), len(self.hypotheses) - 1)
        return self.hypotheses[idx]

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "selected_index": self.selected_index,
            "hypotheses": [
                {
                    "candidate_id": h.candidate_id,
                    "strategy": h.strategy,
                    "conclusion": h.conclusion,
                    "confidence": h.confidence,
                    "rationale_hash": h.rationale_hash,
                    "action_plan": h.action_plan,
                    "tool_calls": h.tool_calls,
                    "metadata": h.metadata,
                }
                for h in self.hypotheses
            ],
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# STRATEGY SELECTOR
# -----------------------------------------------------------------------------

# Task complexity keywords
_COMPLEX_KEYWORDS = frozenset({
    "analiz", "analyze", "karşılaştır", "compare", "tasarla", "design",
    "optimize", "debug", "neden", "why", "nasıl", "how", "plan",
    "strateji", "strategy", "mimari", "architecture", "trade-off",
    "refactor", "implementasyon", "implementation", "değerlendir", "evaluate",
    "araştır", "research", "çok adımlı", "multi-step",
})

_SIMPLE_KEYWORDS = frozenset({
    "ne", "what", "kim", "who", "ne zaman", "when", "nerede", "where",
    "tanımla", "define", "listele", "list", "göster", "show",
})


def _estimate_complexity(task: str) -> float:
    """
    Estimates task complexity between 0-1.
    """
    task_lower = task.lower()
    words = set(task_lower.split())

    complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in task_lower)
    simple_hits = sum(1 for kw in _SIMPLE_KEYWORDS if kw in task_lower)

    # Longer tasks are usually more complex
    length_factor = min(1.0, len(words) / 50.0)

    # Number of question marks increases complexity
    question_factor = min(1.0, task.count("?") * 0.2)

    score = (
        0.3 * min(1.0, complex_hits * 0.25)
        + 0.2 * length_factor
        + 0.1 * question_factor
        - 0.2 * min(1.0, simple_hits * 0.3)
    )
    # Sabit +0.3 bias: skoru orta-karmaşıklığa çeker; bu kasıtlı/heuristik bir
    # ayardır ve select_strategy eşiklerini (0.4 -> cot, 0.7 -> tot) doğrudan
    # etkiler. Eşikler değişirse bu bias yeniden gözden geçirilmelidir.
    return max(0.0, min(1.0, score + 0.3))  # bias toward medium complexity


def select_strategy(task: str) -> str:
    """
    Selects the most appropriate strategy for the task.
    """
    complexity = _estimate_complexity(task)
    if complexity >= 0.7:
        return "tot"
    elif complexity >= 0.4:
        return "cot"
    return "direct"


# -----------------------------------------------------------------------------
# REASONING ENGINE
# -----------------------------------------------------------------------------

class ReasoningEngine:
    """
    Multi-strategy reasoning engine.

    - direct: direct answer for simple tasks
    - cot: chain of thought - step-by-step reasoning
    - tot: tree of thought - branching exploration, with pruning
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[str], str]] = None,
        max_cot_steps: int = 7,
        max_tot_branches: int = 3,
        max_tot_depth: int = 4,
        prune_threshold: float = 0.3,
    ) -> None:
        self.generate_fn = generate_fn
        self.max_cot_steps = max(1, int(max_cot_steps))
        self.max_tot_branches = max(1, int(max_tot_branches))
        self.max_tot_depth = max(1, int(max_tot_depth))
        self.prune_threshold = float(prune_threshold)
        self._node_counter = 0

    def _generate(self, prompt: str) -> str:
        """LLM call (fallback: echo)."""
        if self.generate_fn is not None:
            try:
                return self.generate_fn(prompt)
            except Exception as e:
                # Hata sessizce yutulmasın: en azından uyarı olarak loglanır.
                # Davranış korunur (aynı fallback string'i döner) ama hata artık
                # teşhis edilebilir.
                logger.warning("generation failed in _generate: %s", e, exc_info=True)
                return f"[reasoning-fallback] generation error: {e}"
        return f"[no-model] {prompt[:200]}"

    # -----------------------------------------------------------------
    # MAIN ENTRY POINT
    # -----------------------------------------------------------------
    def reason(
        self,
        task: str,
        context: str = "",
        strategy: str = "auto",
    ) -> ReasoningResult:
        """
        Solves the task with reasoning.

        Args:
            task: Task to solve
            context: Additional context (memory, documents, etc.)
            strategy: "auto", "direct", "cot", "tot"
        """
        t0 = time.monotonic()

        if strategy == "auto":
            strategy = select_strategy(task)

        if strategy == "tot":
            result = self._tree_of_thought(task, context)
        elif strategy == "cot":
            result = self._chain_of_thought(task, context)
        else:
            result = self._direct(task, context)

        result.reasoning_time_ms = (time.monotonic() - t0) * 1000.0
        return result

    def generate_hypotheses(
        self,
        task: str,
        context: str = "",
        strategy: str = "auto",
        max_candidates: int = 3,
    ) -> HypothesisSet:
        """
        Produces multiple candidates for the same task.
        """
        candidate_budget = max(1, int(max_candidates))
        primary = select_strategy(task) if strategy == "auto" else str(strategy).lower().strip()

        ordered = [primary]
        for alt in ("tot", "cot", "direct"):
            if alt not in ordered:
                ordered.append(alt)

        hypotheses: List[Hypothesis] = []
        for idx, candidate_strategy in enumerate(ordered[:candidate_budget], start=1):
            result = self.reason(task, context=context, strategy=candidate_strategy)
            digest = hashlib.sha256(result.conclusion.strip().encode("utf-8")).hexdigest()[:16]
            hypotheses.append(
                Hypothesis(
                    candidate_id=f"h{idx}",
                    strategy=result.strategy,
                    conclusion=result.conclusion,
                    confidence=result.confidence,
                    rationale_hash=digest,
                    action_plan=list(result.action_plan),
                    tool_calls=list(result.tool_calls),
                    metadata={
                        "reasoning_time_ms": result.reasoning_time_ms,
                        "thought_count": len(result.thoughts),
                    },
                )
            )

        if not hypotheses:
            hypotheses.append(
                Hypothesis(
                    candidate_id="h1",
                    strategy="direct",
                    conclusion="",
                    confidence=0.0,
                    rationale_hash="",
                )
            )

        best_idx = max(range(len(hypotheses)), key=lambda i: hypotheses[i].confidence)
        return HypothesisSet(
            task=task,
            hypotheses=hypotheses,
            selected_index=best_idx,
            metadata={
                "candidate_count": len(hypotheses),
                "primary_strategy": primary,
            },
        )

    # -----------------------------------------------------------------
    # DIRECT RESPONSE
    # -----------------------------------------------------------------
    def _direct(self, task: str, context: str) -> ReasoningResult:
        prompt = self._build_prompt(task, context, mode="direct")
        response = self._generate(prompt)

        return ReasoningResult(
            strategy="direct",
            conclusion=response,
            confidence=0.6,
            thoughts=[
                ThoughtStep(
                    step_number=1,
                    thought="Direct response — task assessed as simple",
                    confidence=0.6,
                )
            ],
        )

    # -----------------------------------------------------------------
    # CHAIN OF THOUGHT
    # -----------------------------------------------------------------
    def _chain_of_thought(self, task: str, context: str) -> ReasoningResult:
        """
        Reach conclusion by thinking step by step.
        """
        steps: List[ThoughtStep] = []
        action_plan: List[str] = []
        tool_calls: List[str] = []

        # Step 1: Understand the task
        step1_prompt = self._build_prompt(
            task, context,
            mode="cot_decompose",
            instruction=(
                "Bu görevi alt görevlere böl. Her alt görevi numaralı liste olarak yaz. "
                "Hangi araçlara ihtiyaç olabileceğini belirt."
            ),
        )
        decomposition = self._generate(step1_prompt)
        steps.append(ThoughtStep(
            step_number=1,
            thought=f"Task Decomposition: {decomposition}",
            confidence=0.7,
        ))

        # Steps 2-N: Think for each subtask
        accumulated_reasoning = decomposition
        for i in range(2, self.max_cot_steps + 1):
            step_prompt = self._build_prompt(
                task, context,
                mode="cot_step",
                instruction=(
                    f"Adım {i}: Önceki düşüncelerine dayanarak bir sonraki mantıksal adımı düşün.\n"
                    f"Önceki düşünceler:\n{accumulated_reasoning}\n\n"
                    "Eğer bir araç kullanman gerekiyorsa [TOOL: araç_adı] formatında belirt.\n"
                    "Eğer bir aksiyon öneriyorsan [ACTION: aksiyon] formatında belirt.\n"
                    "Eğer sonuca ulaştıysan [CONCLUSION: sonuç] formatında belirt."
                ),
            )
            step_response = self._generate(step_prompt)
            accumulated_reasoning += f"\nAdım {i}: {step_response}"

            # Tool and action extraction
            confidence = self._estimate_step_confidence(step_response, i)
            tool = self._extract_tag(step_response, "TOOL")
            action = self._extract_tag(step_response, "ACTION")
            conclusion = self._extract_tag(step_response, "CONCLUSION")

            step = ThoughtStep(
                step_number=i,
                thought=step_response,
                confidence=confidence,
                tool_suggestion=tool,
                action_proposal=action or conclusion,
            )
            steps.append(step)

            if tool:
                tool_calls.append(tool)
            if action:
                action_plan.append(action)

            # Stop if conclusion reached
            if conclusion:
                break

        # Final synthesis
        final_prompt = self._build_prompt(
            task, context,
            mode="cot_synthesize",
            instruction=(
                "Tüm düşünce adımlarını sentezle ve kesin bir sonuç ver.\n"
                f"Düşünce zinciri:\n{accumulated_reasoning}"
            ),
        )
        final_conclusion = self._generate(final_prompt)

        avg_confidence = (
            sum(s.confidence for s in steps) / len(steps) if steps else 0.5
        )

        return ReasoningResult(
            strategy="cot",
            conclusion=final_conclusion,
            confidence=min(0.95, avg_confidence + 0.1),
            thoughts=steps,
            action_plan=action_plan,
            tool_calls=tool_calls,
        )

    # -----------------------------------------------------------------
    # TREE OF THOUGHT
    # -----------------------------------------------------------------
    def _tree_of_thought(self, task: str, context: str) -> ReasoningResult:
        """
        Branching thought tree - select best path.
        """
        self._node_counter = 0

        # Root node
        root_prompt = self._build_prompt(
            task, context,
            mode="tot_root",
            instruction=(
                "Bu karmaşık görev için 3 farklı yaklaşım öner. "
                "Her yaklaşımı `---` ile ayır. Her biri için güven yüzdesi ver."
            ),
        )
        root_response = self._generate(root_prompt)

        root = ThoughtNode(
            node_id=self._next_node_id(),
            depth=0,
            thought=f"Root analysis: {task}",
            confidence=0.5,
        )

        # Create branches
        branches = self._split_branches(root_response)
        for branch_text in branches[: self.max_tot_branches]:
            branch_conf = self._extract_confidence(branch_text)
            child = ThoughtNode(
                node_id=self._next_node_id(),
                depth=1,
                thought=branch_text.strip(),
                confidence=branch_conf,
            )
            root.children.append(child)

            # Pruning
            if branch_conf < self.prune_threshold:
                child.pruned = True
                continue

            # Deepen
            self._expand_node(child, task, context, depth=1)

        # Find best path
        best_path = self._find_best_path(root)
        total_nodes = self._count_nodes(root)

        # Synthesize conclusion from best path
        path_summary = " → ".join(
            n.thought[:100] for n in best_path
        )
        synth_prompt = self._build_prompt(
            task, context,
            mode="tot_synthesize",
            instruction=(
                f"En iyi düşünce yolu: {path_summary}\n"
                "Bu yolu takip ederek kesin bir sonuç ve aksiyon planı ver."
            ),
        )
        conclusion = self._generate(synth_prompt)

        # Extract actions
        action_plan: List[str] = []
        tool_calls: List[str] = []
        for node in best_path:
            tool = self._extract_tag(node.thought, "TOOL")
            action = self._extract_tag(node.thought, "ACTION")
            if tool:
                tool_calls.append(tool)
            if action:
                action_plan.append(action)

        tree = ThoughtTree(
            root=root,
            best_path=best_path,
            exploration_count=total_nodes,
            total_nodes=total_nodes,
        )

        path_conf = (
            sum(n.confidence for n in best_path) / len(best_path)
            if best_path else 0.5
        )

        return ReasoningResult(
            strategy="tot",
            conclusion=conclusion,
            confidence=min(0.95, path_conf),
            thoughts=[
                ThoughtStep(
                    step_number=i + 1,
                    thought=n.thought,
                    confidence=n.confidence,
                    action_proposal=n.action_proposal,
                )
                for i, n in enumerate(best_path)
            ],
            tree=tree,
            action_plan=action_plan,
            tool_calls=tool_calls,
            metadata={"total_nodes": total_nodes, "pruned": self._count_pruned(root)},
        )

    def _expand_node(
        self, node: ThoughtNode, task: str, context: str, depth: int
    ) -> None:
        """Expand node."""
        if depth >= self.max_tot_depth:
            return

        expand_prompt = self._build_prompt(
            task, context,
            mode="tot_expand",
            instruction=(
                f"Mevcut düşünce: {node.thought}\n"
                "Bu düşünceyi 2 farklı şekilde derinleştir. `---` ile ayır."
            ),
        )
        response = self._generate(expand_prompt)
        branches = self._split_branches(response)

        for branch_text in branches[: self.max_tot_branches]:
            conf = self._extract_confidence(branch_text)
            child = ThoughtNode(
                node_id=self._next_node_id(),
                depth=depth + 1,
                thought=branch_text.strip(),
                confidence=conf,
            )
            node.children.append(child)

            if conf < self.prune_threshold:
                child.pruned = True
                continue

            if depth + 1 < self.max_tot_depth:
                self._expand_node(child, task, context, depth + 1)

    def _find_best_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Find highest confidence path (greedy)."""
        path = [root]
        current = root
        while current.children:
            active = [c for c in current.children if not c.pruned]
            if not active:
                break
            best = max(active, key=lambda n: n.confidence)
            path.append(best)
            current = best
        return path

    # -----------------------------------------------------------------
    # HELPER METHODS
    # -----------------------------------------------------------------
    def _build_prompt(
        self,
        task: str,
        context: str,
        mode: str,
        instruction: str = "",
    ) -> str:
        parts = [
            "[SYSTEM] Sen MertFormer Titan AGI'nin akıl yürütme motorusun.",
            f"Mod: {mode}",
        ]
        if context:
            parts.append(f"[CONTEXT]\n{context}\n[/CONTEXT]")
        parts.append(f"[TASK]\n{task}\n[/TASK]")
        if instruction:
            parts.append(f"[INSTRUCTION]\n{instruction}\n[/INSTRUCTION]")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        """Extracts value from [TAG: value] format."""
        marker = f"[{tag}:"
        idx = text.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        end = text.find("]", start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def _estimate_step_confidence(text: str, step_number: int) -> float:
        """Estimates step confidence.

        UYARI: Bu değer kelime-sayımı + sabit bias'a dayalı bir HEURISTIK
        PROXY'dir; kalibre edilmiş bir olasılık DEĞİLDİR. 0.0-1.0 aralığında
        sunulması bir kalibrasyon iddiası içermez.
        """
        confidence = 0.6

        # Certainty markers increase confidence
        high_conf = ["kesinlikle", "certainly", "clearly", "definitely", "açıkça"]
        low_conf = ["belki", "maybe", "perhaps", "possibly", "muhtemelen", "might"]

        text_lower = text.lower()
        for word in high_conf:
            if word in text_lower:
                confidence += 0.1
        for word in low_conf:
            if word in text_lower:
                confidence -= 0.1

        # Later steps are usually more confident
        confidence += min(0.15, step_number * 0.03)

        return max(0.1, min(0.95, confidence))

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extracts confidence percentage from text."""
        import re
        match = re.search(r"(\d{1,3})%", text)
        if match:
            return min(0.95, max(0.1, int(match.group(1)) / 100.0))
        return 0.5

    @staticmethod
    def _split_branches(text: str) -> List[str]:
        """Splits branches separated by ---."""
        parts = [p.strip() for p in text.split("---") if p.strip()]
        if not parts:
            return [text.strip()] if text.strip() else []
        return parts

    def _next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    @staticmethod
    def _count_nodes(node: ThoughtNode) -> int:
        count = 1
        for child in node.children:
            count += ReasoningEngine._count_nodes(child)
        return count

    @staticmethod
    def _count_pruned(node: ThoughtNode) -> int:
        count = 1 if node.pruned else 0
        for child in node.children:
            count += ReasoningEngine._count_pruned(child)
        return count
