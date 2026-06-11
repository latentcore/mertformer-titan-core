from __future__ import annotations
"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI SELF AUDITOR
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

TR: Çıktı denetimi — hallüsinasyon tespiti, tutarlılık kontrolü, güvenlik doğrulaması.
EN: Output audit — hallucination detection, consistency check, safety verification.
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class ConsistencyScore:
    """Consistency score."""
    score: float  # 0.0 - 1.0
    overlapping_facts: int = 0
    contradictions: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class GroundingScore:
    """Grounding score."""
    score: float
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class SafetyScore:
    """Safety score."""
    score: float
    violations: List[str] = field(default_factory=list)
    is_safe: bool = True


@dataclass
class UncertaintyScore:
    """Uncertainty score."""
    score: float  # 0.0 (certain) - 1.0 (very uncertain)
    hedging_count: int = 0
    speculation_detected: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Audit report."""
    overall_score: float  # 0.0 - 1.0
    consistency: ConsistencyScore = field(default_factory=lambda: ConsistencyScore(0.5))
    grounding: GroundingScore = field(default_factory=lambda: GroundingScore(0.5))
    safety: SafetyScore = field(default_factory=lambda: SafetyScore(1.0))
    uncertainty: UncertaintyScore = field(default_factory=lambda: UncertaintyScore(0.5))
    should_retry: bool = False
    revision_hints: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# SELF AUDITOR
# -----------------------------------------------------------------------------

# Uncertainty markers
_HEDGING_WORDS_TR = frozenset({
    "belki", "muhtemelen", "olabilir", "sanırım", "tahminimce",
    "bence", "galiba", "herhalde", "yanlış olmam",
    "emin değilim", "kesin değil", "olası", "ihtimal",
})
_HEDGING_WORDS_EN = frozenset({
    "maybe", "perhaps", "possibly", "might", "could be",
    "i think", "i believe", "probably", "presumably",
    "not sure", "uncertain", "speculate", "guess",
    "it seems", "apparently", "arguably",
})

# Safety threat words
_SAFETY_THREATS = frozenset({
    "hack", "exploit", "bypass", "inject", "attack",
    "phishing", "malware", "ransomware", "keylogger",
    "exfiltrate", "backdoor", "rootkit", "unauthorized",
    "illegal", "weapon", "bomb", "drug", "abuse",
})

# Contradiction pairs
_CONTRADICTION_PAIRS = [
    ("yes", "no"), ("true", "false"), ("always", "never"),
    ("all", "none"), ("increase", "decrease"), ("better", "worse"),
    ("possible", "impossible"), ("correct", "incorrect"),
    ("safe", "dangerous"), ("success", "failure"),
    ("evet", "hayır"), ("doğru", "yanlış"),
    ("her zaman", "asla"), ("hepsi", "hiçbiri"),
    ("artış", "azalış"), ("güvenli", "tehlikeli"),
    ("başarı", "başarısızlık"), ("mümkün", "imkansız"),
]


class SelfAuditor:
    """
    Module that audits its own outputs.

    - Consistency: agreement between response and context
    - Grounding: verifiability of claims
    - Safety: harmful content detection
    - Uncertainty: hedging/speculation detection
    """

    def __init__(self, alignment_contracts: Optional[object] = None) -> None:
        self.alignment_contracts = alignment_contracts

    def audit(
        self,
        task: str,
        response: str,
        context: str = "",
        facts: Optional[List[str]] = None,
    ) -> AuditReport:
        """Generate full audit report."""
        consistency = self.check_consistency(response, context)
        grounding = self.check_grounding(response, facts or [])
        safety = self.check_safety(response)
        uncertainty = self.detect_uncertainty(response)

        # Calculate overall score
        overall = (
            0.30 * consistency.score
            + 0.25 * grounding.score
            + 0.25 * safety.score
            + 0.20 * (1.0 - uncertainty.score)  # low uncertainty = high score
        )

        # Retry decision
        should_retry = (
            overall < 0.4
            or consistency.contradictions > 2
            or not safety.is_safe
            or (grounding.ungrounded_claims > 3 and grounding.score < 0.3)
        )

        # Revision hints
        revision_hints: List[str] = []
        if consistency.score < 0.4:
            revision_hints.append("Yanıtın bağlamla tutarsız — kaynağa göre yeniden formüle et")
        if grounding.score < 0.3:
            revision_hints.append("Temelsiz iddialar var — kanıta dayalı ifadeler kullan")
        if not safety.is_safe:
            revision_hints.append("Güvenlik ihlali tespit edildi — zararlı içeriği kaldır")
        if uncertainty.score > 0.7:
            revision_hints.append("Çok fazla belirsizlik — daha kesin ifadeler kullan")

        return AuditReport(
            overall_score=max(0.0, min(1.0, overall)),
            consistency=consistency,
            grounding=grounding,
            safety=safety,
            uncertainty=uncertainty,
            should_retry=should_retry,
            revision_hints=revision_hints,
        )

    def check_consistency(self, response: str, context: str) -> ConsistencyScore:
        """Checks consistency between response and context."""
        if not context:
            return ConsistencyScore(score=0.6, notes=["No context provided for consistency check"])

        resp_lower = response.lower()
        ctx_lower = context.lower()

        # Word overlap
        resp_words = set(resp_lower.split())
        ctx_words = set(ctx_lower.split())
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "ve", "ile", "bir", "bu", "o", "da", "de", "için", "gibi"}
        resp_words -= stop_words
        ctx_words -= stop_words

        if not resp_words or not ctx_words:
            return ConsistencyScore(score=0.5, notes=["Insufficient text for analysis"])

        overlap = len(resp_words & ctx_words)
        total = max(1, len(resp_words))
        overlap_ratio = overlap / total

        # Contradiction detection
        contradictions = 0
        contradiction_notes: List[str] = []
        for pos, neg in _CONTRADICTION_PAIRS:
            if pos in resp_lower and neg in ctx_lower:
                contradictions += 1
                contradiction_notes.append(f"Response says '{pos}', context says '{neg}'")
            elif neg in resp_lower and pos in ctx_lower:
                contradictions += 1
                contradiction_notes.append(f"Response says '{neg}', context says '{pos}'")

        score = overlap_ratio - (contradictions * 0.15)
        score = max(0.0, min(1.0, score))

        notes: List[str] = []
        if overlap_ratio > 0.3:
            notes.append(f"Good overlap: {overlap_ratio:.0%}")
        else:
            notes.append(f"Low overlap: {overlap_ratio:.0%}")
        notes.extend(contradiction_notes)

        return ConsistencyScore(
            score=score,
            overlapping_facts=overlap,
            contradictions=contradictions,
            notes=notes,
        )

    def check_grounding(self, response: str, facts: List[str]) -> GroundingScore:
        """Checks if claims are evidence-based."""
        if not facts:
            return GroundingScore(score=0.5, notes=["No facts provided for grounding"])

        # Extract sentences from response
        sentences = re.split(r'[.!?]\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return GroundingScore(score=0.5, notes=["No substantial claims found"])

        grounded = 0
        ungrounded = 0

        for sentence in sentences:
            sent_lower = sentence.lower()
            sent_words = set(sent_lower.split())
            sent_words -= {"the", "a", "an", "is", "are", "ve", "bir", "bu", "o"}

            # Overlap with any fact?
            max_overlap = 0.0
            for fact in facts:
                fact_words = set(fact.lower().split())
                fact_words -= {"the", "a", "an", "is", "are", "ve", "bir", "bu", "o"}
                if not sent_words or not fact_words:
                    continue
                overlap = len(sent_words & fact_words) / max(1, len(sent_words))
                max_overlap = max(max_overlap, overlap)

            if max_overlap > 0.2:
                grounded += 1
            else:
                ungrounded += 1

        total = grounded + ungrounded
        score = grounded / max(1, total)

        notes: List[str] = []
        if score > 0.6:
            notes.append(f"Well-grounded: {grounded}/{total} claims supported")
        elif score > 0.3:
            notes.append(f"Partially grounded: {grounded}/{total} claims supported")
        else:
            notes.append(f"Poorly grounded: only {grounded}/{total} claims supported")

        return GroundingScore(
            score=score,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            notes=notes,
        )

    def check_safety(self, response: str) -> SafetyScore:
        """Harmful content check."""
        resp_lower = response.lower()
        violations: List[str] = []

        for threat in _SAFETY_THREATS:
            if threat in resp_lower:
                violations.append(f"Safety threat detected: '{threat}'")

        # Also check AlignmentContracts if available
        if self.alignment_contracts is not None:
            try:
                align_violations = self.alignment_contracts.check_prompt(response)
                for v in align_violations:
                    violations.append(f"Alignment violation [{v.rule_id}]: {v.message}")
            except Exception:
                pass

        is_safe = len(violations) == 0
        score = 1.0 if is_safe else max(0.0, 1.0 - len(violations) * 0.2)

        return SafetyScore(score=score, violations=violations, is_safe=is_safe)

    def detect_uncertainty(self, response: str) -> UncertaintyScore:
        """Uncertainty and hedging detection."""
        resp_lower = response.lower()
        hedging_count = 0
        notes: List[str] = []

        for word in _HEDGING_WORDS_TR:
            count = resp_lower.count(word)
            if count > 0:
                hedging_count += count
                notes.append(f"Hedging (TR): '{word}' × {count}")

        for word in _HEDGING_WORDS_EN:
            count = resp_lower.count(word)
            if count > 0:
                hedging_count += count
                notes.append(f"Hedging (EN): '{word}' × {count}")

        # Question mark in response indicates uncertainty
        question_marks = response.count("?")
        speculation = question_marks > 2 or hedging_count > 3

        # Uncertainty score
        word_count = max(1, len(response.split()))
        hedging_density = hedging_count / word_count
        score = min(1.0, hedging_density * 10.0 + question_marks * 0.05)

        return UncertaintyScore(
            score=score,
            hedging_count=hedging_count,
            speculation_detected=speculation,
            notes=notes,
        )
