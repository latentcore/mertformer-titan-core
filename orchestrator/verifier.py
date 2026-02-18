"""Verification and uncertainty scoring for swarm outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VerificationResult:
    pass_check: bool
    confidence: float
    uncertainty: float
    consistency: float
    notes: tuple[str, ...]


class SwarmVerifier:
    def __init__(self, min_confidence: float = 0.45, min_consistency: float = 0.35) -> None:
        self.min_confidence = float(min_confidence)
        self.min_consistency = float(min_consistency)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        denom = len(a | b)
        return 0.0 if denom == 0 else len(a & b) / float(denom)

    def verify(self, task: str, outputs: Sequence[str]) -> VerificationResult:
        notes: list[str] = []
        clean = [o.strip() for o in outputs if o and o.strip()]
        if not clean:
            notes.append("no_output")
            return VerificationResult(False, 0.0, 1.0, 0.0, tuple(notes))

        unique_ratio = len(set(clean)) / float(len(clean))
        if unique_ratio < 0.5:
            notes.append("high_duplication")

        task_tokens = set((task or "").lower().split())
        matched = 0
        for item in clean:
            tokens = set(item.lower().split())
            matched += 1 if task_tokens & tokens else 0
        relevance = matched / float(len(clean))

        # Pairwise semantic consistency proxy via token-overlap.
        overlaps = []
        tokenized = [set(item.lower().split()) for item in clean]
        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                overlaps.append(self._jaccard(tokenized[i], tokenized[j]))
        consistency = sum(overlaps) / float(len(overlaps)) if overlaps else 1.0

        confidence = max(0.0, min(1.0, 0.5 * relevance + 0.3 * unique_ratio + 0.2 * consistency))
        uncertainty = 1.0 - confidence
        pass_check = confidence >= self.min_confidence and consistency >= self.min_consistency

        if pass_check:
            notes.append("verification_pass")
        else:
            notes.append("verification_low_confidence")
        if consistency < self.min_consistency:
            notes.append("consistency_low")

        return VerificationResult(pass_check, confidence, uncertainty, consistency, tuple(notes))
