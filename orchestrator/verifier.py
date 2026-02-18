"""Verification and uncertainty scoring for swarm outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VerificationResult:
    pass_check: bool
    confidence: float
    uncertainty: float
    notes: tuple[str, ...]


class SwarmVerifier:
    def verify(self, task: str, outputs: Sequence[str]) -> VerificationResult:
        notes: list[str] = []
        clean = [o.strip() for o in outputs if o and o.strip()]
        if not clean:
            notes.append("no_output")
            return VerificationResult(False, 0.0, 1.0, tuple(notes))

        unique_ratio = len(set(clean)) / float(len(clean))
        if unique_ratio < 0.5:
            notes.append("high_duplication")

        task_tokens = set((task or "").lower().split())
        matched = 0
        for item in clean:
            tokens = set(item.lower().split())
            matched += 1 if task_tokens & tokens else 0
        relevance = matched / float(len(clean))

        confidence = max(0.0, min(1.0, 0.55 * relevance + 0.45 * unique_ratio))
        uncertainty = 1.0 - confidence
        pass_check = confidence >= 0.45

        if pass_check:
            notes.append("verification_pass")
        else:
            notes.append("verification_low_confidence")

        return VerificationResult(pass_check, confidence, uncertainty, tuple(notes))
