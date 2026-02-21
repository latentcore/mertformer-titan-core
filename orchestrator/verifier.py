"""Verification and uncertainty scoring for swarm outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class VerificationResult:
    pass_check: bool
    confidence: float
    uncertainty: float
    consistency: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GateDecision:
    pass_gate: bool
    confidence: float
    uncertainty: float
    consistency: float
    safety_pass: bool
    uncertainty_pass: bool
    notes: tuple[str, ...]
    gate_scores: dict[str, float]


class SwarmVerifier:
    def __init__(
        self,
        min_confidence: float = 0.45,
        min_consistency: float = 0.35,
        max_uncertainty: float = 0.65,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_consistency = float(min_consistency)
        self.max_uncertainty = float(max_uncertainty)

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

    @staticmethod
    def _contains_safety_risk(text: str) -> bool:
        lowered = (text or "").lower()
        unsafe_markers = (
            "hack",
            "exploit",
            "malware",
            "phishing",
            "exfiltrate",
            "backdoor",
            "ransomware",
            "weapon",
            "bomb",
        )
        return any(marker in lowered for marker in unsafe_markers)

    def verify_episode(self, trace: Sequence[object]) -> GateDecision:
        """
        Verify an end-to-end episode trace and return a single gate decision.
        """
        notes: list[str] = []
        outputs: list[str] = []
        safety_flags: list[str] = []

        for step in trace:
            if isinstance(step, Mapping):
                payload: Mapping[str, Any] = step
                text = str(
                    payload.get("output")
                    or payload.get("thought")
                    or payload.get("action")
                    or payload.get("content")
                    or ""
                )
                if bool(payload.get("blocked", False)):
                    safety_flags.append("blocked_step")
                if payload.get("safety_flag"):
                    safety_flags.append(str(payload["safety_flag"]))
            elif isinstance(step, str):
                text = step
            else:
                text = str(step)

            if text.strip():
                outputs.append(text.strip())
            if self._contains_safety_risk(text):
                safety_flags.append("unsafe_content")

        task = outputs[0] if outputs else "episode_trace"
        base = self.verify(task=task, outputs=outputs)

        safety_pass = len(safety_flags) == 0
        uncertainty_pass = base.uncertainty <= self.max_uncertainty
        pass_gate = base.pass_check and safety_pass and uncertainty_pass

        notes.extend(list(base.notes))
        if safety_pass:
            notes.append("safety_pass")
        else:
            notes.append("safety_block")
            notes.extend(f"safety:{item}" for item in safety_flags)
        if uncertainty_pass:
            notes.append("uncertainty_pass")
        else:
            notes.append("uncertainty_high")

        deduped_notes: list[str] = []
        for note in notes:
            if note not in deduped_notes:
                deduped_notes.append(note)

        return GateDecision(
            pass_gate=pass_gate,
            confidence=base.confidence,
            uncertainty=base.uncertainty,
            consistency=base.consistency,
            safety_pass=safety_pass,
            uncertainty_pass=uncertainty_pass,
            notes=tuple(deduped_notes),
            gate_scores={
                "confidence": base.confidence,
                "consistency": base.consistency,
                "uncertainty_margin": self.max_uncertainty - base.uncertainty,
            },
        )
