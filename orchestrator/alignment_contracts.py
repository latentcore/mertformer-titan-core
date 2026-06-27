"""Formalized alignment contract checks (simulation-grade).

inert / out-of-scope: 45K egitim yolunda kapali (feature-flag); bu modul
orchestrator katmanina ait olup egitim hattinda etkin degildir.

DURUSTLUK NOTU: Asagidaki kontrol GERCEK bir hizalama/guvenlik denetimi
DEGILDIR. Sadece sabit bir engelli-kelime listesine karsi buyuk/kucuk harf
duyarsiz substring eslemesi yapar (keyword filtresi). Semantik hizalama,
niyet analizi veya gercek policy denetimi YOKTUR -- stub/simulation-grade
bir yer tutucudur. Bu sonuc bir gecme-kapisi (pass-gate) olarak
kullanilmamalidir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class AlignmentViolation:
    rule_id: str
    message: str
    severity: str


class AlignmentContracts:
    def __init__(self) -> None:
        self.blocked_tokens = (
            "exfiltrate",
            "bypass policy",
            "stealth monitoring",
            "unauthorized surveillance",
            "covert tracking",
        )

    def check_prompt(self, prompt: str) -> List[AlignmentViolation]:
        """Sabit engelli-kelime listesine karsi substring eslemesi (keyword
        filtresi). Gercek semantik hizalama denetimi DEGIL; stub-grade."""
        text = (prompt or "").lower()
        violations: List[AlignmentViolation] = []
        for token in self.blocked_tokens:
            if token in text:
                violations.append(
                    AlignmentViolation(
                        rule_id="A-POL-001",
                        message=f"Blocked token detected: {token}",
                        severity="high",
                    )
                )
        return violations

    def check_batch(self, prompts: Iterable[str]) -> List[AlignmentViolation]:
        out: List[AlignmentViolation] = []
        for p in prompts:
            out.extend(self.check_prompt(p))
        return out

