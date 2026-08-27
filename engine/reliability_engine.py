"""
CHDM v0.1 §9 — Assessment Reliability. Non-compensatory: governed
entirely by whether any DMEG exists (LOW) or, absent any DMEG, by
whether non-material limitations were noted (MEDIUM) or not (HIGH).
Structurally guarantees REL-01 / INV-11 (LOW always carries at least one
real DMEG id as its limiting factor).
"""

from __future__ import annotations

from domain.dmeg import DMEG
from domain.enums import AssessmentReliabilityLevel
from domain.reliability import AssessmentReliability


def evaluate_reliability(
    dmegs: tuple[DMEG, ...],
    non_dmeg_limitations: tuple[str, ...] = (),
) -> AssessmentReliability:
    if dmegs:
        return AssessmentReliability(
            level=AssessmentReliabilityLevel.LOW,
            limiting_factor_refs=tuple(d.dmeg_id for d in dmegs),
        )
    if non_dmeg_limitations:
        return AssessmentReliability(
            level=AssessmentReliabilityLevel.MEDIUM,
            limiting_factor_refs=non_dmeg_limitations,
        )
    return AssessmentReliability(level=AssessmentReliabilityLevel.HIGH)
