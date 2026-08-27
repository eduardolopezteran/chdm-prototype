"""
CHDM v0.1 §9 — Assessment Reliability.
"""

from dataclasses import dataclass, field

from .enums import AssessmentReliabilityLevel


@dataclass(frozen=True)
class AssessmentReliability:
    level: AssessmentReliabilityLevel
    limiting_factor_refs: tuple[str, ...] = field(default_factory=tuple)
    # e.g. ("DMEG-D6-001", "DMEG-VALUE-002") — non-compensatory: governed by
    # the least reliable materially-contributing evidence, not overall completeness.

    def __post_init__(self) -> None:
        if self.level == AssessmentReliabilityLevel.LOW and not self.limiting_factor_refs:
            # INV-11 / REL-01: Low overall Reliability with no DMEG is invalid.
            # limiting_factor_refs standing empty while LOW is the same defect
            # surfaced at the domain-object level.
            raise ValueError(
                "INV-11/REL-01 violation: AssessmentReliability.level == LOW "
                "requires at least one limiting_factor_ref (an open DMEG). "
                "'Overall Reliability = Low with no DMEG is invalid' (CHDM v0.1 §9)."
            )
        # INV-10: never expressed numerically or as a percentage — enforced
        # structurally by AssessmentReliabilityLevel being an Enum, not a number.
