"""
CHDM v0.1 §4, §7 — Dimension State (governed output per dimension).
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import DimensionCode, DimensionStateValue, RequirementClass
from .reason_code import ReasonCode


@dataclass(frozen=True)
class DimensionState:
    dimension: DimensionCode
    state: DimensionStateValue
    requirement_class: RequirementClass
    reason_code: ReasonCode
    contributing_evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    dimension_reliability: Optional[str] = None  # "HIGH" | "MEDIUM" | "LOW" — dimension-level,
                                                    # distinct from overall AssessmentReliability (§9)
    applicability_rule_ref: Optional[str] = None   # required if state == NOT_APPLICABLE (§7 NA rule)

    def __post_init__(self) -> None:
        if self.state == DimensionStateValue.NOT_APPLICABLE and not self.applicability_rule_ref:
            # INV-03: NA without a canonical applicability rule is invalid.
            raise ValueError(
                "INV-03 violation: DimensionState.state == NOT_APPLICABLE requires "
                "applicability_rule_ref citing a canonical applicability rule — NA must "
                "never be assigned merely because evidence is unavailable, unknown, "
                "stale or uncollected (CHDM v0.1 §7)."
            )
        if self.state != DimensionStateValue.NOT_APPLICABLE and not self.contributing_evidence_refs:
            raise ValueError(
                "DimensionState must cite contributing evidence for any resolved "
                "state (§3.5 traceability)."
            )
