"""
CHDM v0.1 §6 — Risk Model. RiskRecord holds the two independent severity
axes plus evidence status for one CR mechanism within one assessment.
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import RiskMechanismCode, RiskSeverity, EvidenceState
from .reason_code import ReasonCode


@dataclass(frozen=True)
class RiskRecord:
    mechanism: RiskMechanismCode
    potential_severity: Optional[RiskSeverity]
    activated_severity: Optional[RiskSeverity]
    evidence_status: EvidenceState
    reason_code: Optional[ReasonCode]                 # required if activated_severity is set
    contributing_evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.activated_severity is not None:
            # INV-07: activated Material/Critical severity requires
            # Current + Confirmed evidence satisfying the governing rule.
            if self.evidence_status != EvidenceState.CURRENT_CONFIRMED:
                raise ValueError(
                    "INV-07 violation: RiskRecord.activated_severity is set but "
                    "evidence_status is not CURRENT_CONFIRMED. Activated Material/"
                    "Critical severity requires Current + Confirmed evidence "
                    "satisfying the canonical severity condition (CHDM v0.1 §6.1)."
                )
            if self.activated_severity in (RiskSeverity.MATERIAL, RiskSeverity.CRITICAL) and not self.reason_code:
                raise ValueError(
                    "RiskRecord with activated MATERIAL/CRITICAL severity must "
                    "carry a ReasonCode (§13.4)."
                )
            if not self.contributing_evidence_refs:
                raise ValueError(
                    "RiskRecord with activated severity must cite contributing "
                    "evidence (§3.5 traceability)."
                )

    @property
    def is_activated(self) -> bool:
        return self.activated_severity is not None
