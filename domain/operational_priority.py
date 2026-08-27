"""
CHDM v0.1 §10 — Operational Priority.

Note: the OPU + ER0 invariant (INV-14) spans two separate governed
objects (OperationalPriorityResult and EvidenceReviewResult) and cannot be
checked from either object alone. It is enforced in
engine/invariants.py at evaluation-time assembly, per the Technical
Architecture's three-layer validation design (object creation /
evaluation-time combination / serialization).
"""

from dataclasses import dataclass, field

from .enums import OperationalPriority as OperationalPriorityValue
from .reason_code import ReasonCode


@dataclass(frozen=True)
class OperationalPriorityResult:
    value: OperationalPriorityValue
    reason_code: ReasonCode
    contributing_risk_or_dimension_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.value == OperationalPriorityValue.OP1 and not self.contributing_risk_or_dimension_refs:
            # INV-15: OP1 with no Current + Confirmed Critical risk is invalid —
            # OP1 must always be traceable to at least one activated Critical risk.
            raise ValueError(
                "INV-15 violation: OperationalPriorityResult.value == OP1 requires "
                "at least one contributing risk reference (a Current + Confirmed "
                "Critical risk). OP1 with no confirmed Critical risk is invalid "
                "(CHDM v0.1 §10.1)."
            )
