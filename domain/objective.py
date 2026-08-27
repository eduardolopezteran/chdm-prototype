"""
CHDM v0.1 §5 — Objective Outcome and Value Evidence Basis.
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import ObjectiveOutcomeState, ValueEvidenceBasis
from .reason_code import ReasonCode


@dataclass(frozen=True)
class Objective:
    objective_id: str
    text: Optional[str]                 # None only if genuinely Unknown
    source_evidence_ref: Optional[str] = None
    is_known: bool = True                # False -> objective_outcome_state must be UNKNOWN

    def __post_init__(self) -> None:
        if not self.is_known and self.text is not None:
            raise ValueError(
                "Objective marked unknown (is_known=False) must not carry text — "
                "the product must never fabricate an objective to enable a conclusion (§5)."
            )


@dataclass(frozen=True)
class ObjectiveOutcome:
    """The governed, deterministic result of evaluating an Objective
    against confirmed evidence — never set directly by AI or by a human
    typing a value (only ever produced by engine/objective_engine.py)."""
    objective_id: str
    state: ObjectiveOutcomeState
    value_evidence_basis: tuple[ValueEvidenceBasis, ...]   # multi-valued, no hierarchy (§5.2)
    reason_code: ReasonCode
    contributing_evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.state == ObjectiveOutcomeState.ACHIEVED:
            # §5.3 / INV-04 / INV-05: Proxy Supported alone, or Unverified
            # Claim alone, must never produce Achieved.
            direct_bases = {
                ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE,
                ValueEvidenceBasis.CUSTOMER_CONFIRMED,
                ValueEvidenceBasis.INDEPENDENTLY_VERIFIED,
            }
            if not (set(self.value_evidence_basis) & direct_bases):
                raise ValueError(
                    "INV-04/INV-05 violation: Objective Outcome = ACHIEVED requires at "
                    "least one direct evidence basis (Measured Operational Evidence, "
                    "Customer Confirmed, or Independently Verified); Proxy Supported or "
                    "Unverified Claim alone cannot produce Achieved (CHDM v0.1 §5.3)."
                )
        if not self.contributing_evidence_refs:
            raise ValueError(
                "ObjectiveOutcome must cite contributing evidence — a governed "
                "output without traceable evidence is invalid (§3.5)."
            )
