"""
CHDM v0.1 §3 — Evidence Ontology.

EvidenceObject is the basic evidentiary unit (§3.1). For Milestone 1,
evidence is hand-authored directly at its target state/confirmation
(BAR-01 §6: "evidence and confirmation states in fixtures are
hand-authored directly at their target status, not produced by a live
extraction/confirmation flow") — there is no live AI extraction pipeline
in this milestone. The dataclass still carries `provenance` so the
engine's rules (e.g. §3.3 confirmation boundary) are exercised exactly as
they would be against real extraction output later.
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import EvidenceState, SignalDirection, Provenance, DimensionCode


@dataclass(frozen=True)
class EvidenceObject:
    """CHDM v0.1 §3.1 canonical fields."""
    evidence_id: str
    dimension: Optional[DimensionCode]      # construct informed by the evidence; may be None
                                              # for evidence that informs Objective/Risk only
    indicator_observation: str               # what was actually observed (not an interpretation)
    source: str                               # structured input, note, QBR, survey, service record, etc.
    provenance: Provenance
    observed_date_period: Optional[str] = None
    applicability: bool = True                # Applicable / Not Applicable
    evidence_state: EvidenceState = EvidenceState.CURRENT_UNVERIFIED
    signal_direction: SignalDirection = SignalDirection.NON_DIRECTIONAL
    source_excerpt_reference: Optional[str] = None
    risk_mechanism_refs: tuple[str, ...] = field(default_factory=tuple)  # e.g. ("CR-01",)

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("EvidenceObject.evidence_id must not be empty")
        if not self.indicator_observation:
            raise ValueError(
                "EvidenceObject.indicator_observation must not be empty — "
                "an observation and an interpretation are different objects (§3.1)."
            )
        # §3.3 confirmation boundary is enforced at evaluation time, not
        # construction time here, because AI-extracted evidence legitimately
        # starts Current+Unverified and is promoted later via a separate
        # confirmation action (never mutated in place — see
        # engine/evidence_engine.py confirmation handling once built).

    @property
    def is_current_confirmed(self) -> bool:
        return self.evidence_state == EvidenceState.CURRENT_CONFIRMED
