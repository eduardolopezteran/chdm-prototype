"""
CHDM v0.1 §6 — Risk Model. Deterministic evaluation over RiskSeverityClaim
objects for one mechanism at a time.

Potential severity aggregates across ANY usable evidence (Confirmed,
Unverified, Stale, Contradictory — anything except Unavailable/Not
Applicable, per §3.2). Activated severity aggregates ONLY over
Current+Confirmed evidence (§3.3, §6.1, INV-07) — this is the
confirmation boundary, enforced identically for every mechanism rather
than reimplemented per-mechanism.
"""

from __future__ import annotations

from domain.enums import RiskMechanismCode, RiskSeverity, EvidenceState
from domain.reason_code import ReasonCode
from domain.risk_record import RiskRecord
from domain.signals import RiskSeverityClaim
from .evidence_engine import is_current_confirmed, is_usable_as_potential

_TIER_ORDER = {"WATCH": 1, "MATERIAL": 2, "CRITICAL": 3}
_TIER_TO_SEVERITY = {
    "WATCH": RiskSeverity.WATCH,
    "MATERIAL": RiskSeverity.MATERIAL,
    "CRITICAL": RiskSeverity.CRITICAL,
}

_MECHANISM_OBJECT_ID = {
    RiskMechanismCode.CR_01: "CHDM-RISK-CHAMPION-001",
    RiskMechanismCode.CR_02: "CHDM-RISK-SERVICE-001",
    RiskMechanismCode.CR_03: "CHDM-RISK-COMMERCIAL-001",
    RiskMechanismCode.CR_06: "CHDM-RISK-AUTOMATION-001",
    RiskMechanismCode.CR_08: "CHDM-RISK-VALUEFAILURE-001",
}

IMPLEMENTED_MECHANISMS = frozenset(_MECHANISM_OBJECT_ID.keys())


def evaluate_risk(
    mechanism: RiskMechanismCode,
    claims: tuple[RiskSeverityClaim, ...],
) -> RiskRecord:
    if mechanism not in IMPLEMENTED_MECHANISMS:
        raise NotImplementedError(
            f"{mechanism.value} is not implemented in Build Milestone 1 "
            "(registry/risk_mechanisms.yaml marks it 'deferred' — a scoping "
            "decision authorized by BAR-01 §4, not a methodology gap)."
        )

    mechanism_claims = tuple(c for c in claims if c.mechanism == mechanism)

    potential_claims = tuple(c for c in mechanism_claims if is_usable_as_potential(c.evidence_state))
    activated_claims = tuple(c for c in mechanism_claims if is_current_confirmed(c.evidence_state))

    potential_severity = _max_tier(potential_claims)
    activated_severity = _max_tier(activated_claims)

    if activated_severity is not None:
        governing = _claims_at_tier(activated_claims, activated_severity)
        evidence_status = EvidenceState.CURRENT_CONFIRMED
        reason_text = (
            f"Activated {activated_severity} — Current + Confirmed evidence satisfies "
            f"the canonical {activated_severity} condition for {mechanism.value} (CHDM v0.1 §6)."
        )
    elif potential_severity is not None:
        governing = _claims_at_tier(potential_claims, potential_severity)
        evidence_status = governing[0].evidence_state
        reason_text = (
            f"Potential {potential_severity} identified from unconfirmed evidence; "
            f"not activated — only Current + Confirmed evidence may activate severity "
            f"(CHDM v0.1 §3.3, §6.1)."
        )
    else:
        governing = ()
        evidence_status = EvidenceState.UNAVAILABLE
        reason_text = f"No evidence available for {mechanism.value}."

    reason_code = ReasonCode(
        code=f"RISK-{mechanism.value}-{activated_severity or potential_severity or 'NONE'}",
        governing_object_id=_MECHANISM_OBJECT_ID[mechanism],
        human_readable_text=reason_text,
    ) if governing else None

    return RiskRecord(
        mechanism=mechanism,
        potential_severity=_TIER_TO_SEVERITY.get(potential_severity) if potential_severity else None,
        activated_severity=_TIER_TO_SEVERITY.get(activated_severity) if activated_severity else None,
        evidence_status=evidence_status,
        reason_code=reason_code,
        contributing_evidence_refs=tuple(c.evidence_id for c in governing),
    )


def _max_tier(claims: tuple[RiskSeverityClaim, ...]) -> str | None:
    if not claims:
        return None
    return max((c.tier for c in claims), key=lambda t: _TIER_ORDER[t])


def _claims_at_tier(claims: tuple[RiskSeverityClaim, ...], tier: str) -> tuple[RiskSeverityClaim, ...]:
    return tuple(c for c in claims if c.tier == tier)
