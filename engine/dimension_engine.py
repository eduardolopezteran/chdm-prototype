"""
CHDM v0.1 §4 — Dimension State evaluation.

Build Milestone 1 implements only the dimensions Scenario Lab S1-S6
requires: D1 (Value Realization), D2 (Product Adoption), D6 (Relationship
Health). Any other dimension code raises NotImplementedForMilestone1 —
explicit, not a silent default — because building a general text-parsing
interpreter for D3-D5/D7/D8A/D8B's prose state rules is out of scope
until a scenario actually requires it (per the executive build
instruction: "dimension-state evaluation required by S1-S6").

D1 is derived entirely from already-computed upstream constructs
(ObjectiveOutcome + the CR-08 RiskRecord) — no additional signal type
needed, because §4.2's D1 state rules are themselves phrased directly in
terms of Objective Outcome and CR-08 activation.

D2 and D6 use DimensionQualifierSignal (domain/signals.py) because their
state rules reference qualitative account characteristics (breadth,
concentration, automation reliability, sponsor coverage) that are not
themselves outputs of any other engine construct — exactly the same
"already-classified structured claim, confirmation-gated" pattern used
for risk mechanisms.
"""

from __future__ import annotations

from domain.enums import DimensionCode, DimensionStateValue, RequirementClass, RiskSeverity
from domain.dimension_state import DimensionState
from domain.objective import ObjectiveOutcome
from domain.enums import ObjectiveOutcomeState
from domain.reason_code import ReasonCode
from domain.risk_record import RiskRecord
from domain.signals import DimensionQualifierSignal
from .evidence_engine import is_current_confirmed
from .errors import NotImplementedForMilestone1

IMPLEMENTED_DIMENSIONS = frozenset({DimensionCode.D1, DimensionCode.D2, DimensionCode.D6})

_D2_CONCERNING = {"WORKFLOWS_NOT_OCCURRING", "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED"}
_D2_MIXED = {"NARROW_BREADTH_OR_CONCENTRATION"}
_D2_SUPPORTED = {"INTENDED_WORKFLOWS_OPERATING_NORMALLY", "AUTOMATION_RELIABLE_LOW_LOGIN_OK"}

_D6_CONCERNING = {"CHAMPION_LOST_NO_SUCCESSOR"}
_D6_MIXED = {"SUCCESSION_UNCLEAR_OR_CONCENTRATED"}
_D6_SUPPORTED = {"APPROPRIATE_SPONSOR_COVERAGE"}


def evaluate_d1(
    objective_outcome: ObjectiveOutcome,
    cr08_risk: RiskRecord | None,
    requirement_class: RequirementClass,
) -> DimensionState:
    evidence_refs = objective_outcome.contributing_evidence_refs
    if cr08_risk is not None:
        evidence_refs = evidence_refs + cr08_risk.contributing_evidence_refs

    # Special contradiction rule (§4.2, §12.2) — takes precedence over
    # every other D1 rule.
    if objective_outcome.state == ObjectiveOutcomeState.DISPUTED:
        return DimensionState(
            dimension=DimensionCode.D1,
            state=DimensionStateValue.MIXED,
            requirement_class=requirement_class,
            reason_code=_reason("D1-DISPUTED-MIXED", "CHDM-DIM-VALUE-001",
                                 "Objective Outcome = Disputed deterministically produces D1 = Mixed, "
                                 "D1 Reliability = Low (CHDM v0.1 §4.2, §12.2)."),
            contributing_evidence_refs=objective_outcome.contributing_evidence_refs,
            dimension_reliability="LOW",
        )

    cr08_concerning = (
        cr08_risk is not None
        and cr08_risk.activated_severity in (RiskSeverity.MATERIAL, RiskSeverity.CRITICAL)
    )

    if objective_outcome.state == ObjectiveOutcomeState.NOT_ACHIEVED or cr08_concerning:
        return DimensionState(
            DimensionCode.D1, DimensionStateValue.CONCERNING, requirement_class,
            _reason("D1-CONCERNING", "CHDM-DIM-VALUE-001",
                    "Objective Outcome = Not Achieved, or a confirmed Material/Critical "
                    "CR-08 value-risk is active (CHDM v0.1 §4.2)."),
            evidence_refs,
        )
    if objective_outcome.state == ObjectiveOutcomeState.ACHIEVED:
        return DimensionState(
            DimensionCode.D1, DimensionStateValue.SUPPORTED, requirement_class,
            _reason("D1-SUPPORTED", "CHDM-DIM-VALUE-001",
                    "Objective Outcome = Achieved; adequate current confirmed direct "
                    "evidence exists; no material contradiction remains (§4.2)."),
            evidence_refs,
        )
    if objective_outcome.state == ObjectiveOutcomeState.PROGRESSING:
        return DimensionState(
            DimensionCode.D1, DimensionStateValue.MIXED, requirement_class,
            _reason("D1-MIXED-PROGRESSING", "CHDM-DIM-VALUE-001",
                    "Lifecycle-appropriate Progressing (§4.2)."),
            evidence_refs,
        )
    if objective_outcome.state == ObjectiveOutcomeState.UNKNOWN:
        return DimensionState(
            DimensionCode.D1, DimensionStateValue.INSUFFICIENT_EVIDENCE, requirement_class,
            _reason("D1-IE-UNKNOWN", "CHDM-DIM-VALUE-001",
                    "Objective Outcome = Unknown or necessary evidence is "
                    "missing/stale/unverified (§4.2)."),
            evidence_refs,
        )
    raise NotImplementedForMilestone1(
        f"D1 evaluation for ObjectiveOutcomeState.{objective_outcome.state.value} "
        "(e.g. Not Yet Expected) is not required by S1-S6 and is not implemented "
        "in Build Milestone 1."
    )


def evaluate_d2(signals: tuple[DimensionQualifierSignal, ...]) -> DimensionState:
    return _evaluate_qualifier_dimension(
        DimensionCode.D2, RequirementClass.UR, signals,
        _D2_CONCERNING, _D2_MIXED, _D2_SUPPORTED,
    )


def evaluate_d6(signals: tuple[DimensionQualifierSignal, ...]) -> DimensionState:
    return _evaluate_qualifier_dimension(
        DimensionCode.D6, RequirementClass.UR, signals,
        _D6_CONCERNING, _D6_MIXED, _D6_SUPPORTED,
    )


def _evaluate_qualifier_dimension(
    dimension: DimensionCode,
    requirement_class: RequirementClass,
    signals: tuple[DimensionQualifierSignal, ...],
    concerning_qualifiers: set[str],
    mixed_qualifiers: set[str],
    supported_qualifiers: set[str],
) -> DimensionState:
    relevant = tuple(s for s in signals if s.dimension == dimension)
    confirmed = tuple(s for s in relevant if is_current_confirmed(s.evidence_state))
    confirmed_qualifiers = {s.qualifier for s in confirmed}

    dim_id = {DimensionCode.D2: "CHDM-DIM-ADOPTION-001", DimensionCode.D6: "CHDM-DIM-RELATIONSHIP-001"}[dimension]

    if confirmed_qualifiers & concerning_qualifiers:
        hits = tuple(s for s in confirmed if s.qualifier in concerning_qualifiers)
        return DimensionState(
            dimension, DimensionStateValue.CONCERNING, requirement_class,
            _reason(f"{dimension.value}-CONCERNING", dim_id,
                    f"Confirmed concerning qualifier(s): {sorted(confirmed_qualifiers & concerning_qualifiers)} "
                    f"(CHDM v0.1 §4)."),
            tuple(s.evidence_id for s in hits),
        )
    if confirmed_qualifiers & mixed_qualifiers:
        hits = tuple(s for s in confirmed if s.qualifier in mixed_qualifiers)
        return DimensionState(
            dimension, DimensionStateValue.MIXED, requirement_class,
            _reason(f"{dimension.value}-MIXED", dim_id,
                    f"Confirmed qualifier(s) weakening durability: "
                    f"{sorted(confirmed_qualifiers & mixed_qualifiers)} (CHDM v0.1 §4)."),
            tuple(s.evidence_id for s in hits),
        )
    if confirmed_qualifiers & supported_qualifiers:
        hits = tuple(s for s in confirmed if s.qualifier in supported_qualifiers)
        return DimensionState(
            dimension, DimensionStateValue.SUPPORTED, requirement_class,
            _reason(f"{dimension.value}-SUPPORTED", dim_id,
                    f"Confirmed supporting qualifier(s): {sorted(confirmed_qualifiers & supported_qualifiers)} "
                    f"(CHDM v0.1 §4)."),
            tuple(s.evidence_id for s in hits),
        )
    return DimensionState(
        dimension, DimensionStateValue.INSUFFICIENT_EVIDENCE, requirement_class,
        _reason(f"{dimension.value}-IE", dim_id,
                "No confirmed qualifying evidence available to establish a governed "
                "state (CHDM v0.1 §4)."),
        tuple(s.evidence_id for s in relevant) or ("NO_EVIDENCE_CONSIDERED",),
    )


def _reason(code: str, governing_object_id: str, text: str) -> ReasonCode:
    return ReasonCode(code=code, governing_object_id=governing_object_id, human_readable_text=text)
