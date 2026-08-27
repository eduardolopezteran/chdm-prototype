"""
Milestone 3A — recompute() end-to-end tests: confirmation decisions ->
Milestone 1 signals -> engine.evaluate() (unmodified), full round trip.
Mirrors tests/test_evaluate_end_to_end.py's Example A/B shape, but the
signals here are produced by confirmation.recompute.recompute() from
confirmation decisions over hand-authored, pipeline-finalized extraction
objects, not hand-authored Milestone 1 signals directly.
"""

import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import (
    ConfirmationAction, DimensionCode, EvidenceState, Lifecycle,
    OperationalPriority, RiskMechanismCode, RiskSeverity,
)
from domain.objective import Objective
from engine.registry_loader import load_and_validate
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    CandidateEvidenceClassification, CandidateRiskSignal, ExtractionSystemFields,
    ObjectiveCandidate, ObservationRef, SourceSpan, StakeholderObservation,
)

from confirmation.active_evidence import reconstruct_active_evidence
from confirmation.enums import ConfirmationTargetKind
from confirmation.recompute import recompute
from confirmation.schemas import create_confirmation_record

REGISTRY = load_and_validate()
OBJECTIVE = Objective("OBJ-1", "Reduce monthly reconciliation effort by 50%", is_known=True)


def make_account():
    return AccountAssessment(
        assessment_id="ASSESS-M3A-1",
        scope=Scope("SCOPE-M3A-1", "Fictional Northwind Robotics", "Reconciliation Suite"),
        lifecycle=Lifecycle.L3, objective=OBJECTIVE,
    )


def _system(obs_id: str) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="test-provider", model_version="test-v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
        evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def build_extraction_result():
    stakeholder = StakeholderObservation(
        source_evidence_id="E-CHAMPION-1", source_span=_span("the champion has left the company"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Jane Doe", system=_system("OBS-STAKE-1"),
    )
    objective_candidate = ObjectiveCandidate(
        source_evidence_id="E-VAL-1", source_span=_span("reconciliation time dropped by half"),
        basis=InferenceBasis.EXPLICIT, objective_text="Reduce monthly reconciliation effort by 50%",
        system=_system("OBS-OBJ-1"),
    )
    risk_signal = CandidateRiskSignal(
        source_evidence_id="E-CHAMPION-1", source_span=_span("the champion has left the company"),
        basis=InferenceBasis.INFERRED_CANDIDATE, mechanism="CR-01", proposed_severity_tier="CRITICAL",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=0),
        resolved_observation_id="OBS-STAKE-1", system=_system("OBS-RISK-1"),
    )
    evidence_classification = CandidateEvidenceClassification(
        source_evidence_id="E-VAL-1", source_span=_span("confirmed by the finance ops report"),
        basis=InferenceBasis.INFERRED_CANDIDATE, proposed_basis="MEASURED_OPERATIONAL_EVIDENCE",
        supports="ACHIEVED",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.OBJECTIVE_CANDIDATE, index=0),
        resolved_observation_id="OBS-OBJ-1", system=_system("OBS-EVIDCLASS-1"),
    )
    return ExtractionResult(
        accepted=(stakeholder, objective_candidate), candidate_contradictions=(),
        candidate_risk_signals=(risk_signal,), candidate_evidence_classifications=(evidence_classification,),
        rejected=(), dedup_audit=(), traces=(),
    )


_D6_OVERRIDE = {"OBS-STAKE-1": (DimensionCode.D6, "CHAMPION_LOST_NO_SUCCESSOR")}


def _run(records, dimension_qualifier_overrides=_D6_OVERRIDE):
    extraction_result = build_extraction_result()
    active = reconstruct_active_evidence(extraction_result, records)
    return recompute(
        make_account(), REGISTRY, active,
        dimensions_to_evaluate=(DimensionCode.D1, DimensionCode.D6),
        dimension_qualifier_overrides=dimension_qualifier_overrides,
    )


def _confirm(target_kind, target_id):
    return create_confirmation_record(
        target_kind=target_kind, target_observation_id=target_id,
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )


# ---- 1. Confirmed CR-01 activates the expected governed behavior ----

def test_confirmed_cr01_critical_activates_op1():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-STAKE-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, "OBS-RISK-1"),
    )
    diag = _run(records)
    result = diag.result
    assert result.operational_priority.value == OperationalPriority.OP1
    assert result.risk_records[RiskMechanismCode.CR_01].activated_severity == RiskSeverity.CRITICAL
    assert result.dimension_states[DimensionCode.D6].state.value == "CONCERNING"


# ---- 2. Rejecting the same candidate removes that effect ----

def test_rejecting_risk_signal_removes_op1_activation():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-STAKE-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-1"),
        create_confirmation_record(
            target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
            action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
            reason="Extracted fact is materially inaccurate.",
        ),
    )
    diag = _run(records)
    result = diag.result
    assert result.operational_priority.value != OperationalPriority.OP1
    assert diag.active_evidence.is_excluded("OBS-RISK-1")
    cr01 = result.risk_records.get(RiskMechanismCode.CR_01)
    assert cr01 is None or cr01.activated_severity is None


# ---- 3. Cannot Confirm preserves uncertainty ----

def test_cannot_confirm_preserves_uncertainty_no_activation():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-STAKE-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-1"),
        create_confirmation_record(
            target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
            action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
            reason="Conflicting evidence remains unresolved.",
        ),
    )
    diag = _run(records)
    result = diag.result
    item = diag.active_evidence.by_observation_id("OBS-RISK-1")
    assert item.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    assert result.risk_records[RiskMechanismCode.CR_01].potential_severity == RiskSeverity.CRITICAL
    assert result.risk_records[RiskMechanismCode.CR_01].activated_severity is None
    assert result.operational_priority.value != OperationalPriority.OP1


# ---- 4. Correcting a candidate changes the downstream result ----

def test_correcting_tier_from_critical_to_watch_changes_result():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-STAKE-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-1"),
        create_confirmation_record(
            target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
            action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
            corrected_representation={"proposed_severity_tier": "WATCH"},
        ),
    )
    diag = _run(records)
    result = diag.result
    cr01 = result.risk_records[RiskMechanismCode.CR_01]
    assert cr01.potential_severity == RiskSeverity.WATCH
    assert cr01.activated_severity == RiskSeverity.WATCH  # a confirmed Watch does activate at Watch
    assert result.operational_priority.value != OperationalPriority.OP1  # but never elevates to OP1 (CRITICAL-only)


# ---- 5. Semantic fact confirmed while its candidate classification is rejected ----

def test_semantic_confirmed_candidate_rejected_are_independent():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-STAKE-1"),
        create_confirmation_record(
            target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
            action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
            reason="Evidence does not support the proposed statement.",
        ),
    )
    diag = _run(records)
    stake_item = diag.active_evidence.by_observation_id("OBS-STAKE-1")
    assert stake_item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert diag.active_evidence.is_excluded("OBS-RISK-1")
    # The D6 dimension signal (built from the confirmed semantic fact via
    # dimension_qualifier_overrides) is unaffected by the risk signal's
    # rejection -- two fully independent decisions.
    assert diag.dimension_signals[0].evidence_state == EvidenceState.CURRENT_CONFIRMED


# ---- 6. Reproducibility ----

def test_recompute_reproducible():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-STAKE-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, "OBS-RISK-1"),
    )
    r1 = _run(records).result
    r2 = _run(records).result
    assert r1.operational_priority.value == r2.operational_priority.value
    assert {k: v.state for k, v in r1.dimension_states.items()} == {k: v.state for k, v in r2.dimension_states.items()}


# ---- 7. dimension_qualifier_overrides referencing an excluded item raises ----

def test_dimension_qualifier_override_on_rejected_item_raises():
    records = (
        create_confirmation_record(
            target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-STAKE-1",
            action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
            reason="Source refers to the wrong account.",
        ),
    )
    try:
        _run(records)
    except ValueError as exc:
        assert "not an active" in str(exc)
    else:
        raise AssertionError("Expected ValueError for override referencing an excluded item.")


TESTS = [
    test_confirmed_cr01_critical_activates_op1,
    test_rejecting_risk_signal_removes_op1_activation,
    test_cannot_confirm_preserves_uncertainty_no_activation,
    test_correcting_tier_from_critical_to_watch_changes_result,
    test_semantic_confirmed_candidate_rejected_are_independent,
    test_recompute_reproducible,
    test_dimension_qualifier_override_on_rejected_item_raises,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
