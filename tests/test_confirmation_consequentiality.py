"""
Milestone 3A — consequentiality tests. compute_consequentiality() reuses
engine.dmeg_engine.differs() exactly as engine/evaluate.py's own internal
differential steps do; these tests prove it correctly distinguishes a
review decision that changes the governed outcome from one that does not.
"""

import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import ConfirmationAction, DimensionCode, EvidenceState, Lifecycle
from domain.objective import Objective
from engine.registry_loader import load_and_validate
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    CandidateRiskSignal, ExtractionSystemFields, ObservationRef, SourceSpan,
    StakeholderObservation,
)

from confirmation.consequentiality import compute_consequentiality
from confirmation.enums import ConfirmationTargetKind
from confirmation.schemas import create_confirmation_record

REGISTRY = load_and_validate()
OBJECTIVE = Objective("OBJ-1", "Reduce monthly reconciliation effort by 50%", is_known=True)


def make_account():
    return AccountAssessment(
        assessment_id="ASSESS-M3A-CONSEQ-1",
        scope=Scope("SCOPE-M3A-CONSEQ-1", "Fictional Southgate Analytics", "Forecast Rollout"),
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


# ---- 1. Consequential: the only CR-01 signal in the assessment ----

def build_single_signal_extraction_result():
    stakeholder = StakeholderObservation(
        source_evidence_id="E-CHAMPION-1", source_span=_span("the champion has left the company"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Jane Doe", system=_system("OBS-STAKE-1"),
    )
    risk_signal = CandidateRiskSignal(
        source_evidence_id="E-CHAMPION-1", source_span=_span("the champion has left the company"),
        basis=InferenceBasis.INFERRED_CANDIDATE, mechanism="CR-01", proposed_severity_tier="CRITICAL",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=0),
        resolved_observation_id="OBS-STAKE-1", system=_system("OBS-RISK-1"),
    )
    return ExtractionResult(
        accepted=(stakeholder,), candidate_contradictions=(), candidate_risk_signals=(risk_signal,),
        candidate_evidence_classifications=(), rejected=(), dedup_audit=(), traces=(),
    )


def test_confirming_the_only_critical_risk_is_consequential():
    extraction_result = build_single_signal_extraction_result()
    confirm_risk = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    report = compute_consequentiality(
        make_account(), REGISTRY, extraction_result, (confirm_risk,), "OBS-RISK-1",
        dimensions_to_evaluate=(DimensionCode.D1,),
    )
    assert report.is_consequential is True
    assert report.outcome_with_review != report.outcome_without_review


# ---- 2. Non-consequential: a bystander semantic observation that no
#         risk claim, value signal, or dimension_qualifier_override ever
#         references. Reviewing it cannot change anything evaluate() sees,
#         since it never becomes part of any of the three signal tuples --
#         a clean "reviewing this changes nothing downstream" case.

def build_bystander_extraction_result():
    stakeholder = StakeholderObservation(
        source_evidence_id="E-CHAMPION-1", source_span=_span("the champion has left the company"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Jane Doe", system=_system("OBS-STAKE-1"),
    )
    risk_signal = CandidateRiskSignal(
        source_evidence_id="E-CHAMPION-1", source_span=_span("the champion has left the company"),
        basis=InferenceBasis.INFERRED_CANDIDATE, mechanism="CR-01", proposed_severity_tier="CRITICAL",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=0),
        resolved_observation_id="OBS-STAKE-1", system=_system("OBS-RISK-1"),
    )
    bystander = StakeholderObservation(
        source_evidence_id="E-INTRO-1", source_span=_span("account introduced by our partner team"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Partner Contact", system=_system("OBS-STAKE-BYSTANDER"),
    )
    return ExtractionResult(
        accepted=(stakeholder, bystander), candidate_contradictions=(),
        candidate_risk_signals=(risk_signal,), candidate_evidence_classifications=(),
        rejected=(), dedup_audit=(), traces=(),
    )


def test_reviewing_an_unreferenced_bystander_observation_is_not_consequential():
    extraction_result = build_bystander_extraction_result()
    confirm_risk = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    confirm_bystander = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-STAKE-BYSTANDER",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    report = compute_consequentiality(
        make_account(), REGISTRY, extraction_result, (confirm_risk, confirm_bystander),
        "OBS-STAKE-BYSTANDER", dimensions_to_evaluate=(DimensionCode.D1,),
    )
    assert report.is_consequential is False
    assert report.outcome_with_review == report.outcome_without_review


TESTS = [
    test_confirming_the_only_critical_risk_is_consequential,
    test_reviewing_an_unreferenced_bystander_observation_is_not_consequential,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
