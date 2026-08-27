"""
Milestone 3B — sample scenario.

Hand-authored, pipeline-finalized extraction fixtures (same construction
style as tests/test_confirmation_recompute_end_to_end.py), standing in for
a real extraction run since Milestone 3B has no persistence layer and does
not run extraction itself. Gives the review queue a realistic mix: two
candidate risk signals (CR-01, CR-02), one candidate value-evidence
classification, one MissingInformationCandidate, and one bystander
semantic observation with no candidate classification at all (so the UI
can show both lanes -- items with and without an AI-proposed
classification).

dimension_qualifier_overrides is FIXED here and never reviewer-editable
(see ui/state.py and the approved Milestone 3B checkpoint's D2/D6
read-only decision) -- it represents a pre-declared mapping of which
confirmed evidence would supply which D2/D6 qualifier, not something a
CSM picks in this prototype. M3-OD-01 (no AI candidate exists for the
dimension/qualifier axis) remains unresolved; this is a display-only
stand-in for that axis, not a workaround for it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import DimensionCode, EvidenceState, Lifecycle
from domain.objective import Objective
from engine.registry_loader import load_and_validate
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    CandidateEvidenceClassification, CandidateRiskSignal, ExperienceObservation,
    ExtractionSystemFields, MissingInformationCandidate, ObjectiveCandidate,
    ObservationRef, ServiceObservation, SourceSpan, StakeholderObservation,
)


def _system(obs_id: str) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="sample-scenario", model_version="v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
        evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def build_sample_scenario():
    """Returns (account, registry, extraction_result, dimensions_to_evaluate,
    dimension_qualifier_overrides)."""
    registry = load_and_validate()
    objective = Objective("OBJ-1", "Reduce monthly reconciliation effort by 50%", is_known=True)
    account = AccountAssessment(
        assessment_id="ASSESS-M3B-SAMPLE-1",
        scope=Scope("SCOPE-M3B-SAMPLE-1", "Fictional Northwind Robotics", "Reconciliation Suite"),
        lifecycle=Lifecycle.L3, objective=objective,
    )

    stakeholder = StakeholderObservation(
        source_evidence_id="E-CHAMPION-1",
        source_span=_span("Our main sponsor, Jane Doe, left the company last month and no successor has been named."),
        basis=InferenceBasis.EXPLICIT, person_identifier="Jane Doe",
        system=_system("OBS-STAKE-1"),
    )
    risk_signal_champion = CandidateRiskSignal(
        source_evidence_id="E-CHAMPION-1",
        source_span=_span("left the company last month and no successor has been named"),
        basis=InferenceBasis.INFERRED_CANDIDATE, mechanism="CR-01", proposed_severity_tier="CRITICAL",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=0),
        resolved_observation_id="OBS-STAKE-1", system=_system("OBS-RISK-CHAMPION"),
    )

    objective_candidate = ObjectiveCandidate(
        source_evidence_id="E-VAL-1", source_span=_span("Monthly reconciliation time dropped from 40 hours to 18 hours."),
        basis=InferenceBasis.EXPLICIT, objective_text="Reduce monthly reconciliation effort by 50%",
        system=_system("OBS-OBJ-1"),
    )
    evidence_classification = CandidateEvidenceClassification(
        source_evidence_id="E-VAL-1", source_span=_span("confirmed by the finance ops monthly report"),
        basis=InferenceBasis.INFERRED_CANDIDATE, proposed_basis="MEASURED_OPERATIONAL_EVIDENCE",
        supports="ACHIEVED",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.OBJECTIVE_CANDIDATE, index=0),
        resolved_observation_id="OBS-OBJ-1", system=_system("OBS-EVIDCLASS-1"),
    )

    service = ServiceObservation(
        source_evidence_id="E-SERVICE-1",
        source_span=_span("A critical data-sync outage affecting reconciliation was never fully resolved."),
        basis=InferenceBasis.EXPLICIT, incident_or_condition="Unresolved critical data-sync outage",
        system=_system("OBS-SERVICE-1"),
    )
    risk_signal_service = CandidateRiskSignal(
        source_evidence_id="E-SERVICE-1", source_span=_span("critical data-sync outage affecting reconciliation was never fully resolved"),
        basis=InferenceBasis.INFERRED_CANDIDATE, mechanism="CR-02", proposed_severity_tier="MATERIAL",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.SERVICE_OBSERVATION, index=0),
        resolved_observation_id="OBS-SERVICE-1", system=_system("OBS-RISK-SERVICE"),
    )

    missing_renewal = MissingInformationCandidate(
        missing_item="renewal date", reviewed_evidence_ids=("E-CHAMPION-1", "E-VAL-1", "E-SERVICE-1"),
        system=_system("OBS-MISSING-RENEWAL"),
    )

    bystander = ExperienceObservation(
        source_evidence_id="E-INTRO-1", source_span=_span("Account was introduced by our partner referral team."),
        basis=InferenceBasis.EXPLICIT, statement="Account was introduced by our partner referral team.",
        system=_system("OBS-EXPERIENCE-BYSTANDER"),
    )

    extraction_result = ExtractionResult(
        accepted=(stakeholder, objective_candidate, service, missing_renewal, bystander),
        candidate_contradictions=(),
        candidate_risk_signals=(risk_signal_champion, risk_signal_service),
        candidate_evidence_classifications=(evidence_classification,),
        rejected=(), dedup_audit=(), traces=(),
    )

    dimensions_to_evaluate = (DimensionCode.D1, DimensionCode.D6)
    # Read-only D2/D6 stand-in (M3-OD-01 unresolved, no picker in this UI):
    # if/when OBS-STAKE-1 is confirmed, its evidence_state feeds this fixed
    # D6 qualifier -- a CSM can confirm the underlying FACT, but cannot
    # choose or edit which dimension/qualifier it maps to.
    dimension_qualifier_overrides = {"OBS-STAKE-1": (DimensionCode.D6, "CHAMPION_LOST_NO_SUCCESSOR")}

    return account, registry, extraction_result, dimensions_to_evaluate, dimension_qualifier_overrides
