import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import DimensionCode, EvidenceState

from extraction.enums import InferenceBasis, MissingInformationBasis
from extraction.schemas import (
    CandidateContradiction, CandidateDimensionQualifier, CandidateEvidenceClassification,
    CandidateRiskSignal, CommercialObservation, ExtractionSystemFields,
    MissingInformationCandidate, ObservationRef, ObjectiveCandidate, SourceSpan,
    StakeholderObservation,
)
from extraction.enums import ObservationType


def _span(text="Roberto left the company in July"):
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def test_source_span_valid():
    s = _span()
    assert s.end_char - s.start_char == len(s.text)


def test_source_span_rejects_negative_start():
    try:
        SourceSpan("x", -1, 0)
        assert False
    except ValueError:
        pass


def test_source_span_rejects_end_not_after_start():
    try:
        SourceSpan("x", 5, 5)
        assert False
    except ValueError:
        pass


def test_source_span_rejects_length_inconsistency():
    try:
        SourceSpan("abc", 0, 10)
        assert False
    except ValueError:
        pass


def test_system_fields_pending_not_populated():
    s = ExtractionSystemFields.pending()
    assert s.is_populated is False


def test_system_fields_rejects_confirmed_evidence_state():
    try:
        ExtractionSystemFields(evidence_state=EvidenceState.CURRENT_CONFIRMED)
        assert False, "must reject CURRENT_CONFIRMED"
    except ValueError:
        pass


def test_system_fields_accepts_unverified_and_stale():
    ExtractionSystemFields(evidence_state=EvidenceState.CURRENT_UNVERIFIED)
    ExtractionSystemFields(evidence_state=EvidenceState.STALE)


def test_stakeholder_observation_requires_person_identifier():
    try:
        StakeholderObservation("E1", _span(), InferenceBasis.EXPLICIT, person_identifier="")
        assert False
    except ValueError:
        pass


def test_stakeholder_observation_basic_construction():
    obs = StakeholderObservation("E1", _span(), InferenceBasis.EXPLICIT, person_identifier="Roberto",
                                  continuity_event="DEPARTED")
    assert obs.system.is_populated is False
    assert obs.basis == InferenceBasis.EXPLICIT


def test_commercial_observation_rejects_bad_event_type():
    try:
        CommercialObservation("E1", _span("renewal talk"), InferenceBasis.EXPLICIT,
                               event_type="not_a_real_type", description="x")
        assert False
    except ValueError:
        pass


def test_commercial_observation_accepts_allowed_event_type():
    CommercialObservation("E1", _span("renewal talk"), InferenceBasis.EXPLICIT,
                           event_type="renewal", description="x")


def test_missing_information_requires_reviewed_evidence_ids():
    try:
        MissingInformationCandidate("customer_objective", ())
        assert False
    except ValueError:
        pass


def test_missing_information_basis_is_fixed():
    mi = MissingInformationCandidate("customer_objective", ("E1",))
    assert mi.basis == MissingInformationBasis.NOT_FOUND_IN_REVIEWED_EVIDENCE


def test_missing_information_rejects_empty_item():
    try:
        MissingInformationCandidate("", ("E1",))
        assert False
    except ValueError:
        pass


def test_candidate_contradiction_status_must_be_candidate():
    ref_a = ObservationRef(ObservationType.OBJECTIVE_CANDIDATE, 0)
    ref_b = ObservationRef(ObservationType.ADOPTION_OBSERVATION, 0)
    c = CandidateContradiction(ref_a, ref_b, conflict_description="conflict")
    assert c.status == "CANDIDATE"
    try:
        CandidateContradiction(ref_a, ref_b, conflict_description="conflict", status="RESOLVED")
        assert False, "must reject any status other than CANDIDATE"
    except ValueError:
        pass


def test_observation_ref_rejects_negative_index():
    try:
        ObservationRef(ObservationType.OBJECTIVE_CANDIDATE, -1)
        assert False
    except ValueError:
        pass


def test_objective_candidate_requires_text():
    try:
        ObjectiveCandidate("E1", _span("reduce cost"), InferenceBasis.EXPLICIT, objective_text="")
        assert False
    except ValueError:
        pass


def _risk_ref():
    return ObservationRef(ObservationType.STAKEHOLDER_OBSERVATION, 0)


def test_candidate_risk_signal_accepts_mvp_mechanism_and_tier():
    obj = CandidateRiskSignal(
        source_evidence_id="E1", source_span=_span("Roberto left"), basis=InferenceBasis.EXPLICIT,
        mechanism="CR-01", proposed_severity_tier="CRITICAL", supporting_observation_ref=_risk_ref(),
    )
    assert obj.mechanism == "CR-01"
    assert obj.resolved_observation_id is None


def test_candidate_risk_signal_rejects_deferred_mechanism():
    # CR-03 added here under the PMO Option B decision (Milestone 2C MVP
    # scope reduction): it is now deferred from AI-automated candidate
    # classification the same way CR-04/05/06/07 already were, even
    # though (unlike those four) it remains fully implemented in the
    # deterministic engine. This test proves CR-03 is now rejected at
    # construction, not merely discouraged in prompt wording.
    for bad_mechanism in ("CR-03", "CR-04", "CR-05", "CR-06", "CR-07", "not-a-mechanism"):
        try:
            CandidateRiskSignal(
                source_evidence_id="E1", source_span=_span("Roberto left"), basis=InferenceBasis.EXPLICIT,
                mechanism=bad_mechanism, proposed_severity_tier="WATCH", supporting_observation_ref=_risk_ref(),
            )
            assert False, f"must reject deferred/unknown mechanism {bad_mechanism!r}"
        except ValueError:
            pass


def test_mvp_risk_mechanisms_are_exactly_cr01_cr02_cr08():
    # Structural guard for the PMO Option B decision: the AI-candidate-
    # classification MVP subset is exactly CR-01/CR-02/CR-08. CR-03 is
    # deliberately absent (deferred from automated classification only --
    # it remains fully governed deterministically in
    # registry/risk_mechanisms.yaml and engine/risk_engine.py, untouched
    # by this extraction-layer tuple).
    from extraction.schemas import _MVP_IMPLEMENTED_RISK_MECHANISMS
    assert _MVP_IMPLEMENTED_RISK_MECHANISMS == ("CR-01", "CR-02", "CR-08")
    assert "CR-03" not in _MVP_IMPLEMENTED_RISK_MECHANISMS


def test_candidate_risk_signal_rejects_resolved_tier():
    try:
        CandidateRiskSignal(
            source_evidence_id="E1", source_span=_span("Roberto left"), basis=InferenceBasis.EXPLICIT,
            mechanism="CR-01", proposed_severity_tier="RESOLVED", supporting_observation_ref=_risk_ref(),
        )
        assert False, "must reject RESOLVED — that is a lifecycle state, not a candidate tier"
    except ValueError:
        pass


def test_candidate_evidence_classification_accepts_mvp_basis_and_supports():
    obj = CandidateEvidenceClassification(
        source_evidence_id="E1", source_span=_span("resolution time at 3.6 hours"),
        basis=InferenceBasis.EXPLICIT, proposed_basis="MEASURED_OPERATIONAL_EVIDENCE",
        supports="ACHIEVED", supporting_observation_ref=_risk_ref(),
    )
    assert obj.proposed_basis == "MEASURED_OPERATIONAL_EVIDENCE"
    assert obj.resolved_observation_id is None


def test_candidate_evidence_classification_rejects_absence_bases():
    for bad_basis in ("UNVERIFIED_CLAIM", "INSUFFICIENT_EVIDENCE", "not-a-basis"):
        try:
            CandidateEvidenceClassification(
                source_evidence_id="E1", source_span=_span("resolution time at 3.6 hours"),
                basis=InferenceBasis.EXPLICIT, proposed_basis=bad_basis,
                supports="ACHIEVED", supporting_observation_ref=_risk_ref(),
            )
            assert False, f"must reject absence-describing basis {bad_basis!r}"
        except ValueError:
            pass


def test_candidate_evidence_classification_rejects_bad_supports():
    try:
        CandidateEvidenceClassification(
            source_evidence_id="E1", source_span=_span("resolution time at 3.6 hours"),
            basis=InferenceBasis.EXPLICIT, proposed_basis="MEASURED_OPERATIONAL_EVIDENCE",
            supports="MAYBE", supporting_observation_ref=_risk_ref(),
        )
        assert False
    except ValueError:
        pass


def test_candidate_classification_types_never_declare_evidence_state_as_confirmed():
    """Milestone 2C implementation constraint 1: evidence_state remains
    system-enforced. Mirrors test_system_fields_rejects_confirmed_
    evidence_state, applied via the same ExtractionSystemFields
    construction path both new types use."""
    try:
        CandidateRiskSignal(
            source_evidence_id="E1", source_span=_span("Roberto left"), basis=InferenceBasis.EXPLICIT,
            mechanism="CR-01", proposed_severity_tier="WATCH", supporting_observation_ref=_risk_ref(),
            system=ExtractionSystemFields(evidence_state=EvidenceState.CURRENT_CONFIRMED),
        )
        assert False, "must reject CURRENT_CONFIRMED on the shared ExtractionSystemFields path"
    except ValueError:
        pass


def _adoption_ref():
    return ObservationRef(ObservationType.ADOPTION_OBSERVATION, 0)


def _stakeholder_ref():
    return ObservationRef(ObservationType.STAKEHOLDER_OBSERVATION, 0)


def test_candidate_dimension_qualifier_accepts_valid_d2_qualifier():
    obj = CandidateDimensionQualifier(
        dimension=DimensionCode.D2, qualifier="WORKFLOWS_NOT_OCCURRING", basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=_adoption_ref(), source_evidence_id="E1", source_span=_span("no workflow"),
    )
    assert obj.dimension == DimensionCode.D2
    assert obj.qualifier == "WORKFLOWS_NOT_OCCURRING"


def test_candidate_dimension_qualifier_accepts_valid_d6_qualifier():
    obj = CandidateDimensionQualifier(
        dimension=DimensionCode.D6, qualifier="CHAMPION_LOST_NO_SUCCESSOR", basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=_stakeholder_ref(), source_evidence_id="E1", source_span=_span("champion left"),
    )
    assert obj.dimension == DimensionCode.D6
    assert obj.qualifier == "CHAMPION_LOST_NO_SUCCESSOR"


def test_candidate_dimension_qualifier_rejects_d6_qualifier_for_d2():
    """A D6-only qualifier value must never be accepted under dimension=D2
    -- the two qualifier vocabularies are structurally separated, not just
    validated by the model-facing schema (Milestone 4B approved
    architecture: 'each channel exposes only its canonical CHDM qualifier
    vocabulary')."""
    try:
        CandidateDimensionQualifier(
            dimension=DimensionCode.D2, qualifier="CHAMPION_LOST_NO_SUCCESSOR", basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=_adoption_ref(),
        )
        assert False, "must reject a D6 qualifier value under dimension=D2"
    except ValueError:
        pass


def test_candidate_dimension_qualifier_rejects_d2_qualifier_for_d6():
    try:
        CandidateDimensionQualifier(
            dimension=DimensionCode.D6, qualifier="WORKFLOWS_NOT_OCCURRING", basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=_stakeholder_ref(),
        )
        assert False, "must reject a D2 qualifier value under dimension=D6"
    except ValueError:
        pass


def test_candidate_dimension_qualifier_rejects_unimplemented_dimension():
    """Milestone 4B scope: only D2 and D6 have an implemented
    candidate-qualifier channel. Any other DimensionCode (e.g. D1, which
    IS implemented in the deterministic engine but has NO candidate-
    qualifier channel of its own) must be structurally rejected here."""
    try:
        CandidateDimensionQualifier(
            dimension=DimensionCode.D1, qualifier="WORKFLOWS_NOT_OCCURRING", basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=_adoption_ref(),
        )
        assert False, "must reject a dimension with no implemented candidate-qualifier channel"
    except ValueError:
        pass


def test_all_5_d2_qualifiers_are_individually_accepted():
    for q in (
        "INTENDED_WORKFLOWS_OPERATING_NORMALLY", "AUTOMATION_RELIABLE_LOW_LOGIN_OK",
        "NARROW_BREADTH_OR_CONCENTRATION", "WORKFLOWS_NOT_OCCURRING",
        "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED",
    ):
        obj = CandidateDimensionQualifier(
            dimension=DimensionCode.D2, qualifier=q, basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=_adoption_ref(),
        )
        assert obj.qualifier == q


def test_all_4_d6_qualifiers_are_individually_accepted():
    for q in (
        "APPROPRIATE_SPONSOR_COVERAGE", "CHAMPION_LOST_NO_SUCCESSOR",
        "CHAMPION_DEPARTURE_UNCONFIRMED", "SUCCESSION_UNCLEAR_OR_CONCENTRATED",
    ):
        obj = CandidateDimensionQualifier(
            dimension=DimensionCode.D6, qualifier=q, basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=_stakeholder_ref(),
        )
        assert obj.qualifier == q


def test_candidate_dimension_qualifier_never_declares_evidence_state_as_confirmed():
    """Mirrors test_candidate_classification_types_never_declare_evidence_
    state_as_confirmed above, applied to the new Milestone 4B type via the
    same shared ExtractionSystemFields construction path."""
    try:
        CandidateDimensionQualifier(
            dimension=DimensionCode.D2, qualifier="WORKFLOWS_NOT_OCCURRING", basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=_adoption_ref(),
            system=ExtractionSystemFields(evidence_state=EvidenceState.CURRENT_CONFIRMED),
        )
        assert False, "must reject CURRENT_CONFIRMED on the shared ExtractionSystemFields path"
    except ValueError:
        pass


def test_candidate_dimension_qualifier_permits_pending_grounding_fields():
    """Unlike CandidateRiskSignal/CandidateEvidenceClassification,
    CandidateDimensionQualifier's source_evidence_id/source_span are
    Optional and default to None -- the inherited-grounding exception
    means these are populated by the pipeline from the resolved
    supporting observation, not supplied at model-output-parse time the
    way stage-1 types are. A provisional (pre-inheritance) construction
    with both left at their defaults must not raise."""
    obj = CandidateDimensionQualifier(
        dimension=DimensionCode.D2, qualifier="WORKFLOWS_NOT_OCCURRING", basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=_adoption_ref(),
    )
    assert obj.source_evidence_id is None
    assert obj.source_span is None


TESTS = [
    test_source_span_valid, test_source_span_rejects_negative_start,
    test_source_span_rejects_end_not_after_start, test_source_span_rejects_length_inconsistency,
    test_system_fields_pending_not_populated, test_system_fields_rejects_confirmed_evidence_state,
    test_system_fields_accepts_unverified_and_stale,
    test_stakeholder_observation_requires_person_identifier, test_stakeholder_observation_basic_construction,
    test_commercial_observation_rejects_bad_event_type, test_commercial_observation_accepts_allowed_event_type,
    test_missing_information_requires_reviewed_evidence_ids, test_missing_information_basis_is_fixed,
    test_missing_information_rejects_empty_item,
    test_candidate_contradiction_status_must_be_candidate, test_observation_ref_rejects_negative_index,
    test_objective_candidate_requires_text,
    test_candidate_risk_signal_accepts_mvp_mechanism_and_tier,
    test_candidate_risk_signal_rejects_deferred_mechanism,
    test_mvp_risk_mechanisms_are_exactly_cr01_cr02_cr08,
    test_candidate_risk_signal_rejects_resolved_tier,
    test_candidate_evidence_classification_accepts_mvp_basis_and_supports,
    test_candidate_evidence_classification_rejects_absence_bases,
    test_candidate_evidence_classification_rejects_bad_supports,
    test_candidate_classification_types_never_declare_evidence_state_as_confirmed,
    test_candidate_dimension_qualifier_accepts_valid_d2_qualifier,
    test_candidate_dimension_qualifier_accepts_valid_d6_qualifier,
    test_candidate_dimension_qualifier_rejects_d6_qualifier_for_d2,
    test_candidate_dimension_qualifier_rejects_d2_qualifier_for_d6,
    test_candidate_dimension_qualifier_rejects_unimplemented_dimension,
    test_all_5_d2_qualifiers_are_individually_accepted,
    test_all_4_d6_qualifiers_are_individually_accepted,
    test_candidate_dimension_qualifier_never_declares_evidence_state_as_confirmed,
    test_candidate_dimension_qualifier_permits_pending_grounding_fields,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
