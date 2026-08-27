import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import DimensionCode, EvidenceState

from extraction.dedup import deduplicate
from extraction.enums import InferenceBasis, ObservationType
from extraction.schemas import (
    CandidateDimensionQualifier, CandidateEvidenceClassification, CandidateRiskSignal,
    ExtractionSystemFields, MissingInformationCandidate, ObservationRef, SourceSpan,
    StakeholderObservation,
)


def _finalized(person_identifier, continuity_event, evidence_id, start, end, text, obs_id):
    span = SourceSpan(text=text, start_char=start, end_char=end)
    obs = StakeholderObservation(evidence_id, span, InferenceBasis.EXPLICIT, person_identifier=person_identifier,
                                  continuity_event=continuity_event)
    system = ExtractionSystemFields(
        observation_id=obs_id, model_provider="fake", model_version="v1",
        extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )
    from dataclasses import replace
    return replace(obs, system=system)


def _finalized_mic(missing_item, reviewed_evidence_ids, obs_id):
    obs = MissingInformationCandidate(missing_item=missing_item, reviewed_evidence_ids=tuple(reviewed_evidence_ids))
    system = ExtractionSystemFields(
        observation_id=obs_id, model_provider="fake", model_version="v1",
        extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )
    from dataclasses import replace
    return replace(obs, system=system)


def test_exact_duplicate_collapses():
    text = "Roberto left the company in July"
    a = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-1")
    b = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 1
    assert canonical_map["OBS-1"] == "OBS-1"
    assert canonical_map["OBS-2"] == "OBS-1"
    assert len(audit) == 1
    assert audit[0].duplicate_observation_id == "OBS-2"
    assert audit[0].canonical_observation_id == "OBS-1"


def test_non_overlapping_mentions_both_retained():
    text = "Roberto (our sponsor) left the company in July. Roberto departed the org in July per HR."
    span1 = "Roberto (our sponsor) left the company in July"
    span2 = "Roberto departed the org in July"
    s1 = text.index(span1)
    s2 = text.index(span2)
    a = _finalized("Roberto", "DEPARTED", "E1", s1, s1 + len(span1), span1, "OBS-1")
    b = _finalized("Roberto", "DEPARTED", "E1", s2, s2 + len(span2), span2, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "non-overlapping spans must not be silently merged (spec §12)"
    assert audit == ()


def test_different_evidence_ids_never_dedup():
    text = "Roberto left the company in July"
    a = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-1")
    b = _finalized("Roberto", "DEPARTED", "E2", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2
    assert audit == ()


def test_materially_different_value_same_span_retained():
    text = "Roberto left the company in July"
    a = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-1")
    b = _finalized("Roberto", "ROLE_CHANGE", "E1", 0, len(text), text, "OBS-2")  # different claim, ambiguity -> retain both
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "materially different observations must never be silently merged (spec §12)"


def test_dedup_preserves_full_audit_trail_never_deletes_silently():
    text = "Roberto left the company in July"
    a = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-1")
    b = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-2")
    c = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-3")
    kept, canonical_map, audit = deduplicate((a, b, c))
    assert len(kept) == 1
    assert len(audit) == 2  # both OBS-2 and OBS-3 collapse into OBS-1, each with its own audit record
    assert {r.duplicate_observation_id for r in audit} == {"OBS-2", "OBS-3"}
    assert all(r.canonical_observation_id == "OBS-1" for r in audit)


def test_two_identical_missing_information_candidates_collapse():
    """Milestone 2B.2 closure fix: a live run crashed comparing two
    MissingInformationCandidate instances (AttributeError on
    source_evidence_id, which that type doesn't have). Two identical
    claims -- same missing_item, same reviewed_evidence_ids -- must
    collapse to one, exactly like any other exact duplicate."""
    a = _finalized_mic("customer_objective", ["E1"], "OBS-1")
    b = _finalized_mic("customer_objective", ["E1"], "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 1
    assert isinstance(kept[0], MissingInformationCandidate)
    assert canonical_map["OBS-2"] == "OBS-1"
    assert len(audit) == 1
    assert audit[0].observation_type == "MissingInformationCandidate"


def test_same_missing_item_identical_reviewed_scope_order_independent():
    """The same reviewed evidence scope must dedup regardless of the
    order reviewed_evidence_ids were listed in -- scope is a set, not a
    sequence."""
    a = _finalized_mic("value_confirmation", ["E1", "E2"], "OBS-1")
    b = _finalized_mic("value_confirmation", ["E2", "E1"], "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 1
    assert canonical_map["OBS-2"] == "OBS-1"


def test_same_missing_item_different_reviewed_scope_both_retained():
    """The same missing_item claimed against DIFFERENT reviewed evidence
    scopes is NOT the same claim -- collapsing them would silently
    discard which scope was actually reviewed (schemas.py: a
    MissingInformationCandidate must never claim information is absent
    globally, only within the specific evidence it reviewed). Both must
    be retained, exactly like spec §12's "if ambiguity exists, retain
    both" policy for span-grounded types."""
    a = _finalized_mic("customer_objective", ["E1"], "OBS-1")
    b = _finalized_mic("customer_objective", ["E2"], "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "different reviewed_evidence_ids scopes must never be silently merged"
    assert audit == ()


def test_missing_information_candidate_vs_span_grounded_observation_no_crash():
    """A MissingInformationCandidate compared against a normal
    span-grounded observation must never attempt to read
    source_evidence_id/source_span off the MissingInformationCandidate --
    the type check must short-circuit before either branch's
    type-specific attributes are ever touched. Must not raise, and both
    must be retained (they are never the same type, so never duplicates)."""
    mic = _finalized_mic("customer_objective", ["E1"], "OBS-1")
    text = "Roberto left the company in July"
    span_grounded = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((mic, span_grounded))
    assert len(kept) == 2
    assert audit == ()


def test_no_attribute_error_for_heterogeneous_observation_list():
    """A realistic mixed batch (span-grounded observations plus more than
    one MissingInformationCandidate, some duplicate, some not) must run
    through deduplicate() end-to-end without raising, and must produce
    the structurally correct kept/collapsed partition for every pair."""
    text = "Roberto left the company in July"
    s1 = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-1")
    s2 = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-2")  # dup of s1
    m1 = _finalized_mic("customer_objective", ["E1"], "OBS-3")
    m2 = _finalized_mic("customer_objective", ["E1"], "OBS-4")  # dup of m1
    m3 = _finalized_mic("customer_objective", ["E2"], "OBS-5")  # different scope, not a dup
    m4 = _finalized_mic("value_confirmation", ["E1"], "OBS-6")  # different missing_item, not a dup

    kept, canonical_map, audit = deduplicate((s1, s2, m1, m2, m3, m4))

    kept_ids = {o.system.observation_id for o in kept}
    assert kept_ids == {"OBS-1", "OBS-3", "OBS-5", "OBS-6"}
    assert canonical_map["OBS-2"] == "OBS-1"
    assert canonical_map["OBS-4"] == "OBS-3"
    assert {r.duplicate_observation_id for r in audit} == {"OBS-2", "OBS-4"}


def _finalized_crs(mechanism, tier, evidence_id, start, end, text, obs_id, ref_index=0):
    span = SourceSpan(text=text, start_char=start, end_char=end)
    obs = CandidateRiskSignal(
        source_evidence_id=evidence_id, source_span=span, basis=InferenceBasis.EXPLICIT,
        mechanism=mechanism, proposed_severity_tier=tier,
        supporting_observation_ref=ObservationRef(ObservationType.STAKEHOLDER_OBSERVATION, ref_index),
    )
    system = ExtractionSystemFields(
        observation_id=obs_id, model_provider="fake", model_version="v1",
        extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )
    from dataclasses import replace
    return replace(obs, system=system)


def _finalized_cec(proposed_basis, supports, evidence_id, start, end, text, obs_id):
    span = SourceSpan(text=text, start_char=start, end_char=end)
    obs = CandidateEvidenceClassification(
        source_evidence_id=evidence_id, source_span=span, basis=InferenceBasis.EXPLICIT,
        proposed_basis=proposed_basis, supports=supports,
        supporting_observation_ref=ObservationRef(ObservationType.OBJECTIVE_CANDIDATE, 0),
    )
    system = ExtractionSystemFields(
        observation_id=obs_id, model_provider="fake", model_version="v1",
        extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )
    from dataclasses import replace
    return replace(obs, system=system)


def test_candidate_risk_signal_exact_duplicate_collapses():
    text = "no successor has been named"
    a = _finalized_crs("CR-01", "CRITICAL", "E1", 0, len(text), text, "OBS-1")
    b = _finalized_crs("CR-01", "CRITICAL", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 1
    assert canonical_map["OBS-2"] == "OBS-1"
    assert len(audit) == 1


def test_candidate_risk_signal_different_tier_both_retained():
    text = "no successor has been named"
    a = _finalized_crs("CR-01", "CRITICAL", "E1", 0, len(text), text, "OBS-1")
    b = _finalized_crs("CR-01", "WATCH", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "different proposed_severity_tier is a materially different claim"


def test_candidate_risk_signal_different_mechanism_both_retained():
    text = "no successor has been named"
    a = _finalized_crs("CR-01", "CRITICAL", "E1", 0, len(text), text, "OBS-1")
    b = _finalized_crs("CR-02", "CRITICAL", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "different mechanism is a materially different claim"


def test_candidate_evidence_classification_exact_duplicate_collapses():
    text = "resolution time at 3.6 hours"
    a = _finalized_cec("MEASURED_OPERATIONAL_EVIDENCE", "ACHIEVED", "E1", 0, len(text), text, "OBS-1")
    b = _finalized_cec("MEASURED_OPERATIONAL_EVIDENCE", "ACHIEVED", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 1
    assert canonical_map["OBS-2"] == "OBS-1"


def test_candidate_evidence_classification_different_supports_both_retained():
    text = "resolution time at 3.6 hours"
    a = _finalized_cec("MEASURED_OPERATIONAL_EVIDENCE", "ACHIEVED", "E1", 0, len(text), text, "OBS-1")
    b = _finalized_cec("MEASURED_OPERATIONAL_EVIDENCE", "PROGRESSING", "E1", 0, len(text), text, "OBS-2")
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "different supports is a materially different claim"


def test_candidate_classification_types_never_collide_with_each_other_or_other_types():
    """CandidateRiskSignal, CandidateEvidenceClassification, and a plain
    StakeholderObservation must never be treated as duplicates of each
    other even when sharing evidence_id/span, mirroring
    test_missing_information_candidate_vs_span_grounded_observation_
    no_crash's type-isolation guarantee for the two Milestone 2C types."""
    text = "no successor has been named"
    crs = _finalized_crs("CR-01", "CRITICAL", "E1", 0, len(text), text, "OBS-1")
    cec = _finalized_cec("MEASURED_OPERATIONAL_EVIDENCE", "ACHIEVED", "E1", 0, len(text), text, "OBS-2")
    stakeholder = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-3")
    kept, canonical_map, audit = deduplicate((crs, cec, stakeholder))
    assert len(kept) == 3
    assert audit == ()


def _finalized_dq(dimension, qualifier, evidence_id, start, end, text, obs_id, ref_type, ref_index=0):
    span = SourceSpan(text=text, start_char=start, end_char=end)
    obs = CandidateDimensionQualifier(
        dimension=dimension, qualifier=qualifier, basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=ObservationRef(ref_type, ref_index),
        source_evidence_id=evidence_id, source_span=span, resolved_observation_id="OBS-SUPPORTING",
    )
    system = ExtractionSystemFields(
        observation_id=obs_id, model_provider="fake", model_version="v1",
        extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )
    from dataclasses import replace
    return replace(obs, system=system)


def test_dimension_qualifier_exact_duplicate_collapses():
    """Inherited grounding (Milestone 4B): two stage-2 proposals for the
    SAME supporting observation naturally share identical source_
    evidence_id/source_span, so the existing generic dedup machinery
    collapses them correctly with no special-casing."""
    text = "has not run the reconciliation workflow at all this year"
    a = _finalized_dq(
        DimensionCode.D2, "WORKFLOWS_NOT_OCCURRING", "E1", 0, len(text), text, "OBS-1",
        ObservationType.ADOPTION_OBSERVATION,
    )
    b = _finalized_dq(
        DimensionCode.D2, "WORKFLOWS_NOT_OCCURRING", "E1", 0, len(text), text, "OBS-2",
        ObservationType.ADOPTION_OBSERVATION,
    )
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 1
    assert canonical_map["OBS-2"] == "OBS-1"
    assert len(audit) == 1


def test_dimension_qualifier_different_qualifier_both_retained():
    text = "usage concentrated in one module only"
    a = _finalized_dq(
        DimensionCode.D2, "NARROW_BREADTH_OR_CONCENTRATION", "E1", 0, len(text), text, "OBS-1",
        ObservationType.ADOPTION_OBSERVATION,
    )
    b = _finalized_dq(
        DimensionCode.D2, "WORKFLOWS_NOT_OCCURRING", "E1", 0, len(text), text, "OBS-2",
        ObservationType.ADOPTION_OBSERVATION,
    )
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "different qualifier is a materially different claim"


def test_dimension_qualifier_different_dimension_both_retained():
    """D2 and D6 share ONE dataclass (approved architecture) -- must not
    collide as duplicates even if (hypothetically) they shared identical
    source_evidence_id/source_span, since `dimension` is part of
    _primary_value's matching tuple precisely to prevent this."""
    text = "shared span text for this defensive test only"
    a = _finalized_dq(
        DimensionCode.D2, "WORKFLOWS_NOT_OCCURRING", "E1", 0, len(text), text, "OBS-1",
        ObservationType.ADOPTION_OBSERVATION,
    )
    b = _finalized_dq(
        DimensionCode.D6, "CHAMPION_LOST_NO_SUCCESSOR", "E1", 0, len(text), text, "OBS-2",
        ObservationType.STAKEHOLDER_OBSERVATION,
    )
    kept, canonical_map, audit = deduplicate((a, b))
    assert len(kept) == 2, "different dimension is always a materially different claim"


def test_dimension_qualifier_never_collides_with_other_candidate_classification_types():
    text = "no successor has been named for the departed champion"
    dq = _finalized_dq(
        DimensionCode.D6, "CHAMPION_LOST_NO_SUCCESSOR", "E1", 0, len(text), text, "OBS-1",
        ObservationType.STAKEHOLDER_OBSERVATION,
    )
    crs = _finalized_crs("CR-01", "CRITICAL", "E1", 0, len(text), text, "OBS-2")
    stakeholder = _finalized("Roberto", "DEPARTED", "E1", 0, len(text), text, "OBS-3")
    kept, canonical_map, audit = deduplicate((dq, crs, stakeholder))
    assert len(kept) == 3
    assert audit == ()


TESTS = [
    test_exact_duplicate_collapses, test_non_overlapping_mentions_both_retained,
    test_different_evidence_ids_never_dedup, test_materially_different_value_same_span_retained,
    test_dedup_preserves_full_audit_trail_never_deletes_silently,
    test_two_identical_missing_information_candidates_collapse,
    test_same_missing_item_identical_reviewed_scope_order_independent,
    test_same_missing_item_different_reviewed_scope_both_retained,
    test_missing_information_candidate_vs_span_grounded_observation_no_crash,
    test_no_attribute_error_for_heterogeneous_observation_list,
    test_candidate_risk_signal_exact_duplicate_collapses,
    test_candidate_risk_signal_different_tier_both_retained,
    test_candidate_risk_signal_different_mechanism_both_retained,
    test_candidate_evidence_classification_exact_duplicate_collapses,
    test_candidate_evidence_classification_different_supports_both_retained,
    test_candidate_classification_types_never_collide_with_each_other_or_other_types,
    test_dimension_qualifier_exact_duplicate_collapses,
    test_dimension_qualifier_different_qualifier_both_retained,
    test_dimension_qualifier_different_dimension_both_retained,
    test_dimension_qualifier_never_collides_with_other_candidate_classification_types,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
