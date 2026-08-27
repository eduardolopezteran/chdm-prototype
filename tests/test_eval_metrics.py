"""
Milestone 2B.2 closure — deterministic tests for eval/metrics.py.

NO network calls anywhere in this file. Uses FakeExtractionProvider to
produce real, pipeline-finalized ExtractionResult.accepted observations
(same as tests/test_extraction_pipeline_fake_provider.py), then feeds
them through eval.metrics.classify_accepted_observations directly --
this is the actual code path eval/run_eval.py exercises, so a test
passing here is a real guarantee about live-run scoring behavior, not
just a unit-level guarantee about metrics.py in isolation.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import DimensionCode, EvidenceState, Provenance
from domain.evidence import EvidenceObject

from extraction.enums import ObservationType, RejectionReason
from extraction.pipeline import (
    ExtractionValidationFailure, run_dimension_qualifier_classification, run_extraction,
)
from extraction.provider import FakeExtractionProvider
from extraction.schemas import (
    AtomicPredicateEvidence, CandidateDimensionQualifier, ExtractionSystemFields, InferenceBasis,
    ObservationRef, SourceSpan,
)

from eval.metrics import (
    _dimension_qualifier_span_matches, _span_matches,
    aggregate_metrics, classify_accepted_observations, classify_candidate_classifications,
    classify_dimension_qualifiers, dimension_qualifier_atomic_predicate_detail,
)
from eval.run_eval import score_case, select_cases


def _evidence(evidence_id, text, source="account_note"):
    return EvidenceObject(evidence_id, None, text, source, Provenance.USER_PROVIDED)


def _span(text):
    return {"text": text}


# A minimal case, structurally identical to eval/labeled_set.yaml's real
# case 21 (inferred-but-not-explicit objective from a support-ticket
# pattern): one EXPLICIT primary SERVICE_OBSERVATION expectation, plus an
# inferred_candidates_permitted entry for the (never-explicit) objective.
_CASE_21_LIKE = {
    "id": "test_21_like",
    "source_text": (
        "Over the last quarter, the customer has opened four support "
        "tickets asking whether their CRM data can be connected to our "
        "platform automatically."
    ),
    "expected_observations": [
        {
            "type": "SERVICE_OBSERVATION",
            "basis": "EXPLICIT",
            "role": "primary",
            "span_substrings": [
                "the customer has opened four support tickets asking whether "
                "their CRM data can be connected to our platform automatically"
            ],
        },
    ],
    "inferred_candidates_permitted": [
        {
            "text": "the customer wants their CRM data connected to the platform automatically",
            "basis": "INFERRED_CANDIDATE",
            "type": "OBJECTIVE_CANDIDATE",
            "span_substrings": [
                "Over the last quarter, the customer has opened four support "
                "tickets asking whether their CRM data can be connected to "
                "our platform automatically"
            ],
        },
    ],
}


def test_permitted_inferred_candidate_not_scored_as_wrong_type():
    """Milestone 2B.2 closure: the exact case-21 defect from
    prompt_v2_ontology_eval1 -- the model emits a basis-correct
    INFERRED_CANDIDATE ObjectiveCandidate matching a permitted entry, but
    its span overlaps (by containment) the SERVICE_OBSERVATION slot's
    span_substrings. Before the fix, this fell through to the wrong-type
    fallback and scored WRONG_TYPE purely because it had no normal slot
    of its own. After the fix, it must be credited as VALID_UNLABELED
    with matched_inferred_permitted=True, and must NOT consume/claim the
    unrelated SERVICE_OBSERVATION slot."""
    e = _evidence("E1", _CASE_21_LIKE["source_text"])
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "E1",
             "source_span": _span("asking whether their CRM data can be connected to our platform automatically"),
             "basis": "INFERRED_CANDIDATE", "objective_text": "Enable automatic CRM data connection"},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert len(result.accepted) == 1

    classification = classify_accepted_observations(_CASE_21_LIKE, result.accepted)
    entries = classification["classified"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["classification"] == "VALID_UNLABELED"
    assert entry["matched_inferred_permitted"] is True
    assert entry["classification"] != "WRONG_TYPE"

    # The SERVICE_OBSERVATION slot must remain unclaimed -- crediting the
    # inferred candidate must not accidentally satisfy an unrelated slot.
    slots = classification["slots"]
    assert len(slots) == 1
    assert slots[0]["claimed"] is False


def test_explicit_basis_for_permitted_entry_still_scores_ordinarily():
    """The fix must be narrowly scoped to INFERRED_CANDIDATE matches
    only. If the model instead emits an EXPLICIT-basis ObjectiveCandidate
    for the same span (an actual OC-01 boundary violation -- promoting an
    implied purpose to EXPLICIT), it must NOT be waved through by the
    inferred_candidates_permitted cross-reference. It should fall through
    to ordinary wrong-type/unlabeled scoring exactly as before this fix."""
    e = _evidence("E1", _CASE_21_LIKE["source_text"])
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "E1",
             "source_span": _span("asking whether their CRM data can be connected to our platform automatically"),
             "basis": "EXPLICIT", "objective_text": "Enable automatic CRM data connection"},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    classification = classify_accepted_observations(_CASE_21_LIKE, result.accepted)
    entry = classification["classified"][0]
    assert entry["matched_inferred_permitted"] is False
    # Falls through to the wrong-type fallback against the unclaimed
    # SERVICE_OBSERVATION slot, same as before this fix.
    assert entry["classification"] == "WRONG_TYPE"


def test_second_inferred_observation_does_not_double_claim_permitted_entry():
    """A permitted inferred-candidate entry can only be claimed once --
    mirrors the existing `slots[*]["claimed"]` pattern so a duplicate
    can't be silently credited twice."""
    e = _evidence("E1", _CASE_21_LIKE["source_text"])
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "E1",
             "source_span": _span("asking whether their CRM data can be connected to our platform automatically"),
             "basis": "INFERRED_CANDIDATE", "objective_text": "Enable automatic CRM data connection"},
            {"source_evidence_id": "E1",
             "source_span": _span("Over the last quarter, the customer has opened four support tickets"),
             "basis": "INFERRED_CANDIDATE", "objective_text": "Enable automatic CRM data connection, restated"},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.accepted) == 2
    classification = classify_accepted_observations(_CASE_21_LIKE, result.accepted)
    matched = [c for c in classification["classified"] if c["matched_inferred_permitted"]]
    assert len(matched) == 1
    # The second one must NOT also be silently VALID_UNLABELED-via-
    # inferred-permitted; it falls through to whatever the existing
    # (unchanged) rules give it.
    other = [c for c in classification["classified"] if not c["matched_inferred_permitted"]][0]
    assert other["classification"] in ("WRONG_TYPE", "VALID_UNLABELED", "DUPLICATE")


# Milestone 2C: a minimal case, structurally identical to
# eval/labeled_set.yaml's real case 24 (CR-01 sponsor departure).
_CASE_24_LIKE = {
    "id": "test_24_like",
    "source_text": "Marcus told us today he is leaving. No successor has been named.",
    "expected_observations": [
        {
            "type": "STAKEHOLDER_OBSERVATION",
            "basis": "EXPLICIT",
            "role": "primary",
            "span_substrings": ["No successor has been named"],
        },
    ],
    "expected_candidate_risk_signals": [
        {
            "mechanism": "CR-01",
            "proposed_severity_tier": "CRITICAL",
            "role": "primary",
            "span_substrings": ["No successor has been named"],
            "supporting_observation_span_substrings": ["No successor has been named"],
        },
    ],
    "expected_candidate_evidence_classifications": [],
}


def _run_case_24_like(mechanism="CR-01", proposed_severity_tier="CRITICAL", ref_index=0):
    e = _evidence("E1", _CASE_24_LIKE["source_text"])
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span("No successor has been named"),
             "basis": "EXPLICIT", "person_identifier": "Marcus", "continuity_event": "NOT_IDENTIFIED"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span("No successor has been named"), "basis": "EXPLICIT",
             "mechanism": mechanism, "proposed_severity_tier": proposed_severity_tier,
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": ref_index}},
        ],
    }
    return run_extraction((e,), FakeExtractionProvider(raw))


def test_candidate_risk_signal_matched_with_correct_mechanism_and_tier():
    result = _run_case_24_like()
    cc = classify_candidate_classifications(
        _CASE_24_LIKE, result.accepted, result.candidate_risk_signals, result.candidate_evidence_classifications,
    )
    entries = cc["risk_signals"]["classified"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["classification"] == "MATCHED_EXPECTED"
    assert entry["mechanism_correct"] is True
    assert entry["proposed_severity_tier_correct"] is True
    assert entry["reference_correct"] is True
    assert cc["risk_signals"]["slots"][0]["claimed"] is True


def test_candidate_risk_signal_matched_but_wrong_tier_flagged():
    """A span match against the right expected fact, but the WRONG
    proposed_severity_tier, must still count as MATCHED_EXPECTED (the
    extractor found the right fact) while flagging the tier as
    incorrect -- never silently credited as fully correct."""
    result = _run_case_24_like(proposed_severity_tier="WATCH")
    cc = classify_candidate_classifications(
        _CASE_24_LIKE, result.accepted, result.candidate_risk_signals, result.candidate_evidence_classifications,
    )
    entry = cc["risk_signals"]["classified"][0]
    assert entry["classification"] == "MATCHED_EXPECTED"
    assert entry["mechanism_correct"] is True
    assert entry["proposed_severity_tier_correct"] is False


def test_candidate_risk_signal_unexpected_candidate_flagged():
    """A candidate risk signal whose span matches NO expected slot at all
    (an unanticipated fabrication) must be flagged UNEXPECTED_CANDIDATE,
    never silently credited as a match."""
    e = _evidence("E1", "Nothing risk-relevant is stated here at all, just routine notes.")
    raw = {
        "service_observations": [
            {"source_evidence_id": "E1", "source_span": _span("routine notes"),
             "basis": "EXPLICIT", "incident_or_condition": "routine notes"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span("routine notes"), "basis": "EXPLICIT",
             "mechanism": "CR-02", "proposed_severity_tier": "WATCH",
             "supporting_observation_ref": {"observation_type": "SERVICE_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    empty_case = {"id": "test_negative", "source_text": "irrelevant",
                  "expected_observations": [], "expected_candidate_risk_signals": [],
                  "expected_candidate_evidence_classifications": []}
    cc = classify_candidate_classifications(
        empty_case, result.accepted, result.candidate_risk_signals, result.candidate_evidence_classifications,
    )
    assert len(cc["risk_signals"]["classified"]) == 1
    assert cc["risk_signals"]["classified"][0]["classification"] == "UNEXPECTED_CANDIDATE"


def test_aggregate_metrics_reports_candidate_risk_signal_rates_end_to_end():
    """Real code path: eval.run_eval.score_case -> eval.metrics.
    aggregate_metrics, exactly what a live run does. Confirms the
    Milestone 2C aggregate summary keys exist and compute sane rates from
    a single, fully-correct case."""
    provider = FakeExtractionProvider({
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span("No successor has been named"),
             "basis": "EXPLICIT", "person_identifier": "Marcus", "continuity_event": "NOT_IDENTIFIED"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span("No successor has been named"), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    })
    e = _evidence("E1", _CASE_24_LIKE["source_text"])
    calls_before = len(provider.call_log)
    result = run_extraction((e,), provider)
    case_result = score_case(_CASE_24_LIKE, result, provider, calls_before)
    aggregate = aggregate_metrics([case_result])

    summary = aggregate["candidate_risk_signals"]
    assert summary["matched_expected"] == 1
    assert summary["unexpected_candidates"] == 0
    assert summary["mechanism_accuracy_rate"] == 1.0
    assert summary["tier_accuracy_rate"] == 1.0
    assert summary["reference_accuracy_rate"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["false_candidate_rate"] == 0.0

    empty_evidence_summary = aggregate["candidate_evidence_classifications"]
    assert empty_evidence_summary["expected_total"] == 0
    assert empty_evidence_summary["matched_expected"] == 0


# Milestone 4B: a minimal case, structurally identical to
# eval/labeled_set.yaml's real case 37 (WORKFLOWS_NOT_OCCURRING).
_CASE_37_LIKE = {
    "id": "test_37_like",
    "source_text": "The customer's team has not run the quarterly reconciliation workflow at all this year.",
    "expected_observations": [
        {
            "type": "ADOPTION_OBSERVATION",
            "basis": "EXPLICIT",
            "role": "primary",
            "span_substrings": ["has not run the quarterly reconciliation workflow at all this year"],
        },
    ],
    "expected_dimension_d2_qualifiers": [
        {
            "qualifier": "WORKFLOWS_NOT_OCCURRING",
            "role": "primary",
            "supporting_observation_span_substrings": ["has not run the quarterly reconciliation workflow at all this year"],
        },
    ],
    "expected_dimension_d6_qualifiers": [],
}


def _run_case_37_like(qualifier="WORKFLOWS_NOT_OCCURRING", ref_index=0):
    e = _evidence("E1", _CASE_37_LIKE["source_text"])
    stage1_raw = {
        "adoption_observations": [
            {"source_evidence_id": "E1",
             "source_span": _span("has not run the quarterly reconciliation workflow at all this year"),
             "basis": "EXPLICIT", "workflow_or_use_case": "reconciliation", "observed_behavior": "not run"},
        ],
    }
    provider1 = FakeExtractionProvider(stage1_raw)
    result1 = run_extraction((e,), provider1)
    dq_raw = {
        "candidate_d2_qualifiers": [
            {"qualifier": qualifier, "basis": "EXPLICIT",
             "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": ref_index}},
        ],
    }
    provider2 = FakeExtractionProvider(stage1_raw, dimension_qualifier_responses=dq_raw)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    return result2, provider2


def test_dimension_qualifier_matched_with_correct_qualifier():
    result, _ = _run_case_37_like()
    dqc = classify_dimension_qualifiers(
        _CASE_37_LIKE, result.accepted, result.candidate_d2_qualifiers, result.candidate_d6_qualifiers,
    )
    entries = dqc["d2_qualifiers"]["classified"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["classification"] == "MATCHED_EXPECTED"
    assert entry["qualifier_correct"] is True
    assert dqc["d2_qualifiers"]["slots"][0]["claimed"] is True
    assert dqc["d6_qualifiers"]["classified"] == []


def test_dimension_qualifier_matched_but_wrong_qualifier_flagged():
    """A proposal referencing the RIGHT supporting observation, but the
    WRONG qualifier value, must still count as MATCHED_EXPECTED (the
    classifier found the right observation to classify) while flagging
    the qualifier itself as incorrect -- never silently credited as fully
    correct, mirroring the 2C mechanism/tier disposition."""
    result, _ = _run_case_37_like(qualifier="NARROW_BREADTH_OR_CONCENTRATION")
    dqc = classify_dimension_qualifiers(
        _CASE_37_LIKE, result.accepted, result.candidate_d2_qualifiers, result.candidate_d6_qualifiers,
    )
    entry = dqc["d2_qualifiers"]["classified"][0]
    assert entry["classification"] == "MATCHED_EXPECTED"
    assert entry["qualifier_correct"] is False


def test_dimension_qualifier_unexpected_candidate_flagged():
    """A D2 qualifier proposed for an observation with NO matching
    expected slot at all (an unanticipated / non-force-fit violation)
    must be flagged UNEXPECTED_CANDIDATE, never silently credited."""
    e = _evidence("E1", "The reporting dashboard works exactly as designed, no issues at all.")
    stage1_raw = {
        "adoption_observations": [
            {"source_evidence_id": "E1", "source_span": _span("works exactly as designed"),
             "basis": "EXPLICIT", "workflow_or_use_case": "reporting", "observed_behavior": "works fine"},
        ],
    }
    provider1 = FakeExtractionProvider(stage1_raw)
    result1 = run_extraction((e,), provider1)
    dq_raw = {
        "candidate_d2_qualifiers": [
            {"qualifier": "INTENDED_WORKFLOWS_OPERATING_NORMALLY", "basis": "EXPLICIT",
             "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": 0}},
        ],
    }
    provider2 = FakeExtractionProvider(stage1_raw, dimension_qualifier_responses=dq_raw)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    empty_case = {
        "id": "test_negative", "source_text": "irrelevant", "expected_observations": [],
        "expected_dimension_d2_qualifiers": [], "expected_dimension_d6_qualifiers": [],
    }
    dqc = classify_dimension_qualifiers(
        empty_case, result2.accepted, result2.candidate_d2_qualifiers, result2.candidate_d6_qualifiers,
    )
    assert len(dqc["d2_qualifiers"]["classified"]) == 1
    assert dqc["d2_qualifiers"]["classified"][0]["classification"] == "UNEXPECTED_CANDIDATE"


def test_aggregate_metrics_reports_dimension_qualifier_rates_end_to_end():
    """Real code path: eval.run_eval.score_case -> eval.metrics.
    aggregate_metrics, exactly what a live run does for the stage-2
    classifier. Confirms the Milestone 4B aggregate summary keys exist
    and compute sane rates from a single, fully-correct case, and that
    dimension_qualifier_stage_failures is 0 when nothing failed."""
    result, provider = _run_case_37_like()
    e = _evidence("E1", _CASE_37_LIKE["source_text"])
    case_result = score_case(_CASE_37_LIKE, result, provider, 0, dq_calls_before=0)
    aggregate = aggregate_metrics([case_result])

    summary = aggregate["candidate_d2_qualifiers"]
    assert summary["matched_expected"] == 1
    assert summary["unexpected_candidates"] == 0
    assert summary["qualifier_accuracy_rate"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["false_candidate_rate"] == 0.0

    empty_d6_summary = aggregate["candidate_d6_qualifiers"]
    assert empty_d6_summary["expected_total"] == 0
    assert empty_d6_summary["matched_expected"] == 0

    assert aggregate["dimension_qualifier_stage_failures"] == 0


def test_aggregate_metrics_counts_dimension_qualifier_stage_failures():
    """A case where the stage-2 call itself fails must be counted in
    dimension_qualifier_stage_failures -- distinct from a case that ran
    successfully and simply proposed nothing (abstention, counted as a
    missed_candidate_count instead, never as a stage failure)."""
    e = _evidence("E1", _CASE_37_LIKE["source_text"])
    stage1_raw = {
        "adoption_observations": [
            {"source_evidence_id": "E1",
             "source_span": _span("has not run the quarterly reconciliation workflow at all this year"),
             "basis": "EXPLICIT", "workflow_or_use_case": "reconciliation", "observed_behavior": "not run"},
        ],
    }
    provider1 = FakeExtractionProvider(stage1_raw)
    result1 = run_extraction((e,), provider1)
    provider2 = FakeExtractionProvider(
        stage1_raw, dimension_qualifier_responses={}, raise_dimension_qualifier_service_error=True,
    )
    result2 = run_dimension_qualifier_classification(result1, provider2)
    case_result = score_case(_CASE_37_LIKE, result2, provider2, 0, dq_calls_before=0)
    aggregate = aggregate_metrics([case_result])
    assert aggregate["dimension_qualifier_stage_failures"] == 1
    assert case_result["dimension_qualifier_stage_failure"] is not None


# Milestone 4B isolated-classifier architecture checkpoint, item F
# (REQUIRED). The three deterministic cases explicitly named in the
# authorization: (a) complete cited observation -> match; (b) longer
# cited observation containing the expected supporting substring ->
# match; (c) short cited fragment merely contained INSIDE a longer
# expected substring -> no match (this is the reverse-containment
# direction that was removed -- it is the exact mechanism by which live
# cases 35/40 incorrectly certified a compound expected condition from a
# single, incomplete supporting observation).

def test_dimension_qualifier_span_matches_complete_cited_observation():
    """(a) The cited observation's span IS, verbatim, the full expected
    supporting substring -- straightforward exact match."""
    assert _dimension_qualifier_span_matches(
        "No successor has been named", ["No successor has been named"],
    ) is True


def test_dimension_qualifier_span_matches_longer_observation_contains_expected():
    """(b) The cited observation's span is LONGER than, and contains, the
    expected supporting substring -- still a match, since the full
    expected fact is genuinely present within what this one observation
    actually says."""
    assert _dimension_qualifier_span_matches(
        "Marcus told us today he is leaving and no successor has been named for his role",
        ["no successor has been named"],
    ) is True


def test_dimension_qualifier_span_matches_short_fragment_inside_longer_expected_rejected():
    """(c) The cited observation's span is a SHORT FRAGMENT that is
    merely contained INSIDE a longer expected substring -- must NOT
    match. This is the reverse-containment direction removed by item F:
    a single supporting observation covering only PART of a longer,
    compound expected condition must not be credited as if it supported
    the whole thing."""
    assert _dimension_qualifier_span_matches(
        "no successor has been named",
        ["Marcus told us today he is leaving and no successor has been named for his role"],
    ) is False


def test_dimension_qualifier_span_matches_differs_from_general_span_matches_on_reverse_direction():
    """Confirms the two matchers are deliberately DIFFERENT functions
    with different behavior on the same reverse-containment input --
    `_span_matches` (used by every other, non-D2/D6 call site) still
    accepts it, `_dimension_qualifier_span_matches` (D2/D6 only) does
    not. This pins the scope decision itself, not just one function's
    behavior."""
    short_observation = "no successor has been named"
    longer_expected = ["Marcus told us today he is leaving and no successor has been named for his role"]
    assert _span_matches(short_observation, longer_expected) is True
    assert _dimension_qualifier_span_matches(short_observation, longer_expected) is False


# ===========================================================================
# Milestone 4B v3 evaluator-provenance checkpoint --
# eval.metrics.dimension_qualifier_atomic_predicate_detail. Pure unit
# tests against hand-built AtomicPredicateEvidence / ExtractionValidation
# Failure / CandidateDimensionQualifier fixtures (no pipeline/provider
# call needed to test the merging logic itself), plus one end-to-end test
# through eval.run_eval.score_case proving the field is actually wired
# into the live-report shape. This checkpoint is READ-ONLY over already-
# produced pipeline output -- it does not change grounding, duplicate
# detection, or composition decisions, only whether they are visible in
# the eval report per predicate.
# ===========================================================================
_DQ_SPAN = SourceSpan(text="placeholder span text", start_char=0, end_char=len("placeholder span text"))


def _dq_system(obs_id: str) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="fake", model_version="fake-v1",
        extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )


def _predicate(predicate_id, dimension, obs_id, evidence_text="ev", basis=InferenceBasis.EXPLICIT):
    return AtomicPredicateEvidence(
        predicate_id=predicate_id, dimension=dimension, resolved_observation_id=obs_id,
        evidence_text=evidence_text, basis=basis,
    )


def _composed_candidate(dimension, qualifier, obs_id):
    ref_type = ObservationType.ADOPTION_OBSERVATION if dimension == DimensionCode.D2 else ObservationType.STAKEHOLDER_OBSERVATION
    return CandidateDimensionQualifier(
        dimension=dimension, qualifier=qualifier, basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=ObservationRef(observation_type=ref_type, index=0),
        source_evidence_id="E1", source_span=_DQ_SPAN, resolved_observation_id=obs_id,
        system=_dq_system(f"DIMQ-{obs_id}"),
    )


def test_atomic_predicate_detail_marks_full_grounded_set_as_composed():
    predicates = (
        _predicate("RELIABLE_AUTOMATION_OPERATION", DimensionCode.D2, "OBS-1"),
        _predicate("LOW_LOGIN_OR_MANUAL_ACTIVITY", DimensionCode.D2, "OBS-1"),
        _predicate("LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION", DimensionCode.D2, "OBS-1"),
    )
    composed = (_composed_candidate(DimensionCode.D2, "AUTOMATION_RELIABLE_LOW_LOGIN_OK", "OBS-1"),)
    detail = dimension_qualifier_atomic_predicate_detail(predicates, (), composed, ())

    assert len(detail) == 3
    for entry in detail:
        assert entry["grounding_passed"] is True
        assert entry["rejection_reason"] is None
        assert entry["composed"] is True
        assert entry["dimension"] == "D2"
        assert entry["resolved_observation_id"] == "OBS-1"


def test_atomic_predicate_detail_marks_incomplete_set_as_not_composed():
    """Two of three required D2 predicates -- grounded, but never
    composed, since the composer never produced a candidate for this
    observation. This is the exact "incomplete predicate sets" case the
    checkpoint required the evaluator to keep visible (never silently
    dropped just because nothing composed)."""
    predicates = (
        _predicate("RELIABLE_AUTOMATION_OPERATION", DimensionCode.D2, "OBS-2"),
        _predicate("LOW_LOGIN_OR_MANUAL_ACTIVITY", DimensionCode.D2, "OBS-2"),
    )
    detail = dimension_qualifier_atomic_predicate_detail(predicates, (), (), ())

    assert len(detail) == 2
    for entry in detail:
        assert entry["grounding_passed"] is True
        assert entry["composed"] is False


def test_atomic_predicate_detail_serializes_rejected_ungrounded_predicate():
    rejection = ExtractionValidationFailure(
        "candidate_d6_atomic_predicates",
        RejectionReason.DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED,
        "not grounded",
        {"predicate_id": "NO_SUCCESSOR_OR_CONTINUING_COVERAGE", "evidence_text": "fabricated text", "basis": "EXPLICIT"},
        resolved_observation_id="OBS-3", dimension=DimensionCode.D6,
    )
    detail = dimension_qualifier_atomic_predicate_detail((), (rejection,), (), ())

    assert len(detail) == 1
    entry = detail[0]
    assert entry["predicate_id"] == "NO_SUCCESSOR_OR_CONTINUING_COVERAGE"
    assert entry["evidence_text"] == "fabricated text"
    assert entry["dimension"] == "D6"
    assert entry["resolved_observation_id"] == "OBS-3"
    assert entry["grounding_passed"] is False
    assert entry["rejection_reason"] == "DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED"
    assert entry["composed"] is False


def test_atomic_predicate_detail_serializes_rejected_duplicate_predicate():
    rejection = ExtractionValidationFailure(
        "candidate_d2_atomic_predicates",
        RejectionReason.DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE,
        "duplicate",
        {"predicate_id": "RELIABLE_AUTOMATION_OPERATION", "evidence_text": "dup text", "basis": "EXPLICIT"},
        resolved_observation_id="OBS-4", dimension=DimensionCode.D2,
    )
    detail = dimension_qualifier_atomic_predicate_detail((), (rejection,), (), ())

    assert len(detail) == 1
    assert detail[0]["rejection_reason"] == "DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE"
    assert detail[0]["grounding_passed"] is False
    assert detail[0]["composed"] is False


def test_atomic_predicate_detail_ignores_unrelated_rejections():
    """A rejection with an unrelated observation_type (e.g. the ordinary
    simple-qualifier path) or an unrelated reason must never appear in
    this atomic-predicate-only view -- it is already covered by the
    existing generic rejected_detail report field."""
    unrelated_type = ExtractionValidationFailure(
        "candidate_d2_qualifiers", RejectionReason.SCHEMA_INVALID, "bad shape", {"qualifier": "X"},
    )
    unrelated_reason = ExtractionValidationFailure(
        "candidate_d2_atomic_predicates", RejectionReason.BOUNDARY_VIOLATION, "boundary",
        {"predicate_id": "RELIABLE_AUTOMATION_OPERATION"},
        resolved_observation_id="OBS-5", dimension=DimensionCode.D2,
    )
    detail = dimension_qualifier_atomic_predicate_detail((), (unrelated_type, unrelated_reason), (), ())
    assert detail == []


def test_atomic_predicate_detail_keeps_d2_and_d6_independent_in_same_run():
    """A D2 full set and a D6 full set in the SAME run must each be
    correctly flagged composed, without cross-dimension contamination --
    even if (hypothetically) they shared a resolved_observation_id, the
    (resolved_observation_id, dimension) pair keeps them distinct."""
    d2_predicates = (
        _predicate("RELIABLE_AUTOMATION_OPERATION", DimensionCode.D2, "OBS-SHARED"),
        _predicate("LOW_LOGIN_OR_MANUAL_ACTIVITY", DimensionCode.D2, "OBS-SHARED"),
        _predicate("LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION", DimensionCode.D2, "OBS-SHARED"),
    )
    d6_predicates = (
        _predicate("CONFIRMED_CHAMPION_DEPARTURE", DimensionCode.D6, "OBS-SHARED"),
        _predicate("NO_SUCCESSOR_OR_CONTINUING_COVERAGE", DimensionCode.D6, "OBS-SHARED"),
    )
    d2_composed = (_composed_candidate(DimensionCode.D2, "AUTOMATION_RELIABLE_LOW_LOGIN_OK", "OBS-SHARED"),)
    d6_composed = (_composed_candidate(DimensionCode.D6, "CHAMPION_LOST_NO_SUCCESSOR", "OBS-SHARED"),)

    detail = dimension_qualifier_atomic_predicate_detail(
        d2_predicates + d6_predicates, (), d2_composed, d6_composed,
    )
    assert len(detail) == 5
    for entry in detail:
        assert entry["composed"] is True
    d2_entries = [e for e in detail if e["dimension"] == "D2"]
    d6_entries = [e for e in detail if e["dimension"] == "D6"]
    assert len(d2_entries) == 3
    assert len(d6_entries) == 2


def test_atomic_predicate_detail_wired_end_to_end_via_score_case():
    """Proves task-level wiring, not just the pure function: a real
    pipeline run producing a full D2 atomic-predicate set must surface
    `dimension_qualifier_atomic_predicate_detail` inside eval.run_eval.
    score_case's returned per-case report."""
    adoption_text = (
        "The nightly sync integration has completed successfully every night for six months. "
        "Direct user logins to the reporting workflow are rare. "
        "That is expected because the sync integration is what actually performs the reporting workflow."
    )
    case = {
        "id": "test_atomic_predicate_wiring",
        "source_text": adoption_text,
        "expected_observations": [],
        "expected_dimension_d2_qualifiers": [],
        "expected_dimension_d6_qualifiers": [],
    }
    stage1 = {
        "adoption_observations": [
            {
                "source_evidence_id": "E1", "source_span": {"text": adoption_text}, "basis": "EXPLICIT",
                "workflow_or_use_case": "nightly reporting sync", "observed_behavior": "automated, rare logins",
            }
        ],
    }
    dq_response = {
        "candidate_d2_atomic_predicates": [
            {"predicate_id": "RELIABLE_AUTOMATION_OPERATION", "basis": "EXPLICIT",
             "evidence_text": "completed successfully every night for six months"},
            {"predicate_id": "LOW_LOGIN_OR_MANUAL_ACTIVITY", "basis": "EXPLICIT",
             "evidence_text": "Direct user logins to the reporting workflow are rare"},
            {"predicate_id": "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION", "basis": "EXPLICIT",
             "evidence_text": "the sync integration is what actually performs the reporting workflow"},
        ],
    }
    evidence = (EvidenceObject("E1", None, adoption_text, "synthetic_note", Provenance.USER_PROVIDED),)
    provider = FakeExtractionProvider(stage1, dimension_qualifier_responses=dq_response)
    calls_before = len(provider.call_log)
    result = run_extraction(evidence, provider)
    dq_calls_before = len(provider.dimension_qualifier_call_log)
    result = run_dimension_qualifier_classification(result, provider)

    report = score_case(case, result, provider, calls_before, dq_calls_before=dq_calls_before)

    assert "dimension_qualifier_atomic_predicate_detail" in report
    entries = report["dimension_qualifier_atomic_predicate_detail"]
    assert len(entries) == 3
    assert all(e["grounding_passed"] is True and e["composed"] is True for e in entries)
    assert {e["predicate_id"] for e in entries} == {
        "RELIABLE_AUTOMATION_OPERATION", "LOW_LOGIN_OR_MANUAL_ACTIVITY", "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION",
    }


# ===========================================================================
# Milestone 4B D2 atomic-predicate targeted live probe -- eval.run_eval.
# select_cases. Pure, network-free tests for the --case-ids diagnostic
# filter (never exercised by constructing a real AnthropicExtractionProvider).
# ===========================================================================
_SELECT_CASES_FIXTURE = [
    {"id": "01_clear_customer_objective"},
    {"id": "07_healthy_automated_workflow_low_human_activity"},
    {"id": "23_pure_adoption_milestone_no_goal_language"},
    {"id": "35_d2_qualifier_automation_reliable_low_login_ok"},
    {"id": "36_d2_qualifier_narrow_breadth_or_concentration"},
]


def test_select_cases_default_none_returns_everything_unfiltered():
    selected, prefixes = select_cases(_SELECT_CASES_FIXTURE, None)
    assert selected == _SELECT_CASES_FIXTURE
    assert prefixes is None


def test_select_cases_empty_string_returns_everything_unfiltered():
    selected, prefixes = select_cases(_SELECT_CASES_FIXTURE, "")
    assert selected == _SELECT_CASES_FIXTURE
    assert prefixes is None


def test_select_cases_filters_to_exactly_the_requested_prefixes():
    selected, prefixes = select_cases(_SELECT_CASES_FIXTURE, "07,23,35")
    assert [c["id"] for c in selected] == [
        "07_healthy_automated_workflow_low_human_activity",
        "23_pure_adoption_milestone_no_goal_language",
        "35_d2_qualifier_automation_reliable_low_login_ok",
    ]
    assert prefixes == ["07", "23", "35"]
    # Case 36 (a different prefix) and case 01 must NOT be selected.
    assert "36_d2_qualifier_narrow_breadth_or_concentration" not in [c["id"] for c in selected]
    assert "01_clear_customer_objective" not in [c["id"] for c in selected]


def test_select_cases_raises_when_no_case_matches():
    try:
        select_cases(_SELECT_CASES_FIXTURE, "99")
        assert False, "expected SystemExit for a prefix matching nothing"
    except SystemExit:
        pass


def test_select_cases_tolerates_whitespace_around_prefixes():
    selected, prefixes = select_cases(_SELECT_CASES_FIXTURE, " 07 , 23 ")
    assert [c["id"] for c in selected] == [
        "07_healthy_automated_workflow_low_human_activity",
        "23_pure_adoption_milestone_no_goal_language",
    ]
    assert prefixes == ["07", "23"]


def test_select_cases_zero_padded_prefix_deterministically_resolves_case_07():
    # Regression test for the exact real-world defect found via the live
    # prompt_v4_4b_dimqual_v3_atomic_probe1 run: "07" (correctly
    # zero-padded, matching how every case id in eval/labeled_set.yaml is
    # actually formatted) must resolve to exactly case 07 and nothing
    # else -- not case 01, not case 07's own longer id truncated, etc.
    selected, prefixes = select_cases(_SELECT_CASES_FIXTURE, "07")
    assert [c["id"] for c in selected] == ["07_healthy_automated_workflow_low_human_activity"]
    assert prefixes == ["07"]


def test_select_cases_raises_when_any_single_prefix_matches_nothing_even_if_others_match():
    # Regression test for the exact harness defect surfaced by the real
    # probe run: --case-ids "07,23,35" was actually invoked as "7,23,35"
    # (missing the leading zero). "7" matches ZERO case ids (all case ids
    # are zero-padded), while "23" and "35" both matched real cases. The
    # OLD select_cases silently returned a 2-case subset with no error --
    # Case 07 disappeared from what was supposed to be a 3-case targeted
    # probe without any signal anywhere in the run. This must now raise
    # SystemExit naming the unmatched prefix instead of silently
    # narrowing the run.
    try:
        select_cases(_SELECT_CASES_FIXTURE, "7,23,35")
        assert False, "expected SystemExit when one prefix in a multi-prefix filter matches nothing"
    except SystemExit as exc:
        message = str(exc)
        assert "zero-padded" in message
        # The message must name exactly the unmatched prefix ("7"), not
        # the ones that did match ("23", "35") -- confirms this is a
        # precise per-prefix diagnosis, not a generic "matched nothing".
        assert "['7']" in message
        assert "'23'" not in message and "'35'" not in message


TESTS = [
    test_dimension_qualifier_span_matches_complete_cited_observation,
    test_dimension_qualifier_span_matches_longer_observation_contains_expected,
    test_dimension_qualifier_span_matches_short_fragment_inside_longer_expected_rejected,
    test_dimension_qualifier_span_matches_differs_from_general_span_matches_on_reverse_direction,
    test_permitted_inferred_candidate_not_scored_as_wrong_type,
    test_explicit_basis_for_permitted_entry_still_scores_ordinarily,
    test_second_inferred_observation_does_not_double_claim_permitted_entry,
    test_candidate_risk_signal_matched_with_correct_mechanism_and_tier,
    test_candidate_risk_signal_matched_but_wrong_tier_flagged,
    test_candidate_risk_signal_unexpected_candidate_flagged,
    test_aggregate_metrics_reports_candidate_risk_signal_rates_end_to_end,
    test_dimension_qualifier_matched_with_correct_qualifier,
    test_dimension_qualifier_matched_but_wrong_qualifier_flagged,
    test_dimension_qualifier_unexpected_candidate_flagged,
    test_aggregate_metrics_reports_dimension_qualifier_rates_end_to_end,
    test_aggregate_metrics_counts_dimension_qualifier_stage_failures,
    test_atomic_predicate_detail_marks_full_grounded_set_as_composed,
    test_atomic_predicate_detail_marks_incomplete_set_as_not_composed,
    test_atomic_predicate_detail_serializes_rejected_ungrounded_predicate,
    test_atomic_predicate_detail_serializes_rejected_duplicate_predicate,
    test_atomic_predicate_detail_ignores_unrelated_rejections,
    test_atomic_predicate_detail_keeps_d2_and_d6_independent_in_same_run,
    test_atomic_predicate_detail_wired_end_to_end_via_score_case,
    test_select_cases_default_none_returns_everything_unfiltered,
    test_select_cases_empty_string_returns_everything_unfiltered,
    test_select_cases_filters_to_exactly_the_requested_prefixes,
    test_select_cases_raises_when_no_case_matches,
    test_select_cases_tolerates_whitespace_around_prefixes,
    test_select_cases_zero_padded_prefix_deterministically_resolves_case_07,
    test_select_cases_raises_when_any_single_prefix_matches_nothing_even_if_others_match,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
