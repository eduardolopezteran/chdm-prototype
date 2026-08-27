"""
Deterministic pipeline tests (spec §19) — NO network calls anywhere in
this file. Every provider used here is FakeExtractionProvider. Live-model
behavior is exercised only by tests/test_extraction_live_model.py
(Checkpoint 2B), which is a separate file precisely so CI can distinguish
deterministic-code failures from model/API variability (spec §19).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import EvidenceState, Provenance
from domain.evidence import EvidenceObject

from extraction.enums import InferenceBasis, RejectionReason
from extraction.pipeline import run_extraction
from extraction.provider import FakeExtractionProvider


def _evidence(evidence_id, text, source="account_note"):
    return EvidenceObject(evidence_id, None, text, source, Provenance.USER_PROVIDED)


def _span(text, in_text=None):
    # Milestone 2B baseline fix: the model-facing contract no longer
    # accepts start_char/end_char (see extraction/json_schemas.py) — the
    # application derives them from `text` via extraction.validation.
    # resolve_source_span. `in_text` is accepted for call-site
    # compatibility but is no longer used to compute offsets here.
    return {"text": text}


def test_sponsor_departure_no_replacement_spec_example():
    """Spec §4's own worked example, run through the full pipeline."""
    text = "Ana told us Roberto left the company in July. No replacement sponsor has been named."
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span("Roberto left the company in July", text),
             "basis": "EXPLICIT", "person_identifier": "Roberto", "continuity_event": "DEPARTED"},
            {"source_evidence_id": "E1", "source_span": _span("No replacement sponsor has been named", text),
             "basis": "EXPLICIT", "person_identifier": "replacement sponsor", "continuity_event": "NOT_IDENTIFIED"},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert len(result.accepted) == 2
    assert result.rejected == ()
    for obs in result.accepted:
        assert obs.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED
        assert obs.system.model_provider == "FakeExtractionProvider"
        assert obs.system.observation_id is not None
        assert obs.system.trace_id is not None
    assert len(result.traces) == 2


def test_current_unverified_creation_boundary():
    """Every accepted observation must be Current+Unverified — never
    Confirmed, regardless of the model's stated confidence/basis."""
    text = "Roberto left the company in July"
    e = _evidence("E1", text)
    raw = {"stakeholder_observations": [
        {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
         "person_identifier": "Roberto"},
    ]}
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert all(o.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED for o in result.accepted)


def test_explicit_vs_inferred_classification_preserved():
    text = "Roberto was our executive sponsor. He left the company last month."
    e = _evidence("E1", text)
    raw = {"stakeholder_observations": [
        {"source_evidence_id": "E1", "source_span": _span("Roberto was our executive sponsor", text),
         "basis": "EXPLICIT", "person_identifier": "Roberto", "sponsor_or_champion_relationship": "SPONSOR"},
        {"source_evidence_id": "E1", "source_span": _span("He left the company last month", text),
         "basis": "EXPLICIT", "person_identifier": "Roberto", "continuity_event": "DEPARTED"},
        {"source_evidence_id": "E1", "source_span": _span("He left the company last month", text),
         "basis": "INFERRED_CANDIDATE", "person_identifier": "Roberto",
         "continuity_event": "SPONSOR_CONTINUITY_POSSIBLY_UNRESOLVED"},
    ]}
    result = run_extraction((e,), FakeExtractionProvider(raw))
    bases = [o.basis for o in result.accepted]
    assert bases.count(InferenceBasis.EXPLICIT) == 2
    assert bases.count(InferenceBasis.INFERRED_CANDIDATE) == 1
    inferred = [o for o in result.accepted if o.basis == InferenceBasis.INFERRED_CANDIDATE][0]
    assert inferred.continuity_event == "SPONSOR_CONTINUITY_POSSIBLY_UNRESOLVED"


def test_candidate_contradiction_created_without_resolution():
    text_a = "The customer confirmed the expected reduction in close time was achieved."
    text_b = "Average close time remains unchanged from baseline."
    ea, eb = _evidence("EA", text_a, "qbr_note"), _evidence("EB", text_b, "ops_report")
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "reduce close time", "stated_outcome": "achieved per customer"},
        ],
        "adoption_observations": [
            {"source_evidence_id": "EB", "source_span": _span(text_b, text_b), "basis": "EXPLICIT",
             "workflow_or_use_case": "close time", "observed_behavior": "unchanged from baseline"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
             "conflict_description": "Customer states improvement; operational data says unchanged.",
             "methodology_construct_hint": "objective realized outcome"},
        ],
    }
    result = run_extraction((ea, eb), FakeExtractionProvider(raw))
    assert len(result.candidate_contradictions) == 1
    c = result.candidate_contradictions[0]
    assert c.status == "CANDIDATE"
    assert c.resolved_observation_id_a is not None and c.resolved_observation_id_b is not None
    assert c.resolved_observation_id_a != c.resolved_observation_id_b


def test_prohibited_field_rejection_counted():
    text = "Roberto left the company in July"
    e = _evidence("E1", text)
    raw = {"stakeholder_observations": [
        {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
         "person_identifier": "Roberto", "activated_severity": "CRITICAL"},
    ]}
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.accepted == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.BOUNDARY_VIOLATION


def test_malformed_output_one_retry_then_fails_cleanly():
    provider = FakeExtractionProvider([{"totally_bogus_key": 1}, {"still_bogus": 2}])
    e = _evidence("E1", "some text")
    result = run_extraction((e,), provider)
    assert provider.call_count == 2
    assert result.request_failure is not None
    assert result.accepted == ()


def test_malformed_output_retry_recovers():
    provider = FakeExtractionProvider([{"bogus": 1}, {"stakeholder_observations": []}])
    e = _evidence("E1", "some text")
    result = run_extraction((e,), provider)
    assert provider.call_count == 2
    assert result.request_failure is None
    assert result.accepted == ()


def test_model_service_failure_returns_without_changing_state():
    provider = FakeExtractionProvider({}, raise_service_error=True)
    e = _evidence("E1", "some text")
    result = run_extraction((e,), provider)
    assert result.request_failure is not None
    assert result.accepted == ()
    assert result.rejected == ()


def test_empty_extraction_is_valid():
    provider = FakeExtractionProvider({})
    e = _evidence("E1", "Called in to a bad connection, rescheduled for Thursday.")
    result = run_extraction((e,), provider)
    assert result.request_failure is None
    assert result.accepted == ()
    assert result.rejected == ()


def test_missing_information_candidate_end_to_end():
    e = _evidence("E1", "Weekly sync notes with no stated objective.")
    raw = {"missing_information_candidates": [
        {"missing_item": "customer_objective", "reviewed_evidence_ids": ["E1"]},
    ]}
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.accepted) == 1
    mi = result.accepted[0]
    assert mi.missing_item == "customer_objective"
    assert mi.reviewed_evidence_ids == ("E1",)


def test_contradiction_integrity_valid_two_distinct_observations():
    """Milestone 2B.2 closure, integrity condition set item 1: a
    contradiction referencing two genuinely distinct accepted
    observations must still be accepted, with both resolved IDs distinct
    -- the new integrity rule must not reject the legitimate case."""
    text_a = "The customer confirmed the expected reduction in close time was achieved."
    text_b = "Average close time remains unchanged from baseline."
    ea, eb = _evidence("EA", text_a, "qbr_note"), _evidence("EB", text_b, "ops_report")
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "reduce close time", "stated_outcome": "achieved per customer"},
        ],
        "adoption_observations": [
            {"source_evidence_id": "EB", "source_span": _span(text_b, text_b), "basis": "EXPLICIT",
             "workflow_or_use_case": "close time", "observed_behavior": "unchanged from baseline"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
             "conflict_description": "Customer states improvement; operational data says unchanged.",
             "methodology_construct_hint": "objective realized outcome"},
        ],
    }
    result = run_extraction((ea, eb), FakeExtractionProvider(raw))
    assert len(result.candidate_contradictions) == 1
    c = result.candidate_contradictions[0]
    assert c.resolved_observation_id_a is not None and c.resolved_observation_id_b is not None
    assert c.resolved_observation_id_a != c.resolved_observation_id_b
    assert not any(r.reason == RejectionReason.CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE
                   for r in result.rejected)


def test_contradiction_integrity_same_id_a_b_rejected():
    """Milestone 2B.2 closure, integrity condition 3 (A != B). This is the
    exact defect a live Prompt v2 run exposed on case 12: the model
    referenced the SAME accepted observation for both observation_ref_a
    and observation_ref_b. Must be rejected, not silently retained as a
    one-sided "contradiction", and no fabricated second observation may
    be invented to paper over it."""
    text_a = "The customer confirmed the expected reduction in close time was achieved."
    ea = _evidence("EA", text_a, "qbr_note")
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "reduce close time", "stated_outcome": "achieved per customer"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "conflict_description": "Self-referential -- should never happen but must be caught.",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((ea,), FakeExtractionProvider(raw))
    assert result.candidate_contradictions == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].observation_type == "candidate_contradictions"
    assert result.rejected[0].reason == RejectionReason.CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE
    # The one legitimate observation must still be accepted on its own --
    # rejecting the contradiction must not remove the underlying observation.
    assert len(result.accepted) == 1


def test_contradiction_integrity_nonexistent_ref_a_rejected():
    """Milestone 2B.2 closure, integrity condition 1 (reference A must
    exist): observation_ref_a points to an index that was never
    produced."""
    text_b = "Average close time remains unchanged from baseline."
    eb = _evidence("EB", text_b, "ops_report")
    raw = {
        "adoption_observations": [
            {"source_evidence_id": "EB", "source_span": _span(text_b, text_b), "basis": "EXPLICIT",
             "workflow_or_use_case": "close time", "observed_behavior": "unchanged from baseline"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
             "conflict_description": "References a non-existent objective_candidates[0].",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((eb,), FakeExtractionProvider(raw))
    assert result.candidate_contradictions == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.CONTRADICTION_REFERENCES_REJECTED_ITEM
    assert len(result.accepted) == 1


def test_contradiction_integrity_nonexistent_ref_b_rejected():
    """Milestone 2B.2 closure, integrity condition 2 (reference B must
    exist): observation_ref_b points to an index that was never
    produced."""
    text_a = "The customer confirmed the expected reduction in close time was achieved."
    ea = _evidence("EA", text_a, "qbr_note")
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "reduce close time", "stated_outcome": "achieved per customer"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
             "conflict_description": "References a non-existent adoption_observations[0].",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((ea,), FakeExtractionProvider(raw))
    assert result.candidate_contradictions == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.CONTRADICTION_REFERENCES_REJECTED_ITEM
    assert len(result.accepted) == 1


def test_contradiction_integrity_rejected_observation_reference_rejected():
    """Milestone 2B.2 closure, integrity condition 4 (both observations
    must have survived extraction validation): a contradiction referencing
    an item that WAS present in the model's raw output but was itself
    rejected (here, for a boundary violation) must also be rejected --
    never silently retained by falling back to whatever survived."""
    text_a = "Roberto left the company in July"
    text_b = "Average close time remains unchanged from baseline."
    ea, eb = _evidence("EA", text_a), _evidence("EB", text_b, "ops_report")
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "person_identifier": "Roberto", "activated_severity": "CRITICAL"},  # prohibited key -> rejected
        ],
        "adoption_observations": [
            {"source_evidence_id": "EB", "source_span": _span(text_b, text_b), "basis": "EXPLICIT",
             "workflow_or_use_case": "close time", "observed_behavior": "unchanged from baseline"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
             "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
             "conflict_description": "References an item that failed boundary validation.",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((ea, eb), FakeExtractionProvider(raw))
    assert result.candidate_contradictions == ()
    reasons = [r.reason for r in result.rejected]
    assert RejectionReason.BOUNDARY_VIOLATION in reasons
    assert RejectionReason.CONTRADICTION_REFERENCES_REJECTED_ITEM in reasons
    # The valid adoption_observations[0] item must still be accepted.
    assert len(result.accepted) == 1


def test_contradiction_integrity_dedup_collapse_rejected():
    """Milestone 2B.2 closure: the second, structurally distinct way the
    same integrity rule can be violated -- observation_ref_a and
    observation_ref_b start out pointing at two DIFFERENT raw items (so
    the direct key_a == key_b check does not fire), but those two items
    are near-duplicates of each other (same evidence, overlapping span,
    same normalized objective_text) and legitimately collapse into one
    canonical observation during deduplication. The resulting
    contradiction would reference one real observation twice and must be
    rejected exactly like the direct same-index case."""
    text_a = "The customer confirmed the expected reduction in close time was achieved."
    ea = _evidence("EA", text_a, "qbr_note")
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "reduce close time", "stated_outcome": "achieved per customer"},
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "REDUCE CLOSE TIME", "stated_outcome": "achieved, confirmed again"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 1},
             "conflict_description": "References two items that dedup will collapse into one.",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((ea,), FakeExtractionProvider(raw))
    # Sanity check: dedup really did collapse the two near-duplicates,
    # otherwise this test isn't exercising the code path it claims to.
    assert len(result.accepted) == 1
    assert len(result.dedup_audit) == 1
    assert result.candidate_contradictions == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE


def test_contradiction_referencing_candidate_evidence_classification_rejected_not_crashed():
    """Reproduces the exact live crash (Case 11, prompt_v4_optionb_2c_eval1
    run): a candidate_contradictions entry whose observation_ref_b cites
    CANDIDATE_EVIDENCE_CLASSIFICATION -- a Milestone 2C type that is NOT a
    resolvable reference target (enums.OBSERVATION_TYPE_TO_ARRAY_KEY
    deliberately excludes it). Before the fix, json_schemas.py's
    _OBSERVATION_REF_SCHEMA enumerated ALL of ObservationType, so this
    passed schema validation and then hit pipeline._resolve_array_ref()'s
    raw dict lookup, raising an uncaught KeyError instead of a graceful
    rejection. Must now be rejected as SCHEMA_INVALID at the shape-
    validation gate, before reference resolution ever runs, and must
    never crash run_extraction()."""
    text_a = "The customer confirmed the expected reduction in close time was achieved."
    ea = _evidence("EA", text_a, "qbr_note")
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "objective_text": "reduce close time", "stated_outcome": "achieved per customer"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
             "observation_ref_b": {"observation_type": "CANDIDATE_EVIDENCE_CLASSIFICATION", "index": 0},
             "conflict_description": "Should be structurally impossible, not a crash.",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((ea,), FakeExtractionProvider(raw))  # must not raise
    assert result.request_failure is None
    assert result.candidate_contradictions == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].observation_type == "candidate_contradictions"
    assert result.rejected[0].reason == RejectionReason.SCHEMA_INVALID
    # The one legitimate observation must still be accepted independently.
    assert len(result.accepted) == 1


def test_contradiction_referencing_candidate_risk_signal_rejected_not_crashed():
    """Same defect, other Milestone 2C type (CANDIDATE_RISK_SIGNAL) and
    other reference slot (observation_ref_a), for full coverage of the
    live failure's shape."""
    text_b = "Average close time remains unchanged from baseline."
    eb = _evidence("EB", text_b, "ops_report")
    raw = {
        "adoption_observations": [
            {"source_evidence_id": "EB", "source_span": _span(text_b, text_b), "basis": "EXPLICIT",
             "workflow_or_use_case": "close time", "observed_behavior": "unchanged from baseline"},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "CANDIDATE_RISK_SIGNAL", "index": 0},
             "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
             "conflict_description": "Should be structurally impossible, not a crash.",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((eb,), FakeExtractionProvider(raw))  # must not raise
    assert result.request_failure is None
    assert result.candidate_contradictions == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.SCHEMA_INVALID
    assert len(result.accepted) == 1


def test_contradiction_can_reference_missing_information_candidate():
    """Deliberately-supported behavior (predates this fix; see
    RejectionReason.CONTRADICTION_OBSERVATION_NOT_TRACEABLE_TO_EVIDENCE's
    docstring and pipeline._source_ref()'s reviewed_evidence_ids fallback,
    both written specifically for this case) that the schema-tightening
    fix above must NOT break: a contradiction CAN validly reference a
    MissingInformationCandidate as one of its two sides -- e.g. "we found
    no stated objective anywhere reviewed" apparently conflicting with a
    separate fact implying one exists. No prior test exercised this
    end-to-end; added here alongside the crash-reproduction tests so the
    positive and negative sides of the same boundary are both proven."""
    text_a = "The rollout plan references hitting a 20% efficiency target."
    ea = _evidence("EA", text_a, "planning_doc")
    raw = {
        "strategic_observations": [
            {"source_evidence_id": "EA", "source_span": _span(text_a, text_a), "basis": "EXPLICIT",
             "event": "rollout plan references a 20% efficiency target"},
        ],
        "missing_information_candidates": [
            {"missing_item": "customer_objective", "reviewed_evidence_ids": ["EA"]},
        ],
        "candidate_contradictions": [
            {"observation_ref_a": {"observation_type": "STRATEGIC_OBSERVATION", "index": 0},
             "observation_ref_b": {"observation_type": "MISSING_INFORMATION_CANDIDATE", "index": 0},
             "conflict_description": "Plan implies a stated objective, but no objective was ever explicitly captured.",
             "methodology_construct_hint": None},
        ],
    }
    result = run_extraction((ea,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert not any(r.observation_type == "candidate_contradictions" for r in result.rejected)
    assert len(result.candidate_contradictions) == 1
    c = result.candidate_contradictions[0]
    assert c.resolved_observation_id_a is not None
    assert c.resolved_observation_id_b is not None


def test_resolve_array_ref_raises_controlled_exception_not_keyerror():
    """Unit-tests pipeline._resolve_array_ref()'s defense-in-depth
    directly: given an observation_type with no resolvable array key (a
    Milestone 2C candidate-classification type), it must raise the
    internal, caught-by-both-call-sites _UnsupportedReferenceType signal
    -- never a raw KeyError. Both call sites (candidate_contradictions
    resolution and _build_candidate_classification) are already proven
    end-to-end above; this proves the shared primitive itself is correct
    even in isolation, exactly like test_source_ref_reports_unknown_
    source_for_untraceable_observation does for _source_ref."""
    from extraction.pipeline import _resolve_array_ref, _UnsupportedReferenceType

    for bad_type in ("CANDIDATE_RISK_SIGNAL", "CANDIDATE_EVIDENCE_CLASSIFICATION"):
        try:
            _resolve_array_ref({"observation_type": bad_type, "index": 0})
            assert False, f"must raise _UnsupportedReferenceType for {bad_type!r}, not resolve silently"
        except _UnsupportedReferenceType as e:
            assert e.obs_type.value == bad_type
        except KeyError:
            assert False, f"must never raise a raw KeyError for {bad_type!r}"

    # Sanity check the happy path still works for a real resolvable type.
    assert _resolve_array_ref({"observation_type": "OBJECTIVE_CANDIDATE", "index": 3}) == ("objective_candidates", 3)


def test_source_ref_reports_unknown_source_for_untraceable_observation():
    """Milestone 2B.2 closure, integrity condition 5 (both observations
    must remain traceable to source evidence). Every one of the 8 real
    accepted-observation dataclasses carries source_evidence_id or
    reviewed_evidence_ids by construction, so the pipeline-level
    CONTRADICTION_OBSERVATION_NOT_TRACEABLE_TO_EVIDENCE branch cannot be
    reached through a real end-to-end run -- this unit-tests the
    traceability primitive (`pipeline._source_ref`) directly against a
    minimal stand-in object that deliberately has neither field, so the
    guard itself is proven correct even though the real dataclasses never
    trigger it."""
    from extraction.pipeline import _source_ref

    class _NoTraceabilityFields:
        pass

    assert _source_ref(_NoTraceabilityFields()) == "UNKNOWN_SOURCE"

    class _HasEvidenceId:
        source_evidence_id = "E1"

    assert _source_ref(_HasEvidenceId()) == "E1"


def test_traceability_completeness():
    text = "Roberto left the company in July"
    e = _evidence("E1", text)
    raw = {"stakeholder_observations": [
        {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
         "person_identifier": "Roberto"},
    ]}
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.traces) == len(result.accepted)
    trace = result.traces[0]
    obs = result.accepted[0]
    assert trace.subject_object_ref == obs.system.observation_id
    assert trace.trace_id == obs.system.trace_id
    assert "FakeExtractionProvider" in trace.model_version
    assert e.evidence_id in trace.chain
    assert trace.reason_code.governing_object_id  # must be present, non-empty


def test_candidate_risk_signal_end_to_end_derives_evidence_id_from_reference():
    """Milestone 2C implementation constraint 1: the candidate risk
    signal's source_evidence_id must come from the observation its
    supporting_observation_ref resolves to -- never from a model-stated
    field (there is no such field in the model-facing schema)."""
    text = "Marcus, our only point of contact, told us today he is leaving. No successor has been named."
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span("No successor has been named", text),
             "basis": "EXPLICIT", "person_identifier": "Marcus", "continuity_event": "NOT_IDENTIFIED"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span("No successor has been named", text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert result.rejected == ()
    assert len(result.accepted) == 1
    assert len(result.candidate_risk_signals) == 1
    crs = result.candidate_risk_signals[0]
    assert crs.mechanism == "CR-01"
    assert crs.proposed_severity_tier == "CRITICAL"
    assert crs.source_evidence_id == "E1"  # derived from the reference, matches the supporting observation
    assert crs.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    assert crs.system.observation_id is not None
    assert crs.resolved_observation_id == result.accepted[0].system.observation_id


def test_candidate_evidence_classification_end_to_end():
    text = "The customer's goal was faster close. Analytics show close time at 3.6 hours."
    e = _evidence("E1", text)
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "E1", "source_span": _span("The customer's goal was faster close", text),
             "basis": "EXPLICIT", "objective_text": "faster close"},
        ],
        "candidate_evidence_classifications": [
            {"source_span": _span("Analytics show close time at 3.6 hours", text), "basis": "EXPLICIT",
             "proposed_basis": "MEASURED_OPERATIONAL_EVIDENCE", "supports": "ACHIEVED",
             "supporting_observation_ref": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.rejected == ()
    assert len(result.candidate_evidence_classifications) == 1
    cec = result.candidate_evidence_classifications[0]
    assert cec.proposed_basis == "MEASURED_OPERATIONAL_EVIDENCE"
    assert cec.supports == "ACHIEVED"
    assert cec.source_evidence_id == "E1"
    assert cec.resolved_observation_id == result.accepted[0].system.observation_id


def test_candidate_risk_signal_accepted_excluded_from_accepted_field():
    """`accepted` must keep its original Milestone 2B contract (exactly
    the 8 positive/missing-information types) -- candidate risk signals
    are returned via their own dedicated field, even though both were
    validated/deduplicated in the same pipeline run."""
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus", "continuity_event": "NOT_IDENTIFIED"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.accepted) == 1
    assert not any(o.__class__.__name__ == "CandidateRiskSignal" for o in result.accepted)
    assert len(result.candidate_risk_signals) == 1


def test_candidate_risk_signal_nonexistent_reference_rejected():
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_risk_signals == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM


def test_candidate_risk_signal_reference_to_rejected_item_rejected():
    """A supporting_observation_ref pointing at an item that WAS present
    in the model's raw output but was itself rejected (boundary
    violation) must also be rejected -- never falls back to a different
    item or silently retains the classification."""
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus", "activated_severity": "CRITICAL"},  # prohibited key -> rejected
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_risk_signals == ()
    reasons = [r.reason for r in result.rejected]
    assert RejectionReason.BOUNDARY_VIOLATION in reasons
    assert RejectionReason.CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM in reasons
    assert result.accepted == ()


def test_candidate_risk_signal_model_supplied_source_evidence_id_rejected():
    """Milestone 2C implementation constraint 1: source_evidence_id is
    application-derived only. additionalProperties: false must reject a
    model attempting to supply it directly, exactly like
    test_validate_candidate_classification_shape_rejects_source_evidence_id
    but exercised through the full pipeline."""
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus"},
        ],
        "candidate_risk_signals": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_risk_signals == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.SCHEMA_INVALID
    assert len(result.accepted) == 1  # the underlying observation is unaffected


def test_candidate_risk_signal_deferred_mechanism_rejected():
    text = "The integration has some occasional hiccups."
    e = _evidence("E1", text)
    raw = {
        "service_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "incident_or_condition": "occasional hiccups"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-06", "proposed_severity_tier": "WATCH",
             "supporting_observation_ref": {"observation_type": "SERVICE_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_risk_signals == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.SCHEMA_INVALID


def test_candidate_risk_signal_disallowed_reference_type_rejected():
    """The supporting_observation_ref enum structurally prevents citing a
    MissingInformationCandidate (or any type outside the 7 semantic
    types) -- must fail schema validation, never resolve."""
    e = _evidence("E1", "Weekly sync notes with no stated objective.")
    raw = {
        "missing_information_candidates": [
            {"missing_item": "customer_objective", "reviewed_evidence_ids": ["E1"]},
        ],
        "candidate_risk_signals": [
            {"source_span": _span("Weekly sync notes", "Weekly sync notes with no stated objective."),
             "basis": "EXPLICIT", "mechanism": "CR-01", "proposed_severity_tier": "WATCH",
             "supporting_observation_ref": {"observation_type": "MISSING_INFORMATION_CANDIDATE", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_risk_signals == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.SCHEMA_INVALID
    assert len(result.accepted) == 1  # the missing-information candidate is unaffected


def test_candidate_evidence_classification_disallowed_reference_type_rejected():
    """Symmetric coverage: the same supporting_observation_ref enum
    restriction applies to CandidateEvidenceClassification, not just
    CandidateRiskSignal -- must fail schema validation, never resolve."""
    e = _evidence("E1", "Weekly sync notes with no stated objective.")
    raw = {
        "missing_information_candidates": [
            {"missing_item": "customer_objective", "reviewed_evidence_ids": ["E1"]},
        ],
        "candidate_evidence_classifications": [
            {"source_span": _span("Weekly sync notes", "Weekly sync notes with no stated objective."),
             "basis": "EXPLICIT", "proposed_basis": "PROXY_SUPPORTED", "supports": "PROGRESSING",
             "supporting_observation_ref": {"observation_type": "MISSING_INFORMATION_CANDIDATE", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_evidence_classifications == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.SCHEMA_INVALID
    assert len(result.accepted) == 1  # the missing-information candidate is unaffected


def test_candidate_risk_signal_boundary_violation_rejected():
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL", "activated_severity": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.candidate_risk_signals == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.BOUNDARY_VIOLATION


def test_candidate_risk_signal_reference_repointed_through_dedup():
    """The supporting observation the candidate risk signal cites is
    itself a near-duplicate that dedup collapses into a canonical
    survivor -- resolved_observation_id must point at the SURVIVOR's
    final id, not the original (possibly-collapsed) one, exactly
    mirroring how CandidateContradiction is re-pointed."""
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus", "continuity_event": "NOT_IDENTIFIED"},
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "MARCUS", "continuity_event": "NOT_IDENTIFIED"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             # references the SECOND (index 1) stakeholder observation, which dedup will collapse into the first
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 1}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.accepted) == 1  # the two near-duplicate stakeholder observations collapsed to one
    assert len(result.dedup_audit) == 1
    assert len(result.candidate_risk_signals) == 1
    assert result.candidate_risk_signals[0].resolved_observation_id == result.accepted[0].system.observation_id


def test_candidate_risk_signal_itself_deduplicated():
    """Two candidate risk signals proposing the same mechanism/tier from
    the same evidence/span are a genuine duplicate and must collapse,
    exactly like the 7 positive observation types."""
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.candidate_risk_signals) == 1
    assert len(result.dedup_audit) == 1


def test_candidate_classification_trace_records_created():
    text = "No successor has been named"
    e = _evidence("E1", text)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": _span(text, text), "basis": "EXPLICIT",
             "person_identifier": "Marcus"},
        ],
        "candidate_risk_signals": [
            {"source_span": _span(text, text), "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert len(result.traces) == 2  # the stakeholder observation + the candidate risk signal
    crs = result.candidate_risk_signals[0]
    trace = next(t for t in result.traces if t.subject_object_ref == crs.system.observation_id)
    assert trace.chain == (crs.source_evidence_id, crs.resolved_observation_id)


TESTS = [
    test_sponsor_departure_no_replacement_spec_example,
    test_current_unverified_creation_boundary,
    test_explicit_vs_inferred_classification_preserved,
    test_candidate_contradiction_created_without_resolution,
    test_prohibited_field_rejection_counted,
    test_malformed_output_one_retry_then_fails_cleanly,
    test_malformed_output_retry_recovers,
    test_model_service_failure_returns_without_changing_state,
    test_empty_extraction_is_valid,
    test_missing_information_candidate_end_to_end,
    test_contradiction_integrity_valid_two_distinct_observations,
    test_contradiction_integrity_same_id_a_b_rejected,
    test_contradiction_integrity_nonexistent_ref_a_rejected,
    test_contradiction_integrity_nonexistent_ref_b_rejected,
    test_contradiction_integrity_rejected_observation_reference_rejected,
    test_contradiction_integrity_dedup_collapse_rejected,
    test_contradiction_referencing_candidate_evidence_classification_rejected_not_crashed,
    test_contradiction_referencing_candidate_risk_signal_rejected_not_crashed,
    test_contradiction_can_reference_missing_information_candidate,
    test_resolve_array_ref_raises_controlled_exception_not_keyerror,
    test_source_ref_reports_unknown_source_for_untraceable_observation,
    test_traceability_completeness,
    test_candidate_risk_signal_end_to_end_derives_evidence_id_from_reference,
    test_candidate_evidence_classification_end_to_end,
    test_candidate_risk_signal_accepted_excluded_from_accepted_field,
    test_candidate_risk_signal_nonexistent_reference_rejected,
    test_candidate_risk_signal_reference_to_rejected_item_rejected,
    test_candidate_risk_signal_model_supplied_source_evidence_id_rejected,
    test_candidate_risk_signal_deferred_mechanism_rejected,
    test_candidate_risk_signal_disallowed_reference_type_rejected,
    test_candidate_evidence_classification_disallowed_reference_type_rejected,
    test_candidate_risk_signal_boundary_violation_rejected,
    test_candidate_risk_signal_reference_repointed_through_dedup,
    test_candidate_risk_signal_itself_deduplicated,
    test_candidate_classification_trace_records_created,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
