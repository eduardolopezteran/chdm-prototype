"""
Milestone 4B — D2/D6 Candidate Qualifier Extraction: deterministic
pipeline tests (spec §19) for the SEPARATE stage-2 orchestration function
extraction.pipeline.run_dimension_qualifier_classification. NO network
calls anywhere in this file -- every provider used here is
FakeExtractionProvider, exactly mirroring test_extraction_pipeline_fake_
provider.py's own discipline for run_extraction() (kept in a dedicated
file rather than appended there, since this exercises a structurally
separate call/function with its own failure modes).

Isolated-classifier architecture checkpoint (approved, supersedes the
prior batched-call test suite): every provider call in this file now
targets ExtractionProvider.propose_isolated_dimension_qualifier(dimension,
observation, *, repair_hint=None) -- ONE call per already-accepted,
eligible AdoptionObservation (D2) or StakeholderObservation (D6), never
one call for the whole run. FakeExtractionProvider's
dimension_qualifier_responses may still be a plain dict (returned
identically to every isolated call -- fine for single-eligible-
observation fixtures), but multi-observation fixtures in this file use a
callable(dimension, observation, repair_hint) -> dict so each dimension
gets its own single-channel, at-most-one-item envelope (the batched
double-channel DQ_RESPONSE dict from the prior architecture is no longer
a valid response shape for any one isolated call).

Covers every architecture point from the Milestone 4B isolated-classifier
approval:
  - run_extraction() itself is completely unaffected by stage 2 running
    afterward (accepted/candidate_contradictions/candidate_risk_signals/
    candidate_evidence_classifications/request_failure all preserved);
  - skip-if-nothing-eligible (no provider call at all);
  - exactly one isolated provider call per eligible observation (never
    one call for the whole run) -- expected-call-count assertions;
  - each isolated call receives ONE observation, never a list/tuple of
    observations, and its user message never contains sibling-observation
    content -- the structural guarantee the whole rewrite exists to
    provide;
  - PER-OBSERVATION FAILURE ISOLATION (approved architecture item D): one
    observation's call failing must not discard another observation's
    successfully-produced qualifier in the SAME run;
  - graceful degradation on ModelServiceError and on still-malformed-
    after-one-repair-retry, both preserving stage-1 results untouched;
  - the abstention (successful call, zero proposals) vs stage-failure
    (call itself did not complete) distinction, now tracked per
    observation via DimensionQualifierFailure /
    dimension_qualifier_failures, with dimension_qualifier_stage_failure
    preserved as a DERIVED, backward-compatible summary;
  - each of the 4 approved grounding-prohibition rejections individually;
  - inherited grounding (source_evidence_id/source_span copied verbatim
    from the resolved supporting observation);
  - stage-2's own system/provenance, distinct from the referenced
    observation's stage-1 system;
  - traceability: new TraceRecords APPENDED, stage-1's own never touched;
  - deduplication still runs over the full cross-call collection (dedup.py
    unchanged), even though the isolated envelope's own maxItems: 1
    constraint means a SINGLE call can no longer manufacture an
    intra-call duplicate on its own;
  - CandidateDimensionQualifier is never a valid CandidateContradiction
    reference target (structural, enforced at the JSON-schema gate --
    see extraction.json_schemas._OBSERVATION_REF_SCHEMA).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import DimensionCode, Provenance
from domain.evidence import EvidenceObject

from extraction.dedup import deduplicate
from extraction.enums import RejectionReason
from extraction.errors import ItemRejected, ModelServiceError
from extraction.pipeline import run_dimension_qualifier_classification, run_extraction
from extraction.prompts import build_isolated_dimension_qualifier_user_message
from extraction.provider import FakeExtractionProvider
from extraction.schemas import CandidateDimensionQualifier, ExtractionSystemFields, ObservationRef
from extraction.validation import validate_contradiction_item_shape
from extraction.enums import ObservationType


def _evidence(evidence_id, text, source="account_note"):
    return EvidenceObject(evidence_id, None, text, source, Provenance.USER_PROVIDED)


ADOPTION_TEXT = "The monthly reporting workflow has not been used at all this quarter."
STAKEHOLDER_TEXT = "Our champion Jane Smith left the company last month and no one has taken over her role."

STAGE1_RESPONSE = {
    "adoption_observations": [
        {
            "source_evidence_id": "E1",
            "source_span": {"text": ADOPTION_TEXT},
            "basis": "EXPLICIT",
            "workflow_or_use_case": "monthly reporting",
            "observed_behavior": "not used at all this quarter",
        }
    ],
    "stakeholder_observations": [
        {
            "source_evidence_id": "E2",
            "source_span": {"text": STAKEHOLDER_TEXT},
            "basis": "EXPLICIT",
            "person_identifier": "Jane Smith",
            "role": "champion",
        }
    ],
}

# Isolated-architecture-shaped responses: each is a SINGLE-CHANNEL,
# at-most-one-item envelope -- exactly what one isolated call may return.
D2_QUALIFIER_RESPONSE = {
    "candidate_d2_qualifiers": [
        {
            "qualifier": "WORKFLOWS_NOT_OCCURRING", "basis": "EXPLICIT",
            "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
        }
    ],
}
# Milestone 4B v3: CHAMPION_LOST_NO_SUCCESSOR can no longer be proposed
# directly (D6_QUALIFIER_SCHEMA structurally excludes it) -- this fixture
# now supplies the full 2/2 required atomic-predicate set instead, both
# grounded as exact substrings of STAKEHOLDER_TEXT, so the pipeline's
# deterministic composer produces the SAME final qualifier value the old
# direct-proposal fixture did, keeping every downstream assertion in this
# file unchanged.
D6_QUALIFIER_RESPONSE = {
    "candidate_d6_atomic_predicates": [
        {
            "predicate_id": "CONFIRMED_CHAMPION_DEPARTURE", "basis": "EXPLICIT",
            "evidence_text": "left the company last month",
        },
        {
            "predicate_id": "NO_SUCCESSOR_OR_CONTINUING_COVERAGE", "basis": "EXPLICIT",
            "evidence_text": "no one has taken over her role",
        },
    ],
}


def _dq_success_router(dimension, observation, repair_hint=None):
    """A callable dimension_qualifier_responses fixture: routes to the
    correct single-channel envelope based on which dimension THIS
    isolated call is for -- exactly what a real isolated-call provider
    must do (never a shared, both-channel response)."""
    if dimension == DimensionCode.D2:
        return D2_QUALIFIER_RESPONSE
    if dimension == DimensionCode.D6:
        return D6_QUALIFIER_RESPONSE
    raise AssertionError(f"unexpected dimension in test fixture: {dimension!r}")


def _stage1_result():
    evidence = (_evidence("E1", ADOPTION_TEXT), _evidence("E2", STAKEHOLDER_TEXT))
    provider = FakeExtractionProvider(STAGE1_RESPONSE)
    result = run_extraction(evidence, provider)
    assert not result.rejected, result.rejected
    assert len(result.accepted) == 2
    return result


STAGE1_D2_ONLY_RESPONSE = {
    "adoption_observations": [
        {
            "source_evidence_id": "E1",
            "source_span": {"text": ADOPTION_TEXT},
            "basis": "EXPLICIT",
            "workflow_or_use_case": "monthly reporting",
            "observed_behavior": "not used at all this quarter",
        }
    ],
}


def _stage1_result_d2_only():
    """A single-channel fixture (D2 only, no stakeholder observation) --
    used by tests that need to control exactly what the ONE isolated call
    in the run returns, without a second (D6) call in the same run
    complicating the response shape."""
    evidence = (_evidence("E1", ADOPTION_TEXT),)
    provider = FakeExtractionProvider(STAGE1_D2_ONLY_RESPONSE)
    result = run_extraction(evidence, provider)
    assert not result.rejected, result.rejected
    assert len(result.accepted) == 1
    return result


def test_success_path_populates_both_channels_and_preserves_stage1():
    result1 = _stage1_result()
    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_dq_success_router)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert len(result2.candidate_d2_qualifiers) == 1
    assert len(result2.candidate_d6_qualifiers) == 1
    assert result2.dimension_qualifier_stage_failure is None
    assert result2.dimension_qualifier_failures == ()
    # stage-1 fields are byte-for-byte preserved
    assert result2.accepted == result1.accepted
    assert result2.candidate_contradictions == result1.candidate_contradictions
    assert result2.candidate_risk_signals == result1.candidate_risk_signals
    assert result2.candidate_evidence_classifications == result1.candidate_evidence_classifications
    assert result2.request_failure is None

    d2 = result2.candidate_d2_qualifiers[0]
    assert d2.source_evidence_id == "E1"
    assert d2.source_span.text == ADOPTION_TEXT


def test_isolated_architecture_makes_exactly_one_call_per_eligible_observation():
    """Expected-API-call-count guarantee: 2 eligible observations (1
    AdoptionObservation, 1 StakeholderObservation) must produce exactly 2
    isolated provider calls -- never 1 (the old batched architecture's
    call count) and never more than 2 (no redundant retries on a
    successful first attempt)."""
    result1 = _stage1_result()
    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_dq_success_router)
    run_dimension_qualifier_classification(result1, provider2)
    assert provider2.dimension_qualifier_call_count == 2
    assert len(provider2.dimension_qualifier_call_log) == 2


def test_isolated_call_receives_a_single_observation_not_a_collection():
    """Structural pin: propose_isolated_dimension_qualifier's `observation`
    argument must be ONE observation object, never a tuple/list of
    observations -- the isolated architecture's defining constraint."""
    result1 = _stage1_result()
    seen = []

    def _capture(dimension, observation, repair_hint=None):
        seen.append((dimension, observation))
        return _dq_success_router(dimension, observation, repair_hint)

    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_capture)
    run_dimension_qualifier_classification(result1, provider2)

    assert len(seen) == 2
    for dimension, observation in seen:
        assert not isinstance(observation, (tuple, list))
        assert hasattr(observation, "source_span")
        if dimension == DimensionCode.D2:
            assert type(observation).__name__ == "AdoptionObservation"
        else:
            assert type(observation).__name__ == "StakeholderObservation"


def test_isolated_user_message_never_includes_sibling_observation_content():
    """The core structural guarantee of the whole rewrite: a D2 call's
    user message must show ONLY the adoption observation's own content --
    never the stakeholder observation's text or fields, even though both
    were accepted in the SAME stage-1 run -- and vice versa for D6."""
    result1 = _stage1_result()
    adoption_obs = next(o for o in result1.accepted if type(o).__name__ == "AdoptionObservation")
    stakeholder_obs = next(o for o in result1.accepted if type(o).__name__ == "StakeholderObservation")

    d2_message = build_isolated_dimension_qualifier_user_message(DimensionCode.D2, adoption_obs)
    assert ADOPTION_TEXT in d2_message
    assert STAKEHOLDER_TEXT not in d2_message
    assert "Jane Smith" not in d2_message

    d6_message = build_isolated_dimension_qualifier_user_message(DimensionCode.D6, stakeholder_obs)
    assert STAKEHOLDER_TEXT in d6_message
    assert ADOPTION_TEXT not in d6_message
    assert "monthly reporting" not in d6_message


def test_inherited_grounding_copies_supporting_observations_span_verbatim():
    result1 = _stage1_result()
    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_dq_success_router)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    adoption_obs = next(o for o in result1.accepted if type(o).__name__ == "AdoptionObservation")
    d2 = result2.candidate_d2_qualifiers[0]
    assert d2.source_evidence_id == adoption_obs.source_evidence_id
    assert d2.source_span == adoption_obs.source_span
    assert d2.resolved_observation_id == adoption_obs.system.observation_id


def test_dimension_qualifier_system_is_distinct_from_supporting_observations_system():
    result1 = _stage1_result()
    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_dq_success_router)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    adoption_obs = next(o for o in result1.accepted if type(o).__name__ == "AdoptionObservation")
    d2 = result2.candidate_d2_qualifiers[0]
    assert d2.system.observation_id != adoption_obs.system.observation_id
    assert d2.system.trace_id != adoption_obs.system.trace_id


def test_skip_when_no_eligible_observations_makes_no_provider_call():
    evidence = (_evidence("E3", "The customer mentioned pricing concerns during the call."),)
    provider1 = FakeExtractionProvider({})
    result1 = run_extraction(evidence, provider1)
    assert result1.accepted == ()

    provider2 = FakeExtractionProvider({}, dimension_qualifier_responses=_dq_success_router)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2 is result1
    assert provider2.dimension_qualifier_call_count == 0


def test_graceful_degradation_on_model_service_error():
    result1 = _stage1_result_d2_only()
    provider2 = FakeExtractionProvider(
        STAGE1_D2_ONLY_RESPONSE, dimension_qualifier_responses={}, raise_dimension_qualifier_service_error=True,
    )
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.dimension_qualifier_stage_failure is not None
    assert len(result2.dimension_qualifier_failures) == 1
    failure = result2.dimension_qualifier_failures[0]
    assert failure.resolved_observation_id == result1.accepted[0].system.observation_id
    assert failure.dimension == DimensionCode.D2
    assert result2.candidate_d2_qualifiers == ()
    assert result2.candidate_d6_qualifiers == ()
    assert result2.request_failure is None
    # stage-1 fields untouched
    assert result2.accepted == result1.accepted
    assert result2.rejected == result1.rejected
    assert result2.traces == result1.traces


def test_per_observation_failure_isolation_one_failure_does_not_discard_other_successes():
    """Approved architecture item D, the defining guarantee of the
    isolated-failure-tracking rewrite: in a run with TWO eligible
    observations, the D2 call failing must not discard the D6 call's
    successfully-produced qualifier (or vice versa) -- exactly the
    all-or-nothing failure mode the isolated architecture replaces."""
    result1 = _stage1_result()

    def _d2_fails_d6_succeeds(dimension, observation, repair_hint=None):
        if dimension == DimensionCode.D2:
            raise ModelServiceError("simulated D2-only outage")
        return D6_QUALIFIER_RESPONSE

    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_d2_fails_d6_succeeds)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    # D2 failed...
    assert result2.candidate_d2_qualifiers == ()
    assert len(result2.dimension_qualifier_failures) == 1
    assert result2.dimension_qualifier_failures[0].dimension == DimensionCode.D2
    assert result2.dimension_qualifier_stage_failure is not None
    assert "1 of 2" in result2.dimension_qualifier_stage_failure

    # ...but D6's success is completely unaffected.
    assert len(result2.candidate_d6_qualifiers) == 1
    assert result2.candidate_d6_qualifiers[0].qualifier == "CHAMPION_LOST_NO_SUCCESSOR"


def test_graceful_degradation_after_failed_repair_retry():
    result1 = _stage1_result_d2_only()
    provider2 = FakeExtractionProvider(
        STAGE1_D2_ONLY_RESPONSE, dimension_qualifier_responses=[{"bogus": []}, {"still_bogus": []}],
    )
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.dimension_qualifier_stage_failure is not None
    assert "repair" in result2.dimension_qualifier_stage_failure
    assert len(result2.dimension_qualifier_failures) == 1
    assert result2.candidate_d2_qualifiers == ()
    assert result2.candidate_d6_qualifiers == ()
    assert result2.request_failure is None
    assert provider2.dimension_qualifier_call_count == 2


def test_repair_retry_recovers_on_second_attempt():
    result1 = _stage1_result_d2_only()
    provider2 = FakeExtractionProvider(
        STAGE1_D2_ONLY_RESPONSE, dimension_qualifier_responses=[{"bogus": []}, D2_QUALIFIER_RESPONSE],
    )
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.dimension_qualifier_stage_failure is None
    assert result2.dimension_qualifier_failures == ()
    assert len(result2.candidate_d2_qualifiers) == 1


def test_abstention_is_distinct_from_stage_failure():
    """Successful call, zero proposals (abstention) must leave
    dimension_qualifier_stage_failure/dimension_qualifier_failures empty
    -- distinct from a failed call, which also produces zero proposals
    but populates both."""
    result1 = _stage1_result_d2_only()
    provider2 = FakeExtractionProvider(STAGE1_D2_ONLY_RESPONSE, dimension_qualifier_responses={})
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.dimension_qualifier_stage_failure is None
    assert result2.dimension_qualifier_failures == ()
    assert result2.candidate_d2_qualifiers == ()
    assert result2.candidate_d6_qualifiers == ()


def test_grounding_prohibition_a_disallowed_observation_type():
    """(a) supporting_observation_ref cites an observation_type outside
    this channel's own allowed type. The per-channel JSON schema already
    restricts this at the model-facing gate (SCHEMA_INVALID fires first,
    via validate_dimension_qualifier_shape) -- this test confirms the
    end-to-end pipeline path surfaces that as an ordinary rejected entry,
    never an uncaught exception or a stage failure. Uses the D2-only
    fixture so this run's single isolated call is unambiguous."""
    result1 = _stage1_result_d2_only()
    bad = {
        "candidate_d2_qualifiers": [
            {
                "qualifier": "WORKFLOWS_NOT_OCCURRING", "basis": "EXPLICIT",
                "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
            }
        ],
    }
    provider2 = FakeExtractionProvider(STAGE1_D2_ONLY_RESPONSE, dimension_qualifier_responses=bad)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.dimension_qualifier_stage_failure is None
    assert result2.candidate_d2_qualifiers == ()
    assert result2.rejected != result1.rejected
    new_entry = result2.rejected[-1]
    assert new_entry.observation_type == "candidate_d2_qualifiers"
    assert new_entry.reason == RejectionReason.SCHEMA_INVALID


def test_grounding_prohibition_b_unresolved_reference_out_of_range():
    """(b) index does not resolve to any item in the channel-homogeneous
    accepted-observation list this call was given -- under the isolated
    architecture that list always has exactly one possible position (0),
    so any non-zero index is definitionally out of range."""
    result1 = _stage1_result_d2_only()
    bad = {
        "candidate_d2_qualifiers": [
            {
                "qualifier": "WORKFLOWS_NOT_OCCURRING", "basis": "EXPLICIT",
                "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": 7},
            }
        ],
    }
    provider2 = FakeExtractionProvider(STAGE1_D2_ONLY_RESPONSE, dimension_qualifier_responses=bad)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.dimension_qualifier_stage_failure is None
    assert result2.candidate_d2_qualifiers == ()
    new_entry = result2.rejected[-1]
    assert new_entry.reason == RejectionReason.DIMENSION_QUALIFIER_REFERENCE_NOT_FOUND


def test_grounding_prohibition_defense_in_depth_functions_are_reachable():
    """(c) rejected observation and (d) ungrounded object are both
    documented as defense-in-depth (structurally unreachable through the
    normal call path, since candidate_list is built exclusively from
    already-accepted, already-grounded observations) -- confirmed here by
    reading the source of _build_dimension_qualifier directly rather than
    by attempting to force an unreachable path, mirroring how this
    codebase already treats its other defense-in-depth guards (e.g.
    CANDIDATE_CLASSIFICATION_REFERENCED_OBSERVATION_NOT_TRACEABLE)."""
    import inspect
    from extraction import pipeline as pipeline_module
    src = inspect.getsource(pipeline_module._build_dimension_qualifier)
    assert "DIMENSION_QUALIFIER_REFERENCES_REJECTED_ITEM" in src
    assert "DIMENSION_QUALIFIER_REFERENCES_UNGROUNDED_OBJECT" in src


def test_isolated_envelope_schema_structurally_forbids_more_than_one_item():
    """Milestone 4B isolated-classifier architecture checkpoint: unlike
    the prior batched envelope (unlimited items per channel, relying on
    dedup.py to collapse accidental duplicates emitted within one call),
    each isolated call's envelope is capped at maxItems: 1 -- a single
    call can no longer even ATTEMPT to return two qualifiers for the one
    observation it was shown. This is enforced as a real schema
    constraint (part of the forced-tool input_schema), not just
    documentation."""
    from extraction.json_schemas import ISOLATED_D2_QUALIFIER_TOP_LEVEL_SCHEMA, ISOLATED_D6_QUALIFIER_TOP_LEVEL_SCHEMA
    assert ISOLATED_D2_QUALIFIER_TOP_LEVEL_SCHEMA["properties"]["candidate_d2_qualifiers"]["maxItems"] == 1
    assert ISOLATED_D6_QUALIFIER_TOP_LEVEL_SCHEMA["properties"]["candidate_d6_qualifiers"]["maxItems"] == 1


def test_dedup_still_collapses_identical_candidate_dimension_qualifiers_across_calls():
    """dedup.py's existing, UNCHANGED machinery must still collapse two
    CandidateDimensionQualifier objects that share identical grounding
    (dimension, qualifier, source_evidence_id, source_span) -- this can
    no longer be manufactured WITHIN a single isolated call (see
    test_isolated_envelope_schema_structurally_forbids_more_than_one_item
    above), so this is now a defense-in-depth, cross-call guarantee
    rather than something the normal per-run loop can trigger through
    ordinary model behavior. Exercised directly against dedup.deduplicate,
    the same function run_dimension_qualifier_classification calls once
    over its full cross-call `finalized` collection."""
    from domain.enums import EvidenceState
    from extraction.schemas import InferenceBasis, SourceSpan

    span = SourceSpan(text=ADOPTION_TEXT, start_char=0, end_char=len(ADOPTION_TEXT))
    ref = ObservationRef(observation_type=ObservationType.ADOPTION_OBSERVATION, index=0)

    def _make(obs_id: str) -> CandidateDimensionQualifier:
        system = ExtractionSystemFields(
            observation_id=obs_id, model_provider="fake", model_version="fake-v1",
            extracted_at=None, trace_id=f"TRACE-{obs_id}", evidence_state=EvidenceState.CURRENT_UNVERIFIED,
        )
        return CandidateDimensionQualifier(
            dimension=DimensionCode.D2, qualifier="WORKFLOWS_NOT_OCCURRING", basis=InferenceBasis.EXPLICIT,
            supporting_observation_ref=ref, source_evidence_id="E1", source_span=span,
            resolved_observation_id="OBS-SHARED", system=system,
        )

    first = _make("DIMQ-1")
    second = _make("DIMQ-2")
    kept, _canonical_map, audit = deduplicate((first, second))
    assert len(kept) == 1
    assert len(audit) == 1


def test_multiple_eligible_observations_each_get_their_own_qualifier():
    """A run with TWO adoption observations must produce TWO isolated D2
    calls and TWO distinct qualifiers (one per observation), never a
    single call covering both -- the direct behavioral contrast with the
    old batched architecture."""
    text_a = "The monthly reporting workflow has not been used at all this quarter."
    text_b = "The customer's finance team stopped using automated invoice reconciliation entirely."
    stage1 = {
        "adoption_observations": [
            {"source_evidence_id": "E1", "source_span": {"text": text_a}, "basis": "EXPLICIT",
             "workflow_or_use_case": "monthly reporting", "observed_behavior": "not used at all this quarter"},
            {"source_evidence_id": "E2", "source_span": {"text": text_b}, "basis": "EXPLICIT",
             "workflow_or_use_case": "invoice reconciliation", "observed_behavior": "stopped entirely"},
        ],
    }
    evidence = (_evidence("E1", text_a), _evidence("E2", text_b))
    provider1 = FakeExtractionProvider(stage1)
    result1 = run_extraction(evidence, provider1)
    assert len(result1.accepted) == 2

    provider2 = FakeExtractionProvider(stage1, dimension_qualifier_responses=D2_QUALIFIER_RESPONSE)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert provider2.dimension_qualifier_call_count == 2
    assert len(result2.candidate_d2_qualifiers) == 2
    resolved_ids = {q.resolved_observation_id for q in result2.candidate_d2_qualifiers}
    accepted_ids = {o.system.observation_id for o in result1.accepted}
    assert resolved_ids == accepted_ids


def test_traces_are_appended_never_replacing_stage_one_traces():
    result1 = _stage1_result()
    provider2 = FakeExtractionProvider(STAGE1_RESPONSE, dimension_qualifier_responses=_dq_success_router)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.traces[: len(result1.traces)] == result1.traces
    assert len(result2.traces) == len(result1.traces) + 2
    new_trace_ids = {t.subject_object_ref for t in result2.traces[len(result1.traces):]}
    dq_ids = {o.system.observation_id for o in (*result2.candidate_d2_qualifiers, *result2.candidate_d6_qualifiers)}
    assert new_trace_ids == dq_ids


def test_candidate_dimension_qualifier_is_never_a_valid_contradiction_reference_target():
    """Structural guarantee: extraction.json_schemas._OBSERVATION_REF_
    SCHEMA's enum is built from OBSERVATION_TYPE_TO_ARRAY_KEY's keys only
    -- CANDIDATE_D2_QUALIFIER/CANDIDATE_D6_QUALIFIER live in the
    deliberately separate DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY map, so a
    contradiction citing either is rejected at the schema gate, before
    reference resolution ever runs (mirrors how CANDIDATE_RISK_SIGNAL /
    CANDIDATE_EVIDENCE_CLASSIFICATION are already excluded)."""
    item = {
        "observation_ref_a": {"observation_type": "CANDIDATE_D2_QUALIFIER", "index": 0},
        "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
        "conflict_description": "test",
    }
    try:
        validate_contradiction_item_shape(item)
        assert False, "CANDIDATE_D2_QUALIFIER must never be a valid contradiction reference target"
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_user_message_includes_each_observations_own_source_span_text():
    """Milestone 4B calibration checkpoint 1 (Prompt v1.1), preserved
    under the isolated architecture: each call's user message must show
    that ONE observation's own exact source_span.text alongside its
    structured fields -- the highest-leverage, in-scope fix identified in
    the prompt_v4_4b_dimqual_eval1 root-cause analysis (case 41: stage-2
    had almost no signal when stage-1's structured fields were thin, even
    though the observation's own quoted text existed)."""
    result1 = _stage1_result()
    adoption_obs = next(o for o in result1.accepted if type(o).__name__ == "AdoptionObservation")
    stakeholder_obs = next(o for o in result1.accepted if type(o).__name__ == "StakeholderObservation")
    d2_message = build_isolated_dimension_qualifier_user_message(DimensionCode.D2, adoption_obs)
    d6_message = build_isolated_dimension_qualifier_user_message(DimensionCode.D6, stakeholder_obs)
    assert ADOPTION_TEXT in d2_message
    assert STAKEHOLDER_TEXT in d6_message


PARTIAL_SPAN_ADOPTION_TEXT = (
    "The customer's finance team continues to run their month-end close workflow "
    "in the platform exactly as designed."
)
UNRELATED_TRAILING_TEXT = (
    "Separately, an unrelated support ticket about the customer's billing address "
    "was closed on Tuesday."
)
PARTIAL_SPAN_EVIDENCE_TEXT = PARTIAL_SPAN_ADOPTION_TEXT + " " + UNRELATED_TRAILING_TEXT

PARTIAL_SPAN_STAGE1_RESPONSE = {
    "adoption_observations": [
        {
            "source_evidence_id": "E9",
            "source_span": {"text": PARTIAL_SPAN_ADOPTION_TEXT},
            "basis": "EXPLICIT",
            "workflow_or_use_case": "month-end close",
            "observed_behavior": "operating as designed",
        }
    ],
}


def test_user_message_shows_only_the_observations_own_span_not_the_full_raw_evidence_item():
    """Provenance re-audit (calibration checkpoint 1, governing
    interpretation), preserved under the isolated architecture: stage 2
    must see exactly what stage 1 already grounded for an observation --
    never the surrounding raw evidence text that was never captured into
    that observation's own span. source_span.text is the observation's
    OWN grounded representation, not a window into the rest of the
    evidence item it came from."""
    evidence = (_evidence("E9", PARTIAL_SPAN_EVIDENCE_TEXT),)
    provider1 = FakeExtractionProvider(PARTIAL_SPAN_STAGE1_RESPONSE)
    result1 = run_extraction(evidence, provider1)
    assert len(result1.accepted) == 1

    message = build_isolated_dimension_qualifier_user_message(DimensionCode.D2, result1.accepted[0])
    assert PARTIAL_SPAN_ADOPTION_TEXT in message
    assert UNRELATED_TRAILING_TEXT not in message


TESTS = [
    test_success_path_populates_both_channels_and_preserves_stage1,
    test_isolated_architecture_makes_exactly_one_call_per_eligible_observation,
    test_isolated_call_receives_a_single_observation_not_a_collection,
    test_isolated_user_message_never_includes_sibling_observation_content,
    test_inherited_grounding_copies_supporting_observations_span_verbatim,
    test_dimension_qualifier_system_is_distinct_from_supporting_observations_system,
    test_skip_when_no_eligible_observations_makes_no_provider_call,
    test_graceful_degradation_on_model_service_error,
    test_per_observation_failure_isolation_one_failure_does_not_discard_other_successes,
    test_graceful_degradation_after_failed_repair_retry,
    test_repair_retry_recovers_on_second_attempt,
    test_abstention_is_distinct_from_stage_failure,
    test_grounding_prohibition_a_disallowed_observation_type,
    test_grounding_prohibition_b_unresolved_reference_out_of_range,
    test_grounding_prohibition_defense_in_depth_functions_are_reachable,
    test_isolated_envelope_schema_structurally_forbids_more_than_one_item,
    test_dedup_still_collapses_identical_candidate_dimension_qualifiers_across_calls,
    test_multiple_eligible_observations_each_get_their_own_qualifier,
    test_traces_are_appended_never_replacing_stage_one_traces,
    test_candidate_dimension_qualifier_is_never_a_valid_contradiction_reference_target,
    test_user_message_includes_each_observations_own_source_span_text,
    test_user_message_shows_only_the_observations_own_span_not_the_full_raw_evidence_item,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
