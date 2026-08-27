import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extraction.enums import ObservationType
from extraction.errors import ItemRejected, RejectionReason
from extraction.validation import (
    build_observation, resolve_source_span, scan_for_prohibited_keys, validate_candidate_classification_shape,
    validate_contradiction_item_shape, validate_dimension_qualifier_shape,
    validate_isolated_dimension_qualifier_top_level_shape, validate_source_span, validate_top_level_shape,
)
from extraction.schemas import SourceSpan

EVIDENCE_TEXT = "Roberto left the company in July. No replacement sponsor has been named."


def _valid_item(**overrides):
    # Milestone 2B baseline fix: the model supplies only source_span.text —
    # start_char/end_char are no longer part of the model-facing contract
    # (see extraction/json_schemas.py / validation.resolve_source_span).
    item = {
        "source_evidence_id": "E1",
        "source_span": {"text": "Roberto left the company in July"},
        "basis": "EXPLICIT",
        "person_identifier": "Roberto",
        "continuity_event": "DEPARTED",
    }
    item.update(overrides)
    return item


def test_build_observation_accepts_valid_item():
    obs = build_observation(ObservationType.STAKEHOLDER_OBSERVATION, _valid_item(), {"E1": EVIDENCE_TEXT})
    assert obs.person_identifier == "Roberto"
    assert obs.system.is_populated is False  # pipeline attaches system fields, not validation


def test_build_observation_derives_span_offsets_from_text():
    """Milestone 2B baseline fix: offsets are never model-supplied —
    the application derives them from the model's literal text."""
    obs = build_observation(ObservationType.STAKEHOLDER_OBSERVATION, _valid_item(), {"E1": EVIDENCE_TEXT})
    expected_text = "Roberto left the company in July"
    assert obs.source_span.start_char == EVIDENCE_TEXT.index(expected_text)
    assert obs.source_span.end_char - obs.source_span.start_char == len(expected_text)
    assert obs.source_span.text == expected_text


def test_build_observation_rejects_prohibited_key():
    item = _valid_item(activated_severity="CRITICAL")
    try:
        build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": EVIDENCE_TEXT})
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.BOUNDARY_VIOLATION


def test_build_observation_rejects_model_set_system_field():
    for key, value in [
        ("model_provider", "anthropic"), ("model_version", "x"),
        ("extracted_at", "2026-01-01"), ("trace_id", "TRACE-1"),
        ("evidence_state", "CURRENT_CONFIRMED"), ("observation_id", "OBS-1"),
    ]:
        item = _valid_item(**{key: value})
        try:
            build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": EVIDENCE_TEXT})
            assert False, f"must reject model-set {key}"
        except ItemRejected as e:
            assert e.reason == RejectionReason.BOUNDARY_VIOLATION, key


def test_build_observation_rejects_span_not_found():
    # Milestone 2B baseline fix: since offsets are derived from the text,
    # SPAN_NOT_FOUND now means "this text does not appear verbatim in the
    # cited evidence at all" (there is no offset for the model to get
    # wrong anymore).
    item = _valid_item(source_span={"text": "this exact phrase is not in the evidence"})
    try:
        build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": EVIDENCE_TEXT})
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SPAN_NOT_FOUND


def test_build_observation_rejects_ambiguous_span_text():
    """Milestone 2B baseline fix: if the model's literal text appears more
    than once verbatim in the evidence, the system must not silently pick
    the first occurrence — it must reject as SPAN_AMBIGUOUS."""
    evidence = "Roberto said hi. Roberto said hi again, later that day."
    item = _valid_item(source_span={"text": "Roberto said hi"})
    try:
        build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": evidence})
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SPAN_AMBIGUOUS


def test_build_observation_rejects_model_supplied_offsets():
    """Milestone 2B baseline fix: start_char/end_char are no longer part
    of the model-facing schema at all — additionalProperties: false must
    reject them outright, not accept them as ignored advisory values."""
    item = _valid_item(source_span={"text": "Roberto left the company in July", "start_char": 0, "end_char": 33})
    try:
        build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": EVIDENCE_TEXT})
        assert False, "model-supplied offsets must be rejected, not silently accepted or ignored"
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_build_observation_rejects_unknown_evidence_id():
    item = _valid_item(source_evidence_id="E-DOES-NOT-EXIST")
    try:
        build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": EVIDENCE_TEXT})
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.UNKNOWN_EVIDENCE_ID


def test_build_observation_rejects_missing_required_field():
    item = _valid_item()
    del item["person_identifier"]
    try:
        build_observation(ObservationType.STAKEHOLDER_OBSERVATION, item, {"E1": EVIDENCE_TEXT})
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_missing_information_candidate_no_span_required():
    item = {"missing_item": "customer_objective", "reviewed_evidence_ids": ["E1"]}
    obs = build_observation(ObservationType.MISSING_INFORMATION_CANDIDATE, item, {"E1": EVIDENCE_TEXT})
    assert obs.missing_item == "customer_objective"


def test_missing_information_rejects_unknown_reviewed_evidence_id():
    item = {"missing_item": "x", "reviewed_evidence_ids": ["E-GHOST"]}
    try:
        build_observation(ObservationType.MISSING_INFORMATION_CANDIDATE, item, {"E1": EVIDENCE_TEXT})
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.UNKNOWN_EVIDENCE_ID


def test_scan_for_prohibited_keys_finds_nested():
    nested = {"a": {"b": [{"dimension_state": "CONCERNING"}]}}
    hits = scan_for_prohibited_keys(nested)
    assert "dimension_state" in hits


def test_scan_for_prohibited_keys_clean():
    clean = {"a": {"b": [{"person_identifier": "Roberto"}]}}
    assert scan_for_prohibited_keys(clean) == []


def test_resolve_source_span_derives_offsets_from_single_occurrence():
    span_text = "Roberto left the company in July"
    span = resolve_source_span(EVIDENCE_TEXT, span_text)
    assert span.start_char == EVIDENCE_TEXT.index(span_text)
    assert span.end_char - span.start_char == len(span_text)
    assert span.text == span_text


def test_resolve_source_span_rejects_zero_occurrences():
    try:
        resolve_source_span(EVIDENCE_TEXT, "text that never appears anywhere in the evidence")
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SPAN_NOT_FOUND


def test_resolve_source_span_rejects_ambiguous_multiple_occurrences():
    text = "Roberto said hi. Roberto said hi again."
    try:
        resolve_source_span(text, "Roberto said hi")
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SPAN_AMBIGUOUS


def test_resolve_source_span_preserves_raw_item_on_rejection():
    raw = {"source_evidence_id": "E1", "source_span": {"text": "not present"}}
    try:
        resolve_source_span(EVIDENCE_TEXT, "not present", raw_item=raw)
        assert False
    except ItemRejected as e:
        assert e.raw_item is raw


def test_validate_source_span_exact_match():
    span = SourceSpan("Roberto", 0, 7)
    assert validate_source_span("Roberto left", span) is True


def test_validate_source_span_offset_mismatch():
    span = SourceSpan("Roberto", 1, 8)
    assert validate_source_span("Roberto left", span) is False


def test_validate_source_span_out_of_bounds():
    span = SourceSpan("xyz", 100, 103)
    assert validate_source_span("short text", span) is False


def test_validate_top_level_shape_rejects_unknown_key():
    try:
        validate_top_level_shape({"not_a_real_array": []})
        assert False
    except Exception:
        pass


def test_validate_top_level_shape_accepts_empty():
    validate_top_level_shape({})
    validate_top_level_shape({"stakeholder_observations": []})


def test_validate_contradiction_item_shape_accepts_valid():
    item = {
        "observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
        "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
        "conflict_description": "conflict",
    }
    validate_contradiction_item_shape(item)  # must not raise


def test_validate_contradiction_item_shape_rejects_boundary_violation():
    item = {
        "observation_ref_a": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
        "observation_ref_b": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
        "conflict_description": "conflict",
        "dmeg": True,
    }
    try:
        validate_contradiction_item_shape(item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.BOUNDARY_VIOLATION


def _valid_risk_signal_item(**overrides):
    item = {
        "source_span": {"text": "no successor has been named"},
        "basis": "EXPLICIT",
        "mechanism": "CR-01",
        "proposed_severity_tier": "CRITICAL",
        "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
    }
    item.update(overrides)
    return item


def test_validate_candidate_classification_shape_accepts_valid_risk_signal():
    validate_candidate_classification_shape(ObservationType.CANDIDATE_RISK_SIGNAL, _valid_risk_signal_item())


def test_validate_candidate_classification_shape_rejects_boundary_violation():
    item = _valid_risk_signal_item(activated_severity="CRITICAL")
    try:
        validate_candidate_classification_shape(ObservationType.CANDIDATE_RISK_SIGNAL, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.BOUNDARY_VIOLATION


def test_validate_candidate_classification_shape_rejects_source_evidence_id():
    """Milestone 2C implementation constraint 1: source_evidence_id is
    application-derived only — there is no field for the model to
    populate. additionalProperties: false on CANDIDATE_RISK_SIGNAL_SCHEMA
    must reject it outright, exactly the same 'no field exists'
    enforcement SOURCE_SPAN_SCHEMA already uses for start_char/end_char."""
    item = _valid_risk_signal_item(source_evidence_id="E1")
    try:
        validate_candidate_classification_shape(ObservationType.CANDIDATE_RISK_SIGNAL, item)
        assert False, "source_evidence_id must never be accepted on a candidate classification item"
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_candidate_classification_shape_rejects_deferred_mechanism():
    item = _valid_risk_signal_item(mechanism="CR-06")
    try:
        validate_candidate_classification_shape(ObservationType.CANDIDATE_RISK_SIGNAL, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_candidate_classification_shape_rejects_disallowed_ref_type():
    """The supporting_observation_ref's observation_type enum is
    restricted to the 7 semantic types — MISSING_INFORMATION_CANDIDATE
    (and, by the same mechanism, CANDIDATE_CONTRADICTION / either
    candidate-classification type) must be structurally impossible to
    cite as a supporting observation."""
    item = _valid_risk_signal_item(
        supporting_observation_ref={"observation_type": "MISSING_INFORMATION_CANDIDATE", "index": 0},
    )
    try:
        validate_candidate_classification_shape(ObservationType.CANDIDATE_RISK_SIGNAL, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def _valid_evidence_classification_item(**overrides):
    item = {
        "source_span": {"text": "resolution time at 3.6 hours"},
        "basis": "EXPLICIT",
        "proposed_basis": "MEASURED_OPERATIONAL_EVIDENCE",
        "supports": "ACHIEVED",
        "supporting_observation_ref": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
    }
    item.update(overrides)
    return item


def test_validate_candidate_classification_shape_accepts_valid_evidence_classification():
    validate_candidate_classification_shape(
        ObservationType.CANDIDATE_EVIDENCE_CLASSIFICATION, _valid_evidence_classification_item(),
    )


def test_validate_candidate_classification_shape_rejects_absence_basis():
    item = _valid_evidence_classification_item(proposed_basis="UNVERIFIED_CLAIM")
    try:
        validate_candidate_classification_shape(ObservationType.CANDIDATE_EVIDENCE_CLASSIFICATION, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def _valid_d2_qualifier_item(**overrides):
    item = {
        "qualifier": "WORKFLOWS_NOT_OCCURRING",
        "basis": "EXPLICIT",
        "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
    }
    item.update(overrides)
    return item


def _valid_d6_qualifier_item(**overrides):
    # Milestone 4B v3: CHAMPION_LOST_NO_SUCCESSOR is no longer directly
    # proposable (structurally excluded from D6_QUALIFIER_SCHEMA's enum --
    # see extraction/json_schemas.py) -- APPROPRIATE_SPONSOR_COVERAGE is
    # one of the 3 remaining directly-proposable D6 values.
    item = {
        "qualifier": "APPROPRIATE_SPONSOR_COVERAGE",
        "basis": "EXPLICIT",
        "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
    }
    item.update(overrides)
    return item


def test_validate_dimension_qualifier_shape_accepts_valid_d2_item():
    validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D2_QUALIFIER, _valid_d2_qualifier_item())


def test_validate_dimension_qualifier_shape_accepts_valid_d6_item():
    validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D6_QUALIFIER, _valid_d6_qualifier_item())


def test_validate_dimension_qualifier_shape_rejects_boundary_violation():
    item = _valid_d2_qualifier_item(dimension_state="CONCERNING")
    try:
        validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D2_QUALIFIER, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.BOUNDARY_VIOLATION


def test_validate_dimension_qualifier_shape_rejects_source_span():
    """Milestone 4B inherited-grounding exception: unlike the two
    Milestone 2C candidate-classification types (which omit only
    source_evidence_id), a D2/D6 qualifier item declares NEITHER
    source_span NOR source_evidence_id -- there is no field for the model
    to occupy for either, since both are inherited verbatim from the
    resolved supporting observation by the pipeline."""
    item = _valid_d2_qualifier_item(source_span={"text": "no workflow"})
    try:
        validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D2_QUALIFIER, item)
        assert False, "source_span must never be accepted on a dimension-qualifier item"
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_dimension_qualifier_shape_rejects_source_evidence_id():
    item = _valid_d2_qualifier_item(source_evidence_id="E1")
    try:
        validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D2_QUALIFIER, item)
        assert False, "source_evidence_id must never be accepted on a dimension-qualifier item"
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_dimension_qualifier_shape_rejects_d6_qualifier_on_d2_channel():
    """The per-channel JSON schema enum restricts D2 items to the 5 D2
    qualifier values only -- a D6 value must fail schema validation, not
    merely dataclass validation (defense in depth, same discipline as
    extraction.schemas.CandidateDimensionQualifier's own __post_init__
    check)."""
    item = _valid_d2_qualifier_item(qualifier="APPROPRIATE_SPONSOR_COVERAGE")
    try:
        validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D2_QUALIFIER, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_dimension_qualifier_shape_rejects_d2_channel_referencing_stakeholder_observation():
    """Schema-level half of grounding prohibition (c) -- disallowed
    observation type. D2's supporting_observation_ref enum permits only
    ADOPTION_OBSERVATION."""
    item = _valid_d2_qualifier_item(
        supporting_observation_ref={"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
    )
    try:
        validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D2_QUALIFIER, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_dimension_qualifier_shape_rejects_d6_channel_referencing_adoption_observation():
    item = _valid_d6_qualifier_item(
        supporting_observation_ref={"observation_type": "ADOPTION_OBSERVATION", "index": 0},
    )
    try:
        validate_dimension_qualifier_shape(ObservationType.CANDIDATE_D6_QUALIFIER, item)
        assert False
    except ItemRejected as e:
        assert e.reason == RejectionReason.SCHEMA_INVALID


def test_validate_isolated_dimension_qualifier_top_level_shape_accepts_empty_envelope():
    """Milestone 4B isolated-classifier architecture. Each isolated call's
    envelope is single-channel (keyed by the dimension the call was for)
    and holds at most one item — an empty array is a valid, deliberate-
    abstention response for either channel."""
    validate_isolated_dimension_qualifier_top_level_shape(
        ObservationType.CANDIDATE_D2_QUALIFIER, {"candidate_d2_qualifiers": []},
    )
    validate_isolated_dimension_qualifier_top_level_shape(
        ObservationType.CANDIDATE_D6_QUALIFIER, {"candidate_d6_qualifiers": []},
    )


def test_validate_isolated_dimension_qualifier_top_level_shape_accepts_missing_keys():
    validate_isolated_dimension_qualifier_top_level_shape(ObservationType.CANDIDATE_D2_QUALIFIER, {})
    validate_isolated_dimension_qualifier_top_level_shape(ObservationType.CANDIDATE_D6_QUALIFIER, {})


def test_validate_isolated_dimension_qualifier_top_level_shape_rejects_unknown_top_level_key():
    import jsonschema
    try:
        validate_isolated_dimension_qualifier_top_level_shape(
            ObservationType.CANDIDATE_D2_QUALIFIER, {"candidate_d2_qualifiers": [], "confirmed": True},
        )
        assert False
    except jsonschema.exceptions.ValidationError:
        pass


def test_validate_isolated_dimension_qualifier_top_level_shape_rejects_wrong_channel_key():
    """The isolated envelope is dimension-scoped: a D2 call's response
    schema declares ONLY candidate_d2_qualifiers as a known property, so
    the D6 channel key is an unknown top-level key for that call (and
    vice versa) — this is the structural guarantee that a D2 call cannot
    even shape-validate a D6-shaped response, or vice versa."""
    import jsonschema
    try:
        validate_isolated_dimension_qualifier_top_level_shape(
            ObservationType.CANDIDATE_D2_QUALIFIER, {"candidate_d6_qualifiers": []},
        )
        assert False
    except jsonschema.exceptions.ValidationError:
        pass
    try:
        validate_isolated_dimension_qualifier_top_level_shape(
            ObservationType.CANDIDATE_D6_QUALIFIER, {"candidate_d2_qualifiers": []},
        )
        assert False
    except jsonschema.exceptions.ValidationError:
        pass


TESTS = [
    test_build_observation_accepts_valid_item, test_build_observation_derives_span_offsets_from_text,
    test_build_observation_rejects_prohibited_key,
    test_build_observation_rejects_model_set_system_field, test_build_observation_rejects_span_not_found,
    test_build_observation_rejects_ambiguous_span_text, test_build_observation_rejects_model_supplied_offsets,
    test_build_observation_rejects_unknown_evidence_id, test_build_observation_rejects_missing_required_field,
    test_missing_information_candidate_no_span_required, test_missing_information_rejects_unknown_reviewed_evidence_id,
    test_scan_for_prohibited_keys_finds_nested, test_scan_for_prohibited_keys_clean,
    test_resolve_source_span_derives_offsets_from_single_occurrence, test_resolve_source_span_rejects_zero_occurrences,
    test_resolve_source_span_rejects_ambiguous_multiple_occurrences, test_resolve_source_span_preserves_raw_item_on_rejection,
    test_validate_source_span_exact_match, test_validate_source_span_offset_mismatch,
    test_validate_source_span_out_of_bounds,
    test_validate_top_level_shape_rejects_unknown_key, test_validate_top_level_shape_accepts_empty,
    test_validate_contradiction_item_shape_accepts_valid, test_validate_contradiction_item_shape_rejects_boundary_violation,
    test_validate_candidate_classification_shape_accepts_valid_risk_signal,
    test_validate_candidate_classification_shape_rejects_boundary_violation,
    test_validate_candidate_classification_shape_rejects_source_evidence_id,
    test_validate_candidate_classification_shape_rejects_deferred_mechanism,
    test_validate_candidate_classification_shape_rejects_disallowed_ref_type,
    test_validate_candidate_classification_shape_accepts_valid_evidence_classification,
    test_validate_candidate_classification_shape_rejects_absence_basis,
    test_validate_dimension_qualifier_shape_accepts_valid_d2_item,
    test_validate_dimension_qualifier_shape_accepts_valid_d6_item,
    test_validate_dimension_qualifier_shape_rejects_boundary_violation,
    test_validate_dimension_qualifier_shape_rejects_source_span,
    test_validate_dimension_qualifier_shape_rejects_source_evidence_id,
    test_validate_dimension_qualifier_shape_rejects_d6_qualifier_on_d2_channel,
    test_validate_dimension_qualifier_shape_rejects_d2_channel_referencing_stakeholder_observation,
    test_validate_dimension_qualifier_shape_rejects_d6_channel_referencing_adoption_observation,
    test_validate_isolated_dimension_qualifier_top_level_shape_accepts_empty_envelope,
    test_validate_isolated_dimension_qualifier_top_level_shape_accepts_missing_keys,
    test_validate_isolated_dimension_qualifier_top_level_shape_rejects_unknown_top_level_key,
    test_validate_isolated_dimension_qualifier_top_level_shape_rejects_wrong_channel_key,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
