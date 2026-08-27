"""
Milestone 4B v3 — atomic-predicate + deterministic composition
architecture. Deterministic pipeline/schema tests (spec §19) for the new
response-contract change to run_dimension_qualifier_classification: for
exactly the two compound qualifiers scoped to this architecture
(AUTOMATION_RELIABLE_LOW_LOGIN_OK on D2, CHAMPION_LOST_NO_SUCCESSOR on
D6), the model is structurally barred from proposing the compound
qualifier name directly and instead proposes independently-grounded
atomic predicates on a sibling envelope key
(candidate_d2_atomic_predicates / candidate_d6_atomic_predicates), which
the application composes deterministically.

Kept in a dedicated file (rather than appended to
test_extraction_dimension_qualifier_pipeline.py) because this exercises a
structurally new response-contract path with its own failure modes,
mirroring how that file was itself kept separate from
test_extraction_pipeline_fake_provider.py. NO network calls anywhere in
this file -- every provider used here is FakeExtractionProvider.

Covers the required v3 checkpoint scenarios:
  1.  D2 full atomic-predicate set (3/3 grounded) composes
      AUTOMATION_RELIABLE_LOW_LOGIN_OK.
  2.  D2 partial set (2/3) is silent abstention -- no candidate, no
      rejection record.
  3.  D2 zero atomic predicates proposed -- no candidate, no rejection.
  4.  D6 full atomic-predicate set (2/2 grounded) composes
      CHAMPION_LOST_NO_SUCCESSOR.
  5.  D6 partial set (1/2) is silent abstention.
  6.  Ungrounded atomic-predicate evidence_text (not an exact substring of
      the observation's own source_span.text) is rejected with
      DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED, and does NOT
      count toward the required set.
  7.  Duplicate predicate_id within the same call: the second occurrence
      is rejected with DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE
      (never silently deduplicated); the first, if grounded, still
      counts.
  8.  Schema-level structural proof: the 2 compound qualifier values are
      impossible to emit via the direct candidate_d2_qualifiers /
      candidate_d6_qualifiers path (jsonschema.ValidationError).
  9.  Case-15/24/35-style regression fixtures: a departure-alone /
      automation-alone observation that supplies only ONE of the required
      predicates never composes the compound qualifier.
  10. The other 7 qualifiers (4 remaining D2, 3 remaining D6) are
      completely unaffected -- direct-path composition still works
      exactly as before.
  11. Prompt version is "v3".
  12. Provenance: dimension_qualifier_predicate_evidence contains EVERY
      grounded predicate, including from incomplete (non-composing)
      sets -- not just the predicates that contributed to a successful
      composition.
  13. Milestone 4B EXPLICIT-basis composition gate (DIMENSION_QUALIFIER_
      TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION): originally D2-only
      (approved after a live probe showed Case 35 Observation A composing
      from 2 grounded-but-INFERRED_CANDIDATE predicates); extended to D6
      after the normalized full-suite live run found the identical failure
      shape in Case 24 (CHAMPION_LOST_NO_SUCCESSOR composed from an
      EXPLICIT departure predicate + an INFERRED_CANDIDATE no-successor
      predicate grounded to "Marcus, our only point of contact"). Both
      dimensions now require every required predicate to be EXPLICIT for
      composition eligibility; an ineligible-but-complete set abstains
      exactly like an incomplete one, with every grounded predicate
      (EXPLICIT or INFERRED_CANDIDATE) still unconditionally preserved in
      provenance.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jsonschema
import pytest

from domain.enums import DimensionCode, Provenance
from domain.evidence import EvidenceObject

from extraction.enums import DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION, ObservationType, RejectionReason
from extraction.json_schemas import (
    CANDIDATE_D2_QUALIFIER_SCHEMA, CANDIDATE_D6_QUALIFIER_SCHEMA,
    D2_QUALIFIER_SCHEMA, D6_QUALIFIER_SCHEMA,
)
from extraction.pipeline import run_dimension_qualifier_classification, run_extraction
from extraction.prompts import DIMENSION_QUALIFIER_PROMPT_VERSION
from extraction.provider import FakeExtractionProvider


def _evidence(evidence_id, text, source="account_note"):
    return EvidenceObject(evidence_id, None, text, source, Provenance.USER_PROVIDED)


# ---------------------------------------------------------------------------
# D2 fixtures. ADOPTION_TEXT contains enough real language to ground all 3
# required atomic predicates as distinct substrings.
# ---------------------------------------------------------------------------
ADOPTION_TEXT = (
    "The nightly sync integration has completed successfully every night for six months. "
    "Direct user logins to the reporting workflow are rare. "
    "That is expected because the sync integration is what actually performs the reporting workflow."
)
D2_STAGE1_RESPONSE = {
    "adoption_observations": [
        {
            "source_evidence_id": "E1",
            "source_span": {"text": ADOPTION_TEXT},
            "basis": "EXPLICIT",
            "workflow_or_use_case": "nightly reporting sync",
            "observed_behavior": "automated sync, rare direct logins",
        }
    ],
}


def _d2_stage1_result():
    evidence = (_evidence("E1", ADOPTION_TEXT),)
    provider = FakeExtractionProvider(D2_STAGE1_RESPONSE)
    result = run_extraction(evidence, provider)
    assert not result.rejected, result.rejected
    assert len(result.accepted) == 1
    return result


D2_PREDICATE_RELIABLE = {
    "predicate_id": "RELIABLE_AUTOMATION_OPERATION", "basis": "EXPLICIT",
    "evidence_text": "completed successfully every night for six months",
}
D2_PREDICATE_LOW_LOGIN = {
    "predicate_id": "LOW_LOGIN_OR_MANUAL_ACTIVITY", "basis": "EXPLICIT",
    "evidence_text": "Direct user logins to the reporting workflow are rare",
}
D2_PREDICATE_EXPLAINED = {
    "predicate_id": "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION", "basis": "EXPLICIT",
    "evidence_text": "the sync integration is what actually performs the reporting workflow",
}


# ---------------------------------------------------------------------------
# D6 fixtures. STAKEHOLDER_TEXT contains enough real language to ground
# both required atomic predicates as distinct substrings.
# ---------------------------------------------------------------------------
STAKEHOLDER_TEXT = (
    "Our champion Jane Smith left the company last month and no one has taken over her role."
)
D6_STAGE1_RESPONSE = {
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


def _d6_stage1_result():
    evidence = (_evidence("E2", STAKEHOLDER_TEXT),)
    provider = FakeExtractionProvider(D6_STAGE1_RESPONSE)
    result = run_extraction(evidence, provider)
    assert not result.rejected, result.rejected
    assert len(result.accepted) == 1
    return result


D6_PREDICATE_DEPARTURE = {
    "predicate_id": "CONFIRMED_CHAMPION_DEPARTURE", "basis": "EXPLICIT",
    "evidence_text": "left the company last month",
}
D6_PREDICATE_NO_SUCCESSOR = {
    "predicate_id": "NO_SUCCESSOR_OR_CONTINUING_COVERAGE", "basis": "EXPLICIT",
    "evidence_text": "no one has taken over her role",
}


# ---------------------------------------------------------------------------
# D6 Case-24-shaped fixture -- the exact live-run finding that triggered the
# D6 EXPLICIT-basis gate extension (Milestone 4B normalized full-suite live
# run, prompt_v4_4b_dimqual_v3_2_normalized_full_eval1, disposed FAIL):
# CHAMPION_LOST_NO_SUCCESSOR composed from CONFIRMED_CHAMPION_DEPARTURE
# (EXPLICIT) + NO_SUCCESSOR_OR_CONTINUING_COVERAGE (INFERRED_CANDIDATE,
# grounded to "Marcus, our only point of contact" -- a plausible inference
# about sole-contact concentration, not an explicit statement that no
# successor or continuing coverage exists).
# ---------------------------------------------------------------------------
CASE24_STAKEHOLDER_TEXT = (
    "Our champion has left the company as of this quarter, and Marcus, "
    "our only point of contact, now fields any ad hoc requests."
)
CASE24_D6_STAGE1_RESPONSE = {
    "stakeholder_observations": [
        {
            "source_evidence_id": "E4",
            "source_span": {"text": CASE24_STAKEHOLDER_TEXT},
            "basis": "EXPLICIT",
            "person_identifier": "unnamed champion",
            "role": "champion",
        }
    ],
}


def _d6_case24_stage1_result():
    evidence = (_evidence("E4", CASE24_STAKEHOLDER_TEXT),)
    provider = FakeExtractionProvider(CASE24_D6_STAGE1_RESPONSE)
    result = run_extraction(evidence, provider)
    assert not result.rejected, result.rejected
    assert len(result.accepted) == 1
    return result


CASE24_PREDICATE_DEPARTURE = {
    "predicate_id": "CONFIRMED_CHAMPION_DEPARTURE", "basis": "EXPLICIT",
    "evidence_text": "has left the company as of this quarter",
}
CASE24_PREDICATE_NO_SUCCESSOR_INFERRED = {
    "predicate_id": "NO_SUCCESSOR_OR_CONTINUING_COVERAGE", "basis": "INFERRED_CANDIDATE",
    "evidence_text": "Marcus, our only point of contact",
}


# ===========================================================================
# 1. D2 full set composes AUTOMATION_RELIABLE_LOW_LOGIN_OK
# ===========================================================================
def test_d2_full_atomic_predicate_set_composes_compound_qualifier():
    result1 = _d2_stage1_result()
    response = {
        "candidate_d2_atomic_predicates": [D2_PREDICATE_RELIABLE, D2_PREDICATE_LOW_LOGIN, D2_PREDICATE_EXPLAINED],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert len(result2.candidate_d2_qualifiers) == 1
    composed = result2.candidate_d2_qualifiers[0]
    assert composed.qualifier == "AUTOMATION_RELIABLE_LOW_LOGIN_OK"
    assert composed.dimension == DimensionCode.D2
    assert composed.basis.value == "EXPLICIT"
    assert composed.resolved_observation_id == result1.accepted[0].system.observation_id
    assert composed.source_span.text == ADOPTION_TEXT
    assert result2.dimension_qualifier_failures == ()
    # Provenance: all 3 grounded predicates preserved.
    assert len(result2.dimension_qualifier_predicate_evidence) == 3
    predicate_ids = {e.predicate_id for e in result2.dimension_qualifier_predicate_evidence}
    assert predicate_ids == {
        "RELIABLE_AUTOMATION_OPERATION", "LOW_LOGIN_OR_MANUAL_ACTIVITY", "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION",
    }


# ===========================================================================
# 2. D2 partial set (2/3) -> silent abstention
# ===========================================================================
def test_d2_partial_atomic_predicate_set_is_silent_abstention():
    result1 = _d2_stage1_result()
    response = {
        "candidate_d2_atomic_predicates": [D2_PREDICATE_RELIABLE, D2_PREDICATE_LOW_LOGIN],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d2_qualifiers == ()
    assert result2.dimension_qualifier_failures == ()
    assert result2.dimension_qualifier_stage_failure is None
    # The rejected tuple must gain no new entries -- an incomplete set is
    # never a rejection, it is deliberate abstention.
    assert result2.rejected == result1.rejected
    # But the 2 grounded predicates are still preserved for audit.
    assert len(result2.dimension_qualifier_predicate_evidence) == 2


# ===========================================================================
# 3. D2 zero atomic predicates -> no candidate
# ===========================================================================
def test_d2_zero_atomic_predicates_produces_no_candidate():
    result1 = _d2_stage1_result()
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses={})
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.candidate_d2_qualifiers == ()
    assert result2.dimension_qualifier_predicate_evidence == ()


# ===========================================================================
# 4. D6 full set composes CHAMPION_LOST_NO_SUCCESSOR
# ===========================================================================
def test_d6_full_atomic_predicate_set_composes_compound_qualifier():
    result1 = _d6_stage1_result()
    response = {"candidate_d6_atomic_predicates": [D6_PREDICATE_DEPARTURE, D6_PREDICATE_NO_SUCCESSOR]}
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert len(result2.candidate_d6_qualifiers) == 1
    composed = result2.candidate_d6_qualifiers[0]
    assert composed.qualifier == "CHAMPION_LOST_NO_SUCCESSOR"
    assert composed.dimension == DimensionCode.D6
    assert composed.resolved_observation_id == result1.accepted[0].system.observation_id
    assert len(result2.dimension_qualifier_predicate_evidence) == 2


# ===========================================================================
# 5. D6 partial set (1/2) -> silent abstention. This is also the Case-24/15
#    style regression: a departure-alone observation must never compose.
# ===========================================================================
def test_d6_partial_atomic_predicate_set_is_silent_abstention():
    result1 = _d6_stage1_result()
    response = {"candidate_d6_atomic_predicates": [D6_PREDICATE_DEPARTURE]}
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d6_qualifiers == ()
    assert result2.dimension_qualifier_failures == ()
    assert result2.rejected == result1.rejected
    assert len(result2.dimension_qualifier_predicate_evidence) == 1
    assert result2.dimension_qualifier_predicate_evidence[0].predicate_id == "CONFIRMED_CHAMPION_DEPARTURE"


# ===========================================================================
# 6. Ungrounded evidence_text -> rejected, does not count toward the set
# ===========================================================================
def test_ungrounded_atomic_predicate_evidence_is_rejected_and_excluded_from_composition():
    result1 = _d6_stage1_result()
    bad_no_successor = {
        "predicate_id": "NO_SUCCESSOR_OR_CONTINUING_COVERAGE", "basis": "EXPLICIT",
        # Not a substring of STAKEHOLDER_TEXT at all.
        "evidence_text": "the account has been reassigned to a new success manager",
    }
    response = {"candidate_d6_atomic_predicates": [D6_PREDICATE_DEPARTURE, bad_no_successor]}
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    # Composition never happens -- only 1 of 2 required predicates grounded.
    assert result2.candidate_d6_qualifiers == ()
    new_rejected = [r for r in result2.rejected if r not in result1.rejected]
    assert len(new_rejected) == 1
    assert new_rejected[0].reason == RejectionReason.DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED
    assert new_rejected[0].observation_type == "candidate_d6_atomic_predicates"
    # Only the grounded predicate is preserved in provenance.
    assert len(result2.dimension_qualifier_predicate_evidence) == 1
    assert result2.dimension_qualifier_predicate_evidence[0].predicate_id == "CONFIRMED_CHAMPION_DEPARTURE"


# ===========================================================================
# 7. Duplicate predicate_id within the same call -> explicit rejection,
#    never silent deduplication; first (grounded) occurrence still counts.
# ===========================================================================
def test_duplicate_atomic_predicate_id_is_explicitly_rejected_not_silently_deduplicated():
    """The envelope's maxItems (3 for D2, matching the required-predicate
    count exactly) means a duplicate necessarily consumes a slot that
    could otherwise have supplied the missing third predicate -- so this
    fixture uses 2 required predicates + 1 duplicate (3 items, within the
    schema's maxItems limit) rather than all 3 required plus a 4th
    duplicate item (which would itself exceed maxItems and fail at the
    envelope-shape gate before ever reaching per-item duplicate
    detection). The duplicate is rejected explicitly; the missing third
    predicate means composition still correctly abstains -- duplicate
    rejection and set-completeness are independent checks, exercised
    together here."""
    result1 = _d2_stage1_result()
    duplicate_reliable = dict(D2_PREDICATE_RELIABLE)  # same predicate_id, same evidence_text
    response = {
        "candidate_d2_atomic_predicates": [D2_PREDICATE_RELIABLE, duplicate_reliable, D2_PREDICATE_LOW_LOGIN],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    new_rejected = [r for r in result2.rejected if r not in result1.rejected]
    assert len(new_rejected) == 1
    assert new_rejected[0].reason == RejectionReason.DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE
    # The required set is now incomplete (only RELIABLE + LOW_LOGIN
    # grounded; EXPLAINED is missing because the duplicate occupied the
    # slot instead) -- silent abstention, not a second rejection.
    assert result2.candidate_d2_qualifiers == ()
    assert result2.dimension_qualifier_failures == ()
    # Provenance never records the rejected duplicate -- only the 2
    # unique, grounded predicates.
    assert len(result2.dimension_qualifier_predicate_evidence) == 2
    predicate_ids = {e.predicate_id for e in result2.dimension_qualifier_predicate_evidence}
    assert predicate_ids == {"RELIABLE_AUTOMATION_OPERATION", "LOW_LOGIN_OR_MANUAL_ACTIVITY"}


# ===========================================================================
# 8. Schema-level: the 2 compound qualifiers are structurally impossible
#    via the direct proposal path.
# ===========================================================================
def test_compound_qualifiers_structurally_excluded_from_direct_schema_enum():
    assert "AUTOMATION_RELIABLE_LOW_LOGIN_OK" not in D2_QUALIFIER_SCHEMA["enum"]
    assert "CHAMPION_LOST_NO_SUCCESSOR" not in D6_QUALIFIER_SCHEMA["enum"]

    bad_d2_item = {
        "qualifier": "AUTOMATION_RELIABLE_LOW_LOGIN_OK", "basis": "EXPLICIT",
        "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad_d2_item, schema=CANDIDATE_D2_QUALIFIER_SCHEMA)

    bad_d6_item = {
        "qualifier": "CHAMPION_LOST_NO_SUCCESSOR", "basis": "EXPLICIT",
        "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad_d6_item, schema=CANDIDATE_D6_QUALIFIER_SCHEMA)


# ===========================================================================
# 9. Case-07/23/24/35-style regression: a single-half-condition observation
#    (via the model proposing only one grounded predicate) never composes,
#    even across several different partial combinations.
# ===========================================================================
@pytest.mark.parametrize("predicates", [
    [D2_PREDICATE_RELIABLE],
    [D2_PREDICATE_LOW_LOGIN],
    [D2_PREDICATE_EXPLAINED],
    [D2_PREDICATE_LOW_LOGIN, D2_PREDICATE_EXPLAINED],
])
def test_d2_every_incomplete_combination_abstains(predicates):
    result1 = _d2_stage1_result()
    response = {"candidate_d2_atomic_predicates": predicates}
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.candidate_d2_qualifiers == ()


# ===========================================================================
# 10. Regression: the other 7 qualifiers (unchanged direct-proposal path)
#     still compose exactly as before -- one D2, one D6 example.
# ===========================================================================
def test_unaffected_d2_qualifier_still_composes_via_direct_path():
    result1 = _d2_stage1_result()
    response = {
        "candidate_d2_qualifiers": [
            {
                "qualifier": "INTENDED_WORKFLOWS_OPERATING_NORMALLY", "basis": "EXPLICIT",
                "supporting_observation_ref": {"observation_type": "ADOPTION_OBSERVATION", "index": 0},
            }
        ],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert len(result2.candidate_d2_qualifiers) == 1
    assert result2.candidate_d2_qualifiers[0].qualifier == "INTENDED_WORKFLOWS_OPERATING_NORMALLY"


def test_unaffected_d6_qualifier_still_composes_via_direct_path():
    result1 = _d6_stage1_result()
    response = {
        "candidate_d6_qualifiers": [
            {
                "qualifier": "APPROPRIATE_SPONSOR_COVERAGE", "basis": "EXPLICIT",
                "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0},
            }
        ],
    }
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert len(result2.candidate_d6_qualifiers) == 1
    assert result2.candidate_d6_qualifiers[0].qualifier == "APPROPRIATE_SPONSOR_COVERAGE"


# ===========================================================================
# 11. Prompt version
# ===========================================================================
def test_dimension_qualifier_prompt_version_is_v3_2():
    # Bumped v3.1 -> v3.2 in the D2 atomic-predicate SECOND targeted
    # live-probe calibration checkpoint (RELIABLE_AUTOMATION_OPERATION
    # wording only -- see extraction/prompts.py's own v3.1 -> v3.2
    # history comment; the architecture/composer/schema this file's
    # other tests exercise is unchanged).
    assert DIMENSION_QUALIFIER_PROMPT_VERSION == "v3.2"


# ===========================================================================
# 12. Provenance across multiple observations in the same run: an
#     incomplete set's evidence is preserved exactly like a complete set's.
# ===========================================================================
def test_provenance_preserves_predicates_from_both_complete_and_incomplete_sets_in_same_run():
    text_a = ADOPTION_TEXT
    text_b = (
        "The finance team's invoice sync has been stable with zero failures for a year. "
        "There is nothing else stated about login activity for this workflow."
    )
    stage1 = {
        "adoption_observations": [
            {"source_evidence_id": "E1", "source_span": {"text": text_a}, "basis": "EXPLICIT",
             "workflow_or_use_case": "nightly reporting sync", "observed_behavior": "automated, rare logins"},
            {"source_evidence_id": "E3", "source_span": {"text": text_b}, "basis": "EXPLICIT",
             "workflow_or_use_case": "invoice sync", "observed_behavior": "stable"},
        ],
    }
    evidence = (_evidence("E1", text_a), _evidence("E3", text_b))
    provider1 = FakeExtractionProvider(stage1)
    result1 = run_extraction(evidence, provider1)
    assert len(result1.accepted) == 2

    obs_a = next(o for o in result1.accepted if o.source_evidence_id == "E1")
    obs_b = next(o for o in result1.accepted if o.source_evidence_id == "E3")

    def _router(dimension, observation, repair_hint=None):
        if observation.system.observation_id == obs_a.system.observation_id:
            return {
                "candidate_d2_atomic_predicates": [
                    D2_PREDICATE_RELIABLE, D2_PREDICATE_LOW_LOGIN, D2_PREDICATE_EXPLAINED,
                ],
            }
        # obs_b: only ONE grounded predicate -- an incomplete set.
        return {
            "candidate_d2_atomic_predicates": [
                {
                    "predicate_id": "RELIABLE_AUTOMATION_OPERATION", "basis": "EXPLICIT",
                    "evidence_text": "has been stable with zero failures for a year",
                },
            ],
        }

    provider2 = FakeExtractionProvider(stage1, dimension_qualifier_responses=_router)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    # Only obs_a's full set composed.
    assert len(result2.candidate_d2_qualifiers) == 1
    assert result2.candidate_d2_qualifiers[0].resolved_observation_id == obs_a.system.observation_id

    # But provenance preserves ALL 4 grounded predicates across both
    # observations -- 3 from the complete set, 1 from the incomplete one.
    assert len(result2.dimension_qualifier_predicate_evidence) == 4
    by_obs = {}
    for e in result2.dimension_qualifier_predicate_evidence:
        by_obs.setdefault(e.resolved_observation_id, []).append(e.predicate_id)
    assert len(by_obs[obs_a.system.observation_id]) == 3
    assert len(by_obs[obs_b.system.observation_id]) == 1


# ===========================================================================
# 13. Milestone 4B D2 EXPLICIT-basis composition gate (architecture
#     checkpoint, approved after a live probe on Prompt v3.2 showed Case 35
#     Observation A completing its 3-predicate D2 set via 2 grounded-but-
#     INFERRED_CANDIDATE predicates on a reliability-only observation).
#     Scoped to D2 only via DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_
#     BASIS_FOR_COMPOSITION -- completeness is no longer sufficient for D2;
#     every required predicate must ALSO be basis == EXPLICIT. D6 is
#     unaffected. This governs composition eligibility only -- provenance
#     (dimension_qualifier_predicate_evidence) still records every grounded
#     predicate regardless of whether the gate blocks composition.
# ===========================================================================
def test_explicit_basis_composition_gate_now_covers_both_d2_and_d6():
    # Direct structural pin on the map itself, so an accidental flip is
    # caught immediately regardless of downstream behavior. Originally
    # D2=True / D6=False (D2-only checkpoint); extended to D6=True after
    # the normalized full-suite live run found the identical failure shape
    # in Case 24 (see the D6-scoped tests below and extraction/enums.py's
    # history comment on DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_
    # FOR_COMPOSITION for the full rationale).
    assert DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION[
        ObservationType.CANDIDATE_D2_QUALIFIER
    ] is True
    assert DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION[
        ObservationType.CANDIDATE_D6_QUALIFIER
    ] is True


def test_d2_complete_set_with_one_inferred_candidate_predicate_does_not_compose():
    result1 = _d2_stage1_result()
    inferred_explained = dict(D2_PREDICATE_EXPLAINED, basis="INFERRED_CANDIDATE")
    response = {
        "candidate_d2_atomic_predicates": [D2_PREDICATE_RELIABLE, D2_PREDICATE_LOW_LOGIN, inferred_explained],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    # Complete (3/3 grounded) but NOT all EXPLICIT -> ineligible, abstains
    # exactly like an incomplete set: no candidate, no rejection record.
    assert result2.candidate_d2_qualifiers == ()
    assert result2.dimension_qualifier_failures == ()
    assert result2.rejected == result1.rejected
    # Provenance: all 3 grounded predicates still preserved for audit even
    # though composition was blocked -- collection and composition are
    # separate steps.
    assert len(result2.dimension_qualifier_predicate_evidence) == 3
    predicate_ids = {e.predicate_id for e in result2.dimension_qualifier_predicate_evidence}
    assert predicate_ids == {
        "RELIABLE_AUTOMATION_OPERATION", "LOW_LOGIN_OR_MANUAL_ACTIVITY", "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION",
    }


def test_d2_case_35_observation_a_shape_does_not_compose():
    # Direct regression pin for the exact live-probe finding that
    # authorized this gate: a reliability-only observation whose text
    # never states login/activity levels, where the model nonetheless
    # grounds LOW_LOGIN_OR_MANUAL_ACTIVITY and LOW_ACTIVITY_EXPLAINED_BY_
    # AUTOMATION to real substrings of the SAME reliability sentence and
    # self-labels both INFERRED_CANDIDATE.
    result1 = _d2_stage1_result()
    response = {
        "candidate_d2_atomic_predicates": [
            D2_PREDICATE_RELIABLE,  # EXPLICIT -- genuinely supported
            dict(D2_PREDICATE_LOW_LOGIN, basis="INFERRED_CANDIDATE",
                 evidence_text="completed successfully every night for six months"),
            dict(D2_PREDICATE_EXPLAINED, basis="INFERRED_CANDIDATE",
                 evidence_text="completed successfully every night for six months"),
        ],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d2_qualifiers == ()
    assert len(result2.dimension_qualifier_predicate_evidence) == 3


def test_d2_gate_is_symmetric_across_which_predicate_is_inferred():
    # The gate must not be accidentally keyed to a specific predicate_id --
    # INFERRED_CANDIDATE on RELIABLE_AUTOMATION_OPERATION (with the other 2
    # EXPLICIT) must also block composition.
    result1 = _d2_stage1_result()
    inferred_reliable = dict(D2_PREDICATE_RELIABLE, basis="INFERRED_CANDIDATE")
    response = {
        "candidate_d2_atomic_predicates": [inferred_reliable, D2_PREDICATE_LOW_LOGIN, D2_PREDICATE_EXPLAINED],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d2_qualifiers == ()
    assert len(result2.dimension_qualifier_predicate_evidence) == 3


def test_d6_departure_explicit_no_successor_inferred_now_abstains_preserving_both_predicates():
    # UPDATED, not deleted: this test previously proved a complete D6 set
    # with one INFERRED_CANDIDATE predicate still composes (D6 was
    # explicitly NOT scoped by the D2-only gate). The normalized full-suite
    # live run (Case 24) found the same failure shape in D6 that motivated
    # the original D2 gate, so the governing policy changed: D6 now also
    # requires EXPLICIT basis on every required predicate for composition
    # eligibility. Required regression check 2 of the D6 extension
    # checkpoint. Complete (2/2 grounded) but NOT all EXPLICIT -> now
    # ineligible; abstains exactly like an incomplete set: no candidate, no
    # rejection record.
    result1 = _d6_stage1_result()
    inferred_no_successor = dict(D6_PREDICATE_NO_SUCCESSOR, basis="INFERRED_CANDIDATE")
    response = {"candidate_d6_atomic_predicates": [D6_PREDICATE_DEPARTURE, inferred_no_successor]}
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d6_qualifiers == ()
    assert result2.dimension_qualifier_failures == ()
    assert result2.rejected == result1.rejected
    # Both grounded predicates remain preserved in provenance/audit even
    # though composition was blocked -- collection and composition remain
    # separate steps, unchanged by this extension.
    assert len(result2.dimension_qualifier_predicate_evidence) == 2
    predicate_ids = {e.predicate_id for e in result2.dimension_qualifier_predicate_evidence}
    assert predicate_ids == {"CONFIRMED_CHAMPION_DEPARTURE", "NO_SUCCESSOR_OR_CONTINUING_COVERAGE"}


def test_d6_complete_set_both_explicit_still_composes():
    # Required regression check 1 of the D6 extension checkpoint: the
    # legitimate, fully-EXPLICIT case must keep composing under the
    # extended gate, with composed basis EXPLICIT.
    result1 = _d6_stage1_result()
    response = {"candidate_d6_atomic_predicates": [D6_PREDICATE_DEPARTURE, D6_PREDICATE_NO_SUCCESSOR]}
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert len(result2.candidate_d6_qualifiers) == 1
    composed = result2.candidate_d6_qualifiers[0]
    assert composed.qualifier == "CHAMPION_LOST_NO_SUCCESSOR"
    assert composed.basis.value == "EXPLICIT"


def test_d6_case_24_predicate_shape_abstains():
    # Required regression check 3 of the D6 extension checkpoint: direct
    # regression pin for the exact live finding that triggered this
    # extension -- CONFIRMED_CHAMPION_DEPARTURE EXPLICIT +
    # NO_SUCCESSOR_OR_CONTINUING_COVERAGE INFERRED_CANDIDATE, grounded to
    # "Marcus, our only point of contact" (a plausible inference about
    # sole-contact concentration, not an explicit statement that no
    # successor or continuing coverage exists).
    result1 = _d6_case24_stage1_result()
    response = {
        "candidate_d6_atomic_predicates": [
            CASE24_PREDICATE_DEPARTURE, CASE24_PREDICATE_NO_SUCCESSOR_INFERRED,
        ],
    }
    provider2 = FakeExtractionProvider(CASE24_D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d6_qualifiers == ()
    assert len(result2.dimension_qualifier_predicate_evidence) == 2


def test_d6_gate_is_symmetric_across_which_predicate_is_inferred():
    # Required regression check 4 of the D6 extension checkpoint: the gate
    # must not be accidentally keyed to a specific predicate_id --
    # INFERRED_CANDIDATE on CONFIRMED_CHAMPION_DEPARTURE (with NO_SUCCESSOR_
    # OR_CONTINUING_COVERAGE EXPLICIT) must also block composition.
    result1 = _d6_stage1_result()
    inferred_departure = dict(D6_PREDICATE_DEPARTURE, basis="INFERRED_CANDIDATE")
    response = {"candidate_d6_atomic_predicates": [inferred_departure, D6_PREDICATE_NO_SUCCESSOR]}
    provider2 = FakeExtractionProvider(D6_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert result2.candidate_d6_qualifiers == ()
    assert len(result2.dimension_qualifier_predicate_evidence) == 2


def test_d6_case_40_style_explicit_departure_and_explicit_no_successor_still_composes():
    # Required regression check 5 of the D6 extension checkpoint, using the
    # Case-40 benchmark shape directly: "Our champion Priya Nair has left
    # the company as of last week, and the account team confirms no one
    # else has taken over her responsibilities" -- both facts explicit,
    # single observation. Must keep composing under the extended gate
    # exactly as it did before this checkpoint.
    case40_text = (
        "Our champion Priya Nair has left the company as of last week, and "
        "the account team confirms no one else has taken over her responsibilities."
    )
    stage1 = {
        "stakeholder_observations": [
            {
                "source_evidence_id": "E5",
                "source_span": {"text": case40_text},
                "basis": "EXPLICIT",
                "person_identifier": "Priya Nair",
                "role": "champion",
            }
        ],
    }
    evidence = (_evidence("E5", case40_text),)
    provider1 = FakeExtractionProvider(stage1)
    result1 = run_extraction(evidence, provider1)
    assert not result1.rejected, result1.rejected

    response = {
        "candidate_d6_atomic_predicates": [
            {"predicate_id": "CONFIRMED_CHAMPION_DEPARTURE", "basis": "EXPLICIT",
             "evidence_text": "has left the company as of last week"},
            {"predicate_id": "NO_SUCCESSOR_OR_CONTINUING_COVERAGE", "basis": "EXPLICIT",
             "evidence_text": "the account team confirms no one else has taken over her responsibilities"},
        ],
    }
    provider2 = FakeExtractionProvider(stage1, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert len(result2.candidate_d6_qualifiers) == 1
    assert result2.candidate_d6_qualifiers[0].qualifier == "CHAMPION_LOST_NO_SUCCESSOR"
    assert result2.candidate_d6_qualifiers[0].basis.value == "EXPLICIT"


def test_d2_all_explicit_set_still_composes_with_explicit_composed_basis():
    # Regression pin: the gate must not block the legitimate all-EXPLICIT
    # case it was designed to keep passing.
    result1 = _d2_stage1_result()
    response = {
        "candidate_d2_atomic_predicates": [D2_PREDICATE_RELIABLE, D2_PREDICATE_LOW_LOGIN, D2_PREDICATE_EXPLAINED],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)

    assert len(result2.candidate_d2_qualifiers) == 1
    assert result2.candidate_d2_qualifiers[0].basis.value == "EXPLICIT"


def test_d2_behavior_unchanged_by_d6_gate_extension():
    # Required regression check 6 of the D6 extension checkpoint: D2's own
    # gate behavior must be completely unaffected by extending the gate to
    # D6. Direct regression pin re-running the exact Case-35-shaped D2
    # scenario the D2 gate was originally built for.
    result1 = _d2_stage1_result()
    response = {
        "candidate_d2_atomic_predicates": [
            D2_PREDICATE_RELIABLE,
            dict(D2_PREDICATE_LOW_LOGIN, basis="INFERRED_CANDIDATE"),
            dict(D2_PREDICATE_EXPLAINED, basis="INFERRED_CANDIDATE"),
        ],
    }
    provider2 = FakeExtractionProvider(D2_STAGE1_RESPONSE, dimension_qualifier_responses=response)
    result2 = run_dimension_qualifier_classification(result1, provider2)
    assert result2.candidate_d2_qualifiers == ()
    assert len(result2.dimension_qualifier_predicate_evidence) == 3


TESTS = [
    test_d2_full_atomic_predicate_set_composes_compound_qualifier,
    test_d2_partial_atomic_predicate_set_is_silent_abstention,
    test_d2_zero_atomic_predicates_produces_no_candidate,
    test_d6_full_atomic_predicate_set_composes_compound_qualifier,
    test_d6_partial_atomic_predicate_set_is_silent_abstention,
    test_ungrounded_atomic_predicate_evidence_is_rejected_and_excluded_from_composition,
    test_duplicate_atomic_predicate_id_is_explicitly_rejected_not_silently_deduplicated,
    test_compound_qualifiers_structurally_excluded_from_direct_schema_enum,
    test_unaffected_d2_qualifier_still_composes_via_direct_path,
    test_unaffected_d6_qualifier_still_composes_via_direct_path,
    test_dimension_qualifier_prompt_version_is_v3_2,
    test_provenance_preserves_predicates_from_both_complete_and_incomplete_sets_in_same_run,
    test_explicit_basis_composition_gate_now_covers_both_d2_and_d6,
    test_d2_complete_set_with_one_inferred_candidate_predicate_does_not_compose,
    test_d2_case_35_observation_a_shape_does_not_compose,
    test_d2_gate_is_symmetric_across_which_predicate_is_inferred,
    test_d6_departure_explicit_no_successor_inferred_now_abstains_preserving_both_predicates,
    test_d2_all_explicit_set_still_composes_with_explicit_composed_basis,
    test_d6_complete_set_both_explicit_still_composes,
    test_d6_case_24_predicate_shape_abstains,
    test_d6_gate_is_symmetric_across_which_predicate_is_inferred,
    test_d6_case_40_style_explicit_departure_and_explicit_no_successor_still_composes,
    test_d2_behavior_unchanged_by_d6_gate_extension,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    for predicates in (
        [D2_PREDICATE_RELIABLE], [D2_PREDICATE_LOW_LOGIN], [D2_PREDICATE_EXPLAINED],
        [D2_PREDICATE_LOW_LOGIN, D2_PREDICATE_EXPLAINED],
    ):
        test_d2_every_incomplete_combination_abstains(predicates)
        print("PASS  test_d2_every_incomplete_combination_abstains")
    print(f"\n{len(TESTS) + 4}/{len(TESTS) + 4} passed")
