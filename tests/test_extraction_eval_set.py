"""
Structural regression check on eval/labeled_set.yaml — no model calls.
Confirms the labeled fixture itself stays well-formed and every declared
span_substrings entry genuinely resolves against its source text, so a
live evaluation run is always scored against a fixture that is itself
correct (not silently drifted).

Milestone 2B.1: updated for the 20-case, role/multi-span-substring
labeled-set schema (evaluation-only change; the 15 original case IDs and
source texts are unchanged, only re-annotated).

Milestone 2B.2: updated for the 23-case schema after the OC-01 ontology
clarification. Case 12's ontology_ambiguity/alternate_types accommodation
is retired (canonical type ADOPTION_OBSERVATION only); case 20's side-a
label was corrected from OBJECTIVE_CANDIDATE to STRATEGIC_OBSERVATION;
three new cases (21-23) were added targeting the OC-01 boundary directly.
Prompt text also changed (v1 -> v2) in this same milestone, so a
PROMPT_VERSION guard test was added here as well (spec item 6: "add
prompt-version tests if useful").

Milestone 2C: updated for the 33-case schema after Candidate CHDM
Classification Extraction was added. 10 new cases (24-33) exercise
`expected_candidate_risk_signals` / `expected_candidate_evidence_
classifications` -- see labeled_set.yaml's own Milestone 2C revision
note for the exact per-case field shape. Prompt text changed again
(v2 -> v3), so the PROMPT_VERSION guard test is updated accordingly.

Milestone 4B: updated for the 42-case schema after D2/D6 Candidate
Qualifier Extraction was added. 9 new cases (34-42) exercise
`expected_dimension_d2_qualifiers` / `expected_dimension_d6_qualifiers`
-- see labeled_set.yaml's own Milestone 4B revision note for the exact
per-case field shape. PROMPT_VERSION (stage 1) is UNCHANGED at v4 (the
stage-2 classifier has its own, separate DIMENSION_QUALIFIER_PROMPT_
VERSION, guarded by its own test below) -- the two-stage architecture
means stage 1's prompt text was never touched by this milestone.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Optional

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extraction.prompts import (
    DIMENSION_QUALIFIER_PROMPT_VERSION, PROMPT_VERSION, _ISOLATED_D2_SYSTEM_PROMPT, _ISOLATED_D6_SYSTEM_PROMPT,
)

LABELED_SET_PATH = ROOT / "eval" / "labeled_set.yaml"

REQUIRED_CASE_ID_PREFIXES = [f"{i:02d}_" for i in range(1, 43)]
VALID_D2_QUALIFIERS = {
    "INTENDED_WORKFLOWS_OPERATING_NORMALLY", "AUTOMATION_RELIABLE_LOW_LOGIN_OK",
    "NARROW_BREADTH_OR_CONCENTRATION", "WORKFLOWS_NOT_OCCURRING",
    "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED",
}
VALID_D6_QUALIFIERS = {
    "APPROPRIATE_SPONSOR_COVERAGE", "CHAMPION_LOST_NO_SUCCESSOR",
    "CHAMPION_DEPARTURE_UNCONFIRMED", "SUCCESSION_UNCLEAR_OR_CONCENTRATED",
}
VALID_ROLES = {"primary", "supporting", "optional-valid"}
# CR-03 removed under the PMO Option B decision (Milestone 2C MVP scope
# reduction): deferred from AI-automated candidate classification only --
# it remains fully governed deterministically (registry/risk_mechanisms.yaml,
# engine/risk_engine.py), untouched by this extraction-layer benchmark set.
VALID_MECHANISMS = {"CR-01", "CR-02", "CR-08"}
VALID_SEVERITY_TIERS = {"WATCH", "MATERIAL", "CRITICAL"}
VALID_EVIDENCE_BASES = {
    "PROXY_SUPPORTED", "MEASURED_OPERATIONAL_EVIDENCE", "CUSTOMER_CONFIRMED", "INDEPENDENTLY_VERIFIED",
}
VALID_SUPPORTS = {"ACHIEVED", "PROGRESSING", "NOT_ACHIEVED"}


def _load():
    with open(LABELED_SET_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _source_text_for(case: dict, source: Optional[str]) -> str:
    if source:
        return case.get(f"source_text_{source}", "")
    return case.get("source_text", case.get("source_text_a", ""))


def test_labeled_set_has_42_cases():
    data = _load()
    assert len(data["cases"]) == 42


def test_labeled_set_covers_all_42_required_categories():
    data = _load()
    ids = [c["id"] for c in data["cases"]]
    for prefix in REQUIRED_CASE_ID_PREFIXES:
        assert any(i.startswith(prefix) for i in ids), f"missing case category {prefix}"


def test_case_ids_are_unique():
    data = _load()
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids))


def test_every_expected_span_substring_resolves_against_its_source_text():
    data = _load()
    for case in data["cases"]:
        for eo in case.get("expected_observations", []):
            source_text = _source_text_for(case, eo.get("source"))
            for sub in eo.get("span_substrings", []):
                assert sub in source_text, (
                    f"{case['id']}: span_substring {sub!r} not found in "
                    f"source_text{'_' + eo['source'] if eo.get('source') else ''}"
                )


def test_every_expected_observation_has_a_valid_role():
    data = _load()
    for case in data["cases"]:
        for eo in case.get("expected_observations", []):
            assert eo.get("role") in VALID_ROLES, (
                f"{case['id']}: expected_observations entry missing a valid role "
                f"(got {eo.get('role')!r})"
            )


def test_grounded_but_irrelevant_permitted_spans_resolve_and_are_scoped_to_case_14():
    data = _load()
    for case in data["cases"]:
        irrelevant = case.get("grounded_but_irrelevant_permitted", [])
        if not irrelevant:
            continue
        # Milestone 2B.1 Case 14 disposition: this list exists specifically
        # to score case 14's known irrelevant-but-grounded content. Any
        # OTHER case using it would be a silent benchmark loosening that
        # was never reviewed/authorized -- fail loudly if that happens.
        assert case["id"].startswith("14_"), (
            f"{case['id']}: grounded_but_irrelevant_permitted is only authorized "
            f"for case 14 (Milestone 2B.1 Case 14 disposition) — found elsewhere."
        )
        for irr in irrelevant:
            assert irr["span_substrings"], f"{case['id']}: irrelevant entry with no span_substrings"
            for sub in irr["span_substrings"]:
                assert sub in case["source_text"], (
                    f"{case['id']}: grounded_but_irrelevant_permitted span {sub!r} "
                    f"not found in source_text"
                )


def test_no_case_has_ontology_ambiguity_accommodation():
    """Milestone 2B.2 (OC-01 ontology clarification): the Milestone 2B.1
    Case 12b `alternate_types`/`ontology_ambiguity` accommodation was a
    scoped, temporary evaluation accommodation pending ontology review,
    not a permanent feature of the benchmark. That review (OC-01/OC-02)
    concluded case 12b is canonically ADOPTION_OBSERVATION only, so the
    accommodation is retired. This test now asserts NO case uses it —
    superseding the prior test (`test_ontology_ambiguity_is_scoped_to_
    case_12_only`) whose premise (case 12 legitimately carries it) no
    longer holds. If a future case reintroduces this marker, that is a
    new ontology question requiring its own separate review, not a
    silent reversion to this retired accommodation."""
    data = _load()
    for case in data["cases"]:
        for eo in case.get("expected_observations", []):
            assert not eo.get("ontology_ambiguity") and not eo.get("alternate_types"), (
                f"{case['id']}: ontology_ambiguity/alternate_types found, but this "
                f"accommodation was retired by the OC-01 ontology clarification "
                f"(Milestone 2B.2) — any reintroduction needs its own review"
            )


def test_case_12_has_no_alternate_types():
    """Milestone 2B.2 Case 12b disposition (OC-01/OC-02): the "average
    close time unchanged from baseline" fact is canonically
    ADOPTION_OBSERVATION only. A model emitting OBJECTIVE_CANDIDATE for
    it must score WRONG_TYPE, not be credited via the retired
    ontology-ambiguity accommodation — mirrors the existing Case 05
    guard below."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("12_"))
    for eo in case["expected_observations"]:
        assert not eo.get("alternate_types")
        assert not eo.get("ontology_ambiguity")
        if eo.get("source") == "b":
            assert eo["type"] == "ADOPTION_OBSERVATION"


def test_case_20_side_a_is_strategic_not_objective():
    """Milestone 2B.2 Case 20 disposition (OC-01/OC-03): "implementation
    was completed ahead of schedule" never states a desired future
    outcome -- it is project/status information, exactly like side b.
    Both sides are canonically STRATEGIC_OBSERVATION. This corrects the
    original Milestone 2B.1 benchmark label, which was itself wrong
    under the clarified ontology, not just a model deficiency."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("20_"))
    types_by_source = {eo.get("source"): eo["type"] for eo in case["expected_observations"]}
    assert types_by_source.get("a") == "STRATEGIC_OBSERVATION"
    assert types_by_source.get("b") == "STRATEGIC_OBSERVATION"


def test_case_05_has_no_alternate_types():
    """Milestone 2B.1 Case 05 disposition: the usage-growth fact stays
    canonically ADOPTION_OBSERVATION only. A model emitting
    OBJECTIVE_CANDIDATE for it must score WRONG_TYPE, not be credited via
    a benchmark-widening alternate type."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("05_"))
    for eo in case["expected_observations"]:
        assert eo["type"] == "ADOPTION_OBSERVATION"
        assert not eo.get("alternate_types")
        assert not eo.get("ontology_ambiguity")


def test_case_17_is_strategic_not_objective():
    """Milestone 2B.2 Case 17 disposition (OC-01/OC-03, approved
    separately from the initial two corrections): "completed phase 1 of
    the rollout on schedule; phase 2 configuration begins next week"
    describes rollout/project status and milestone progression, not a
    stated desired customer outcome. Canonical type is
    STRATEGIC_OBSERVATION, aligned with the Case 20 correction. A model
    emitting OBJECTIVE_CANDIDATE for this fact must score WRONG_TYPE."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("17_"))
    for eo in case["expected_observations"]:
        assert eo["type"] == "STRATEGIC_OBSERVATION"


def test_every_case_has_a_description_and_source_text():
    data = _load()
    for case in data["cases"]:
        assert case.get("description")
        assert case.get("source_text") or (case.get("source_text_a") and case.get("source_text_b"))


def test_case_14_is_the_designated_empty_extraction_case():
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("14_"))
    assert case["expected_observations"] == []
    # Milestone 2B.1: two grounded-but-irrelevant spans are a known,
    # explicitly permitted exception -- not part of expected_observations.
    assert len(case.get("grounded_but_irrelevant_permitted", [])) == 2


def test_case_12_is_the_designated_contradiction_case():
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("12_"))
    assert case.get("expected_contradiction", {}).get("required") is True


def test_case_20_is_the_second_contradiction_case():
    """Milestone 2B.1 spec §8/§9: at least one additional contradiction
    example beyond case 12."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("20_"))
    assert case.get("expected_contradiction", {}).get("required") is True
    assert "source_text_a" in case and "source_text_b" in case


def test_at_least_two_contradiction_cases_total():
    data = _load()
    contradiction_cases = [c for c in data["cases"] if c.get("expected_contradiction", {}).get("required")]
    assert len(contradiction_cases) >= 2


def test_case_15_permits_only_inferred_not_explicit_derived_claim():
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("15_"))
    inferred = case.get("inferred_candidates_permitted", [])
    assert len(inferred) >= 1
    assert all(ic["basis"] == "INFERRED_CANDIDATE" for ic in inferred)


def test_case_16_is_the_inferred_objective_case():
    """Milestone 2B.1 new case: objective evidence that is inferable but
    not explicit — must never surface as an EXPLICIT-basis observation."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("16_"))
    assert case["expected_observations"] == []
    inferred = case.get("inferred_candidates_permitted", [])
    assert len(inferred) >= 1
    assert all(ic["basis"] == "INFERRED_CANDIDATE" for ic in inferred)


def test_new_ambiguity_cases_16_through_20_exist_and_are_well_formed():
    data = _load()
    ids = {c["id"] for c in data["cases"]}
    for i in range(16, 21):
        prefix = f"{i:02d}_"
        assert any(cid.startswith(prefix) for cid in ids), f"missing new Milestone 2B.1 case {prefix}"


def test_new_oc01_validation_cases_21_through_23_exist_and_are_well_formed():
    """Milestone 2B.2 spec item 4: three small, targeted validation cases
    probing the OC-01 boundary -- inferred-but-not-explicit objective (21),
    pure strategic/project-status with no usage language (22), pure
    adoption/milestone with no goal language (23)."""
    data = _load()
    by_id = {c["id"]: c for c in data["cases"]}
    for i in range(21, 24):
        prefix = f"{i:02d}_"
        assert any(cid.startswith(prefix) for cid in by_id), f"missing new Milestone 2B.2 case {prefix}"

    case21 = next(c for c in data["cases"] if c["id"].startswith("21_"))
    assert case21["expected_observations"] == [] or all(
        eo["type"] != "OBJECTIVE_CANDIDATE" for eo in case21["expected_observations"]
    )
    inferred21 = case21.get("inferred_candidates_permitted", [])
    assert len(inferred21) >= 1
    assert all(ic["basis"] == "INFERRED_CANDIDATE" for ic in inferred21)
    assert any(ic.get("type") == "OBJECTIVE_CANDIDATE" for ic in inferred21)

    case22 = next(c for c in data["cases"] if c["id"].startswith("22_"))
    types22 = {eo["type"] for eo in case22["expected_observations"]}
    assert types22 == {"STRATEGIC_OBSERVATION"}

    case23 = next(c for c in data["cases"] if c["id"].startswith("23_"))
    types23 = {eo["type"] for eo in case23["expected_observations"]}
    assert types23 == {"ADOPTION_OBSERVATION"}


def test_prompt_version_is_v4():
    """PMO Option B decision (Milestone 2C MVP scope reduction, approved on
    the prompt_v3_refine2_2c_eval3 CONDITIONAL decision): CR-03 (Commercial
    Continuity) is deferred from AI-automated candidate-risk-signal
    classification -- it never once produced a correct candidate across
    three live rounds, and was newly confused with CR-08 in the third. The
    MVP candidate_risk_signals mechanism set is now exactly CR-01/CR-02/
    CR-08. This is a structural scope decision, not a v3.x calibration
    refinement, hence the version-family break from v3.2 to v4 rather than
    v3.3. PROMPT_VERSION must be bumped so every evaluation report is
    traceable to the exact prompt text that produced it."""
    assert PROMPT_VERSION == "v4"


def test_new_2c_cases_24_through_33_exist():
    data = _load()
    ids = {c["id"] for c in data["cases"]}
    for i in range(24, 34):
        prefix = f"{i:02d}_"
        assert any(cid.startswith(prefix) for cid in ids), f"missing new Milestone 2C case {prefix}"


def test_candidate_risk_signal_span_substrings_resolve_and_are_well_formed():
    data = _load()
    for case in data["cases"]:
        for crs in case.get("expected_candidate_risk_signals", []):
            source_text = _source_text_for(case, crs.get("source"))
            assert crs["mechanism"] in VALID_MECHANISMS, (
                f"{case['id']}: expected_candidate_risk_signals mechanism "
                f"{crs['mechanism']!r} is not an MVP-implemented mechanism"
            )
            assert crs["proposed_severity_tier"] in VALID_SEVERITY_TIERS, (
                f"{case['id']}: expected_candidate_risk_signals proposed_severity_tier "
                f"{crs['proposed_severity_tier']!r} is not WATCH/MATERIAL/CRITICAL"
            )
            for sub in crs["span_substrings"]:
                assert sub in source_text, (
                    f"{case['id']}: candidate risk signal span_substring {sub!r} not found in source_text"
                )
            for sub in crs["supporting_observation_span_substrings"]:
                assert sub in source_text, (
                    f"{case['id']}: candidate risk signal supporting_observation_span_substrings "
                    f"{sub!r} not found in source_text"
                )
                # The supporting reference must identify a REAL expected
                # observation in this same case -- not a dangling pointer to
                # text that merely happens to appear in the source.
                assert any(
                    sub in s for eo in case.get("expected_observations", []) for s in eo.get("span_substrings", [])
                ), (
                    f"{case['id']}: candidate risk signal's supporting_observation_span_substrings "
                    f"{sub!r} does not match any expected_observations entry"
                )


def test_candidate_evidence_classification_span_substrings_resolve_and_are_well_formed():
    data = _load()
    for case in data["cases"]:
        for cec in case.get("expected_candidate_evidence_classifications", []):
            source_text = _source_text_for(case, cec.get("source"))
            assert cec["proposed_basis"] in VALID_EVIDENCE_BASES, (
                f"{case['id']}: expected_candidate_evidence_classifications proposed_basis "
                f"{cec['proposed_basis']!r} is not one of the 4 MVP-proposable bases"
            )
            assert cec["supports"] in VALID_SUPPORTS, (
                f"{case['id']}: expected_candidate_evidence_classifications supports "
                f"{cec['supports']!r} is not ACHIEVED/PROGRESSING/NOT_ACHIEVED"
            )
            for sub in cec["span_substrings"]:
                assert sub in source_text, (
                    f"{case['id']}: candidate evidence classification span_substring {sub!r} not found in source_text"
                )
            for sub in cec["supporting_observation_span_substrings"]:
                assert sub in source_text, (
                    f"{case['id']}: candidate evidence classification supporting_observation_span_substrings "
                    f"{sub!r} not found in source_text"
                )
                assert any(
                    sub in s for eo in case.get("expected_observations", []) for s in eo.get("span_substrings", [])
                ), (
                    f"{case['id']}: candidate evidence classification's supporting_observation_span_substrings "
                    f"{sub!r} does not match any expected_observations entry"
                )


def test_case_32_is_the_designated_negative_candidate_classification_case():
    """Milestone 2C anti-fabrication case: correct behavior is proposing
    NEITHER candidate_risk_signals nor candidate_evidence_classifications,
    mirroring case 14's 'empty is the correct primary result' pattern
    applied to the two Milestone 2C types specifically."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("32_"))
    assert case.get("expected_candidate_risk_signals") == []
    assert case.get("expected_candidate_evidence_classifications") == []


def test_all_3_mvp_mechanisms_are_covered_across_2c_cases():
    """PMO Option B decision (Milestone 2C MVP scope reduction): the
    benchmark must exercise exactly the three AI-automated MVP mechanisms
    (CR-01/02/08). CR-03 must NOT appear in any case's
    expected_candidate_risk_signals -- case 26 (formerly a positive CR-03
    test) was reframed to expect none, per test_case_26_is_mvp_boundary_
    case_not_positive_cr03 below."""
    data = _load()
    seen = {
        crs["mechanism"]
        for case in data["cases"]
        for crs in case.get("expected_candidate_risk_signals", [])
    }
    assert seen == VALID_MECHANISMS, f"expected all 3 MVP mechanisms covered, got {seen}"
    assert "CR-03" not in seen


def test_case_26_is_mvp_boundary_case_not_positive_cr03():
    """PMO Option B decision: case 26 was reframed from a positive CR-03
    candidate-risk-signal test into an MVP-boundary test. It must still
    extract the underlying CommercialObservation, but must expect ZERO
    candidate risk signals (no CR-03, since it's now structurally
    unavailable; no force-fit onto CR-01/CR-02/CR-08 either)."""
    data = _load()
    case26 = next(c for c in data["cases"] if c["id"].startswith("26_"))
    assert case26.get("expected_candidate_risk_signals") == []
    types26 = {eo["type"] for eo in case26["expected_observations"]}
    assert "COMMERCIAL_OBSERVATION" in types26


def test_all_4_evidence_bases_are_covered_across_2c_cases():
    """Milestone 2C implementation constraint 2: the benchmark must
    exercise all four authorized proposed value-evidence bases."""
    data = _load()
    seen = {
        cec["proposed_basis"]
        for case in data["cases"]
        for cec in case.get("expected_candidate_evidence_classifications", [])
    }
    assert seen == VALID_EVIDENCE_BASES, f"expected all 4 MVP evidence bases covered, got {seen}"


def test_cases_24_and_27_include_a_distractor_observation():
    """Milestone 2C implementation constraint 2 (valid/invalid observation-
    reference behavior): cases 24 and 27 each include a second, unrelated
    observation type so the correct supporting_observation_ref target is
    a genuine test rather than the only option available."""
    data = _load()
    for prefix in ("24_", "27_"):
        case = next(c for c in data["cases"] if c["id"].startswith(prefix))
        types = {eo["type"] for eo in case["expected_observations"]}
        assert len(types) >= 2, f"{case['id']}: expected at least 2 distinct observation types (distractor present)"


def test_dimension_qualifier_span_substrings_resolve_and_are_well_formed():
    data = _load()
    for case in data["cases"]:
        for dq in case.get("expected_dimension_d2_qualifiers", []):
            assert dq["qualifier"] in VALID_D2_QUALIFIERS, (
                f"{case['id']}: expected_dimension_d2_qualifiers qualifier {dq['qualifier']!r} "
                f"is not one of the 5 D2 qualifiers"
            )
            source_text = _source_text_for(case, dq.get("source"))
            for sub in dq["supporting_observation_span_substrings"]:
                assert sub in source_text, (
                    f"{case['id']}: expected_dimension_d2_qualifiers "
                    f"supporting_observation_span_substrings {sub!r} not found in source_text"
                )
                assert any(
                    sub in s for eo in case.get("expected_observations", []) for s in eo.get("span_substrings", [])
                ), (
                    f"{case['id']}: expected_dimension_d2_qualifiers supporting_observation_"
                    f"span_substrings {sub!r} does not match any expected_observations entry"
                )
        for dq in case.get("expected_dimension_d6_qualifiers", []):
            assert dq["qualifier"] in VALID_D6_QUALIFIERS, (
                f"{case['id']}: expected_dimension_d6_qualifiers qualifier {dq['qualifier']!r} "
                f"is not one of the 4 D6 qualifiers"
            )
            source_text = _source_text_for(case, dq.get("source"))
            for sub in dq["supporting_observation_span_substrings"]:
                assert sub in source_text, (
                    f"{case['id']}: expected_dimension_d6_qualifiers "
                    f"supporting_observation_span_substrings {sub!r} not found in source_text"
                )
                assert any(
                    sub in s for eo in case.get("expected_observations", []) for s in eo.get("span_substrings", [])
                ), (
                    f"{case['id']}: expected_dimension_d6_qualifiers supporting_observation_"
                    f"span_substrings {sub!r} does not match any expected_observations entry"
                )


def test_new_4b_cases_34_through_42_exist():
    data = _load()
    ids = {c["id"] for c in data["cases"]}
    for i in range(34, 43):
        prefix = f"{i:02d}_"
        assert any(cid.startswith(prefix) for cid in ids), f"missing new Milestone 4B case {prefix}"


# Benchmark-normalization checkpoint (Case 35, D2 EXPLICIT-basis composition
# gate): AUTOMATION_RELIABLE_LOW_LOGIN_OK is deliberately EXCLUDED from this
# benchmark-coverage assertion. It remains a fully valid member of
# VALID_D2_QUALIFIERS (see the membership check in
# test_dimension_qualifier_span_substrings_resolve_and_are_well_formed) --
# nothing about its definition or its place in the governed D2 vocabulary
# changed. What changed is that no case in the 42-case benchmark can any
# longer legitimately exercise its COMPOSITION end-to-end: composing it
# requires all 3 required atomic predicates to be EXPLICIT and grounded in
# the SAME isolated observation, and the only natural-language source shape
# that motivated this qualifier (a reliability fact + a low-login fact)
# splits across two clauses/observations in every case that was ever
# authored for it (Case 35, and Case 07's long-standing precedent). Adding a
# new single-observation fixture purely to keep this assertion green would
# be reverse-engineering benchmark content to satisfy a test rather than to
# express a governed truth -- exactly what the normalization authorization
# said not to do. AUTOMATION_RELIABLE_LOW_LOGIN_OK's composition correctness
# is instead verified deterministically, independent of live model behavior,
# by tests/test_extraction_atomic_predicate_composition.py (see
# test_d2_all_explicit_set_still_composes_with_explicit_composed_basis and
# the sibling EXPLICIT-basis-gate tests in that file).
D2_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE = VALID_D2_QUALIFIERS - {
    "AUTOMATION_RELIABLE_LOW_LOGIN_OK",
}


def test_remaining_4_d2_qualifiers_are_covered_across_4b_cases():
    """Milestone 4B checkpoint (post Case-35 normalization): the benchmark
    must exercise the 4 D2 qualifiers whose composition can be demonstrated
    within the one-observation/EXPLICIT-basis architecture.
    AUTOMATION_RELIABLE_LOW_LOGIN_OK is intentionally excluded here -- see
    D2_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE above -- and is instead
    covered by the deterministic composer tests."""
    data = _load()
    seen = {
        dq["qualifier"]
        for case in data["cases"]
        for dq in case.get("expected_dimension_d2_qualifiers", [])
    }
    assert seen == D2_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE, (
        f"expected the 4 benchmark-coverable D2 qualifiers covered, got {seen}"
    )


def test_automation_reliable_low_login_ok_remains_valid_vocabulary_but_uncomposed_in_benchmark():
    """Milestone 4B checkpoint: AUTOMATION_RELIABLE_LOW_LOGIN_OK must still
    be a recognized, governed D2 qualifier (it can still legitimately
    compose in principle -- e.g. from a single observation that states
    reliability AND low activity AND the explanatory link all at once) even
    though no CURRENT benchmark case exercises its composition end-to-end
    after the Case 35 normalization."""
    assert "AUTOMATION_RELIABLE_LOW_LOGIN_OK" in VALID_D2_QUALIFIERS
    data = _load()
    seen = {
        dq["qualifier"]
        for case in data["cases"]
        for dq in case.get("expected_dimension_d2_qualifiers", [])
    }
    assert "AUTOMATION_RELIABLE_LOW_LOGIN_OK" not in seen, (
        "if a case now legitimately exercises this compound's composition, "
        "restore it to D2_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE and to "
        "the exhaustive coverage test rather than leaving this stale"
    )


# Benchmark-normalization checkpoint (Case 40, D6 EXPLICIT-basis
# composition gate extension): CHAMPION_LOST_NO_SUCCESSOR is deliberately
# EXCLUDED from this benchmark-coverage assertion, on the same terms as
# AUTOMATION_RELIABLE_LOW_LOGIN_OK was excluded from D2's coverage
# assertion after the Case 35 normalization (see
# D2_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE above). It remains a fully
# valid member of VALID_D6_QUALIFIERS -- nothing about its definition or
# its place in the governed D6 vocabulary changed. What changed is that no
# case in the 42-case benchmark can any longer legitimately exercise its
# COMPOSITION end-to-end: a live D6-focused gate probe
# (prompt_v4_4b_dimqual_v3_2_d6_explicit_gate_probe1) confirmed stage-1
# splits Case 40's Priya Nair departure and no-successor facts into two
# separate StakeholderObservations, so no single observation's predicate
# set is ever complete under the frozen one-observation composition rule.
# Its composition correctness is instead verified deterministically,
# independent of live model behavior, by tests/
# test_extraction_atomic_predicate_composition.py (see
# test_d6_complete_set_both_explicit_still_composes and the sibling D6
# EXPLICIT-basis-gate tests in that file).
D6_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE = VALID_D6_QUALIFIERS - {
    "CHAMPION_LOST_NO_SUCCESSOR",
}


def test_remaining_3_d6_qualifiers_are_covered_across_4b_cases():
    """Milestone 4B checkpoint (post Case-40 normalization): the benchmark
    must exercise the 3 D6 qualifiers whose composition can be
    demonstrated within the one-observation/EXPLICIT-basis architecture.
    CHAMPION_LOST_NO_SUCCESSOR is intentionally excluded here -- see
    D6_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE above -- and is instead
    covered by the deterministic composer tests."""
    data = _load()
    seen = {
        dq["qualifier"]
        for case in data["cases"]
        for dq in case.get("expected_dimension_d6_qualifiers", [])
    }
    assert seen == D6_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE, (
        f"expected the 3 benchmark-coverable D6 qualifiers covered, got {seen}"
    )


def test_champion_lost_no_successor_remains_valid_vocabulary_but_uncomposed_in_benchmark():
    """Milestone 4B checkpoint: CHAMPION_LOST_NO_SUCCESSOR must still be a
    recognized, governed D6 qualifier (it can still legitimately compose
    in principle -- e.g. from a single observation that states both the
    departure and the no-successor fact at once) even though no CURRENT
    benchmark case exercises its composition end-to-end after the Case 40
    normalization."""
    assert "CHAMPION_LOST_NO_SUCCESSOR" in VALID_D6_QUALIFIERS
    data = _load()
    seen = {
        dq["qualifier"]
        for case in data["cases"]
        for dq in case.get("expected_dimension_d6_qualifiers", [])
    }
    assert "CHAMPION_LOST_NO_SUCCESSOR" not in seen, (
        "if a case now legitimately exercises this compound's composition, "
        "restore it to D6_QUALIFIERS_EXPECTED_IN_BENCHMARK_COVERAGE and to "
        "the exhaustive coverage test rather than leaving this stale"
    )


def test_case_40_distinguishes_champion_lost_from_departure_unconfirmed():
    """Milestone 4B architecture negotiation flagged this pair as the
    trickiest to distinguish. UPDATED for the Case 40 normalization
    (D6 EXPLICIT-basis composition gate extension): CHAMPION_LOST_
    NO_SUCCESSOR no longer composes here (stage-1 splits Priya's departure
    and no-successor facts into two separate observations, and the frozen
    one-observation rule correctly blocks cross-observation composition).
    Case 40 must still propose CHAMPION_DEPARTURE_UNCONFIRMED for Tom
    Reyes -- his signal must never be collapsed into a confirmed-departure
    qualifier just because a departure is plausible -- and must still
    represent all 3 underlying atomic facts (Priya's departure, Priya's
    no-successor coverage, Tom's unconfirmed signal) as distinct
    observations."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("40_"))
    qualifiers = {dq["qualifier"] for dq in case["expected_dimension_d6_qualifiers"]}
    assert qualifiers == {"CHAMPION_DEPARTURE_UNCONFIRMED"}
    assert "CHAMPION_LOST_NO_SUCCESSOR" not in qualifiers
    assert len(case["expected_observations"]) == 3


def test_case_42_is_the_designated_cooccurrence_and_non_force_fit_case():
    """Milestone 4B: at least one case must exercise D2 and D6 qualifiers
    proposed from the SAME evidence batch (co-occurrence), and at least
    one case must include an observation that correctly receives NO
    qualifier proposal (non-force-fit)."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("42_"))
    assert len(case["expected_dimension_d2_qualifiers"]) >= 1
    assert len(case["expected_dimension_d6_qualifiers"]) >= 1
    # non-force-fit: more expected_observations than qualifier slots --
    # at least one observation must correctly receive no proposal.
    total_qualifier_slots = len(case["expected_dimension_d2_qualifiers"]) + len(case["expected_dimension_d6_qualifiers"])
    assert len(case["expected_observations"]) > total_qualifier_slots


def test_dimension_qualifier_prompt_version_is_v3_2():
    """Milestone 4B D2 atomic-predicate SECOND targeted live-probe
    calibration checkpoint: bumped from v3.1 to v3.2 -- a further
    CALIBRATION (not a family break), scoped to EXACTLY ONE predicate
    definition (RELIABLE_AUTOMATION_OPERATION), after the v3.1 rerun
    (prompt_v4_4b_dimqual_v3_1_atomic_probe1) was disposed PARTIAL PASS /
    OVERALL FAIL: Cases 07 and 23 fully passed the compound non-force-fit
    gate, but Case 35 Observation B still incorrectly emitted RELIABLE_
    AUTOMATION_OPERATION from "the integration handles the workflow
    without manual intervention" -- automation MODE language, not a
    reliability/track-record claim -- which completed that observation's
    3/3 predicate set and caused an incorrect composition. LOW_LOGIN_OR_
    MANUAL_ACTIVITY and LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION are frozen
    and unchanged in this checkpoint (per explicit instruction). See
    extraction/prompts.py's own v3.1 -> v3.2 history comment for the full
    rationale. Kept structurally distinct from PROMPT_VERSION (stage 1,
    unchanged at v4)."""
    assert DIMENSION_QUALIFIER_PROMPT_VERSION == "v3.2"


def test_isolated_d2_prompt_pins_v3_1_shared_vocabulary_calibration():
    """Milestone 4B D2 atomic-predicate targeted live-probe calibration
    checkpoint: pins the "shared vocabulary is not shared evidence"
    framing sentence and the LOW_LOGIN_OR_MANUAL_ACTIVITY / LOW_ACTIVITY_
    EXPLAINED_BY_AUTOMATION synthetic invalid examples added in v3.1 to
    close the gap the live probe exposed on cases 23/35 -- generic
    automation-existence/execution-mode language being accepted as
    sufficient for predicates it does not establish. Both predicates are
    FROZEN in the v3.2 checkpoint (RELIABLE_AUTOMATION_OPERATION is the
    only predicate that changed further -- see the v3.2-specific tests
    below), so their v3.1 examples must still be present unmodified.
    These examples are deliberately NOT the verbatim benchmark sentences
    from cases 07/23/35 (per explicit instruction not to copy benchmark
    text into the prompt)."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert "SHARED VOCABULARY IS NOT SHARED EVIDENCE" in prompt
    assert "the report generator produces the file without any manual steps" in prompt
    assert "the sync job runs automatically every night" in prompt
    # None of the synthetic examples may be the actual benchmark
    # sentences from cases 07/23/35 -- confirms "do not copy benchmark
    # text verbatim into the prompt" was honored.
    assert "runs the platform's nightly batch job every night without any manual intervention" not in prompt
    assert "The nightly data sync between the customer's ERP and our platform has run automatically without failure for the past six months" not in prompt


# ===========================================================================
# Milestone 4B D2 atomic-predicate SECOND targeted live-probe calibration
# checkpoint (Prompt v3.2): RELIABLE_AUTOMATION_OPERATION only. 5 required
# deterministic prompt/contract tests, one per authorized semantic category.
# Per this project's established discipline, these pin that the approved
# WORDING exists -- they do not and cannot prove live model behavior,
# which can only be validated by a live run.
# ===========================================================================
def test_v3_2_reliable_automation_rejects_existence_alone():
    """Category 1: automation existence alone must not satisfy
    RELIABLE_AUTOMATION_OPERATION."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert "automation existing or being in place" in prompt
    assert "the workflow is fully automated and requires no manual steps" in prompt


def test_v3_2_reliable_automation_rejects_no_manual_intervention_alone():
    """Category 2: absence of manual intervention alone must not satisfy
    RELIABLE_AUTOMATION_OPERATION."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert 'the absence of manual intervention ("without manual intervention," "without manual steps")' in prompt


def test_v3_2_reliable_automation_rejects_scheduled_cadence_alone():
    """Category 3: scheduled cadence alone (e.g. "runs every night") must
    not satisfy RELIABLE_AUTOMATION_OPERATION -- this is the specific
    residual gap Case 23 raised and Case 35 confirmed."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert "scheduled CADENCE alone" in prompt
    assert "cadence describes WHEN something runs, not whether it works" in prompt
    assert "the sync job runs every night" in prompt
    # Distinct from the pre-existing, unrelated LOW_ACTIVITY_EXPLAINED_BY_
    # AUTOMATION example "the sync job runs automatically every night" --
    # confirm this new cadence-only example is not accidentally the same
    # string (it omits "automatically").
    assert "the sync job runs every night\" -- that states cadence" in prompt


def test_v3_2_reliable_automation_accepts_repeated_successful_operation():
    """Category 4: explicit repeated successful operation over an
    observed period must satisfy RELIABLE_AUTOMATION_OPERATION."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert "repeated successful executions" in prompt
    assert "the sync has completed successfully every run for the past quarter" in prompt


def test_v3_2_reliable_automation_accepts_explicit_failure_free_period():
    """Category 5: an explicit failure-free period must satisfy
    RELIABLE_AUTOMATION_OPERATION."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert "an explicit absence of failures over a period or run history" in prompt
    assert "the integration has had zero failed runs in the last 90 days" in prompt


def test_isolated_d2_prompt_excludes_compound_qualifier_from_direct_vocabulary():
    """Milestone 4B v3 checkpoint: pins that AUTOMATION_RELIABLE_LOW_
    LOGIN_OK is no longer offered as a directly-proposable value in the
    isolated D2 system prompt's qualifier vocabulary list, and that the
    new atomic-predicate instruction section (all 3 required predicate
    IDs) is present instead -- a structural guard against the v3
    response-contract change being silently reverted or dropped in a
    future revision. Prompt WORDING QUALITY is still validated only by a
    live run (never by string-matching alone, per this project's
    established discipline) -- this test only pins that the approved
    text exists, not that it produces correct model behavior."""
    prompt = _ISOLATED_D2_SYSTEM_PROMPT
    assert "AUTOMATION_RELIABLE_LOW_LOGIN_OK is NOT a value you may propose here" in prompt
    # RELIABLE_AUTOMATION_OPERATION's opening clause was reworded in the
    # Prompt v3.2 calibration checkpoint ("explicitly establishes" ->
    # "explicitly and affirmatively establishes") -- see the v3.2-scoped
    # tests below for that predicate's own content pins.
    assert "RELIABLE_AUTOMATION_OPERATION: this observation explicitly and affirmatively establishes" in prompt
    assert "LOW_LOGIN_OR_MANUAL_ACTIVITY: this observation explicitly establishes" in prompt
    assert "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION: this observation explicitly establishes" in prompt
    assert "Never propose the same predicate_id more than once in this call." in prompt
    # Structural pin: the value must not appear as a proposable D2_QUALIFIER_SCHEMA enum member.
    from extraction.json_schemas import D2_QUALIFIER_SCHEMA
    assert "AUTOMATION_RELIABLE_LOW_LOGIN_OK" not in D2_QUALIFIER_SCHEMA["enum"]


def test_isolated_d6_prompt_excludes_compound_qualifier_from_direct_vocabulary():
    """Milestone 4B v3 checkpoint: pins that CHAMPION_LOST_NO_SUCCESSOR is
    no longer offered as a directly-proposable value in the isolated D6
    system prompt's qualifier vocabulary list, and that the new atomic-
    predicate instruction section (both required predicate IDs) is
    present instead -- mirrors the D2 pin test above exactly, scoped to
    D6's own 2 required atomic predicates."""
    prompt = _ISOLATED_D6_SYSTEM_PROMPT
    assert "CHAMPION_LOST_NO_SUCCESSOR is NOT a value you may propose here" in prompt
    assert "CONFIRMED_CHAMPION_DEPARTURE: this observation explicitly establishes" in prompt
    assert "NO_SUCCESSOR_OR_CONTINUING_COVERAGE: this observation explicitly states" in prompt
    assert "Never propose the same predicate_id more than once in this call." in prompt
    from extraction.json_schemas import D6_QUALIFIER_SCHEMA
    assert "CHAMPION_LOST_NO_SUCCESSOR" not in D6_QUALIFIER_SCHEMA["enum"]


def test_isolated_d6_prompt_pins_champion_departure_unconfirmed_clarification():
    """Milestone 4B Prompt v2.1 calibration checkpoint (approved
    addition, checked explicitly against cases 03/24/15 in the checkpoint
    disposition): CHAMPION_DEPARTURE_UNCONFIRMED requires the FACT of
    departure to be uncertain -- a confirmed departure with unaddressed
    successor status must abstain outright, never fall back to
    CHAMPION_DEPARTURE_UNCONFIRMED as a catch-all."""
    prompt = _ISOLATED_D6_SYSTEM_PROMPT
    assert "CHAMPION_DEPARTURE_UNCONFIRMED is valid ONLY when the FACT of departure itself is uncertain" in prompt
    assert "does NOT qualify for CHAMPION_DEPARTURE_UNCONFIRMED merely because successor status is unknown" in prompt
    assert "never fall back to CHAMPION_DEPARTURE_UNCONFIRMED as a catch-all" in prompt


def test_optional_valid_entries_exist_for_previously_under_labeled_cases():
    """Milestone 2B.1 spec §3/§5: real grounded facts observed in
    baseline_v1_fix1.json (e.g. case 02's stakeholders/adoption/service
    facts) must now be captured as optional-valid expectations rather
    than being invisible to scoring."""
    data = _load()
    case = next(c for c in data["cases"] if c["id"].startswith("02_"))
    optional_valid = [eo for eo in case["expected_observations"] if eo.get("role") == "optional-valid"]
    assert len(optional_valid) >= 4


TESTS = [
    test_labeled_set_has_42_cases,
    test_labeled_set_covers_all_42_required_categories,
    test_case_ids_are_unique,
    test_every_expected_span_substring_resolves_against_its_source_text,
    test_every_expected_observation_has_a_valid_role,
    test_grounded_but_irrelevant_permitted_spans_resolve_and_are_scoped_to_case_14,
    test_no_case_has_ontology_ambiguity_accommodation,
    test_case_12_has_no_alternate_types,
    test_case_20_side_a_is_strategic_not_objective,
    test_case_05_has_no_alternate_types,
    test_case_17_is_strategic_not_objective,
    test_every_case_has_a_description_and_source_text,
    test_case_14_is_the_designated_empty_extraction_case,
    test_case_12_is_the_designated_contradiction_case,
    test_case_20_is_the_second_contradiction_case,
    test_at_least_two_contradiction_cases_total,
    test_case_15_permits_only_inferred_not_explicit_derived_claim,
    test_case_16_is_the_inferred_objective_case,
    test_new_ambiguity_cases_16_through_20_exist_and_are_well_formed,
    test_new_oc01_validation_cases_21_through_23_exist_and_are_well_formed,
    test_prompt_version_is_v4,
    test_optional_valid_entries_exist_for_previously_under_labeled_cases,
    test_new_2c_cases_24_through_33_exist,
    test_candidate_risk_signal_span_substrings_resolve_and_are_well_formed,
    test_candidate_evidence_classification_span_substrings_resolve_and_are_well_formed,
    test_case_32_is_the_designated_negative_candidate_classification_case,
    test_all_3_mvp_mechanisms_are_covered_across_2c_cases,
    test_case_26_is_mvp_boundary_case_not_positive_cr03,
    test_all_4_evidence_bases_are_covered_across_2c_cases,
    test_cases_24_and_27_include_a_distractor_observation,
    test_dimension_qualifier_span_substrings_resolve_and_are_well_formed,
    test_new_4b_cases_34_through_42_exist,
    test_remaining_4_d2_qualifiers_are_covered_across_4b_cases,
    test_automation_reliable_low_login_ok_remains_valid_vocabulary_but_uncomposed_in_benchmark,
    test_remaining_3_d6_qualifiers_are_covered_across_4b_cases,
    test_champion_lost_no_successor_remains_valid_vocabulary_but_uncomposed_in_benchmark,
    test_case_40_distinguishes_champion_lost_from_departure_unconfirmed,
    test_case_42_is_the_designated_cooccurrence_and_non_force_fit_case,
    test_dimension_qualifier_prompt_version_is_v3_2,
    test_isolated_d2_prompt_excludes_compound_qualifier_from_direct_vocabulary,
    test_isolated_d2_prompt_pins_v3_1_shared_vocabulary_calibration,
    test_v3_2_reliable_automation_rejects_existence_alone,
    test_v3_2_reliable_automation_rejects_no_manual_intervention_alone,
    test_v3_2_reliable_automation_rejects_scheduled_cadence_alone,
    test_v3_2_reliable_automation_accepts_repeated_successful_operation,
    test_v3_2_reliable_automation_accepts_explicit_failure_free_period,
    test_isolated_d6_prompt_excludes_compound_qualifier_from_direct_vocabulary,
    test_isolated_d6_prompt_pins_champion_departure_unconfirmed_clarification,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
