"""
Milestone 4C — D2/D6 Human Confirmation Integration tests.

Closes M3-OD-01 for D2/D6: a Milestone 4B CandidateDimensionQualifier (simple
or deterministically composed compound) is now reviewable through the SAME
confirmation architecture as every other candidate type (HumanConfirmationRecord
/ state_machine / active_evidence / recompute / consequentiality) -- no new
confirmation subsystem. Mirrors the hand-authored, pipeline-finalized fixture
style already used by tests/test_confirmation_active_evidence.py and
tests/test_confirmation_recompute_end_to_end.py, rather than running the full
AI extraction pipeline (that is exercised elsewhere, in tests/test_extraction_*
and tests/test_extraction_dimension_qualifier_pipeline.py).

Covers, in order: CONFIRM (D2, D6, simple and compound), REJECT, CANNOT_CONFIRM
(including the required CANNOT_CONFIRM -> no DMEG -> no ER1 regression),
CORRECT's three governing constraints (dimension immutable, basis immutable,
compound-qualifier manufacture prohibited) plus a normal in-vocabulary
correction, atomic-predicate provenance preservation, "no composer
re-execution" during confirmation, D2/D6 deterministic recomputation and
reproducibility, multiple-qualifier precedence (zero engine change), separate-
decision independence from the supporting stage-1 observation, the kept
dimension_qualifier_overrides path concatenating safely, full audit-chain
reconstruction, Confirm-vs-never-reviewed consequentiality and its
non-mutation, and the three UI-layer pure-function additions (ui/labels.py's
qualifier dropdown, ui/item_card.py's per-type non-editable-fields function,
ui/review_queue.py's atomic_predicate_evidence_for lookup).
"""

import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import ConfirmationAction, DimensionCode, EvidenceReviewStatus, EvidenceState, Lifecycle
from domain.objective import Objective
from engine.registry_loader import load_and_validate
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    AdoptionObservation, AtomicPredicateEvidence, CandidateDimensionQualifier,
    ExtractionSystemFields, ObservationRef, SourceSpan, StakeholderObservation,
)

from confirmation.active_evidence import reconstruct_active_evidence
from confirmation.consequentiality import compute_consequentiality
from confirmation.enums import ConfirmationTargetKind
from confirmation.recompute import recompute
from confirmation.schemas import create_confirmation_record

from ui.item_card import _non_editable_fields_for
from ui.labels import correct_field_options
from ui.review_queue import atomic_predicate_evidence_for

REGISTRY = load_and_validate()
OBJECTIVE = Objective("OBJ-1", "Reduce monthly reconciliation effort by 50%", is_known=True)


def make_account():
    return AccountAssessment(
        assessment_id="ASSESS-M4C-1",
        scope=Scope("SCOPE-M4C-1", "Fictional Northwind Robotics", "Reconciliation Suite"),
        lifecycle=Lifecycle.L3, objective=OBJECTIVE,
    )


def _system(obs_id: str, evidence_state=EvidenceState.CURRENT_UNVERIFIED) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="test-provider", model_version="test-v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
        evidence_state=evidence_state,
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def build_extraction_result():
    """One stage-1 stakeholder fact (D6-eligible, backing a compound
    candidate), one stage-1 adoption fact (D2-eligible, backing a simple
    candidate), a simple D6 candidate on a SECOND stakeholder fact, and the
    2 grounded atomic predicates that (in a real run) composed the D6
    compound candidate -- exactly the shape extraction/pipeline.py's stage 2
    produces, hand-authored here per this test file's established style."""
    stake_compound = StakeholderObservation(
        source_evidence_id="E-CHAMPION-1", source_span=_span("Priya, our champion, has left the company"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Priya", system=_system("OBS-STAKE-COMPOUND"),
    )
    stake_simple = StakeholderObservation(
        source_evidence_id="E-SUCC-1", source_span=_span("relationship continuity is concentrated in one person"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Jordan", system=_system("OBS-STAKE-SIMPLE"),
    )
    adoption = AdoptionObservation(
        source_evidence_id="E-ADOPT-1", source_span=_span("usage is real but confined to the invoicing module"),
        basis=InferenceBasis.EXPLICIT, workflow_or_use_case="Invoicing", observed_behavior="Confined usage",
        system=_system("OBS-ADOPT-1"),
    )

    d6_compound = CandidateDimensionQualifier(
        dimension=DimensionCode.D6, qualifier="CHAMPION_LOST_NO_SUCCESSOR", basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=0),
        source_evidence_id="E-CHAMPION-1", source_span=stake_compound.source_span,
        resolved_observation_id="OBS-STAKE-COMPOUND", system=_system("OBS-D6-COMPOUND"),
    )
    d6_simple = CandidateDimensionQualifier(
        dimension=DimensionCode.D6, qualifier="SUCCESSION_UNCLEAR_OR_CONCENTRATED", basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=1),
        source_evidence_id="E-SUCC-1", source_span=stake_simple.source_span,
        resolved_observation_id="OBS-STAKE-SIMPLE", system=_system("OBS-D6-SIMPLE"),
    )
    d2_simple = CandidateDimensionQualifier(
        dimension=DimensionCode.D2, qualifier="NARROW_BREADTH_OR_CONCENTRATION", basis=InferenceBasis.EXPLICIT,
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.ADOPTION_OBSERVATION, index=0),
        source_evidence_id="E-ADOPT-1", source_span=adoption.source_span,
        resolved_observation_id="OBS-ADOPT-1", system=_system("OBS-D2-SIMPLE"),
    )

    predicate_evidence = (
        AtomicPredicateEvidence(
            predicate_id="CONFIRMED_CHAMPION_DEPARTURE", dimension=DimensionCode.D6,
            resolved_observation_id="OBS-STAKE-COMPOUND", evidence_text="Priya, our champion, has left the company",
            basis=InferenceBasis.EXPLICIT,
        ),
        AtomicPredicateEvidence(
            predicate_id="NO_SUCCESSOR_OR_CONTINUING_COVERAGE", dimension=DimensionCode.D6,
            resolved_observation_id="OBS-STAKE-COMPOUND", evidence_text="has left the company",
            basis=InferenceBasis.EXPLICIT,
        ),
    )

    return ExtractionResult(
        accepted=(stake_compound, stake_simple, adoption), candidate_contradictions=(),
        candidate_risk_signals=(), candidate_evidence_classifications=(),
        rejected=(), dedup_audit=(), traces=(),
        candidate_d2_qualifiers=(d2_simple,), candidate_d6_qualifiers=(d6_simple, d6_compound),
        dimension_qualifier_predicate_evidence=predicate_evidence,
    )


def _confirm(target_id):
    return create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id=target_id,
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )


def _run(records, dimensions_to_evaluate, dimension_qualifier_overrides=None):
    extraction_result = build_extraction_result()
    active = reconstruct_active_evidence(extraction_result, records)
    return recompute(
        make_account(), REGISTRY, active,
        dimensions_to_evaluate=dimensions_to_evaluate,
        dimension_qualifier_overrides=dimension_qualifier_overrides,
    )


# ---- 1. D2 CONFIRM: simple candidate activates the expected D2 state ----

def test_confirm_d2_simple_qualifier_produces_signal_and_activates_mixed():
    diag = _run((_confirm("OBS-D2-SIMPLE"),), (DimensionCode.D2,))
    # The builder emits one signal per active qualifier candidate REGARDLESS
    # of confirmation state (all 3 fixture candidates are active here); only
    # the D2 one is confirmed, and only D2 is evaluated.
    assert len(diag.dimension_signals) == 3
    sig = next(s for s in diag.dimension_signals if s.signal_id == "OBS-D2-SIMPLE")
    assert sig.dimension == DimensionCode.D2
    assert sig.qualifier == "NARROW_BREADTH_OR_CONCENTRATION"
    assert sig.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert sig.evidence_refs == ("OBS-ADOPT-1",)  # PMO decision: supporting stage-1 id, never a synthetic id
    assert diag.result.dimension_states[DimensionCode.D2].state.value == "MIXED"


# ---- 2. D6 CONFIRM: compound candidate activates the expected D6 state ----

def test_confirm_d6_compound_qualifier_produces_signal_and_activates_concerning():
    diag = _run((_confirm("OBS-D6-COMPOUND"),), (DimensionCode.D6,))
    assert len(diag.dimension_signals) == 3  # all 3 active fixture candidates, evidence_state-agnostic builder
    sig = next(s for s in diag.dimension_signals if s.signal_id == "OBS-D6-COMPOUND")
    assert sig.dimension == DimensionCode.D6
    assert sig.qualifier == "CHAMPION_LOST_NO_SUCCESSOR"
    assert sig.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert sig.evidence_refs == ("OBS-STAKE-COMPOUND",)
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "CONCERNING"


# ---- 3. REJECT excludes the candidate; no signal, no side effects ----

def test_reject_d6_qualifier_excludes_and_produces_no_signal():
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-COMPOUND",
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
        reason="Evidence does not actually establish no successor exists.",
    )
    diag = _run((record,), (DimensionCode.D6,))
    # the rejected candidate produces no signal at all; the 2 remaining
    # active candidates (D2 simple, D6 simple) still do.
    assert "OBS-D6-COMPOUND" not in {s.signal_id for s in diag.dimension_signals}
    assert len(diag.dimension_signals) == 2
    assert diag.active_evidence.is_excluded("OBS-D6-COMPOUND")
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "INSUFFICIENT_EVIDENCE"
    # sibling candidate and the supporting stage-1 observation are untouched
    assert diag.active_evidence.by_observation_id("OBS-D6-SIMPLE") is not None
    assert diag.active_evidence.by_observation_id("OBS-STAKE-COMPOUND").evidence_state == EvidenceState.CURRENT_UNVERIFIED


# ---- 4. CANNOT_CONFIRM preserves uncertainty; no activation ----

def test_cannot_confirm_d6_qualifier_preserves_uncertainty_no_activation():
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-COMPOUND",
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
        reason="Need to verify with the account team before treating this as final.",
    )
    diag = _run((record,), (DimensionCode.D6,))
    item = diag.active_evidence.by_observation_id("OBS-D6-COMPOUND")
    assert item.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    # the builder still builds a signal (evidence_state-agnostic, per its own
    # docstring) -- confirmation-gating happens downstream in
    # engine/dimension_engine.py's unmodified is_current_confirmed() check.
    assert len(diag.dimension_signals) == 3
    sig = next(s for s in diag.dimension_signals if s.signal_id == "OBS-D6-COMPOUND")
    assert sig.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "INSUFFICIENT_EVIDENCE"


# ---- 5. Required regression: CANNOT_CONFIRM -> no DMEG -> no hardcoded ER1 ----

def test_cannot_confirm_d6_qualifier_triggers_no_dmeg_and_no_er1():
    """The discovered, favorable fact from the pre-implementation
    verification checkpoint: engine/evaluate.py's DMEG differential-
    generation section (§4, paths 4a/4b/4c) never iterates dimension_signals
    at all -- it only ever looks at risk_claims (4a) and D1 (4b/4c). A
    pending/CANNOT_CONFIRM'd D2/D6 qualifier candidate therefore has ZERO
    DMEG/ER1 pathway today, direct or indirect. This extraction result has
    no risk signals and no evidence classifications at all, isolating the
    proof to exactly this candidate."""
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-COMPOUND",
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
        reason="Need to verify with the account team before treating this as final.",
    )
    diag = _run((record,), (DimensionCode.D6,))
    assert diag.result.dmegs == ()
    assert diag.result.evidence_review.value == EvidenceReviewStatus.ER0


# ---- 6. CORRECT: `dimension` is immutable ----

def test_correct_may_not_change_dimension():
    extraction_result = build_extraction_result()
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D2-SIMPLE",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"dimension": DimensionCode.D6},
    )
    with pytest.raises(ValueError, match="may not change `dimension`"):
        reconstruct_active_evidence(extraction_result, (record,))


# ---- 7. CORRECT: `basis` is immutable ----

def test_correct_may_not_change_basis():
    extraction_result = build_extraction_result()
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D2-SIMPLE",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"basis": InferenceBasis.INFERRED_CANDIDATE},
    )
    with pytest.raises(ValueError, match="may not change `basis`"):
        reconstruct_active_evidence(extraction_result, (record,))


# ---- 8. CORRECT: a simple candidate may never be corrected into a compound qualifier ----

def test_correct_may_not_manufacture_compound_qualifier_on_a_simple_candidate():
    extraction_result = build_extraction_result()
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-SIMPLE",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"qualifier": "CHAMPION_LOST_NO_SUCCESSOR"},
    )
    with pytest.raises(ValueError, match="Milestone 4B's 2 compound"):
        reconstruct_active_evidence(extraction_result, (record,))


def test_correct_may_not_propose_qualifier_outside_governed_vocabulary():
    extraction_result = build_extraction_result()
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-SIMPLE",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"qualifier": "NARROW_BREADTH_OR_CONCENTRATION"},  # a D2 value, not D6
    )
    with pytest.raises(ValueError, match="not in the governed vocabulary"):
        reconstruct_active_evidence(extraction_result, (record,))


# ---- 9. CORRECT: a normal in-vocabulary qualifier correction still works ----

def test_correct_normal_in_vocabulary_qualifier_change_succeeds_and_flows_to_recompute():
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-SIMPLE",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"qualifier": "APPROPRIATE_SPONSOR_COVERAGE"},
    )
    diag = _run((record,), (DimensionCode.D6,))
    item = diag.active_evidence.by_observation_id("OBS-D6-SIMPLE")
    assert item.is_correction is True
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.representation["qualifier"] == "APPROPRIATE_SPONSOR_COVERAGE"
    sig = next(s for s in diag.dimension_signals if s.signal_id == "OBS-D6-SIMPLE")
    assert sig.qualifier == "APPROPRIATE_SPONSOR_COVERAGE"
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "SUPPORTED"


# ---- 10. CORRECT: a compound candidate corrected to its OWN compound value is allowed ----

def test_correct_compound_candidate_to_its_own_existing_compound_value_is_allowed():
    """The guard only blocks MANUFACTURING a compound value that wasn't
    already there -- it does not block every CORRECT on an already-compound
    candidate (e.g. correcting some other field while the qualifier itself
    is re-affirmed unchanged)."""
    extraction_result = build_extraction_result()
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-COMPOUND",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"qualifier": "CHAMPION_LOST_NO_SUCCESSOR"},
    )
    active = reconstruct_active_evidence(extraction_result, (record,))
    item = active.by_observation_id("OBS-D6-COMPOUND")
    assert item.representation["qualifier"] == "CHAMPION_LOST_NO_SUCCESSOR"


# ---- 11. Atomic-predicate provenance is preserved, never copied onto the signal ----

def test_atomic_predicate_provenance_preserved_and_never_copied_onto_signal():
    extraction_result = build_extraction_result()
    obs = next(o for o in extraction_result.candidate_d6_qualifiers if o.system.observation_id == "OBS-D6-COMPOUND")
    predicates = atomic_predicate_evidence_for(extraction_result, obs)
    assert len(predicates) == 2
    assert {p.predicate_id for p in predicates} == {
        "CONFIRMED_CHAMPION_DEPARTURE", "NO_SUCCESSOR_OR_CONTINUING_COVERAGE",
    }
    assert predicates == extraction_result.dimension_qualifier_predicate_evidence  # identical objects, never copied

    diag = _run((_confirm("OBS-D6-COMPOUND"),), (DimensionCode.D6,))
    sig = next(s for s in diag.dimension_signals if s.signal_id == "OBS-D6-COMPOUND")
    # the signal's evidence_refs is exactly the ONE supporting stage-1 id --
    # atomic predicate_ids are never flattened onto it.
    assert sig.evidence_refs == ("OBS-STAKE-COMPOUND",)
    assert "CONFIRMED_CHAMPION_DEPARTURE" not in sig.evidence_refs

    # atomic provenance is still independently reachable after confirmation,
    # by following evidence_refs[0] back into the SAME extraction_result key.
    still_reachable = tuple(
        p for p in extraction_result.dimension_qualifier_predicate_evidence
        if p.resolved_observation_id == sig.evidence_refs[0] and p.dimension == sig.dimension
    )
    assert still_reachable == predicates


def test_atomic_predicate_evidence_for_is_empty_for_a_simple_candidate():
    extraction_result = build_extraction_result()
    obs = next(o for o in extraction_result.candidate_d6_qualifiers if o.system.observation_id == "OBS-D6-SIMPLE")
    assert atomic_predicate_evidence_for(extraction_result, obs) == ()


# ---- 12. No composer re-execution during confirmation ----

def test_confirmation_never_re_invokes_stage2_composition_logic():
    """Structural proof: confirmation/recompute.py imports nothing from
    extraction.pipeline (the module the atomic-predicate composer lives in)
    -- it cannot re-run composition even in principle. Combined with a
    behavioral proof: a compound candidate with an EMPTY predicate-evidence
    set (as if the composer's completeness check had never run for it --
    never possible from a real pipeline run, but exactly what would be true
    if recompute() tried to re-derive completeness itself) still confirms
    and produces a signal with its own qualifier value untouched, because
    recompute() only ever reads the CONFIRMED representation's own
    `qualifier` field -- it never re-checks predicate-evidence completeness,
    which is a Milestone 4B extraction-time concern, frozen for 4C."""
    import confirmation.recompute as recompute_module
    assert "extraction.pipeline" not in {
        getattr(v, "__module__", None) for v in vars(recompute_module).values()
    }
    assert not hasattr(recompute_module, "run_dimension_qualifier_classification")

    bare_result = ExtractionResult(
        accepted=(), candidate_contradictions=(), candidate_risk_signals=(),
        candidate_evidence_classifications=(), rejected=(), dedup_audit=(), traces=(),
        candidate_d6_qualifiers=(
            CandidateDimensionQualifier(
                dimension=DimensionCode.D6, qualifier="CHAMPION_LOST_NO_SUCCESSOR", basis=InferenceBasis.EXPLICIT,
                supporting_observation_ref=ObservationRef(observation_type=ObservationType.STAKEHOLDER_OBSERVATION, index=0),
                source_evidence_id="E-1", source_span=_span("no successor identified anywhere"),
                resolved_observation_id="OBS-STAKE-BARE", system=_system("OBS-D6-BARE"),
            ),
        ),
        dimension_qualifier_predicate_evidence=(),  # deliberately empty -- see docstring above
    )
    active = reconstruct_active_evidence(bare_result, (_confirm("OBS-D6-BARE"),))
    diag = recompute(
        make_account(), REGISTRY, active,
        dimensions_to_evaluate=(DimensionCode.D6,), dimension_qualifier_overrides=None,
    )
    assert diag.dimension_signals[0].qualifier == "CHAMPION_LOST_NO_SUCCESSOR"
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "CONCERNING"


# ---- 13. Deterministic recomputation / reproducibility ----

def test_d6_recompute_is_reproducible():
    records = (_confirm("OBS-D6-COMPOUND"),)
    r1 = _run(records, (DimensionCode.D6,)).result
    r2 = _run(records, (DimensionCode.D6,)).result
    assert r1.dimension_states[DimensionCode.D6].state == r2.dimension_states[DimensionCode.D6].state
    assert r1.evidence_review.value == r2.evidence_review.value


# ---- 14. Multiple confirmed qualifiers for the same dimension: precedence, zero engine change ----

def test_multiple_confirmed_d6_qualifiers_follow_existing_concerning_over_mixed_precedence():
    records = (_confirm("OBS-D6-COMPOUND"), _confirm("OBS-D6-SIMPLE"))  # CONCERNING + MIXED confirmed together
    diag = _run(records, (DimensionCode.D6,))
    d6_sigs = [s for s in diag.dimension_signals if s.dimension == DimensionCode.D6]
    assert len(d6_sigs) == 2
    assert all(s.evidence_state == EvidenceState.CURRENT_CONFIRMED for s in d6_sigs)
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "CONCERNING"  # CONCERNING wins, unmodified rule


# ---- 15. Confirming a qualifier candidate never implicitly confirms its supporting observation ----

def test_confirming_qualifier_does_not_confirm_supporting_stakeholder_observation():
    diag = _run((_confirm("OBS-D6-COMPOUND"),), (DimensionCode.D6,))
    supporting = diag.active_evidence.by_observation_id("OBS-STAKE-COMPOUND")
    assert supporting is not None
    assert supporting.evidence_state == EvidenceState.CURRENT_UNVERIFIED  # never promoted
    assert supporting.confirmation_id is None  # never reviewed at all


# ---- 16. dimension_qualifier_overrides (kept, PMO decision) concatenates safely ----

def test_overrides_path_concatenates_with_confirmed_candidate_signals():
    overrides = {"OBS-STAKE-SIMPLE": (DimensionCode.D6, "APPROPRIATE_SPONSOR_COVERAGE")}
    diag = _run((_confirm("OBS-D6-COMPOUND"),), (DimensionCode.D6,), dimension_qualifier_overrides=overrides)
    assert len(diag.dimension_signals) == 4  # 3 from the confirmed-candidate builder + 1 from the override path
    sources = {s.signal_id for s in diag.dimension_signals}
    assert "OBS-D6-COMPOUND" in sources
    assert "OBS-STAKE-SIMPLE-D6" in sources
    # CONCERNING (confirmed candidate) beats SUPPORTED (override) -- same
    # unmodified precedence rule as test 14 above.
    assert diag.result.dimension_states[DimensionCode.D6].state.value == "CONCERNING"


# ---- 17. Full audit-chain reconstruction across re-review ----

def test_full_audit_chain_reconstruction_across_re_review():
    first = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, target_observation_id="OBS-D6-COMPOUND",
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
        reason="Awaiting confirmation from the account team.",
    )
    second = _confirm("OBS-D6-COMPOUND")
    active = reconstruct_active_evidence(build_extraction_result(), (first, second))
    item = active.by_observation_id("OBS-D6-COMPOUND")
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.confirmation_id == second.confirmation_id  # latest terminal record governs
    # nothing is deleted -- both records remain independently inspectable,
    # exactly what ui/audit_view.py's history view reads.
    all_records = (first, second)
    assert all(r.target_observation_id == "OBS-D6-COMPOUND" for r in all_records)
    assert first.action == ConfirmationAction.CANNOT_CONFIRM
    assert second.action == ConfirmationAction.CONFIRM


# ---- 18. Consequentiality now works for a D2/D6 qualifier candidate ----

def test_confirming_d6_compound_is_consequential():
    extraction_result = build_extraction_result()
    records = (_confirm("OBS-D6-COMPOUND"),)
    report = compute_consequentiality(
        make_account(), REGISTRY, extraction_result, records, "OBS-D6-COMPOUND",
        dimensions_to_evaluate=(DimensionCode.D6,),
    )
    assert report.is_consequential is True  # CONCERNING (confirmed) vs INSUFFICIENT_EVIDENCE (never reviewed)


def test_compute_consequentiality_does_not_mutate_inputs():
    extraction_result = build_extraction_result()
    records = (_confirm("OBS-D6-COMPOUND"),)
    before_d6 = extraction_result.candidate_d6_qualifiers
    r1 = compute_consequentiality(
        make_account(), REGISTRY, extraction_result, records, "OBS-D6-COMPOUND",
        dimensions_to_evaluate=(DimensionCode.D6,),
    )
    r2 = compute_consequentiality(
        make_account(), REGISTRY, extraction_result, records, "OBS-D6-COMPOUND",
        dimensions_to_evaluate=(DimensionCode.D6,),
    )
    assert extraction_result.candidate_d6_qualifiers is before_d6  # never replaced/mutated
    assert len(records) == 1  # never appended to
    assert r1.is_consequential == r2.is_consequential
    assert r1.outcome_with_review == r2.outcome_with_review
    assert r1.outcome_without_review == r2.outcome_without_review


# ---- 19. UI pure-function additions ----

def test_labels_correct_field_options_excludes_compound_values_for_a_simple_candidate():
    extraction_result = build_extraction_result()
    simple_obs = next(o for o in extraction_result.candidate_d6_qualifiers if o.system.observation_id == "OBS-D6-SIMPLE")
    options = correct_field_options(simple_obs, "qualifier")
    assert "CHAMPION_LOST_NO_SUCCESSOR" not in options
    assert "SUCCESSION_UNCLEAR_OR_CONCENTRATED" in options  # its own current value stays offered
    assert "APPROPRIATE_SPONSOR_COVERAGE" in options


def test_labels_correct_field_options_keeps_compound_value_for_a_compound_candidate():
    extraction_result = build_extraction_result()
    compound_obs = next(o for o in extraction_result.candidate_d6_qualifiers if o.system.observation_id == "OBS-D6-COMPOUND")
    options = correct_field_options(compound_obs, "qualifier")
    assert "CHAMPION_LOST_NO_SUCCESSOR" in options


def test_item_card_non_editable_fields_excludes_basis_and_dimension_only_for_dimension_qualifier():
    extraction_result = build_extraction_result()
    qualifier_obs = extraction_result.candidate_d6_qualifiers[0]
    fields = _non_editable_fields_for(qualifier_obs)
    assert "basis" in fields
    assert "dimension" in fields

    other_obs = extraction_result.accepted[0]  # a StakeholderObservation
    other_fields = _non_editable_fields_for(other_obs)
    assert "basis" not in other_fields  # basis stays correctable for every other type
    assert "dimension" in other_fields  # globally safe -- no other type has this field name anyway


TESTS = [
    test_confirm_d2_simple_qualifier_produces_signal_and_activates_mixed,
    test_confirm_d6_compound_qualifier_produces_signal_and_activates_concerning,
    test_reject_d6_qualifier_excludes_and_produces_no_signal,
    test_cannot_confirm_d6_qualifier_preserves_uncertainty_no_activation,
    test_cannot_confirm_d6_qualifier_triggers_no_dmeg_and_no_er1,
    test_correct_may_not_change_dimension,
    test_correct_may_not_change_basis,
    test_correct_may_not_manufacture_compound_qualifier_on_a_simple_candidate,
    test_correct_may_not_propose_qualifier_outside_governed_vocabulary,
    test_correct_normal_in_vocabulary_qualifier_change_succeeds_and_flows_to_recompute,
    test_correct_compound_candidate_to_its_own_existing_compound_value_is_allowed,
    test_atomic_predicate_provenance_preserved_and_never_copied_onto_signal,
    test_atomic_predicate_evidence_for_is_empty_for_a_simple_candidate,
    test_confirmation_never_re_invokes_stage2_composition_logic,
    test_d6_recompute_is_reproducible,
    test_multiple_confirmed_d6_qualifiers_follow_existing_concerning_over_mixed_precedence,
    test_confirming_qualifier_does_not_confirm_supporting_stakeholder_observation,
    test_overrides_path_concatenates_with_confirmed_candidate_signals,
    test_full_audit_chain_reconstruction_across_re_review,
    test_confirming_d6_compound_is_consequential,
    test_compute_consequentiality_does_not_mutate_inputs,
    test_labels_correct_field_options_excludes_compound_values_for_a_simple_candidate,
    test_labels_correct_field_options_keeps_compound_value_for_a_compound_candidate,
    test_item_card_non_editable_fields_excludes_basis_and_dimension_only_for_dimension_qualifier,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
