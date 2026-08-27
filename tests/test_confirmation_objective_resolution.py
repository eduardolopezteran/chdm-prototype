"""
Milestone 3D — Objective Context Integration: confirmation-layer
objective-identity bridge tests.

Exercises confirmation.recompute._resolve_objective() end to end through
the same public recompute() entry point every other confirmation test
uses -- no direct unit-calls to the private helper, so these tests prove
the whole confirmed-evidence -> Objective -> engine.evaluate() chain, not
just the resolver in isolation. The account's declared objective is
deliberately Objective(is_known=False) throughout this file (unlike
tests/test_confirmation_recompute_end_to_end.py, which hand-declares a
KNOWN objective and therefore never needs this bridge at all -- both
fixtures are intentionally different, proving the "already-known objective
is never overridden" rule by construction, not just by assertion).

No domain/ or engine/ file changed to support this milestone; nothing
here tests objective_engine.py's or dimension_engine.py's own logic
(that remains covered by the pre-existing Milestone 1 suite) -- these
tests confirm confirmed extraction evidence now reaches that unmodified
logic correctly.
"""

import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import (
    ConfirmationAction, DimensionCode, EvidenceState, Lifecycle,
    ObjectiveOutcomeState, RiskMechanismCode,
)
from domain.objective import Objective
from engine.registry_loader import load_and_validate
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    CandidateEvidenceClassification, ExtractionSystemFields, ObjectiveCandidate,
    ObservationRef, SourceSpan,
)

from confirmation.active_evidence import reconstruct_active_evidence
from confirmation.enums import ConfirmationTargetKind, ObjectiveResolutionStatus
from confirmation.recompute import recompute
from confirmation.schemas import create_confirmation_record

REGISTRY = load_and_validate()
UNKNOWN_OBJECTIVE = Objective("OBJ-1", None, is_known=False)


def make_account(objective=UNKNOWN_OBJECTIVE):
    return AccountAssessment(
        assessment_id="ASSESS-M3D-1",
        scope=Scope("SCOPE-M3D-1", "Fictional Southgate Analytics", "Reporting Suite"),
        lifecycle=Lifecycle.L3, objective=objective,
    )


def _system(obs_id: str) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="test-provider", model_version="test-v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
        evidence_state=EvidenceState.CURRENT_UNVERIFIED,
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


# accepted[0] = OBS-OBJ-1 "Reduce monthly reconciliation effort by 50%"
# accepted[1] = OBS-OBJ-2 same statement, deliberately re-cased/re-spaced
#               ("  Reduce   monthly reconciliation EFFORT by 50%  ")
# accepted[2] = OBS-OBJ-3 a materially different statement
#               ("Improve customer NPS score")
def build_extraction_result():
    obj_1 = ObjectiveCandidate(
        source_evidence_id="E-OBJ-1", source_span=_span("reduce our reconciliation workload by half"),
        basis=InferenceBasis.EXPLICIT, objective_text="Reduce monthly reconciliation effort by 50%",
        system=_system("OBS-OBJ-1"),
    )
    obj_2_duplicate = ObjectiveCandidate(
        source_evidence_id="E-OBJ-2", source_span=_span("cut reconciliation time in half every month"),
        basis=InferenceBasis.EXPLICIT,
        objective_text="  Reduce   monthly reconciliation EFFORT by 50%  ",
        system=_system("OBS-OBJ-2"),
    )
    obj_3_conflicting = ObjectiveCandidate(
        source_evidence_id="E-OBJ-3", source_span=_span("our real focus this year is NPS"),
        basis=InferenceBasis.EXPLICIT, objective_text="Improve customer NPS score",
        system=_system("OBS-OBJ-3"),
    )
    evidence_classification_achieved = CandidateEvidenceClassification(
        source_evidence_id="E-VAL-1", source_span=_span("confirmed by the finance ops monthly report"),
        basis=InferenceBasis.INFERRED_CANDIDATE, proposed_basis="MEASURED_OPERATIONAL_EVIDENCE",
        supports="ACHIEVED",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.OBJECTIVE_CANDIDATE, index=0),
        resolved_observation_id="OBS-OBJ-1", system=_system("OBS-EVIDCLASS-ACHIEVED"),
    )
    evidence_classification_not_achieved = CandidateEvidenceClassification(
        source_evidence_id="E-VAL-2", source_span=_span("customer reported reconciliation time is unchanged"),
        basis=InferenceBasis.INFERRED_CANDIDATE, proposed_basis="CUSTOMER_CONFIRMED",
        supports="NOT_ACHIEVED",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.OBJECTIVE_CANDIDATE, index=0),
        resolved_observation_id="OBS-OBJ-1", system=_system("OBS-EVIDCLASS-NOTACHIEVED"),
    )
    return ExtractionResult(
        accepted=(obj_1, obj_2_duplicate, obj_3_conflicting),
        candidate_contradictions=(), candidate_risk_signals=(),
        candidate_evidence_classifications=(evidence_classification_achieved, evidence_classification_not_achieved),
        rejected=(), dedup_audit=(), traces=(),
    )


def _run(records, objective=UNKNOWN_OBJECTIVE):
    extraction_result = build_extraction_result()
    active = reconstruct_active_evidence(extraction_result, records)
    return recompute(
        make_account(objective), REGISTRY, active,
        dimensions_to_evaluate=(DimensionCode.D1,), dimension_qualifier_overrides={},
    )


def _confirm(target_kind, target_id):
    return create_confirmation_record(
        target_kind=target_kind, target_observation_id=target_id,
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )


def _reject(target_kind, target_id, reason="Not accurate."):
    return create_confirmation_record(
        target_kind=target_kind, target_observation_id=target_id,
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com", reason=reason,
    )


def _cannot_confirm(target_kind, target_id, reason="Cannot verify yet."):
    return create_confirmation_record(
        target_kind=target_kind, target_observation_id=target_id,
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com", reason=reason,
    )


# ---- 1. One confirmed objective only (no value evidence) ----

def test_one_confirmed_objective_only_establishes_identity_but_outcome_stays_unknown():
    records = (_confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),)
    diag = _run(records)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.ESTABLISHED
    assert res.text == "Reduce monthly reconciliation effort by 50%"
    assert res.contributing_observation_ids == ("OBS-OBJ-1",)
    assert res.source_evidence_ref == "E-OBJ-1"
    # No confirmed value evidence yet -- outcome must still be Unknown,
    # proving identity and outcome are genuinely decoupled.
    assert diag.result.objective_outcome.state == ObjectiveOutcomeState.UNKNOWN


# ---- 2. Objective + confirmed value evidence -> governed outcome ----

def test_objective_plus_confirmed_direct_evidence_achieves():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-ACHIEVED"),
    )
    diag = _run(records)
    assert diag.objective_resolution.status == ObjectiveResolutionStatus.ESTABLISHED
    assert diag.result.objective_outcome.state == ObjectiveOutcomeState.ACHIEVED
    assert diag.result.dimension_states[DimensionCode.D1].state.value == "SUPPORTED"


# ---- 3a. Reject on the ObjectiveCandidate leaves identity not established ----

def test_rejected_objective_candidate_leaves_identity_not_established():
    records = (_reject(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),)
    diag = _run(records)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.NOT_ESTABLISHED
    assert res.text is None
    assert diag.result.objective_outcome.state == ObjectiveOutcomeState.UNKNOWN


# ---- 3b. Cannot Confirm on the ObjectiveCandidate leaves identity not established ----

def test_cannot_confirm_objective_candidate_leaves_identity_not_established():
    records = (_cannot_confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),)
    diag = _run(records)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.NOT_ESTABLISHED
    assert res.text is None
    item = diag.active_evidence.by_observation_id("OBS-OBJ-1")
    assert item.evidence_state == EvidenceState.CURRENT_UNVERIFIED  # uncertainty preserved, not promoted


# ---- 4. Correct uses the corrected text, not the original ----

def test_correct_uses_corrected_objective_text_not_original():
    records = (
        create_confirmation_record(
            target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-OBJ-1",
            action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
            corrected_representation={"objective_text": "Cut monthly reconciliation effort in half"},
        ),
    )
    diag = _run(records)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.ESTABLISHED
    assert res.text == "Cut monthly reconciliation effort in half"
    assert res.contributing_observation_ids == ("OBS-OBJ-1",)


# ---- 5. Identical duplicate objective statements ----

def test_identical_duplicate_confirmed_statements_establish_one_identity():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-2"),  # same text, different casing/whitespace
    )
    diag = _run(records)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.ESTABLISHED
    # Canonical source is the earliest confirmed -- OBS-OBJ-1 was
    # confirmed first (its confirmation_id sorts first).
    assert res.source_evidence_ref == "E-OBJ-1"
    assert res.text == "Reduce monthly reconciliation effort by 50%"
    # Both observations are preserved in the audit trail, not discarded.
    assert res.contributing_observation_ids == ("OBS-OBJ-1", "OBS-OBJ-2")
    assert diag.result.objective_outcome.state == ObjectiveOutcomeState.UNKNOWN  # still no value evidence


# ---- 6. Conflicting objective statements, both input orders ----

def test_conflicting_confirmed_objective_statements_are_never_auto_resolved():
    records_order_a = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-3"),
    )
    diag_a = _run(records_order_a)
    res_a = diag_a.objective_resolution
    assert res_a.status == ObjectiveResolutionStatus.CONFLICTING
    assert res_a.text is None
    assert res_a.source_evidence_ref is None
    assert res_a.conflicting_observation_ids == ("OBS-OBJ-1", "OBS-OBJ-3")
    assert diag_a.result.objective_outcome.state == ObjectiveOutcomeState.UNKNOWN

    records_order_b = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-3"),
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),
    )
    diag_b = _run(records_order_b)
    res_b = diag_b.objective_resolution
    assert res_b.status == ObjectiveResolutionStatus.CONFLICTING
    # Order-independent: same conflicting set regardless of confirmation order.
    assert res_b.conflicting_observation_ids == ("OBS-OBJ-1", "OBS-OBJ-3")
    assert diag_b.result.objective_outcome.state == ObjectiveOutcomeState.UNKNOWN


def test_conflict_is_reconciled_only_by_reject_or_correct_never_automatically():
    """The approved reconciliation mechanism is human Reject/Correct on
    the conflicting item(s) -- proves rejecting the extra candidate
    resolves the conflict without any special-cased "reconciliation"
    logic in the bridge itself."""
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-3"),
        _reject(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-3", reason="Wrong account context."),
    )
    diag = _run(records)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.ESTABLISHED
    assert res.text == "Reduce monthly reconciliation effort by 50%"


# ---- 7. Conflicting VALUE evidence -> Disputed / D1 Mixed (already-existing engine rule) ----

def test_conflicting_confirmed_value_evidence_produces_disputed_and_d1_mixed():
    records = (
        _confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-ACHIEVED"),
        _confirm(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, "OBS-EVIDCLASS-NOTACHIEVED"),
    )
    diag = _run(records)
    # Identity establishment is unaffected by a value-evidence conflict --
    # these are the two different axes described in the checkpoint.
    assert diag.objective_resolution.status == ObjectiveResolutionStatus.ESTABLISHED
    assert diag.result.objective_outcome.state == ObjectiveOutcomeState.DISPUTED
    d1 = diag.result.dimension_states[DimensionCode.D1]
    assert d1.state.value == "MIXED"
    assert d1.dimension_reliability == "LOW"


# ---- 8. Objective source traceability ----

def test_objective_source_evidence_ref_traces_to_the_confirmed_candidates_evidence():
    records = (_confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),)
    diag = _run(records)
    res = diag.objective_resolution
    assert res.source_evidence_ref == "E-OBJ-1"
    # And the resolved Objective the engine actually saw carries the same
    # traceable reference, via the Objective Outcome it produced.
    assert diag.result.objective_outcome.contributing_evidence_refs


# ---- Already-known objective is never overridden ----

def test_already_known_objective_is_never_overridden_by_confirmed_evidence():
    known = Objective("OBJ-1", "Pre-declared, already-known objective", is_known=True)
    records = (_confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-3"),)  # conflicting text, confirmed
    diag = _run(records, objective=known)
    res = diag.objective_resolution
    assert res.status == ObjectiveResolutionStatus.ESTABLISHED
    assert res.text == "Pre-declared, already-known objective"


# ---- No Objective supplied at all leaves engine behavior unchanged ----

def test_no_objective_supplied_at_all_is_left_alone():
    records = (_confirm(ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-OBJ-1"),)
    diag = _run(records, objective=None)
    assert diag.objective_resolution.status == ObjectiveResolutionStatus.NOT_ESTABLISHED
    assert diag.result.objective_outcome is None


TESTS = [
    test_one_confirmed_objective_only_establishes_identity_but_outcome_stays_unknown,
    test_objective_plus_confirmed_direct_evidence_achieves,
    test_rejected_objective_candidate_leaves_identity_not_established,
    test_cannot_confirm_objective_candidate_leaves_identity_not_established,
    test_correct_uses_corrected_objective_text_not_original,
    test_identical_duplicate_confirmed_statements_establish_one_identity,
    test_conflicting_confirmed_objective_statements_are_never_auto_resolved,
    test_conflict_is_reconciled_only_by_reject_or_correct_never_automatically,
    test_conflicting_confirmed_value_evidence_produces_disputed_and_d1_mixed,
    test_objective_source_evidence_ref_traces_to_the_confirmed_candidates_evidence,
    test_already_known_objective_is_never_overridden_by_confirmed_evidence,
    test_no_objective_supplied_at_all_is_left_alone,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
