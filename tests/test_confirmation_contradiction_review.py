"""
Milestone 3E — contradiction-review tests.

Hand-authors pipeline-finalized extraction objects directly (system fields
already populated), same style as tests/test_confirmation_active_evidence.py,
plus a CandidateContradiction referencing two of them. Proves:
  - ContradictionReviewRecord validation (reviewer required, DISMISS
    requires a non-empty reason, ACKNOWLEDGE may leave it optional);
  - resolve_contradiction_terminal's re-review / duplicate-sequence
    behavior, mirroring confirmation.state_machine.resolve_terminal;
  - conflict_status's live derivation from a real ActiveEvidenceSet across
    every combination the approved checkpoint named (both active; one
    rejected; one corrected; both rejected);
  - contradiction review is a COMPLETELY SEPARATE journal from
    HumanConfirmationRecord: it never appears in state.confirmation_records
    machinery, never affects active_evidence, and the underlying
    observations remain independently confirmable/correctable/rejectable
    regardless of any contradiction-review action taken on the marker that
    references them.
"""

import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.enums import ConfirmationAction, EvidenceState
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    AdoptionObservation, CandidateContradiction, ExtractionSystemFields, ObservationRef, SourceSpan,
    StakeholderObservation,
)

from confirmation.active_evidence import reconstruct_active_evidence
from confirmation.contradiction_review import (
    ContradictionReviewRecord,
    conflict_status,
    create_contradiction_review_record,
    group_by_contradiction,
    resolve_contradiction_terminal,
)
from confirmation.enums import ConfirmationTargetKind, ContradictionReviewAction
from confirmation.schemas import create_confirmation_record


def _system(obs_id: str, evidence_state=EvidenceState.CURRENT_UNVERIFIED) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="test-provider", model_version="test-v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
        evidence_state=evidence_state,
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def build_extraction_result_with_contradiction():
    obs_a = AdoptionObservation(
        source_evidence_id="EVID-1", source_span=_span("used weekly by the finance team"),
        basis=InferenceBasis.EXPLICIT, workflow_or_use_case="Weekly reconciliation",
        observed_behavior="Used weekly by the finance team", system=_system("OBS-A"),
    )
    obs_b = StakeholderObservation(
        source_evidence_id="EVID-2", source_span=_span("champion says adoption has stalled"),
        basis=InferenceBasis.EXPLICIT, person_identifier="Champion",
        role="Champion", system=_system("OBS-B"),
    )
    contradiction = CandidateContradiction(
        observation_ref_a=ObservationRef(ObservationType.ADOPTION_OBSERVATION, 0),
        observation_ref_b=ObservationRef(ObservationType.STAKEHOLDER_OBSERVATION, 0),
        conflict_description="One item says adoption is active weekly use; the other says adoption has stalled.",
        methodology_construct_hint="the objective's realized outcome",
        resolved_observation_id_a="OBS-A", resolved_observation_id_b="OBS-B",
        system=_system("CONTRA-1"),
    )
    return ExtractionResult(
        accepted=(obs_a, obs_b), candidate_contradictions=(contradiction,),
        candidate_risk_signals=(), candidate_evidence_classifications=(),
        rejected=(), dedup_audit=(), traces=(),
    ), contradiction


# ---- ContradictionReviewRecord validation ----

def test_acknowledge_allows_empty_reason():
    record = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
    )
    assert record.action == ContradictionReviewAction.ACKNOWLEDGE
    assert record.reason is None
    assert record.record_id.startswith("CONTRA-REVIEW-")


def test_dismiss_requires_non_empty_reason():
    with pytest.raises(ValueError, match="reason is required"):
        create_contradiction_review_record(
            contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Dana",
        )
    with pytest.raises(ValueError, match="reason is required"):
        create_contradiction_review_record(
            contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Dana", reason="   ",
        )


def test_dismiss_with_reason_succeeds():
    record = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Dana",
        reason="The two observations describe different time periods, not a real conflict.",
    )
    assert record.action == ContradictionReviewAction.DISMISS
    assert record.reason


def test_reviewer_required():
    with pytest.raises(ValueError, match="reviewer is required"):
        create_contradiction_review_record(
            contradiction_id="CONTRA-1", action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="",
        )


def test_contradiction_id_required():
    with pytest.raises(ValueError, match="contradiction_id must not be empty"):
        ContradictionReviewRecord(
            record_id="CONTRA-REVIEW-000001", sequence=1, contradiction_id="",
            action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
        )


# ---- resolve_contradiction_terminal / group_by_contradiction ----

def test_resolve_terminal_empty_is_none():
    assert resolve_contradiction_terminal(()) is None


def test_resolve_terminal_single_record():
    record = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
    )
    assert resolve_contradiction_terminal((record,)) is record


def test_resolve_terminal_re_review_latest_wins():
    ack = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
    )
    dismiss = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Priya",
        reason="Reviewed again -- not actually contradictory.",
    )
    terminal = resolve_contradiction_terminal((ack, dismiss))
    assert terminal is dismiss
    assert terminal.action == ContradictionReviewAction.DISMISS


def test_resolve_terminal_duplicate_sequence_raises():
    ack = ContradictionReviewRecord(
        record_id="CONTRA-REVIEW-000001", sequence=1, contradiction_id="CONTRA-1",
        action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
    )
    dup = ContradictionReviewRecord(
        record_id="CONTRA-REVIEW-000002", sequence=1, contradiction_id="CONTRA-1",
        action=ContradictionReviewAction.DISMISS, reviewer="Priya", reason="conflict",
    )
    with pytest.raises(ValueError, match="Ambiguous terminal disposition"):
        resolve_contradiction_terminal((ack, dup))


def test_group_by_contradiction_separates_targets():
    r1 = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
    )
    r2 = create_contradiction_review_record(
        contradiction_id="CONTRA-2", action=ContradictionReviewAction.ACKNOWLEDGE, reviewer="Dana",
    )
    grouped = group_by_contradiction((r1, r2))
    assert set(grouped) == {"CONTRA-1", "CONTRA-2"}
    assert grouped["CONTRA-1"] == [r1]


# ---- conflict_status: live derivation ----

def test_conflict_status_both_active():
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    active = reconstruct_active_evidence(extraction_result, ())
    status = conflict_status(active, contradiction)
    assert status.both_active
    assert status.side_a.status == "active"
    assert status.side_b.status == "active"


def test_conflict_status_one_rejected():
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    reject = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-A",
        action=ConfirmationAction.REJECT, reviewer="Dana", reason="not accurate",
    )
    active = reconstruct_active_evidence(extraction_result, (reject,))
    status = conflict_status(active, contradiction)
    assert not status.both_active
    assert not status.both_rejected
    assert status.side_a.status == "rejected"
    assert status.side_b.status == "active"


def test_conflict_status_one_corrected():
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    correct = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-B",
        action=ConfirmationAction.CORRECT, reviewer="Dana",
        corrected_representation={"observed_statement_or_behavior": "Adoption slowed but has not stalled."},
    )
    active = reconstruct_active_evidence(extraction_result, (correct,))
    status = conflict_status(active, contradiction)
    assert status.side_a.status == "active"
    assert status.side_b.status == "corrected"


def test_conflict_status_both_rejected():
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    reject_a = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-A",
        action=ConfirmationAction.REJECT, reviewer="Dana", reason="not accurate",
    )
    reject_b = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-B",
        action=ConfirmationAction.REJECT, reviewer="Dana", reason="not accurate either",
    )
    active = reconstruct_active_evidence(extraction_result, (reject_a, reject_b))
    status = conflict_status(active, contradiction)
    assert status.both_rejected
    assert not status.both_active


# ---- Independence from HumanConfirmationRecord / underlying observations ----

def test_contradiction_review_does_not_touch_confirmation_records_journal():
    # ContradictionReviewRecord and HumanConfirmationRecord are entirely
    # separate journals by construction -- this test simply documents that
    # a contradiction-review record is never a HumanConfirmationRecord and
    # carries no target_kind/EvidenceState-shaped fields at all.
    record = create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Dana",
        reason="not a real conflict",
    )
    assert not hasattr(record, "target_kind")
    assert not hasattr(record, "corrected_representation")


def test_underlying_observations_remain_independently_confirmable_after_dismiss():
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    # Dismissing the contradiction marker itself...
    create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Dana",
        reason="not a real conflict",
    )
    # ...has no bearing on whether the two referenced observations can
    # still be normally confirmed -- active_evidence never sees the
    # contradiction-review journal at all (reconstruct_active_evidence's
    # signature doesn't even accept one).
    confirm_a = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-A",
        action=ConfirmationAction.CONFIRM, reviewer="Dana",
    )
    confirm_b = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-B",
        action=ConfirmationAction.CONFIRM, reviewer="Dana",
    )
    active = reconstruct_active_evidence(extraction_result, (confirm_a, confirm_b))
    assert active.by_observation_id("OBS-A").evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert active.by_observation_id("OBS-B").evidence_state == EvidenceState.CURRENT_CONFIRMED


def test_candidate_contradiction_status_field_never_mutated():
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    create_contradiction_review_record(
        contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS, reviewer="Dana",
        reason="not a real conflict",
    )
    # The underlying CandidateContradiction object is never written back to
    # by this module -- status stays the literal "CANDIDATE" string
    # forever (extraction/schemas.py's own __post_init__ guard).
    assert contradiction.status == "CANDIDATE"


def test_contradiction_not_indexed_into_active_evidence_itself():
    # A CandidateContradiction is never itself an ActiveEvidenceItem or an
    # ExclusionRecord target -- confirmation/active_evidence.py's
    # _index_extraction_items() never iterates candidate_contradictions at
    # all (Milestone 3E design: contradictions are reviewed through an
    # entirely separate, parallel mechanism, never through the
    # HumanConfirmationRecord/ActiveEvidenceSet machinery built for
    # observations).
    extraction_result, contradiction = build_extraction_result_with_contradiction()
    active = reconstruct_active_evidence(extraction_result, ())
    assert active.by_observation_id("CONTRA-1") is None
    assert not active.is_excluded("CONTRA-1")
