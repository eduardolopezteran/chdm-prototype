"""
Milestone 3A — active-evidence reconstruction tests.

Hand-authors pipeline-finalized extraction objects directly (system fields
already populated), mirroring tests/test_evaluate_end_to_end.py's
hand-authored-signal style, rather than running the full AI extraction
pipeline (that is exercised elsewhere, in tests/test_extraction_*).
"""

import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.enums import ConfirmationAction, EvidenceState
from extraction.enums import ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    AdoptionObservation, CandidateEvidenceClassification, CandidateRiskSignal,
    ExtractionSystemFields, MissingInformationCandidate, ObservationRef, SourceSpan,
)
from extraction.enums import InferenceBasis

from confirmation.active_evidence import reconstruct_active_evidence
from confirmation.enums import ConfirmationTargetKind, ExclusionReason
from confirmation.schemas import create_confirmation_record


def _system(obs_id: str, evidence_state=EvidenceState.CURRENT_UNVERIFIED) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="test-provider", model_version="test-v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
        evidence_state=evidence_state,
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def build_extraction_result():
    adoption = AdoptionObservation(
        source_evidence_id="EVID-1", source_span=_span("used weekly by the finance team"),
        basis=InferenceBasis.EXPLICIT, workflow_or_use_case="Weekly reconciliation",
        observed_behavior="Used weekly by the finance team", system=_system("OBS-ADOPT-1"),
    )
    missing = MissingInformationCandidate(
        missing_item="renewal date", reviewed_evidence_ids=("EVID-1", "EVID-2"),
        system=_system("OBS-MISSING-1"),
    )
    risk_signal = CandidateRiskSignal(
        source_evidence_id="EVID-1", source_span=_span("champion has not logged in for 60 days"),
        basis=InferenceBasis.INFERRED_CANDIDATE, mechanism="CR-01", proposed_severity_tier="CRITICAL",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.ADOPTION_OBSERVATION, index=0),
        resolved_observation_id="OBS-ADOPT-1", system=_system("OBS-RISK-1"),
    )
    evidence_classification = CandidateEvidenceClassification(
        source_evidence_id="EVID-1", source_span=_span("reconciliation time dropped by half"),
        basis=InferenceBasis.INFERRED_CANDIDATE, proposed_basis="MEASURED_OPERATIONAL_EVIDENCE",
        supports="ACHIEVED",
        supporting_observation_ref=ObservationRef(observation_type=ObservationType.ADOPTION_OBSERVATION, index=0),
        resolved_observation_id="OBS-ADOPT-1", system=_system("OBS-EVIDCLASS-1"),
    )
    return ExtractionResult(
        accepted=(adoption, missing), candidate_contradictions=(),
        candidate_risk_signals=(risk_signal,), candidate_evidence_classifications=(evidence_classification,),
        rejected=(), dedup_audit=(), traces=(),
    )


# ---- 1. Never reviewed -> stays active, Current+Unverified ----

def test_unreviewed_item_stays_active_unverified():
    active = reconstruct_active_evidence(build_extraction_result(), ())
    item = active.by_observation_id("OBS-ADOPT-1")
    assert item is not None
    assert item.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    assert item.confirmation_id is None
    assert not active.is_excluded("OBS-ADOPT-1")


# ---- 2. Unverified -> Confirmed ----

def test_confirm_promotes_to_current_confirmed():
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (record,))
    item = active.by_observation_id("OBS-RISK-1")
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.confirmation_id == record.confirmation_id
    assert item.is_correction is False


# ---- 3. Unverified -> Rejected: excluded from active evaluation ----

def test_reject_excludes_from_active_evidence():
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
        reason="Extracted fact is materially inaccurate.",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (record,))
    assert active.by_observation_id("OBS-RISK-1") is None
    assert active.is_excluded("OBS-RISK-1")
    excl = next(e for e in active.excluded if e.target_observation_id == "OBS-RISK-1")
    assert excl.reason == ExclusionReason.REJECTED
    assert excl.confirmation_id == record.confirmation_id
    assert excl.detail == "Extracted fact is materially inaccurate."


# ---- 4. Unverified -> Cannot Confirm: preserves uncertainty ----

def test_cannot_confirm_preserves_uncertainty():
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
        reason="Conflicting evidence remains unresolved.",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (record,))
    item = active.by_observation_id("OBS-RISK-1")
    assert item is not None
    assert not active.is_excluded("OBS-RISK-1")
    assert item.evidence_state == EvidenceState.CURRENT_UNVERIFIED  # unchanged, not promoted
    assert item.confirmation_id == record.confirmation_id  # still auditable as reviewed


# ---- 5. Unverified -> Corrected: original preserved, correction active ----

def test_correct_uses_corrected_representation_and_preserves_original():
    original = build_extraction_result()
    original_risk_obs = original.candidate_risk_signals[0]
    record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"proposed_severity_tier": "WATCH"},
    )
    active = reconstruct_active_evidence(original, (record,))
    item = active.by_observation_id("OBS-RISK-1")
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.is_correction is True
    assert item.representation["proposed_severity_tier"] == "WATCH"
    assert item.representation["mechanism"] == "CR-01"  # everything else carried over unchanged
    assert item.original is original_risk_obs  # original never discarded


# ---- 6. Semantic fact confirmed while its candidate classification is rejected ----

def test_semantic_observation_and_candidate_classification_are_independent_decisions():
    confirm_semantic = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-ADOPT-1",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    reject_risk = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
        reason="Evidence does not support the proposed statement.",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (confirm_semantic, reject_risk))
    assert active.by_observation_id("OBS-ADOPT-1").evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert active.is_excluded("OBS-RISK-1")


# ---- 7. Confirmation traceability ----

def test_confirmation_id_is_traceable_on_active_items_and_exclusions():
    confirm = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION,
        target_observation_id="OBS-EVIDCLASS-1", action=ConfirmationAction.CONFIRM,
        reviewer="a.reviewer@example.com",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (confirm,))
    item = active.by_observation_id("OBS-EVIDCLASS-1")
    assert item.confirmation_id == confirm.confirmation_id


# ---- 8. Unknown target raises rather than silently ignoring ----

def test_unknown_target_observation_id_raises():
    bad_record = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-DOES-NOT-EXIST",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    with pytest.raises(ValueError, match="not present in the supplied ExtractionResult"):
        reconstruct_active_evidence(build_extraction_result(), (bad_record,))


# ---- 9. Re-review: latest disposition governs ----

def test_re_review_latest_record_governs():
    first = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
        reason="Reviewer lacks sufficient basis to decide.",
    )
    second = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (first, second))
    item = active.by_observation_id("OBS-RISK-1")
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.confirmation_id == second.confirmation_id


# ---- 10. MissingInformationCandidate can be confirmed too ----

def test_missing_information_candidate_can_be_confirmed():
    confirm = create_confirmation_record(
        target_kind=ConfirmationTargetKind.MISSING_INFORMATION_CANDIDATE,
        target_observation_id="OBS-MISSING-1", action=ConfirmationAction.CONFIRM,
        reviewer="a.reviewer@example.com",
    )
    active = reconstruct_active_evidence(build_extraction_result(), (confirm,))
    item = active.by_observation_id("OBS-MISSING-1")
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.source_evidence_id == "EVID-1"  # first of reviewed_evidence_ids, representative only


# ---- 11. Correction regression: REJECTED -> CONFIRM raises end-to-end ----

def test_reject_then_bare_confirm_raises_through_reconstruct_active_evidence():
    reject = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
        reason="Source refers to the wrong account.",
    )
    bare_confirm = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    with pytest.raises(ValueError, match="A REJECTED item may never transition directly to CONFIRM"):
        reconstruct_active_evidence(build_extraction_result(), (reject, bare_confirm))


def test_reject_then_correct_reinstates_via_new_replacement_not_the_original():
    """The approved reinstatement path works end-to-end: REJECTED ->
    CORRECT produces a NEW active representation (Current+Confirmed), while
    the rejected original is preserved on `.original` but never itself
    resurrected as the authoritative object."""
    reject = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
        reason="Extracted fact is materially inaccurate.",
    )
    original_risk_obs = build_extraction_result().candidate_risk_signals[0]
    correct = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-1",
        action=ConfirmationAction.CORRECT, reviewer="a.reviewer@example.com",
        corrected_representation={"proposed_severity_tier": "MATERIAL"},
    )
    extraction_result = build_extraction_result()
    active = reconstruct_active_evidence(extraction_result, (reject, correct))
    item = active.by_observation_id("OBS-RISK-1")
    assert not active.is_excluded("OBS-RISK-1")
    assert item.evidence_state == EvidenceState.CURRENT_CONFIRMED
    assert item.is_correction is True
    assert item.representation["proposed_severity_tier"] == "MATERIAL"
    # the rejected original is preserved for audit, never mutated, and is
    # NOT itself what downstream signal-building reads (that reads
    # `.representation`, the new replacement) -- so it is preserved, not
    # resurrected as authoritative.
    assert item.original is extraction_result.candidate_risk_signals[0]
    assert item.original.proposed_severity_tier == "CRITICAL"  # original untouched


TESTS = [
    test_unreviewed_item_stays_active_unverified,
    test_confirm_promotes_to_current_confirmed,
    test_reject_excludes_from_active_evidence,
    test_cannot_confirm_preserves_uncertainty,
    test_correct_uses_corrected_representation_and_preserves_original,
    test_semantic_observation_and_candidate_classification_are_independent_decisions,
    test_confirmation_id_is_traceable_on_active_items_and_exclusions,
    test_unknown_target_observation_id_raises,
    test_re_review_latest_record_governs,
    test_missing_information_candidate_can_be_confirmed,
    test_reject_then_bare_confirm_raises_through_reconstruct_active_evidence,
    test_reject_then_correct_reinstates_via_new_replacement_not_the_original,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
