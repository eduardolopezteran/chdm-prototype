"""
Milestone 3E — ui/actions.submit_contradiction_review + ui/review_queue.find_any_item.

Mirrors tests/test_ui_actions.py's style for the existing submit_action
glue: proves the new contradiction-review entry point appends a record
only on success, never touches state.confirmation_records or
state.current_diagnostic (contradiction review never triggers a
recompute() -- the governed diagnostic is unaffected by design), and that
find_any_item() correctly resolves both a semantic observation and a
MissingInformationCandidate reference without the caller needing to guess
which kind it is.
"""
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import DimensionCode, Lifecycle
from domain.objective import Objective
from engine.registry_loader import load_and_validate
from extraction.enums import InferenceBasis, ObservationType
from extraction.pipeline import ExtractionResult
from extraction.schemas import (
    AdoptionObservation, CandidateContradiction, ExtractionSystemFields, MissingInformationCandidate,
    ObservationRef, SourceSpan,
)

from confirmation.enums import ContradictionReviewAction

from ui.actions import initial_recompute, submit_contradiction_review
from ui.review_queue import find_any_item
from ui.state import AppState


def _system(obs_id: str) -> ExtractionSystemFields:
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="test-provider", model_version="test-v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"TRACE-{obs_id}",
    )


def _span(text: str) -> SourceSpan:
    return SourceSpan(text=text, start_char=0, end_char=len(text))


def _fresh_state() -> AppState:
    obs_a = AdoptionObservation(
        source_evidence_id="EVID-1", source_span=_span("used weekly by the finance team"),
        basis=InferenceBasis.EXPLICIT, workflow_or_use_case="Weekly reconciliation",
        observed_behavior="Used weekly by the finance team", system=_system("OBS-A"),
    )
    missing = MissingInformationCandidate(
        missing_item="renewal date", reviewed_evidence_ids=("EVID-1",), system=_system("OBS-MISSING-1"),
    )
    contradiction = CandidateContradiction(
        observation_ref_a=ObservationRef(ObservationType.ADOPTION_OBSERVATION, 0),
        observation_ref_b=ObservationRef(ObservationType.ADOPTION_OBSERVATION, 0),
        conflict_description="placeholder conflict for UI-level glue test",
        resolved_observation_id_a="OBS-A", resolved_observation_id_b="OBS-MISSING-1",
        system=_system("CONTRA-1"),
    )
    extraction_result = ExtractionResult(
        accepted=(obs_a, missing), candidate_contradictions=(contradiction,),
        candidate_risk_signals=(), candidate_evidence_classifications=(),
        rejected=(), dedup_audit=(), traces=(),
    )
    registry = load_and_validate()
    objective = Objective("OBJ-1", "Reduce reconciliation time", is_known=True)
    account = AccountAssessment(
        assessment_id="ASSESS-M3E-TEST",
        scope=Scope("SCOPE-M3E-TEST", "Test account", "Milestone 3E test"),
        lifecycle=Lifecycle.L3, objective=objective,
    )
    state = AppState(
        account=account, registry=registry, extraction_result=extraction_result,
        dimensions_to_evaluate=(DimensionCode.D1, DimensionCode.D2, DimensionCode.D6),
        dimension_qualifier_overrides={}, reviewer="a.reviewer@example.com",
    )
    state.current_diagnostic = initial_recompute(state)
    return state


def test_submit_contradiction_review_acknowledge_appends_record():
    state = _fresh_state()
    record = submit_contradiction_review(
        state, contradiction_id="CONTRA-1", action=ContradictionReviewAction.ACKNOWLEDGE,
    )
    assert state.contradiction_review_records == [record]
    assert record.action == ContradictionReviewAction.ACKNOWLEDGE


def test_submit_contradiction_review_dismiss_requires_reason_and_does_not_append():
    state = _fresh_state()
    with pytest.raises(ValueError, match="reason is required"):
        submit_contradiction_review(
            state, contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS,
        )
    assert state.contradiction_review_records == []


def test_contradiction_review_never_touches_confirmation_records_or_diagnostic():
    state = _fresh_state()
    diagnostic_before = state.current_diagnostic
    submit_contradiction_review(
        state, contradiction_id="CONTRA-1", action=ContradictionReviewAction.DISMISS,
        reason="Not actually contradictory -- different time windows.",
    )
    # No recompute() call happens for contradiction review -- the governed
    # diagnostic object is untouched (identity check, not just equality).
    assert state.current_diagnostic is diagnostic_before
    assert state.confirmation_records == []


def test_find_any_item_resolves_semantic_observation():
    state = _fresh_state()
    result = find_any_item(state.extraction_result, "OBS-A")
    assert result is not None
    kind, obs = result
    assert kind.value == "SEMANTIC_OBSERVATION"
    assert obs.system.observation_id == "OBS-A"


def test_find_any_item_resolves_missing_information_candidate():
    state = _fresh_state()
    result = find_any_item(state.extraction_result, "OBS-MISSING-1")
    assert result is not None
    kind, obs = result
    assert kind.value == "MISSING_INFORMATION_CANDIDATE"
    assert obs.missing_item == "renewal date"


def test_find_any_item_returns_none_for_unknown_id():
    state = _fresh_state()
    assert find_any_item(state.extraction_result, "OBS-DOES-NOT-EXIST") is None
