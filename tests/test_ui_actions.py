"""
Milestone 3B — ui/actions.py orchestration glue tests.

Exercises the thin adapter over the real confirmation/ backend -- not a
re-test of confirmation/'s own logic (that is covered by the 282 backend
tests). These tests prove the glue: appends a record only on success,
keeps current/previous diagnostics in sync, and the lazy consequentiality
cache invalidates correctly against the confirmation-history version.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.enums import ConfirmationAction, OperationalPriority

from confirmation.enums import ConfirmationTargetKind

from ui.actions import evaluate_consequentiality, get_cached_consequentiality, initial_recompute, submit_action
from ui.sample_data import build_sample_scenario
from ui.state import AppState


def _fresh_state() -> AppState:
    account, registry, extraction_result, dims, overrides = build_sample_scenario()
    state = AppState(
        account=account, registry=registry, extraction_result=extraction_result,
        dimensions_to_evaluate=dims, dimension_qualifier_overrides=overrides,
        reviewer="a.reviewer@example.com",
    )
    state.current_diagnostic = initial_recompute(state)
    return state


def test_initial_recompute_with_empty_history_succeeds():
    state = _fresh_state()
    assert state.current_diagnostic is not None
    assert state.confirmation_history_version == 0


def test_submit_action_appends_record_and_updates_diagnostics():
    state = _fresh_state()
    before = state.current_diagnostic
    submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
        target_observation_id="OBS-RISK-CHAMPION", action=ConfirmationAction.CONFIRM,
    )
    assert state.confirmation_history_version == 1
    assert state.previous_diagnostic is before
    assert state.current_diagnostic is not before


def test_submit_action_invalid_reject_without_reason_raises_and_does_not_append():
    state = _fresh_state()
    with pytest.raises(ValueError, match="reason is required"):
        submit_action(
            state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
            target_observation_id="OBS-RISK-CHAMPION", action=ConfirmationAction.REJECT,
        )
    assert state.confirmation_history_version == 0  # nothing appended on a failed submission


def test_submit_action_illegal_reject_then_confirm_raises_through_the_glue():
    state = _fresh_state()
    submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
        target_observation_id="OBS-RISK-CHAMPION", action=ConfirmationAction.REJECT,
        reason="Extracted fact is materially inaccurate.",
    )
    with pytest.raises(ValueError, match="A REJECTED item may never transition directly to CONFIRM"):
        submit_action(
            state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
            target_observation_id="OBS-RISK-CHAMPION", action=ConfirmationAction.CONFIRM,
        )


def test_consequentiality_cache_starts_empty():
    state = _fresh_state()
    assert get_cached_consequentiality(state, "OBS-RISK-CHAMPION") is None


def test_consequentiality_cache_populated_after_evaluate():
    state = _fresh_state()
    report = evaluate_consequentiality(state, "OBS-RISK-CHAMPION")
    assert get_cached_consequentiality(state, "OBS-RISK-CHAMPION") is report


def test_consequentiality_cache_invalidated_by_a_new_confirmation_action():
    state = _fresh_state()
    evaluate_consequentiality(state, "OBS-RISK-CHAMPION")
    assert get_cached_consequentiality(state, "OBS-RISK-CHAMPION") is not None
    # A new action on a DIFFERENT target still bumps the shared history
    # version -- the cached entry must now read as stale (not-yet-
    # evaluated), per the approved lazy-cache design, since the
    # underlying diagnostic it was computed against is no longer current.
    submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
        target_observation_id="OBS-RISK-SERVICE", action=ConfirmationAction.CONFIRM,
    )
    assert get_cached_consequentiality(state, "OBS-RISK-CHAMPION") is None


def test_end_to_end_confirming_champion_departure_raises_op_to_op1():
    state = _fresh_state()
    submit_action(
        state, target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION,
        target_observation_id="OBS-STAKE-1", action=ConfirmationAction.CONFIRM,
    )
    submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
        target_observation_id="OBS-RISK-CHAMPION", action=ConfirmationAction.CONFIRM,
    )
    assert state.current_diagnostic.result.operational_priority.value == OperationalPriority.OP1


TESTS = [
    test_initial_recompute_with_empty_history_succeeds,
    test_submit_action_appends_record_and_updates_diagnostics,
    test_submit_action_invalid_reject_without_reason_raises_and_does_not_append,
    test_submit_action_illegal_reject_then_confirm_raises_through_the_glue,
    test_consequentiality_cache_starts_empty,
    test_consequentiality_cache_populated_after_evaluate,
    test_consequentiality_cache_invalidated_by_a_new_confirmation_action,
    test_end_to_end_confirming_champion_departure_raises_op_to_op1,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
