"""
Milestone 3A — HumanConfirmationRecord schema tests, including the
reviewer-rationale amendment: REJECT / CANNOT_CONFIRM require a non-empty
reason; CONFIRM / CORRECT leave reason optional (CORRECT still requires
corrected_representation).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.enums import ConfirmationAction
from confirmation.enums import ConfirmationTargetKind
from confirmation.schemas import HumanConfirmationRecord, create_confirmation_record


def _base_kwargs(**overrides):
    kwargs = dict(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION,
        target_observation_id="OBS-000001",
        action=ConfirmationAction.CONFIRM,
        reviewer="eduardo.lteran@example.com",
    )
    kwargs.update(overrides)
    return kwargs


def test_reject_without_reason_fails():
    with pytest.raises(ValueError, match="reason is required"):
        create_confirmation_record(**_base_kwargs(action=ConfirmationAction.REJECT))


def test_reject_with_reason_is_valid():
    record = create_confirmation_record(
        **_base_kwargs(action=ConfirmationAction.REJECT, reason="AI merged two separate events."),
    )
    assert record.action == ConfirmationAction.REJECT
    assert record.reason == "AI merged two separate events."


def test_cannot_confirm_without_reason_fails():
    with pytest.raises(ValueError, match="reason is required"):
        create_confirmation_record(**_base_kwargs(action=ConfirmationAction.CANNOT_CONFIRM))


def test_cannot_confirm_with_reason_is_valid():
    record = create_confirmation_record(
        **_base_kwargs(action=ConfirmationAction.CANNOT_CONFIRM, reason="Evidence is incomplete."),
    )
    assert record.action == ConfirmationAction.CANNOT_CONFIRM


def test_confirm_without_reason_is_valid():
    record = create_confirmation_record(**_base_kwargs(action=ConfirmationAction.CONFIRM))
    assert record.reason is None


def test_correct_without_reason_is_valid_if_corrected_representation_valid():
    record = create_confirmation_record(**_base_kwargs(
        action=ConfirmationAction.CORRECT,
        corrected_representation={"observed_behavior": "Corrected: used weekly, not daily."},
    ))
    assert record.reason is None
    assert record.corrected_representation["observed_behavior"].startswith("Corrected")


def test_correct_without_corrected_representation_fails():
    with pytest.raises(ValueError, match="corrected_representation is required"):
        create_confirmation_record(**_base_kwargs(action=ConfirmationAction.CORRECT))


def test_correct_with_empty_corrected_representation_fails():
    with pytest.raises(ValueError, match="corrected_representation is required"):
        create_confirmation_record(**_base_kwargs(action=ConfirmationAction.CORRECT, corrected_representation={}))


def test_reviewer_required_non_empty():
    with pytest.raises(ValueError, match="reviewer is required"):
        create_confirmation_record(**_base_kwargs(reviewer=""))


def test_reviewer_cannot_be_reserved_non_human_identifier():
    """Structural guard: AI cannot self-confirm."""
    for bad_reviewer in ("AI", "system", "Model", "extraction_pipeline", "BOT"):
        with pytest.raises(ValueError, match="reserved non-human identifier"):
            create_confirmation_record(**_base_kwargs(reviewer=bad_reviewer))


def test_target_observation_id_required():
    with pytest.raises(ValueError, match="target_observation_id must not be empty"):
        create_confirmation_record(**_base_kwargs(target_observation_id=""))


def test_create_confirmation_record_assigns_monotonic_sequence():
    r1 = create_confirmation_record(**_base_kwargs())
    r2 = create_confirmation_record(**_base_kwargs())
    assert r2.sequence > r1.sequence
    assert r1.confirmation_id != r2.confirmation_id


def test_hand_authored_record_with_duplicate_sequence_still_constructs():
    """HumanConfirmationRecord itself does not enforce global sequence
    uniqueness (that would require a shared registry) -- ambiguity is
    caught downstream by state_machine.resolve_terminal(), tested
    separately. Hand-authoring is still useful for state-machine tests
    that need explicit control over ordering."""
    record = HumanConfirmationRecord(
        confirmation_id="CONF-TEST-1", sequence=1,
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION,
        target_observation_id="OBS-000001", action=ConfirmationAction.CONFIRM,
        reviewer="a.reviewer@example.com",
    )
    assert record.sequence == 1


TESTS = [
    test_reject_without_reason_fails,
    test_reject_with_reason_is_valid,
    test_cannot_confirm_without_reason_fails,
    test_cannot_confirm_with_reason_is_valid,
    test_confirm_without_reason_is_valid,
    test_correct_without_reason_is_valid_if_corrected_representation_valid,
    test_correct_without_corrected_representation_fails,
    test_correct_with_empty_corrected_representation_fails,
    test_reviewer_required_non_empty,
    test_reviewer_cannot_be_reserved_non_human_identifier,
    test_target_observation_id_required,
    test_create_confirmation_record_assigns_monotonic_sequence,
    test_hand_authored_record_with_duplicate_sequence_still_constructs,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
