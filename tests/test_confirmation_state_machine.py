"""
Milestone 3A — confirmation append-only journal resolution tests.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.enums import ConfirmationAction
from confirmation.enums import ConfirmationTargetKind
from confirmation.schemas import HumanConfirmationRecord
from confirmation.state_machine import group_by_target, resolve_terminal, validate_transition_sequence


def _record(seq, action, target="OBS-1", reason=None, corrected=None):
    return HumanConfirmationRecord(
        confirmation_id=f"CONF-{seq:06d}", sequence=seq,
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION,
        target_observation_id=target, action=action,
        reviewer="a.reviewer@example.com", reason=reason,
        corrected_representation=corrected,
    )


def test_resolve_terminal_none_when_no_records():
    assert resolve_terminal(()) is None


def test_resolve_terminal_picks_highest_sequence():
    r1 = _record(1, ConfirmationAction.CANNOT_CONFIRM, reason="Evidence is incomplete.")
    r2 = _record(2, ConfirmationAction.CONFIRM)
    terminal = resolve_terminal([r2, r1])  # order-independent
    assert terminal is r2


def test_resolve_terminal_supports_rereview_confirm_after_cannot_confirm():
    r1 = _record(1, ConfirmationAction.CANNOT_CONFIRM, reason="Source is ambiguous.")
    r2 = _record(2, ConfirmationAction.CONFIRM)
    assert resolve_terminal([r1, r2]).action == ConfirmationAction.CONFIRM


def test_resolve_terminal_supports_reject_after_confirm():
    """Re-review can move the other direction too -- a later audit finds a
    problem with something previously confirmed."""
    r1 = _record(1, ConfirmationAction.CONFIRM)
    r2 = _record(2, ConfirmationAction.REJECT, reason="Extracted fact is materially inaccurate.")
    assert resolve_terminal([r1, r2]).action == ConfirmationAction.REJECT


def test_resolve_terminal_raises_on_duplicate_sequence():
    r1 = _record(1, ConfirmationAction.CONFIRM)
    r2 = _record(1, ConfirmationAction.REJECT, reason="Source refers to the wrong account.")
    with pytest.raises(ValueError, match="Ambiguous terminal disposition"):
        resolve_terminal([r1, r2])


def test_group_by_target():
    r1 = _record(1, ConfirmationAction.CONFIRM, target="OBS-1")
    r2 = _record(2, ConfirmationAction.CONFIRM, target="OBS-2")
    r3 = _record(3, ConfirmationAction.REJECT, target="OBS-1", reason="AI merged two separate events.")
    grouped = group_by_target([r1, r2, r3])
    assert set(grouped) == {"OBS-1", "OBS-2"}
    assert grouped["OBS-1"] == [r1, r3]
    assert grouped["OBS-2"] == [r2]


# ---- Correction regression: REJECTED -> CONFIRM must be illegal ----

def test_reject_then_bare_confirm_is_illegal():
    """A rejected item must never be reinstated as active/authoritative by
    a bare re-CONFIRM of the original -- that would resurrect exactly the
    object a human reviewer determined was wrong. Governing rule:
    REJECTED -> CONFIRM is illegal."""
    r1 = _record(1, ConfirmationAction.REJECT, reason="Source refers to the wrong account.")
    r2 = _record(2, ConfirmationAction.CONFIRM)
    with pytest.raises(ValueError, match="A REJECTED item may never transition directly to CONFIRM"):
        resolve_terminal([r1, r2])


def test_reject_then_confirm_illegal_even_if_not_the_final_hop():
    """The illegal transition is caught wherever in the chain it occurs,
    not just at the final hop -- REJECT -> CONFIRM -> CORRECT still
    contains the forbidden direct reinstatement and must still raise."""
    r1 = _record(1, ConfirmationAction.REJECT, reason="AI merged two separate events.")
    r2 = _record(2, ConfirmationAction.CONFIRM)
    r3 = _record(3, ConfirmationAction.CORRECT, corrected={"observed_behavior": "corrected text"})
    with pytest.raises(ValueError, match="A REJECTED item may never transition directly to CONFIRM"):
        resolve_terminal([r1, r2, r3])


def test_reject_then_correct_is_the_legal_reinstatement_path():
    """The approved reinstatement path: REJECTED -> CORRECT (a new
    replacement representation), never REJECTED -> CONFIRM directly."""
    r1 = _record(1, ConfirmationAction.REJECT, reason="Source refers to the wrong account.")
    r2 = _record(2, ConfirmationAction.CORRECT, corrected={"observed_behavior": "corrected text"})
    terminal = resolve_terminal([r1, r2])
    assert terminal is r2
    assert terminal.action == ConfirmationAction.CORRECT


def test_reject_then_cannot_confirm_remains_legal():
    """Only REJECTED -> CONFIRM is restricted -- every other transition,
    including REJECTED -> CANNOT_CONFIRM, is unaffected by this
    correction."""
    r1 = _record(1, ConfirmationAction.REJECT, reason="Evidence does not support the proposed statement.")
    r2 = _record(2, ConfirmationAction.CANNOT_CONFIRM, reason="Reviewer lacks sufficient basis to decide.")
    terminal = resolve_terminal([r1, r2])
    assert terminal.action == ConfirmationAction.CANNOT_CONFIRM


def test_confirm_then_reject_remains_legal():
    """Sanity check that the correction is directional -- CONFIRM -> REJECT
    (an audit finds a problem with something previously confirmed) is not
    restricted, only REJECT -> CONFIRM is."""
    r1 = _record(1, ConfirmationAction.CONFIRM)
    r2 = _record(2, ConfirmationAction.REJECT, reason="Extracted fact is materially inaccurate.")
    validate_transition_sequence([r1, r2])  # does not raise
    assert resolve_terminal([r1, r2]).action == ConfirmationAction.REJECT


TESTS = [
    test_resolve_terminal_none_when_no_records,
    test_resolve_terminal_picks_highest_sequence,
    test_resolve_terminal_supports_rereview_confirm_after_cannot_confirm,
    test_resolve_terminal_supports_reject_after_confirm,
    test_resolve_terminal_raises_on_duplicate_sequence,
    test_group_by_target,
    test_reject_then_bare_confirm_is_illegal,
    test_reject_then_confirm_illegal_even_if_not_the_final_hop,
    test_reject_then_correct_is_the_legal_reinstatement_path,
    test_reject_then_cannot_confirm_remains_legal,
    test_confirm_then_reject_remains_legal,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
