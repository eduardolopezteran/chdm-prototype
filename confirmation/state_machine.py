"""
Milestone 3A — confirmation state machine.

Confirmation records are an append-only journal (mirrors this codebase's
existing audit-first philosophy: nothing is ever mutated or deleted — see
extraction/pipeline.py's dedup_audit / ExtractionValidationFailure
patterns). A target may be reviewed more than once — re-review is
explicitly supported (e.g. CANNOT_CONFIRM now, CONFIRM later once more
evidence arrives, or CONFIRM now, REJECT later if a downstream check finds
a problem). The TERMINAL disposition for a target is always its
highest-`sequence` record.

Correction (post-implementation review): this module previously claimed
"no action is forbidden from following any other" — that was wrong. The
approved state-machine rule forbids exactly one direct transition:

    REJECTED -> CONFIRM   is ILLEGAL

A rejected item must never be reinstated as the active/authoritative
object by a bare re-CONFIRM of the original -- that would resurrect
exactly the object a human reviewer determined was wrong. Reinstatement
must go through CORRECT instead: REJECTED -> CORRECT -> (new replacement
representation) -> Current+Confirmed. The rejected original stays in the
audit history forever (see active_evidence.ActiveEvidenceItem.original)
but is never again treated as authoritative. Every other transition
remains unrestricted -- this module still does not forbid, for example,
CONFIRM -> REJECT (an audit finds a problem with something previously
confirmed) or CANNOT_CONFIRM -> CONFIRM (more evidence arrives).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from domain.enums import ConfirmationAction

from .schemas import HumanConfirmationRecord

# The only currently-governed illegal direct transitions. Expressed as
# (previous_action, next_action) pairs so this stays a table lookup, not
# scattered if/else logic, should more restrictions ever be approved.
_ILLEGAL_DIRECT_TRANSITIONS: FrozenSet[Tuple[ConfirmationAction, ConfirmationAction]] = frozenset({
    (ConfirmationAction.REJECT, ConfirmationAction.CONFIRM),
})


def group_by_target(
    records: Sequence[HumanConfirmationRecord],
) -> Dict[str, List[HumanConfirmationRecord]]:
    """All confirmation records, grouped by target_observation_id. Order
    within each group is preserved from the input sequence (callers that
    care about chronology should pass records in recorded order, but
    resolve_terminal() below does not rely on input order — it sorts by
    `sequence` explicitly)."""
    grouped: Dict[str, List[HumanConfirmationRecord]] = {}
    for record in records:
        grouped.setdefault(record.target_observation_id, []).append(record)
    return grouped


def validate_transition_sequence(records: Sequence[HumanConfirmationRecord]) -> None:
    """Walks ONE target's full journal, ordered by `sequence`, and enforces
    every currently-governed illegal direct transition (see
    _ILLEGAL_DIRECT_TRANSITIONS above). Checks every consecutive pair in
    the whole history, not just the final hop -- an illegal transition is
    illegal wherever in the chain it occurs, even if a later, legal record
    moved the terminal disposition somewhere else. Raises ValueError naming
    both offending records if violated; records must already be sorted by
    sequence (resolve_terminal() does this before calling in)."""
    for prev, curr in zip(records, records[1:]):
        if (prev.action, curr.action) in _ILLEGAL_DIRECT_TRANSITIONS:
            raise ValueError(
                f"Illegal transition for target_observation_id={curr.target_observation_id!r}: "
                f"{prev.action.value} (sequence={prev.sequence}, confirmation_id="
                f"{prev.confirmation_id!r}) -> {curr.action.value} (sequence={curr.sequence}, "
                f"confirmation_id={curr.confirmation_id!r}). A REJECTED item may never "
                "transition directly to CONFIRM -- reinstate it via CORRECT (a new "
                "replacement representation), never by re-confirming the rejected original."
            )


def resolve_terminal(
    records: Sequence[HumanConfirmationRecord],
) -> Optional[HumanConfirmationRecord]:
    """The authoritative record for a single target: the one with the
    highest `sequence` among the records supplied. Returns None for an
    empty sequence (never reviewed). Raises if two records for the same
    target somehow share a sequence number — create_confirmation_record()'s
    monotonic counter should make this impossible in practice; this is a
    defense-in-depth guard against hand-authored test records with
    duplicate sequences, exactly like this codebase's other "should never
    fire in practice" guards (see extraction/enums.py RejectionReason
    comments). Also validates the full transition sequence (see
    validate_transition_sequence above) before returning the terminal
    record -- an illegal REJECTED -> CONFIRM anywhere in a target's
    history raises here, so no caller can ever observe a resolved
    "reinstated by bare CONFIRM" result."""
    if not records:
        return None
    ordered = sorted(records, key=lambda r: r.sequence)
    if len(ordered) >= 2 and ordered[-1].sequence == ordered[-2].sequence:
        raise ValueError(
            f"Ambiguous terminal disposition for target_observation_id="
            f"{ordered[-1].target_observation_id!r}: two records share "
            f"sequence={ordered[-1].sequence} — confirmation records must "
            "have distinct, monotonically-assigned sequence numbers."
        )
    validate_transition_sequence(ordered)
    return ordered[-1]
