"""
Milestone 3E — Contradiction Review Integration.

A CandidateContradiction (extraction.schemas.CandidateContradiction) is a
flagged RELATIONSHIP between two separately-reviewable evidence items --
it is never itself evidence to confirm/correct/reject. Deliberately NOT
modeled as a HumanConfirmationRecord / ConfirmationTargetKind target:
mixing the two would blur exactly the distinction this milestone exists to
preserve (reviewing the conflict MARKER vs. reviewing the underlying
observations). This module is a small, separate, parallel journal -- same
append-only, audit-first philosophy as confirmation/schemas.py's
HumanConfirmationRecord, but its own record type, its own action
vocabulary (ContradictionReviewAction, confirmation/enums.py --
ACKNOWLEDGE/DISMISS, never domain/), and its own resolution function.
confirmation/state_machine.py's REJECTED->CONFIRM illegal-transition rule
does not apply here -- there is no equivalent forbidden transition between
ACKNOWLEDGE and DISMISS in either direction, since neither sets anything
governed either way.

Neither action ever:
  - confirms, corrects, or rejects either referenced observation
    (extraction/schemas.py's CandidateContradiction.status stays the
    literal "CANDIDATE" string forever -- this module never writes back
    to the extraction object, exactly like every other confirmation
    mechanism in this codebase never mutates its source);
  - sets ObjectiveOutcome, D1, Operational Priority, Evidence Review, or
    Reliability;
  - creates any new governed CHDM state.

conflict_status() below is the approved "dynamic conflict status" display
helper: it derives the CURRENT status of the two referenced observations
live, from the supplied ActiveEvidenceSet, every time it is called. It
never stores anything and never mutates a ContradictionReviewRecord --
a record's own disposition only ever changes via an explicit new
ACKNOWLEDGE/DISMISS action on the marker itself (resolve_contradiction_
terminal below), never as a side effect of the referenced observations'
state changing.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .enums import ContradictionReviewAction

_contradiction_review_id_counter = itertools.count(1)


@dataclass(frozen=True)
class ContradictionReviewRecord:
    """An append-only journal entry recording one reviewer's disposition
    of one flagged CandidateContradiction marker. `contradiction_id` is the
    contradiction's own system-assigned observation_id (extraction/
    schemas.py's CandidateContradiction.system.observation_id, e.g.
    "CONTRA-######-########") -- contradictions are confirmed structurally
    addressable targets (Milestone 2B/2B.2 closure), this module simply
    never routes them through the HumanConfirmationRecord machinery built
    for observations. Re-review is supported (e.g. ACKNOWLEDGE now,
    DISMISS later once a reviewer concludes the pair wasn't really
    contradictory after all) -- see resolve_contradiction_terminal."""
    record_id: str
    sequence: int
    contradiction_id: str
    action: ContradictionReviewAction
    reviewer: str
    reason: Optional[str] = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.contradiction_id:
            raise ValueError("ContradictionReviewRecord.contradiction_id must not be empty.")
        if not self.reviewer or not self.reviewer.strip():
            raise ValueError(
                "ContradictionReviewRecord.reviewer is required and must be non-empty -- "
                "every contradiction-review decision must be attributable to a named human "
                "reviewer, same authority boundary as HumanConfirmationRecord."
            )
        if self.action == ContradictionReviewAction.DISMISS and (not self.reason or not self.reason.strip()):
            raise ValueError(
                "ContradictionReviewRecord.reason is required and must be non-empty when "
                "action=DISMISS -- the reviewer must state why the AI's flagged pair is not "
                "actually contradictory. ACKNOWLEDGE may leave reason optional."
            )


def create_contradiction_review_record(
    *,
    contradiction_id: str,
    action: ContradictionReviewAction,
    reviewer: str,
    reason: Optional[str] = None,
    recorded_at: Optional[datetime] = None,
) -> ContradictionReviewRecord:
    """The only supported way to construct a ContradictionReviewRecord with
    a correctly system-assigned record_id/sequence -- mirrors confirmation/
    schemas.py's create_confirmation_record pattern exactly, kept as a
    separate counter/function so the two record types' id spaces never
    collide or get confused with each other (a CONF-###### id and a
    CONTRA-REVIEW-###### id are never interchangeable)."""
    n = next(_contradiction_review_id_counter)
    kwargs: dict = dict(
        record_id=f"CONTRA-REVIEW-{n:06d}",
        sequence=n,
        contradiction_id=contradiction_id,
        action=action,
        reviewer=reviewer,
        reason=reason,
    )
    if recorded_at is not None:
        kwargs["recorded_at"] = recorded_at
    return ContradictionReviewRecord(**kwargs)


def group_by_contradiction(
    records: Sequence[ContradictionReviewRecord],
) -> Dict[str, List[ContradictionReviewRecord]]:
    """All contradiction-review records, grouped by contradiction_id.
    Mirrors confirmation.state_machine.group_by_target's contract exactly,
    for the separate ContradictionReviewRecord journal."""
    grouped: Dict[str, List[ContradictionReviewRecord]] = {}
    for record in records:
        grouped.setdefault(record.contradiction_id, []).append(record)
    return grouped


def resolve_contradiction_terminal(
    records: Sequence[ContradictionReviewRecord],
) -> Optional[ContradictionReviewRecord]:
    """The authoritative record for one contradiction marker: the one with
    the highest `sequence` among the records supplied. Returns None for an
    empty sequence (never reviewed). Raises if two records somehow share a
    sequence number (defense-in-depth guard, mirrors confirmation.
    state_machine.resolve_terminal — create_contradiction_review_record's
    monotonic counter should make this impossible in practice). No
    illegal-transition table here (unlike state_machine.py's REJECTED ->
    CONFIRM rule): there is no equivalent forbidden transition between
    ACKNOWLEDGE and DISMISS, since neither ever sets anything governed."""
    if not records:
        return None
    ordered = sorted(records, key=lambda r: r.sequence)
    if len(ordered) >= 2 and ordered[-1].sequence == ordered[-2].sequence:
        raise ValueError(
            f"Ambiguous terminal disposition for contradiction_id={ordered[-1].contradiction_id!r}: "
            f"two records share sequence={ordered[-1].sequence} — contradiction-review records "
            "must have distinct, monotonically-assigned sequence numbers."
        )
    return ordered[-1]


@dataclass(frozen=True)
class ReferencedObservationStatus:
    """The live status of ONE side of a flagged conflict, as of right now.
    `status` is one of "active" (still Current+Unverified/Confirmed and not
    excluded), "rejected" (excluded via ExclusionReason.REJECTED),
    "corrected" (active as a corrected replacement version), or
    "not_found" (the referenced observation_id is not in the current
    active-evidence index at all -- should not occur in practice for a
    validly-resolved CandidateContradiction, but display code must not
    crash if it somehow does)."""
    observation_id: str
    status: str


@dataclass(frozen=True)
class ConflictStatus:
    """The live, derived status of both sides of one flagged conflict.
    Never stored on a ContradictionReviewRecord and never used to mutate
    one -- see this module's docstring. Purely a read model over the
    ActiveEvidenceSet the caller already has from
    state.current_diagnostic.active_evidence."""
    side_a: ReferencedObservationStatus
    side_b: ReferencedObservationStatus

    @property
    def both_active(self) -> bool:
        return self.side_a.status == "active" and self.side_b.status == "active"

    @property
    def both_rejected(self) -> bool:
        return self.side_a.status == "rejected" and self.side_b.status == "rejected"


def _referenced_status(active_evidence, observation_id: Optional[str]) -> ReferencedObservationStatus:
    if not observation_id:
        return ReferencedObservationStatus(observation_id or "", "not_found")
    if active_evidence.is_excluded(observation_id):
        return ReferencedObservationStatus(observation_id, "rejected")
    item = active_evidence.by_observation_id(observation_id)
    if item is None:
        return ReferencedObservationStatus(observation_id, "not_found")
    if item.is_correction:
        return ReferencedObservationStatus(observation_id, "corrected")
    return ReferencedObservationStatus(observation_id, "active")


def conflict_status(active_evidence, contradiction) -> ConflictStatus:
    """Approved 'dynamic conflict status': derives the current disposition
    of both referenced observations live from `active_evidence`
    (confirmation.schemas.ActiveEvidenceSet, already reconstructed by the
    caller for this recompute) -- both active; one rejected; one
    corrected; both rejected; etc. Read-only, side-effect-free."""
    return ConflictStatus(
        side_a=_referenced_status(active_evidence, contradiction.resolved_observation_id_a),
        side_b=_referenced_status(active_evidence, contradiction.resolved_observation_id_b),
    )
