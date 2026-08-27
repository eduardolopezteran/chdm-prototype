"""
Milestone 3A — Human Confirmation Backend: schemas.

A HumanConfirmationRecord is an append-only journal entry — never mutated,
never deleted — mirroring this codebase's existing audit-first patterns
(extraction.pipeline.ExtractionValidationFailure / DedupAuditRecord).
Re-review is explicitly supported: a target may accumulate more than one
record over time (e.g. CANNOT_CONFIRM now, CONFIRM later once more
evidence arrives). The record with the highest `sequence` is authoritative
for that target — see state_machine.resolve_terminal().

Confirmation-record amendment (reviewer rationale, applied on top of the
original Milestone 3A checkpoint): REJECT remains a review disposition,
never an EvidenceState — domain.enums.EvidenceState is untouched by this
package. REJECT and CANNOT_CONFIRM each require a non-empty reviewer
reason/comment (auditability only — this does not change evidence-state
semantics or any deterministic CHDM logic). CONFIRM may leave reason
optional. CORRECT may leave reason optional, but corrected_representation
itself is mandatory.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from domain.enums import ConfirmationAction, EvidenceState

from .enums import ConfirmationTargetKind, ExclusionReason, ObjectiveResolutionStatus

# Reserved reviewer identifiers that can never attribute a confirmation
# decision — a structural guard for "AI cannot self-confirm" (Milestone 3A
# spec). Real reviewer identity/authn is out of scope for this
# backend-only milestone; this only blocks the obviously-wrong cases.
_RESERVED_REVIEWER_IDENTIFIERS = frozenset({
    "ai", "system", "model", "extraction_pipeline", "bot", "automation", "pipeline",
})

_confirmation_id_counter = itertools.count(1)


@dataclass(frozen=True)
class HumanConfirmationRecord:
    confirmation_id: str
    sequence: int
    target_kind: ConfirmationTargetKind
    target_observation_id: str
    action: ConfirmationAction
    reviewer: str
    reason: Optional[str] = None
    corrected_representation: Optional[Mapping[str, Any]] = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.target_observation_id:
            raise ValueError("HumanConfirmationRecord.target_observation_id must not be empty.")
        if not self.reviewer or not self.reviewer.strip():
            raise ValueError(
                "HumanConfirmationRecord.reviewer is required and must be non-empty — "
                "every confirmation decision must be attributable to a named human "
                "reviewer (AI cannot self-confirm, Milestone 3A authority boundary)."
            )
        if self.reviewer.strip().lower() in _RESERVED_REVIEWER_IDENTIFIERS:
            raise ValueError(
                f"HumanConfirmationRecord.reviewer={self.reviewer!r} is a reserved "
                "non-human identifier — AI cannot self-confirm (Milestone 3A authority "
                "boundary)."
            )
        # Reviewer-rationale amendment: REJECT / CANNOT_CONFIRM require a
        # non-empty reason. CONFIRM leaves it optional. CORRECT leaves it
        # optional too, but corrected_representation is separately
        # mandatory below.
        if self.action in (ConfirmationAction.REJECT, ConfirmationAction.CANNOT_CONFIRM):
            if not self.reason or not self.reason.strip():
                raise ValueError(
                    f"HumanConfirmationRecord.reason is required and must be non-empty "
                    f"when action={self.action.value} (auditability requirement — the "
                    "reviewer must state why the item was rejected, or why it could not "
                    "be confirmed)."
                )
        if self.action == ConfirmationAction.CORRECT:
            if not self.corrected_representation:
                raise ValueError(
                    "HumanConfirmationRecord.corrected_representation is required and "
                    "must be non-empty when action=CORRECT."
                )


def create_confirmation_record(
    *,
    target_kind: ConfirmationTargetKind,
    target_observation_id: str,
    action: ConfirmationAction,
    reviewer: str,
    reason: Optional[str] = None,
    corrected_representation: Optional[Mapping[str, Any]] = None,
    recorded_at: Optional[datetime] = None,
) -> HumanConfirmationRecord:
    """The only supported way to construct a HumanConfirmationRecord with a
    correctly system-assigned confirmation_id/sequence — mirrors
    extraction.pipeline._new_id / engine.trace.build_trace's own
    monotonic-counter pattern. Callers should not hand-author
    confirmation_id/sequence themselves (tests that need to control
    ordering explicitly may still construct HumanConfirmationRecord
    directly)."""
    n = next(_confirmation_id_counter)
    kwargs: dict = dict(
        confirmation_id=f"CONF-{n:06d}",
        sequence=n,
        target_kind=target_kind,
        target_observation_id=target_observation_id,
        action=action,
        reviewer=reviewer,
        reason=reason,
        corrected_representation=corrected_representation,
    )
    if recorded_at is not None:
        kwargs["recorded_at"] = recorded_at
    return HumanConfirmationRecord(**kwargs)


@dataclass(frozen=True)
class ActiveEvidenceItem:
    """A single item admitted into the active-evidence set used to build
    Milestone 1 signals. Deliberately NOT one of the extraction.schemas
    dataclasses: ExtractionSystemFields.__post_init__ structurally forbids
    evidence_state=CURRENT_CONFIRMED on every AI-extraction type, and this
    package must never route around that guardrail. Promotion to
    Current+Confirmed only ever happens by constructing an
    ActiveEvidenceItem here (active_evidence.reconstruct_active_evidence),
    and only in direct response to a HumanConfirmationRecord with
    action=CONFIRM or action=CORRECT."""
    target_kind: ConfirmationTargetKind
    observation_id: str
    source_evidence_id: str
    evidence_state: EvidenceState
    is_correction: bool
    original: Any
    representation: Mapping[str, Any]
    confirmation_id: Optional[str] = None


@dataclass(frozen=True)
class ExclusionRecord:
    target_observation_id: str
    target_kind: ConfirmationTargetKind
    reason: ExclusionReason
    confirmation_id: Optional[str]
    detail: Optional[str] = None


@dataclass(frozen=True)
class ActiveEvidenceSet:
    items: tuple
    excluded: tuple

    def by_observation_id(self, observation_id: str) -> Optional[ActiveEvidenceItem]:
        for item in self.items:
            if item.observation_id == observation_id:
                return item
        return None

    def is_excluded(self, observation_id: str) -> bool:
        return any(e.target_observation_id == observation_id for e in self.excluded)


@dataclass(frozen=True)
class ObjectiveResolution:
    """Milestone 3D — confirmation-layer-only integration/audit metadata
    describing how (or whether) the Objective handed to engine.evaluate()
    was established. Deliberately NOT a new governed CHDM construct: it
    carries no rule of its own, is never consumed by anything in engine/,
    and does not change domain/objective.py's Objective in any way beyond
    supplying its ordinary constructor arguments. See
    confirmation.recompute._resolve_objective().

    `text` / `source_evidence_ref` here mirror whatever the resolved
    Objective ended up with (empty/None when status is NOT_ESTABLISHED or
    CONFLICTING, since the engine then receives an unresolved/unknown
    objective under the existing, unmodified is_known=False rule).
    `contributing_observation_ids` lists every confirmed ObjectiveCandidate
    observation_id that agreed with the established text (for audit, even
    when more than one made the identical statement); `conflicting_
    observation_ids` lists every confirmed ObjectiveCandidate observation_id
    across ALL groups when status is CONFLICTING (for audit -- nothing is
    discarded, nothing is picked)."""
    status: ObjectiveResolutionStatus
    objective_id: str
    text: Optional[str] = None
    source_evidence_ref: Optional[str] = None
    contributing_observation_ids: tuple = ()
    conflicting_observation_ids: tuple = ()
    detail: Optional[str] = None


@dataclass(frozen=True)
class RecomputeDiagnostic:
    """Everything a caller needs to audit one recompute() run: the
    engine.evaluate.EvaluationResult itself, the active-evidence set it was
    built from, and the exact signal tuples handed to evaluate().

    Milestone 3D: objective_resolution records how the Objective actually
    handed to evaluate() was established -- see ObjectiveResolution above.
    Required (no default) so every caller sees it explicitly rather than
    silently defaulting to some assumed status."""
    result: Any
    active_evidence: ActiveEvidenceSet
    value_signals: tuple
    dimension_signals: tuple
    risk_claims: tuple
    objective_resolution: ObjectiveResolution


@dataclass(frozen=True)
class ConsequentialityReport:
    """Whether ONE target's current review disposition changes the
    governed outcome, compared to that target never having been reviewed
    at all. outcome_with_review / outcome_without_review are hashable
    outcome signatures (see consequentiality._outcome_signature), kept on
    the report for audit rather than the full EvaluationResult objects."""
    target_observation_id: str
    is_consequential: bool
    outcome_with_review: tuple
    outcome_without_review: tuple
