"""
Milestone 3B — review queue construction.

Pure function: given an ExtractionResult and the confirmation journal so
far, decide which items still need review and in what order. Does NOT
call compute_consequentiality() itself -- ordering here is driven entirely
by a caller-supplied lookup (ui/actions.get_cached_consequentiality),
which is how the approved lazy-evaluation design stays out of this
module. That keeps build_review_queue() unit-testable without Streamlit,
a live backend evaluation, or any I/O.

"Evidence Requiring Review" (the diagnostic step this queue drives) means:
never reviewed at all, or reviewed but left as CANNOT_CONFIRM (uncertainty
deliberately preserved, not resolved). CONFIRM / CORRECT / REJECT are all
terminal review outcomes -- those items are considered reviewed and drop
out of this primary queue (they remain reachable via ui/audit_view.py,
and a Rejected item can still receive a later CORRECT).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from domain.enums import ConfirmationAction

from confirmation.enums import ConfirmationTargetKind
from confirmation.state_machine import group_by_target, resolve_terminal
from extraction.schemas import MissingInformationCandidate

_NEEDS_REVIEW_TERMINAL_ACTIONS = (None, ConfirmationAction.CANNOT_CONFIRM)


def grounded_text(obs) -> str:
    """The human-readable grounded quote for any reviewable extraction
    item. MissingInformationCandidate is the one type with no source_span
    at all (Checkpoint 2A refinement 4: absence has nothing to quote) --
    it is grounded to a reviewed evidence SCOPE instead
    (reviewed_evidence_ids), not a specific span. Every other reviewable
    type has source_span.text. Shared here (rather than duplicated in
    ui/item_card.py and ui/audit_view.py) after the Milestone 3B manual UX
    smoke pass hit this exact AttributeError in both places independently."""
    if hasattr(obs, "source_span"):
        return obs.source_span.text
    return f'"{obs.missing_item}" not found in reviewed evidence: {", ".join(obs.reviewed_evidence_ids)}'


@dataclass(frozen=True)
class QueueEntry:
    target_kind: ConfirmationTargetKind
    target_observation_id: str
    label: str
    status: str  # "unreviewed" | "cannot_confirm"
    consequentiality: Optional[bool]  # True/False if known and current, None if not yet evaluated


def _label_for(obs, target_kind: ConfirmationTargetKind) -> str:
    if target_kind == ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL:
        return f"Candidate risk: {obs.mechanism} ({obs.proposed_severity_tier})"
    if target_kind == ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION:
        return f"Candidate value evidence: {obs.proposed_basis} -> {obs.supports}"
    if target_kind == ConfirmationTargetKind.MISSING_INFORMATION_CANDIDATE:
        return f"Missing information: {obs.missing_item}"
    if target_kind == ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER:
        return f"Candidate {obs.dimension.value} qualifier: {obs.qualifier}"
    text = obs.source_span.text
    return text if len(text) <= 80 else text[:77] + "..."


def _all_extraction_items(extraction_result):
    for obs in extraction_result.accepted:
        kind = (
            ConfirmationTargetKind.MISSING_INFORMATION_CANDIDATE
            if isinstance(obs, MissingInformationCandidate)
            else ConfirmationTargetKind.SEMANTIC_OBSERVATION
        )
        yield obs.system.observation_id, kind, obs
    for obs in extraction_result.candidate_risk_signals:
        yield obs.system.observation_id, ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, obs
    for obs in extraction_result.candidate_evidence_classifications:
        yield obs.system.observation_id, ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, obs
    # Milestone 4C: candidate_d2_qualifiers / candidate_d6_qualifiers both
    # index into the SAME target kind (see confirmation/active_evidence.py's
    # _index_extraction_items for the identical pattern and rationale).
    for obs in extraction_result.candidate_d2_qualifiers:
        yield obs.system.observation_id, ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, obs
    for obs in extraction_result.candidate_d6_qualifiers:
        yield obs.system.observation_id, ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, obs


def atomic_predicate_evidence_for(extraction_result, obs) -> tuple:
    """Milestone 4C — shared lookup so ui/item_card.py (active review) and
    ui/audit_view.py (historical review) never duplicate this logic (same
    reason grounded_text() above is shared, per its own docstring history).
    Returns every AtomicPredicateEvidence Milestone 4B recorded for this
    CandidateDimensionQualifier's own (resolved_observation_id, dimension)
    key -- empty for a simple (non-compound) candidate, or for any other
    target kind. This is a pure, read-only lookup against
    ExtractionResult.dimension_qualifier_predicate_evidence: it never
    copies, mutates, or re-derives that provenance -- compound provenance
    stays exactly where Milestone 4B put it, resolved fresh every time by
    this function rather than flattened onto the candidate or the signal."""
    if not hasattr(obs, "resolved_observation_id") or not hasattr(obs, "dimension"):
        return ()
    return tuple(
        p for p in extraction_result.dimension_qualifier_predicate_evidence
        if p.resolved_observation_id == obs.resolved_observation_id and p.dimension == obs.dimension
    )


def find_item(extraction_result, target_kind: ConfirmationTargetKind, target_observation_id: str):
    """Looks up the original extraction dataclass instance for one target
    -- used by ui/item_card.py and ui/audit_view.py to render source
    traceability. Returns None if not found (caller's responsibility to
    handle; this function does not raise, unlike active_evidence's
    stricter unknown-target guard, since this is read-only display code)."""
    for observation_id, kind, obs in _all_extraction_items(extraction_result):
        if observation_id == target_observation_id and kind == target_kind:
            return obs
    return None


def find_any_item(extraction_result, observation_id: str):
    """Milestone 3E: resolves an observation_id to its (target_kind, obs)
    pair WITHOUT requiring the caller to already know which kind it is --
    used to display a CandidateContradiction's two referenced observations.
    A contradiction can reference any of the 7 span-grounded positive types
    or a MissingInformationCandidate (extraction/schemas.py:
    CandidateContradiction can never reference a CandidateRiskSignal or a
    CandidateEvidenceClassification), so the caller cannot simply guess
    ConfirmationTargetKind.SEMANTIC_OBSERVATION the way find_item() above
    requires it to. Returns None if not found (read-only display code,
    same not-found contract as find_item)."""
    for oid, kind, obs in _all_extraction_items(extraction_result):
        if oid == observation_id:
            return kind, obs
    return None


def build_review_queue(
    extraction_result,
    confirmation_records,
    consequentiality_lookup: Callable[[str], Optional[bool]],
) -> List[QueueEntry]:
    """Ordering per the approved checkpoint: known Consequential first,
    then not-yet-evaluated, then known Not consequential. Unevaluated is
    NEVER sorted as though it were non-consequential."""
    grouped = group_by_target(confirmation_records)
    entries: List[QueueEntry] = []
    for observation_id, kind, obs in _all_extraction_items(extraction_result):
        terminal = resolve_terminal(grouped.get(observation_id, ()))
        terminal_action = terminal.action if terminal is not None else None
        if terminal_action not in _NEEDS_REVIEW_TERMINAL_ACTIONS:
            continue
        status = "cannot_confirm" if terminal_action == ConfirmationAction.CANNOT_CONFIRM else "unreviewed"
        entries.append(QueueEntry(
            target_kind=kind, target_observation_id=observation_id,
            label=_label_for(obs, kind), status=status,
            consequentiality=consequentiality_lookup(observation_id),
        ))

    tier = {True: 0, None: 1, False: 2}
    indexed = list(enumerate(entries))  # preserves original extraction order as a stable tiebreaker
    indexed.sort(key=lambda pair: (tier[pair[1].consequentiality], pair[0]))
    return [entry for _, entry in indexed]
