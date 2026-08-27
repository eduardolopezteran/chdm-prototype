"""
Milestone 3A — active-evidence reconstruction.

Turns a Milestone 2 extraction.pipeline.ExtractionResult plus an
append-only journal of HumanConfirmationRecord decisions into the set of
items actually eligible to participate in a Milestone 1 evaluate() run
right now. This is the real, production successor `bridge_to_milestone1.py`
was explicitly written to be a test-only placeholder for (see that
module's docstring: "This is NOT the production human-confirmation
architecture") — implemented as a new module, not a modification of that
test-only file, so the existing Milestone 2B smoke test keeps asserting
exactly what it always has.

Per-target resolution (state_machine.resolve_terminal):
  - No record at all           -> active, evidence_state unchanged
                                   (Current+Unverified/Stale, exactly as
                                   extracted — never reviewed, never
                                   promoted).
  - Terminal action = CONFIRM  -> active, evidence_state promoted to
                                   Current+Confirmed.
  - Terminal action = REJECT   -> excluded entirely (ExclusionRecord).
                                   REJECT is a review disposition, not an
                                   EvidenceState — domain.enums.EvidenceState
                                   is never touched by this decision.
  - Terminal action =
    CANNOT_CONFIRM              -> active, evidence_state UNCHANGED
                                   (uncertainty preserved — reviewed, but
                                   not validated one way or the other).
  - Terminal action = CORRECT  -> active, evidence_state promoted to
                                   Current+Confirmed, using the corrected
                                   representation (original fields
                                   overlaid with corrected_representation).
                                   The original extraction item is always
                                   preserved on `ActiveEvidenceItem.original`
                                   for audit — never discarded.

Confirmation-boundary enforcement (CHDM v0.1 §3.3: only a human may
produce Current+Confirmed evidence) lives HERE: ActiveEvidenceItem is a
new dataclass, deliberately NOT one of the extraction.schemas dataclasses,
specifically because ExtractionSystemFields.__post_init__ structurally
forbids evidence_state=CURRENT_CONFIRMED on every AI-extraction type.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Sequence, Tuple

from domain.enums import ConfirmationAction, EvidenceState
from domain.signals import DIMENSION_QUALIFIERS
from extraction.schemas import CandidateDimensionQualifier, MissingInformationCandidate

from .enums import ConfirmationTargetKind, ExclusionReason
from .schemas import ActiveEvidenceItem, ActiveEvidenceSet, ExclusionRecord, HumanConfirmationRecord
from .state_machine import group_by_target, resolve_terminal

# Milestone 4C: the 2 compound D2/D6 qualifiers Milestone 4B structurally
# bars the model from proposing directly (extraction/enums.py
# DIMENSION_QUALIFIER_TYPE_TO_COMPOSED_QUALIFIER) — a human CORRECT action
# must never be able to manufacture one either. See the CORRECT branch in
# reconstruct_active_evidence() below.
_COMPOUND_DIMENSION_QUALIFIERS = frozenset({"AUTOMATION_RELIABLE_LOW_LOGIN_OK", "CHAMPION_LOST_NO_SUCCESSOR"})


def _representation(obs) -> dict:
    return {f.name: getattr(obs, f.name) for f in dataclasses.fields(obs) if f.name != "system"}


def _target_kind_for(obs) -> ConfirmationTargetKind:
    from extraction.schemas import CandidateEvidenceClassification, CandidateRiskSignal

    if isinstance(obs, MissingInformationCandidate):
        return ConfirmationTargetKind.MISSING_INFORMATION_CANDIDATE
    if isinstance(obs, CandidateRiskSignal):
        return ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL
    if isinstance(obs, CandidateEvidenceClassification):
        return ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION
    if isinstance(obs, CandidateDimensionQualifier):
        return ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER
    return ConfirmationTargetKind.SEMANTIC_OBSERVATION


def _index_extraction_items(extraction_result) -> Dict[str, tuple]:
    """observation_id -> (target_kind, original_dataclass_instance)."""
    index: Dict[str, tuple] = {}
    for obs in extraction_result.accepted:
        index[obs.system.observation_id] = (_target_kind_for(obs), obs)
    for obs in extraction_result.candidate_risk_signals:
        index[obs.system.observation_id] = (ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, obs)
    for obs in extraction_result.candidate_evidence_classifications:
        index[obs.system.observation_id] = (ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION, obs)
    # Milestone 4C: candidate_d2_qualifiers / candidate_d6_qualifiers are
    # two separate ExtractionResult fields (extraction/pipeline.py keeps
    # them split by dimension), but both index into the SAME
    # ConfirmationTargetKind — the object's own `.dimension` field is
    # already the authoritative D2/D6 discriminator, mirroring how
    # CANDIDATE_RISK_SIGNAL covers all 3 mechanisms under one target kind.
    for obs in extraction_result.candidate_d2_qualifiers:
        index[obs.system.observation_id] = (ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, obs)
    for obs in extraction_result.candidate_d6_qualifiers:
        index[obs.system.observation_id] = (ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER, obs)
    return index


def _validate_dimension_qualifier_correction(obs, corrected_representation: dict) -> None:
    """Milestone 4C governing requirement: a CORRECT action on a
    CandidateDimensionQualifier must never be able to manufacture
    AUTOMATION_RELIABLE_LOW_LOGIN_OK or CHAMPION_LOST_NO_SUCCESSOR unless
    the ORIGINAL candidate already had that exact compound qualifier —
    otherwise a reviewer could bypass Milestone 4B's frozen, three-round-
    calibrated same-observation/EXPLICIT-basis composition authority
    control by simply typing the compound name into a simple candidate's
    Correct form. `dimension` and `basis` are structural/provenance fields
    and are not correctable at all (enforced by the caller never offering
    them — see ui/item_card.py's _NON_EDITABLE_FIELDS — but re-checked
    here too, defense in depth, since this function is the actual
    authority boundary, never the UI alone). `qualifier`, if present in
    corrected_representation, must remain in the governed vocabulary for
    the candidate's OWN (uncorrectable) dimension."""
    if "dimension" in corrected_representation and corrected_representation["dimension"] != obs.dimension:
        raise ValueError(
            "CandidateDimensionQualifier correction may not change `dimension` — "
            "a D2 candidate can never be corrected into a D6 claim (or vice versa); "
            "that would require a different supporting_observation_ref type entirely, "
            "not a correction of this candidate."
        )
    if "basis" in corrected_representation and corrected_representation["basis"] != obs.basis:
        raise ValueError(
            "CandidateDimensionQualifier correction may not change `basis` — it is "
            "the model's self-reported provenance and has already done its only job "
            "(gating Milestone 4B compound-composition eligibility, which already "
            "happened deterministically before this candidate was ever reviewed)."
        )
    corrected_qualifier = corrected_representation.get("qualifier")
    if corrected_qualifier is None:
        return
    allowed = DIMENSION_QUALIFIERS.get(obs.dimension, ())
    if corrected_qualifier not in allowed:
        raise ValueError(
            f"CandidateDimensionQualifier correction qualifier={corrected_qualifier!r} is not "
            f"in the governed vocabulary for dimension={obs.dimension!r}: {allowed}."
        )
    if corrected_qualifier in _COMPOUND_DIMENSION_QUALIFIERS and obs.qualifier != corrected_qualifier:
        raise ValueError(
            f"CandidateDimensionQualifier correction may not set qualifier="
            f"{corrected_qualifier!r} — this is one of Milestone 4B's 2 compound "
            "qualifiers, structurally produced ONLY by deterministic atomic-predicate "
            "composition (same-observation, EXPLICIT-basis). A human correction may "
            "never manufacture it on a candidate that wasn't already that exact "
            "compound qualifier — that would bypass the frozen composition authority "
            "control entirely."
        )


def _representative_evidence_id(obs) -> str:
    source_evidence_id = getattr(obs, "source_evidence_id", None)
    if source_evidence_id is not None:
        return source_evidence_id
    # MissingInformationCandidate has no single source_evidence_id (it is
    # scoped to reviewed_evidence_ids, plural) — use the first reviewed id
    # as the representative evidence_id for signal-building purposes; the
    # full reviewed scope remains available via
    # representation["reviewed_evidence_ids"].
    reviewed = getattr(obs, "reviewed_evidence_ids", ())
    return reviewed[0] if reviewed else "UNKNOWN_SOURCE"


def reconstruct_active_evidence(
    extraction_result,
    confirmation_records: Sequence[HumanConfirmationRecord],
) -> ActiveEvidenceSet:
    index = _index_extraction_items(extraction_result)
    grouped = group_by_target(confirmation_records)

    unknown_targets = set(grouped) - set(index)
    if unknown_targets:
        raise ValueError(
            "HumanConfirmationRecord(s) reference target_observation_id(s) not present "
            f"in the supplied ExtractionResult: {sorted(unknown_targets)!r}."
        )

    items: list = []
    excluded: list = []

    for observation_id, (kind, obs) in index.items():
        terminal = resolve_terminal(grouped.get(observation_id, ()))
        source_evidence_id = _representative_evidence_id(obs)

        if terminal is None:
            items.append(ActiveEvidenceItem(
                target_kind=kind, observation_id=observation_id,
                source_evidence_id=source_evidence_id,
                evidence_state=obs.system.evidence_state,
                is_correction=False, original=obs,
                representation=_representation(obs), confirmation_id=None,
            ))
            continue

        if terminal.action == ConfirmationAction.REJECT:
            excluded.append(ExclusionRecord(
                target_observation_id=observation_id, target_kind=kind,
                reason=ExclusionReason.REJECTED, confirmation_id=terminal.confirmation_id,
                detail=terminal.reason,
            ))
            continue

        if terminal.action == ConfirmationAction.CANNOT_CONFIRM:
            items.append(ActiveEvidenceItem(
                target_kind=kind, observation_id=observation_id,
                source_evidence_id=source_evidence_id,
                evidence_state=obs.system.evidence_state,  # uncertainty preserved, NOT promoted
                is_correction=False, original=obs,
                representation=_representation(obs), confirmation_id=terminal.confirmation_id,
            ))
            continue

        if terminal.action == ConfirmationAction.CONFIRM:
            items.append(ActiveEvidenceItem(
                target_kind=kind, observation_id=observation_id,
                source_evidence_id=source_evidence_id,
                evidence_state=EvidenceState.CURRENT_CONFIRMED,
                is_correction=False, original=obs,
                representation=_representation(obs), confirmation_id=terminal.confirmation_id,
            ))
            continue

        if terminal.action == ConfirmationAction.CORRECT:
            if kind == ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER:
                # Milestone 4C authority boundary — see
                # _validate_dimension_qualifier_correction's own docstring.
                # This is the actual enforcement point; the UI's dropdown
                # restriction (ui/labels.py) is defense in depth only.
                _validate_dimension_qualifier_correction(obs, dict(terminal.corrected_representation))
            representation = {**_representation(obs), **dict(terminal.corrected_representation)}
            items.append(ActiveEvidenceItem(
                target_kind=kind, observation_id=observation_id,
                source_evidence_id=representation.get("source_evidence_id", source_evidence_id),
                evidence_state=EvidenceState.CURRENT_CONFIRMED,
                is_correction=True, original=obs,
                representation=representation, confirmation_id=terminal.confirmation_id,
            ))
            continue

        raise AssertionError(f"Unhandled ConfirmationAction: {terminal.action}")  # pragma: no cover

    return ActiveEvidenceSet(items=tuple(items), excluded=tuple(excluded))
