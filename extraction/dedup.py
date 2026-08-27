"""
Build Milestone 2 — deterministic, audit-preserving deduplication
(spec §12).

Covers the 7 span-grounded observation types PLUS MissingInformationCandidate,
PLUS (Milestone 2C) the 2 span-grounded candidate-classification types,
CandidateRiskSignal and CandidateEvidenceClassification, PLUS (Milestone
4B) CandidateDimensionQualifier (not CandidateContradiction —
contradictions are handled separately in pipeline.py and never enter
this function's `observations` input). No second LLM call or LLM
judgment is used to decide canonical truth — matching criteria are
purely structural. For the span-grounded types: same dataclass type,
same source_evidence_id, materially overlapping source_span, and a
normalized-equivalent primary semantic value (for the 2 Milestone 2C
types: (mechanism, proposed_severity_tier) and (proposed_basis,
supports) respectively; for Milestone 4B's CandidateDimensionQualifier:
(dimension, qualifier) — see `_primary_value` below). If any of those
don't clearly match, BOTH observations are retained (spec §12: "If
ambiguity exists, retain both rather than silently merge materially
different observations.").

Milestone 4B note: CandidateDimensionQualifier uses INHERITED grounding
(source_evidence_id/source_span copied verbatim from its supporting
observation — see extraction.schemas.CandidateDimensionQualifier's
docstring), so this generic span-grounded matching logic requires no
special-casing to handle it correctly: two duplicate qualifier proposals
for the SAME supporting observation naturally inherit identical
source_evidence_id/source_span, so they collapse exactly like any other
span-grounded duplicate would.

MissingInformationCandidate is not span-grounded (schemas.py: it has no
source_evidence_id/source_span, only `missing_item` + `reviewed_evidence_ids`
— see the Milestone 2B.2 closure note on `_is_duplicate` below for its
dedicated matching rule, added after a live run crashed comparing two
MissingInformationCandidate instances against the span-grounded-only
logic that used to run unconditionally here).

Nothing is ever deleted without a trace: `deduplicate()` returns the
audit records for every collapsed duplicate alongside the kept list.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    AdoptionObservation, CandidateDimensionQualifier, CandidateEvidenceClassification,
    CandidateRiskSignal, CommercialObservation, ExperienceObservation,
    MissingInformationCandidate, ObjectiveCandidate, ServiceObservation,
    StakeholderObservation, StrategicObservation,
)

_SPAN_OVERLAP_THRESHOLD = 0.5  # fraction of the smaller span's length that must overlap


@dataclass(frozen=True)
class DedupAuditRecord:
    duplicate_observation_id: str
    canonical_observation_id: str
    observation_type: str
    reason: str


def _norm(s) -> str:
    return (s or "").strip().lower()


def _primary_value(obs) -> tuple:
    """The normalized tuple of fields that determines whether two
    observations assert the "same source fact" (spec §12's stable-field
    list). Deliberately narrow and type-specific rather than comparing
    every field, so cosmetic differences (e.g. a slightly different
    optional `role` string) don't block a real duplicate from being
    caught, while a materially different claim never matches."""
    if isinstance(obs, ObjectiveCandidate):
        return (_norm(obs.objective_text),)
    if isinstance(obs, StakeholderObservation):
        return (_norm(obs.person_identifier), _norm(obs.continuity_event))
    if isinstance(obs, AdoptionObservation):
        return (_norm(obs.workflow_or_use_case), _norm(obs.observed_behavior))
    if isinstance(obs, ServiceObservation):
        return (_norm(obs.incident_or_condition),)
    if isinstance(obs, CommercialObservation):
        return (_norm(obs.event_type), _norm(obs.description))
    if isinstance(obs, ExperienceObservation):
        return (_norm(obs.statement),)
    if isinstance(obs, StrategicObservation):
        return (_norm(obs.event),)
    # Milestone 2C: the 2 candidate-classification types. Both remain
    # span-grounded (source_evidence_id + source_span), so the generic
    # `_is_duplicate` fallback branch below already handles them once
    # `_primary_value` returns a real tuple here instead of falling
    # through to the identity default.
    if isinstance(obs, CandidateRiskSignal):
        return (_norm(obs.mechanism), _norm(obs.proposed_severity_tier))
    if isinstance(obs, CandidateEvidenceClassification):
        return (_norm(obs.proposed_basis), _norm(obs.supports))
    # Milestone 4B: CandidateDimensionQualifier. Both D2 and D6 channels
    # share this ONE dataclass (approved architecture), so `dimension` is
    # included here precisely so a D2 proposal and a D6 proposal are
    # never treated as candidate duplicates of each other even if (by
    # some future coincidence) they shared a source_evidence_id/span --
    # in practice this cannot currently happen (D2 only ever inherits
    # grounding from an AdoptionObservation, D6 only ever from a
    # StakeholderObservation), but this keeps the matching criterion
    # explicit and correct rather than relying on that invariant holding
    # silently forever.
    if isinstance(obs, CandidateDimensionQualifier):
        return (_norm(obs.dimension.value), _norm(obs.qualifier))
    # Any type not handled above (there is currently none among the 10
    # span-grounded-or-classification accepted observation types —
    # MissingInformationCandidate has its own dedicated branch in
    # `_is_duplicate`, never reaching here) falls back to per-instance
    # identity, i.e. never matches anything. A defensive default, not a
    # documented behavior for any real type.
    return (id(obs),)


def _span_overlap_ratio(a, b) -> float:
    lo = max(a.start_char, b.start_char)
    hi = min(a.end_char, b.end_char)
    if hi <= lo:
        return 0.0
    overlap = hi - lo
    shorter = min(a.end_char - a.start_char, b.end_char - b.start_char)
    return overlap / shorter if shorter else 0.0


def _is_duplicate(a, b) -> bool:
    if type(a) is not type(b):
        return False

    # Milestone 2B.2 closure fix: a live run crashed here
    # (`AttributeError: 'MissingInformationCandidate' object has no
    # attribute 'source_evidence_id'`) because the code below used to run
    # unconditionally for every same-typed pair, including two
    # MissingInformationCandidate instances -- which are not
    # span-grounded and never had a source_evidence_id/source_span to
    # compare (schemas.py). MissingInformationCandidate gets its own
    # matching rule instead of a fake source_evidence_id: same normalized
    # `missing_item`, AND the exact same `reviewed_evidence_ids` scope.
    # Scope is deliberately exact-set equality, not overlap or subset --
    # "not found in evidence reviewed" is only the same claim if it was
    # reviewed against the SAME evidence; two claims naming the same
    # missing item but scoped to different reviewed evidence are
    # materially different claims and must both be retained (spec §12's
    # own "if ambiguity exists, retain both" policy, applied here to
    # scope rather than span).
    if isinstance(a, MissingInformationCandidate):
        return (
            _norm(a.missing_item) == _norm(b.missing_item)
            and frozenset(a.reviewed_evidence_ids) == frozenset(b.reviewed_evidence_ids)
        )

    return (
        a.source_evidence_id == b.source_evidence_id
        and _span_overlap_ratio(a.source_span, b.source_span) >= _SPAN_OVERLAP_THRESHOLD
        and _primary_value(a) == _primary_value(b)
    )


def deduplicate(observations: tuple) -> tuple[tuple, dict[str, str], tuple[DedupAuditRecord, ...]]:
    """`observations` must already carry populated `system.observation_id`
    values (i.e. this runs AFTER system-metadata attachment, per
    pipeline.py's ordering). Returns (kept, canonical_id_map, audit) where
    canonical_id_map maps EVERY input observation_id (kept or collapsed)
    to the observation_id that survives — used by pipeline.py to
    re-point any CandidateContradiction reference that targeted a
    now-collapsed duplicate."""
    kept: list = []
    canonical_map: dict[str, str] = {}
    audit: list[DedupAuditRecord] = []

    for obs in observations:
        oid = obs.system.observation_id
        match = next((k for k in kept if _is_duplicate(k, obs)), None)
        if match is not None:
            canonical_map[oid] = match.system.observation_id
            reason = (
                "same missing_item within the same reviewed evidence scope"
                if isinstance(obs, MissingInformationCandidate)
                else "same evidence item, overlapping source span, normalized-equivalent value"
            )
            audit.append(DedupAuditRecord(
                duplicate_observation_id=oid,
                canonical_observation_id=match.system.observation_id,
                observation_type=type(obs).__name__,
                reason=reason,
            ))
        else:
            canonical_map[oid] = oid
            kept.append(obs)

    return tuple(kept), canonical_map, tuple(audit)
