"""
Milestone 3A — recompute.

Rebuilds the three Milestone 1 signal tuples (ValueEvidenceSignal,
DimensionQualifierSignal, RiskSeverityClaim) from a confirmation-resolved
ActiveEvidenceSet, then calls the existing, unmodified
engine.evaluate.evaluate() — this module never re-implements any
deterministic CHDM rule; it only supplies confirmation-governed inputs to
the rule engine that already exists. Recomputation therefore automatically
picks up every downstream recalculation evaluate() already performs:
dimension states, risk records (potential/activated severity), DMEGs,
reliability, operational priority, and evidence review.

Mechanism/tier and basis/supports for risk and value signals are taken
directly from the AI's own CandidateRiskSignal / CandidateEvidenceClassification
proposal once confirmed or corrected — that IS the governed classification
once a human ratifies it, never re-derived here.

Dimension/qualifier signals, historically (through Milestone 4B): the one
axis with no AI candidate-classification type at all (M3-OD-01). Milestone
4B resolved that for D2/D6 with CandidateDimensionQualifier; Milestone 4C
closes the loop by feeding CONFIRMED/CORRECTed candidates of that type
into `_build_dimension_signals_from_confirmed_candidates()` below, the
real successor to the manual override path for this axis, structurally
mirroring `_build_value_signals`/`_build_risk_claims`. `dimension_qualifier_
overrides` / `_build_dimension_signals()` is KEPT (approved PMO decision,
Milestone 4C checkpoint) rather than removed — it remains useful for
hand-authored fixtures, Scenario Lab scenarios, and any future dimension
that gets no AI candidate type of its own — and its output is simply
concatenated with the confirmed-candidate signals in `recompute()` below.
The two sources can never conflict destructively: engine/dimension_engine.py's
existing, unmodified precedence rule (CONCERNING > MIXED > SUPPORTED >
INSUFFICIENT_EVIDENCE, applied to the UNION of every confirmed qualifier
for a dimension) already handles multiple signals for the same dimension
deterministically, exactly as it does for multiple confirmed candidates
alone.

Milestone 3D — Objective Context Integration. Adds one more derived
input alongside the three signal-builders above: `_resolve_objective()`
turns confirmed/corrected ObjectiveCandidate evidence into the actual
`domain.objective.Objective` handed to `engine.evaluate()`, closing the
gap where a confirmed ObjectiveCandidate previously had no effect on
anything (it was promoted to Current+Confirmed in ActiveEvidenceSet like
any other semantic observation, but nothing downstream ever read it).
This is a confirmation-layer bridge only:
  - No `domain/` or `engine/` file is touched. `objective_engine.py`'s
    deterministic ACHIEVED/PROGRESSING/NOT_ACHIEVED/DISPUTED/UNKNOWN
    precedence and `dimension_engine.evaluate_d1()`'s DISPUTED-takes-
    precedence rule are exactly as they were — this module only changes
    what `Objective` they are handed, never how they decide.
  - No new governed ObjectiveOutcome state is introduced.
    `ObjectiveResolutionStatus` (confirmation/enums.py) and
    `ObjectiveResolution` (confirmation/schemas.py) are integration/audit
    metadata only, for the UI/caller to explain WHY Objective Outcome is
    Unknown (no confirmed objective evidence yet, vs. confirmed evidence
    that conflicts) — the engine itself still just sees an
    is_known=False Objective either way and renders UNKNOWN under the
    existing, unmodified rule.
  - Only `ObjectiveCandidate.objective_text` is promoted into
    `Objective.text` (approved checkpoint decision A). `measure`/
    `target`/`timeframe`/`stated_outcome` remain extraction/provenance
    detail only, visible in ui/item_card.py's existing technical-detail
    view, never promoted into a governed field — not because they are
    permanently out of scope, but because doing so was not authorized
    for this milestone (candidate for a future explicit methodology
    amendment, not decided here).
  - An already-known objective (the caller-supplied `account.objective`
    has `is_known=True`) is never overridden by confirmed extraction
    evidence — this bridge only ever ESTABLISHES an objective identity
    that isn't already known; it never second-guesses one that is. This
    is also what keeps every pre-3D test that hand-declares a known
    Objective (e.g. tests/test_confirmation_recompute_end_to_end.py)
    passing unchanged: those never confirm the ObjectiveCandidate in
    their fixtures, and now never need to.
  - No semantic similarity, embeddings, fuzzy matching, or LLM
    involvement in reconciling objective identity (approved checkpoint,
    objective-equivalence rule) — only deterministic normalization
    (whitespace/case), see `_normalize_objective_text()`. Two or more
    confirmed ObjectiveCandidate statements are the SAME objective
    identity only if their normalized text matches exactly; otherwise
    objective identity is CONFLICTING and is left unresolved — never
    auto-picked, auto-concatenated, or averaged. Reconciliation remains
    entirely a human Reject/Correct action on the conflicting item(s),
    exactly like every other confirmation decision in this codebase.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from domain.enums import DimensionCode, RiskMechanismCode, ValueEvidenceBasis
from domain.objective import Objective
from domain.signals import DimensionQualifierSignal, RiskSeverityClaim, ValueEvidenceSignal
from engine.evaluate import evaluate
from engine.evidence_engine import is_current_confirmed

from extraction.schemas import ObjectiveCandidate

from .enums import ConfirmationTargetKind, ObjectiveResolutionStatus
from .schemas import ActiveEvidenceItem, ActiveEvidenceSet, ObjectiveResolution, RecomputeDiagnostic

# observation_id -> (DimensionCode, qualifier_string)
DimensionQualifierOverrides = Mapping[str, Tuple[DimensionCode, str]]


def _build_value_signals(active: ActiveEvidenceSet) -> tuple:
    signals = []
    for item in active.items:
        if item.target_kind != ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION:
            continue
        signals.append(ValueEvidenceSignal(
            signal_id=item.observation_id,
            evidence_id=item.source_evidence_id,
            evidence_state=item.evidence_state,
            basis=ValueEvidenceBasis(item.representation["proposed_basis"]),
            supports=item.representation["supports"],
        ))
    return tuple(signals)


def _build_risk_claims(active: ActiveEvidenceSet) -> tuple:
    claims = []
    for item in active.items:
        if item.target_kind != ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL:
            continue
        claims.append(RiskSeverityClaim(
            signal_id=item.observation_id,
            mechanism=RiskMechanismCode(item.representation["mechanism"]),
            evidence_id=item.source_evidence_id,
            evidence_state=item.evidence_state,
            tier=item.representation["proposed_severity_tier"],
        ))
    return tuple(claims)


def _build_dimension_signals_from_confirmed_candidates(active: ActiveEvidenceSet) -> tuple:
    """Milestone 4C — the real successor to `dimension_qualifier_overrides`
    for dimensions that now HAVE an AI candidate-classification type
    (M3-OD-01, resolved for D2/D6 by Milestone 4B's CandidateDimensionQualifier).
    Structurally mirrors `_build_value_signals`/`_build_risk_claims` above
    exactly: builds one DimensionQualifierSignal per active item of this
    target kind, REGARDLESS of evidence_state (a CANNOT_CONFIRM'd item's
    evidence_state stays whatever it already was, e.g. CURRENT_UNVERIFIED —
    it is not filtered out here). Confirmation-gating happens exactly once,
    downstream, in engine/dimension_engine.py's existing, unmodified
    `is_current_confirmed(s.evidence_state)` check — identical to how an
    unconfirmed CandidateRiskSignal/CandidateEvidenceClassification already
    flows all the way to risk_engine/objective_engine before being filtered
    there, never here. This function performs NO deterministic judgment of
    its own: qualifier/dimension are copied verbatim from the (possibly
    CORRECT-overlaid) representation, never re-derived or re-validated —
    Milestone 4B's atomic-predicate composition and Milestone 4C's own
    CORRECT-guard (confirmation/active_evidence.py) are the only places
    that ever decide what qualifier a candidate is allowed to assert.

    evidence_refs (PMO decision, Milestone 4C checkpoint): populated with
    the supporting Stage-1 observation's own id
    (CandidateDimensionQualifier.resolved_observation_id, already inherited
    verbatim from Milestone 4B's inherited-grounding design) — never a
    synthetic atomic-predicate id. Atomic-predicate provenance (for the 2
    compound qualifiers) is never copied onto the signal; it stays exactly
    where Milestone 4B put it (ExtractionResult.dimension_qualifier_
    predicate_evidence, addressable via (resolved_observation_id,
    dimension)) and remains reachable from this signal by following
    evidence_refs[0] back to that same key — audit chain preserved without
    flattening compound provenance into the signal itself."""
    signals = []
    for item in active.items:
        if item.target_kind != ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER:
            continue
        resolved_observation_id = item.representation.get("resolved_observation_id")
        signals.append(DimensionQualifierSignal(
            signal_id=item.observation_id,
            dimension=item.representation["dimension"],
            evidence_id=item.source_evidence_id,
            evidence_state=item.evidence_state,
            qualifier=item.representation["qualifier"],
            evidence_refs=(resolved_observation_id,) if resolved_observation_id else (),
        ))
    return tuple(signals)


def _build_dimension_signals(
    active: ActiveEvidenceSet,
    dimension_qualifier_overrides: Optional[DimensionQualifierOverrides],
) -> tuple:
    if not dimension_qualifier_overrides:
        return ()
    signals = []
    for observation_id, (dimension, qualifier) in dimension_qualifier_overrides.items():
        item = active.by_observation_id(observation_id)
        if item is None:
            raise ValueError(
                f"dimension_qualifier_overrides references observation_id="
                f"{observation_id!r}, which is not an active (non-excluded) item in "
                "the supplied ActiveEvidenceSet."
            )
        signals.append(DimensionQualifierSignal(
            signal_id=f"{item.observation_id}-{dimension.value}",
            dimension=dimension,
            evidence_id=item.source_evidence_id,
            evidence_state=item.evidence_state,
            qualifier=qualifier,
        ))
    return tuple(signals)


def _normalize_objective_text(text: str) -> str:
    """Milestone 3D, approved objective-equivalence rule: deterministic
    normalization ONLY -- collapse all internal whitespace runs to a
    single space, strip leading/trailing whitespace, casefold. No
    semantic similarity, embeddings, fuzzy matching, or LLM involvement.
    Two confirmed ObjectiveCandidate statements are the same objective
    identity only if this normalized form matches exactly; anything else
    (a synonym, a reworded restatement, a superset/subset phrasing) is
    treated as a genuine conflict, never silently reconciled."""
    return " ".join(text.split()).casefold()


def _resolve_objective(
    active_evidence: ActiveEvidenceSet,
    fallback_objective: Optional[Objective],
) -> Tuple[Optional[Objective], ObjectiveResolution]:
    """Milestone 3D — the confirmation-layer objective-identity bridge.
    Returns (the Objective to hand evaluate(), audit metadata describing
    how it got there). See this module's docstring for the full set of
    approved constraints this function must honor.

    Three cases hand the UNCHANGED `fallback_objective` straight back
    (never mutated, never wrapped) so `recompute()` can skip
    `dataclasses.replace()` entirely via an identity check:
      1. `fallback_objective` is already known (`is_known=True`) --
         confirmed extraction evidence never overrides an
         already-declared objective.
      2. `fallback_objective` is None -- the caller deliberately did not
         supply an Objective at all (a different, narrower signal than
         "Unknown"; engine.evaluate() itself skips ObjectiveOutcome
         entirely in this case, unchanged by this milestone), so this
         bridge does not manufacture one.
      3. No confirmed ObjectiveCandidate evidence exists yet, or the
         confirmed statements conflict -- resolution status is
         NOT_ESTABLISHED or CONFLICTING respectively, and the engine
         continues to receive exactly the is_known=False Objective it
         always would have, rendering ObjectiveOutcome=UNKNOWN under the
         existing, unmodified rule. Only the audit metadata explains WHY.
    """
    if fallback_objective is None:
        return None, ObjectiveResolution(
            status=ObjectiveResolutionStatus.NOT_ESTABLISHED,
            objective_id="OBJ-UNSPECIFIED",
            detail="No Objective supplied for this account; objective evaluation not in scope.",
        )

    if fallback_objective.is_known:
        return fallback_objective, ObjectiveResolution(
            status=ObjectiveResolutionStatus.ESTABLISHED,
            objective_id=fallback_objective.objective_id,
            text=fallback_objective.text,
            source_evidence_ref=fallback_objective.source_evidence_ref,
            detail="Objective was already known/declared; not derived from confirmed extraction evidence.",
        )

    candidates: List[ActiveEvidenceItem] = [
        item for item in active_evidence.items
        if item.target_kind == ConfirmationTargetKind.SEMANTIC_OBSERVATION
        and isinstance(item.original, ObjectiveCandidate)
        and is_current_confirmed(item.evidence_state)
    ]

    if not candidates:
        return fallback_objective, ObjectiveResolution(
            status=ObjectiveResolutionStatus.NOT_ESTABLISHED,
            objective_id=fallback_objective.objective_id,
            detail="No confirmed ObjectiveCandidate evidence establishes an objective yet.",
        )

    groups: Dict[str, List[ActiveEvidenceItem]] = {}
    for item in candidates:
        text = item.representation.get("objective_text") or ""
        groups.setdefault(_normalize_objective_text(text), []).append(item)

    if len(groups) > 1:
        conflicting_ids = tuple(sorted(item.observation_id for members in groups.values() for item in members))
        return fallback_objective, ObjectiveResolution(
            status=ObjectiveResolutionStatus.CONFLICTING,
            objective_id=fallback_objective.objective_id,
            conflicting_observation_ids=conflicting_ids,
            detail=(
                f"{len(groups)} confirmed objective statements conflict (normalized text differs) -- "
                "objective identity is unresolved. Reject or Correct all but one confirmed statement "
                "to reconcile; this bridge never auto-picks, concatenates, or averages a conflict."
            ),
        )

    # Exactly one normalized-text group: identity established. Canonical
    # source = the earliest confirmed candidate in the group (approved
    # checkpoint: Objective.source_evidence_ref is singular in the
    # current domain model, so an explicit, deterministic tiebreak is
    # required even when every member says the same thing).
    # confirmation_id is a zero-padded, strictly monotonically increasing
    # counter (confirmation.schemas.create_confirmation_record's
    # itertools.count) even across different targets, so a plain string
    # sort on it is chronological.
    winning_items = next(iter(groups.values()))
    canonical = sorted(winning_items, key=lambda item: (item.confirmation_id or ""))[0]
    contributing_ids = tuple(sorted(item.observation_id for item in winning_items))

    resolved = Objective(
        objective_id=fallback_objective.objective_id,
        text=canonical.representation.get("objective_text"),
        source_evidence_ref=canonical.source_evidence_id,
        is_known=True,
    )
    detail = (
        f"Established from confirmed statement {canonical.observation_id}."
        if len(winning_items) == 1 else
        f"Established from {len(winning_items)} identically-worded confirmed statements "
        f"(after whitespace/case normalization); canonical source is the earliest confirmed "
        f"({canonical.observation_id}) -- Objective.source_evidence_ref is singular in the "
        "current domain model, so this is a deliberate, documented limitation, not a redesign."
    )
    return resolved, ObjectiveResolution(
        status=ObjectiveResolutionStatus.ESTABLISHED,
        objective_id=resolved.objective_id, text=resolved.text, source_evidence_ref=resolved.source_evidence_ref,
        contributing_observation_ids=contributing_ids, detail=detail,
    )


def recompute(
    account,
    registry,
    active_evidence: ActiveEvidenceSet,
    *,
    dimensions_to_evaluate: Sequence[DimensionCode] = (),
    dimension_qualifier_overrides: Optional[DimensionQualifierOverrides] = None,
) -> RecomputeDiagnostic:
    value_signals = _build_value_signals(active_evidence)
    risk_claims = _build_risk_claims(active_evidence)
    # Milestone 4C: confirmed/corrected CandidateDimensionQualifier items
    # (the real M3-OD-01 resolution for D2/D6) concatenated with the
    # legacy manual-override path (kept, PMO decision) — see this module's
    # docstring and _build_dimension_signals_from_confirmed_candidates's
    # own docstring for the full rationale.
    dimension_signals = (
        _build_dimension_signals_from_confirmed_candidates(active_evidence)
        + _build_dimension_signals(active_evidence, dimension_qualifier_overrides)
    )

    resolved_objective, objective_resolution = _resolve_objective(active_evidence, account.objective)
    resolved_account = (
        account if resolved_objective is account.objective
        else dataclasses.replace(account, objective=resolved_objective)
    )

    result = evaluate(
        resolved_account, registry,
        value_signals=value_signals,
        dimension_signals=dimension_signals,
        risk_claims=risk_claims,
        dimensions_to_evaluate=tuple(dimensions_to_evaluate),
    )
    return RecomputeDiagnostic(
        result=result,
        active_evidence=active_evidence,
        value_signals=value_signals,
        dimension_signals=dimension_signals,
        risk_claims=risk_claims,
        objective_resolution=objective_resolution,
    )
