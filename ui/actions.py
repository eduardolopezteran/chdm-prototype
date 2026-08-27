"""
Milestone 3B — orchestration glue.

Thin adapter over the completed Milestone 3A `confirmation/` backend.
Every governed value the UI renders comes from a fresh
confirmation.recompute.recompute() call made HERE -- this module must
never patch a diagnostic value client-side, and must never reproduce
confirmation-state, active-evidence, or consequentiality logic itself. If
a rule needs to change, it changes in confirmation/, never here.

Consequentiality is lazy, per the approved checkpoint: this module never
calls compute_consequentiality() for a whole queue eagerly. It is only
ever invoked (by ui/app.py / ui/item_card.py) for the current top queue
item and for an item the reviewer explicitly opens. Results are cached in
AppState.consequentiality_cache, keyed against
AppState.confirmation_history_version; a cache hit at a stale version is
treated as "not yet evaluated," never silently reused.

I5 build block: generate_explanation/get_cached_explanation follow the
identical lazy/cached/staleness-by-confirmation-history-version pattern as
consequentiality above, for the on-demand Grounded Explanation +
Diagnostic Questions layer (explanation/). generate_explanation() never
calls into domain/, engine/, or confirmation/ beyond reading
state.current_diagnostic, which this module already produces via ordinary
recompute() elsewhere -- it performs no recomputation of its own and never
writes back into any governed object.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from domain.enums import ConfirmationAction

from confirmation.active_evidence import reconstruct_active_evidence
from confirmation.consequentiality import compute_consequentiality
from confirmation.contradiction_review import ContradictionReviewRecord, create_contradiction_review_record
from confirmation.enums import ConfirmationTargetKind, ContradictionReviewAction
from confirmation.recompute import recompute
from confirmation.schemas import ConsequentialityReport, RecomputeDiagnostic, create_confirmation_record

from explanation.pipeline import generate_explanation_and_questions
from explanation.provider import AnthropicExplanationProvider, ExplanationProvider
from explanation.schemas import ExplanationResult


def _recompute_from_state(state) -> RecomputeDiagnostic:
    active = reconstruct_active_evidence(state.extraction_result, tuple(state.confirmation_records))
    return recompute(
        state.account, state.registry, active,
        dimensions_to_evaluate=state.dimensions_to_evaluate,
        dimension_qualifier_overrides=state.dimension_qualifier_overrides,
    )


def initial_recompute(state) -> RecomputeDiagnostic:
    """The diagnostic result before any reviewer action this session.
    confirmation_records may be empty here -- that is a normal, valid
    input to recompute(), not a special case handled separately."""
    return _recompute_from_state(state)


def submit_action(
    state,
    *,
    target_kind: ConfirmationTargetKind,
    target_observation_id: str,
    action: ConfirmationAction,
    reason: Optional[str] = None,
    corrected_representation: Optional[Dict[str, Any]] = None,
) -> RecomputeDiagnostic:
    """The single entry point every Confirm / Correct / Reject / Cannot
    Confirm button calls. Raises ValueError exactly as the backend raises
    it (from create_confirmation_record's __post_init__ guards, from
    state_machine's illegal-transition check inside reconstruct_active_
    evidence, or from recompute()'s own guards) -- callers must catch and
    display it, never suppress or reinterpret it. On success, the new
    record is appended to the append-only journal and a fresh diagnostic
    is computed and stored; the caller should re-render from
    state.current_diagnostic afterward, never from a locally-held stale
    copy."""
    record = create_confirmation_record(
        target_kind=target_kind,
        target_observation_id=target_observation_id,
        action=action,
        reviewer=state.reviewer,
        reason=reason,
        corrected_representation=corrected_representation,
    )
    # Only appended after create_confirmation_record succeeds -- an
    # invalid submission (e.g. REJECT with no reason, or a reserved
    # reviewer identifier) never touches the journal.
    state.confirmation_records.append(record)
    diag = _recompute_from_state(state)
    state.previous_diagnostic = state.current_diagnostic
    state.current_diagnostic = diag
    return diag


def submit_contradiction_review(
    state,
    *,
    contradiction_id: str,
    action: ContradictionReviewAction,
    reason: Optional[str] = None,
) -> ContradictionReviewRecord:
    """Milestone 3E entry point for Acknowledge/Dismiss on a flagged
    CandidateContradiction marker. Deliberately separate from
    submit_action() above: neither ContradictionReviewAction ever
    confirms/corrects/rejects an observation, and this function never
    calls recompute() -- the governed diagnostic is completely unaffected
    by contradiction review (approved checkpoint constraint), so there is
    nothing to re-render from state.current_diagnostic afterward; only
    state.contradiction_review_records changes. Raises ValueError exactly
    as ContradictionReviewRecord.__post_init__ raises it (e.g. DISMISS
    with no reason, or a missing reviewer) -- callers must catch and
    display it, same contract as submit_action."""
    record = create_contradiction_review_record(
        contradiction_id=contradiction_id,
        action=action,
        reviewer=state.reviewer,
        reason=reason,
    )
    # Only appended after create_contradiction_review_record succeeds --
    # an invalid submission never touches the journal, mirroring
    # submit_action's own append-only-on-success contract.
    state.contradiction_review_records.append(record)
    return record


def get_cached_consequentiality(state, target_observation_id: str) -> Optional[ConsequentialityReport]:
    """None means "not yet evaluated at the current confirmation-history
    version" -- callers (ui/review_queue.py's ordering, ui/item_card.py's
    badge) must render this as "Consequentiality not yet evaluated," never
    silently treat it as non-consequential."""
    cached = state.consequentiality_cache.get(target_observation_id)
    if cached is None:
        return None
    version, report = cached
    if version != state.confirmation_history_version:
        return None  # stale -- a later action may have changed the answer
    return report


def evaluate_consequentiality(state, target_observation_id: str) -> ConsequentialityReport:
    """Actually runs compute_consequentiality() (two additional
    recompute() calls under the hood) and caches the result at the
    current history version. Call this ONLY for the current top queue
    item, or when a reviewer opens an item -- never for the whole queue on
    every render (approved performance constraint)."""
    report = compute_consequentiality(
        state.account, state.registry, state.extraction_result,
        tuple(state.confirmation_records), target_observation_id,
        dimensions_to_evaluate=state.dimensions_to_evaluate,
        dimension_qualifier_overrides=state.dimension_qualifier_overrides,
    )
    state.consequentiality_cache[target_observation_id] = (state.confirmation_history_version, report)
    return report


def get_cached_explanation(state) -> Optional[ExplanationResult]:
    """I5 build block. None means "not yet generated at the current
    confirmation-history version" -- ui/explanation_panel.py must render
    this as "not yet generated," never silently show a stale explanation
    of a since-superseded governed result. Mirrors
    get_cached_consequentiality's exact staleness convention above."""
    if state.explanation_cache is None:
        return None
    version, result = state.explanation_cache
    if version != state.confirmation_history_version:
        return None  # stale -- a later confirmation action may have changed the governed result
    return result


def generate_explanation(state, *, api_key: Optional[str] = None) -> ExplanationResult:
    """I5 build block. On-demand only -- called exclusively from
    ui/explanation_panel.py's "Generate AI explanation" button handler,
    never automatically on render (approved decision: no synchronous
    generation). Always operates on state.current_diagnostic, which is
    itself always produced by confirmation.recompute.recompute() (see
    _recompute_from_state above / ui/app.py) -- this function never
    recomputes anything itself and never sees anything upstream of the
    already-governed RecomputeDiagnostic.

    api_key sourcing mirrors ui/setup_screen.py's own live-mode pattern
    exactly (os.environ.get("ANTHROPIC_API_KEY") or
    st.session_state.get("live_api_key")) -- no new key-handling mechanism
    is introduced by this module; the caller passes whichever key value
    setup_screen.py's own helper already resolved."""
    provider: ExplanationProvider = AnthropicExplanationProvider(api_key=api_key)
    result = generate_explanation_and_questions(
        state.current_diagnostic, provider,
        assessment_id=state.account.assessment_id,
        methodology_version=state.account.methodology_version,
    )
    state.explanation_cache = (state.confirmation_history_version, result)
    return result
