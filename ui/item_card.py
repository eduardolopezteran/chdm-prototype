"""
Milestone 3B — item card: source traceability, consequentiality badge, and
the four review actions (Confirm / Correct / Reject / Cannot Confirm).

Two independent lanes are never merged here: a candidate classification's
card is a separate card from the semantic observation it interprets, each
submitting its own confirmation.schemas.HumanConfirmationRecord via
ui/actions.submit_action -- confirming/correcting/rejecting one never
touches the other.

Available actions are restricted per the approved state-machine rule:
REJECTED -> CONFIRM is illegal, so a rejected item's card offers only
Correct (the approved reinstatement path) and Cannot Confirm -- it never
even renders a bare Confirm button, avoiding a doomed submit-then-error
round trip.

Milestone 4A additions:
  - Plain-language labels throughout (ui/labels.py) -- the card header, the
    Correct form's field labels, and constrained Correct fields
    (basis/mechanism/proposed_severity_tier/proposed_basis/supports) are
    now dropdowns sourced from the SAME schemas extraction/validation.py
    enforces, instead of free text.
  - Reordered: the four review-action tabs now render BEFORE the
    "Full extracted representation (technical detail)" expander, not
    after -- a reviewer sees the decision to make before the raw JSON.
  - The checkbox + everything it reveals is now wrapped in `@st.fragment`
    (_render_review_controls below) so opening/closing a card only
    reruns that card's own fragment, not the whole script -- this is
    what fixes the Milestone 4A-reported scroll-position jump, since the
    rest of the page's DOM is untouched by a fragment-scoped rerun. The
    actual Confirm/Correct/Reject/Cannot Confirm buttons still call bare
    `st.rerun()` inside `_submit()` (default scope="app", UNCHANGED from
    Milestone 3B) -- an action must still force the OUTER ui/app.py
    script to rerun, since that script (not this fragment) decides which
    section (Evidence Requiring Review vs. Audit/history) the item
    belongs in afterward.
"""
from __future__ import annotations

import dataclasses

import streamlit as st

from domain.enums import ConfirmationAction

from confirmation.state_machine import group_by_target, resolve_terminal
from extraction.schemas import CandidateDimensionQualifier

from . import actions
from .labels import correct_field_options, field_label, target_kind_label
from .review_queue import atomic_predicate_evidence_for, find_item, grounded_text

_CONSEQUENTIALITY_LABELS = {
    True: ("\U0001F534", "Consequential"),
    False: ("⚪", "Not consequential"),
    None: ("\U0001F7E1", "Consequentiality not yet evaluated"),
}

# Milestone 4C: `dimension` is unique to CandidateDimensionQualifier, so it
# is safe to exclude globally, for every type, outright. `basis` is NOT
# globally safe -- it is legitimately correctable on every other type -- so
# it is excluded only for CandidateDimensionQualifier via
# _non_editable_fields_for below, rather than added here.
_NON_EDITABLE_FIELDS = {"source_span", "supporting_observation_ref", "resolved_observation_id", "dimension"}


def _non_editable_fields_for(obs) -> set:
    """Milestone 4C governing requirement: "`dimension` and `basis` are not
    editable" for a CandidateDimensionQualifier's Correct form. `dimension`
    is handled by the global _NON_EDITABLE_FIELDS set above (safe for every
    type); `basis` cannot be, since it IS a legitimately correctable field
    for every other correctable extraction type (CandidateRiskSignal,
    CandidateEvidenceClassification, the 7 span-grounded positive types) --
    so it is added here only for this specific type. This is defense in
    depth (keeping the field off the form entirely) around the real
    authority boundary, confirmation/active_evidence.py's
    _validate_dimension_qualifier_correction, which independently rejects
    any attempt to change `basis` even if a caller bypassed this UI."""
    if isinstance(obs, CandidateDimensionQualifier):
        return _NON_EDITABLE_FIELDS | {"basis"}
    return _NON_EDITABLE_FIELDS

_ACTION_LABELS = {
    ConfirmationAction.CONFIRM: "Confirm",
    ConfirmationAction.CORRECT: "Correct",
    ConfirmationAction.REJECT: "Reject",
    ConfirmationAction.CANNOT_CONFIRM: "Cannot Confirm",
}


def _representation_view(obs) -> dict:
    return {f.name: getattr(obs, f.name) for f in dataclasses.fields(obs) if f.name != "system"}


def _terminal_action_for(state, target_observation_id: str):
    grouped = group_by_target(state.confirmation_records)
    terminal = resolve_terminal(grouped.get(target_observation_id, ()))
    return terminal.action if terminal is not None else None


def _available_actions(terminal_action):
    if terminal_action == ConfirmationAction.REJECT:
        # REJECTED -> CONFIRM is illegal (approved state-machine rule).
        # Reinstatement must go through Correct; Cannot Confirm remains
        # available too (not restricted by this rule).
        return [ConfirmationAction.CORRECT, ConfirmationAction.CANNOT_CONFIRM]
    return [ConfirmationAction.CONFIRM, ConfirmationAction.CORRECT,
            ConfirmationAction.REJECT, ConfirmationAction.CANNOT_CONFIRM]


def _consequentiality_badge(state, target_observation_id: str, *, evaluate_now: bool) -> None:
    report = actions.get_cached_consequentiality(state, target_observation_id)
    if report is None and evaluate_now:
        report = actions.evaluate_consequentiality(state, target_observation_id)
    value = None if report is None else report.is_consequential
    icon, label = _CONSEQUENTIALITY_LABELS[value]
    st.caption(f"{icon} {label}")


@st.fragment
def _render_review_controls(state, target_kind, target_observation_id: str, obs) -> None:
    """Milestone 4A: the checkbox and everything it reveals live in their
    own fragment (see this module's docstring). Toggling "Review this
    item" therefore reruns only this fragment -- the rest of the page
    (everything above and below this card) is never redrawn, which is
    what keeps the browser from jumping the viewport. Confirm/Correct/
    Reject/Cannot Confirm submissions still force a full app rerun (see
    _submit's bare st.rerun() call, unchanged since Milestone 3B) --
    those need ui/app.py's OUTER script to re-run, since that script (not
    this fragment) decides whether the item now belongs in "Evidence
    Requiring Review" or "Audit / history"."""
    opened = st.checkbox("Review this item", key=f"open-{target_observation_id}")
    if not opened:
        return

    # Lazy consequentiality evaluation on open (approved Milestone 3B
    # checkpoint): a plain cache-miss check, not gated behind a single
    # shared "currently open" pointer -- more than one card can be
    # expanded at once (Streamlit does not make checkboxes mutually
    # exclusive), and each expanded card must independently reach its
    # action row on every render, not just the first one checked. An
    # earlier version gated this behind a shared
    # state.opened_target_observation_id + st.rerun(), which caused every
    # render pass to bounce back to whichever item was checked first and
    # never reach any other open card's actions -- caught during the
    # Milestone 3B manual UX smoke pass.
    state.opened_target_observation_id = target_observation_id
    if actions.get_cached_consequentiality(state, target_observation_id) is None:
        actions.evaluate_consequentiality(state, target_observation_id)

    # Milestone 4A: the decision (Confirm/Correct/Reject/Cannot Confirm)
    # now renders BEFORE the raw technical detail, not after -- a
    # reviewer sees what they're being asked to decide before the JSON.
    _render_actions(state, target_kind, target_observation_id, obs)


def render(state, target_kind, target_observation_id: str, *, is_top_item: bool) -> None:
    obs = find_item(state.extraction_result, target_kind, target_observation_id)
    if obs is None:
        st.error(f"Item {target_observation_id!r} not found in the extraction result.")
        return

    with st.container(border=True):
        st.markdown(f"**{target_kind_label(target_kind)}**")
        st.write(f"> {grounded_text(obs)}")
        source_label = getattr(obs, "source_evidence_id", None) or "(scoped evidence review -- see reviewed_evidence_ids)"
        st.caption(f"Source evidence: {source_label}")

        # Milestone 4C: compound-qualifier provenance stays visibly
        # distinguishable from a simple candidate's, per the approved
        # architecture ("do not flatten deterministic-composition
        # provenance"). atomic_predicate_evidence_for() is a pure,
        # read-only lookup -- it returns () for a simple candidate (nothing
        # rendered) and for every other target kind, and never copies or
        # mutates the underlying AtomicPredicateEvidence records.
        atomic_evidence = atomic_predicate_evidence_for(state.extraction_result, obs)
        if atomic_evidence:
            with st.expander(f"Compound qualifier basis ({len(atomic_evidence)} atomic predicates)", expanded=False):
                st.caption(
                    "This qualifier was deterministically composed from the following "
                    "grounded, same-observation, EXPLICIT-basis atomic predicates:"
                )
                for predicate in atomic_evidence:
                    st.write(f"- **{predicate.predicate_id}** ({predicate.basis.value}): \"{predicate.evidence_text}\"")

        _consequentiality_badge(state, target_observation_id, evaluate_now=is_top_item)

        error = state.errors_by_target.get(target_observation_id)
        if error:
            st.error(error)

        _render_review_controls(state, target_kind, target_observation_id, obs)


def _render_actions(state, target_kind, target_observation_id: str, obs) -> None:
    reviewer_missing = not state.reviewer.strip()
    if reviewer_missing:
        st.warning("Enter a reviewer name in the sidebar before taking any action.")

    terminal_action = _terminal_action_for(state, target_observation_id)
    available = _available_actions(terminal_action)
    if terminal_action == ConfirmationAction.REJECT:
        st.info(
            "This item was rejected. It cannot be re-confirmed directly -- "
            "the only path back to active is Correct (a new replacement version)."
        )

    tabs = st.tabs([_ACTION_LABELS[a] for a in available])

    for tab, action in zip(tabs, available):
        with tab:
            if action == ConfirmationAction.CONFIRM:
                st.write("Confirming this as an accurate, verified representation.")
                if st.button("Confirm", key=f"confirm-{target_observation_id}", disabled=reviewer_missing):
                    _submit(state, target_kind, target_observation_id, action)

            elif action == ConfirmationAction.CORRECT:
                st.write("The original stays in history; this creates a NEW confirmed replacement version.")
                corrected = {}
                for field_name, value in _representation_view(obs).items():
                    if field_name in _non_editable_fields_for(obs):
                        continue
                    label = field_label(field_name)
                    options = correct_field_options(obs, field_name)
                    current = str(value)
                    if options is not None:
                        # Milestone 4A: constrained fields (basis, mechanism,
                        # proposed_severity_tier, proposed_basis, supports)
                        # are a dropdown over the SAME values
                        # extraction/validation.py would accept -- a
                        # reviewer can never propose a value the pipeline
                        # itself would reject. `current` should always
                        # already be a valid option (the pipeline validated
                        # it on the way in); the extra branch below is
                        # defense-in-depth only, so an unexpected value
                        # never crashes the form.
                        select_options = list(options) if current in options else [current, *options]
                        new_value = st.selectbox(
                            label, select_options, index=select_options.index(current),
                            key=f"correct-{target_observation_id}-{field_name}",
                        )
                    else:
                        new_value = st.text_input(
                            label, value=current, key=f"correct-{target_observation_id}-{field_name}",
                        )
                    if new_value != current:
                        corrected[field_name] = new_value
                if st.button("Submit correction", key=f"submit-correct-{target_observation_id}", disabled=reviewer_missing):
                    if not corrected:
                        st.error("Change at least one field before submitting a correction.")
                    else:
                        _submit(state, target_kind, target_observation_id, action, corrected_representation=corrected)

            elif action == ConfirmationAction.REJECT:
                reason = st.text_area("Reason (required)", key=f"reject-reason-{target_observation_id}")
                st.caption(
                    "This item will be left out of the analysis. It stays visible in the review "
                    "history, and you can bring it back later with Correct."
                )
                if st.button("Reject", key=f"reject-{target_observation_id}", disabled=reviewer_missing):
                    _submit(state, target_kind, target_observation_id, action, reason=reason)

            elif action == ConfirmationAction.CANNOT_CONFIRM:
                reason = st.text_area("Reason (required)", key=f"cannot-confirm-reason-{target_observation_id}")
                st.caption(
                    "This item stays part of the analysis as unverified. It may keep showing up as "
                    "needing review until it's confirmed, corrected, or rejected."
                )
                if st.button("Cannot Confirm", key=f"cannot-confirm-{target_observation_id}", disabled=reviewer_missing):
                    _submit(state, target_kind, target_observation_id, action, reason=reason)


def _submit(state, target_kind, target_observation_id, action, *, reason=None, corrected_representation=None):
    # st.rerun() is required for correctness: without it, the buttons
    # already rendered for THIS card in the current pass (computed from
    # the PRE-action terminal disposition) stay on-screen unchanged for
    # the remainder of this same pass -- e.g. a just-rejected item would
    # still show a live "Confirm" button until the reviewer's next,
    # unrelated interaction, which is a worse and more confusing defect
    # than the cost of a rerun. ui/app.py's rendered_target_ids guard
    # independently prevents any item's card from being rendered twice
    # within one pass (covering the moment before this rerun completes).
    #
    # KNOWN LIMITATION (see MANIFEST.txt): under Streamlit's AppTest
    # headless test harness specifically, triggering st.rerun() here was
    # observed to intermittently reset OTHER, unrelated open item cards'
    # checkbox state when many cards were expanded simultaneously in a
    # single scripted test. Streamlit's session_state model is designed to
    # persist widget state by key across reruns, and this pattern
    # (mutate-then-rerun on button click) is standard practice -- this may
    # be an AppTest-specific artifact rather than real browser behavior.
    # Flagged for verification in a live browser session before this
    # prototype is used for actual review sessions with many items open
    # at once; not re-tested by the 20 automated ui/ tests, which exercise
    # ui/actions.py and ui/review_queue.py directly, not through Streamlit
    # widget interactions.
    try:
        actions.submit_action(
            state, target_kind=target_kind, target_observation_id=target_observation_id,
            action=action, reason=(reason or None), corrected_representation=corrected_representation,
        )
        state.errors_by_target.pop(target_observation_id, None)
    except ValueError as exc:
        state.errors_by_target[target_observation_id] = str(exc)
    st.rerun()
