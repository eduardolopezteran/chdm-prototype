"""
Milestone 3E — Flagged Conflicts (contradiction review).

Surfaces extraction.schemas.CandidateContradiction markers in a section
deliberately SEPARATE from "Evidence Requiring Review" (ui/review_queue.py
/ ui/item_card.py): a contradiction is an AI-flagged RELATIONSHIP between
two separately-reviewable observations, never itself evidence to
confirm/correct/reject (approved Milestone 3E checkpoint). Acknowledge and
Dismiss here (confirmation.enums.ContradictionReviewAction) never touch
either referenced observation and never change any governed diagnostic
value -- resolution of the underlying disagreement still only happens by
reviewing the two observations themselves via ui/item_card.py, reachable
from "Evidence Requiring Review" or "Audit / history" as normal.

Reviewed/dismissed markers are never removed from this section -- every
CandidateContradiction the extraction produced is always listed here, with
its review history (if any) shown inline, so nothing is ever hidden after
a disposition is recorded (approved audit requirement).

Milestone 4A: same two changes as ui/item_card.py, for consistency between
the two review surfaces. (1) The checkbox + everything it reveals is now
an `@st.fragment` (_render_review_controls below) -- opening/closing a
flagged-conflict card only reruns that card's own fragment, matching the
scroll-position fix applied to Evidence Requiring Review cards; Acknowledge
/Dismiss submissions still call bare `st.rerun()` (default scope="app",
unchanged) inside _submit(). Unlike item_card.py, a contradiction marker
never moves between sections after a disposition is recorded (it always
stays listed in "Flagged Conflicts" -- approved 3E audit requirement), so
the outer-rerun requirement here is weaker than item_card.py's, but the
bare `st.rerun()` call is kept anyway for consistency and because it is
already known-correct, rather than introducing an additional
scope="fragment" variant with no independent way to verify it in this
sandbox. (2) Technical details (ids, ObservationRef internals, full
review history) now render AFTER the Acknowledge/Dismiss controls, not
mixed in above them -- same "decision before raw detail" ordering as
ui/item_card.py.
"""
from __future__ import annotations

import streamlit as st

from confirmation.contradiction_review import (
    ContradictionReviewAction,
    conflict_status,
    resolve_contradiction_terminal,
)

from . import actions
from .labels import CONTRADICTION_ACTION_LABEL
from .review_queue import find_any_item, grounded_text

_STATUS_LABEL = {
    "active": "still active",
    "rejected": "rejected",
    "corrected": "corrected (replacement version)",
    "not_found": "not found in this analysis",
}


def _side_label(side) -> str:
    return _STATUS_LABEL.get(side.status, side.status)


def _plain_conflict_summary(status) -> str:
    if status.both_active:
        return "Both flagged items are still active."
    if status.both_rejected:
        return "Both flagged items have been rejected -- this conflict is likely no longer material."
    return f"Side A is {_side_label(status.side_a)}. Side B is {_side_label(status.side_b)}."


@st.fragment
def _render_review_controls(state, contradiction, contradiction_id: str) -> None:
    """Milestone 4A: see this module's docstring -- fixes the same
    scroll-position issue as ui/item_card.py's _render_review_controls,
    by the same mechanism (fragment-scoped rerun on open/close)."""
    error = state.contradiction_errors.get(contradiction_id)
    opened = st.checkbox("Review this flagged conflict", key=f"open-contra-{contradiction_id}")
    if not opened:
        return

    reviewer_missing = not state.reviewer.strip()
    if reviewer_missing:
        st.warning("Enter a reviewer name in the sidebar before taking any action.")
    if error:
        st.error(error)

    tabs = st.tabs(["Acknowledge", "Dismiss"])
    with tabs[0]:
        st.write(
            "You have seen this flagged conflict. Resolution still happens by "
            "reviewing the two items above, not by acknowledging the flag itself."
        )
        reason = st.text_area("Note (optional)", key=f"ack-note-{contradiction_id}")
        if st.button("Acknowledge", key=f"ack-{contradiction_id}", disabled=reviewer_missing):
            _submit(state, contradiction_id, ContradictionReviewAction.ACKNOWLEDGE, reason)
    with tabs[1]:
        st.write("You have determined this flagged pair is not actually contradictory.")
        reason = st.text_area("Reason (required)", key=f"dismiss-reason-{contradiction_id}")
        if st.button("Dismiss", key=f"dismiss-{contradiction_id}", disabled=reviewer_missing):
            _submit(state, contradiction_id, ContradictionReviewAction.DISMISS, reason)

    records = [r for r in state.contradiction_review_records if r.contradiction_id == contradiction_id]
    with st.expander("Technical details", expanded=False):
        st.json({
            "contradiction_id": contradiction_id,
            "resolved_observation_id_a": contradiction.resolved_observation_id_a,
            "resolved_observation_id_b": contradiction.resolved_observation_id_b,
            "observation_ref_a": {
                "observation_type": contradiction.observation_ref_a.observation_type.value,
                "index": contradiction.observation_ref_a.index,
            },
            "observation_ref_b": {
                "observation_type": contradiction.observation_ref_b.observation_type.value,
                "index": contradiction.observation_ref_b.index,
            },
        })
        if records:
            st.caption("Review history (most recent first):")
            for record in sorted(records, key=lambda r: r.sequence, reverse=True):
                label = CONTRADICTION_ACTION_LABEL.get(record.action, record.action.value)
                line = f"- **{label}** by {record.reviewer} at {record.recorded_at.isoformat()} (`{record.record_id}`)"
                if record.reason:
                    line += f"  \n  Note: {record.reason}"
                st.markdown(line)


def _render_one(state, contradiction) -> None:
    contradiction_id = contradiction.system.observation_id
    all_records = [r for r in state.contradiction_review_records if r.contradiction_id == contradiction_id]
    terminal = resolve_contradiction_terminal(all_records)

    side_a = find_any_item(state.extraction_result, contradiction.resolved_observation_id_a)
    side_b = find_any_item(state.extraction_result, contradiction.resolved_observation_id_b)

    with st.container(border=True):
        st.markdown("**Flagged conflict**")
        st.write(contradiction.conflict_description)
        if contradiction.methodology_construct_hint:
            st.caption(f"May affect: {contradiction.methodology_construct_hint}")

        if side_a is not None:
            st.write(f"> Side A: {grounded_text(side_a[1])}")
        if side_b is not None:
            st.write(f"> Side B: {grounded_text(side_b[1])}")

        active_evidence = state.current_diagnostic.active_evidence
        status = conflict_status(active_evidence, contradiction)
        st.caption(_plain_conflict_summary(status))

        if terminal is not None:
            label = CONTRADICTION_ACTION_LABEL.get(terminal.action, terminal.action.value)
            note = f" -- {terminal.reason}" if terminal.reason else ""
            st.info(f"{label} by {terminal.reviewer} at {terminal.recorded_at.isoformat()}{note}")

        _render_review_controls(state, contradiction, contradiction_id)


def _submit(state, contradiction_id: str, action: ContradictionReviewAction, reason: str) -> None:
    # st.rerun() for the same correctness reason as ui/item_card.py's
    # _submit: without it, this card's already-rendered widgets stay
    # stale for the remainder of this pass.
    try:
        actions.submit_contradiction_review(
            state, contradiction_id=contradiction_id, action=action, reason=(reason or None),
        )
        state.contradiction_errors.pop(contradiction_id, None)
    except ValueError as exc:
        state.contradiction_errors[contradiction_id] = str(exc)
    st.rerun()


def render(state) -> None:
    contradictions = state.extraction_result.candidate_contradictions
    st.subheader("Flagged Conflicts")
    st.caption(
        "The AI flagged these pairs of extracted items as possibly contradicting each other. "
        "Acknowledging or dismissing a flag never confirms, corrects, or rejects the underlying "
        "items -- review those separately in Evidence Requiring Review below."
    )
    if not contradictions:
        st.write("No flagged conflicts in this analysis.")
        return
    for contradiction in contradictions:
        _render_one(state, contradiction)
