"""
Milestone 3B — audit/history presentation.

Lightweight, explicitly not a full production audit console. Reads
state.current_diagnostic.active_evidence (already computed by the real
backend recompute() call in ui/actions.py) for active/excluded status --
never recomputes or re-derives that here.

Milestone 4A: every enum/field name shown here now goes through
ui/labels.py's plain-language maps (checkpoint item 10) -- the raw
`confirmation_id` moved to the end of each history line, de-emphasized,
rather than leading with it.

Milestone 4C: a historical D2/D6 compound qualifier's atomic-predicate
provenance is shown here too, right after the original AI proposal --
the same read-only atomic_predicate_evidence_for() lookup ui/item_card.py
uses during active review (approved architecture requirement: compound
provenance must remain visibly distinguishable wherever a qualifier is
reviewed, active or historical -- never flattened).
"""
from __future__ import annotations

import streamlit as st

from .labels import CONFIRMATION_ACTION_LABEL, EVIDENCE_STATE_LABEL, field_label
from .review_queue import atomic_predicate_evidence_for, find_item, grounded_text


def render(state, target_kind, target_observation_id: str) -> None:
    obs = find_item(state.extraction_result, target_kind, target_observation_id)
    if obs is None:
        st.error(f"Item {target_observation_id!r} not found.")
        return

    st.write(f"**Original AI proposal** (`{target_observation_id}`):")
    st.write(f"> {grounded_text(obs)}")

    atomic_evidence = atomic_predicate_evidence_for(state.extraction_result, obs)
    if atomic_evidence:
        with st.expander(f"Compound qualifier basis ({len(atomic_evidence)} atomic predicates)", expanded=False):
            st.caption(
                "This qualifier was deterministically composed from the following "
                "grounded, same-observation, EXPLICIT-basis atomic predicates:"
            )
            for predicate in atomic_evidence:
                st.write(f"- **{predicate.predicate_id}** ({predicate.basis.value}): \"{predicate.evidence_text}\"")

    active_evidence = state.current_diagnostic.active_evidence
    if active_evidence.is_excluded(target_observation_id):
        st.write("Status: **Excluded from the analysis** (rejected; retained here for audit)")
    else:
        item = active_evidence.by_observation_id(target_observation_id)
        if item is None:
            st.write("Status: not part of the current analysis.")
        elif item.is_correction:
            st.write("Status: **Active** — a corrected replacement version, Confirmed")
        else:
            state_label = EVIDENCE_STATE_LABEL.get(item.evidence_state, item.evidence_state.value)
            st.write(f"Status: **Active** — {state_label}")

    records = [r for r in state.confirmation_records if r.target_observation_id == target_observation_id]
    if not records:
        st.write("No review history yet.")
        return

    st.write("**Review history** (most recent first):")
    for record in sorted(records, key=lambda r: r.sequence, reverse=True):
        action_label = CONFIRMATION_ACTION_LABEL.get(record.action, record.action.value)
        line = (
            f"- **{action_label}** by {record.reviewer} at {record.recorded_at.isoformat()} "
            f"(`{record.confirmation_id}`)"
        )
        if record.reason:
            line += f"  \n  Reason: {record.reason}"
        if record.corrected_representation:
            changed = ", ".join(
                f"{field_label(k)}: {v}" for k, v in dict(record.corrected_representation).items()
            )
            line += f"  \n  Changed: {changed}"
        st.markdown(line)
