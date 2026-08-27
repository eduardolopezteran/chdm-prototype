"""
Milestone 3B — "what to investigate next."

Reuses ui/review_queue.build_review_queue() (the same pure function that
drives the primary review queue) rather than deriving its own notion of
priority -- there is exactly one queue-ordering rule in this codebase.
"""
from __future__ import annotations

import streamlit as st

from . import actions
from .labels import target_kind_label
from .review_queue import build_review_queue

_ICON = {True: "\U0001F534", False: "⚪", None: "\U0001F7E1"}


def _consequentiality_lookup(state):
    def lookup(observation_id):
        report = actions.get_cached_consequentiality(state, observation_id)
        return None if report is None else report.is_consequential
    return lookup


def render(state) -> None:
    st.subheader("What to investigate next")
    queue = build_review_queue(
        state.extraction_result, tuple(state.confirmation_records), _consequentiality_lookup(state),
    )
    if not queue:
        st.success("No items currently require review.")
        return

    st.write(f"{len(queue)} item(s) still require review.")
    top = queue[0]
    st.write(
        f"Highest priority: {_ICON[top.consequentiality]} **{top.label}** "
        f"({target_kind_label(top.target_kind)})"
    )

    result = state.current_diagnostic.result
    if result.dmegs:
        st.write(f"{len(result.dmegs)} open DMEG(s) remain unresolved.")
