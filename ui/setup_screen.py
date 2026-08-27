"""
Milestone 3C — evidence-entry setup screen (approved checkpoint decision 2:
"Sample scenario" and "Paste customer evidence").

Runs BEFORE the Milestone 3B diagnostic/confirmation flow. Once "Run
analysis" succeeds, ui/app.py builds an AppState from the real
extraction.pipeline.ExtractionResult this screen produced and switches to
the review flow -- this module has no confirmation-state, active-
evidence, or recomputation logic of its own; it only gets evidence text
and a provider choice to ui/extraction_bridge.run_pipeline().

Every label here is deliberately plain and user-facing (approved
checkpoint decision 4) -- no ExtractionResult, provider class name,
schema, trace id, model metadata, or internal enum name appears anywhere
on this screen. Provenance stays available only in the existing
"Technical details" section once review starts (ui/item_card.py).
"""
from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from . import extraction_bridge, sample_scenarios


def _live_mode_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(st.session_state.get("live_api_key"))


def render() -> Optional["extraction_bridge.ExtractionRunOutcome"]:
    """Renders the setup screen. Returns a successful ExtractionRunOutcome
    once the reviewer runs an analysis that did not fail outright this
    pass; otherwise returns None (still on the setup screen, possibly
    showing a failure or validation message from the previous attempt)."""
    st.title("CHDM Human Confirmation — Prototype")
    st.caption(
        "Start by choosing evidence to analyze. Nothing is added to the diagnostic "
        "until you confirm it in the review step that follows."
    )

    failure = st.session_state.pop("last_extraction_failure", None)
    if failure:
        st.error("Extraction could not be completed. You can try again below.")
        with st.expander("Technical details", expanded=False):
            st.write(failure)

    error = st.session_state.pop("setup_error", None)
    if error:
        st.warning(error)

    mode = st.radio(
        "How would you like to start?",
        options=["Sample scenario", "Paste customer evidence"],
        key="entry_mode",
    )

    st.divider()
    st.subheader("AI extraction mode")
    if not _live_mode_available():
        st.caption(
            "Live AI extraction needs an API key for this session. It is used only to run "
            "the analysis below and is never saved or written to a file."
        )
        st.text_input("API key for live AI extraction (optional)", type="password", key="live_api_key")
    live_available = _live_mode_available()
    use_live = st.checkbox(
        "Use live AI extraction",
        value=False,
        disabled=not live_available,
        help=(
            "Off (default): deterministic demo extraction -- no AI call, no network, always the "
            "same result for the same input. Good for development, tests, and repeatable demos. "
            "On: sends the evidence below to a live AI model for real extraction."
        ),
    )

    st.divider()
    fake_response = None
    if mode == "Sample scenario":
        options = sample_scenarios.list_scenarios()
        chosen_key = st.selectbox(
            "Sample scenario", options=[o.key for o in options],
            format_func=lambda k: sample_scenarios.get_scenario(k).title,
        )
        scenario = sample_scenarios.get_scenario(chosen_key)
        st.write(scenario.description)
        raw_text = sample_scenarios.raw_text_for(scenario)
        with st.expander("Evidence text for this scenario", expanded=False):
            st.write(raw_text)
        fake_response = scenario.fake_response
        raw_texts = [raw_text]
    else:
        st.write("Paste one or more notes, emails, or call summaries below.")
        pasted = st.text_area("Customer evidence", height=200, key="pasted_evidence")
        raw_texts = [pasted]

    run_clicked = st.button("Run analysis", type="primary")
    if not run_clicked:
        return None

    evidence_batch = extraction_bridge.build_evidence_batch(raw_texts)
    if not evidence_batch:
        st.session_state["setup_error"] = "Enter some customer evidence before running analysis."
        st.rerun()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("live_api_key") or None
    with st.spinner("Running analysis..."):
        outcome = extraction_bridge.run_pipeline(
            evidence_batch, use_live=use_live, api_key=api_key, fake_response=fake_response,
        )

    if outcome.extraction_result.request_failure is not None:
        st.session_state["last_extraction_failure"] = outcome.extraction_result.request_failure
        st.rerun()

    return outcome
