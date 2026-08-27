"""
I5 build block -- Grounded Explanation / Diagnostic Questions panel.

Renders immediately after diagnostic_panel.render(state) (ui/app.py), which
already renders the deterministic "Diagnostic Result / Why this result /
Uncertainty" panel from state.current_diagnostic. This module NEVER
computes or alters anything that panel shows -- it only ever adds an
on-demand AI narrative and phrased questions underneath it, via a button
(approved decision: on-demand generation, not synchronous). If generation
was never triggered, or fails at any stage, this module renders nothing
beyond a plain, low-emphasis notice -- the deterministic panel above it is
always complete and correct on its own, with or without this panel.
"""
from __future__ import annotations

import os

import streamlit as st

from . import actions

_FAILURE_MESSAGES = {
    "PROVIDER_ERROR": "The AI explanation service could not be reached.",
    "MALFORMED_OUTPUT": "The AI response could not be understood.",
    "CITATION_LINKING_FAILED": "The AI response referenced information outside the verified result.",
    "PROHIBITED_CONTENT": "The AI response did not meet this product's content rules.",
}


def _resolve_api_key() -> str | None:
    # Mirrors ui/setup_screen.py's own live-mode key sourcing exactly --
    # no new key-handling mechanism introduced by this panel.
    return os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("live_api_key") or None


def render(state) -> None:
    st.subheader("AI Explanation & Diagnostic Questions")
    st.caption(
        "Optional. Explains the deterministic result above in plain language and phrases a "
        "few questions about what remains unresolved. Never changes the result itself."
    )

    cached = actions.get_cached_explanation(state)

    col1, col2 = st.columns([1, 3])
    with col1:
        generate_clicked = st.button("Generate AI explanation", key="generate_ai_explanation")
    if cached is not None:
        with col2:
            st.caption("Showing a previously generated explanation for the current result.")

    if generate_clicked:
        api_key = _resolve_api_key()
        if not api_key:
            st.warning(
                "No API key is configured for live AI extraction/explanation. "
                "Set one on the setup screen to use this feature."
            )
        else:
            with st.spinner("Generating explanation..."):
                cached = actions.generate_explanation(state, api_key=api_key)

    if cached is None:
        return  # never generated this run -- render nothing further, deterministic panel stands alone

    if cached.is_fallback:
        message = _FAILURE_MESSAGES.get(cached.fallback_reason.value, "AI explanation is unavailable.")
        st.info(f"{message} The deterministic result above is unaffected.")
        return

    with st.expander("AI Explanation", expanded=True):
        st.caption("Narrative explanation -- not a rule output. See the panel above for the governed result.")
        st.write(cached.explanation.generated_text)

    if cached.questions:
        with st.expander(f"Diagnostic Questions ({len(cached.questions)})", expanded=True):
            for q in cached.questions:
                st.write(f"**{q.rank}.** {q.text}")
                st.caption(f"Targets: `{q.source_gap_ref}` -- {q.stake_description}")
    else:
        with st.expander("Diagnostic Questions", expanded=False):
            st.write("No unresolved material items require a diagnostic question right now.")
