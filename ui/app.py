"""
Milestone 3B/3C — Streamlit entrypoint.

Wires the approved diagnostic-first flow: (Milestone 3C) evidence entry ->
diagnostic result -> why -> uncertainty -> flagged conflicts ->
evidence requiring review -> confirmation action -> recomputed result ->
what to investigate next. This module holds no governed logic itself -- every rendered value traces back
to a confirmation.recompute.recompute() call made in ui/actions.py, and
every extracted item traces back to a real
extraction.pipeline.run_extraction() call made in ui/extraction_bridge.py
(Milestone 3C -- ui/sample_data.py's hand-authored, pipeline-bypassing
fixture is no longer the default source of extraction data; it remains
only as a fixture for the pre-existing ui/ test suite).

Run with:  streamlit run run_ui.py   (from the chdm-engine/ directory)
"""
from __future__ import annotations

import streamlit as st

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import DimensionCode, Lifecycle
from domain.objective import Objective
from engine.registry_loader import load_and_validate

from . import (
    actions,
    audit_view,
    contradiction_view,
    diagnostic_panel,
    diff_view,
    explanation_panel,
    extraction_bridge,
    extraction_review,
    item_card,
    next_steps,
    setup_screen,
)
from .review_queue import build_review_queue
from .state import AppState

st.set_page_config(
    page_title="CHDM Human Confirmation (Prototype)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"] p {
    white-space: normal;
    overflow: visible;
    text-overflow: unset;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

# Milestone 1 implements dimension-state evaluation for exactly D1/D2/D6
# (engine/dimension_engine.py's IMPLEMENTED_DIMENSIONS) -- any other code
# raises NotImplementedForMilestone1. D1 is the dimension a confirmed
# CandidateEvidenceClassification can move, via Objective Outcome; D2/D6 are
# moved by a confirmed/corrected CandidateDimensionQualifier (Milestone 4C,
# M3-OD-01 resolved -- see confirmation/recompute.py's
# _build_dimension_signals_from_confirmed_candidates).
_DIMENSIONS_TO_EVALUATE = (DimensionCode.D1, DimensionCode.D2, DimensionCode.D6)


def _build_state_from_outcome(outcome: extraction_bridge.ExtractionRunOutcome) -> AppState:
    """Milestone 3C. Builds the AppState the review flow renders from a
    REAL extraction.pipeline.ExtractionResult (ui/extraction_bridge.py),
    rather than ui/sample_data.py's hand-authored fixture. No
    dimension_qualifier_overrides are set here: unlike Milestone 3B's
    fixture (which pre-declared a fixed D6 mapping as a display-only
    stand-in), a real extraction run's D2/D6 state comes from whatever
    CandidateDimensionQualifier items the stage-2 classifier proposed
    (extraction_bridge.py's run_pipeline now calls
    run_dimension_qualifier_classification) once a reviewer confirms or
    corrects them (Milestone 4C, M3-OD-01 resolved). Until reviewed, D2/D6
    simply show as not-yet-evaluated here, which is the honest result, not
    a workaround.

    Known, flagged architecture gap (not resolved in this milestone, same
    disclosure-not-silence approach as M3-OD-01): AccountAssessment.objective
    is a separate, already-known governed input in the Milestone 1 domain
    model, and there is currently no wiring anywhere in confirmation/ or
    engine/ that promotes a confirmed ObjectiveCandidate into it. This
    module deliberately does NOT fabricate objective text to work around
    that gap (Objective.__post_init__ / CHDM v0.1 spec section 5: "the
    product must never fabricate an objective to enable a conclusion") --
    the objective is left genuinely Unknown, so Objective Outcome
    correctly and honestly renders UNKNOWN for every Milestone 3C
    evidence-entry-driven analysis, even after a value-evidence
    classification is confirmed. Flagged in MANIFEST.txt and the
    Milestone 3C report as a candidate follow-up decision, not decided
    silently here."""
    registry = load_and_validate()
    objective = Objective(objective_id="OBJ-1", text=None, is_known=False)
    account = AccountAssessment(
        assessment_id="ASSESS-M3C-SESSION",
        scope=Scope("SCOPE-M3C-SESSION", "New account analysis", "Milestone 3C prototype"),
        lifecycle=Lifecycle.L3, objective=objective,
    )
    state = AppState(
        account=account, registry=registry, extraction_result=outcome.extraction_result,
        dimensions_to_evaluate=_DIMENSIONS_TO_EVALUATE, dimension_qualifier_overrides={},
        provider_label=outcome.provider_label,
    )
    state.current_diagnostic = actions.initial_recompute(state)
    return state


def get_state() -> AppState:
    return st.session_state.get("app_state")


def _lookup(state):
    def inner(observation_id):
        report = actions.get_cached_consequentiality(state, observation_id)
        return None if report is None else report.is_consequential
    return inner


def main() -> None:
    # Milestone 3C: an AppState only exists once an evidence-entry run has
    # actually succeeded (setup_screen.render() returns a non-None
    # outcome). Before that, there is nothing to review yet -- render the
    # setup screen only, and return. This replaces Milestone 3B's
    # always-available hand-authored sample_data.py fixture as the app's
    # starting point.
    state = get_state()
    if state is None:
        outcome = setup_screen.render()
        if outcome is None:
            return
        st.session_state["app_state"] = _build_state_from_outcome(outcome)
        st.rerun()

    st.title("CHDM Human Confirmation — Prototype")
    st.caption(
        "Backend authority and recomputation only, no production auth/persistence/deployment. "
        "Every value below comes from a fresh, rule-based recomputation over currently confirmed "
        "evidence, run against a real extraction result."
    )

    with st.sidebar:
        st.header("Reviewer")
        state.reviewer = st.text_input("Your name", value=state.reviewer)
        st.caption("No production authentication in this prototype (explicitly out of scope).")
        st.divider()
        st.caption(f"Extraction: {state.provider_label}")

        if st.button("Start a new analysis"):
            # No persistence in this prototype (explicitly out of scope,
            # unchanged since Milestone 3B) — starting over simply
            # discards this session's in-memory state and returns to the
            # setup screen.
            for key in ("app_state", "last_extraction_failure", "setup_error"):
                st.session_state.pop(key, None)
            st.rerun()

    extraction_review.render(state.extraction_result)

    diagnostic_panel.render(state)

    st.divider()
    explanation_panel.render(state)

    st.divider()
    if state.previous_diagnostic is not None:
        st.subheader("What changed after your last action")
        diff = diff_view.diagnostic_diff(state.previous_diagnostic.result, state.current_diagnostic.result)
        if diff:
            for line in diff:
                st.write(f"- {line}")
        else:
            st.write("No change to the diagnostic result. This review has been recorded in the audit trail.")

    # Tracks every target_observation_id whose item card (and its
    # `open-<id>` checkbox widget) has already been rendered in THIS
    # script pass. Needed because a single confirmation action can, within
    # one pass, move an item out of the primary queue and into the
    # excluded/audit branch below -- without this guard, both branches
    # could try to render the very same widget key in the same pass
    # (StreamlitDuplicateElementKey), caught during the Milestone 3B
    # manual UX smoke pass. No st.rerun() is used to route around this
    # instead, since an earlier attempt at that intermittently reset
    # OTHER open items' checkbox state.
    rendered_target_ids: set = set()

    st.divider()
    st.subheader("Evidence Requiring Review")
    queue = build_review_queue(state.extraction_result, tuple(state.confirmation_records), _lookup(state))
    if not queue:
        st.success("No items currently require review.")
    else:
        for index, entry in enumerate(queue):
            item_card.render(state, entry.target_kind, entry.target_observation_id, is_top_item=(index == 0))
            rendered_target_ids.add(entry.target_observation_id)

    st.divider()
    next_steps.render(state)

    st.divider()
    with st.expander("Audit / history (all reviewed items)", expanded=False):
        reviewed_ids = sorted({r.target_observation_id for r in state.confirmation_records})
        if not reviewed_ids:
            st.write("No items reviewed yet.")
        for observation_id in reviewed_ids:
            first_record = next(r for r in state.confirmation_records if r.target_observation_id == observation_id)
            audit_view.render(state, first_record.target_kind, observation_id)
            # Rejected items intentionally drop out of "Evidence Requiring
            # Review" above (they are resolved, not pending) -- but Reject
            # must still be reachable via Correct (the approved
            # reinstatement path), so re-render the same item card's
            # action row here rather than leaving audit view read-only-only.
            if (
                observation_id not in rendered_target_ids
                and state.current_diagnostic.active_evidence.is_excluded(observation_id)
            ):
                item_card.render(state, first_record.target_kind, observation_id, is_top_item=False)
                rendered_target_ids.add(observation_id)
            st.markdown("---")


if __name__ == "__main__":
    main()
