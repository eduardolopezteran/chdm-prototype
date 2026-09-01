"""
Milestone 3B/3D — Diagnostic Result / Why / Uncertainty panel.

Renders exactly what confirmation.recompute.recompute() returned in
state.current_diagnostic -- never a client-side patched value. Covers
steps 1-3 of the approved sequence: diagnostic result -> why -> uncertainty.
"""
from __future__ import annotations

import re

import streamlit as st

from confirmation.enums import ObjectiveResolutionStatus
from .labels import EVIDENCE_REVIEW_LABEL, OPERATIONAL_PRIORITY_LABEL

_READ_ONLY_DIMENSIONS = {"D2", "D6"}  # M3-OD-01 unresolved -- see MANIFEST.txt

# Milestone 3D — plain-language framing for ObjectiveResolution.status.
# Deliberately distinguishes "no confirmed evidence yet" from "confirmed
# evidence conflicts" (approved checkpoint constraint: the UI must
# explicitly state WHICH of these is the reason Objective Outcome is
# Unknown, never leave it as an undifferentiated "Unknown"). Purely a
# display concern -- objective_resolution itself carries no governed
# CHDM state (confirmation/schemas.py's ObjectiveResolution docstring).
_OBJECTIVE_STATUS_LABEL = {
    ObjectiveResolutionStatus.ESTABLISHED: "Objective established",
    ObjectiveResolutionStatus.NOT_ESTABLISHED: "Objective not yet established",
    ObjectiveResolutionStatus.CONFLICTING: "Objective identity unresolved -- confirmed statements conflict",
}


def _render_objective_resolution(state) -> None:
    res = state.current_diagnostic.objective_resolution
    label = _OBJECTIVE_STATUS_LABEL.get(res.status, res.status.value)
    if res.status == ObjectiveResolutionStatus.ESTABLISHED:
        st.write(f"**{label}:** {res.text}")
    elif res.status == ObjectiveResolutionStatus.CONFLICTING:
        st.warning(
            f"**{label}.** {len(res.conflicting_observation_ids)} confirmed objective statements "
            "disagree with each other. This -- not simply missing evidence -- is why Objective "
            "Outcome is Unknown below. Reject or Correct all but one confirmed statement to resolve it."
        )
    else:
        st.write(f"**{label}.** No confirmed evidence yet states what this account is trying to achieve.")


def render(state) -> None:
    result = state.current_diagnostic.result

    st.subheader("Diagnostic Result")
    cols = st.columns(4)
    op_label = OPERATIONAL_PRIORITY_LABEL.get(result.operational_priority.value, result.operational_priority.value.value)
    er_label = EVIDENCE_REVIEW_LABEL.get(result.evidence_review.value, result.evidence_review.value.value)
    cols[0].metric("Operational Priority", op_label)
    cols[1].metric("Evidence Review", er_label)
    cols[2].metric("Reliability", result.reliability.level.value)
    obj_state = result.objective_outcome.state.value if result.objective_outcome else "N/A"
    cols[3].metric("Objective Outcome", obj_state)

    _render_objective_resolution(state)

_DIMENSION_PUBLIC_LABELS = {
    "D1": "Objective Outcome",
    "D2": "Product Adoption",
    "D6": "Relationship Health",
}

_CITATION_PATTERN = re.compile(r"\s*\(CHDM[^)]*\)\.?\s*$")

def _public_text(text: str) -> str:
    """Strip internal spec citations like '(CHDM v0.1 §4.2)' from reviewer-facing copy."""
    return _CITATION_PATTERN.sub("", text).rstrip()


def render(state) -> None:
    result = state.current_diagnostic.result
    st.subheader("Diagnostic Result")
    cols = st.columns(4)
    op_label = OPERATIONAL_PRIORITY_LABEL.get(result.operational_priority.value, result.operational_priority.value.value)
    er_label = EVIDENCE_REVIEW_LABEL.get(result.evidence_review.value, result.evidence_review.value.value)
    obj_state = result.objective_outcome.state.value if result.objective_outcome else "N/A"
    metrics = [
        ("Operational Priority", op_label),
        ("Evidence Review", er_label),
        ("Reliability", result.reliability.level.value),
        ("Objective Outcome", obj_state),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"**{label}**")
        col.markdown(f"### {value}")
        
    _render_objective_resolution(state)

    with st.expander("Why this result", expanded=False):
        st.write(
            f"**Operational Priority:** "
            f"{_public_text(result.operational_priority.reason_code.human_readable_text)}"
        )
        for dim, dim_state in result.dimension_states.items():
            label = _DIMENSION_PUBLIC_LABELS.get(dim.value, dim.value)
            state_text = dim_state.state.value.replace("_", " ").capitalize()
            st.write(
                f"**{label}:** {state_text}.  \n"
                f"{_public_text(dim_state.reason_code.human_readable_text)}"
            )
        for mech, risk in result.risk_records.items():
            potential = risk.potential_severity.value if risk.potential_severity else None
            activated = risk.activated_severity.value if risk.activated_severity else None
            if activated:
                st.write(f"**Risk status:** {activated.replace('_', ' ').capitalize()} condition is active.")
            elif potential:
                st.write(
                    f"**Risk status:** A potential {potential.replace('_', ' ').lower()} condition "
                    "has been identified, but it is not activated because the supporting evidence "
                    "has not met the confirmation requirements."
                )

    is_er1 = result.evidence_review.value.value == "ER1"
    with st.expander("Uncertainty", expanded=is_er1):
        review_text = "Required" if is_er1 else "Not currently required"
        st.write(f"**Evidence review:** {review_text}")
        if not result.dmegs:
            st.write("**Open material evidence gaps:** None")
        else:
            st.write("**Open material evidence gaps:**")
            for dmeg in result.dmegs:
                impact_text = (
                    "This unresolved gap could affect Operational Priority."
                    if dmeg.affects_operational_priority
                    else "This unresolved gap does not currently affect Operational Priority."
                )
                st.write(f"- {impact_text}")
        st.write(f"**Reliability:** {result.reliability.level.value}")
        if result.reliability.limiting_factor_refs:
            st.write("**Reliability note:** Limited by unresolved material evidence gaps.")
        else:
            st.write("**Reliability note:** No unresolved material evidence gap is currently limiting the assessment.")
