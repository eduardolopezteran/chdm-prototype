"""
Milestone 3C — plain-language display of extraction-time rejections.

Deliberately distinct from a human Reject action (confirmation/ +
ui/item_card.py): an extraction-time rejection means the AI's OWN
candidate claim failed schema/grounding validation inside
extraction.pipeline.run_extraction() and never became a reviewable item
at all (extraction.pipeline.ExtractionValidationFailure) -- there is
nothing here for a CSM to confirm, correct, or reject, because it was
never accepted into the review queue in the first place. A human Reject,
by contrast, is a reviewer's own decision about an item that WAS accepted
and WAS presented for review (ui/review_queue.py / ui/item_card.py). This
module never touches the confirmation journal and calls no
confirmation/ function -- pure display over
extraction.pipeline.ExtractionResult.rejected.
"""
from __future__ import annotations

import streamlit as st

_PLAIN_REASON = {
    "SCHEMA_INVALID": "The AI's response for this item wasn't formatted correctly.",
    "MALFORMED_TOP_LEVEL_OUTPUT": "The AI's overall response wasn't formatted correctly.",
    "BOUNDARY_VIOLATION": "The AI attempted to set a governed conclusion directly, which isn't "
                           "allowed, so the underlying evidence was not captured.",
    "MISSING_SPAN": "The AI didn't quote the exact original text for this item.",
    "SPAN_NOT_FOUND": "The AI's quoted text could not be found in the original evidence.",
    "SPAN_AMBIGUOUS": "The AI's quoted text matched more than one place in the evidence, "
                       "so it couldn't be pinned down exactly.",
    "UNKNOWN_EVIDENCE_ID": "The AI referenced evidence that wasn't provided.",
    "CONTRADICTION_REFERENCES_REJECTED_ITEM": "A flagged conflict referenced an item that "
                                               "couldn't itself be included.",
    "CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE": "A flagged conflict referenced the same "
                                                         "evidence on both sides.",
    "CONTRADICTION_OBSERVATION_NOT_TRACEABLE_TO_EVIDENCE": "A flagged conflict couldn't be traced "
                                                             "back to specific evidence.",
    "CONTRADICTION_REFERENCES_UNSUPPORTED_TYPE": "A flagged conflict referenced something that "
                                                   "can't be compared this way.",
    "CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM": "An AI interpretation referenced evidence "
                                                           "that couldn't itself be included.",
    "CANDIDATE_CLASSIFICATION_REFERENCED_OBSERVATION_NOT_TRACEABLE": "An AI interpretation couldn't "
                                                                      "be linked back to specific evidence.",
    "CANDIDATE_CLASSIFICATION_REFERENCES_UNSUPPORTED_TYPE": "An AI interpretation referenced "
                                                              "something that can't be interpreted this way.",
}
_DEFAULT_PLAIN_REASON = "This item could not be included in the analysis."


def _plain_reason(reason) -> str:
    key = reason.value if hasattr(reason, "value") else str(reason)
    return _PLAIN_REASON.get(key, _DEFAULT_PLAIN_REASON)


def render(extraction_result) -> None:
    rejected = extraction_result.rejected
    if not rejected:
        return
    # A single, non-nested expander (Streamlit does not support nested
    # expanders) -- plain-language summary first, one shared "Technical
    # details" block at the bottom, mirroring ui/item_card.py's existing
    # pattern of keeping raw/internal detail below the decision-relevant
    # content rather than between it.
    with st.expander(f"Some evidence couldn't be included in this analysis ({len(rejected)})", expanded=False):
        st.caption(
            "These are AI extraction issues, not human review decisions -- they never reached "
            "the review queue below, so there is nothing here to confirm, correct, or reject."
        )
        for item in rejected:
            st.write(f"- {_plain_reason(item.reason)}")
        st.markdown("**Technical details**")
        st.json([
            {
                "observation_type": item.observation_type,
                "reason": item.reason.value if hasattr(item.reason, "value") else str(item.reason),
                "detail": item.detail,
            }
            for item in rejected
        ])
