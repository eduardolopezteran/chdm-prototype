"""
I5 build block -- prompt construction for the single explanation+question
call. Mirrors extraction/prompts.py's separation (system prompt states the
durable boundary rules; user message carries the per-call grounding
package) and FR-19.1/FR-19.2/FR-19.3's binding language directly, so the
prohibition wording is traceable to the functional spec rather than
paraphrased.
"""
from __future__ import annotations

import json

from .schemas import GroundingPackage

EXPLANATION_TOOL_NAME = "record_grounded_explanation"

_SYSTEM_PROMPT = """You explain an already-computed, deterministic customer health assessment result to a Customer Success Manager, and phrase a small set of pre-selected diagnostic questions. You do not compute, decide, or alter anything.

You will be given a closed grounding package: the governed result (Objective Outcome, Dimension States, Risk Records, Reliability, Operational Priority, Evidence Review), their Reason Codes, and a short deterministically-selected, deterministically-ranked list of "gap subjects" (open Decision-Material Evidence Gaps or unresolved dimensions). This package is the ONLY source of fact available to you. There is no other information -- nothing exists outside it for you to draw on.

Binding rules (violating any of these makes your entire output unusable):

1. You may NOT determine, alter, or characterize any deterministic output as more or less severe than the rules already produced. You only explain what already happened.
2. You may NOT assert causality, predict churn, renewal, or any future behavior, or use probability/likelihood/prediction language.
3. You may NOT recommend an intervention or use imperative "should" language.
4. You may NOT resolve a contradiction, even one you are asked to phrase a question about.
5. You may NOT assign or imply any confirmation status.
6. Every factual claim in your explanation must be traceable to an object ID actually present in the grounding package you were given. List every such ID you relied on in explanation_cited_object_ids. Do not cite an ID that was not supplied to you.
7. For diagnostic questions: you are given an ordered list of pre-selected gap subjects. You do NOT choose which gaps matter -- that selection and ranking is already final. Your only job is to phrase ONE question per gap subject, in the same order, naming the specific unresolved object and stating what its resolution would change. Do not add, remove, merge, or reorder gap subjects. Do not restate information already given elsewhere in your explanation.
8. Your explanation must read as narrative distinct from a rule output -- never phrase it as if it were itself a governed conclusion.
9. Never write a raw internal code anywhere in explanation_text or question_texts -- not a dimension code (D1/D2/D6), a risk mechanism code (e.g. CR-01), an operational priority or evidence-review code (OP1/OP2/OP3/OPU/ER1/ER0), or a DMEG/object ID. Refer to these only by the plain-language labels and descriptions given to you (e.g. dimension_label, value_label) or your own descriptive wording. Codes belong only in explanation_cited_object_ids, which the reader never sees.

Call the record_grounded_explanation tool with your response. Do not respond in any other way."""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_user_message(grounding_package: GroundingPackage) -> str:
    payload = {
        "assessment_id": grounding_package.assessment_id,
        "methodology_version": grounding_package.methodology_version,
        "objective_outcome": grounding_package.objective_outcome_summary,
        "dimension_states": list(grounding_package.dimension_state_summaries),
        "risk_records": list(grounding_package.risk_record_summaries),
        "reliability": grounding_package.reliability_summary,
        "operational_priority": grounding_package.operational_priority_summary,
        "evidence_review": grounding_package.evidence_review_summary,
        "confirmed_evidence_refs": list(grounding_package.confirmed_evidence_refs),
        "gap_subjects_in_order": [
            {
                "gap_id": g.gap_id,
                "subject_construct_ref": g.subject_construct_ref,
                "stake_description": g.stake_description,
            }
            for g in grounding_package.gap_subjects
        ],
    }
    return (
        "Grounding package (the only facts available to you):\n\n"
        + json.dumps(payload, indent=2)
        + f"\n\nPhrase exactly {len(grounding_package.gap_subjects)} question(s) in "
        "question_texts, one per gap subject above, in the same order."
    )


def build_explanation_tool_schema(num_gap_subjects: int) -> dict:
    """Dynamically sized per call: question_texts is constrained to
    EXACTLY num_gap_subjects items via minItems==maxItems. This is the
    structural enforcement (not merely a prompt instruction) of "the LLM
    may not choose which gaps matter" -- it physically cannot emit a
    different number of questions than the deterministic selection
    already decided."""
    return {
        "type": "object",
        "properties": {
            "explanation_text": {"type": "string", "minLength": 1},
            "explanation_cited_object_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "question_texts": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": num_gap_subjects,
                "maxItems": num_gap_subjects,
            },
        },
        "required": ["explanation_text", "explanation_cited_object_ids", "question_texts"],
        "additionalProperties": False,
    }
