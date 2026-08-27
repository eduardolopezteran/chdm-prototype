"""
Milestone 4A — shared plain-language display maps.

Single source of truth for every place the UI turns an internal enum
value or a raw extraction dataclass field name into something a reviewer
without CHDM/Python background can read, mirroring the pattern already
established by ui/extraction_review.py's _PLAIN_REASON map (Milestone 3C)
and review_queue.py's shared grounded_text() (Milestone 3B) -- both were
added after the SAME kind of thing (a display rule) was found duplicated
or missing in more than one ui/ module. Every map below has a documented
fallback for anything unmapped, so an unmapped value degrades to a
readable (if not perfectly worded) string instead of crashing or leaking
a raw enum repr.

Governed values themselves are UNCHANGED: this module only decides how
to PRINT domain.enums.OperationalPriority / EvidenceReviewStatus /
ConfirmationAction and confirmation.enums.ConfirmationTargetKind /
ContradictionReviewAction values, and how to label extraction dataclass
field names in the Correct form. Nothing here is read by confirmation/,
engine/, or extraction/ -- it is purely a rendering concern, one
direction only (governed value -> display string), never the reverse.
"""
from __future__ import annotations

from domain.enums import ConfirmationAction, EvidenceReviewStatus, EvidenceState, OperationalPriority

from confirmation.enums import ConfirmationTargetKind, ContradictionReviewAction
from extraction.enums import InferenceBasis, MissingInformationBasis
from extraction.json_schemas import (
    EVIDENCE_BASIS_SCHEMA,
    MECHANISM_SCHEMA,
    SEVERITY_TIER_SCHEMA,
    SUPPORTS_SCHEMA,
)

# ---- What kind of item is this card about? (checkpoint item 7) ----

TARGET_KIND_LABEL = {
    ConfirmationTargetKind.SEMANTIC_OBSERVATION: "Extracted observation",
    ConfirmationTargetKind.MISSING_INFORMATION_CANDIDATE: "Missing information",
    ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL: "Possible risk signal",
    ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION: "Possible value evidence",
    ConfirmationTargetKind.CANDIDATE_DIMENSION_QUALIFIER: "Candidate D2/D6 qualifier",
}


def target_kind_label(target_kind) -> str:
    """Fallback mirrors the pre-Milestone-4A rendering (title-cased enum
    value) so an unmapped future target_kind still reads reasonably."""
    return TARGET_KIND_LABEL.get(target_kind, target_kind.value.replace("_", " ").title())


# ---- D2/D6 qualifier plain-language meaning (Milestone 4C) ----
# Wording taken directly from the state_rule/safeguard prose comments
# already sitting next to each code in domain/signals.py's own
# DIMENSION_QUALIFIERS ("not an invented taxonomy") -- this map does not
# add any new claim about what a qualifier means, only restates the
# already-approved one-line description in reviewer-facing language.

QUALIFIER_MEANING = {
    "INTENDED_WORKFLOWS_OPERATING_NORMALLY": "The customer's intended workflows are operating as expected.",
    "AUTOMATION_RELIABLE_LOW_LOGIN_OK": (
        "Automation is reliably handling the workflow, so low direct login activity is expected, not concerning."
    ),
    "NARROW_BREADTH_OR_CONCENTRATION": "Usage is real but narrow or concentrated in a small part of the product.",
    "WORKFLOWS_NOT_OCCURRING": "An intended workflow does not appear to be happening at all.",
    "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED": (
        "Adoption is materially declining with no explanation on record."
    ),
    "APPROPRIATE_SPONSOR_COVERAGE": "An appropriate executive sponsor remains actively engaged.",
    "CHAMPION_LOST_NO_SUCCESSOR": "A champion has confirmed left, and no successor or continuing coverage exists.",
    "CHAMPION_DEPARTURE_UNCONFIRMED": (
        "A stakeholder's departure is a genuine, unresolved signal, not yet confirmed."
    ),
    "SUCCESSION_UNCLEAR_OR_CONCENTRATED": (
        "Relationship continuity is concentrated in one person, or who would take over is unclear."
    ),
}


def qualifier_meaning(qualifier: str) -> str:
    """Fallback mirrors target_kind_label's own philosophy: an unmapped
    future qualifier still reads as a readable (if generic) sentence
    instead of crashing or leaking a raw enum-like string."""
    return QUALIFIER_MEANING.get(qualifier, qualifier.replace("_", " ").capitalize() + ".")


# ---- Operational Priority / Evidence Review (checkpoint item 2) ----
# domain/enums.py already carries the plain meaning in a comment next to
# each code (e.g. "OP1 = Urgent Review") that the UI never surfaced before
# 4A. The code itself is kept in parentheses -- still useful for anyone
# cross-referencing the CHDM v0.1 methodology registry/spec by code.

OPERATIONAL_PRIORITY_LABEL = {
    OperationalPriority.OP1: "Urgent Review (OP1)",
    OperationalPriority.OP2: "Review Required (OP2)",
    OperationalPriority.OP3: "Routine Monitoring (OP3)",
    OperationalPriority.OPU: "Undetermined (OPU)",
}

EVIDENCE_REVIEW_LABEL = {
    EvidenceReviewStatus.ER1: "Evidence Review Required (ER1)",
    EvidenceReviewStatus.ER0: "Evidence Review Not Required (ER0)",
}


# ---- Evidence state (checkpoint item 10) ----

EVIDENCE_STATE_LABEL = {
    EvidenceState.CURRENT_CONFIRMED: "Confirmed",
    EvidenceState.CURRENT_UNVERIFIED: "Unverified",
    EvidenceState.STALE: "Stale",
    EvidenceState.CONTRADICTORY: "Contradictory",
    EvidenceState.UNAVAILABLE: "Unavailable",
    EvidenceState.NOT_APPLICABLE: "Not applicable",
}


# ---- Review-history actions (checkpoint item 10) ----

CONFIRMATION_ACTION_LABEL = {
    ConfirmationAction.CONFIRM: "Confirmed",
    ConfirmationAction.CORRECT: "Corrected",
    ConfirmationAction.REJECT: "Rejected",
    ConfirmationAction.CANNOT_CONFIRM: "Marked as unable to confirm",
}

CONTRADICTION_ACTION_LABEL = {
    ContradictionReviewAction.ACKNOWLEDGE: "Acknowledged",
    ContradictionReviewAction.DISMISS: "Dismissed",
}


# ---- Correct-form field labels (checkpoint item 1) ----
# Covers every editable field across every correctable extraction type
# (extraction/schemas.py) -- i.e. every field NOT in
# ui/item_card.py's _NON_EDITABLE_FIELDS. Anything added to a schema later
# and not listed here falls back to a readable title-cased version of the
# raw field name (same fallback philosophy as TARGET_KIND_LABEL above),
# so a missing entry here is a cosmetic gap, never a crash.

FIELD_LABEL = {
    "source_evidence_id": "Source evidence ID",
    "basis": "Basis (stated vs. inferred)",
    # ObjectiveCandidate
    "objective_text": "Objective statement",
    "stated_outcome": "Stated outcome",
    "measure": "How it's measured",
    "target": "Target",
    "timeframe": "Timeframe",
    # StakeholderObservation
    "person_identifier": "Person",
    "role": "Role",
    "stakeholder_type": "Stakeholder type",
    "sponsor_or_champion_relationship": "Sponsor/champion relationship",
    "continuity_event": "Continuity event",
    "effective_date": "Effective date",
    # AdoptionObservation
    "workflow_or_use_case": "Workflow or use case",
    "observed_behavior": "Observed behavior",
    "adoption_nature": "Nature of adoption",
    "human_vs_automated": "Human vs. automated",
    "evidence_date": "Evidence date",
    # ServiceObservation
    "incident_or_condition": "Incident or condition",
    "severity_language": "Severity (as described)",
    "affected_workflow": "Affected workflow",
    "status": "Status",
    # CommercialObservation
    "event_type": "Event type",
    "description": "Description",
    "commercial_decision_active_candidate": "Active commercial decision?",
    # ExperienceObservation
    "statement": "Statement",
    "stakeholder": "Stakeholder",
    # StrategicObservation
    "event": "Event",
    "affected_org_or_context": "Affected org or context",
    "event_date": "Event date",
    # MissingInformationCandidate
    "missing_item": "What's missing",
    "reviewed_evidence_ids": "Evidence reviewed",
    # CandidateRiskSignal
    "mechanism": "Risk mechanism",
    "proposed_severity_tier": "Proposed severity",
    # CandidateEvidenceClassification
    "proposed_basis": "Type of evidence",
    "supports": "Suggests objective is...",
    # CandidateDimensionQualifier (Milestone 4C)
    "dimension": "Dimension",
    "qualifier": "Qualifier",
    # CandidateContradiction (not correctable today, labeled for
    # completeness/consistency wherever a contradiction's fields are
    # ever shown in a details view)
    "conflict_description": "Description of the conflict",
    "methodology_construct_hint": "What this might affect",
}


def field_label(field_name: str) -> str:
    return FIELD_LABEL.get(field_name, field_name.replace("_", " ").capitalize())


# ---- Dropdown options for the Correct form's constrained fields ----
# (checkpoint item 3). Sourced directly from the SAME schemas
# extraction/validation.py enforces on the way in -- MECHANISM_SCHEMA /
# SEVERITY_TIER_SCHEMA / EVIDENCE_BASIS_SCHEMA / SUPPORTS_SCHEMA -- so a
# reviewer's Correct submission can never propose a value the pipeline
# itself would never have accepted, and the dropdown options can never
# silently drift out of sync with what extraction/ actually validates.
# `basis` is the one field whose valid values depend on WHICH extraction
# type is being corrected (InferenceBasis for the 7 span-grounded
# positive/candidate types; the single-valued MissingInformationBasis for
# MissingInformationCandidate) -- see correct_field_options() below,
# which is keyed by the actual object, not just the field name.

_MECHANISM_OPTIONS = tuple(MECHANISM_SCHEMA["enum"])
_SEVERITY_TIER_OPTIONS = tuple(SEVERITY_TIER_SCHEMA["enum"])
_EVIDENCE_BASIS_OPTIONS = tuple(EVIDENCE_BASIS_SCHEMA["enum"])
_SUPPORTS_OPTIONS = tuple(SUPPORTS_SCHEMA["enum"])
_INFERENCE_BASIS_OPTIONS = tuple(b.value for b in InferenceBasis)
_MISSING_INFORMATION_BASIS_OPTIONS = tuple(b.value for b in MissingInformationBasis)

# Milestone 4C: the 2 compound D2/D6 qualifiers Milestone 4B structurally
# bars the model from proposing directly, and confirmation/active_evidence.py
# _validate_dimension_qualifier_correction (the actual authority boundary)
# structurally bars a human CORRECT action from manufacturing on a
# candidate that wasn't already that exact compound qualifier. This module
# copy exists only so the dropdown itself never OFFERS the bypass in the
# first place -- defense in depth, never the enforcement point.
_COMPOUND_QUALIFIERS = frozenset({"AUTOMATION_RELIABLE_LOW_LOGIN_OK", "CHAMPION_LOST_NO_SUCCESSOR"})


def correct_field_options(obs, field_name: str):
    """Returns the fixed set of valid values for a constrained Correct
    field, or None if the field is free text. `obs` is the ORIGINAL
    extraction object being corrected (needed because `basis` means a
    different enum depending on its type, and `qualifier`'s governed
    vocabulary depends on the candidate's own dimension)."""
    if field_name == "basis":
        from extraction.schemas import MissingInformationCandidate
        if isinstance(obs, MissingInformationCandidate):
            return _MISSING_INFORMATION_BASIS_OPTIONS
        return _INFERENCE_BASIS_OPTIONS
    if field_name == "mechanism":
        return _MECHANISM_OPTIONS
    if field_name == "proposed_severity_tier":
        return _SEVERITY_TIER_OPTIONS
    if field_name == "proposed_basis":
        return _EVIDENCE_BASIS_OPTIONS
    if field_name == "supports":
        return _SUPPORTS_OPTIONS
    if field_name == "qualifier":
        from domain.signals import DIMENSION_QUALIFIERS
        allowed = DIMENSION_QUALIFIERS.get(obs.dimension, ())
        # A simple candidate's dropdown never offers either compound name
        # (that's the "remove the two compound values from the UI
        # correction choices for simple candidates" requirement). A
        # candidate that already IS one of the compound qualifiers keeps
        # its own current value selectable (item_card.py's select_options
        # fallback would add it back anyway if omitted here, since
        # `current` is always prepended when missing from `options` --
        # this branch simply makes the intent explicit rather than relying
        # on that fallback).
        if obs.qualifier in _COMPOUND_QUALIFIERS:
            return allowed
        return tuple(q for q in allowed if q not in _COMPOUND_QUALIFIERS)
    return None
