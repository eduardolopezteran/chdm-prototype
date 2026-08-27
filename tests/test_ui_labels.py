"""
Milestone 4A — ui/labels.py plain-language map tests.

Pure-function tests, no Streamlit dependency at import/run time (mirrors
the existing ui/ test suite's style, e.g. tests/test_ui_review_queue.py).
Two things matter here: (1) every real governed enum value this codebase
can actually produce has an explicit, deliberate label -- no "no missing
case" gaps, same completeness bar Milestone 3C's test_ui_extraction_
bridge.py already holds _PLAIN_REASON to; (2) the fallback for anything
NOT explicitly mapped degrades to a readable string rather than crashing.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import ConfirmationAction, EvidenceReviewStatus, EvidenceState, OperationalPriority

from confirmation.enums import ConfirmationTargetKind, ContradictionReviewAction
from extraction.schemas import AdoptionObservation, ExtractionSystemFields, MissingInformationCandidate, SourceSpan
from extraction.enums import InferenceBasis

from ui.labels import (
    CONFIRMATION_ACTION_LABEL,
    CONTRADICTION_ACTION_LABEL,
    EVIDENCE_REVIEW_LABEL,
    EVIDENCE_STATE_LABEL,
    FIELD_LABEL,
    OPERATIONAL_PRIORITY_LABEL,
    correct_field_options,
    field_label,
    target_kind_label,
)


def _system(obs_id):
    from datetime import datetime, timezone
    return ExtractionSystemFields(
        observation_id=obs_id, model_provider="t", model_version="v1",
        extracted_at=datetime.now(timezone.utc), trace_id=f"T-{obs_id}",
    )


def _span(text):
    return SourceSpan(text=text, start_char=0, end_char=len(text))


# ---- Completeness: every real enum member has a deliberate label ----

def test_every_confirmation_target_kind_has_a_label():
    for kind in ConfirmationTargetKind:
        label = target_kind_label(kind)
        assert label and not label.isupper()


def test_every_operational_priority_has_a_label():
    for op in OperationalPriority:
        assert op in OPERATIONAL_PRIORITY_LABEL
        assert op.value in OPERATIONAL_PRIORITY_LABEL[op]  # code kept visible in parentheses


def test_every_evidence_review_status_has_a_label():
    for er in EvidenceReviewStatus:
        assert er in EVIDENCE_REVIEW_LABEL
        assert er.value in EVIDENCE_REVIEW_LABEL[er]


def test_every_confirmation_action_has_a_label():
    for action in ConfirmationAction:
        assert action in CONFIRMATION_ACTION_LABEL


def test_every_contradiction_review_action_has_a_label():
    for action in ContradictionReviewAction:
        assert action in CONTRADICTION_ACTION_LABEL


def test_every_evidence_state_has_a_label():
    for state in EvidenceState:
        assert state in EVIDENCE_STATE_LABEL


# ---- target_kind_label ----

def test_target_kind_label_known_values():
    assert target_kind_label(ConfirmationTargetKind.SEMANTIC_OBSERVATION) == "Extracted observation"
    assert target_kind_label(ConfirmationTargetKind.MISSING_INFORMATION_CANDIDATE) == "Missing information"
    assert target_kind_label(ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL) == "Possible risk signal"
    assert target_kind_label(ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION) == "Possible value evidence"


def test_target_kind_label_never_leaks_raw_enum_value():
    for kind in ConfirmationTargetKind:
        label = target_kind_label(kind)
        assert kind.value not in label or " " in label  # never the bare SCREAMING_SNAKE token alone


# ---- field_label ----

def test_field_label_known_field():
    assert field_label("mechanism") == "Risk mechanism"
    assert field_label("proposed_severity_tier") == "Proposed severity"


def test_field_label_fallback_for_unmapped_field():
    assert field_label("some_future_field") == "Some future field"


# ---- correct_field_options ----

def test_correct_field_options_mechanism():
    assert correct_field_options(None, "mechanism") == ("CR-01", "CR-02", "CR-08")


def test_correct_field_options_severity_tier():
    assert correct_field_options(None, "proposed_severity_tier") == ("WATCH", "MATERIAL", "CRITICAL")


def test_correct_field_options_proposed_basis():
    assert correct_field_options(None, "proposed_basis") == (
        "PROXY_SUPPORTED", "MEASURED_OPERATIONAL_EVIDENCE", "CUSTOMER_CONFIRMED", "INDEPENDENTLY_VERIFIED",
    )


def test_correct_field_options_supports():
    assert correct_field_options(None, "supports") == ("ACHIEVED", "PROGRESSING", "NOT_ACHIEVED")


def test_correct_field_options_basis_for_positive_observation_type():
    obs = AdoptionObservation(
        source_evidence_id="E-1", source_span=_span("text"), basis=InferenceBasis.EXPLICIT,
        workflow_or_use_case="x", observed_behavior="y", system=_system("OBS-1"),
    )
    options = correct_field_options(obs, "basis")
    assert set(options) == {"EXPLICIT", "INFERRED_CANDIDATE"}


def test_correct_field_options_basis_for_missing_information_candidate():
    obs = MissingInformationCandidate(
        missing_item="renewal date", reviewed_evidence_ids=("E-1",), system=_system("OBS-2"),
    )
    options = correct_field_options(obs, "basis")
    assert options == ("NOT_FOUND_IN_REVIEWED_EVIDENCE",)


def test_correct_field_options_none_for_free_text_field():
    assert correct_field_options(None, "source_evidence_id") is None
    assert correct_field_options(None, "objective_text") is None
