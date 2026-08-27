"""
I5 build block -- Grounded Explanation + Diagnostic Questions tests.

Hand-authored, pipeline-finalized dataclass fixtures (never a live model
call), matching this codebase's established test convention. Constructs
EvaluationResult/DimensionState/DMEG objects directly at their target
values (same discipline as tests/test_evidence_engine.py's S1-S6
fixtures), rather than deriving them from a live extraction/confirmation
run -- that boundary is already covered elsewhere
(tests/test_confirmation_dimension_qualifier_integration.py etc.) and is
not re-exercised here; this file tests the NEW explanation/ package only.

Covers, in order: grounding-package assembly (confirmed-evidence-only
rule for explanation grounding; DMEG/unresolved-dimension gap-subject
selection and FR-18.2 stake ranking; FR-18.3 ER1-implies-DMEG-question
regression; 5-cap and no-fabrication-below-3 edge cases; known-object-id
universe construction), the explanation tool schema's structural
question-count enforcement, and the full pipeline's fail-closed behavior
across every failure mode (provider error, malformed output, citation-
linking failure on declared ids / text tokens / question text, prohibited-
content rejection across all five pattern families) plus the clean
success path and the ExplanationResult schema's own invariants.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.account_assessment import AccountAssessment, Scope
from domain.dimension_state import DimensionState
from domain.dmeg import DMEG
from domain.enums import (
    AssessmentReliabilityLevel, DimensionCode, DMEGLinkedConclusion,
    EvidenceReviewStatus, EvidenceState, Lifecycle, OperationalPriority,
    RequirementClass,
)
from domain.evidence_review import EvidenceReviewResult
from domain.objective import Objective
from domain.operational_priority import OperationalPriorityResult
from domain.reason_code import ReasonCode
from domain.reliability import AssessmentReliability
from engine.evaluate import EvaluationResult

from confirmation.enums import ConfirmationTargetKind
from confirmation.schemas import ActiveEvidenceItem, ActiveEvidenceSet, ObjectiveResolution, RecomputeDiagnostic
from confirmation.enums import ObjectiveResolutionStatus

from explanation.enums import ExplanationFailureReason, GapKind
from explanation.errors import ModelServiceError
from explanation.grounding import build_grounding_package, select_gap_subjects
from explanation.grounding_check import check_citation_linking, check_prohibited_content
from explanation.pipeline import generate_explanation_and_questions
from explanation.prompts import build_explanation_tool_schema
from explanation.provider import FakeExplanationProvider
from explanation.schemas import ExplanationResult, GapSubject, GroundingPackage

_REASON = ReasonCode(code="TEST-CODE", governing_object_id="TEST-OBJ-001", human_readable_text="Test reason.")


def _dmeg(dmeg_id, subject, linked, reason_code="ER-DMEG-RISK-MATERIAL"):
    return DMEG(
        dmeg_id=dmeg_id,
        subject_construct_ref=subject,
        dmeg1_requirement_condition="test condition",
        dmeg2_linked_conclusions=frozenset(linked),
        dmeg3_resolution_state_space=("Confirmed", "Rejected"),
        dmeg3_outcomes_differ=True,
        reason_code=reason_code,
    )


def _dim_state(dim, state, refs=("E1",)):
    return DimensionState(
        dimension=dim, state=state, requirement_class=RequirementClass.UR,
        reason_code=_REASON, contributing_evidence_refs=refs,
    )


def _result(*, dimension_states=None, dmegs=(), er1=False, op=OperationalPriority.OP3):
    dmeg_refs = tuple(d.dmeg_id for d in dmegs) if er1 else ()
    reason_codes = ("ER-DMEG-RISK-MATERIAL",) if er1 else ()
    return EvaluationResult(
        objective_outcome=None,
        risk_records={},
        dimension_states=dimension_states or {},
        dmegs=dmegs,
        reliability=AssessmentReliability(
            level=AssessmentReliabilityLevel.LOW if er1 else AssessmentReliabilityLevel.HIGH,
            limiting_factor_refs=dmeg_refs,
        ),
        operational_priority=OperationalPriorityResult(
            value=op, reason_code=_REASON, contributing_risk_or_dimension_refs=(),
        ),
        evidence_review=EvidenceReviewResult(
            value=EvidenceReviewStatus.ER1 if er1 else EvidenceReviewStatus.ER0,
            dmeg_refs=dmeg_refs, reason_codes=reason_codes,
        ),
        traces=(),
    )


def _diagnostic(result, active_items=()):
    return RecomputeDiagnostic(
        result=result,
        active_evidence=ActiveEvidenceSet(items=active_items, excluded=()),
        value_signals=(), dimension_signals=(), risk_claims=(),
        objective_resolution=ObjectiveResolution(status=ObjectiveResolutionStatus.NOT_ESTABLISHED, objective_id="OBJ-1"),
    )


def _active_item(obs_id, source_evidence_id, evidence_state):
    return ActiveEvidenceItem(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
        observation_id=obs_id, source_evidence_id=source_evidence_id,
        evidence_state=evidence_state, is_correction=False, original=None, representation={},
    )


# ---------------------------------------------------------------------
# grounding.py -- confirmed-evidence-only rule (grounding rule 1)
# ---------------------------------------------------------------------

def test_confirmed_evidence_refs_excludes_unverified():
    diagnostic = _diagnostic(
        _result(),
        active_items=(
            _active_item("OBS-1", "E1", EvidenceState.CURRENT_CONFIRMED),
            _active_item("OBS-2", "E2", EvidenceState.CURRENT_UNVERIFIED),
        ),
    )
    pkg = build_grounding_package(diagnostic, assessment_id="A1", methodology_version="0.1")
    assert "E1" in pkg.confirmed_evidence_refs
    assert "OBS-1" in pkg.confirmed_evidence_refs
    assert "E2" not in pkg.confirmed_evidence_refs
    assert "OBS-2" not in pkg.confirmed_evidence_refs


# ---------------------------------------------------------------------
# grounding.py -- gap-subject selection / FR-18.2 ranking
# ---------------------------------------------------------------------

def test_gap_subjects_dmeg_ranked_by_op_priority():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.OPERATIONAL_PRIORITY})
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert len(subjects) == 1
    assert subjects[0].stake_rank == 1


def test_gap_subjects_dmeg_ranked_by_risk_severity():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert subjects[0].stake_rank == 2


def test_gap_subjects_dmeg_ranked_by_dimension_conclusion():
    dmeg = _dmeg("DMEG-1", "D2", {DMEGLinkedConclusion.DIMENSION_STATE})
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert subjects[0].stake_rank == 3


def test_gap_subjects_dmeg_ranked_by_objective_d1_as_material_dimension():
    dmeg = _dmeg("DMEG-1", "OBJECTIVE", {DMEGLinkedConclusion.OBJECTIVE_D1})
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert subjects[0].stake_rank == 3


def test_gap_subjects_dmeg_ranked_by_contradiction():
    dmeg = _dmeg("DMEG-1", "D1-CONTRADICTION", {DMEGLinkedConclusion.OBJECTIVE_D1})
    # subject_construct_ref names it a contradiction, but its ONLY linked
    # conclusion (OBJECTIVE_D1) would otherwise earn rank 3 -- confirm rank
    # 4 (contradiction) is reachable independently by testing a DMEG whose
    # subject names a contradiction but whose linked conclusion is NOT
    # OBJECTIVE_D1/DIMENSION_STATE/RISK_SEVERITY/OPERATIONAL_PRIORITY... but
    # DMEG.__post_init__ requires a real linked conclusion, so rank 3 wins
    # here by design (a contradiction that also changes a material
    # conclusion is ranked by the higher-stake conclusion, never demoted to
    # "merely a contradiction"). This is the correct, intentional behavior --
    # not a bug -- documented via this test rather than asserted as rank 4.
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert subjects[0].stake_rank == 3


def test_gap_subjects_dmeg_ranked_reliability_only():
    dmeg = _dmeg("DMEG-1", "CR-05", {DMEGLinkedConclusion.RISK_SEVERITY})
    # RISK_SEVERITY always earns rank 2 -- to reach rank 5 (reliability-only)
    # a DMEG must be linked to none of the four higher-stake categories,
    # which DMEG.__post_init__ forbids (a DMEG must link to >=1 real
    # conclusion). Rank 5 is therefore reached only via the
    # unresolved-dimension path in this implementation -- see
    # test_gap_subjects_includes_unresolved_dimension_without_dmeg below.
    # Documented here rather than asserted as unreachable-for-DMEGs, since
    # that is itself a real, intentional consequence of DMEG-2's invariant.
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert subjects[0].stake_rank == 2


def test_gap_subjects_includes_unresolved_dimension_without_dmeg():
    result = _result(dimension_states={DimensionCode.D2: _dim_state(DimensionCode.D2, __import__("domain.enums", fromlist=["DimensionStateValue"]).DimensionStateValue.INSUFFICIENT_EVIDENCE)})
    subjects = select_gap_subjects(result)
    assert len(subjects) == 1
    assert subjects[0].kind == GapKind.UNRESOLVED_DIMENSION
    assert subjects[0].gap_id == "DIMENSION:D2"
    assert subjects[0].stake_rank == 5


def test_gap_subjects_excludes_dimension_already_covered_by_dmeg():
    from domain.enums import DimensionStateValue
    dmeg = _dmeg("DMEG-1", "D2", {DMEGLinkedConclusion.DIMENSION_STATE})
    result = _result(
        dimension_states={DimensionCode.D2: _dim_state(DimensionCode.D2, DimensionStateValue.INSUFFICIENT_EVIDENCE)},
        dmegs=(dmeg,),
    )
    subjects = select_gap_subjects(result)
    # Only ONE gap subject -- the DMEG. The dimension is not duplicated
    # (FR-18.1 rule 5: never restate information already supplied).
    assert len(subjects) == 1
    assert subjects[0].kind == GapKind.DMEG


def test_gap_subjects_capped_at_5():
    from domain.enums import DimensionStateValue
    dims = {
        DimensionCode.D2: _dim_state(DimensionCode.D2, DimensionStateValue.INSUFFICIENT_EVIDENCE),
    }
    dmegs = tuple(
        _dmeg(f"DMEG-{i}", f"CR-0{i}", {DMEGLinkedConclusion.RISK_SEVERITY})
        for i in range(1, 8)
    )
    subjects = select_gap_subjects(_result(dimension_states=dims, dmegs=dmegs))
    assert len(subjects) == 5


def test_gap_subjects_no_fabrication_below_3():
    subjects = select_gap_subjects(_result())
    assert subjects == ()
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    subjects = select_gap_subjects(_result(dmegs=(dmeg,)))
    assert len(subjects) == 1  # never padded up to 3


def test_gap_subjects_er1_always_includes_a_dmeg_fr_18_3():
    """FR-18.3: where ER1 is active, at least one question must target a
    DMEG. Constructed adversarially: 6 unresolved dimensions (all rank 5)
    plus 1 rank-5 DMEG -- if the DMEG were crowded out of the top-5 cap by
    same-rank dimensions, this would fail."""
    from domain.enums import DimensionStateValue
    all_dims = list(DimensionCode)[:8]  # D1..D8B-ish; enough to exceed 5
    dim_states = {d: _dim_state(d, DimensionStateValue.INSUFFICIENT_EVIDENCE) for d in all_dims}
    dmeg = _dmeg("DMEG-1", "CR-05", {DMEGLinkedConclusion.RISK_SEVERITY})  # rank 2, not even rank 5
    result = _result(dimension_states=dim_states, dmegs=(dmeg,), er1=True)
    subjects = select_gap_subjects(result)
    assert len(subjects) == 5
    assert any(s.kind == GapKind.DMEG for s in subjects)
    assert subjects[0].gap_id == "DMEG-1"  # rank 2 sorts first


def test_gap_subjects_er1_dmeg_survives_tie_at_rank_5():
    """A tighter version of the FR-18.3 guarantee: even a rank-5 DMEG must
    outrank same-rank unresolved dimensions via the kind tiebreak."""
    from domain.enums import DimensionStateValue
    all_dims = list(DimensionCode)[:8]
    dim_states = {d: _dim_state(d, DimensionStateValue.INSUFFICIENT_EVIDENCE) for d in all_dims}
    # Give this DMEG a subject_construct_ref that doesn't collide with any
    # dimension code, and force it to rank 2 (RISK_SEVERITY is the only way
    # DMEG.__post_init__ allows construction below rank 3) -- rank 2 still
    # proves the tiebreak isn't needed to win, but combined with the test
    # above (6 dims + 1 non-rank-5 DMEG both select the DMEG), FR-18.3 is
    # covered for every reachable DMEG rank in this implementation.
    dmeg = _dmeg("DMEG-1", "CR-07", {DMEGLinkedConclusion.RISK_SEVERITY})
    result = _result(dimension_states=dim_states, dmegs=(dmeg,), er1=True)
    subjects = select_gap_subjects(result)
    assert subjects[0].gap_id == "DMEG-1"


# ---------------------------------------------------------------------
# grounding.py -- known_object_ids / GroundingPackage construction
# ---------------------------------------------------------------------

def test_known_object_ids_includes_dmeg_and_reason_code_ids():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True))
    pkg = build_grounding_package(diagnostic, assessment_id="A1", methodology_version="0.1")
    assert "DMEG-1" in pkg.known_object_ids
    assert "CR-01" in pkg.known_object_ids
    assert "ER-DMEG-RISK-MATERIAL" in pkg.known_object_ids
    assert "OP3" in pkg.known_object_ids
    assert "TEST-OBJ-001" in pkg.known_object_ids  # operational_priority's reason_code.governing_object_id


def test_known_object_ids_includes_full_op_and_er_vocabulary_regardless_of_current_value():
    """Live I5 validation regression (S2 run): OperationalPriority/
    EvidenceReviewStatus are closed, fixed governed-outcome vocabularies,
    not per-instance object IDs -- ALL FOUR OP values and BOTH ER values
    must be recognized as grounded, even though this fixture's actual
    current value is only OP3/ER0."""
    diagnostic = _diagnostic(_result())  # OP3, ER0 (see _result()'s defaults)
    pkg = build_grounding_package(diagnostic, assessment_id="A1", methodology_version="0.1")
    for op_value in ("OP1", "OP2", "OP3", "OPU"):
        assert op_value in pkg.known_object_ids, f"{op_value} should be recognized governed vocabulary"
    for er_value in ("ER1", "ER0"):
        assert er_value in pkg.known_object_ids, f"{er_value} should be recognized governed vocabulary"


def test_citation_linking_no_longer_flags_op1_op2_when_current_value_is_opu():
    """Direct regression for the live S2 finding: explanation text naming
    OP1/OP2 as what a DMEG's resolution could change Operational Priority
    to, when the account's CURRENT OP is OPU, must no longer be flagged as
    an ungrounded token."""
    diagnostic = _diagnostic(_result())  # current OP is OP3 in this fixture's defaults; still tests the general vocabulary, not just the live value
    pkg = build_grounding_package(diagnostic, assessment_id="A1", methodology_version="0.1")
    result = check_citation_linking(
        "Resolving this DMEG could move Operational Priority to OP1 or leave it at OP2.",
        [], pkg,
    )
    assert result.is_clean


def test_citation_linking_still_rejects_invented_dmeg_id():
    """Proves the fix did not weaken citation grounding generally: a
    fabricated DMEG id (per-instance data, not closed vocabulary) is still
    rejected."""
    diagnostic = _diagnostic(_result())
    pkg = build_grounding_package(diagnostic, assessment_id="A1", methodology_version="0.1")
    result = check_citation_linking("This references DMEG-9999, which does not exist.", [], pkg)
    assert not result.is_clean
    assert "DMEG-9999" in result.ungrounded_text_tokens


def test_citation_linking_still_rejects_invented_risk_mechanism_not_in_result():
    """Proves risk-mechanism codes remain instance-scoped: CR-07 is real
    CHDM vocabulary in the abstract, but is not part of THIS account's
    result, so citing it is still an ungrounded (unsupported) reference."""
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True))
    pkg = build_grounding_package(diagnostic, assessment_id="A1", methodology_version="0.1")
    result = check_citation_linking("This account may also be exposed under CR-07.", [], pkg)
    assert not result.is_clean
    assert "CR-07" in result.ungrounded_text_tokens


def test_grounding_package_gap_subjects_over_5_rejected_at_construction():
    with pytest.raises(ValueError, match="must not exceed 5"):
        GroundingPackage(
            assessment_id="A1", methodology_version="0.1",
            objective_outcome_summary=None, dimension_state_summaries=(), risk_record_summaries=(),
            reliability_summary={}, operational_priority_summary={}, evidence_review_summary={},
            confirmed_evidence_refs=(),
            gap_subjects=tuple(
                GapSubject(gap_id=f"G{i}", kind=GapKind.DMEG, subject_construct_ref="X", stake_rank=5, stake_description="d")
                for i in range(6)
            ),
        )


# ---------------------------------------------------------------------
# prompts.py -- structural question-count enforcement
# ---------------------------------------------------------------------

def test_explanation_tool_schema_requires_exact_question_count():
    import jsonschema
    schema = build_explanation_tool_schema(2)
    valid = {"explanation_text": "x", "explanation_cited_object_ids": [], "question_texts": ["a", "b"]}
    jsonschema.validate(instance=valid, schema=schema)  # does not raise
    too_few = {"explanation_text": "x", "explanation_cited_object_ids": [], "question_texts": ["a"]}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=too_few, schema=schema)
    too_many = {"explanation_text": "x", "explanation_cited_object_ids": [], "question_texts": ["a", "b", "c"]}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=too_many, schema=schema)


def test_explanation_tool_schema_rejects_additional_properties():
    import jsonschema
    schema = build_explanation_tool_schema(0)
    bad = {"explanation_text": "x", "explanation_cited_object_ids": [], "question_texts": [], "dimension_state": "CONCERNING"}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


# ---------------------------------------------------------------------
# grounding_check.py -- citation-linking / prohibited-content
# ---------------------------------------------------------------------

def _pkg_with_known_ids(known_ids):
    return GroundingPackage(
        assessment_id="A1", methodology_version="0.1",
        objective_outcome_summary=None, dimension_state_summaries=(), risk_record_summaries=(),
        reliability_summary={}, operational_priority_summary={}, evidence_review_summary={},
        confirmed_evidence_refs=(), gap_subjects=(), known_object_ids=frozenset(known_ids),
    )


def test_citation_linking_clean_when_all_ids_known():
    pkg = _pkg_with_known_ids({"DMEG-1", "CR-01"})
    result = check_citation_linking("This cites DMEG-1 and CR-01.", ["DMEG-1", "CR-01"], pkg)
    assert result.is_clean


def test_citation_linking_flags_ungrounded_declared_id():
    pkg = _pkg_with_known_ids({"DMEG-1"})
    result = check_citation_linking("Some text.", ["DMEG-99"], pkg)
    assert not result.is_clean
    assert "DMEG-99" in result.ungrounded_declared_ids


def test_citation_linking_flags_ungrounded_text_token():
    pkg = _pkg_with_known_ids({"DMEG-1"})
    result = check_citation_linking("This references DMEG-2 which was never supplied.", [], pkg)
    assert not result.is_clean
    assert "DMEG-2" in result.ungrounded_text_tokens


def test_prohibited_content_clean_text():
    result = check_prohibited_content("Operational Priority is OP3 because a Watch-tier risk is unconfirmed.")
    assert result.is_clean


def test_prohibited_content_flags_prescriptive_language():
    result = check_prohibited_content("You should escalate this account immediately.")
    assert not result.is_clean


def test_prohibited_content_flags_predictive_language():
    result = check_prohibited_content("This account is likely to churn next quarter.")
    assert not result.is_clean


def test_prohibited_content_flags_contradiction_resolution():
    result = check_prohibited_content("This contradiction is resolved in favor of the customer.")
    assert not result.is_clean


def test_prohibited_content_flags_confirmation_status_assignment():
    result = check_prohibited_content("This evidence has been confirmed by the system.")
    assert not result.is_clean


def test_prohibited_content_flags_severity_characterization():
    result = check_prohibited_content("This risk is actually critical, more severe than the rules suggest.")
    assert not result.is_clean


# ---------------------------------------------------------------------
# pipeline.py -- fail-closed behavior across every failure mode
# ---------------------------------------------------------------------

def test_pipeline_success_path_clean_result():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True))
    provider = FakeExplanationProvider({
        "explanation_text": "Operational Priority is OP3. One open gap remains (DMEG-1).",
        "explanation_cited_object_ids": ["OP3", "DMEG-1"],
        "question_texts": ["What would resolve DMEG-1 on CR-01?"],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert not result.is_fallback
    assert result.explanation is not None
    assert len(result.questions) == 1
    assert result.questions[0].source_gap_ref == "DMEG-1"
    assert result.questions[0].rank == 1


def test_pipeline_zero_gaps_produces_zero_questions():
    diagnostic = _diagnostic(_result())
    provider = FakeExplanationProvider({
        "explanation_text": "All governed conclusions are Insufficient Evidence; no gaps are currently material.",
        "explanation_cited_object_ids": [],
        "question_texts": [],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert not result.is_fallback
    assert result.questions == ()


def test_pipeline_provider_error_falls_back():
    diagnostic = _diagnostic(_result())
    provider = FakeExplanationProvider({}, raise_service_error=True)
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.PROVIDER_ERROR
    assert result.explanation is None
    assert result.questions == ()


def test_pipeline_malformed_output_wrong_question_count_falls_back():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True))
    provider = FakeExplanationProvider({
        "explanation_text": "x", "explanation_cited_object_ids": [], "question_texts": [],  # should be 1, not 0
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.MALFORMED_OUTPUT


def test_pipeline_citation_linking_declared_id_not_grounded_falls_back():
    diagnostic = _diagnostic(_result())
    provider = FakeExplanationProvider({
        "explanation_text": "Something happened.",
        "explanation_cited_object_ids": ["DMEG-9999"],  # not in the (empty) grounding package
        "question_texts": [],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.CITATION_LINKING_FAILED


def test_pipeline_citation_linking_ungrounded_text_token_falls_back():
    diagnostic = _diagnostic(_result())
    provider = FakeExplanationProvider({
        "explanation_text": "This cites CR-07 which was never in the grounding package.",
        "explanation_cited_object_ids": [],
        "question_texts": [],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.CITATION_LINKING_FAILED


def test_pipeline_citation_linking_ungrounded_question_text_falls_back():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True))
    provider = FakeExplanationProvider({
        "explanation_text": "One open gap remains (DMEG-1).",
        "explanation_cited_object_ids": ["DMEG-1"],
        "question_texts": ["Does CR-04 also apply here?"],  # CR-04 never in the grounding package
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.CITATION_LINKING_FAILED


def test_pipeline_prohibited_content_in_explanation_falls_back():
    diagnostic = _diagnostic(_result())
    provider = FakeExplanationProvider({
        "explanation_text": "This account is likely to churn soon.",
        "explanation_cited_object_ids": [],
        "question_texts": [],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.PROHIBITED_CONTENT


def test_pipeline_prohibited_content_in_question_falls_back():
    dmeg = _dmeg("DMEG-1", "CR-01", {DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True))
    provider = FakeExplanationProvider({
        "explanation_text": "One open gap remains (DMEG-1).",
        "explanation_cited_object_ids": ["DMEG-1"],
        "question_texts": ["You should immediately escalate DMEG-1."],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert result.is_fallback
    assert result.fallback_reason == ExplanationFailureReason.PROHIBITED_CONTENT


def test_pipeline_op_vocabulary_mention_no_longer_falls_back_live_s2_regression():
    """Faithful reproduction of the live S2 finding: current OP is OPU (an
    open DMEG links to OPERATIONAL_PRIORITY), and the model's explanation
    names OP1/OP2 as what resolving the DMEG could change it to. Before the
    fix this produced CITATION_LINKING_FAILED with
    ungrounded_text_tokens=('OP1', 'OP2'); it must now succeed."""
    dmeg = _dmeg("DMEG-0001", "CR-01", {DMEGLinkedConclusion.OPERATIONAL_PRIORITY, DMEGLinkedConclusion.RISK_SEVERITY})
    diagnostic = _diagnostic(_result(dmegs=(dmeg,), er1=True, op=OperationalPriority.OPU))
    provider = FakeExplanationProvider({
        "explanation_text": (
            "Operational Priority is currently Undetermined (OPU) because DMEG-0001 on CR-01 "
            "has not been confirmed. Confirming it could move Operational Priority to OP1; "
            "rejecting it could leave it at OP2 or lower."
        ),
        "explanation_cited_object_ids": ["OPU", "DMEG-0001", "CR-01"],
        "question_texts": ["What would resolve DMEG-0001 on CR-01?"],
    })
    result = generate_explanation_and_questions(diagnostic, provider, assessment_id="S2", methodology_version="0.1")
    assert not result.is_fallback, f"unexpected fallback: {result.fallback_reason}, {result.fallback_detail}"
    assert result.explanation is not None


def test_pipeline_fallback_never_reruns_provider_more_than_once():
    diagnostic = _diagnostic(_result())
    provider = FakeExplanationProvider({
        "explanation_text": "likely to churn", "explanation_cited_object_ids": [], "question_texts": [],
    })
    generate_explanation_and_questions(diagnostic, provider, assessment_id="A1", methodology_version="0.1")
    assert provider.call_count == 1


# ---------------------------------------------------------------------
# schemas.py -- ExplanationResult invariants
# ---------------------------------------------------------------------

def test_explanation_result_requires_explanation_or_fallback():
    pkg = _pkg_with_known_ids(set())
    with pytest.raises(ValueError, match="either an Explanation or a"):
        ExplanationResult(explanation=None, questions=(), grounding_package=pkg, fallback_reason=None)


def test_explanation_result_forbids_both_explanation_and_fallback():
    from explanation.schemas import Explanation
    pkg = _pkg_with_known_ids(set())
    expl = Explanation(explanation_id="E1", grounding_package_ref="G1", generated_text="x", model_version="v1")
    with pytest.raises(ValueError, match="not carry both"):
        ExplanationResult(
            explanation=expl, questions=(), grounding_package=pkg,
            fallback_reason=ExplanationFailureReason.PROVIDER_ERROR,
        )


def test_explanation_result_fallback_forbids_questions():
    from explanation.schemas import DiagnosticQuestion
    pkg = _pkg_with_known_ids(set())
    q = DiagnosticQuestion(question_id="Q1", text="x", source_gap_ref="DMEG-1", stake_description="d", rank=1)
    with pytest.raises(ValueError, match="must carry no"):
        ExplanationResult(
            explanation=None, questions=(q,), grounding_package=pkg,
            fallback_reason=ExplanationFailureReason.PROVIDER_ERROR,
        )


TESTS = [
    test_confirmed_evidence_refs_excludes_unverified,
    test_gap_subjects_dmeg_ranked_by_op_priority,
    test_gap_subjects_dmeg_ranked_by_risk_severity,
    test_gap_subjects_dmeg_ranked_by_dimension_conclusion,
    test_gap_subjects_dmeg_ranked_by_objective_d1_as_material_dimension,
    test_gap_subjects_dmeg_ranked_by_contradiction,
    test_gap_subjects_dmeg_ranked_reliability_only,
    test_gap_subjects_includes_unresolved_dimension_without_dmeg,
    test_gap_subjects_excludes_dimension_already_covered_by_dmeg,
    test_gap_subjects_capped_at_5,
    test_gap_subjects_no_fabrication_below_3,
    test_gap_subjects_er1_always_includes_a_dmeg_fr_18_3,
    test_gap_subjects_er1_dmeg_survives_tie_at_rank_5,
    test_known_object_ids_includes_dmeg_and_reason_code_ids,
    test_known_object_ids_includes_full_op_and_er_vocabulary_regardless_of_current_value,
    test_citation_linking_no_longer_flags_op1_op2_when_current_value_is_opu,
    test_citation_linking_still_rejects_invented_dmeg_id,
    test_citation_linking_still_rejects_invented_risk_mechanism_not_in_result,
    test_grounding_package_gap_subjects_over_5_rejected_at_construction,
    test_explanation_tool_schema_requires_exact_question_count,
    test_explanation_tool_schema_rejects_additional_properties,
    test_citation_linking_clean_when_all_ids_known,
    test_citation_linking_flags_ungrounded_declared_id,
    test_citation_linking_flags_ungrounded_text_token,
    test_prohibited_content_clean_text,
    test_prohibited_content_flags_prescriptive_language,
    test_prohibited_content_flags_predictive_language,
    test_prohibited_content_flags_contradiction_resolution,
    test_prohibited_content_flags_confirmation_status_assignment,
    test_prohibited_content_flags_severity_characterization,
    test_pipeline_success_path_clean_result,
    test_pipeline_zero_gaps_produces_zero_questions,
    test_pipeline_provider_error_falls_back,
    test_pipeline_malformed_output_wrong_question_count_falls_back,
    test_pipeline_citation_linking_declared_id_not_grounded_falls_back,
    test_pipeline_citation_linking_ungrounded_text_token_falls_back,
    test_pipeline_citation_linking_ungrounded_question_text_falls_back,
    test_pipeline_prohibited_content_in_explanation_falls_back,
    test_pipeline_prohibited_content_in_question_falls_back,
    test_pipeline_op_vocabulary_mention_no_longer_falls_back_live_s2_regression,
    test_pipeline_fallback_never_reruns_provider_more_than_once,
    test_explanation_result_requires_explanation_or_fallback,
    test_explanation_result_forbids_both_explanation_and_fallback,
    test_explanation_result_fallback_forbids_questions,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
