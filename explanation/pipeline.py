"""
I5 build block -- orchestration.

generate_explanation_and_questions() is the ONLY entry point ui/explanation_panel.py
calls. It is a leaf, display-only operation:

  RecomputeDiagnostic (already computed, already governed)
    -> build_grounding_package()            [explanation/grounding.py, pure]
    -> provider.generate()                  [explanation/provider.py, one model call]
    -> jsonschema shape validation           [defense: primary boundary mechanism]
    -> check_citation_linking() (both fields) [TAC-13]
    -> check_prohibited_content() (both fields) [FR-19.1]
    -> ExplanationResult(explanation, questions)   OR
    -> ExplanationResult(fallback_reason=...) on ANY failure at any step

Never calls into domain/, engine/, or confirmation/ for anything beyond
reading the already-finished RecomputeDiagnostic it's handed -- no
recomputation, no state mutation, no write-back. Never calls
extraction/ or confirmation/ to fetch additional (e.g. unconfirmed)
content -- the only inputs are what build_grounding_package() already
assembled from the governed result.

Fail-closed by construction: this module has exactly one success return
path and every other path returns (never raises) an ExplanationResult with
a fallback_reason. ui/explanation_panel.py therefore never needs its own
error handling beyond checking `.is_fallback` -- mirrors the deterministic
CHDM engine's own "never blocks completion" discipline (FR-17.1), applied
here to the AI layer instead: an AI failure never blocks or corrupts the
deterministic result already on screen.
"""
from __future__ import annotations

import itertools

import jsonschema

from confirmation.schemas import RecomputeDiagnostic

from .enums import ExplanationFailureReason
from .errors import ModelServiceError
from .grounding import build_grounding_package
from .grounding_check import check_citation_linking, check_prohibited_content
from .prompts import build_explanation_tool_schema
from .provider import ExplanationProvider
from .schemas import DiagnosticQuestion, Explanation, ExplanationResult, GroundingPackage

_call_counter = itertools.count(1)


def _fallback(grounding_package: GroundingPackage, reason: ExplanationFailureReason, detail: str) -> ExplanationResult:
    return ExplanationResult(
        explanation=None,
        questions=(),
        grounding_package=grounding_package,
        fallback_reason=reason,
        fallback_detail=detail,
    )


def generate_explanation_and_questions(
    diagnostic: RecomputeDiagnostic,
    provider: ExplanationProvider,
    *,
    assessment_id: str,
    methodology_version: str,
) -> ExplanationResult:
    grounding_package = build_grounding_package(
        diagnostic, assessment_id=assessment_id, methodology_version=methodology_version,
    )

    try:
        raw = provider.generate(grounding_package)
    except ModelServiceError as e:
        return _fallback(grounding_package, ExplanationFailureReason.PROVIDER_ERROR, str(e))

    try:
        jsonschema.validate(
            instance=raw,
            schema=build_explanation_tool_schema(len(grounding_package.gap_subjects)),
        )
    except jsonschema.exceptions.ValidationError as e:
        return _fallback(grounding_package, ExplanationFailureReason.MALFORMED_OUTPUT, str(e))

    explanation_text = raw["explanation_text"]
    declared_cited_ids = raw["explanation_cited_object_ids"]
    question_texts = raw["question_texts"]

    citation_result = check_citation_linking(explanation_text, declared_cited_ids, grounding_package)
    if not citation_result.is_clean:
        return _fallback(
            grounding_package, ExplanationFailureReason.CITATION_LINKING_FAILED,
            f"ungrounded_declared_ids={citation_result.ungrounded_declared_ids}, "
            f"ungrounded_text_tokens={citation_result.ungrounded_text_tokens}",
        )
    for q_text in question_texts:
        q_citation_result = check_citation_linking(q_text, [], grounding_package)
        if not q_citation_result.is_clean:
            return _fallback(
                grounding_package, ExplanationFailureReason.CITATION_LINKING_FAILED,
                f"question text cited an ungrounded token: {q_citation_result.ungrounded_text_tokens}",
            )

    prohibited_result = check_prohibited_content(explanation_text)
    if not prohibited_result.is_clean:
        return _fallback(
            grounding_package, ExplanationFailureReason.PROHIBITED_CONTENT,
            f"explanation matched prohibited pattern(s): {prohibited_result.matches}",
        )
    for q_text in question_texts:
        q_prohibited_result = check_prohibited_content(q_text)
        if not q_prohibited_result.is_clean:
            return _fallback(
                grounding_package, ExplanationFailureReason.PROHIBITED_CONTENT,
                f"question matched prohibited pattern(s): {q_prohibited_result.matches}",
            )

    n = next(_call_counter)
    explanation = Explanation(
        explanation_id=f"EXPL-{n:06d}",
        grounding_package_ref=f"GP-{n:06d}",
        generated_text=explanation_text,
        model_version=provider.model_version,
    )
    questions = tuple(
        DiagnosticQuestion(
            question_id=f"DQ-{gap.gap_id}",
            text=q_text,
            source_gap_ref=gap.gap_id,
            stake_description=gap.stake_description,
            rank=index + 1,
        )
        for index, (gap, q_text) in enumerate(zip(grounding_package.gap_subjects, question_texts))
    )

    return ExplanationResult(
        explanation=explanation,
        questions=questions,
        grounding_package=grounding_package,
        fallback_reason=None,
    )
