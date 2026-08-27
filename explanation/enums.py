"""
I5 build block — explanation-layer enums.

Deliberately tiny: this package introduces no new governed CHDM
vocabulary. GapKind distinguishes the two kinds of real, already-governed
objects a diagnostic question may target (an open DMEG, or a dimension
resolved to INSUFFICIENT_EVIDENCE with no open DMEG of its own) -- both
already exist in engine.evaluate.EvaluationResult; this enum only labels
which one a given DiagnosticQuestion cites.
"""
from __future__ import annotations

from enum import Enum


class GapKind(str, Enum):
    DMEG = "DMEG"
    UNRESOLVED_DIMENSION = "UNRESOLVED_DIMENSION"


class ExplanationFailureReason(str, Enum):
    """Why generate_explanation_and_questions() fell back to the
    deterministic-only result (explanation/pipeline.py). Audit/UI detail
    only -- never affects governed state."""
    PROVIDER_ERROR = "PROVIDER_ERROR"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    CITATION_LINKING_FAILED = "CITATION_LINKING_FAILED"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"
