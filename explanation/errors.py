"""
I5 build block -- explanation-layer exceptions. Mirrors extraction/errors.py's
contract exactly: these never touch or change deterministic assessment
state; they only ever affect whether AI-generated explanation/questions
are shown, vs. the deterministic-only fallback.
"""
from __future__ import annotations

from typing import Any, Optional


class ExplanationError(Exception):
    """Base class for every explanation-layer exception."""


class ModelServiceError(ExplanationError):
    """The provider/API call itself failed (network, timeout, non-2xx,
    provider-side error). Must never change deterministic assessment
    state -- callers simply receive no AI explanation/questions this run."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.cause = cause


class MalformedOutputError(ExplanationError):
    """The model response could not be brought into schema-valid shape.
    No partial/fabricated explanation or questions are ever shown."""

    def __init__(self, message: str, *, last_raw_output: Any = None):
        super().__init__(message)
        self.last_raw_output = last_raw_output
