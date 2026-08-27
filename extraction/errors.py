"""
Build Milestone 2 — extraction-specific exceptions (spec §13 failure
handling). None of these ever touch or change deterministic CHDM
assessment state; they only ever affect what does or doesn't enter the
extraction pipeline's accepted-observation list.
"""

from __future__ import annotations

from typing import Any, Optional

from .enums import RejectionReason


class ExtractionError(Exception):
    """Base class for every Milestone 2 extraction-layer exception."""


class ModelServiceError(ExtractionError):
    """The provider/API call itself failed (network, timeout, non-2xx,
    provider-side error). Must never change deterministic assessment
    state — callers simply receive no new observations for this request."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.cause = cause


class MalformedOutputError(ExtractionError):
    """The top-level model response could not be brought into schema-valid
    shape even after one repair/retry attempt (spec §13 "Schema failure").
    Extraction fails for the whole request; no partial observations are
    fabricated from an unparseable response."""

    def __init__(self, message: str, *, last_raw_output: Any = None):
        super().__init__(message)
        self.last_raw_output = last_raw_output


class ItemRejected(ExtractionError):
    """A single candidate item (one entry in one of the model's output
    arrays) failed validation and must not enter the accepted path. Always
    carries a RejectionReason so the caller can record a full audit trail
    (spec §13: "reject... rather than silently retain it")."""

    def __init__(self, reason: RejectionReason, detail: str, *, raw_item: Any = None):
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail
        self.raw_item = raw_item
