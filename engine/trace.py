"""
CHDM v0.1 §3.5, §14, §20 — basic trace generation.

Builds one TraceRecord per governed output, resolving the chain:
  source evidence -> status/confirmation -> methodology object/rule
    -> deterministic conclusion -> interpretation (interpretation/AI
       explanation is out of scope for Milestone 1 — chain stops at the
       deterministic conclusion, which is exactly right per BAR-01 §6's
       explicit exclusion of Grounded Explanation from this milestone).

Queryable without any AI model by construction: every element is a
structured id reference, never natural language requiring
reinterpretation (Technical Architecture §19).
"""

from __future__ import annotations

import itertools

from domain.reason_code import ReasonCode
from domain.trace_record import TraceRecord

_counter = itertools.count(1)


def build_trace(
    subject_object_ref: str,
    evidence_refs: tuple[str, ...],
    reason_code: ReasonCode,
    methodology_version: str = "0.1",
) -> TraceRecord:
    chain = tuple(evidence_refs) + (reason_code.governing_object_id, subject_object_ref)
    return TraceRecord(
        trace_id=f"TRACE-{next(_counter):05d}",
        subject_object_ref=subject_object_ref,
        chain=chain,
        methodology_version=methodology_version,
        reason_code=reason_code,
    )
