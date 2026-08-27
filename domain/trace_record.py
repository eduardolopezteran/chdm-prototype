"""
CHDM v0.1 §3.5, §14 — Provenance and traceability.

Material outputs must preserve a trace from:
  evidence -> status/confirmation -> methodology object/rule
    -> deterministic conclusion -> interpretation

A TraceRecord is the ordered chain of object references for ONE governed
output. It is queryable without any AI model — every element is a
structured reference, not natural language.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .reason_code import ReasonCode


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    subject_object_ref: str            # id of the governed output this trace explains
    chain: tuple[str, ...]              # ordered evidence_id/observation_id/... references
    methodology_version: str
    reason_code: ReasonCode
    created_at: datetime = field(default_factory=datetime.utcnow)
    actor: Optional[str] = None         # human actor id, if a human action is in the chain
    model_version: Optional[str] = None  # only set if an AI-generated object is in the chain

    def __post_init__(self) -> None:
        if not self.chain:
            raise ValueError(
                "TraceRecord.chain must not be empty — every governed "
                "output must resolve a full evidence-to-rule-to-output chain (§3.5)."
            )
        if not self.methodology_version:
            raise ValueError("TraceRecord.methodology_version is required (§14.4).")
