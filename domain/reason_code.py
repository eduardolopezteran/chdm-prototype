"""
CHDM v0.1 §13 — Reason-Code Model.

A ReasonCode attaches a rule citation to a governed output. It never
stands alone: §13.4 requires every consequential output to reference
governing object + governing rule + deterministic reason. ReasonCode
carries the human-readable reason; `governing_object_id` carries the
Methodology Object reference (see methodology_object.py).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonCode:
    code: str                    # e.g. "OP1-CONFIRMED-CRITICAL" — must come
                                   # from registry/reason_codes.yaml families
    governing_object_id: str      # e.g. "CHDM-RISK-CHAMPION-001"
    human_readable_text: str

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("ReasonCode.code must not be empty")
        if not self.governing_object_id:
            # CHDM v0.1 §13.4 / invariant family INV-01/INV-19 generalization:
            # a governed output without a governing methodology rule is invalid.
            raise ValueError(
                "ReasonCode.governing_object_id must not be empty — "
                "a reason code without a governing methodology object "
                "reference is an invalid governed output (§13.4)."
            )
