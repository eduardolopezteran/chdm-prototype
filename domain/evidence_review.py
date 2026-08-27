"""
CHDM v0.1 §11 — Evidence Review.
"""

from dataclasses import dataclass, field

from .enums import EvidenceReviewStatus


@dataclass(frozen=True)
class EvidenceReviewResult:
    value: EvidenceReviewStatus
    dmeg_refs: tuple[str, ...] = field(default_factory=tuple)   # DMEG ids that produced ER1
    reason_codes: tuple[str, ...] = field(default_factory=tuple)  # ER-DMEG-* family codes

    def __post_init__(self) -> None:
        if self.value == EvidenceReviewStatus.ER1:
            # INV-12: ER1 when no DMEG exists is invalid.
            if not self.dmeg_refs:
                raise ValueError(
                    "INV-12 violation: EvidenceReviewResult.value == ER1 requires "
                    "at least one dmeg_ref. ER1 is never triggered merely by a "
                    "blank field, an unresolved Supporting dimension with no "
                    "active trigger, or general desirability of more information "
                    "(CHDM v0.1 §11)."
                )
            if not self.reason_codes:
                raise ValueError(
                    "ER1 must carry at least one evidence-review reason code (§13.2)."
                )
        else:
            # INV-13: ER0 while one or more DMEGs exist is invalid.
            if self.dmeg_refs:
                raise ValueError(
                    "INV-13 violation: EvidenceReviewResult.value == ER0 but "
                    "dmeg_refs is non-empty. ER0 requires that NO DMEG exists "
                    "(CHDM v0.1 §11)."
                )
