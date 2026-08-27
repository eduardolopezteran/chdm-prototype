"""
CHDM v0.1 §11 — Evidence Review. ER1 iff >=1 DMEG exists; ER0 otherwise.
Structurally consistent with INV-12/INV-13 by construction (dmeg_refs is
always non-empty when ER1, always empty when ER0).
"""

from __future__ import annotations

from domain.dmeg import DMEG
from domain.enums import EvidenceReviewStatus
from domain.evidence_review import EvidenceReviewResult


def evaluate_evidence_review(dmegs: tuple[DMEG, ...]) -> EvidenceReviewResult:
    if dmegs:
        return EvidenceReviewResult(
            value=EvidenceReviewStatus.ER1,
            dmeg_refs=tuple(d.dmeg_id for d in dmegs),
            reason_codes=tuple(sorted({d.reason_code for d in dmegs})),
        )
    return EvidenceReviewResult(value=EvidenceReviewStatus.ER0)
