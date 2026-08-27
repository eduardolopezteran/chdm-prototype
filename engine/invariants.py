"""
CHDM v0.1 §15 — evaluation-time (cross-object) invariant checks.

Object-construction-time invariants (INV-03, INV-04/05, INV-07, INV-11,
INV-12, INV-13, INV-15) are already enforced inside the domain
dataclasses themselves (fail-fast at creation). This module enforces the
invariants that span TWO OR MORE separately-constructed governed objects
and therefore cannot be checked by any single object's __post_init__ —
principally INV-14 (OPU + ER0), per Technical Architecture §20's
three-layer validation design (creation-time / evaluation-time /
serialization-time).
"""

from __future__ import annotations

from domain.enums import OperationalPriority, EvidenceReviewStatus, DimensionStateValue
from domain.evidence_review import EvidenceReviewResult
from domain.operational_priority import OperationalPriorityResult
from domain.dimension_state import DimensionState
from .errors import InvariantViolationError


def check_governed_output(
    op_result: OperationalPriorityResult,
    er_result: EvidenceReviewResult,
    dimension_states: tuple[DimensionState, ...] = (),
) -> None:
    """Raises InvariantViolationError on any evaluation-time invariant
    violation. Never coerces to a nearby valid state."""

    # INV-14: OPU + ER0 is invalid.
    if op_result.value == OperationalPriority.OPU and er_result.value == EvidenceReviewStatus.ER0:
        raise InvariantViolationError(
            "INV-14",
            "Operational Priority = OPU with Evidence Review = ER0. If priority is "
            "undetermined because unresolved evidence could elevate it, that evidence "
            "satisfies DMEG by definition, which forces ER1 (CHDM v0.1 §11, §15).",
        )

    # INV-16 defense-in-depth: OP2 must never be justified by an ordinary
    # MIXED dimension alone (dimension_engine structurally never does
    # this — CONCERNING is the only dimension state priority_engine
    # reads — but re-verify here since this spans two constructs).
    if op_result.value == OperationalPriority.OP2:
        referenced_mixed_only = any(
            d.state == DimensionStateValue.MIXED and d.dimension.value in op_result.contributing_risk_or_dimension_refs
            for d in dimension_states
        )
        if referenced_mixed_only:
            raise InvariantViolationError(
                "INV-16",
                "OP2 must never be created solely by ordinary Mixed evidence with no "
                "governed confirmed Material/review-triggering condition (CHDM v0.1 §10.1).",
            )

    # INV-19 defense-in-depth: OP value must be one of the four canonical
    # codes (structurally guaranteed by the enum type, re-asserted here
    # as a serialization-time safety net per Technical Architecture §20).
    if op_result.value not in (OperationalPriority.OP1, OperationalPriority.OP2,
                                OperationalPriority.OP3, OperationalPriority.OPU):
        raise InvariantViolationError(
            "INV-19",
            f"Operational Priority value {op_result.value!r} is not a canonical OP1-3/OPU "
            "value. The retired P1/P2/P3/P4 architecture must never be emitted.",
        )
