"""
CHDM v0.1 §2 (lifecycle), §16 (assessment sequence) — Account Assessment
root aggregate. This is the top-level object a Scenario fixture is loaded
into and the deterministic engine evaluates.

Milestone 1 scope note: this object holds INPUT state only (scope,
lifecycle, overlays, objective, evidence). Computed governed outputs
(DimensionState list, RiskRecord list, ObjectiveOutcome,
AssessmentReliability, OperationalPriorityResult, EvidenceReviewResult,
TraceRecords) are returned by engine/evaluate.py as a separate
EvaluationResult — never mutated onto this input object — so "same input
twice -> same output" (FR-10.2 / TAC-01) is structurally obvious rather
than dependent on discipline.
"""

from dataclasses import dataclass, field
from typing import Optional

from .enums import Lifecycle, Overlay
from .evidence import EvidenceObject
from .objective import Objective


@dataclass(frozen=True)
class Scope:
    """CHDM unit of analysis: customer × product/use case (§0 doctrine)."""
    scope_id: str
    customer_identifier: str    # session-scoped / synthetic for Scenario Lab
    use_case_label: str

    def __post_init__(self) -> None:
        if not self.customer_identifier or not self.use_case_label:
            raise ValueError(
                "Scope requires both customer_identifier and use_case_label — "
                "CHDM is undefined without a declared unit of analysis (§0/§2)."
            )


@dataclass(frozen=True)
class AccountAssessment:
    assessment_id: str
    scope: Scope
    lifecycle: Lifecycle                       # must be human-declared, never inferred (§2)
    overlays: dict[Overlay, bool] = field(default_factory=dict)
    objective: Optional[Objective] = None       # None only pending explicit-Unknown declaration
    evidence: tuple[EvidenceObject, ...] = field(default_factory=tuple)
    methodology_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.lifecycle:
            raise ValueError(
                "AccountAssessment.lifecycle is required and must be a deliberate "
                "human declaration — never inferred (CHDM v0.1 §2)."
            )
        # Every overlay must be explicitly Yes/No where referenced; default
        # missing overlays to False (No) rather than leaving them undefined,
        # since Commercial Decision Active specifically has a deterministic
        # promotion effect on D7 (§4.8) that must never be silently skipped.
        for ov in Overlay:
            if ov not in self.overlays:
                object.__setattr__(
                    self, "overlays", {**self.overlays, ov: False}
                )

    def evidence_for_dimension(self, dimension_code) -> tuple[EvidenceObject, ...]:
        return tuple(e for e in self.evidence if e.dimension == dimension_code)
