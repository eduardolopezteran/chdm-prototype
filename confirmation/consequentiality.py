"""
Milestone 3A — consequentiality.

Reuses engine.dmeg_engine.differs() exactly as engine/evaluate.py's own
internal DMEG differential steps do (see evaluate.py's confirmed-vs-
rejected variant re-run pattern, steps 4a-4c) — this module invents no new
consequentiality mechanism, and does not resolve M3-OD-01. It answers one
narrow question: "does THIS target's current review disposition actually
change the governed outcome, compared to it never having been reviewed at
all?" — exactly the information a human-review queue needs to prioritize
work. Only ever evaluates the paths engine/dimension_engine.py already
supports (IMPLEMENTED_DIMENSIONS = {D1, D2, D6}); a dimension outside that
set is the caller's/evaluate()'s existing responsibility, not something
reintroduced here.
"""
from __future__ import annotations

from typing import Optional, Sequence

from domain.enums import DimensionCode
from engine.dmeg_engine import differs

from .active_evidence import reconstruct_active_evidence
from .recompute import DimensionQualifierOverrides, recompute
from .schemas import ConsequentialityReport


def _outcome_signature(result) -> tuple:
    dims = tuple(sorted(
        (dim.value, state.state.value, state.dimension_reliability)
        for dim, state in result.dimension_states.items()
    ))
    risks = tuple(sorted(
        (
            mech.value,
            rec.potential_severity.value if rec.potential_severity else None,
            rec.activated_severity.value if rec.activated_severity else None,
        )
        for mech, rec in result.risk_records.items()
    ))
    objective = result.objective_outcome.state.value if result.objective_outcome else None
    return (
        result.operational_priority.value.value,
        result.evidence_review.value.value,
        result.reliability.level.value,
        objective,
        dims,
        risks,
    )


def _filtered_overrides(
    overrides: Optional[DimensionQualifierOverrides], active,
) -> Optional[DimensionQualifierOverrides]:
    """Drop any override whose observation_id is not active in THIS
    variant — used for the counterfactual re-run below, where the very
    target under test may have become excluded (e.g. it was REJECTed, so
    the "without review" branch — which drops its record entirely — makes
    it active again as unconfirmed; but a target being tested that is
    itself excluded in the "with review" branch should not raise here,
    since that is an expected, legitimate shape for this specific
    counterfactual comparison, not a caller typo)."""
    if not overrides:
        return overrides
    return {
        observation_id: value
        for observation_id, value in overrides.items()
        if active.by_observation_id(observation_id) is not None
    }


def compute_consequentiality(
    account,
    registry,
    extraction_result,
    confirmation_records: Sequence,
    target_observation_id: str,
    *,
    dimensions_to_evaluate: Sequence[DimensionCode] = (),
    dimension_qualifier_overrides: Optional[DimensionQualifierOverrides] = None,
) -> ConsequentialityReport:
    with_records = tuple(confirmation_records)
    without_records = tuple(
        r for r in confirmation_records if r.target_observation_id != target_observation_id
    )

    active_with = reconstruct_active_evidence(extraction_result, with_records)
    active_without = reconstruct_active_evidence(extraction_result, without_records)

    diag_with = recompute(
        account, registry, active_with,
        dimensions_to_evaluate=dimensions_to_evaluate,
        dimension_qualifier_overrides=_filtered_overrides(dimension_qualifier_overrides, active_with),
    )
    diag_without = recompute(
        account, registry, active_without,
        dimensions_to_evaluate=dimensions_to_evaluate,
        dimension_qualifier_overrides=_filtered_overrides(dimension_qualifier_overrides, active_without),
    )

    sig_with = _outcome_signature(diag_with.result)
    sig_without = _outcome_signature(diag_without.result)
    is_consequential = differs({"with_review": sig_with, "without_review": sig_without})

    return ConsequentialityReport(
        target_observation_id=target_observation_id,
        is_consequential=is_consequential,
        outcome_with_review=sig_with,
        outcome_without_review=sig_without,
    )
