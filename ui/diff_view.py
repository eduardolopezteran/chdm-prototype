"""
Milestone 3B — diagnostic diffing.

Pure function: compares two engine.evaluate.EvaluationResult instances
(both already produced by the real backend recompute() call in
ui/actions.py) and describes exactly what changed, in plain language.
Never recomputes or patches a governed value itself -- this module only
reads the two results it is handed.
"""
from __future__ import annotations

from typing import List, Optional


def diagnostic_diff(before, after) -> List[str]:
    """`before` may be None (first render, nothing to diff against yet) --
    returns an empty list in that case, same as "no change"."""
    if before is None:
        return []
    changes: List[str] = []

    if before.operational_priority.value != after.operational_priority.value:
        changes.append(
            f"Operational Priority: {before.operational_priority.value.value} -> "
            f"{after.operational_priority.value.value} "
            f"({after.operational_priority.reason_code.human_readable_text})"
        )
    if before.evidence_review.value != after.evidence_review.value:
        changes.append(
            f"Evidence Review: {before.evidence_review.value.value} -> "
            f"{after.evidence_review.value.value}"
        )
    if before.reliability.level != after.reliability.level:
        changes.append(f"Reliability: {before.reliability.level.value} -> {after.reliability.level.value}")

    before_obj = before.objective_outcome.state.value if before.objective_outcome else None
    after_obj = after.objective_outcome.state.value if after.objective_outcome else None
    if before_obj != after_obj:
        changes.append(f"Objective Outcome: {before_obj} -> {after_obj}")

    before_dims = {d: s.state.value for d, s in before.dimension_states.items()}
    after_dims = {d: s.state.value for d, s in after.dimension_states.items()}
    for dim in sorted(set(before_dims) | set(after_dims), key=lambda d: d.value):
        b, a = before_dims.get(dim), after_dims.get(dim)
        if b != a:
            changes.append(f"Dimension {dim.value}: {b} -> {a}")

    before_risks = {
        m: (r.potential_severity.value if r.potential_severity else None,
            r.activated_severity.value if r.activated_severity else None)
        for m, r in before.risk_records.items()
    }
    after_risks = {
        m: (r.potential_severity.value if r.potential_severity else None,
            r.activated_severity.value if r.activated_severity else None)
        for m, r in after.risk_records.items()
    }
    for mech in sorted(set(before_risks) | set(after_risks), key=lambda m: m.value):
        b, a = before_risks.get(mech), after_risks.get(mech)
        if b != a:
            changes.append(f"Risk {mech.value}: potential/activated severity {b} -> {a}")

    return changes
