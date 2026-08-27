"""
CHDM v0.1 §10 — Operational Priority.

OP1 > OP2 > OP3 ordinal; OPU deliberately outside that ordering (never a
4th rung — INV-19). Mixed dimensions never contribute to OP2 on their
own (INV-16) — only CONCERNING dimension states are considered here, and
dimension_engine only ever produces CONCERNING from Current+Confirmed
signals, so this is structurally consistent with "Concerning dimension
produced from Current + Confirmed evidence under a governed rule."
"""

from __future__ import annotations

from domain.dimension_state import DimensionState
from domain.enums import DimensionStateValue, OperationalPriority, RiskSeverity
from domain.operational_priority import OperationalPriorityResult
from domain.reason_code import ReasonCode
from domain.risk_record import RiskRecord


def evaluate_operational_priority(
    risk_records: tuple[RiskRecord, ...],
    dimension_states: tuple[DimensionState, ...],
    has_priority_elevating_dmeg: bool,
) -> OperationalPriorityResult:
    confirmed_criticals = tuple(r for r in risk_records if r.activated_severity == RiskSeverity.CRITICAL)
    if confirmed_criticals:
        return OperationalPriorityResult(
            value=OperationalPriority.OP1,
            reason_code=ReasonCode(
                "OP1-CONFIRMED-CRITICAL", "CHDM-RULE-OP-OP1-001",
                "At least one Current + Confirmed Critical risk is active (CHDM v0.1 §10.1). "
                "No favorable evidence may suppress OP1.",
            ),
            contributing_risk_or_dimension_refs=tuple(r.mechanism.value for r in confirmed_criticals),
        )

    confirmed_materials = tuple(r for r in risk_records if r.activated_severity == RiskSeverity.MATERIAL)
    concerning_dims = tuple(d for d in dimension_states if d.state == DimensionStateValue.CONCERNING)
    if confirmed_materials or concerning_dims:
        refs = tuple(r.mechanism.value for r in confirmed_materials) + tuple(d.dimension.value for d in concerning_dims)
        code = "OP2-CONFIRMED-MATERIAL" if confirmed_materials else "OP2-CONFIRMED-DIMENSION-CONCERN"
        return OperationalPriorityResult(
            value=OperationalPriority.OP2,
            reason_code=ReasonCode(
                code, "CHDM-RULE-OP-OP2-001",
                "Activated Material risk, or a Concerning dimension produced from "
                "Current + Confirmed evidence under a governed rule (CHDM v0.1 §10.1).",
            ),
            contributing_risk_or_dimension_refs=refs,
        )

    if has_priority_elevating_dmeg:
        return OperationalPriorityResult(
            value=OperationalPriority.OPU,
            reason_code=ReasonCode(
                "OPU-PRIORITY-DEPENDENCY-UNRESOLVED", "CHDM-RULE-OP-OPU-001",
                "No confirmed OP1/OP2 condition exists and unresolved evidence could "
                "plausibly activate OP1 or OP2 (CHDM v0.1 §10.1). OPU is not a lower "
                "severity than OP2 — it is outside the ordinal sequence (INV-19).",
            ),
        )

    return OperationalPriorityResult(
        value=OperationalPriority.OP3,
        reason_code=ReasonCode(
            "OP3-NO-ESTABLISHED-MATERIAL-CONCERN", "CHDM-RULE-OP-OP3-001",
            "No confirmed OP1/OP2 condition and no priority-elevating DMEG (CHDM v0.1 §10.1).",
        ),
    )
