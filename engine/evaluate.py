"""
CHDM v0.1 §16 — top-level deterministic evaluate() orchestration.

Wires together, in the amendment's governing dependency order (§16):
  Objective -> Dimensions -> Risks -> DMEG -> Reliability -> Operational
  Priority -> Evidence Review

Pure function: same AccountAssessment + same signal sets => same
EvaluationResult, every time (FR-10.2 / TAC-01). No LLM participates
anywhere in this module.

Refinement (executive correction, post-checkpoint-2): whether a DMEG is
priority-elevating is no longer inferred from DMEG count. Each candidate
DMEG is evaluated per-conclusion via genuine re-computation:
  - RISK_SEVERITY: re-run risk_engine on the Confirmed vs. Rejected
    variant of the pending claim; does activated_severity differ?
  - OPERATIONAL_PRIORITY: re-run evaluate_operational_priority with only
    this one mechanism's risk record substituted (Confirmed vs.
    Rejected), all other current risk/dimension state held fixed, and
    has_priority_elevating_dmeg forced False to avoid self-reference;
    does the OP value differ?
  - OBJECTIVE_D1: for CR-08 specifically (the only implemented mechanism
    D1 consumes directly — CHDM v0.1 §4.2), re-run evaluate_d1 on the
    Confirmed vs. Rejected variant; does the D1 state differ?
  - DIMENSION_STATE (D2-D8B): not reachable from a risk-claim differential
    in Milestone 1's architecture, because the only two implemented
    qualifier-driven dimensions (D2, D6) consume DimensionQualifierSignal,
    not RiskRecord — there is no wired dependency from a risk claim to a
    D2/D6 conclusion to differentiate. Not fabricated; simply out of
    scope for this DMEG source. (A DMEG could still reach DIMENSION_STATE
    via a DimensionQualifierSignal differential, which Milestone 1 does
    not yet generate speculatively — S1-S6 fixtures use directly
    confirmed qualifier evidence.)

`priority_engine.evaluate_operational_priority` and
`review_engine.evaluate_evidence_review` now consume, respectively,
`DMEG.affects_operational_priority` (derived from actual OP-outcome
variance) and DMEG existence (unchanged — ER1 remains "any DMEG",
independent of what it's linked to).

Correction (executive review, Checkpoint 3, S4 re-evaluation): the D1
contradiction rule (§12.2, FULLY_GOVERNED) fixes the CURRENT deterministic
classification when Current+Confirmed evidence conflicts (Objective =
Disputed, D1 = Mixed, D1 dimension_reliability = Low) — but that
classification is not itself the end of the DMEG analysis. §12.1's
general contradiction rule states a contradiction "may create... DMEG...
where its resolution could materially alter a governed conclusion," and
DMEG-2's linked-conclusions list names "objective_outcome_or_D1"
explicitly. A Disputed D1 is therefore evaluated for DMEG-3 exactly like
any other candidate DMEG — by genuine differential re-computation across
the contradiction's own permissible resolutions
(dmeg_rules.yaml `unresolved_contradiction: [resolved_toward_a,
resolved_toward_b]`) — never assumed to be either "always a DMEG" or
"never a DMEG" merely because the rule producing Disputed is fully
governed. See step 4c below. (This does not touch UC-01: the
recomputation only ever re-runs D1's own fully-governed rule under each
resolution — it never invents a D2-D8B contradiction-to-state rule.)
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field

from domain.account_assessment import AccountAssessment
from domain.dimension_state import DimensionState
from domain.dmeg import DMEG
from domain.enums import (
    DimensionCode, RiskMechanismCode, RequirementClass, EvidenceState,
    DMEGLinkedConclusion, ObjectiveOutcomeState,
)
from domain.evidence_review import EvidenceReviewResult
from domain.objective import ObjectiveOutcome
from domain.operational_priority import OperationalPriorityResult
from domain.reliability import AssessmentReliability
from domain.risk_record import RiskRecord
from domain.signals import ValueEvidenceSignal, DimensionQualifierSignal, RiskSeverityClaim
from domain.trace_record import TraceRecord

from . import dimension_engine, objective_engine, risk_engine
from .dmeg_engine import evaluate_differential_dmeg, evaluate_lr_direct_evidence_gap
from .evidence_engine import is_current_confirmed
from .invariants import check_governed_output
from .priority_engine import evaluate_operational_priority
from .reliability_engine import evaluate_reliability
from .review_engine import evaluate_evidence_review
from .trace import build_trace

_DIRECT_BASES = objective_engine.DIRECT_BASES
_dmeg_counter = itertools.count(1)


@dataclass(frozen=True)
class EvaluationResult:
    objective_outcome: ObjectiveOutcome | None
    risk_records: dict[RiskMechanismCode, RiskRecord]
    dimension_states: dict[DimensionCode, DimensionState]
    dmegs: tuple[DMEG, ...]
    reliability: AssessmentReliability
    operational_priority: OperationalPriorityResult
    evidence_review: EvidenceReviewResult
    traces: tuple[TraceRecord, ...]


def evaluate(
    account: AccountAssessment,
    registry,
    value_signals: tuple[ValueEvidenceSignal, ...] = (),
    dimension_signals: tuple[DimensionQualifierSignal, ...] = (),
    risk_claims: tuple[RiskSeverityClaim, ...] = (),
    dimensions_to_evaluate: tuple[DimensionCode, ...] = (),
    not_yet_expected: bool = False,
) -> EvaluationResult:
    lifecycle_matrix = registry["dimensions"]["lifecycle_requirement_matrix"]

    # ---- 1. Objective Outcome ----
    objective_outcome = None
    if account.objective is not None:
        objective_outcome = objective_engine.evaluate_objective_outcome(
            account.objective, value_signals, not_yet_expected=not_yet_expected,
        )

    # ---- 2. Risks (computed before dimensions because D1 needs CR-08) ----
    mechanisms_present = tuple(sorted({c.mechanism for c in risk_claims}, key=lambda m: m.value))
    risk_records: dict[RiskMechanismCode, RiskRecord] = {
        m: risk_engine.evaluate_risk(m, risk_claims) for m in mechanisms_present
    }

    # ---- 3. Dimensions ----
    dimension_states: dict[DimensionCode, DimensionState] = {}
    for dim in dimensions_to_evaluate:
        req_class = _requirement_class_for(lifecycle_matrix, dim, account.lifecycle.value)
        if dim == DimensionCode.D1:
            if objective_outcome is None:
                continue
            dimension_states[dim] = dimension_engine.evaluate_d1(
                objective_outcome, risk_records.get(RiskMechanismCode.CR_08), req_class,
            )
        elif dim == DimensionCode.D2:
            dimension_states[dim] = dimension_engine.evaluate_d2(dimension_signals)
        elif dim == DimensionCode.D6:
            dimension_states[dim] = dimension_engine.evaluate_d6(dimension_signals)
        else:
            raise dimension_engine.NotImplementedForMilestone1(
                f"{dim.value} dimension-state evaluation is not implemented in "
                "Build Milestone 1 (not required by S1-S6)."
            )

    # ---- 4. DMEG ----
    dmegs: list[DMEG] = []

    # 4a. Risk confirmation-pending DMEGs. All severity tiers are
    # considered (not just MATERIAL/CRITICAL) — the differential test
    # itself, not a tier pre-filter, decides materiality: a Watch-tier
    # claim whose confirmation cannot change activated_severity simply
    # produces no outcome variance and therefore no DMEG.
    d1_req_class = None
    if DimensionCode.D1 in dimensions_to_evaluate:
        d1_req_class = _requirement_class_for(lifecycle_matrix, DimensionCode.D1, account.lifecycle.value)

    for mechanism in mechanisms_present:
        claims_for_mechanism = tuple(c for c in risk_claims if c.mechanism == mechanism)
        for claim in claims_for_mechanism:
            if is_current_confirmed(claim.evidence_state):
                continue
            confirmed_variant = tuple(
                dataclasses.replace(c, evidence_state=EvidenceState.CURRENT_CONFIRMED) if c is claim else c
                for c in claims_for_mechanism
            )
            rejected_variant = tuple(c for c in claims_for_mechanism if c is not claim)
            rec_confirmed = risk_engine.evaluate_risk(mechanism, confirmed_variant)
            rec_rejected = risk_engine.evaluate_risk(mechanism, rejected_variant)

            outcomes_by_conclusion: dict[DMEGLinkedConclusion, dict[str, object]] = {
                DMEGLinkedConclusion.RISK_SEVERITY: {
                    "Confirmed": rec_confirmed.activated_severity,
                    "Rejected": rec_rejected.activated_severity,
                },
            }

            # OBJECTIVE_D1 — only CR-08 is wired to a D1 conclusion (§4.2).
            if mechanism == RiskMechanismCode.CR_08 and objective_outcome is not None and d1_req_class is not None:
                d1_confirmed = dimension_engine.evaluate_d1(objective_outcome, rec_confirmed, d1_req_class)
                d1_rejected = dimension_engine.evaluate_d1(objective_outcome, rec_rejected, d1_req_class)
                outcomes_by_conclusion[DMEGLinkedConclusion.OBJECTIVE_D1] = {
                    "Confirmed": d1_confirmed.state,
                    "Rejected": d1_rejected.state,
                }

            # OPERATIONAL_PRIORITY — isolated substitution: swap only this
            # mechanism's risk record in/out of the current risk-record set,
            # hold every other risk record and all dimension states fixed,
            # and force has_priority_elevating_dmeg=False so this probe
            # cannot be contaminated by DMEGs not yet (or already) found.
            other_records = tuple(r for m, r in risk_records.items() if m != mechanism)
            op_confirmed = evaluate_operational_priority(
                other_records + (rec_confirmed,), tuple(dimension_states.values()), False,
            )
            op_rejected = evaluate_operational_priority(
                other_records + (rec_rejected,), tuple(dimension_states.values()), False,
            )
            outcomes_by_conclusion[DMEGLinkedConclusion.OPERATIONAL_PRIORITY] = {
                "Confirmed": op_confirmed.value,
                "Rejected": op_rejected.value,
            }

            dmeg = evaluate_differential_dmeg(
                dmeg_id=f"DMEG-{next(_dmeg_counter):04d}",
                subject_construct_ref=mechanism.value,
                dmeg1_requirement_condition=f"CR-triggered risk mechanism {mechanism.value}",
                outcomes_by_conclusion=outcomes_by_conclusion,
                reason_code=_er_dmeg_reason_for_tier(claim.tier),
            )
            if dmeg:
                dmegs.append(dmeg)

    # 4b. D1 LR direct-evidence gap (REQ-05/REQ-06).
    if DimensionCode.D1 in dimensions_to_evaluate and objective_outcome is not None:
        req_class = _requirement_class_for(lifecycle_matrix, DimensionCode.D1, account.lifecycle.value)
        has_direct_confirmed = any(
            s.basis in _DIRECT_BASES and is_current_confirmed(s.evidence_state)
            for s in value_signals
        )
        dmeg = evaluate_lr_direct_evidence_gap(
            dmeg_id=f"DMEG-{next(_dmeg_counter):04d}",
            dimension=DimensionCode.D1,
            lifecycle=account.lifecycle,
            requirement_class=req_class,
            has_any_direct_confirmed_evidence=has_direct_confirmed,
            objective_outcome_state=objective_outcome.state,
        )
        if dmeg:
            dmegs.append(dmeg)

    # 4c. D1 contradiction (Disputed) DMEG (§12.1 general contradiction
    # DMEG-creation condition, applied to the §12.2 D1 rule). Whether the
    # still-unresolved question of WHICH confirmed source is correct is
    # decision-material is judged by re-running D1's own fully-governed
    # rule under each of the contradiction's two permissible resolutions
    # (the Not-Achieved source is corrected/invalidated, or the Achieved
    # source is corrected/invalidated — §12.1's own named resolution
    # paths) and checking whether the resulting Objective/D1 conclusion,
    # and/or Operational Priority, actually varies.
    if (DimensionCode.D1 in dimensions_to_evaluate and objective_outcome is not None
            and objective_outcome.state == ObjectiveOutcomeState.DISPUTED and d1_req_class is not None):
        confirmed_achieved = tuple(
            s for s in value_signals if is_current_confirmed(s.evidence_state) and s.supports == "ACHIEVED"
        )
        confirmed_not_achieved = tuple(
            s for s in value_signals if is_current_confirmed(s.evidence_state) and s.supports == "NOT_ACHIEVED"
        )
        # resolved_toward_a: the Not-Achieved source turns out to be
        # corrected/invalidated -> only the Achieved side stands.
        resolved_a_signals = tuple(s for s in value_signals if s not in confirmed_not_achieved)
        # resolved_toward_b: the Achieved source turns out to be
        # corrected/invalidated -> only the Not-Achieved side stands.
        resolved_b_signals = tuple(s for s in value_signals if s not in confirmed_achieved)

        outcome_a = objective_engine.evaluate_objective_outcome(
            account.objective, resolved_a_signals, not_yet_expected=not_yet_expected,
        )
        outcome_b = objective_engine.evaluate_objective_outcome(
            account.objective, resolved_b_signals, not_yet_expected=not_yet_expected,
        )
        cr08_risk = risk_records.get(RiskMechanismCode.CR_08)
        d1_a = dimension_engine.evaluate_d1(outcome_a, cr08_risk, d1_req_class)
        d1_b = dimension_engine.evaluate_d1(outcome_b, cr08_risk, d1_req_class)

        outcomes_by_conclusion = {
            DMEGLinkedConclusion.OBJECTIVE_D1: {
                "ResolvedTowardAchieved": d1_a.state,
                "ResolvedTowardNotAchieved": d1_b.state,
            },
        }

        other_dim_states = tuple(s for dim, s in dimension_states.items() if dim != DimensionCode.D1)
        op_a = evaluate_operational_priority(tuple(risk_records.values()), other_dim_states + (d1_a,), False)
        op_b = evaluate_operational_priority(tuple(risk_records.values()), other_dim_states + (d1_b,), False)
        outcomes_by_conclusion[DMEGLinkedConclusion.OPERATIONAL_PRIORITY] = {
            "ResolvedTowardAchieved": op_a.value,
            "ResolvedTowardNotAchieved": op_b.value,
        }

        dmeg = evaluate_differential_dmeg(
            dmeg_id=f"DMEG-{next(_dmeg_counter):04d}",
            subject_construct_ref="D1-CONTRADICTION",
            dmeg1_requirement_condition=f"LR (D1) — Current+Confirmed contradiction (§12.2 d1_contradiction_rule)",
            outcomes_by_conclusion=outcomes_by_conclusion,
            reason_code="ER-DMEG-VALUE",
        )
        if dmeg:
            dmegs.append(dmeg)

    dmegs_t = tuple(dmegs)
    # Priority-elevation is read directly off each DMEG's own
    # DMEG-2 linkage (derived from genuine differential re-computation
    # above, or — for the REQ-06 structural gap — from CHDM's own named
    # consequence). A DMEG that does not actually change the Operational
    # Priority outcome does not elevate it, even though it still triggers
    # ER1 below.
    has_priority_elevating_dmeg = any(d.affects_operational_priority for d in dmegs_t)

    # ---- 5. Reliability ----
    reliability = evaluate_reliability(dmegs_t)

    # ---- 6. Operational Priority ----
    operational_priority = evaluate_operational_priority(
        tuple(risk_records.values()), tuple(dimension_states.values()), has_priority_elevating_dmeg,
    )

    # ---- 7. Evidence Review ----
    evidence_review = evaluate_evidence_review(dmegs_t)

    # ---- 8. Invariants ----
    check_governed_output(operational_priority, evidence_review, tuple(dimension_states.values()))

    # ---- 9. Trace ----
    traces = []
    if objective_outcome is not None:
        traces.append(build_trace(
            f"OBJECTIVE_OUTCOME:{objective_outcome.objective_id}",
            objective_outcome.contributing_evidence_refs, objective_outcome.reason_code,
        ))
    for dim, state in dimension_states.items():
        traces.append(build_trace(f"DIMENSION:{dim.value}", state.contributing_evidence_refs, state.reason_code))
    for mech, rec in risk_records.items():
        if rec.reason_code is not None:
            traces.append(build_trace(f"RISK:{mech.value}", rec.contributing_evidence_refs, rec.reason_code))
    traces.append(build_trace("OPERATIONAL_PRIORITY", operational_priority.contributing_risk_or_dimension_refs or ("N/A",), operational_priority.reason_code))

    return EvaluationResult(
        objective_outcome=objective_outcome,
        risk_records=risk_records,
        dimension_states=dimension_states,
        dmegs=dmegs_t,
        reliability=reliability,
        operational_priority=operational_priority,
        evidence_review=evidence_review,
        traces=tuple(traces),
    )


def _requirement_class_for(lifecycle_matrix: dict, dim: DimensionCode, lifecycle_value: str) -> RequirementClass:
    raw = lifecycle_matrix[dim.value][lifecycle_value]
    # "S_CR" in the registry means Supporting-promotable-to-CR; Milestone 1
    # dimensions (D1 LR-always, D2/D6 UR-always) never hit this branch,
    # but map it defensively rather than crash if it's ever reached.
    mapping = {"UR": RequirementClass.UR, "LR": RequirementClass.LR,
               "CR": RequirementClass.CR, "S": RequirementClass.S,
               "S_CR": RequirementClass.S, "NA": RequirementClass.NA}
    return mapping[raw]


def _er_dmeg_reason_for_tier(tier: str) -> str:
    return "ER-DMEG-RISK-CRITICAL" if tier == "CRITICAL" else "ER-DMEG-RISK-MATERIAL"
