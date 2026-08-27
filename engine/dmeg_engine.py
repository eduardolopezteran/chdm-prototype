"""
CHDM v0.1 §8 — Decision-Material Evidence Gap.

Implements the Differential Evaluation Primitive (Technical Architecture
§8/§9): given a governed output computed under each state of an
enumerable resolution-state-space, DMEG-3 is satisfied iff at least two
states produce different outputs for AT LEAST ONE governed-conclusion
category. This module does NOT re-run other engine constructs itself —
the caller (engine/evaluate.py) computes each conclusion's outcomes dict
by re-invoking the relevant sub-evaluation once per candidate state; this
module only judges which conclusions differ and assembles the DMEG
record.

Refinement (executive correction, post-checkpoint-2): a DMEG's linkage to
Operational Priority is never assumed from the DMEG's mere existence. It
is determined per-conclusion, exactly like every other linked conclusion
— "did the Operational Priority outcome actually differ under this
DMEG's resolution-state-space?" A DMEG can be real (ER1-triggering)
while affecting only, say, a dimension-state conclusion, with Operational
Priority unaffected.

Two DMEG-raising modes, both explicitly grounded in CHDM v0.1 text:

  1. `evaluate_differential_dmeg` — the general confirmation-pending case.
  2. `evaluate_lr_direct_evidence_gap` — CHDM v0.1's own named validation
     cases REQ-05 and REQ-06 (§17.4), implemented directly. REQ-06's own
     text ("normally OPU absent established concern") is what licenses
     marking this specific, named gap as OPERATIONAL_PRIORITY-linked —
     it is CHDM's own stated consequence, not an inference from DMEG
     count.
"""

from __future__ import annotations

from typing import Hashable

from domain.dmeg import DMEG
from domain.enums import Lifecycle, ObjectiveOutcomeState, RequirementClass, DimensionCode, DMEGLinkedConclusion


def differs(outcomes: dict[str, Hashable]) -> bool:
    """DMEG-3 outcome-variance test for a single conclusion category: do
    at least two of the evaluated resolution states produce a different
    governed output?"""
    if len(outcomes) < 2:
        return False
    return len(set(outcomes.values())) > 1


def evaluate_differential_dmeg(
    dmeg_id: str,
    subject_construct_ref: str,
    dmeg1_requirement_condition: str,
    outcomes_by_conclusion: dict[DMEGLinkedConclusion, dict[str, Hashable]],
    reason_code: str,
) -> DMEG | None:
    """General confirmation-pending DMEG.

    `outcomes_by_conclusion` maps each governed-conclusion category the
    caller checked to its {resolution_state_name: outcome_value} dict —
    already computed by the caller via real re-evaluation (never invented
    here). Only categories that actually differ become part of
    `dmeg2_linked_conclusions`. If NO category differs, this is not a
    DMEG at all (DMEG-3 fails) and None is returned — the gap is real but
    immaterial, so no DMEG is raised even though it might be "useful to
    obtain" (CHDM v0.1 §8).
    """
    linked = frozenset(
        conclusion for conclusion, outcomes in outcomes_by_conclusion.items()
        if differs(outcomes)
    )
    if not linked:
        return None

    # Use the resolution-state-space from whichever conclusion produced
    # the widest evaluated set (they should generally share the same
    # candidate-state names, e.g. "Confirmed"/"Rejected").
    state_space = tuple(next(iter(outcomes_by_conclusion.values())).keys())

    return DMEG(
        dmeg_id=dmeg_id,
        subject_construct_ref=subject_construct_ref,
        dmeg1_requirement_condition=dmeg1_requirement_condition,
        dmeg2_linked_conclusions=linked,
        dmeg3_resolution_state_space=state_space,
        dmeg3_outcomes_differ=True,
        reason_code=reason_code,
    )


def evaluate_lr_direct_evidence_gap(
    dmeg_id: str,
    dimension: DimensionCode,
    lifecycle: Lifecycle,
    requirement_class: RequirementClass,
    has_any_direct_confirmed_evidence: bool,
    objective_outcome_state: ObjectiveOutcomeState,
) -> DMEG | None:
    """CHDM v0.1 §17.4 REQ-05 / REQ-06, implemented directly (not
    generalized beyond the named cases):

    REQ-06 — L4; no current credible value evidence -> DMEG; Low; ER1;
              "normally OPU absent established concern" (i.e. this named
              case is explicitly Objective/D1-linked AND
              Operational-Priority-linked, per CHDM's own text — not an
              inference from DMEG count).
    REQ-05 — L1; realized value unavailable but Not Yet Expected legitimate
              -> NO DMEG from absent final-value evidence.
    """
    if dimension != DimensionCode.D1 or requirement_class != RequirementClass.LR:
        return None
    if has_any_direct_confirmed_evidence:
        return None

    # REQ-05: legitimate Not Yet Expected at L1 explicitly produces NO DMEG.
    if lifecycle == Lifecycle.L1 and objective_outcome_state == ObjectiveOutcomeState.NOT_YET_EXPECTED:
        return None

    # REQ-06: L4 with no current credible value evidence -> DMEG,
    # explicitly linked to Objective/D1 and to Operational Priority
    # (CHDM v0.1 §17.4 REQ-06's own stated consequence).
    if lifecycle == Lifecycle.L4:
        return DMEG(
            dmeg_id=dmeg_id,
            subject_construct_ref=f"{dimension.value}@{lifecycle.value}",
            dmeg1_requirement_condition="LR at L4 (registry/dimensions.yaml D1 lifecycle_rule)",
            dmeg2_linked_conclusions=frozenset({
                DMEGLinkedConclusion.OBJECTIVE_D1,
                DMEGLinkedConclusion.OPERATIONAL_PRIORITY,
            }),
            dmeg3_resolution_state_space=("direct_evidence_confirmed_achieved", "direct_evidence_absent_stays_progressing_or_unknown"),
            dmeg3_outcomes_differ=True,
            reason_code="ER-DMEG-VALUE",
        )

    # L2/L3 direct-evidence absence: CHDM v0.1 does not name an explicit
    # validation case for this combination (only L1/REQ-05 and L4/REQ-06
    # are named). Out of scope for Milestone 1 rather than guessed.
    return None
