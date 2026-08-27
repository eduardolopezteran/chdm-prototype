"""
CHDM v0.1 §5 — Objective Outcome and Value Evidence Basis.

Deterministic evaluation over a set of ValueEvidenceSignal objects
(already-classified structured claims — see domain/signals.py for why the
engine does not itself interpret raw evidence text).

Precedence, derived directly from §5.4's own conditions (each rule's
condition explicitly excludes the states above it — e.g. Achieved
requires "no current confirmed material evidence contradicts
achievement", which is exactly the Disputed condition; Progressing
requires "no current confirmed evidence establishes material failure",
which is exactly the Not Achieved condition):

  1. DISPUTED         — confirmed evidence for both Achieved and Not Achieved
  2. ACHIEVED          — confirmed direct evidence, uncontradicted
  3. NOT_ACHIEVED      — confirmed evidence of failure, uncontradicted by achievement
  4. PROGRESSING        — confirmed evidence of meaningful progress (proxy allowed — V-OBJ-03),
                           no confirmed failure evidence
  5. NOT_YET_EXPECTED    — explicit `not_yet_expected` declaration, no confirmed failure evidence
  6. UNKNOWN              — default: nothing above resolved

Design note on `not_yet_expected`: CHDM v0.1 §4.2's L1 lifecycle_rule text
("whether outcome realization is reasonably expected") is itself
evaluative, the same way lifecycle assignment or staleness thresholds are
— it is not a crisp formula derivable from Lifecycle alone. Consistent
with lifecycle being a deliberate human declaration (§2), this engine
takes `not_yet_expected` as an explicit input the assessment/fixture
author declares, rather than inferring it from L1/L2 automatically. This
is a scoping choice, not a new methodology rule: CHDM does not specify
the inference and this design does not invent one.
"""

from __future__ import annotations

from domain.enums import ObjectiveOutcomeState, ValueEvidenceBasis
from domain.objective import Objective, ObjectiveOutcome
from domain.reason_code import ReasonCode
from domain.signals import ValueEvidenceSignal
from .evidence_engine import is_current_confirmed

DIRECT_BASES = {
    ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE,
    ValueEvidenceBasis.CUSTOMER_CONFIRMED,
    ValueEvidenceBasis.INDEPENDENTLY_VERIFIED,
}
_DIRECT_BASES = DIRECT_BASES  # backward-compat alias within this module

_REASON_OBJECT = {
    ObjectiveOutcomeState.UNKNOWN: "CHDM-OBJ-OUTCOME-UNKNOWN-001",
    ObjectiveOutcomeState.NOT_YET_EXPECTED: "CHDM-OBJ-OUTCOME-NOTYETEXPECTED-001",
    ObjectiveOutcomeState.PROGRESSING: "CHDM-OBJ-OUTCOME-PROGRESSING-001",
    ObjectiveOutcomeState.ACHIEVED: "CHDM-OBJ-OUTCOME-ACHIEVED-001",
    ObjectiveOutcomeState.NOT_ACHIEVED: "CHDM-OBJ-OUTCOME-NOTACHIEVED-001",
    ObjectiveOutcomeState.DISPUTED: "CHDM-OBJ-OUTCOME-DISPUTED-001",
}

_REASON_TEXT = {
    ObjectiveOutcomeState.UNKNOWN: "Neither progress nor outcome determinable from adequate current evidence (CHDM v0.1 §5.4).",
    ObjectiveOutcomeState.NOT_YET_EXPECTED: "Lifecycle indicates realization not yet reasonably due; no confirmed evidence of material failure (§5.4).",
    ObjectiveOutcomeState.PROGRESSING: "Current + Confirmed evidence demonstrates meaningful progress; achievement not directly established (§5.4, V-OBJ-03).",
    ObjectiveOutcomeState.ACHIEVED: "Current + Confirmed direct evidence supports achievement, uncontradicted (§5.4).",
    ObjectiveOutcomeState.NOT_ACHIEVED: "Current + Confirmed evidence indicates the expected outcome has not occurred (§5.4).",
    ObjectiveOutcomeState.DISPUTED: "Current + Confirmed evidence supports both Achieved and Not Achieved (§5.4, §12.2).",
}


def evaluate_objective_outcome(
    objective: Objective,
    signals: tuple[ValueEvidenceSignal, ...],
    not_yet_expected: bool = False,
) -> ObjectiveOutcome:
    if not objective.is_known:
        return _make(objective, ObjectiveOutcomeState.UNKNOWN, signals, confirmed_only=False)

    confirmed = tuple(s for s in signals if is_current_confirmed(s.evidence_state))
    confirmed_achieved = tuple(s for s in confirmed if s.supports == "ACHIEVED")
    confirmed_not_achieved = tuple(s for s in confirmed if s.supports == "NOT_ACHIEVED")
    confirmed_progressing = tuple(s for s in confirmed if s.supports == "PROGRESSING")

    # 1. DISPUTED
    if confirmed_achieved and confirmed_not_achieved:
        return _make(objective, ObjectiveOutcomeState.DISPUTED, signals, confirmed_only=True)

    # 2. ACHIEVED — requires at least one DIRECT confirmed basis, uncontradicted
    direct_confirmed_achieved = tuple(s for s in confirmed_achieved if s.basis in _DIRECT_BASES)
    if direct_confirmed_achieved and not confirmed_not_achieved:
        return _make(objective, ObjectiveOutcomeState.ACHIEVED, signals, confirmed_only=True)

    # 3. NOT_ACHIEVED — uncontradicted by achievement (already excluded above)
    if confirmed_not_achieved:
        return _make(objective, ObjectiveOutcomeState.NOT_ACHIEVED, signals, confirmed_only=True)

    # 4. PROGRESSING — proxy evidence is explicitly sufficient (V-OBJ-03)
    if confirmed_progressing:
        return _make(objective, ObjectiveOutcomeState.PROGRESSING, signals, confirmed_only=True)

    # 5. NOT_YET_EXPECTED — explicit declaration, no confirmed failure evidence (already excluded)
    if not_yet_expected:
        return _make(objective, ObjectiveOutcomeState.NOT_YET_EXPECTED, signals, confirmed_only=False)

    # 6. UNKNOWN — default
    return _make(objective, ObjectiveOutcomeState.UNKNOWN, signals, confirmed_only=False)


def _make(
    objective: Objective,
    state: ObjectiveOutcomeState,
    signals: tuple[ValueEvidenceSignal, ...],
    confirmed_only: bool,
) -> ObjectiveOutcome:
    relevant = (
        tuple(s for s in signals if is_current_confirmed(s.evidence_state))
        if confirmed_only
        else signals
    )
    bases = tuple(sorted({s.basis for s in relevant}, key=lambda b: b.value)) or (
        (ValueEvidenceBasis.INSUFFICIENT_EVIDENCE,) if not signals else
        tuple(sorted({s.basis for s in signals}, key=lambda b: b.value))
    )
    evidence_refs = tuple(s.evidence_id for s in (relevant or signals)) or (objective.source_evidence_ref,)
    evidence_refs = tuple(r for r in evidence_refs if r) or ("NO_EVIDENCE_CONSIDERED",)

    return ObjectiveOutcome(
        objective_id=objective.objective_id,
        state=state,
        value_evidence_basis=bases,
        reason_code=ReasonCode(
            code=f"OBJ-OUTCOME-{state.value}",
            governing_object_id=_REASON_OBJECT[state],
            human_readable_text=_REASON_TEXT[state],
        ),
        contributing_evidence_refs=evidence_refs,
    )
