import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.dmeg import DMEG
from domain.enums import (
    AssessmentReliabilityLevel, OperationalPriority, EvidenceReviewStatus,
    DimensionStateValue, DimensionCode, RequirementClass, RiskMechanismCode,
    RiskSeverity, EvidenceState, DMEGLinkedConclusion,
)
from domain.dimension_state import DimensionState
from domain.reason_code import ReasonCode
from domain.risk_record import RiskRecord
from engine.reliability_engine import evaluate_reliability
from engine.priority_engine import evaluate_operational_priority
from engine.review_engine import evaluate_evidence_review
from engine.invariants import check_governed_output
from engine.errors import InvariantViolationError

RC = lambda: ReasonCode("X", "CHDM-RULE-TEST-001", "test")


def mk_dmeg(id_="DMEG-1", reason="ER-DMEG-DIMENSION"):
    """A generic, non-priority-elevating DMEG (linked only to a dimension-
    state conclusion) — used wherever a test just needs *a* DMEG to exist
    (for Reliability/ER1 purposes) without asserting anything about
    Operational Priority."""
    return DMEG(id_, "D1", "LR", frozenset({DMEGLinkedConclusion.DIMENSION_STATE}),
                ("Confirmed", "Rejected"), True, reason)


# ---- Reliability ----

def test_reliability_high_no_dmeg_no_limitations():
    r = evaluate_reliability(())
    assert r.level == AssessmentReliabilityLevel.HIGH


def test_reliability_low_with_dmeg():
    r = evaluate_reliability((mk_dmeg(),))
    assert r.level == AssessmentReliabilityLevel.LOW
    assert r.limiting_factor_refs == ("DMEG-1",)


def test_reliability_medium_non_dmeg_limitation_rel01():
    """REL-01 validation case (§17.6): non-material supporting evidence
    missing, no confirmed Material/Critical concern -> overall must NOT
    be Low."""
    r = evaluate_reliability((), non_dmeg_limitations=("D4 unresolved, non-material",))
    assert r.level == AssessmentReliabilityLevel.MEDIUM


# ---- Operational Priority ----

def risk(mechanism, activated):
    return RiskRecord(mechanism, activated, activated, EvidenceState.CURRENT_CONFIRMED,
                       RC(), contributing_evidence_refs=("E1",))


def test_op1_confirmed_critical():
    op = evaluate_operational_priority((risk(RiskMechanismCode.CR_01, RiskSeverity.CRITICAL),), (), False)
    assert op.value == OperationalPriority.OP1


def test_op2_confirmed_material():
    op = evaluate_operational_priority((risk(RiskMechanismCode.CR_02, RiskSeverity.MATERIAL),), (), False)
    assert op.value == OperationalPriority.OP2


def test_op2_from_concerning_dimension():
    d = DimensionState(DimensionCode.D1, DimensionStateValue.CONCERNING, RequirementClass.LR,
                        RC(), contributing_evidence_refs=("E1",))
    op = evaluate_operational_priority((), (d,), False)
    assert op.value == OperationalPriority.OP2


def test_mixed_dimension_alone_does_not_produce_op2_inv16():
    d = DimensionState(DimensionCode.D1, DimensionStateValue.MIXED, RequirementClass.LR,
                        RC(), contributing_evidence_refs=("E1",))
    op = evaluate_operational_priority((), (d,), False)
    assert op.value != OperationalPriority.OP2


def test_opu_when_priority_elevating_dmeg():
    op = evaluate_operational_priority((), (), True)
    assert op.value == OperationalPriority.OPU


def test_op3_default():
    op = evaluate_operational_priority((), (), False)
    assert op.value == OperationalPriority.OP3


def test_critical_not_suppressed_by_absence_of_concern_elsewhere():
    """Non-compensation doctrine sanity: OP1 wins regardless of what else is passed."""
    d = DimensionState(DimensionCode.D2, DimensionStateValue.SUPPORTED, RequirementClass.UR,
                        RC(), contributing_evidence_refs=("E1",))
    op = evaluate_operational_priority((risk(RiskMechanismCode.CR_01, RiskSeverity.CRITICAL),), (d,), False)
    assert op.value == OperationalPriority.OP1


# ---- Evidence Review ----

def test_er0_no_dmeg():
    er = evaluate_evidence_review(())
    assert er.value == EvidenceReviewStatus.ER0


def test_er1_with_dmeg():
    er = evaluate_evidence_review((mk_dmeg(),))
    assert er.value == EvidenceReviewStatus.ER1
    assert er.dmeg_refs == ("DMEG-1",)


# ---- Cross-object invariants ----

def test_inv14_opu_er0_rejected():
    from domain.operational_priority import OperationalPriorityResult
    op = OperationalPriorityResult(OperationalPriority.OPU, RC())
    er = evaluate_evidence_review(())  # ER0
    try:
        check_governed_output(op, er)
        assert False, "expected InvariantViolationError for OPU+ER0"
    except InvariantViolationError as e:
        assert e.invariant_id == "INV-14"


def test_opu_er1_valid_combo_passes():
    from domain.operational_priority import OperationalPriorityResult
    op = OperationalPriorityResult(OperationalPriority.OPU, RC())
    er = evaluate_evidence_review((mk_dmeg(),))  # ER1
    check_governed_output(op, er)  # should not raise


# ---- Checkpoint 2 refinement: priority-elevating vs. non-priority-elevating
# DMEG (DMEGLinkedConclusion-driven, not len(dmegs)>0). Five focused tests
# required by the executive review.

def mk_priority_elevating_dmeg(id_="DMEG-P1"):
    """A DMEG whose differential test showed the OPERATIONAL_PRIORITY
    outcome itself varies (e.g. an unconfirmed potential-Critical CR-01
    claim: Confirmed -> OP1, Rejected -> OP3)."""
    return DMEG(id_, "CR-01", "CR-triggered",
                frozenset({DMEGLinkedConclusion.RISK_SEVERITY, DMEGLinkedConclusion.OPERATIONAL_PRIORITY}),
                ("Confirmed", "Rejected"), True, "ER-DMEG-RISK-CRITICAL")


def mk_non_priority_elevating_dmeg(id_="DMEG-NP1"):
    """A DMEG that is real (DMEG-1/2/3 all satisfied — it affects a
    dimension-state conclusion) but whose differential test showed the
    OPERATIONAL_PRIORITY outcome does NOT vary — e.g. it affects D6 only,
    with no path to Operational Priority in this account's current state."""
    return DMEG(id_, "D6", "UR", frozenset({DMEGLinkedConclusion.DIMENSION_STATE}),
                ("Confirmed", "Rejected"), True, "ER-DMEG-DIMENSION")


def test_1_no_dmeg_produces_er0():
    """1. No DMEG -> ER0."""
    er = evaluate_evidence_review(())
    assert er.value == EvidenceReviewStatus.ER0
    op = evaluate_operational_priority((), (), False)
    assert op.value == OperationalPriority.OP3


def test_2_priority_elevating_dmeg_no_confirmed_op_gives_opu_and_er1():
    """2. Priority-elevating DMEG, no confirmed OP1/OP2 condition ->
    Operational Priority = Undetermined, Evidence Review = ER1."""
    dmegs = (mk_priority_elevating_dmeg(),)
    has_elevating = any(d.affects_operational_priority for d in dmegs)
    assert has_elevating is True
    op = evaluate_operational_priority((), (), has_elevating)
    er = evaluate_evidence_review(dmegs)
    assert op.value == OperationalPriority.OPU
    assert er.value == EvidenceReviewStatus.ER1


def test_3_non_priority_elevating_dmeg_gives_er1_without_forcing_opu():
    """3. A DMEG that is real (ER1-triggering) but does not affect
    Operational Priority must NOT automatically force OPU."""
    dmegs = (mk_non_priority_elevating_dmeg(),)
    has_elevating = any(d.affects_operational_priority for d in dmegs)
    assert has_elevating is False
    op = evaluate_operational_priority((), (), has_elevating)
    er = evaluate_evidence_review(dmegs)
    assert op.value == OperationalPriority.OP3
    assert op.value != OperationalPriority.OPU
    assert er.value == EvidenceReviewStatus.ER1


def test_4_confirmed_op2_plus_potential_op1_dmeg_stays_op2_with_er1():
    """4. A confirmed OP2 condition (Material risk) already establishes
    Operational Priority; an additional DMEG whose potential resolution
    could reach OP1 does not override the already-confirmed OP2, and ER1
    is still raised for the outstanding DMEG."""
    dmegs = (mk_priority_elevating_dmeg(),)
    has_elevating = any(d.affects_operational_priority for d in dmegs)
    op = evaluate_operational_priority(
        (risk(RiskMechanismCode.CR_02, RiskSeverity.MATERIAL),), (), has_elevating,
    )
    er = evaluate_evidence_review(dmegs)
    assert op.value == OperationalPriority.OP2
    assert er.value == EvidenceReviewStatus.ER1


def test_5_confirmed_op1_plus_additional_dmeg_stays_op1_with_er1():
    """5. A confirmed OP1 condition (Critical risk) is never suppressed by
    an additional outstanding DMEG; ER1 is still raised independently."""
    dmegs = (mk_non_priority_elevating_dmeg(),)
    has_elevating = any(d.affects_operational_priority for d in dmegs)
    op = evaluate_operational_priority(
        (risk(RiskMechanismCode.CR_01, RiskSeverity.CRITICAL),), (), has_elevating,
    )
    er = evaluate_evidence_review(dmegs)
    assert op.value == OperationalPriority.OP1
    assert er.value == EvidenceReviewStatus.ER1


TESTS = [
    test_reliability_high_no_dmeg_no_limitations, test_reliability_low_with_dmeg,
    test_reliability_medium_non_dmeg_limitation_rel01,
    test_op1_confirmed_critical, test_op2_confirmed_material, test_op2_from_concerning_dimension,
    test_mixed_dimension_alone_does_not_produce_op2_inv16, test_opu_when_priority_elevating_dmeg,
    test_op3_default, test_critical_not_suppressed_by_absence_of_concern_elsewhere,
    test_er0_no_dmeg, test_er1_with_dmeg,
    test_inv14_opu_er0_rejected, test_opu_er1_valid_combo_passes,
    test_1_no_dmeg_produces_er0,
    test_2_priority_elevating_dmeg_no_confirmed_op_gives_opu_and_er1,
    test_3_non_priority_elevating_dmeg_gives_er1_without_forcing_opu,
    test_4_confirmed_op2_plus_potential_op1_dmeg_stays_op2_with_er1,
    test_5_confirmed_op1_plus_additional_dmeg_stays_op1_with_er1,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
