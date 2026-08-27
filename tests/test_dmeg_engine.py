import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import Lifecycle, ObjectiveOutcomeState, RequirementClass, DimensionCode, DMEGLinkedConclusion
from engine.dmeg_engine import differs, evaluate_differential_dmeg, evaluate_lr_direct_evidence_gap


def test_differs_true_when_outcomes_vary():
    assert differs({"Confirmed": "OP1", "Rejected": "OP3"}) is True


def test_differs_false_when_outcomes_same():
    assert differs({"Confirmed": "OP3", "Rejected": "OP3"}) is False


def test_differs_false_with_single_state():
    assert differs({"Confirmed": "OP1"}) is False


def test_differential_dmeg_constructed_when_outcomes_differ():
    """S2 core logic: CR-01 confirm-vs-reject differential produces a DMEG,
    and — because the OPERATIONAL_PRIORITY outcome itself varies (OP1 vs
    OP3) — it is correctly linked to OPERATIONAL_PRIORITY."""
    dmeg = evaluate_differential_dmeg(
        "DMEG-CR01-001", "CR-01", "UR (D6)",
        outcomes_by_conclusion={
            DMEGLinkedConclusion.RISK_SEVERITY: {"Confirmed": "CRITICAL", "Rejected": None},
            DMEGLinkedConclusion.OPERATIONAL_PRIORITY: {"Confirmed": "OP1", "Rejected": "OP3"},
        },
        reason_code="ER-DMEG-RISK-CRITICAL",
    )
    assert dmeg is not None
    assert dmeg.dmeg3_outcomes_differ is True
    assert dmeg.reason_code == "ER-DMEG-RISK-CRITICAL"
    assert dmeg.affects_operational_priority is True
    assert DMEGLinkedConclusion.RISK_SEVERITY in dmeg.dmeg2_linked_conclusions


def test_differential_dmeg_non_priority_elevating_when_op_outcome_stable():
    """A DMEG whose RISK_SEVERITY outcome varies but whose OPERATIONAL_PRIORITY
    outcome does not (e.g. an already-confirmed OP1 elsewhere swamps it) must
    be linked to RISK_SEVERITY only — affects_operational_priority is False."""
    dmeg = evaluate_differential_dmeg(
        "DMEG-CR02-001", "CR-02", "CR-triggered",
        outcomes_by_conclusion={
            DMEGLinkedConclusion.RISK_SEVERITY: {"Confirmed": "MATERIAL", "Rejected": None},
            DMEGLinkedConclusion.OPERATIONAL_PRIORITY: {"Confirmed": "OP1", "Rejected": "OP1"},
        },
        reason_code="ER-DMEG-RISK-MATERIAL",
    )
    assert dmeg is not None
    assert dmeg.affects_operational_priority is False
    assert dmeg.dmeg2_linked_conclusions == frozenset({DMEGLinkedConclusion.RISK_SEVERITY})


def test_differential_dmeg_none_when_outcomes_identical():
    dmeg = evaluate_differential_dmeg(
        "DMEG-X", "CR-02", "S_CR",
        outcomes_by_conclusion={
            DMEGLinkedConclusion.DIMENSION_STATE: {"Confirmed": "SUPPORTED", "Rejected": "SUPPORTED"},
        },
        reason_code="ER-DMEG-DIMENSION",
    )
    assert dmeg is None


def test_req06_l4_no_direct_evidence_produces_dmeg():
    """S3 core logic: REQ-06 verbatim."""
    dmeg = evaluate_lr_direct_evidence_gap(
        "DMEG-D1-001", DimensionCode.D1, Lifecycle.L4, RequirementClass.LR,
        has_any_direct_confirmed_evidence=False,
        objective_outcome_state=ObjectiveOutcomeState.PROGRESSING,
    )
    assert dmeg is not None
    assert dmeg.reason_code == "ER-DMEG-VALUE"


def test_req06_no_dmeg_when_direct_evidence_present():
    dmeg = evaluate_lr_direct_evidence_gap(
        "DMEG-D1-002", DimensionCode.D1, Lifecycle.L4, RequirementClass.LR,
        has_any_direct_confirmed_evidence=True,
        objective_outcome_state=ObjectiveOutcomeState.ACHIEVED,
    )
    assert dmeg is None


def test_req05_l1_not_yet_expected_no_dmeg():
    dmeg = evaluate_lr_direct_evidence_gap(
        "DMEG-D1-003", DimensionCode.D1, Lifecycle.L1, RequirementClass.LR,
        has_any_direct_confirmed_evidence=False,
        objective_outcome_state=ObjectiveOutcomeState.NOT_YET_EXPECTED,
    )
    assert dmeg is None


def test_non_d1_or_non_lr_returns_none():
    dmeg = evaluate_lr_direct_evidence_gap(
        "DMEG-X", DimensionCode.D2, Lifecycle.L4, RequirementClass.UR,
        has_any_direct_confirmed_evidence=False,
        objective_outcome_state=ObjectiveOutcomeState.UNKNOWN,
    )
    assert dmeg is None


TESTS = [
    test_differs_true_when_outcomes_vary, test_differs_false_when_outcomes_same,
    test_differs_false_with_single_state,
    test_differential_dmeg_constructed_when_outcomes_differ,
    test_differential_dmeg_non_priority_elevating_when_op_outcome_stable,
    test_differential_dmeg_none_when_outcomes_identical,
    test_req06_l4_no_direct_evidence_produces_dmeg,
    test_req06_no_dmeg_when_direct_evidence_present,
    test_req05_l1_not_yet_expected_no_dmeg,
    test_non_d1_or_non_lr_returns_none,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
