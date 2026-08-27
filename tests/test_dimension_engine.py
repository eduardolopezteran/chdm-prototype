import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import (
    EvidenceState, ObjectiveOutcomeState, ValueEvidenceBasis, DimensionStateValue,
    RequirementClass, RiskMechanismCode, RiskSeverity,
)
from domain.objective import ObjectiveOutcome
from domain.reason_code import ReasonCode
from domain.risk_record import RiskRecord
from domain.signals import DimensionQualifierSignal
from engine.dimension_engine import evaluate_d1, evaluate_d2, evaluate_d6
from engine.errors import NotImplementedForMilestone1

RC = lambda: ReasonCode("X", "CHDM-RULE-TEST-001", "test")


def outcome(state, basis=(ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE,)):
    return ObjectiveOutcome("OBJ-1", state, basis, RC(), contributing_evidence_refs=("E1",))


def test_d1_achieved():
    d = evaluate_d1(outcome(ObjectiveOutcomeState.ACHIEVED), None, RequirementClass.LR)
    assert d.state == DimensionStateValue.SUPPORTED


def test_d1_progressing_mixed():
    """S3 core logic: Progressing -> D1 Mixed."""
    d = evaluate_d1(outcome(ObjectiveOutcomeState.PROGRESSING), None, RequirementClass.LR)
    assert d.state == DimensionStateValue.MIXED


def test_d1_unknown_insufficient_evidence():
    d = evaluate_d1(outcome(ObjectiveOutcomeState.UNKNOWN), None, RequirementClass.LR)
    assert d.state == DimensionStateValue.INSUFFICIENT_EVIDENCE


def test_d1_disputed_mixed_low_reliability():
    """S4 core logic: Disputed -> D1 Mixed, D1 Reliability Low (fully governed)."""
    d = evaluate_d1(outcome(ObjectiveOutcomeState.DISPUTED), None, RequirementClass.LR)
    assert d.state == DimensionStateValue.MIXED
    assert d.dimension_reliability == "LOW"


def test_d1_not_achieved_concerning():
    d = evaluate_d1(outcome(ObjectiveOutcomeState.NOT_ACHIEVED), None, RequirementClass.LR)
    assert d.state == DimensionStateValue.CONCERNING


def test_d1_cr08_activated_forces_concerning_even_if_achieved():
    cr08 = RiskRecord(RiskMechanismCode.CR_08, RiskSeverity.MATERIAL, RiskSeverity.MATERIAL,
                       EvidenceState.CURRENT_CONFIRMED, RC(), contributing_evidence_refs=("E2",))
    d = evaluate_d1(outcome(ObjectiveOutcomeState.ACHIEVED), cr08, RequirementClass.LR)
    assert d.state == DimensionStateValue.CONCERNING


def test_d1_not_yet_expected_raises_not_implemented():
    try:
        evaluate_d1(outcome(ObjectiveOutcomeState.NOT_YET_EXPECTED), None, RequirementClass.LR)
        assert False, "expected NotImplementedForMilestone1"
    except NotImplementedForMilestone1:
        pass


def qsig(id_, dim, state, qualifier):
    return DimensionQualifierSignal(id_, dim, id_, state, qualifier)


def test_d2_supported_via_automation_safeguard():
    """Scenario S5 core logic: low human activity + healthy automated
    workflow -> D2 Supported (login count safeguard, §4.3)."""
    from domain.enums import DimensionCode
    signals = (qsig("E1", DimensionCode.D2, EvidenceState.CURRENT_CONFIRMED,
                     "AUTOMATION_RELIABLE_LOW_LOGIN_OK"),)
    d = evaluate_d2(signals)
    assert d.state == DimensionStateValue.SUPPORTED


def test_d2_mixed_narrow_breadth():
    """Scenario S6 core logic: both variants -> D2 Mixed regardless of CR-06."""
    from domain.enums import DimensionCode
    signals = (qsig("E1", DimensionCode.D2, EvidenceState.CURRENT_CONFIRMED,
                     "NARROW_BREADTH_OR_CONCENTRATION"),)
    d = evaluate_d2(signals)
    assert d.state == DimensionStateValue.MIXED


def test_d2_insufficient_evidence_default():
    d = evaluate_d2(())
    assert d.state == DimensionStateValue.INSUFFICIENT_EVIDENCE


def test_d6_concerning_confirmed_champion_lost():
    """Scenario S1 core logic: confirmed champion departure -> D6 Concerning."""
    from domain.enums import DimensionCode
    signals = (qsig("E1", DimensionCode.D6, EvidenceState.CURRENT_CONFIRMED,
                     "CHAMPION_LOST_NO_SUCCESSOR"),)
    d = evaluate_d6(signals)
    assert d.state == DimensionStateValue.CONCERNING


def test_d6_insufficient_evidence_when_unconfirmed():
    """Scenario S2 core logic: same evidence, unconfirmed -> D6 cannot
    resolve to Concerning; falls to Insufficient Evidence."""
    from domain.enums import DimensionCode
    signals = (qsig("E1", DimensionCode.D6, EvidenceState.CURRENT_UNVERIFIED,
                     "CHAMPION_LOST_NO_SUCCESSOR"),)
    d = evaluate_d6(signals)
    assert d.state == DimensionStateValue.INSUFFICIENT_EVIDENCE


TESTS = [
    test_d1_achieved, test_d1_progressing_mixed, test_d1_unknown_insufficient_evidence,
    test_d1_disputed_mixed_low_reliability, test_d1_not_achieved_concerning,
    test_d1_cr08_activated_forces_concerning_even_if_achieved,
    test_d1_not_yet_expected_raises_not_implemented,
    test_d2_supported_via_automation_safeguard, test_d2_mixed_narrow_breadth,
    test_d2_insufficient_evidence_default,
    test_d6_concerning_confirmed_champion_lost, test_d6_insufficient_evidence_when_unconfirmed,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
