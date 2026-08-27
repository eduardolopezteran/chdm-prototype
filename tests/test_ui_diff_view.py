"""
Milestone 3B — ui/diff_view.py tests (pure function, no Streamlit).
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import DimensionCode, EvidenceState, Lifecycle, RiskMechanismCode, ValueEvidenceBasis
from domain.objective import Objective
from domain.signals import DimensionQualifierSignal, RiskSeverityClaim, ValueEvidenceSignal
from engine.evaluate import evaluate
from engine.registry_loader import load_and_validate

from ui.diff_view import diagnostic_diff

REGISTRY = load_and_validate()
OBJECTIVE = Objective("OBJ-1", "Reduce monthly reconciliation effort by 50%", is_known=True)


def make_account():
    return AccountAssessment(
        assessment_id="ASSESS-UI-DIFF-1", scope=Scope("SCOPE-UI-DIFF-1", "Fictional Co", "Suite"),
        lifecycle=Lifecycle.L3, objective=OBJECTIVE,
    )


def _result(evidence_state):
    value_signals = (
        ValueEvidenceSignal("E-VAL-1", "E-VAL-1", EvidenceState.CURRENT_CONFIRMED,
                             ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "ACHIEVED"),
    )
    dim_signals = (
        DimensionQualifierSignal("E-D6-1", DimensionCode.D6, "E-CHAMPION-1", evidence_state,
                                  "CHAMPION_LOST_NO_SUCCESSOR"),
    )
    risk_claims = (
        RiskSeverityClaim("E-RISK-1", RiskMechanismCode.CR_01, "E-CHAMPION-1", evidence_state, "CRITICAL"),
    )
    return evaluate(
        make_account(), REGISTRY, value_signals=value_signals, dimension_signals=dim_signals,
        risk_claims=risk_claims, dimensions_to_evaluate=(DimensionCode.D1, DimensionCode.D6),
    )


def test_before_none_returns_empty_list():
    assert diagnostic_diff(None, _result(EvidenceState.CURRENT_CONFIRMED)) == []


def test_identical_results_produce_no_diff():
    result = _result(EvidenceState.CURRENT_CONFIRMED)
    assert diagnostic_diff(result, result) == []


def test_confirmation_change_produces_op_dimension_and_risk_diffs():
    before = _result(EvidenceState.CURRENT_UNVERIFIED)
    after = _result(EvidenceState.CURRENT_CONFIRMED)
    changes = diagnostic_diff(before, after)
    assert any(c.startswith("Operational Priority") for c in changes)
    assert any(c.startswith("Dimension D6") for c in changes)
    assert any(c.startswith("Risk CR-01") for c in changes)


def test_diff_direction_is_before_arrow_after():
    before = _result(EvidenceState.CURRENT_UNVERIFIED)
    after = _result(EvidenceState.CURRENT_CONFIRMED)
    changes = diagnostic_diff(before, after)
    op_line = next(c for c in changes if c.startswith("Operational Priority"))
    assert "OPU" in op_line and "OP1" in op_line
    assert op_line.index("OPU") < op_line.index("OP1")


TESTS = [
    test_before_none_returns_empty_list,
    test_identical_results_produce_no_diff,
    test_confirmation_change_produces_op_dimension_and_risk_diffs,
    test_diff_direction_is_before_arrow_after,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
