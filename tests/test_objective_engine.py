import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import EvidenceState, ObjectiveOutcomeState, ValueEvidenceBasis
from domain.objective import Objective
from domain.signals import ValueEvidenceSignal
from engine.objective_engine import evaluate_objective_outcome

OBJ = Objective("OBJ-1", "Reduce reconciliation time by 50%", is_known=True)


def sig(id_, state, basis, supports):
    return ValueEvidenceSignal(id_, id_, state, basis, supports)


def test_achieved_from_measured_confirmed():
    signals = (sig("E1", EvidenceState.CURRENT_CONFIRMED,
                    ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "ACHIEVED"),)
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state == ObjectiveOutcomeState.ACHIEVED
    assert ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE in result.value_evidence_basis


def test_v_obj_03_progressing_from_proxy_only():
    """V-OBJ-03 verbatim: milestones complete + usage strong, no
    demonstrated outcome -> Progressing, not Achieved. This is the exact
    logic Scenario S3 depends on."""
    signals = (sig("E1", EvidenceState.CURRENT_CONFIRMED,
                    ValueEvidenceBasis.PROXY_SUPPORTED, "PROGRESSING"),)
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state == ObjectiveOutcomeState.PROGRESSING
    assert result.value_evidence_basis == (ValueEvidenceBasis.PROXY_SUPPORTED,)


def test_proxy_alone_never_achieves():
    """Even if a proxy signal claims to support ACHIEVED, the engine must
    not grant Achieved from a non-direct basis (INV-04)."""
    signals = (sig("E1", EvidenceState.CURRENT_CONFIRMED,
                    ValueEvidenceBasis.PROXY_SUPPORTED, "ACHIEVED"),)
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state != ObjectiveOutcomeState.ACHIEVED


def test_not_achieved():
    signals = (sig("E1", EvidenceState.CURRENT_CONFIRMED,
                    ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "NOT_ACHIEVED"),)
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state == ObjectiveOutcomeState.NOT_ACHIEVED


def test_disputed_v_obj_04():
    """V-OBJ-04: customer says achieved; operational KPI says opposite ->
    Disputed, both bases retained. This is Scenario S4's core logic."""
    signals = (
        sig("E1", EvidenceState.CURRENT_CONFIRMED, ValueEvidenceBasis.CUSTOMER_CONFIRMED, "ACHIEVED"),
        sig("E2", EvidenceState.CURRENT_CONFIRMED, ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "NOT_ACHIEVED"),
    )
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state == ObjectiveOutcomeState.DISPUTED
    assert ValueEvidenceBasis.CUSTOMER_CONFIRMED in result.value_evidence_basis
    assert ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE in result.value_evidence_basis


def test_unverified_signal_does_not_achieve():
    signals = (sig("E1", EvidenceState.CURRENT_UNVERIFIED,
                    ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "ACHIEVED"),)
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state == ObjectiveOutcomeState.UNKNOWN


def test_unknown_objective():
    unknown_obj = Objective("OBJ-2", None, is_known=False)
    result = evaluate_objective_outcome(unknown_obj, ())
    assert result.state == ObjectiveOutcomeState.UNKNOWN


def test_not_yet_expected_explicit_declaration():
    result = evaluate_objective_outcome(OBJ, (), not_yet_expected=True)
    assert result.state == ObjectiveOutcomeState.NOT_YET_EXPECTED


def test_no_confirmed_evidence_defaults_unknown():
    signals = (sig("E1", EvidenceState.STALE, ValueEvidenceBasis.CUSTOMER_CONFIRMED, "ACHIEVED"),)
    result = evaluate_objective_outcome(OBJ, signals)
    assert result.state == ObjectiveOutcomeState.UNKNOWN


TESTS = [
    test_achieved_from_measured_confirmed,
    test_v_obj_03_progressing_from_proxy_only,
    test_proxy_alone_never_achieves,
    test_not_achieved,
    test_disputed_v_obj_04,
    test_unverified_signal_does_not_achieve,
    test_unknown_objective,
    test_not_yet_expected_explicit_declaration,
    test_no_confirmed_evidence_defaults_unknown,
]

if __name__ == "__main__":
    passed = 0
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(TESTS)} passed")
