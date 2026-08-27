import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import EvidenceState, RiskMechanismCode, RiskSeverity
from domain.signals import RiskSeverityClaim
from engine.risk_engine import evaluate_risk


def claim(id_, mechanism, state, tier):
    return RiskSeverityClaim(id_, mechanism, id_, state, tier)


def test_s1_confirmed_critical_champion_departure():
    """Scenario S1 core logic: confirmed champion departure, no successor -> activated Critical."""
    claims = (claim("E1", RiskMechanismCode.CR_01, EvidenceState.CURRENT_CONFIRMED, "CRITICAL"),)
    record = evaluate_risk(RiskMechanismCode.CR_01, claims)
    assert record.potential_severity == RiskSeverity.CRITICAL
    assert record.activated_severity == RiskSeverity.CRITICAL
    assert record.evidence_status == EvidenceState.CURRENT_CONFIRMED
    assert record.reason_code is not None
    assert record.contributing_evidence_refs == ("E1",)


def test_s2_unconfirmed_potential_critical_not_activated():
    """Scenario S2 core logic: identical evidence, unconfirmed -> potential
    Critical, Activated None. Non-compensation / confirmation boundary."""
    claims = (claim("E1", RiskMechanismCode.CR_01, EvidenceState.CURRENT_UNVERIFIED, "CRITICAL"),)
    record = evaluate_risk(RiskMechanismCode.CR_01, claims)
    assert record.potential_severity == RiskSeverity.CRITICAL
    assert record.activated_severity is None
    assert record.is_activated is False


def test_no_evidence_no_risk():
    record = evaluate_risk(RiskMechanismCode.CR_02, ())
    assert record.potential_severity is None
    assert record.activated_severity is None
    assert record.reason_code is None


def test_favorable_unrelated_evidence_cannot_suppress_confirmed_critical():
    """Non-compensation doctrine (§1.3): unrelated favorable claims (here,
    simply absent from CR-01's claim set) cannot suppress an activated
    Critical. Since favorable evidence for OTHER mechanisms/dimensions is
    modeled as separate objects entirely (never merged into one score),
    this is structurally guaranteed rather than needing an explicit
    'cancel' test — verified here by confirming a second, lower-tier
    CONFIRMED claim on the SAME mechanism does not downgrade the max."""
    claims = (
        claim("E1", RiskMechanismCode.CR_01, EvidenceState.CURRENT_CONFIRMED, "CRITICAL"),
        claim("E2", RiskMechanismCode.CR_01, EvidenceState.CURRENT_CONFIRMED, "WATCH"),
    )
    record = evaluate_risk(RiskMechanismCode.CR_01, claims)
    assert record.activated_severity == RiskSeverity.CRITICAL


def test_stale_evidence_contributes_potential_not_activated():
    claims = (claim("E1", RiskMechanismCode.CR_03, EvidenceState.STALE, "MATERIAL"),)
    record = evaluate_risk(RiskMechanismCode.CR_03, claims)
    assert record.potential_severity == RiskSeverity.MATERIAL
    assert record.activated_severity is None


def test_unavailable_evidence_does_not_contribute_even_potential():
    claims = (claim("E1", RiskMechanismCode.CR_08, EvidenceState.UNAVAILABLE, "CRITICAL"),)
    record = evaluate_risk(RiskMechanismCode.CR_08, claims)
    assert record.potential_severity is None
    assert record.activated_severity is None


def test_deferred_mechanism_raises():
    try:
        evaluate_risk(RiskMechanismCode.CR_04, ())
        assert False, "expected NotImplementedError for deferred CR-04"
    except NotImplementedError:
        pass


def test_cr06_material_scenario_lab_only():
    """Scenario S6 (material-variant) core logic."""
    claims = (claim("E1", RiskMechanismCode.CR_06, EvidenceState.CURRENT_CONFIRMED, "MATERIAL"),)
    record = evaluate_risk(RiskMechanismCode.CR_06, claims)
    assert record.activated_severity == RiskSeverity.MATERIAL


TESTS = [
    test_s1_confirmed_critical_champion_departure,
    test_s2_unconfirmed_potential_critical_not_activated,
    test_no_evidence_no_risk,
    test_favorable_unrelated_evidence_cannot_suppress_confirmed_critical,
    test_stale_evidence_contributes_potential_not_activated,
    test_unavailable_evidence_does_not_contribute_even_potential,
    test_deferred_mechanism_raises,
    test_cr06_material_scenario_lab_only,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
