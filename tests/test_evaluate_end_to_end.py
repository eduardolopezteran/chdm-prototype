"""
Checkpoint 2 tests: full deterministic pipeline, hand-authored evidence
(not the finalized S1-S6 fixture files — those come after this
checkpoint). Two representative examples mirroring S1 and S2's core
logic, run end-to-end through:
  Objective -> Dimensions -> Risks -> DMEG -> Reliability -> OP -> ER
plus reproducibility and full-trace checks.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import (
    EvidenceState, Lifecycle, ObjectiveOutcomeState, ValueEvidenceBasis,
    DimensionCode, RiskMechanismCode, RiskSeverity, OperationalPriority,
    EvidenceReviewStatus, AssessmentReliabilityLevel,
)
from domain.objective import Objective
from domain.signals import ValueEvidenceSignal, DimensionQualifierSignal, RiskSeverityClaim
from engine.evaluate import evaluate
from engine.registry_loader import load_and_validate

REGISTRY = load_and_validate()

OBJECTIVE = Objective("OBJ-1", "Reduce monthly reconciliation effort by 50%", is_known=True)


def make_account(lifecycle=Lifecycle.L3):
    return AccountAssessment(
        assessment_id="ASSESS-001",
        scope=Scope("SCOPE-1", "Fictional Northwind Robotics", "Reconciliation Suite"),
        lifecycle=lifecycle,
        objective=OBJECTIVE,
    )


def print_result(label, result):
    print(f"\n=== {label} ===")
    if result.objective_outcome:
        print(f"Objective Outcome : {result.objective_outcome.state.value}  "
              f"(basis={[b.value for b in result.objective_outcome.value_evidence_basis]}, "
              f"reason={result.objective_outcome.reason_code.code})")
    for dim, state in result.dimension_states.items():
        print(f"Dimension {dim.value:<4}: {state.state.value:<22} "
              f"reliability={state.dimension_reliability}  reason={state.reason_code.code}")
    for mech, rec in result.risk_records.items():
        print(f"Risk {mech.value:<6}: potential={rec.potential_severity}, "
              f"activated={rec.activated_severity}  reason={rec.reason_code.code if rec.reason_code else None}")
    print(f"DMEGs             : {[(d.dmeg_id, d.subject_construct_ref, d.reason_code) for d in result.dmegs]}")
    print(f"Reliability        : {result.reliability.level.value}  "
          f"limiting={result.reliability.limiting_factor_refs}")
    print(f"Operational Priority: {result.operational_priority.value.value}  "
          f"reason={result.operational_priority.reason_code.code}")
    print(f"Evidence Review    : {result.evidence_review.value.value}  "
          f"reason_codes={result.evidence_review.reason_codes}")
    print(f"Trace records      : {len(result.traces)}")
    for t in result.traces:
        print(f"  - {t.trace_id}: {t.subject_object_ref} <- chain{t.chain} "
              f"[{t.reason_code.governing_object_id}]")


# ---- Example A: S1-shaped — confirmed champion departure, no successor ----

def build_example_a():
    account = make_account()
    value_signals = (
        ValueEvidenceSignal("E-VAL-1", "E-VAL-1", EvidenceState.CURRENT_CONFIRMED,
                             ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "ACHIEVED"),
    )
    dim_signals = (
        DimensionQualifierSignal("E-D6-1", DimensionCode.D6, "E-CHAMPION-1",
                                  EvidenceState.CURRENT_CONFIRMED, "CHAMPION_LOST_NO_SUCCESSOR"),
    )
    risk_claims = (
        RiskSeverityClaim("E-RISK-1", RiskMechanismCode.CR_01, "E-CHAMPION-1",
                           EvidenceState.CURRENT_CONFIRMED, "CRITICAL"),
    )
    return account, value_signals, dim_signals, risk_claims


def test_example_a_s1_shaped_op1_confirmed_critical():
    account, value_signals, dim_signals, risk_claims = build_example_a()
    result = evaluate(
        account, REGISTRY, value_signals=value_signals, dimension_signals=dim_signals,
        risk_claims=risk_claims, dimensions_to_evaluate=(DimensionCode.D1, DimensionCode.D6),
    )
    print_result("Example A (S1-shaped): confirmed champion departure", result)

    assert result.operational_priority.value == OperationalPriority.OP1
    assert result.risk_records[RiskMechanismCode.CR_01].activated_severity == RiskSeverity.CRITICAL
    assert result.dimension_states[DimensionCode.D6].state.value == "CONCERNING"
    assert result.evidence_review.value == EvidenceReviewStatus.ER0  # no unresolved gap
    assert result.reliability.level == AssessmentReliabilityLevel.HIGH
    assert len(result.dmegs) == 0


# ---- Example B: S2-shaped — identical evidence, unconfirmed ----

def build_example_b():
    account = make_account()
    value_signals = (
        ValueEvidenceSignal("E-VAL-1", "E-VAL-1", EvidenceState.CURRENT_CONFIRMED,
                             ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "ACHIEVED"),
    )
    dim_signals = (
        DimensionQualifierSignal("E-D6-1", DimensionCode.D6, "E-CHAMPION-1",
                                  EvidenceState.CURRENT_UNVERIFIED, "CHAMPION_LOST_NO_SUCCESSOR"),
    )
    risk_claims = (
        RiskSeverityClaim("E-RISK-1", RiskMechanismCode.CR_01, "E-CHAMPION-1",
                           EvidenceState.CURRENT_UNVERIFIED, "CRITICAL"),
    )
    return account, value_signals, dim_signals, risk_claims


def test_example_b_s2_shaped_opu_er1():
    account, value_signals, dim_signals, risk_claims = build_example_b()
    result = evaluate(
        account, REGISTRY, value_signals=value_signals, dimension_signals=dim_signals,
        risk_claims=risk_claims, dimensions_to_evaluate=(DimensionCode.D1, DimensionCode.D6),
    )
    print_result("Example B (S2-shaped): unconfirmed champion departure", result)

    assert result.operational_priority.value == OperationalPriority.OPU
    assert result.risk_records[RiskMechanismCode.CR_01].potential_severity == RiskSeverity.CRITICAL
    assert result.risk_records[RiskMechanismCode.CR_01].activated_severity is None
    assert result.dimension_states[DimensionCode.D6].state.value == "INSUFFICIENT_EVIDENCE"
    assert result.evidence_review.value == EvidenceReviewStatus.ER1
    assert result.reliability.level == AssessmentReliabilityLevel.LOW
    assert len(result.dmegs) == 1


def test_a_and_b_differ_by_exactly_one_confirmation_field():
    """S1/S2 must differ by exactly one confirmation-relevant state
    (functional-spec FR-21.1 / BAR-01 TAC-08) — verified programmatically,
    not just by construction."""
    _, va, da, ra = build_example_a()
    _, vb, db, rb = build_example_b()
    assert va == vb  # value evidence identical
    diffs = []
    if da != db:
        diffs.append(("dimension_signal", da, db))
    if ra != rb:
        diffs.append(("risk_claim", ra, rb))
    # Exactly the D6 signal and the CR-01 claim differ, and both differ
    # ONLY in evidence_state (Confirmed -> Unverified) — same underlying claim.
    assert len(diffs) == 2
    d6a, d6b = da[0], db[0]
    assert d6a.qualifier == d6b.qualifier and d6a.evidence_id == d6b.evidence_id
    assert d6a.evidence_state != d6b.evidence_state
    r1a, r1b = ra[0], rb[0]
    assert r1a.tier == r1b.tier and r1a.evidence_id == r1b.evidence_id
    assert r1a.evidence_state != r1b.evidence_state


def test_reproducibility_identical_inputs_identical_outputs():
    """FR-10.2 / TAC-01: re-running on identical inputs must produce
    identical governed outputs."""
    account, value_signals, dim_signals, risk_claims = build_example_a()
    r1 = evaluate(account, REGISTRY, value_signals, dim_signals, risk_claims,
                   (DimensionCode.D1, DimensionCode.D6))
    r2 = evaluate(account, REGISTRY, value_signals, dim_signals, risk_claims,
                   (DimensionCode.D1, DimensionCode.D6))
    assert r1.operational_priority.value == r2.operational_priority.value
    assert r1.evidence_review.value == r2.evidence_review.value
    assert r1.reliability.level == r2.reliability.level
    assert {k: v.state for k, v in r1.dimension_states.items()} == {k: v.state for k, v in r2.dimension_states.items()}
    assert {k: (v.potential_severity, v.activated_severity) for k, v in r1.risk_records.items()} == \
           {k: (v.potential_severity, v.activated_severity) for k, v in r2.risk_records.items()}


def test_every_governed_output_has_reason_code_and_trace():
    account, value_signals, dim_signals, risk_claims = build_example_b()
    result = evaluate(account, REGISTRY, value_signals, dim_signals, risk_claims,
                       (DimensionCode.D1, DimensionCode.D6))
    assert result.objective_outcome.reason_code.governing_object_id
    for state in result.dimension_states.values():
        assert state.reason_code.governing_object_id
    assert result.operational_priority.reason_code.governing_object_id
    assert len(result.traces) >= 1
    for t in result.traces:
        assert t.methodology_version == "0.1"
        assert t.reason_code.governing_object_id


# ---- Example C: S4-shaped — confirmed contradiction is decision-material ----
# Checkpoint 3 correction: distinct account/objective from the S4 fixture,
# exercising engine/evaluate.py step 4c directly, to confirm the D1
# contradiction DMEG mechanism is genuine engine behavior and not
# something only the S4 fixture happens to trigger.

def build_example_c():
    account = AccountAssessment(
        assessment_id="ASSESS-002",
        scope=Scope("SCOPE-2", "Fictional Southgate Analytics", "Forecast Accuracy Rollout"),
        lifecycle=Lifecycle.L3,
        objective=Objective("OBJ-2", "Cut forecast variance below 5%", is_known=True),
    )
    value_signals = (
        ValueEvidenceSignal("E-VAL-C1", "E-VAL-C1", EvidenceState.CURRENT_CONFIRMED,
                             ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "ACHIEVED"),
        ValueEvidenceSignal("E-VAL-C2", "E-VAL-C2", EvidenceState.CURRENT_CONFIRMED,
                             ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE, "NOT_ACHIEVED"),
    )
    return account, value_signals


def test_example_c_s4_shaped_disputed_contradiction_is_decision_material():
    account, value_signals = build_example_c()
    result = evaluate(
        account, REGISTRY, value_signals=value_signals,
        dimensions_to_evaluate=(DimensionCode.D1,),
    )
    print_result("Example C (S4-shaped): confirmed contradicting objective evidence", result)

    assert result.objective_outcome.state == ObjectiveOutcomeState.DISPUTED
    assert result.dimension_states[DimensionCode.D1].state.value == "MIXED"
    assert result.dimension_states[DimensionCode.D1].dimension_reliability == "LOW"
    # Checkpoint 3 correction: this contradiction IS decision-material
    # (resolving it flips D1 between Supported and Concerning, and OP
    # between OP3 and OP2) — so it correctly raises exactly one DMEG,
    # linked to Operational Priority, driving overall Reliability to Low
    # and Operational Priority to Undetermined rather than a bare OP3.
    assert len(result.dmegs) == 1
    assert result.dmegs[0].affects_operational_priority is True
    assert result.reliability.level == AssessmentReliabilityLevel.LOW
    assert result.operational_priority.value == OperationalPriority.OPU
    assert result.evidence_review.value == EvidenceReviewStatus.ER1


TESTS = [
    test_example_a_s1_shaped_op1_confirmed_critical,
    test_example_b_s2_shaped_opu_er1,
    test_a_and_b_differ_by_exactly_one_confirmation_field,
    test_reproducibility_identical_inputs_identical_outputs,
    test_every_governed_output_has_reason_code_and_trace,
    test_example_c_s4_shaped_disputed_contradiction_is_decision_material,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"\nPASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
