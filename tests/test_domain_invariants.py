"""
Checkpoint 1 smoke tests for domain object construction-time invariant
guards. These are NOT the full CHDM invariant suite (that requires the
engine, built after this checkpoint) — they confirm the dataclasses
themselves refuse to construct the locally-checkable invalid states
(INV-03, INV-04/05, INV-07, INV-11, INV-12, INV-13, INV-15) rather than
silently accepting bad data.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain import (
    DimensionState, DimensionCode, DimensionStateValue, RequirementClass,
    ReasonCode, RiskRecord, RiskMechanismCode, RiskSeverity, EvidenceState,
    ObjectiveOutcome, ObjectiveOutcomeState, ValueEvidenceBasis,
    AssessmentReliability, AssessmentReliabilityLevel,
    OperationalPriorityResult, OperationalPriority,
    EvidenceReviewResult, EvidenceReviewStatus,
)

RC = lambda code="X", obj="CHDM-RULE-TEST-001": ReasonCode(code, obj, "test reason")


def expect_valueerror(fn, label):
    try:
        fn()
        print(f"FAIL  {label}: expected ValueError, none raised")
        return False
    except ValueError:
        print(f"PASS  {label}")
        return True


def expect_ok(fn, label):
    try:
        fn()
        print(f"PASS  {label}")
        return True
    except ValueError as e:
        print(f"FAIL  {label}: unexpected ValueError: {e}")
        return False


results = []

# INV-03: NOT_APPLICABLE without applicability_rule_ref
results.append(expect_valueerror(
    lambda: DimensionState(DimensionCode.D8A, DimensionStateValue.NOT_APPLICABLE,
                            RequirementClass.S, RC()),
    "INV-03: NA without applicability_rule_ref rejected"
))
results.append(expect_ok(
    lambda: DimensionState(DimensionCode.D8A, DimensionStateValue.NOT_APPLICABLE,
                            RequirementClass.NA, RC(), applicability_rule_ref="CHDM-RULE-NA-D8A-001"),
    "NA with applicability_rule_ref accepted"
))

# INV-07: activated severity without Current+Confirmed evidence
results.append(expect_valueerror(
    lambda: RiskRecord(RiskMechanismCode.CR_01, RiskSeverity.CRITICAL, RiskSeverity.CRITICAL,
                        EvidenceState.CURRENT_UNVERIFIED, RC(), contributing_evidence_refs=("E1",)),
    "INV-07: activated Critical on Unverified evidence rejected"
))
results.append(expect_ok(
    lambda: RiskRecord(RiskMechanismCode.CR_01, RiskSeverity.CRITICAL, RiskSeverity.CRITICAL,
                        EvidenceState.CURRENT_CONFIRMED, RC(), contributing_evidence_refs=("E1",)),
    "activated Critical on Confirmed evidence accepted"
))

# INV-04/05: Achieved from Proxy Supported alone
results.append(expect_valueerror(
    lambda: ObjectiveOutcome("OBJ1", ObjectiveOutcomeState.ACHIEVED,
                              (ValueEvidenceBasis.PROXY_SUPPORTED,), RC(),
                              contributing_evidence_refs=("E1",)),
    "INV-04: Achieved from Proxy Supported alone rejected"
))
results.append(expect_ok(
    lambda: ObjectiveOutcome("OBJ1", ObjectiveOutcomeState.ACHIEVED,
                              (ValueEvidenceBasis.MEASURED_OPERATIONAL_EVIDENCE,), RC(),
                              contributing_evidence_refs=("E1",)),
    "Achieved from Measured Operational Evidence accepted"
))
# V-OBJ-03 shape: Progressing from Proxy Supported IS allowed
results.append(expect_ok(
    lambda: ObjectiveOutcome("OBJ1", ObjectiveOutcomeState.PROGRESSING,
                              (ValueEvidenceBasis.PROXY_SUPPORTED,), RC(),
                              contributing_evidence_refs=("E1",)),
    "V-OBJ-03: Progressing from Proxy Supported accepted"
))

# INV-11/REL-01: overall Reliability LOW with no limiting factors (no DMEG)
results.append(expect_valueerror(
    lambda: AssessmentReliability(AssessmentReliabilityLevel.LOW),
    "INV-11/REL-01: Low reliability with no DMEG ref rejected"
))
results.append(expect_ok(
    lambda: AssessmentReliability(AssessmentReliabilityLevel.LOW, limiting_factor_refs=("DMEG-1",)),
    "Low reliability with DMEG ref accepted"
))

# INV-12/13: ER1 without DMEG refs, and ER0 with DMEG refs
results.append(expect_valueerror(
    lambda: EvidenceReviewResult(EvidenceReviewStatus.ER1),
    "INV-12: ER1 without DMEG refs rejected"
))
results.append(expect_valueerror(
    lambda: EvidenceReviewResult(EvidenceReviewStatus.ER0, dmeg_refs=("DMEG-1",)),
    "INV-13: ER0 with DMEG refs rejected"
))
results.append(expect_ok(
    lambda: EvidenceReviewResult(EvidenceReviewStatus.ER1, dmeg_refs=("DMEG-1",),
                                  reason_codes=("ER-DMEG-VALUE",)),
    "ER1 with DMEG refs and reason codes accepted"
))

# INV-15: OP1 without contributing risk refs
results.append(expect_valueerror(
    lambda: OperationalPriorityResult(OperationalPriority.OP1, RC()),
    "INV-15: OP1 without contributing risk refs rejected"
))
results.append(expect_ok(
    lambda: OperationalPriorityResult(OperationalPriority.OP1, RC(),
                                       contributing_risk_or_dimension_refs=("CR-01",)),
    "OP1 with contributing risk ref accepted"
))

print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed")
if not all(results):
    raise SystemExit(1)
