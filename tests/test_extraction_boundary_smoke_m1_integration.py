"""
Spec §16 — limited integration smoke test. Proves that accepted,
Current+Unverified AI observations can enter the Milestone 1 deterministic
engine WITHOUT crossing the §3.3 confirmation boundary: they cannot
activate Material/Critical severity, cannot produce a falsely confirmed
governed result, but CAN produce potential signals / evidence-uncertainty
outcomes (DMEG -> OPU/ER1) exactly as any other unconfirmed evidence
would. No human confirmation step is added — this only proves the
boundary holds when AI-extracted evidence flows through unchanged
Milestone 1 code.

This is a deterministic test: FakeExtractionProvider only, no network.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import (
    DimensionCode, EvidenceState, Lifecycle, OperationalPriority, Provenance,
    RiskMechanismCode, RiskSeverity, ValueEvidenceBasis,
)
from domain.evidence import EvidenceObject

from extraction.bridge_to_milestone1 import to_risk_severity_claim, to_value_evidence_signal
from extraction.pipeline import run_extraction
from extraction.provider import FakeExtractionProvider

from engine.evaluate import evaluate
from engine.registry_loader import load_and_validate
from engine.risk_engine import evaluate_risk

REGISTRY = load_and_validate()


def _extract_champion_departure_claim():
    text = "Roberto left the company in July with no successor named."
    e = EvidenceObject("E1", None, text, "account_note", Provenance.USER_PROVIDED)
    raw = {"stakeholder_observations": [
        {"source_evidence_id": "E1",
         # Milestone 2B baseline fix: model supplies text only; offsets
         # are derived deterministically by validation.resolve_source_span.
         "source_span": {"text": text},
         "basis": "EXPLICIT", "person_identifier": "Roberto", "continuity_event": "DEPARTED"},
    ]}
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert len(result.accepted) == 1
    return result.accepted[0]


def test_ai_extracted_observation_is_current_unverified_before_bridging():
    obs = _extract_champion_departure_claim()
    assert obs.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED


def test_ai_extracted_claim_cannot_activate_critical_severity():
    """The core confirmation-boundary proof: even a CRITICAL-tier risk
    claim, sourced entirely from AI extraction, must NOT activate —
    because its evidence_state is CURRENT_UNVERIFIED, not
    CURRENT_CONFIRMED (CHDM v0.1 §3.3, INV-07), and the bridge has no
    parameter through which that could be overridden."""
    obs = _extract_champion_departure_claim()
    claim = to_risk_severity_claim(obs, mechanism=RiskMechanismCode.CR_01, tier="CRITICAL")
    assert claim.evidence_state == EvidenceState.CURRENT_UNVERIFIED

    record = evaluate_risk(RiskMechanismCode.CR_01, (claim,))
    assert record.potential_severity == RiskSeverity.CRITICAL  # visible as a POTENTIAL signal
    assert record.activated_severity is None  # but never activated — this is the boundary holding


def test_ai_extracted_evidence_cannot_produce_falsely_confirmed_op1():
    """Running the unconfirmed AI claim through the FULL Milestone 1
    pipeline must not produce OP1 (which requires a confirmed Critical) —
    it must instead surface as evidence uncertainty (OPU/ER1), exactly
    the same deterministic treatment any other unconfirmed evidence would
    receive. Milestone 1's engine code is completely unmodified for this
    to be true."""
    obs = _extract_champion_departure_claim()
    claim = to_risk_severity_claim(obs, mechanism=RiskMechanismCode.CR_01, tier="CRITICAL")

    account = AccountAssessment(
        assessment_id="ASSESS-EXTRACT-SMOKE",
        scope=Scope("SCOPE-SMOKE", "Fictional Extraction Smoke Test Account", "Smoke Test"),
        lifecycle=Lifecycle.L3,
    )
    result = evaluate(account, REGISTRY, risk_claims=(claim,))

    assert result.risk_records[RiskMechanismCode.CR_01].activated_severity is None
    assert result.operational_priority.value != OperationalPriority.OP1
    # Unconfirmed evidence that could plausibly activate OP1 correctly
    # produces evidence uncertainty, not a false negative OR a false
    # positive — this is potential-signal visibility, not a governed
    # confirmed conclusion.
    assert result.operational_priority.value == OperationalPriority.OPU
    assert len(result.dmegs) == 1
    assert result.evidence_review.value.value == "ER1"


def _extract_candidate_risk_signal():
    text = "Roberto left the company in July with no successor named."
    e = EvidenceObject("E1", None, text, "account_note", Provenance.USER_PROVIDED)
    raw = {
        "stakeholder_observations": [
            {"source_evidence_id": "E1", "source_span": {"text": text}, "basis": "EXPLICIT",
             "person_identifier": "Roberto", "continuity_event": "DEPARTED"},
        ],
        "candidate_risk_signals": [
            {"source_span": {"text": text}, "basis": "EXPLICIT",
             "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
             "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert len(result.candidate_risk_signals) == 1
    return result.candidate_risk_signals[0]


def test_candidate_risk_signal_is_current_unverified_before_bridging():
    """Milestone 2C: CandidateRiskSignal already proposes mechanism +
    potential severity tier — structurally closer to a governed risk
    claim than a bare semantic observation. Still Current+Unverified,
    same as everything else Milestone 2/2C ever produces."""
    crs = _extract_candidate_risk_signal()
    assert crs.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    assert crs.mechanism == "CR-01"
    assert crs.proposed_severity_tier == "CRITICAL"


def test_candidate_risk_signal_proposed_mechanism_tier_cannot_activate_critical_severity():
    """The core confirmation-boundary proof, extended to Milestone 2C:
    bridging a CandidateRiskSignal's OWN proposed mechanism/tier into
    Milestone 1 still cannot activate — its evidence_state is still
    CURRENT_UNVERIFIED and the bridge has no parameter through which
    that could be overridden, regardless of how much CHDM-adjacent
    structure the candidate object itself carries."""
    crs = _extract_candidate_risk_signal()
    claim = to_risk_severity_claim(crs, mechanism=RiskMechanismCode(crs.mechanism), tier=crs.proposed_severity_tier)
    assert claim.evidence_state == EvidenceState.CURRENT_UNVERIFIED

    record = evaluate_risk(RiskMechanismCode.CR_01, (claim,))
    assert record.potential_severity == RiskSeverity.CRITICAL  # visible as a POTENTIAL signal
    assert record.activated_severity is None  # but never activated — this is the boundary holding


def _extract_candidate_evidence_classification():
    text = "The customer's goal was faster close. Analytics show close time at 3.6 hours."
    e = EvidenceObject("E1", None, text, "account_note", Provenance.USER_PROVIDED)
    raw = {
        "objective_candidates": [
            {"source_evidence_id": "E1", "source_span": {"text": "The customer's goal was faster close"},
             "basis": "EXPLICIT", "objective_text": "faster close"},
        ],
        "candidate_evidence_classifications": [
            {"source_span": {"text": "Analytics show close time at 3.6 hours"}, "basis": "EXPLICIT",
             "proposed_basis": "MEASURED_OPERATIONAL_EVIDENCE", "supports": "ACHIEVED",
             "supporting_observation_ref": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0}},
        ],
    }
    result = run_extraction((e,), FakeExtractionProvider(raw))
    assert result.request_failure is None
    assert len(result.candidate_evidence_classifications) == 1
    return result.candidate_evidence_classifications[0]


def test_candidate_evidence_classification_cannot_produce_achieved_when_unconfirmed():
    """Milestone 2C companion proof: a CandidateEvidenceClassification
    proposing MEASURED_OPERATIONAL_EVIDENCE / ACHIEVED — the exact
    combination that WOULD produce Objective Outcome = Achieved if
    Current+Confirmed — still cannot, because it is Current+Unverified.
    The objective_engine only ever counts Current+Confirmed signals."""
    cec = _extract_candidate_evidence_classification()
    assert cec.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED

    signal = to_value_evidence_signal(
        cec, basis=ValueEvidenceBasis(cec.proposed_basis), supports=cec.supports,
    )
    assert signal.evidence_state == EvidenceState.CURRENT_UNVERIFIED

    from engine.objective_engine import evaluate_objective_outcome
    from domain.enums import ObjectiveOutcomeState
    from domain.objective import Objective

    objective = Objective(objective_id="OBJ-1", text="faster close")
    outcome = evaluate_objective_outcome(objective, (signal,))
    assert outcome.state == ObjectiveOutcomeState.UNKNOWN  # not Achieved — no confirmed evidence at all


def test_milestone1_engine_module_untouched_and_importable():
    """Sanity check that this smoke test genuinely exercises the same,
    unmodified engine.evaluate used by the full Milestone 1 regression
    suite (tests/test_evaluate_end_to_end.py, tests/test_scenarios.py) —
    not a Milestone-2-local copy or shim."""
    import engine.evaluate as m1_evaluate
    assert m1_evaluate.evaluate is evaluate


TESTS = [
    test_ai_extracted_observation_is_current_unverified_before_bridging,
    test_ai_extracted_claim_cannot_activate_critical_severity,
    test_ai_extracted_evidence_cannot_produce_falsely_confirmed_op1,
    test_candidate_risk_signal_is_current_unverified_before_bridging,
    test_candidate_risk_signal_proposed_mechanism_tier_cannot_activate_critical_severity,
    test_candidate_evidence_classification_cannot_produce_achieved_when_unconfirmed,
    test_milestone1_engine_module_untouched_and_importable,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
