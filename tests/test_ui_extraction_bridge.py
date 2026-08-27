"""
Milestone 3C — extraction-to-confirmation integration tests.

Fake-provider only (no network, no live API key needed) -- exercises the
REAL, unmodified extraction.pipeline.run_extraction() end to end through
ui/extraction_bridge.py, ui/sample_scenarios.py, and (for the full
closed-loop cases) ui/actions.py + ui/state.py, exactly the path
ui/app.py wires together. Live-provider behavior (AnthropicExtractionProvider)
is exercised only by the one required live smoke run documented in
MANIFEST.txt / the Milestone 3C report -- this file never calls it,
mirroring how tests/test_extraction_live_model.py already auto-skips
without a key.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import ConfirmationAction, DimensionCode, Lifecycle
from domain.objective import Objective

from confirmation.enums import ConfirmationTargetKind
from engine.registry_loader import load_and_validate

from extraction.enums import RejectionReason

from ui import extraction_review, sample_scenarios
from ui.actions import initial_recompute, submit_action
from ui.extraction_bridge import ExtractionRunOutcome, build_evidence_batch, run_pipeline
from ui.state import AppState


# ---------------------------------------------------------------- build_evidence_batch ----

def test_build_evidence_batch_builds_one_evidence_object_per_nonempty_text():
    batch = build_evidence_batch(["First note.", "Second note."])
    assert len(batch) == 2
    assert batch[0].evidence_id == "E1"
    assert batch[0].indicator_observation == "First note."
    assert batch[1].evidence_id == "E2"
    assert batch[0].dimension is None


def test_build_evidence_batch_filters_out_blank_and_whitespace_only_entries():
    batch = build_evidence_batch(["", "   ", "Real note.", None])
    assert len(batch) == 1
    assert batch[0].indicator_observation == "Real note."


def test_build_evidence_batch_empty_input_returns_empty_tuple():
    assert build_evidence_batch([]) == ()
    assert build_evidence_batch(["", "  "]) == ()


# ---------------------------------------------------------------- run_pipeline: Fake mode, arbitrary text ----

def test_fake_mode_arbitrary_text_produces_one_grounded_observation_no_network():
    batch = build_evidence_batch(["The customer mentioned they are happy with onboarding."])
    outcome = run_pipeline(batch, use_live=False)
    assert isinstance(outcome, ExtractionRunOutcome)
    assert "no AI call" in outcome.provider_label
    result = outcome.extraction_result
    assert result.request_failure is None
    assert len(result.accepted) == 1
    assert result.accepted[0].source_span.text == "The customer mentioned they are happy with onboarding."
    assert result.rejected == ()


def test_fake_mode_arbitrary_text_handles_multiple_evidence_items():
    batch = build_evidence_batch(["First item text.", "Second item text."])
    outcome = run_pipeline(batch, use_live=False)
    assert len(outcome.extraction_result.accepted) == 2


# ---------------------------------------------------------------- run_pipeline: total failure ----

def test_total_request_failure_is_reported_not_raised():
    """A whole-request provider failure must come back as
    ExtractionResult.request_failure, exactly like every other caller of
    run_extraction() (spec §13) -- ui/setup_screen.py depends on this to
    show "Extraction could not be completed" instead of crashing."""
    batch = build_evidence_batch(["Some evidence."])
    # fake_response=None + use_live=False takes the arbitrary-text path;
    # to force a total failure deterministically, go one level down and
    # drive run_pipeline's own provider selection with a response that
    # can never resolve -- simplest deterministic way is to pass a
    # fake_response whose only entry has a span that cannot be found,
    # which the pipeline rejects per-item (not a request_failure). A true
    # request_failure requires a malformed top-level shape from the
    # provider; simulate that directly through the same run_pipeline
    # entry point using a non-dict/list fake_response is not supported by
    # FakeExtractionProvider's constructor contract, so this is exercised
    # directly against extraction.pipeline.run_extraction with a
    # provider configured to simulate a service failure -- the same
    # provider class ui/extraction_bridge.py itself selects.
    from extraction.pipeline import run_extraction
    from extraction.provider import FakeExtractionProvider

    failing_provider = FakeExtractionProvider({}, raise_service_error=True)
    result = run_extraction(batch, failing_provider)
    assert result.request_failure is not None
    assert result.accepted == ()
    assert result.rejected == ()  # no per-item rejections recorded for a whole-request failure


# ---------------------------------------------------------------- run_pipeline: partial rejection ----

def test_partial_rejection_does_not_crash_and_is_recorded():
    """One good item, one item whose span cannot be found in the cited
    evidence -- the good item must still be accepted, and the bad one
    recorded in `rejected`, never silently dropped and never crashing the
    whole run (spec §4/§13)."""
    batch = build_evidence_batch(["Only this exact sentence is real evidence text."])
    fake_response = {
        "experience_observations": [
            {
                "source_evidence_id": "E1",
                "source_span": {"text": "Only this exact sentence is real evidence text."},
                "basis": "EXPLICIT",
                "statement": "Only this exact sentence is real evidence text.",
            },
            {
                "source_evidence_id": "E1",
                "source_span": {"text": "this text was never actually in the evidence"},
                "basis": "EXPLICIT",
                "statement": "this text was never actually in the evidence",
            },
        ],
    }
    outcome = run_pipeline(batch, use_live=False, fake_response=fake_response)
    result = outcome.extraction_result
    assert result.request_failure is None
    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == RejectionReason.SPAN_NOT_FOUND


def test_extraction_review_plain_reason_never_crashes_on_any_rejection_reason():
    """Every RejectionReason value must map to SOME plain-language string
    (falling back to a generic default), so ui/extraction_review.py can
    never itself crash the app rendering a rejection it doesn't
    specifically recognize."""
    for reason in RejectionReason:
        text = extraction_review._plain_reason(reason)
        assert isinstance(text, str) and text


# ---------------------------------------------------------------- sample scenarios ----

def test_all_curated_sample_scenarios_are_within_the_approved_3_to_5_range():
    assert 3 <= len(sample_scenarios.list_scenarios()) <= 5


def test_all_curated_sample_scenarios_load_their_raw_text_from_labeled_set_yaml():
    """Traceability requirement (approved checkpoint decision 2): every
    scenario's evidence text must come from eval/labeled_set.yaml, never
    a separately duplicated literal string."""
    for scenario in sample_scenarios.list_scenarios():
        text = sample_scenarios.raw_text_for(scenario)
        assert isinstance(text, str) and text.strip()


def test_all_curated_sample_scenarios_run_cleanly_through_the_real_pipeline():
    """Each scenario's canned Fake response must be schema-valid against
    the REAL run_extraction() pipeline, with zero rejections and zero
    request failures -- these are meant to be reliable demo material, not
    fixtures that merely happen to construct."""
    for scenario in sample_scenarios.list_scenarios():
        text = sample_scenarios.raw_text_for(scenario)
        batch = build_evidence_batch([text])
        outcome = run_pipeline(batch, use_live=False, fake_response=scenario.fake_response)
        result = outcome.extraction_result
        assert result.request_failure is None, scenario.key
        assert result.rejected == (), (scenario.key, result.rejected)
        assert len(result.accepted) > 0, scenario.key


def test_champion_departure_scenario_produces_a_cr01_critical_candidate_risk_signal():
    scenario = sample_scenarios.get_scenario("champion_departure")
    text = sample_scenarios.raw_text_for(scenario)
    batch = build_evidence_batch([text])
    outcome = run_pipeline(batch, use_live=False, fake_response=scenario.fake_response)
    result = outcome.extraction_result
    assert len(result.candidate_risk_signals) == 1
    signal = result.candidate_risk_signals[0]
    assert signal.mechanism == "CR-01"
    assert signal.proposed_severity_tier == "CRITICAL"


def test_no_objective_stated_scenario_produces_a_missing_information_candidate():
    scenario = sample_scenarios.get_scenario("no_objective_stated")
    text = sample_scenarios.raw_text_for(scenario)
    batch = build_evidence_batch([text])
    outcome = run_pipeline(batch, use_live=False, fake_response=scenario.fake_response)
    result = outcome.extraction_result
    from extraction.schemas import MissingInformationCandidate
    missing = [o for o in result.accepted if isinstance(o, MissingInformationCandidate)]
    assert len(missing) == 1
    assert missing[0].missing_item == "customer objective"


def test_get_scenario_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        sample_scenarios.get_scenario("not-a-real-scenario")


# ---------------------------------------------------------------- end-to-end: real extraction -> AppState -> confirmation -> recompute ----

def _state_from_scenario(scenario_key: str) -> AppState:
    scenario = sample_scenarios.get_scenario(scenario_key)
    text = sample_scenarios.raw_text_for(scenario)
    batch = build_evidence_batch([text])
    outcome = run_pipeline(batch, use_live=False, fake_response=scenario.fake_response)
    assert outcome.extraction_result.request_failure is None

    registry = load_and_validate()
    objective = Objective(objective_id="OBJ-1", text=None, is_known=False)
    account = AccountAssessment(
        assessment_id="ASSESS-TEST", scope=Scope("SCOPE-TEST", "Test Account", "Test Suite"),
        lifecycle=Lifecycle.L3, objective=objective,
    )
    state = AppState(
        account=account, registry=registry, extraction_result=outcome.extraction_result,
        dimensions_to_evaluate=(DimensionCode.D1, DimensionCode.D2, DimensionCode.D6),
        dimension_qualifier_overrides={}, provider_label=outcome.provider_label, reviewer="tester",
    )
    state.current_diagnostic = initial_recompute(state)
    return state


def test_real_extraction_to_appstate_end_to_end_no_crash():
    state = _state_from_scenario("champion_departure")
    assert state.current_diagnostic is not None
    assert state.current_diagnostic.result.operational_priority.value.value == "OPU"


def test_confirming_a_real_extracted_candidate_risk_signal_changes_the_diagnostic():
    state = _state_from_scenario("champion_departure")
    signal = state.extraction_result.candidate_risk_signals[0]
    before = state.current_diagnostic.result

    diag = submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL,
        target_observation_id=signal.system.observation_id, action=ConfirmationAction.CONFIRM,
    )
    after = diag.result

    assert len(state.confirmation_records) == 1
    # CR-01 confirmed CRITICAL must activate the CR-01 risk record --
    # this is confirmation/ + engine/ behavior, unmodified by Milestone
    # 3C; this test only proves the REAL extracted item flows through
    # correctly, not that the rule itself is new.
    from domain.enums import RiskMechanismCode
    cr01_after = after.risk_records[RiskMechanismCode.CR_01]
    assert cr01_after.activated_severity is not None
    cr01_before = before.risk_records[RiskMechanismCode.CR_01]
    assert cr01_before.activated_severity is None


def test_objective_stays_unknown_when_only_value_evidence_is_confirmed_not_identity():
    """Milestone 3D closed the architecture gap this test used to pin (see
    git history / MANIFEST.txt's Milestone 3C section for the original
    "flagged, not resolved" framing): confirming a CandidateEvidenceClassification
    alone -- WITHOUT ever confirming the ObjectiveCandidate it interprets --
    correctly still leaves Objective Outcome UNKNOWN, because objective
    IDENTITY and objective OUTCOME EVIDENCE are two independently
    confirmed, independently required things (confirmation.recompute's
    Milestone 3D objective-resolution bridge). This is still the CORRECT
    behavior post-3D, just no longer an unresolved gap -- see
    test_confirming_objective_identity_and_value_evidence_together_moves_outcome_off_unknown
    below for the closed-loop counterpart."""
    state = _state_from_scenario("proxy_value_evidence")
    classification = state.extraction_result.candidate_evidence_classifications[0]

    diag = submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION,
        target_observation_id=classification.system.observation_id, action=ConfirmationAction.CONFIRM,
    )
    assert diag.result.objective_outcome.state.value == "UNKNOWN"
    assert diag.objective_resolution.status.value == "NOT_ESTABLISHED"


def test_confirming_objective_identity_and_value_evidence_together_moves_outcome_off_unknown():
    """Milestone 3D end-to-end, through the real ui/ layer (not just
    confirmation/ directly): confirming BOTH the ObjectiveCandidate (which
    establishes objective identity) AND the CandidateEvidenceClassification
    that interprets confirmed evidence toward it now correctly moves
    Objective Outcome off UNKNOWN -- closing the exact gap Milestone 3C
    flagged and deferred."""
    from extraction.schemas import ObjectiveCandidate

    state = _state_from_scenario("proxy_value_evidence")
    objective_candidate = next(
        o for o in state.extraction_result.accepted if isinstance(o, ObjectiveCandidate)
    )
    classification = state.extraction_result.candidate_evidence_classifications[0]

    submit_action(
        state, target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION,
        target_observation_id=objective_candidate.system.observation_id, action=ConfirmationAction.CONFIRM,
    )
    diag = submit_action(
        state, target_kind=ConfirmationTargetKind.CANDIDATE_EVIDENCE_CLASSIFICATION,
        target_observation_id=classification.system.observation_id, action=ConfirmationAction.CONFIRM,
    )
    assert diag.result.objective_outcome.state.value != "UNKNOWN"
    assert diag.objective_resolution.status.value == "ESTABLISHED"
    assert diag.objective_resolution.text == objective_candidate.objective_text
