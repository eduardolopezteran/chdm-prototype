"""
Milestone 3C — curated sample scenarios ("Sample scenario" evidence-entry
mode, approved checkpoint decision 2A).

Exactly 4 scenarios, deliberately small (approved checkpoint: "do not
expose all 33 evaluation benchmark cases as a developer-facing demo
selector"). Each scenario's RAW EVIDENCE TEXT is loaded directly from
eval/labeled_set.yaml at the case id recorded below -- eval/labeled_set.yaml
remains the ONE authoritative source for this text; it is never
duplicated as a separate literal string in this file (approved
checkpoint: "preserve one authoritative source... rather than creating
uncontrolled duplicate scenario definitions"). Only a short, plain-
language display title/description is authored here -- that framing text
is demo UX, not evidence content, so it does not belong in the benchmark
file.

Each scenario also carries a hand-authored CANNED raw JSON response
(matching extraction/json_schemas.py's TOP_LEVEL_SCHEMA) representing a
correct extraction of that case's text. This is what FakeExtractionProvider
returns for this scenario in "Deterministic demo extraction" mode, so
Sample scenario + Fake still exercises the REAL run_extraction() pipeline
end to end -- schema validation, exact span resolution, dedup,
candidate-classification reference resolution, system-field attachment --
unlike Milestone 3B's ui/sample_data.py, which hand-authored already-
finalized dataclass instances and never called the pipeline at all. In
"Live AI extraction" mode, the SAME raw evidence text is sent to the real
model instead -- the canned response is not used in live mode.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml

_LABELED_SET_PATH = pathlib.Path(__file__).resolve().parent.parent / "eval" / "labeled_set.yaml"


@dataclass(frozen=True)
class SampleScenario:
    key: str
    title: str                # plain, user-facing
    description: str          # plain, user-facing, one line
    source_case_id: str       # traceability only -- surfaced in Technical details, never the primary label
    fake_response: dict       # canned TOP_LEVEL_SCHEMA-conformant response, used only in Fake mode


def _case_text(case_id: str) -> str:
    with open(_LABELED_SET_PATH, encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]
    for case in cases:
        if case["id"] == case_id:
            return case["source_text"].strip()
    raise KeyError(f"No case {case_id!r} in {_LABELED_SET_PATH}")


def _scenarios() -> tuple:
    return (
        SampleScenario(
            key="champion_departure",
            title="Champion departure, no successor named",
            description="A sponsor announces they are leaving with no replacement in sight.",
            source_case_id="24_cr01_sponsor_departure_critical_with_distractor",
            fake_response={
                "stakeholder_observations": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {
                            "text": "Marcus, our only point of contact and the executive sponsor since "
                                    "signing, told us today he is leaving the company at the end of the week",
                        },
                        "basis": "EXPLICIT", "person_identifier": "Marcus",
                    },
                    {
                        "source_evidence_id": "E1",
                        "source_span": {
                            "text": "No successor has been named and we have no other relationship at the account",
                        },
                        "basis": "EXPLICIT", "person_identifier": "Marcus",
                    },
                ],
                "service_observations": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {"text": "a minor billing portal glitch was fixed Tuesday"},
                        "basis": "EXPLICIT",
                        "incident_or_condition": "a minor billing portal glitch was fixed Tuesday",
                    },
                ],
                "candidate_risk_signals": [
                    {
                        "source_span": {
                            "text": "No successor has been named and we have no other relationship at the account",
                        },
                        "basis": "INFERRED_CANDIDATE", "mechanism": "CR-01", "proposed_severity_tier": "CRITICAL",
                        "supporting_observation_ref": {"observation_type": "STAKEHOLDER_OBSERVATION", "index": 1},
                    },
                ],
            },
        ),
        SampleScenario(
            key="service_outage",
            title="Ongoing service disruption",
            description="A core workflow has been failing for days; the customer has a frustrating workaround.",
            source_case_id="25_cr02_service_failure_material",
            fake_response={
                "service_observations": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {
                            "text": "The customer's core reporting workflow has been failing intermittently "
                                    "for six days",
                        },
                        "basis": "EXPLICIT",
                        "incident_or_condition": "The customer's core reporting workflow has been failing "
                                                  "intermittently for six days",
                    },
                ],
                "experience_observations": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {
                            "text": "their team has a manual workaround in place but says it is "
                                    "time-consuming and they are frustrated",
                        },
                        "basis": "EXPLICIT",
                        "statement": "their team has a manual workaround in place but says it is "
                                     "time-consuming and they are frustrated",
                    },
                ],
                "candidate_risk_signals": [
                    {
                        "source_span": {
                            "text": "The customer's core reporting workflow has been failing intermittently "
                                    "for six days",
                        },
                        "basis": "INFERRED_CANDIDATE", "mechanism": "CR-02", "proposed_severity_tier": "MATERIAL",
                        "supporting_observation_ref": {"observation_type": "SERVICE_OBSERVATION", "index": 0},
                    },
                ],
            },
        ),
        SampleScenario(
            key="proxy_value_evidence",
            title="Indirect evidence of progress toward a goal",
            description="Support tickets are trending down toward a stated efficiency goal, though nothing "
                        "is directly measured yet.",
            source_case_id="28_evidence_classification_proxy_supported",
            fake_response={
                "objective_candidates": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {"text": "reduce support burden on their internal IT team"},
                        "basis": "EXPLICIT", "objective_text": "Reduce support burden on their internal IT team",
                    },
                ],
                "service_observations": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {
                            "text": "Support ticket volume from this account has trended down for three "
                                    "consecutive months",
                        },
                        "basis": "EXPLICIT",
                        "incident_or_condition": "Support ticket volume from this account has trended down "
                                                  "for three consecutive months",
                    },
                ],
                "candidate_evidence_classifications": [
                    {
                        "source_span": {
                            "text": "Support ticket volume from this account has trended down for three "
                                    "consecutive months",
                        },
                        "basis": "INFERRED_CANDIDATE", "proposed_basis": "PROXY_SUPPORTED", "supports": "PROGRESSING",
                        "supporting_observation_ref": {"observation_type": "OBJECTIVE_CANDIDATE", "index": 0},
                    },
                ],
            },
        ),
        SampleScenario(
            key="no_objective_stated",
            title="Routine notes with no stated goal",
            description="A status update with no mention of what the customer is trying to achieve.",
            source_case_id="02_unknown_absent_objective",
            fake_response={
                "stakeholder_observations": [
                    {
                        "source_evidence_id": "E1", "source_span": {"text": "customer's ops lead"},
                        "basis": "EXPLICIT", "person_identifier": "ops lead",
                    },
                    {
                        "source_evidence_id": "E1", "source_span": {"text": "two analysts"},
                        "basis": "EXPLICIT", "person_identifier": "two analysts",
                    },
                ],
                "adoption_observations": [
                    {
                        "source_evidence_id": "E1",
                        "source_span": {
                            "text": "We reviewed the dashboard together and answered some configuration questions",
                        },
                        "basis": "EXPLICIT", "workflow_or_use_case": "dashboard review",
                        "observed_behavior": "We reviewed the dashboard together and answered some "
                                              "configuration questions",
                    },
                ],
                "service_observations": [
                    {
                        "source_evidence_id": "E1", "source_span": {"text": "No blockers reported"},
                        "basis": "EXPLICIT", "incident_or_condition": "No blockers reported",
                    },
                ],
                "missing_information_candidates": [
                    {"missing_item": "customer objective", "reviewed_evidence_ids": ["E1"]},
                ],
            },
        ),
    )


def list_scenarios() -> tuple:
    return _scenarios()


def get_scenario(key: str) -> SampleScenario:
    for s in _scenarios():
        if s.key == key:
            return s
    raise KeyError(f"No sample scenario {key!r}")


def raw_text_for(scenario: SampleScenario) -> str:
    return _case_text(scenario.source_case_id)
