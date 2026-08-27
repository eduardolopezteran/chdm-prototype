"""
Scenario Lab fixture loader (Build Milestone 1).

Pure data-marshalling only — no business logic. This module's only job is
to turn a static YAML scenario fixture (scenarios/S*.yaml) into the same
domain objects any caller of engine.evaluate() would construct by hand,
then hand them to the real engine. It never computes a governed outcome
itself; every governed value in a fixture's `expected:` block is compared
against whatever engine.evaluate() actually returns (tests/test_scenarios.py).

Keeping this separate from the fixtures themselves preserves "no
executable business logic in fixtures" (executive instruction,
Checkpoint 2 authorization): the YAML files contain only declarative
signal/claim data and declared expected outputs; this loader is shared,
scenario-agnostic plumbing identical for all six fixtures.

Correction (Checkpoint 3 executive review): confirmation state
(evidence_state) is authoritative in exactly one place per fixture — the
top-level `evidence:` block, keyed by evidence_id. Every signal/claim
entry (value_evidence_signals, dimension_qualifier_signals,
risk_severity_claims) references an evidence_id and has its
evidence_state DERIVED from that shared block by this loader; a
signal/claim entry must never declare its own evidence_state directly
(enforced below — raises if one does). This is what makes two fixtures
that differ only in one piece of evidence's confirmation status (e.g.
S1/S2) differ at exactly one serialized field, even when multiple
signals/claims (a dimension qualifier AND a risk claim) trace to that
same evidence item — because they now all read confirmation state from
the one place it is declared, rather than each carrying their own copy
that could silently drift out of sync.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import yaml

from domain.account_assessment import AccountAssessment, Scope
from domain.enums import (
    Lifecycle, DimensionCode, EvidenceState, ValueEvidenceBasis,
    RiskMechanismCode,
)
from domain.objective import Objective
from domain.signals import ValueEvidenceSignal, DimensionQualifierSignal, RiskSeverityClaim

SCENARIOS_DIR = pathlib.Path(__file__).resolve().parent


@dataclass(frozen=True)
class ScenarioFixture:
    scenario_id: str
    title: str
    account_description: str
    account: AccountAssessment
    value_signals: tuple[ValueEvidenceSignal, ...]
    dimension_signals: tuple[DimensionQualifierSignal, ...]
    risk_claims: tuple[RiskSeverityClaim, ...]
    dimensions_to_evaluate: tuple[DimensionCode, ...]
    not_yet_expected: bool
    expected: dict[str, Any]
    raw: dict[str, Any]


def _build_evidence_state_index(raw: dict, path: pathlib.Path) -> dict[str, EvidenceState]:
    index: dict[str, EvidenceState] = {}
    for e in raw.get("evidence", ()):
        index[e["evidence_id"]] = EvidenceState(e["evidence_state"])
    return index


def _resolve_evidence_state(entry: dict, evidence_index: dict[str, EvidenceState], path: pathlib.Path) -> EvidenceState:
    if "evidence_state" in entry:
        raise ValueError(
            f"{path.name}: entry {entry.get('signal_id')!r} declares its own evidence_state. "
            "Confirmation state must be declared exactly once, on the fixture's top-level "
            "`evidence:` block (keyed by evidence_id), and derived from there — never "
            "duplicated onto individual signal/claim entries (Checkpoint 3 one-source-of-truth "
            "correction)."
        )
    evidence_id = entry["evidence_id"]
    if evidence_id not in evidence_index:
        raise ValueError(
            f"{path.name}: evidence_id {evidence_id!r} (referenced by signal/claim "
            f"{entry.get('signal_id')!r}) is not declared in this fixture's `evidence:` block."
        )
    return evidence_index[evidence_id]


def load_scenario(path: pathlib.Path) -> ScenarioFixture:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    evidence_index = _build_evidence_state_index(raw, path)
    lifecycle = Lifecycle(raw["lifecycle"])

    objective = None
    if raw.get("objective") is not None:
        obj_raw = raw["objective"]
        objective = Objective(
            objective_id=obj_raw["objective_id"],
            text=obj_raw.get("text"),
            is_known=obj_raw.get("is_known", True),
        )

    account = AccountAssessment(
        assessment_id=raw["scenario_id"],
        scope=Scope(
            scope_id=f"SCOPE-{raw['scenario_id']}",
            customer_identifier=raw.get("customer_identifier", "Fictional Scenario-Lab Account"),
            use_case_label=raw.get("use_case_label", "Scenario Lab"),
        ),
        lifecycle=lifecycle,
        objective=objective,
    )

    value_signals = tuple(
        ValueEvidenceSignal(
            signal_id=s["signal_id"],
            evidence_id=s["evidence_id"],
            evidence_state=_resolve_evidence_state(s, evidence_index, path),
            basis=ValueEvidenceBasis(s["basis"]),
            supports=s["supports"],
        )
        for s in raw.get("value_evidence_signals", ())
    )

    dimension_signals = tuple(
        DimensionQualifierSignal(
            signal_id=s["signal_id"],
            dimension=DimensionCode(s["dimension"]),
            evidence_id=s["evidence_id"],
            evidence_state=_resolve_evidence_state(s, evidence_index, path),
            qualifier=s["qualifier"],
        )
        for s in raw.get("dimension_qualifier_signals", ())
    )

    risk_claims = tuple(
        RiskSeverityClaim(
            signal_id=s["signal_id"],
            mechanism=RiskMechanismCode(s["mechanism"]),
            evidence_id=s["evidence_id"],
            evidence_state=_resolve_evidence_state(s, evidence_index, path),
            tier=s["tier"],
        )
        for s in raw.get("risk_severity_claims", ())
    )

    dimensions_to_evaluate = tuple(DimensionCode(d) for d in raw.get("dimensions_to_evaluate", ()))

    return ScenarioFixture(
        scenario_id=raw["scenario_id"],
        title=raw["title"],
        account_description=raw["account_description"].strip(),
        account=account,
        value_signals=value_signals,
        dimension_signals=dimension_signals,
        risk_claims=risk_claims,
        dimensions_to_evaluate=dimensions_to_evaluate,
        not_yet_expected=raw.get("not_yet_expected", False),
        expected=raw["expected"],
        raw=raw,
    )


def load_all_scenarios() -> tuple[ScenarioFixture, ...]:
    paths = sorted(SCENARIOS_DIR.glob("s*.yaml"))
    return tuple(load_scenario(p) for p in paths)
