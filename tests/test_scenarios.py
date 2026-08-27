"""
CHDM v0.1 Scenario Lab regression suite (Build Milestone 1, S1-S6).

Loads each static YAML fixture (scenarios/s*.yaml — data only, no
executable business logic), runs it through the real deterministic
engine (engine.evaluate.evaluate), and asserts the engine's ACTUAL
governed output against the fixture's DECLARED expected output. No
expected value here is derived from engine behavior after the fact —
every expected value was hand-derived from CHDM v0.1's own governed
rules before being checked against the engine (see the scenario YAML
files' account_description / comments for the methodology citation each
expected value rests on).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import RiskMechanismCode
from engine.evaluate import evaluate
from engine.registry_loader import load_and_validate
from scenarios.loader import load_scenario, SCENARIOS_DIR

REGISTRY = load_and_validate()


def run_scenario(filename):
    fixture = load_scenario(SCENARIOS_DIR / filename)
    result = evaluate(
        fixture.account,
        REGISTRY,
        value_signals=fixture.value_signals,
        dimension_signals=fixture.dimension_signals,
        risk_claims=fixture.risk_claims,
        dimensions_to_evaluate=fixture.dimensions_to_evaluate,
        not_yet_expected=fixture.not_yet_expected,
    )
    return fixture, result


def assert_matches_expected(fixture, result):
    exp = fixture.expected
    label = fixture.scenario_id

    actual_obj_state = result.objective_outcome.state.value if result.objective_outcome else None
    assert actual_obj_state == exp["objective_outcome_state"], (
        f"{label}: objective_outcome_state expected={exp['objective_outcome_state']!r} "
        f"actual={actual_obj_state!r}"
    )

    for dim_str, expected_state in exp.get("dimension_states", {}).items():
        actual_state = result.dimension_states[_dim(dim_str)].state.value
        assert actual_state == expected_state, (
            f"{label}: dimension {dim_str} state expected={expected_state!r} actual={actual_state!r}"
        )

    for dim_str, expected_rel in exp.get("dimension_reliability", {}).items():
        actual_rel = result.dimension_states[_dim(dim_str)].dimension_reliability
        assert actual_rel == expected_rel, (
            f"{label}: dimension {dim_str} dimension_reliability expected={expected_rel!r} actual={actual_rel!r}"
        )

    for mech_str, expected_rec in exp.get("risk_records", {}).items():
        rec = result.risk_records[RiskMechanismCode(mech_str)]
        actual_potential = rec.potential_severity.value if rec.potential_severity else None
        actual_activated = rec.activated_severity.value if rec.activated_severity else None
        assert actual_potential == expected_rec.get("potential_severity"), (
            f"{label}: {mech_str} potential_severity expected={expected_rec.get('potential_severity')!r} "
            f"actual={actual_potential!r}"
        )
        assert actual_activated == expected_rec.get("activated_severity"), (
            f"{label}: {mech_str} activated_severity expected={expected_rec.get('activated_severity')!r} "
            f"actual={actual_activated!r}"
        )

    assert len(result.dmegs) == exp["dmeg_count"], (
        f"{label}: dmeg_count expected={exp['dmeg_count']} actual={len(result.dmegs)}"
    )

    if "dmeg_affects_operational_priority" in exp:
        actual_affects = any(d.affects_operational_priority for d in result.dmegs)
        assert actual_affects == exp["dmeg_affects_operational_priority"], (
            f"{label}: dmeg_affects_operational_priority expected={exp['dmeg_affects_operational_priority']} "
            f"actual={actual_affects}"
        )

    assert result.reliability.level.value == exp["reliability_level"], (
        f"{label}: reliability_level expected={exp['reliability_level']!r} actual={result.reliability.level.value!r}"
    )
    assert result.operational_priority.value.value == exp["operational_priority"], (
        f"{label}: operational_priority expected={exp['operational_priority']!r} "
        f"actual={result.operational_priority.value.value!r}"
    )
    assert result.evidence_review.value.value == exp["evidence_review"], (
        f"{label}: evidence_review expected={exp['evidence_review']!r} actual={result.evidence_review.value.value!r}"
    )


def _dim(dim_str):
    from domain.enums import DimensionCode
    return DimensionCode(dim_str)


def test_s1():
    fixture, result = run_scenario("s1_confirmed_critical_relationship_risk.yaml")
    assert_matches_expected(fixture, result)


def test_s2():
    fixture, result = run_scenario("s2_unconfirmed_potential_critical_relationship_risk.yaml")
    assert_matches_expected(fixture, result)


def test_s3():
    fixture, result = run_scenario("s3_l4_progressing_no_direct_evidence_gap.yaml")
    assert_matches_expected(fixture, result)


def test_s4():
    fixture, result = run_scenario("s4_confirmed_conflicting_objective_evidence.yaml")
    assert_matches_expected(fixture, result)


def test_s5():
    fixture, result = run_scenario("s5_low_activity_healthy_automation.yaml")
    assert_matches_expected(fixture, result)


def test_s6():
    fixture, result = run_scenario("s6_mixed_adoption_with_confirmed_automation_risk.yaml")
    assert_matches_expected(fixture, result)


_ENGINE_INPUT_KEYS = (
    "lifecycle", "not_yet_expected", "objective", "dimensions_to_evaluate",
    "evidence", "value_evidence_signals", "dimension_qualifier_signals", "risk_severity_claims",
)


def _diff_paths(a, b, path=""):
    """Recursive structural diff over the raw (already-parsed) YAML data —
    yields (path, value_a, value_b) for every leaf that differs. Does not
    tolerate reordering (S1/S2 are hand-authored to be structurally
    parallel, so this is a meaningful, not accidental, guarantee)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            yield from _diff_paths(a.get(key, "<MISSING>"), b.get(key, "<MISSING>"), f"{path}.{key}" if path else key)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (f"{path}[len]", len(a), len(b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from _diff_paths(x, y, f"{path}[{i}]")
    else:
        if a != b:
            yield (path, a, b)


def test_s1_s2_differ_at_exactly_one_authoritative_evidence_field():
    """S1/S2 must differ by exactly one confirmation-relevant state
    (functional-spec FR-21.1 / BAR-01 TAC-08). Verified two ways:

    1. Against the RAW serialized YAML (this test) — proves the fixture
       files themselves, not just the loaded objects, differ at exactly
       one path: the `evidence:` block's declared evidence_state for
       E-CHAMPION-1. Since confirmation state is authoritative in that
       one place only (Checkpoint 3 correction — loader.py raises if any
       signal/claim entry tries to declare its own evidence_state), the
       dimension_qualifier_signals and risk_severity_claims blocks are
       now byte-for-byte identical between S1 and S2.
    2. Against the loaded domain objects (test_s1_s2_loaded_signals_share_
       the_derived_evidence_state below) — proves the D6 signal and CR-01
       claim actually pick up that one authoritative value.
    """
    import yaml as _yaml

    with open(SCENARIOS_DIR / "s1_confirmed_critical_relationship_risk.yaml", encoding="utf-8") as f:
        raw1 = _yaml.safe_load(f)
    with open(SCENARIOS_DIR / "s2_unconfirmed_potential_critical_relationship_risk.yaml", encoding="utf-8") as f:
        raw2 = _yaml.safe_load(f)

    scoped1 = {k: raw1.get(k) for k in _ENGINE_INPUT_KEYS}
    scoped2 = {k: raw2.get(k) for k in _ENGINE_INPUT_KEYS}

    diffs = list(_diff_paths(scoped1, scoped2))
    assert diffs == [("evidence[2].evidence_state", "CURRENT_CONFIRMED", "CURRENT_UNVERIFIED")], (
        f"S1/S2 must differ at exactly one authoritative evidence field; actual diff paths: {diffs}"
    )

    # Sanity: that one entry is E-CHAMPION-1, and every other evidence
    # item plus every signal/claim block is untouched.
    assert raw1["evidence"][2]["evidence_id"] == raw2["evidence"][2]["evidence_id"] == "E-CHAMPION-1"
    assert raw1["dimension_qualifier_signals"] == raw2["dimension_qualifier_signals"]
    assert raw1["risk_severity_claims"] == raw2["risk_severity_claims"]
    assert raw1["value_evidence_signals"] == raw2["value_evidence_signals"]


def test_s1_s2_loaded_signals_derive_from_the_one_authoritative_evidence_state():
    """The loaded D6 qualifier signal and CR-01 claim must both change
    together, purely as a downstream consequence of the one `evidence:`
    entry changing — not because they carry independent evidence_state
    fields that happened to both be edited."""
    s1 = load_scenario(SCENARIOS_DIR / "s1_confirmed_critical_relationship_risk.yaml")
    s2 = load_scenario(SCENARIOS_DIR / "s2_unconfirmed_potential_critical_relationship_risk.yaml")

    assert s1.value_signals == s2.value_signals

    from domain.enums import DimensionCode
    d6_s1 = next(s for s in s1.dimension_signals if s.dimension == DimensionCode.D6)
    d6_s2 = next(s for s in s2.dimension_signals if s.dimension == DimensionCode.D6)
    r1_s1 = next(c for c in s1.risk_claims if c.evidence_id == "E-CHAMPION-1")
    r1_s2 = next(c for c in s2.risk_claims if c.evidence_id == "E-CHAMPION-1")

    assert d6_s1.evidence_state == r1_s1.evidence_state  # same evidence item -> same derived state, within S1
    assert d6_s2.evidence_state == r1_s2.evidence_state  # same evidence item -> same derived state, within S2
    assert d6_s1.evidence_state != d6_s2.evidence_state  # the one authoritative field actually changed


def test_loader_rejects_signal_level_evidence_state():
    """Enforce the one-source-of-truth rule at load time: a signal/claim
    entry that declares its own evidence_state must be rejected, not
    silently accepted (which would reopen the drift risk this correction
    closes)."""
    import tempfile
    import yaml as _yaml

    bad = {
        "scenario_id": "S-BAD",
        "title": "invalid fixture",
        "account_description": "test",
        "lifecycle": "L3",
        "not_yet_expected": False,
        "objective": None,
        "dimensions_to_evaluate": [],
        "evidence": [{"evidence_id": "E-1", "evidence_state": "CURRENT_CONFIRMED"}],
        "dimension_qualifier_signals": [
            {"signal_id": "S-1", "dimension": "D2", "evidence_id": "E-1",
             "evidence_state": "CURRENT_CONFIRMED",  # <- not allowed here
             "qualifier": "INTENDED_WORKFLOWS_OPERATING_NORMALLY"},
        ],
        "expected": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump(bad, f)
        temp_path = pathlib.Path(f.name)
    try:
        load_scenario(temp_path)
        assert False, "expected ValueError for signal-level evidence_state"
    except ValueError as e:
        assert "evidence_state" in str(e)
    finally:
        temp_path.unlink()


def test_all_six_scenarios_load_and_evaluate_without_error():
    for fname in [
        "s1_confirmed_critical_relationship_risk.yaml",
        "s2_unconfirmed_potential_critical_relationship_risk.yaml",
        "s3_l4_progressing_no_direct_evidence_gap.yaml",
        "s4_confirmed_conflicting_objective_evidence.yaml",
        "s5_low_activity_healthy_automation.yaml",
        "s6_mixed_adoption_with_confirmed_automation_risk.yaml",
    ]:
        fixture, result = run_scenario(fname)
        assert fname.lower().startswith(fixture.scenario_id.lower() + "_")
        assert result is not None


TESTS = [
    test_s1, test_s2, test_s3, test_s4, test_s5, test_s6,
    test_s1_s2_differ_at_exactly_one_authoritative_evidence_field,
    test_s1_s2_loaded_signals_derive_from_the_one_authoritative_evidence_state,
    test_loader_rejects_signal_level_evidence_state,
    test_all_six_scenarios_load_and_evaluate_without_error,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
