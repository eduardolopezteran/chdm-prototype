"""
Checkpoint 1 tests: registry loads, validates, and is structurally
consistent with the domain model. No deterministic CHDM evaluation logic
is tested here yet (engine/*_engine.py doesn't exist until after this
checkpoint) — this only proves the foundation the engine will be built on
is complete and internally consistent.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.registry_loader import load_and_validate, RegistryValidationError, REGISTRY_DIR


def test_registry_loads_and_validates():
    registry = load_and_validate()
    assert registry.raw, "registry loaded no files"


def test_all_required_files_present_on_disk():
    from engine.registry_loader import REQUIRED_FILES
    for filename in REQUIRED_FILES:
        path = REGISTRY_DIR / filename
        assert path.exists(), f"missing registry file: {filename}"


def test_uc01_present_and_unresolved():
    registry = load_and_validate()
    uc01 = registry["contradiction_rules"]["unresolved_general_contradiction_rule"]
    assert uc01["id"] == "UC-01"
    assert uc01["status"] == "UNRESOLVED — GOVERNED CLARIFICATION REQUIRED"
    assert uc01["blocks_build_milestone_1"] is False
    assert uc01["canonical_rule_object_id"] is None


def test_d1_contradiction_fully_governed():
    registry = load_and_validate()
    d1_rule = registry["contradiction_rules"]["d1_contradiction_rule"]
    assert d1_rule["status"] == "FULLY_GOVERNED"
    assert d1_rule["deterministic_effect"]["dimension_state"] == "MIXED"
    assert d1_rule["deterministic_effect"]["dimension_reliability"] == "LOW"


def test_cr06_scenario_lab_only():
    registry = load_and_validate()
    status = registry["risk_mechanisms"]["mvp_implementation_status"]
    assert status["CR-06"] == "scenario_lab_only"
    assert status["CR-01"] == "implemented"
    assert status["CR-04"] == "deferred"


def test_s3_progressing_validation_case_present():
    """V-OBJ-03 is CHDM v0.1's own governed validation case resolving the
    S3 Objective Outcome question — must be present verbatim, not
    reconstructed differently by the engine layer."""
    registry = load_and_validate()
    progressing_rule = registry["objective_outcome"]["deterministic_rules"]["PROGRESSING"]
    assert progressing_rule["validation_case"] == "V-OBJ-03"


def test_all_22_invariants_present():
    registry = load_and_validate()
    ids = {i["id"] for i in registry["invariants"]["invariants"]}
    assert ids == {f"INV-{n:02d}" for n in range(1, 23)}


def test_invalid_registry_detected(tmp_path, monkeypatch):
    """Sanity check that validate_registry() actually catches a broken
    registry rather than passing everything unconditionally."""
    import shutil
    from engine.registry_loader import load_registry, validate_registry

    broken_dir = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, broken_dir)

    # Corrupt dimensions.yaml by removing D8B.
    import yaml
    dims_path = broken_dir / "dimensions.yaml"
    doc = yaml.safe_load(dims_path.read_text(encoding="utf-8"))
    doc["dimensions"] = [d for d in doc["dimensions"] if d["code"] != "D8B"]
    dims_path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    registry = load_registry(broken_dir)
    try:
        validate_registry(registry)
        assert False, "expected RegistryValidationError for missing D8B"
    except RegistryValidationError as e:
        assert "D8B" in str(e) or "dimensions.yaml" in str(e)


if __name__ == "__main__":
    import traceback

    tests = [
        test_registry_loads_and_validates,
        test_all_required_files_present_on_disk,
        test_uc01_present_and_unresolved,
        test_d1_contradiction_fully_governed,
        test_cr06_scenario_lab_only,
        test_s3_progressing_validation_case_present,
        test_all_22_invariants_present,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
