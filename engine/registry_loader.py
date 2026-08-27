"""
Loads and validates the CHDM v0.1 Methodology Registry (registry/*.yaml).

This module performs structural + exhaustiveness validation only — it
does NOT implement any deterministic CHDM evaluation logic. That is
engine/*_engine.py, built in the next milestone step after this
checkpoint.

Validation performed here:
  1. Every registry file loads as valid YAML.
  2. Every file declares methodology_version == "0.1" (version-pinning,
     §14.4 / TAC-11).
  3. Every governed enum in domain/enums.py has a corresponding,
     exhaustive entry in the registry (no dimension, risk mechanism,
     lifecycle stage, objective-outcome state, or invariant silently
     missing).
  4. UC-01 is present and still marked unresolved (a registry edit that
     silently "resolved" it without going through governance would be
     caught here).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

from domain.enums import (
    Lifecycle,
    DimensionCode,
    RiskMechanismCode,
    ObjectiveOutcomeState,
    ValueEvidenceBasis,
    RequirementClass,
)

REGISTRY_DIR = pathlib.Path(__file__).resolve().parent.parent / "registry"

REQUIRED_FILES = [
    "doctrine.yaml",
    "lifecycle.yaml",
    "evidence_states.yaml",
    "requirement_classes.yaml",
    "objective_outcome.yaml",
    "dimensions.yaml",
    "risk_mechanisms.yaml",
    "dmeg_rules.yaml",
    "reliability_rules.yaml",
    "op_er_rules.yaml",
    "contradiction_rules.yaml",
    "reason_codes.yaml",
    "methodology_object_types.yaml",
    "invariants.yaml",
    "assessment_sequence.yaml",
    "validation_cases.yaml",
]

EXPECTED_METHODOLOGY_VERSION = "0.1"


class RegistryValidationError(Exception):
    """Raised when the registry fails structural or exhaustiveness validation."""


@dataclass
class MethodologyRegistry:
    """In-memory, read-only-at-runtime registry (Technical Architecture §11)."""

    raw: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __getitem__(self, filename_stem: str) -> dict[str, Any]:
        return self.raw[filename_stem]


def load_registry(registry_dir: pathlib.Path = REGISTRY_DIR) -> MethodologyRegistry:
    raw: dict[str, dict[str, Any]] = {}
    missing = []
    for filename in REQUIRED_FILES:
        path = registry_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        with path.open("r", encoding="utf-8") as f:
            raw[path.stem] = yaml.safe_load(f)
    if missing:
        raise RegistryValidationError(
            f"Missing required registry files: {missing}"
        )
    return MethodologyRegistry(raw=raw)


def validate_registry(registry: MethodologyRegistry) -> None:
    """Raises RegistryValidationError on any structural or exhaustiveness defect."""
    errors: list[str] = []

    # --- 1. version pinning ---
    for name, doc in registry.raw.items():
        v = doc.get("methodology_version")
        if v != EXPECTED_METHODOLOGY_VERSION:
            errors.append(
                f"{name}.yaml: methodology_version={v!r}, expected "
                f"{EXPECTED_METHODOLOGY_VERSION!r}"
            )

    # --- 2. lifecycle exhaustiveness ---
    lifecycle_codes = {s["code"] for s in registry["lifecycle"]["lifecycle_stages"]}
    expected_lifecycle = {e.value for e in Lifecycle}
    if lifecycle_codes != expected_lifecycle:
        errors.append(
            f"lifecycle.yaml stages {lifecycle_codes} != Lifecycle enum {expected_lifecycle}"
        )

    # --- 3. dimension exhaustiveness ---
    dim_codes = {d["code"] for d in registry["dimensions"]["dimensions"]}
    expected_dims = {e.value for e in DimensionCode}
    if dim_codes != expected_dims:
        errors.append(
            f"dimensions.yaml dimensions {dim_codes} != DimensionCode enum {expected_dims}"
        )
    matrix_dims = set(registry["dimensions"]["lifecycle_requirement_matrix"].keys())
    if matrix_dims != expected_dims:
        errors.append(
            f"dimensions.yaml lifecycle_requirement_matrix keys {matrix_dims} != "
            f"DimensionCode enum {expected_dims}"
        )

    # --- 4. risk mechanism exhaustiveness ---
    risk_codes = {r["code"] for r in registry["risk_mechanisms"]["risk_mechanisms"]}
    expected_risks = {e.value for e in RiskMechanismCode}
    if risk_codes != expected_risks:
        errors.append(
            f"risk_mechanisms.yaml mechanisms {risk_codes} != RiskMechanismCode enum {expected_risks}"
        )
    mvp_status_codes = set(registry["risk_mechanisms"]["mvp_implementation_status"].keys())
    if mvp_status_codes != expected_risks:
        errors.append(
            f"risk_mechanisms.yaml mvp_implementation_status keys {mvp_status_codes} != "
            f"RiskMechanismCode enum {expected_risks}"
        )

    # --- 5. objective outcome / value evidence basis exhaustiveness ---
    outcome_codes = {
        s["code"] for s in registry["objective_outcome"]["objective_outcome_states"]
    }
    expected_outcomes = {e.value for e in ObjectiveOutcomeState}
    if outcome_codes != expected_outcomes:
        errors.append(
            f"objective_outcome.yaml states {outcome_codes} != "
            f"ObjectiveOutcomeState enum {expected_outcomes}"
        )
    basis_codes = {
        b["code"] for b in registry["objective_outcome"]["value_evidence_bases"]
    }
    expected_bases = {e.value for e in ValueEvidenceBasis}
    if basis_codes != expected_bases:
        errors.append(
            f"objective_outcome.yaml value_evidence_bases {basis_codes} != "
            f"ValueEvidenceBasis enum {expected_bases}"
        )
    deterministic_rule_keys = set(registry["objective_outcome"]["deterministic_rules"].keys())
    if deterministic_rule_keys != expected_outcomes:
        errors.append(
            f"objective_outcome.yaml deterministic_rules keys {deterministic_rule_keys} != "
            f"ObjectiveOutcomeState enum {expected_outcomes}"
        )

    # --- 6. requirement classes exhaustiveness ---
    req_codes = {c["code"] for c in registry["requirement_classes"]["requirement_classes"]}
    expected_req = {e.value for e in RequirementClass}
    if req_codes != expected_req:
        errors.append(
            f"requirement_classes.yaml classes {req_codes} != RequirementClass enum {expected_req}"
        )

    # --- 7. invariants: all 22 present ---
    inv_ids = {i["id"] for i in registry["invariants"]["invariants"]}
    expected_inv = {f"INV-{n:02d}" for n in range(1, 23)}
    if inv_ids != expected_inv:
        errors.append(f"invariants.yaml has {inv_ids}, expected {expected_inv}")

    # --- 8. UC-01 must still be present and unresolved ---
    uc01 = registry["contradiction_rules"]["unresolved_general_contradiction_rule"]
    if uc01.get("id") != "UC-01":
        errors.append("contradiction_rules.yaml: UC-01 id missing or altered")
    if uc01.get("status") != "UNRESOLVED — GOVERNED CLARIFICATION REQUIRED":
        errors.append(
            "UC-01 status has been changed from 'UNRESOLVED — GOVERNED CLARIFICATION "
            "REQUIRED' — resolving UC-01 requires approved methodology governance, "
            "not a registry edit (CHDM v0.1 §19)."
        )
    if uc01.get("canonical_rule_object_id") is not None:
        errors.append(
            "UC-01 canonical_rule_object_id has been assigned — CHDM v0.1 §19 "
            "explicitly withholds this ID until the rule's semantics are approved."
        )
    d1_rule = registry["contradiction_rules"]["d1_contradiction_rule"]
    if d1_rule.get("status") != "FULLY_GOVERNED":
        errors.append("contradiction_rules.yaml: d1_contradiction_rule status altered")

    # --- 9. reason code family sanity (non-empty) ---
    rc = registry["reason_codes"]
    for fam in ("requirement_reasons", "evidence_review_reasons", "operational_priority_reasons"):
        if not rc.get(fam):
            errors.append(f"reason_codes.yaml: {fam} is empty")

    if errors:
        raise RegistryValidationError(
            "Registry validation failed with "
            f"{len(errors)} error(s):\n  - " + "\n  - ".join(errors)
        )


def load_and_validate(registry_dir: pathlib.Path = REGISTRY_DIR) -> MethodologyRegistry:
    registry = load_registry(registry_dir)
    validate_registry(registry)
    return registry
