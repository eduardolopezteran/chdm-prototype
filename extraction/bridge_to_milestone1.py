"""
Build Milestone 2 — TEST-ONLY Milestone 1 integration-smoke adapter
(spec §16, Checkpoint 2A refinement 7).

This is NOT the production human-confirmation architecture (that is an
out-of-scope future milestone). It exists solely so
tests/test_extraction_boundary_smoke_m1_integration.py can prove that an
accepted, Current+Unverified AI observation flows through
`engine.evaluate()` without crossing the §3.3 confirmation boundary — it
can never activate Material/Critical severity or produce a falsely
confirmed governed result.

Every function here does a MECHANICAL field copy only:
  - `evidence_state` is always taken directly from the observation's own
    `system.evidence_state` (populated exclusively by the extraction
    pipeline as CURRENT_UNVERIFIED, or STALE via freshness derivation —
    never CURRENT_CONFIRMED, enforced by ExtractionSystemFields itself).
    No function here accepts an `evidence_state` override parameter, so
    there is no code path through this module that can promote or
    confirm anything.
  - The CHDM-governed target (`dimension`+`qualifier`, `mechanism`+`tier`,
    or `basis`+`supports`) is always supplied explicitly by the caller.
    This module never infers a CHDM dimension, qualifier, risk mechanism,
    tier, or value-evidence-basis from an observation's free-text
    semantic fields — that mapping is exactly the "semantic enrichment"
    Checkpoint 2A refinement 7 prohibits this adapter from performing.
"""

from __future__ import annotations

from domain.enums import DimensionCode, RiskMechanismCode, ValueEvidenceBasis
from domain.signals import DimensionQualifierSignal, RiskSeverityClaim, ValueEvidenceSignal

from .schemas import ObjectiveCandidate


def _require_finalized(obs) -> None:
    if not obs.system.is_populated:
        raise ValueError(
            "bridge_to_milestone1 requires a pipeline-finalized observation "
            "(system fields populated by extraction.pipeline.run_extraction)."
        )


def to_dimension_qualifier_signal(obs, *, dimension: DimensionCode, qualifier: str) -> DimensionQualifierSignal:
    """`dimension`/`qualifier` are supplied by the caller — this function
    never chooses them."""
    _require_finalized(obs)
    return DimensionQualifierSignal(
        signal_id=obs.system.observation_id,
        dimension=dimension,
        evidence_id=obs.source_evidence_id,
        evidence_state=obs.system.evidence_state,
        qualifier=qualifier,
    )


def to_risk_severity_claim(obs, *, mechanism: RiskMechanismCode, tier: str) -> RiskSeverityClaim:
    """`mechanism`/`tier` are supplied by the caller — this function never
    decides which CR mechanism or severity tier an observation implies."""
    _require_finalized(obs)
    return RiskSeverityClaim(
        signal_id=obs.system.observation_id,
        mechanism=mechanism,
        evidence_id=obs.source_evidence_id,
        evidence_state=obs.system.evidence_state,
        tier=tier,
    )


def to_value_evidence_signal(
    obs: ObjectiveCandidate, *, basis: ValueEvidenceBasis, supports: str,
) -> ValueEvidenceSignal:
    """`basis` (the CHDM Value Evidence Basis — Proxy Supported, Measured
    Operational Evidence, etc.) and `supports` are supplied by the
    caller — deliberately not derived from `ObjectiveCandidate.basis`,
    which is the unrelated EXPLICIT/INFERRED_CANDIDATE extraction-
    provenance axis, not a CHDM Value Evidence Basis."""
    _require_finalized(obs)
    return ValueEvidenceSignal(
        signal_id=obs.system.observation_id,
        evidence_id=obs.source_evidence_id,
        evidence_state=obs.system.evidence_state,
        basis=basis,
        supports=supports,
    )
