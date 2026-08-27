"""
Milestone 3A — Human Confirmation Backend.

Backend authority and recomputation only (no UI). This package is the
real, production successor to `extraction.bridge_to_milestone1` (which
remains, unmodified, a TEST-ONLY Milestone 1 integration-smoke adapter —
see its own docstring). It is a new top-level package, deliberately kept
separate from `engine/` (the pure, dependency-free deterministic CHDM
engine) and `extraction/` (the AI-extraction layer), so that neither of
those packages ever needs to import the other. `confirmation/` is the one
place allowed to depend on both.

Core responsibility: turn an `extraction.pipeline.ExtractionResult` plus
an append-only journal of `HumanConfirmationRecord` decisions into the
Milestone 1 signal tuples (`ValueEvidenceSignal`, `DimensionQualifierSignal`,
`RiskSeverityClaim`) that `engine.evaluate.evaluate()` already consumes —
without ever modifying `engine/`, without ever inventing a new
`EvidenceState` value, and without ever letting the AI extraction layer
promote its own output to Current+Confirmed.

Modules:
  enums.py            — ConfirmationTargetKind, ExclusionReason
  schemas.py           — HumanConfirmationRecord and its factory, plus
                          ActiveEvidenceItem / ActiveEvidenceSet /
                          ExclusionRecord / RecomputeDiagnostic /
                          ConsequentialityReport
  state_machine.py     — append-only journal resolution (terminal
                          disposition per target)
  active_evidence.py   — reconstruct_active_evidence()
  recompute.py          — recompute() (signal-building + evaluate() call)
  consequentiality.py   — compute_consequentiality() (reuses
                          engine.dmeg_engine.differs(), invents nothing new)
"""
