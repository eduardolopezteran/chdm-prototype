"""
Structured, schema-validated input signals the deterministic engine
consumes. These correspond to CHDM v0.1's "Extracted Observation" concept
(Technical Architecture §4/§6): a candidate structured claim about a
specific target (an Objective, a Dimension, or a Risk Mechanism), created
by extraction (in the real product) or hand-authored directly at a target
status (Milestone 1 Scenario Lab fixtures — BAR-01 §6).

The engine NEVER re-derives a qualitative judgment from raw text. It only
aggregates and gates already-classified signals against the registry's
deterministic conditions. This is the load-bearing boundary that keeps
the engine "deterministic" in CHDM's sense: the same signal set always
produces the same governed output, and only Current+Confirmed signals may
activate anything (§3.3 confirmation boundary).
"""

from dataclasses import dataclass, field

from .enums import EvidenceState, DimensionCode, RiskMechanismCode, ValueEvidenceBasis


@dataclass(frozen=True)
class ValueEvidenceSignal:
    """A structured claim feeding Objective Outcome evaluation (CHDM v0.1 §5.4).

    `supports` must be one of: "ACHIEVED", "PROGRESSING", "NOT_ACHIEVED" —
    i.e. which conclusion this specific piece of evidence would support IF
    it were Current + Confirmed. The objective_engine aggregates these
    against confirmation state; it never itself decides what a piece of
    evidence "means."
    """
    signal_id: str
    evidence_id: str
    evidence_state: EvidenceState
    basis: ValueEvidenceBasis
    supports: str  # "ACHIEVED" | "PROGRESSING" | "NOT_ACHIEVED"

    def __post_init__(self) -> None:
        if self.supports not in ("ACHIEVED", "PROGRESSING", "NOT_ACHIEVED"):
            raise ValueError(
                f"ValueEvidenceSignal.supports={self.supports!r} must be one of "
                "ACHIEVED, PROGRESSING, NOT_ACHIEVED (CHDM v0.1 §5.4)."
            )


# Controlled per-dimension qualifier vocabulary. Every code here is taken
# directly from a state_rule / safeguard sentence in registry/dimensions.yaml
# — this is not an invented taxonomy, it is that prose turned into stable
# identifiers so it can be matched deterministically instead of re-parsed.
DIMENSION_QUALIFIERS: dict[DimensionCode, tuple[str, ...]] = {
    DimensionCode.D2: (
        "INTENDED_WORKFLOWS_OPERATING_NORMALLY",     # -> SUPPORTED contributor
        "AUTOMATION_RELIABLE_LOW_LOGIN_OK",            # -> SUPPORTED (S5 safeguard)
        "NARROW_BREADTH_OR_CONCENTRATION",              # -> MIXED contributor
        "WORKFLOWS_NOT_OCCURRING",                       # -> CONCERNING contributor
        "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED", # -> CONCERNING contributor
    ),
    DimensionCode.D6: (
        "APPROPRIATE_SPONSOR_COVERAGE",               # -> SUPPORTED contributor
        "CHAMPION_LOST_NO_SUCCESSOR",                    # -> CONCERNING contributor (confirmed)
        "CHAMPION_DEPARTURE_UNCONFIRMED",                 # -> INSUFFICIENT_EVIDENCE contributor
        "SUCCESSION_UNCLEAR_OR_CONCENTRATED",              # -> MIXED contributor
    ),
}


@dataclass(frozen=True)
class DimensionQualifierSignal:
    """A structured claim feeding dimension-state evaluation for
    dimensions whose state rules require more than an upstream
    Objective/Risk result alone (currently D2, D6 — see
    engine/dimension_engine.py for which dimensions are implemented in
    Milestone 1)."""
    signal_id: str
    dimension: DimensionCode
    evidence_id: str
    evidence_state: EvidenceState
    qualifier: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        allowed = DIMENSION_QUALIFIERS.get(self.dimension)
        if allowed is None:
            raise ValueError(
                f"DimensionQualifierSignal.dimension={self.dimension} has no "
                "registered qualifier vocabulary — this dimension's signal-based "
                "evaluation is not implemented in Milestone 1."
            )
        if self.qualifier not in allowed:
            raise ValueError(
                f"DimensionQualifierSignal.qualifier={self.qualifier!r} is not in "
                f"the controlled vocabulary for {self.dimension}: {allowed}"
            )


@dataclass(frozen=True)
class RiskSeverityClaim:
    """A structured claim feeding risk evaluation (CHDM v0.1 §6). `tier`
    must match one of the mechanism's own severity_conditions keys in
    registry/risk_mechanisms.yaml (WATCH/MATERIAL/CRITICAL) — the engine
    does not decide which tier a piece of evidence represents; it only
    aggregates and confirmation-gates already-classified claims, exactly
    as AI extraction proposes potential severity and human confirmation
    gates activation in the real product (§3.3, §6.1, FR-15.2)."""
    signal_id: str
    mechanism: RiskMechanismCode
    evidence_id: str
    evidence_state: EvidenceState
    tier: str  # "WATCH" | "MATERIAL" | "CRITICAL"
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.tier not in ("WATCH", "MATERIAL", "CRITICAL"):
            raise ValueError(
                f"RiskSeverityClaim.tier={self.tier!r} must be WATCH, MATERIAL, "
                "or CRITICAL (CHDM v0.1 §6.1)."
            )
