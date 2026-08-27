"""
Build Milestone 2 — typed candidate-observation schemas (spec §5).

Design principle (Checkpoint 2A refinement 1): every dataclass below
separates SEMANTIC fields (what the model may populate) from SYSTEM
fields (what the extraction pipeline attaches after validation). The
model never sees, sets, or requests: model_provider, model_version,
extracted_at, trace_id, or evidence_state. Those live exclusively on
`ExtractionSystemFields`, which every accepted observation carries, and
which starts out unpopulated (`ExtractionSystemFields.pending()`) when a
dataclass is first constructed from parsed model output —
`pipeline.py._attach_system_metadata()` is the ONLY place that ever
produces a populated one, via `dataclasses.replace`. There is no
constructor path by which model-controlled data can set evidence_state to
CURRENT_CONFIRMED: `ExtractionSystemFields.__post_init__` rejects any
evidence_state other than CURRENT_UNVERIFIED or STALE (the only two
states freshness derivation may legitimately produce for AI-extracted
evidence — see domain/evidence.py's confirmation-boundary note).

None of these dataclasses are consumed by engine.evaluate(). They are a
new, earlier pipeline stage entirely — not a replacement for, or
modification of, domain/signals.py's ValueEvidenceSignal /
DimensionQualifierSignal / RiskSeverityClaim (those remain Milestone 1's
confirmation-ready, engine-consumed types, unchanged). A future
(out-of-scope) human-confirmation milestone would be what turns an
accepted observation here into one of those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from domain.enums import DimensionCode, EvidenceState

from .enums import InferenceBasis, MissingInformationBasis, ObservationType


@dataclass(frozen=True)
class SourceSpan:
    """Checkpoint 2A refinement 2: deterministic source position, not just
    quoted text. `start_char`/`end_char` are 0-indexed, exclusive-end
    offsets into the SPECIFIC evidence item's raw text.

    Milestone 2B baseline-fix update: as of the span-grounding fix, the
    model supplies only `text`; `start_char`/`end_char` are ALWAYS
    system-derived by validation.py's `resolve_source_span` (exact,
    unambiguous substring search against the cited evidence — never
    model-authored, never fuzzy). This class's invariant below
    (`end_char - start_char == len(text)`) is therefore now an internal
    consistency assertion on values the application itself computed, not
    a check on model arithmetic — if it ever fails, that indicates an
    implementation defect in the derivation, not a bad model response."""
    text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("SourceSpan.text must not be empty.")
        if self.start_char < 0:
            raise ValueError("SourceSpan.start_char must be >= 0.")
        if self.end_char <= self.start_char:
            raise ValueError("SourceSpan.end_char must be > start_char.")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError(
                "SourceSpan is internally inconsistent: end_char - start_char "
                f"({self.end_char - self.start_char}) must equal len(text) ({len(self.text)})."
            )


@dataclass(frozen=True)
class ExtractionSystemFields:
    """System-generated metadata (Checkpoint 2A refinement 1). The model
    never populates this. `pending()` is the only way user/model-facing
    code can construct one prior to pipeline attachment; `populated()` is
    the only way the pipeline may fill it in."""
    observation_id: Optional[str] = None
    model_provider: Optional[str] = None
    model_version: Optional[str] = None
    extracted_at: Optional[datetime] = None
    trace_id: Optional[str] = None
    evidence_state: Optional[EvidenceState] = None

    def __post_init__(self) -> None:
        if self.evidence_state is not None and self.evidence_state not in (
            EvidenceState.CURRENT_UNVERIFIED, EvidenceState.STALE,
        ):
            raise ValueError(
                "ExtractionSystemFields.evidence_state must be CURRENT_UNVERIFIED "
                "(the default) or STALE (freshness-derived) — an AI extraction path "
                "must never produce CURRENT_CONFIRMED (§3.3 confirmation boundary; "
                "Checkpoint 2A refinement 1)."
            )

    @property
    def is_populated(self) -> bool:
        return self.observation_id is not None

    @staticmethod
    def pending() -> "ExtractionSystemFields":
        return ExtractionSystemFields()


@dataclass(frozen=True)
class ObservationRef:
    """A CandidateContradiction's reference to one of the OTHER
    observations produced in the SAME extraction response. Deliberately
    NOT a model-invented ID string (that would be adjacent to the banned
    trace_id/observation_id concept) — it is the array key + index the
    model itself is emitting into, which the model already fully
    controls and cannot get inconsistent with itself. Resolved to a final
    system-assigned observation_id by pipeline.py after validation."""
    observation_type: ObservationType
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("ObservationRef.index must be >= 0.")


def _require_system(system: ExtractionSystemFields, cls_name: str) -> None:
    if system is None:
        raise ValueError(f"{cls_name}.system is required (may be pending, never absent).")


@dataclass(frozen=True)
class ObjectiveCandidate:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    objective_text: str
    stated_outcome: Optional[str] = None
    measure: Optional[str] = None
    target: Optional[str] = None
    timeframe: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "ObjectiveCandidate")
        if not self.objective_text:
            raise ValueError("ObjectiveCandidate.objective_text must not be empty.")


@dataclass(frozen=True)
class StakeholderObservation:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    person_identifier: str
    role: Optional[str] = None
    stakeholder_type: Optional[str] = None
    sponsor_or_champion_relationship: Optional[str] = None
    continuity_event: Optional[str] = None
    effective_date: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "StakeholderObservation")
        if not self.person_identifier:
            raise ValueError("StakeholderObservation.person_identifier must not be empty.")


@dataclass(frozen=True)
class AdoptionObservation:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    workflow_or_use_case: str
    observed_behavior: str
    adoption_nature: Optional[str] = None
    human_vs_automated: Optional[str] = None
    evidence_date: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "AdoptionObservation")
        if not self.workflow_or_use_case or not self.observed_behavior:
            raise ValueError(
                "AdoptionObservation requires workflow_or_use_case and observed_behavior."
            )


@dataclass(frozen=True)
class ServiceObservation:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    incident_or_condition: str
    severity_language: Optional[str] = None  # raw language from evidence; NEVER a CHDM severity tier
    affected_workflow: Optional[str] = None
    status: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "ServiceObservation")
        if not self.incident_or_condition:
            raise ValueError("ServiceObservation.incident_or_condition must not be empty.")


@dataclass(frozen=True)
class CommercialObservation:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    event_type: str  # renewal | procurement | budget | payment | pricing | competitive
    description: str
    commercial_decision_active_candidate: Optional[bool] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    _ALLOWED_EVENT_TYPES = ("renewal", "procurement", "budget", "payment", "pricing", "competitive")

    def __post_init__(self) -> None:
        _require_system(self.system, "CommercialObservation")
        if not self.description:
            raise ValueError("CommercialObservation.description must not be empty.")
        if self.event_type not in self._ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"CommercialObservation.event_type={self.event_type!r} must be one of "
                f"{self._ALLOWED_EVENT_TYPES} (spec §5)."
            )


@dataclass(frozen=True)
class ExperienceObservation:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    statement: str
    stakeholder: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "ExperienceObservation")
        if not self.statement:
            raise ValueError("ExperienceObservation.statement must not be empty.")


@dataclass(frozen=True)
class StrategicObservation:
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    event: str
    affected_org_or_context: Optional[str] = None
    event_date: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "StrategicObservation")
        if not self.event:
            raise ValueError("StrategicObservation.event must not be empty.")


@dataclass(frozen=True)
class MissingInformationCandidate:
    """Checkpoint 2A refinement 4: evidence-scope-relative absence only.
    No source_span (there is nothing to quote for an absence) — but the
    reviewed evidence scope is mandatory, so the claim always reads as
    "not found in THESE reviewed items," never "absent from the account."
    """
    missing_item: str
    reviewed_evidence_ids: tuple[str, ...]
    basis: MissingInformationBasis = MissingInformationBasis.NOT_FOUND_IN_REVIEWED_EVIDENCE
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "MissingInformationCandidate")
        if not self.missing_item:
            raise ValueError("MissingInformationCandidate.missing_item must not be empty.")
        if not self.reviewed_evidence_ids:
            raise ValueError(
                "MissingInformationCandidate.reviewed_evidence_ids must not be empty — "
                "the reviewed scope must always be explicit (Checkpoint 2A refinement 4)."
            )
        if self.basis != MissingInformationBasis.NOT_FOUND_IN_REVIEWED_EVIDENCE:
            raise ValueError(
                "MissingInformationCandidate.basis must be NOT_FOUND_IN_REVIEWED_EVIDENCE — "
                "this observation type must never assert global account-level absence."
            )


@dataclass(frozen=True)
class CandidateContradiction:
    """Spec §11: AI may link two observations as an apparent conflict; it
    must never resolve, average, downgrade, or classify the conflict
    (no Disputed, no D1=Mixed, no DMEG — all deterministic/downstream).
    `methodology_construct_hint` is a plain descriptive string (e.g. "the
    objective's realized outcome"), never validated against the registry
    and never consumed by any deterministic rule — it is advisory text
    for a future human reviewer, directly requested by spec §5's "nature
    of apparent conflict" / "methodology construct potentially affected"
    fields. (Distinct from the `risk_mechanism_hint` field proposed for
    the 7 positive observation types, which Checkpoint 2A explicitly
    removed — that would have been per-observation CHDM-taxonomy
    labeling; this is a plain-language note about a conflict a human
    still has to adjudicate.)"""
    observation_ref_a: ObservationRef
    observation_ref_b: ObservationRef
    conflict_description: str
    methodology_construct_hint: Optional[str] = None
    status: str = "CANDIDATE"
    # System-resolved (never model-provided): the final observation_id each
    # ref points to, AFTER system-metadata attachment and AFTER
    # deduplication canonicalization — populated by pipeline.py only.
    # observation_ref_a/b above remain the original array-position
    # provenance pointers into the model's raw output.
    resolved_observation_id_a: Optional[str] = None
    resolved_observation_id_b: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "CandidateContradiction")
        if not self.conflict_description:
            raise ValueError("CandidateContradiction.conflict_description must not be empty.")
        if self.status != "CANDIDATE":
            raise ValueError(
                "CandidateContradiction.status must be 'CANDIDATE' — resolving a "
                "contradiction (Disputed, D1=Mixed, DMEG, or any other governed "
                "disposition) is a deterministic/human-governed downstream function, "
                "never an AI extraction output (spec §11)."
            )


# Milestone 2C — CHDM-registry-scoped candidate values. Extraction-layer
# literal tuples, deliberately NOT imported from domain.enums (module
# docstring: this file stays separate from CHDM's governed vocabulary).
# MUST stay byte-identical to domain.enums.RiskMechanismCode / RiskSeverity
# / ValueEvidenceBasis's string values — checked by
# tests/test_extraction_schemas.py.
_MVP_IMPLEMENTED_RISK_MECHANISMS = ("CR-01", "CR-02", "CR-08")
# CR-01 Sponsor/Champion Continuity, CR-02 Service Failure, CR-08 Value
# Failure/Rejection — the AI-candidate-classification MVP subset,
# authorized by the PMO Option B decision (Milestone 2C, MVP scope
# reduction). This is narrower than registry/risk_mechanisms.yaml's own
# `mvp_implementation_status`, which still marks CR-03 (Commercial
# Continuity) `implemented` for the DETERMINISTIC engine — CR-03 is
# fully present in CHDM, the registry, domain/enums.py, and
# engine/risk_engine.py, completely untouched by this change. What is
# deferred is narrower and specific to this file: automated
# CandidateRiskSignal proposal for CR-03 during Milestone 2C, based on
# empirical evidence across three live evaluation rounds (case 26 never
# once produced a correct CR-03 candidate; the third round additionally
# showed CR-08 being confused with CR-03 on an unambiguous CR-08 case).
# CR-06 is Scenario Lab only; CR-04/05/07 are deferred for other,
# pre-existing reasons. Prompt v4 never offers CR-03 (or CR-04/05/06/07)
# to the model, and this tuple is the SECOND, structural enforcement
# layer (json_schemas.py's mechanism enum is the first) — a model
# attempting one anyway fails schema validation, never silently
# succeeds via dataclass construction alone.
_CANDIDATE_SEVERITY_TIERS = ("WATCH", "MATERIAL", "CRITICAL")
# RESOLVED excluded on purpose: a candidate risk signal proposes a
# potential/prospective severity; it cannot propose a lifecycle
# resolution state (CHDM v0.1 §6.1).
_CANDIDATE_EVIDENCE_BASES = (
    "PROXY_SUPPORTED", "MEASURED_OPERATIONAL_EVIDENCE",
    "CUSTOMER_CONFIRMED", "INDEPENDENTLY_VERIFIED",
)
# UNVERIFIED_CLAIM / INSUFFICIENT_EVIDENCE deliberately excluded: those
# describe an ABSENCE of qualifying evidence-basis, never something an
# extractor proposes about evidence that is actually present in front of
# it (Milestone 2C checkpoint scoping decision).
_CANDIDATE_SUPPORTS_VALUES = ("ACHIEVED", "PROGRESSING", "NOT_ACHIEVED")


@dataclass(frozen=True)
class CandidateRiskSignal:
    """Milestone 2C (spec §4.2/§7/FR-15.2) — an upstream extraction
    capability, not a governed conclusion. Proposes WHICH risk mechanism
    and a POTENTIAL severity tier — never an activated one (CHDM v0.1
    §3.3 confirmation boundary; §6.1). `evidence_state` on `system`
    remains CURRENT_UNVERIFIED like every other extraction output; only
    human confirmation (a future milestone) may change that.

    Grounded twice: (a) its own exact source_span, resolved against the
    evidence text of whichever item `supporting_observation_ref` points
    to; (b) the required structural reference itself. `source_evidence_id`
    is deliberately NOT model-facing (absent from
    CANDIDATE_RISK_SIGNAL_SCHEMA) — the pipeline derives it exclusively
    from the resolved supporting observation, so a candidate risk signal
    can never cite an evidence item different from the observation it
    claims to interpret (Milestone 2C implementation constraint 1)."""
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    mechanism: str                    # "CR-01" | "CR-02" | "CR-08" (CR-03 deferred from MVP, see tuple comment above)
    proposed_severity_tier: str       # "WATCH" | "MATERIAL" | "CRITICAL"
    supporting_observation_ref: ObservationRef
    # System-resolved (never model-provided): the supporting observation's
    # final canonical observation_id, AFTER system-metadata attachment and
    # AFTER deduplication — populated by pipeline.py only, exactly
    # mirroring CandidateContradiction.resolved_observation_id_a/b.
    # supporting_observation_ref above remains the original array-position
    # provenance pointer into the model's raw output.
    resolved_observation_id: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "CandidateRiskSignal")
        if self.mechanism not in _MVP_IMPLEMENTED_RISK_MECHANISMS:
            raise ValueError(
                f"CandidateRiskSignal.mechanism={self.mechanism!r} must be one of "
                f"{_MVP_IMPLEMENTED_RISK_MECHANISMS} — the AI-candidate-classification "
                "MVP subset (PMO Option B decision). CR-03 is deferred from automated "
                "classification specifically (though fully implemented in the "
                "deterministic engine); CR-04/05/06/07 are deferred/scenario-lab-only "
                "for other reasons. None may ever be proposed here."
            )
        if self.proposed_severity_tier not in _CANDIDATE_SEVERITY_TIERS:
            raise ValueError(
                f"CandidateRiskSignal.proposed_severity_tier={self.proposed_severity_tier!r} "
                f"must be one of {_CANDIDATE_SEVERITY_TIERS} (CHDM v0.1 §6.1) — a POTENTIAL "
                "severity only, never activated_severity (§3.3)."
            )


@dataclass(frozen=True)
class CandidateEvidenceClassification:
    """Milestone 2C companion to CandidateRiskSignal, for the Objective /
    value-evidence side (spec §4.2/§7/FR-15.2). Proposes an
    evidence-basis interpretation and which outcome it would support IF
    Current+Confirmed — never a final Objective Outcome, D1, or
    Reliability determination (all remain deterministic/human-governed,
    CHDM v0.1 §3.3/§5.4).

    Same application-derived `source_evidence_id` discipline as
    CandidateRiskSignal — see that class's docstring."""
    source_evidence_id: str
    source_span: SourceSpan
    basis: InferenceBasis
    proposed_basis: str                # PROXY_SUPPORTED | MEASURED_OPERATIONAL_EVIDENCE | CUSTOMER_CONFIRMED | INDEPENDENTLY_VERIFIED
    supports: str                      # "ACHIEVED" | "PROGRESSING" | "NOT_ACHIEVED"
    supporting_observation_ref: ObservationRef
    resolved_observation_id: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "CandidateEvidenceClassification")
        if self.proposed_basis not in _CANDIDATE_EVIDENCE_BASES:
            raise ValueError(
                f"CandidateEvidenceClassification.proposed_basis={self.proposed_basis!r} "
                f"must be one of {_CANDIDATE_EVIDENCE_BASES} (CHDM v0.1 §5.2, MVP-proposable "
                "subset — UNVERIFIED_CLAIM/INSUFFICIENT_EVIDENCE describe absence, not "
                "something an extractor proposes about present evidence)."
            )
        if self.supports not in _CANDIDATE_SUPPORTS_VALUES:
            raise ValueError(
                f"CandidateEvidenceClassification.supports={self.supports!r} must be one of "
                f"{_CANDIDATE_SUPPORTS_VALUES} (CHDM v0.1 §5.4)."
            )


# Milestone 4B: the authoritative D2/D6 qualifier vocabularies, hardcoded
# as literal tuples here — mirroring _MVP_IMPLEMENTED_RISK_MECHANISMS /
# _CANDIDATE_SEVERITY_TIERS / _CANDIDATE_EVIDENCE_BASES above, i.e. NOT
# imported at runtime from domain.signals.DIMENSION_QUALIFIERS. This is a
# deliberate repeat of this file's existing pattern (json_schemas.py's
# per-channel enum schemas are the model-facing gate; these tuples are
# the matching dataclass-level guard) — kept in sync with
# domain.signals.DIMENSION_QUALIFIERS by comment cross-reference, not by
# import, exactly like the other hardcoded-literal enum sets in this
# module. If domain.signals.DIMENSION_QUALIFIERS is ever amended, these
# two tuples (and their json_schemas.py / prompts.py counterparts) must
# be updated to match by hand.
_CANDIDATE_D2_QUALIFIERS = (
    "INTENDED_WORKFLOWS_OPERATING_NORMALLY",
    "AUTOMATION_RELIABLE_LOW_LOGIN_OK",
    "NARROW_BREADTH_OR_CONCENTRATION",
    "WORKFLOWS_NOT_OCCURRING",
    "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED",
)
_CANDIDATE_D6_QUALIFIERS = (
    "APPROPRIATE_SPONSOR_COVERAGE",
    "CHAMPION_LOST_NO_SUCCESSOR",
    "CHAMPION_DEPARTURE_UNCONFIRMED",
    "SUCCESSION_UNCLEAR_OR_CONCENTRATED",
)
_DIMENSION_TO_CANDIDATE_QUALIFIERS = {
    DimensionCode.D2: _CANDIDATE_D2_QUALIFIERS,
    DimensionCode.D6: _CANDIDATE_D6_QUALIFIERS,
}


@dataclass(frozen=True)
class CandidateDimensionQualifier:
    """Milestone 4B (PMO Option A follow-on to Milestone 2C; resolves
    M3-OD-01) — an upstream extraction capability proposing WHICH D2
    (Product Adoption) or D6 (Relationship Health) qualifier best
    describes an already-accepted, already-grounded semantic observation.
    Never a governed DimensionState or DimensionQualifierSignal itself
    (CHDM v0.1 §3.3 confirmation boundary) — `system.evidence_state`
    remains CURRENT_UNVERIFIED like every other extraction output; only
    a future Milestone 4C human-confirmation step may promote it into a
    real domain.signals.DimensionQualifierSignal.

    INHERITED GROUNDING (approved, narrow, explicitly-scoped exception to
    Checkpoint 2A refinement 2 grounding discipline — see extraction.
    pipeline._build_candidate_classification's docstring for that
    discipline's unmodified statement, which still governs every other
    span-grounded type including CandidateRiskSignal and
    CandidateEvidenceClassification above):

    This is inherited grounding for second-stage classification of an
    already-grounded semantic observation. The stage-2 D2/D6 qualifier
    classifier (extraction.pipeline.run_dimension_qualifier_
    classification) operates exclusively on observations that already
    passed stage-1's own independent exact-span grounding
    (AdoptionObservation for D2, StakeholderObservation for D6). Because
    the supporting observation is already exactly grounded, the model is
    NOT asked to reproduce source_span a second time — `source_span` and
    `source_evidence_id` below are copied verbatim from the resolved
    supporting observation by the application, never re-derived or
    re-validated against evidence text. This does NOT relax stage-1's own
    grounding requirement in any way, and is NOT generalized to
    CandidateRiskSignal or CandidateEvidenceClassification, which keep
    their existing independent-span-grounding requirement unchanged.

    Model-facing fields: `supporting_observation_ref`, `qualifier`,
    `basis`. Application-derived fields (never model-facing): `dimension`
    (which channel — candidate_d2_qualifiers vs candidate_d6_qualifiers —
    the item came from), `source_span`, `source_evidence_id`,
    `resolved_observation_id`, `system` (this dataclass's OWN stage-2
    provenance — provider/model/trace_id — which is distinct from, and
    never overwrites, the referenced observation's own stage-1 `system`).

    The 4 approved grounding prohibitions (extraction.pipeline.
    run_dimension_qualifier_classification /
    _resolve_dimension_qualifier_reference, each its own RejectionReason):
    a CandidateDimensionQualifier must never inherit grounding from (a) a
    rejected observation, (b) an unresolved reference, (c) a disallowed
    observation type (D2 may only reference AdoptionObservation; D6 may
    only reference StakeholderObservation), (d) an ungrounded object.

    Like CandidateRiskSignal / CandidateEvidenceClassification,
    CandidateDimensionQualifier is structurally invalid as a
    CandidateContradiction reference target — it is absent from
    SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES and DIMENSION_QUALIFIER_
    TYPE_TO_ARRAY_KEY is deliberately kept separate from
    OBSERVATION_TYPE_TO_ARRAY_KEY for exactly this reason."""
    dimension: DimensionCode
    qualifier: str
    basis: InferenceBasis
    supporting_observation_ref: ObservationRef
    # Application-derived, inherited verbatim from the resolved
    # supporting observation (see class docstring) — never independently
    # model-supplied or independently re-grounded against evidence text.
    source_evidence_id: Optional[str] = None
    source_span: Optional[SourceSpan] = None
    resolved_observation_id: Optional[str] = None
    system: ExtractionSystemFields = field(default_factory=ExtractionSystemFields.pending)

    def __post_init__(self) -> None:
        _require_system(self.system, "CandidateDimensionQualifier")
        allowed = _DIMENSION_TO_CANDIDATE_QUALIFIERS.get(self.dimension)
        if allowed is None:
            raise ValueError(
                f"CandidateDimensionQualifier.dimension={self.dimension!r} must be "
                f"DimensionCode.D2 or DimensionCode.D6 (Milestone 4B scope — the only "
                "two dimensions with an implemented candidate-qualifier channel)."
            )
        if self.qualifier not in allowed:
            raise ValueError(
                f"CandidateDimensionQualifier.qualifier={self.qualifier!r} must be one "
                f"of {allowed} for dimension={self.dimension!r} (domain.signals."
                "DIMENSION_QUALIFIERS — the authoritative CHDM qualifier vocabulary; see "
                "_DIMENSION_TO_CANDIDATE_QUALIFIERS above)."
            )


@dataclass(frozen=True)
class AtomicPredicateEvidence:
    """Milestone 4B v3 (atomic-predicate + deterministic composition
    architecture, approved after the self-attestation design was rejected
    as insufficient — see CandidateDimensionQualifier's docstring history
    for the full architecture context). ONE grounded atomic-predicate
    proposal from the stage-2 classifier, for one of the two compound
    qualifiers currently scoped to this path (AUTOMATION_RELIABLE_LOW_
    LOGIN_OK on D2, CHAMPION_LOST_NO_SUCCESSOR on D6).

    This is PROVENANCE/AUDIT metadata, never a governed domain object and
    never itself a CandidateDimensionQualifier. It is deliberately not a
    field on CandidateDimensionQualifier (which is unchanged by this
    architecture) — instead it lives on ExtractionResult.
    dimension_qualifier_predicate_evidence (extraction/pipeline.py),
    addressable via (resolved_observation_id, dimension) without needing
    a separate foreign key, since composition is 1:1 per observation per
    compound qualifier. EVERY grounded predicate proposed for an eligible
    observation is preserved here, whether or not the full required set
    for that observation was ever complete — an INCOMPLETE set is
    preserved for audit exactly like a COMPLETE one; only a complete set
    additionally produces a composed CandidateDimensionQualifier (see
    extraction.pipeline.run_dimension_qualifier_classification).

    Grounding: `evidence_text` must appear verbatim as an exact substring
    of the SAME observation's own `source_span.text` (checked by
    extraction.validation before this object is ever constructed — no
    fuzzy/normalized/partial matching, mirroring resolve_source_span's own
    discipline). Different predicates for the same observation MAY cite
    different, non-overlapping substrings of that same source_span.text;
    nothing requires them to share or overlap.

    Explicitly acknowledged, NOT eliminated, residual risk: exact-
    substring grounding guarantees the cited text is REAL (present in the
    observation's own span) and structurally prevents cross-observation
    synthesis (there is only ever one observation in scope for an isolated
    call) — both deterministic, 100% guaranteed. It does NOT guarantee the
    cited text actually, semantically ESTABLISHES the specific predicate's
    factual condition; a model could still mis-tag a real-but-
    semantically-insufficient span to a predicate_id. This narrows but
    does not eliminate the risk the self-attestation design's rejection
    was about — reported honestly here rather than oversold as a complete
    guarantee, per explicit approved-architecture instruction."""
    predicate_id: str
    dimension: DimensionCode
    resolved_observation_id: str
    evidence_text: str
    basis: InferenceBasis

    def __post_init__(self) -> None:
        if not self.predicate_id:
            raise ValueError("AtomicPredicateEvidence.predicate_id must not be empty.")
        if not self.resolved_observation_id:
            raise ValueError("AtomicPredicateEvidence.resolved_observation_id must not be empty.")
        if not self.evidence_text:
            raise ValueError("AtomicPredicateEvidence.evidence_text must not be empty.")


# The 7 span-grounded, positive observation dataclasses, keyed by
# ObservationType — single source of truth for pipeline.py / validation.py
# so type <-> dataclass wiring is declared exactly once.
OBSERVATION_TYPE_TO_DATACLASS = {
    ObservationType.OBJECTIVE_CANDIDATE: ObjectiveCandidate,
    ObservationType.STAKEHOLDER_OBSERVATION: StakeholderObservation,
    ObservationType.ADOPTION_OBSERVATION: AdoptionObservation,
    ObservationType.SERVICE_OBSERVATION: ServiceObservation,
    ObservationType.COMMERCIAL_OBSERVATION: CommercialObservation,
    ObservationType.EXPERIENCE_OBSERVATION: ExperienceObservation,
    ObservationType.STRATEGIC_OBSERVATION: StrategicObservation,
    ObservationType.MISSING_INFORMATION_CANDIDATE: MissingInformationCandidate,
}

# Milestone 2C: the 2 candidate-classification dataclasses, keyed
# separately from OBSERVATION_TYPE_TO_DATACLASS above because these are
# NOT processed by the main per-item loop (they require the full
# accepted-item index first, to resolve supporting_observation_ref and
# derive source_evidence_id from it — see pipeline.py's second pass,
# same ordering reason CandidateContradiction is handled separately).
CANDIDATE_CLASSIFICATION_TYPE_TO_DATACLASS = {
    ObservationType.CANDIDATE_RISK_SIGNAL: CandidateRiskSignal,
    ObservationType.CANDIDATE_EVIDENCE_CLASSIFICATION: CandidateEvidenceClassification,
}

# Milestone 4B: the D2/D6 candidate-qualifier dataclass map, kept separate
# from CANDIDATE_CLASSIFICATION_TYPE_TO_DATACLASS above (mirroring
# extraction.enums.DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY's own
# separateness from CANDIDATE_CLASSIFICATION_TYPE_TO_ARRAY_KEY) because
# both channels resolve to the SAME dataclass (CandidateDimensionQualifier
# — one dataclass, two model-output channels, per the approved
# architecture), not two distinct dataclasses like the 2C pair above.
DIMENSION_QUALIFIER_TYPE_TO_DATACLASS = {
    ObservationType.CANDIDATE_D2_QUALIFIER: CandidateDimensionQualifier,
    ObservationType.CANDIDATE_D6_QUALIFIER: CandidateDimensionQualifier,
}
