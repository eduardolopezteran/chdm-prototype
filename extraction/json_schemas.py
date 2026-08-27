"""
Build Milestone 2 — model-facing JSON Schemas (spec §5 + Checkpoint 2A
refinement 1: semantic fields only).

Every schema below is deliberately `"additionalProperties": false` with
an explicit `properties`/`required` allowlist. This is the PRIMARY
boundary enforcement mechanism: model_provider, model_version,
extracted_at, trace_id, and evidence_state are simply not declared
anywhere in these schemas, so a model attempting to set them fails
JSON Schema validation immediately — the same mechanism that rejects a
model attempting to emit `"confirmed": true`, `"activated_severity":
"CRITICAL"`, `"dimension_state": "CONCERNING"`, `"operational_priority":
"OP1"`, `"dmeg": true`, `"churn_probability": 0.8`, or
`"recommended_action": "..."`. validation.py additionally runs an
explicit recursive denylist scan (defense-in-depth) so a rejected
boundary-violation attempt can be classified and counted precisely,
rather than surfacing only as a generic schema error.

Milestone 2B baseline-fix note: `SOURCE_SPAN_SCHEMA` deliberately no
longer declares `start_char`/`end_char`. The live Anthropic baseline
(eval/results/baseline_v1.json) showed the model cannot reliably
self-report exact Python-slice-convention character offsets, which was
rejecting otherwise-correct extractions. The model is now responsible
only for `text` -- the exact verbatim substring; the application derives
`start_char`/`end_char` deterministically by searching for that
substring in the cited evidence (validation.py's `resolve_source_span`).
This does not relax the exact-grounding requirement (Checkpoint 2A
refinement 2): the derivation still requires a byte-exact, unambiguous
match, with no fuzzy/normalized/partial matching. `additionalProperties:
false` here means the model cannot supply offsets even as advisory
values -- there is no field for them to occupy.
"""

from __future__ import annotations

from .enums import ObservationType, SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES, OBSERVATION_TYPE_TO_ARRAY_KEY, DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY, DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_ARRAY_KEY, D2_ATOMIC_PREDICATE_IDS, D6_ATOMIC_PREDICATE_IDS

SOURCE_SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1},
    },
    "required": ["text"],
    "additionalProperties": False,
}

BASIS_SCHEMA = {"type": "string", "enum": ["EXPLICIT", "INFERRED_CANDIDATE"]}

_COMMON_GROUNDING_PROPERTIES = {
    "source_evidence_id": {"type": "string", "minLength": 1},
    "source_span": SOURCE_SPAN_SCHEMA,
    "basis": BASIS_SCHEMA,
}
_COMMON_GROUNDING_REQUIRED = ["source_evidence_id", "source_span", "basis"]


def _observation_schema(extra_properties: dict, extra_required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {**_COMMON_GROUNDING_PROPERTIES, **extra_properties},
        "required": [*_COMMON_GROUNDING_REQUIRED, *extra_required],
        "additionalProperties": False,
    }


OBJECTIVE_CANDIDATE_SCHEMA = _observation_schema(
    {
        "objective_text": {"type": "string", "minLength": 1},
        "stated_outcome": {"type": ["string", "null"]},
        "measure": {"type": ["string", "null"]},
        "target": {"type": ["string", "null"]},
        "timeframe": {"type": ["string", "null"]},
    },
    ["objective_text"],
)

STAKEHOLDER_OBSERVATION_SCHEMA = _observation_schema(
    {
        "person_identifier": {"type": "string", "minLength": 1},
        "role": {"type": ["string", "null"]},
        "stakeholder_type": {"type": ["string", "null"]},
        "sponsor_or_champion_relationship": {"type": ["string", "null"]},
        "continuity_event": {"type": ["string", "null"]},
        "effective_date": {"type": ["string", "null"]},
    },
    ["person_identifier"],
)

ADOPTION_OBSERVATION_SCHEMA = _observation_schema(
    {
        "workflow_or_use_case": {"type": "string", "minLength": 1},
        "observed_behavior": {"type": "string", "minLength": 1},
        "adoption_nature": {"type": ["string", "null"]},
        "human_vs_automated": {"type": ["string", "null"]},
        "evidence_date": {"type": ["string", "null"]},
    },
    ["workflow_or_use_case", "observed_behavior"],
)

SERVICE_OBSERVATION_SCHEMA = _observation_schema(
    {
        "incident_or_condition": {"type": "string", "minLength": 1},
        "severity_language": {"type": ["string", "null"]},
        "affected_workflow": {"type": ["string", "null"]},
        "status": {"type": ["string", "null"]},
    },
    ["incident_or_condition"],
)

COMMERCIAL_OBSERVATION_SCHEMA = _observation_schema(
    {
        "event_type": {
            "type": "string",
            "enum": ["renewal", "procurement", "budget", "payment", "pricing", "competitive"],
        },
        "description": {"type": "string", "minLength": 1},
        "commercial_decision_active_candidate": {"type": ["boolean", "null"]},
    },
    ["event_type", "description"],
)

EXPERIENCE_OBSERVATION_SCHEMA = _observation_schema(
    {
        "statement": {"type": "string", "minLength": 1},
        "stakeholder": {"type": ["string", "null"]},
    },
    ["statement"],
)

STRATEGIC_OBSERVATION_SCHEMA = _observation_schema(
    {
        "event": {"type": "string", "minLength": 1},
        "affected_org_or_context": {"type": ["string", "null"]},
        "event_date": {"type": ["string", "null"]},
    },
    ["event"],
)

MISSING_INFORMATION_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "missing_item": {"type": "string", "minLength": 1},
        "reviewed_evidence_ids": {
            "type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1,
        },
    },
    "required": ["missing_item", "reviewed_evidence_ids"],
    "additionalProperties": False,
    # basis is NOT model-settable: it has exactly one legal value
    # (NOT_FOUND_IN_REVIEWED_EVIDENCE) and is attached by validation.py,
    # matching the same system-vs-semantic split used everywhere else.
}

_OBSERVATION_REF_SCHEMA = {
    "type": "object",
    "properties": {
        # Fixed post-live-defect (Case 11, prompt_v4_optionb_2c_eval1 crash):
        # this previously enumerated ALL of ObservationType, including the
        # two Milestone 2C candidate-classification types
        # (CANDIDATE_RISK_SIGNAL, CANDIDATE_EVIDENCE_CLASSIFICATION) added
        # after this schema was originally written. Those two are not keys
        # in OBSERVATION_TYPE_TO_ARRAY_KEY (enums.py deliberately keeps
        # them in a separate map -- they are not resolvable
        # contradiction/reference targets), so a model citing one in
        # observation_ref_a/b passed this schema gate and then hit
        # pipeline._resolve_array_ref()'s raw dict lookup, raising an
        # uncaught KeyError instead of a graceful rejection. Restricted
        # here to exactly OBSERVATION_TYPE_TO_ARRAY_KEY's keys -- the 7
        # span-grounded semantic types PLUS MissingInformationCandidate,
        # which is deliberately, historically supported as a contradiction
        # target (see RejectionReason.CONTRADICTION_OBSERVATION_NOT_
        # TRACEABLE_TO_EVIDENCE's docstring and pipeline._source_ref()'s
        # reviewed_evidence_ids fallback -- both exist specifically for
        # this case). This is NOT the same set as
        # SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES below (which excludes
        # MissingInformationCandidate for a different, narrower reason —
        # "absence is not evidence to interpret" — that applies to
        # candidate classifications proposing an interpretation, not to a
        # contradiction flagging a conflict against an absence).
        "observation_type": {
            "type": "string",
            "enum": sorted(t.value for t in OBSERVATION_TYPE_TO_ARRAY_KEY),
        },
        "index": {"type": "integer", "minimum": 0},
    },
    "required": ["observation_type", "index"],
    "additionalProperties": False,
}

CANDIDATE_CONTRADICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "observation_ref_a": _OBSERVATION_REF_SCHEMA,
        "observation_ref_b": _OBSERVATION_REF_SCHEMA,
        "conflict_description": {"type": "string", "minLength": 1},
        "methodology_construct_hint": {"type": ["string", "null"]},
    },
    "required": ["observation_ref_a", "observation_ref_b", "conflict_description"],
    "additionalProperties": False,
    # status is NOT model-settable: candidate contradictions are always
    # status="CANDIDATE" (spec §11) — attached by validation.py, never
    # something the model can set to a resolved disposition.
}

MECHANISM_SCHEMA = {"type": "string", "enum": ["CR-01", "CR-02", "CR-08"]}
# AI-candidate-classification MVP subset only (PMO Option B decision,
# Milestone 2C MVP scope reduction) — CR-03, CR-04, CR-05, CR-06, CR-07
# are all structurally impossible to emit here, not just discouraged in
# the prompt. CR-03 is deferred from AUTOMATED classification only; it
# remains fully implemented in registry/risk_mechanisms.yaml and
# engine/risk_engine.py for the deterministic engine, untouched by this
# schema. See extraction/schemas.py's _MVP_IMPLEMENTED_RISK_MECHANISMS
# comment for the full rationale.

SEVERITY_TIER_SCHEMA = {"type": "string", "enum": ["WATCH", "MATERIAL", "CRITICAL"]}
# A POTENTIAL severity only. RESOLVED intentionally excluded — that is a
# lifecycle state, never something a candidate signal proposes.

EVIDENCE_BASIS_SCHEMA = {
    "type": "string",
    "enum": ["PROXY_SUPPORTED", "MEASURED_OPERATIONAL_EVIDENCE", "CUSTOMER_CONFIRMED", "INDEPENDENTLY_VERIFIED"],
}
# UNVERIFIED_CLAIM / INSUFFICIENT_EVIDENCE deliberately excluded — those
# describe an absence of qualifying evidence-basis, not an interpretation
# an extractor proposes about evidence that is actually present.

SUPPORTS_SCHEMA = {"type": "string", "enum": ["ACHIEVED", "PROGRESSING", "NOT_ACHIEVED"]}

# Milestone 2C: like _OBSERVATION_REF_SCHEMA, but the observation_type
# enum is restricted to the 7 span-grounded SEMANTIC types only (built
# from enums.SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES, the single source
# of truth, so this cannot drift from it). A candidate classification can
# never reference a MissingInformationCandidate (absence is not evidence
# to interpret), a CandidateContradiction, or another candidate
# classification — structurally prevented here rather than only checked
# at runtime.
_SUPPORTING_OBSERVATION_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "observation_type": {
            "type": "string",
            "enum": sorted(t.value for t in SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES),
        },
        "index": {"type": "integer", "minimum": 0},
    },
    "required": ["observation_type", "index"],
    "additionalProperties": False,
}

# Milestone 2C — Candidate CHDM Classification Extraction (spec
# §4.2/§7/FR-15.2). Deliberately does NOT declare `source_evidence_id`:
# unlike the 7 positive observation types, a candidate classification's
# evidence citation is application-derived exclusively from the
# supporting observation its `supporting_observation_ref` resolves to
# (Milestone 2C implementation constraint 1) — there is no field for the
# model to occupy even advisorially, exactly the same "no field exists"
# enforcement pattern SOURCE_SPAN_SCHEMA already uses for start_char/
# end_char. `evidence_state`/`observation_id`/system metadata are
# likewise absent, as with every other model-facing schema in this file.
CANDIDATE_RISK_SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "source_span": SOURCE_SPAN_SCHEMA,
        "basis": BASIS_SCHEMA,
        "mechanism": MECHANISM_SCHEMA,
        "proposed_severity_tier": SEVERITY_TIER_SCHEMA,
        "supporting_observation_ref": _SUPPORTING_OBSERVATION_REF_SCHEMA,
    },
    "required": ["source_span", "basis", "mechanism", "proposed_severity_tier", "supporting_observation_ref"],
    "additionalProperties": False,
}

CANDIDATE_EVIDENCE_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "source_span": SOURCE_SPAN_SCHEMA,
        "basis": BASIS_SCHEMA,
        "proposed_basis": EVIDENCE_BASIS_SCHEMA,
        "supports": SUPPORTS_SCHEMA,
        "supporting_observation_ref": _SUPPORTING_OBSERVATION_REF_SCHEMA,
    },
    "required": ["source_span", "basis", "proposed_basis", "supports", "supporting_observation_ref"],
    "additionalProperties": False,
}

TYPE_TO_SCHEMA = {
    ObservationType.OBJECTIVE_CANDIDATE: OBJECTIVE_CANDIDATE_SCHEMA,
    ObservationType.STAKEHOLDER_OBSERVATION: STAKEHOLDER_OBSERVATION_SCHEMA,
    ObservationType.ADOPTION_OBSERVATION: ADOPTION_OBSERVATION_SCHEMA,
    ObservationType.SERVICE_OBSERVATION: SERVICE_OBSERVATION_SCHEMA,
    ObservationType.COMMERCIAL_OBSERVATION: COMMERCIAL_OBSERVATION_SCHEMA,
    ObservationType.EXPERIENCE_OBSERVATION: EXPERIENCE_OBSERVATION_SCHEMA,
    ObservationType.STRATEGIC_OBSERVATION: STRATEGIC_OBSERVATION_SCHEMA,
    ObservationType.MISSING_INFORMATION_CANDIDATE: MISSING_INFORMATION_CANDIDATE_SCHEMA,
}

# Milestone 2C: kept separate from TYPE_TO_SCHEMA above for the same
# reason CANDIDATE_CLASSIFICATION_TYPE_TO_DATACLASS is kept separate from
# OBSERVATION_TYPE_TO_DATACLASS in schemas.py — these two types are
# validated in their own pipeline.py pass, not the main per-item loop.
CANDIDATE_CLASSIFICATION_TYPE_TO_SCHEMA = {
    ObservationType.CANDIDATE_RISK_SIGNAL: CANDIDATE_RISK_SIGNAL_SCHEMA,
    ObservationType.CANDIDATE_EVIDENCE_CLASSIFICATION: CANDIDATE_EVIDENCE_CLASSIFICATION_SCHEMA,
}

# Milestone 4B — D2/D6 Candidate Qualifier Extraction (resolves
# M3-OD-01). These schemas belong to the SEPARATE stage-2 classifier call
# (extraction.pipeline.run_dimension_qualifier_classification /
# extraction.provider's propose_dimension_qualifiers), never merged into
# TOP_LEVEL_SCHEMA / the stage-1 monolithic extraction prompt (explicit
# architecture mandate — adding classification axes to the shared
# extraction prompt risked destabilizing previously-correct semantic
# typing, per Milestone 2C's own empirical evidence).
#
# Deliberately do NOT declare `source_span` OR `source_evidence_id` here
# (unlike CANDIDATE_RISK_SIGNAL_SCHEMA / CANDIDATE_EVIDENCE_CLASSIFICATION_
# SCHEMA above, which omit only `source_evidence_id`). This is the
# INHERITED GROUNDING exception approved for Milestone 4B only (see
# extraction.schemas.CandidateDimensionQualifier's docstring): the model
# classifies an already-accepted, already-exactly-grounded semantic
# observation, so it is never asked to reproduce that observation's span
# a second time. There is simply no field here for the model to occupy,
# exactly the same "no field exists" enforcement pattern SOURCE_SPAN_
# SCHEMA already uses for start_char/end_char.
D2_QUALIFIER_SCHEMA = {
    "type": "string",
    "enum": [
        "INTENDED_WORKFLOWS_OPERATING_NORMALLY",
        "NARROW_BREADTH_OR_CONCENTRATION",
        "WORKFLOWS_NOT_OCCURRING",
        "ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED",
    ],
}
# Milestone 4B v3 — atomic-predicate + deterministic composition
# architecture (response-contract/authority-boundary change, versioned as
# a family break: DIMENSION_QUALIFIER_PROMPT_VERSION = "v3", not a v2.x
# calibration). AUTOMATION_RELIABLE_LOW_LOGIN_OK is DELIBERATELY REMOVED
# from this enum — it is no longer a value the model may propose directly
# on the simple qualifier path at all (structurally, not just by prompt
# instruction). The model may instead propose independently-grounded
# CANDIDATE_D2_ATOMIC_PREDICATE_SCHEMA items on the sibling
# candidate_d2_atomic_predicates array key below; the application alone
# deterministically composes AUTOMATION_RELIABLE_LOW_LOGIN_OK if and only
# if all 3 required atomic predicates are present and grounded (see
# extraction.enums.DIMENSION_QUALIFIER_TYPE_TO_COMPOSED_QUALIFIER /
# extraction.pipeline.run_dimension_qualifier_classification). The
# remaining 4 values are unchanged from v2.1 — this is otherwise the same
# hardcoded-literal pattern as MECHANISM_SCHEMA / SEVERITY_TIER_SCHEMA
# above, kept in sync with domain/signals.py and extraction/schemas.py's
# _CANDIDATE_D2_QUALIFIERS (which still contains all 5 values — that
# tuple governs the FINAL CandidateDimensionQualifier object, produced
# either directly or via composition, and is unchanged by this
# architecture) by comment cross-reference. No explicit NONE/no-proposal
# value is declared — omitting an eligible observation from the
# candidate_d2_qualifiers array IS the abstention signal (unchanged
# architecture decision).

D6_QUALIFIER_SCHEMA = {
    "type": "string",
    "enum": [
        "APPROPRIATE_SPONSOR_COVERAGE",
        "CHAMPION_DEPARTURE_UNCONFIRMED",
        "SUCCESSION_UNCLEAR_OR_CONCENTRATED",
    ],
}
# Milestone 4B v3: CHAMPION_LOST_NO_SUCCESSOR is DELIBERATELY REMOVED from
# this enum for exactly the same reason AUTOMATION_RELIABLE_LOW_LOGIN_OK
# was removed from D2_QUALIFIER_SCHEMA above — see that schema's comment
# for the full rationale; mirrors it exactly, scoped to D6's own 2
# required atomic predicates instead of D2's 3. CHAMPION_DEPARTURE_
# UNCONFIRMED vs the now-composed-only CHAMPION_LOST_NO_SUCCESSOR remains
# the trickiest distinction (genuine uncertainty vs. a confirmed gap) —
# see prompts.py's stage-2 prompt for the explicit guidance given to the
# model on this distinction, now framed in terms of the CONFIRMED_
# CHAMPION_DEPARTURE atomic predicate rather than the compound qualifier
# itself.

# Milestone 4B v3 — the atomic predicate ID vocabularies, model-facing
# schema-level enums. Hardcoded literals mirroring D2_QUALIFIER_SCHEMA /
# D6_QUALIFIER_SCHEMA's own discipline; kept in sync with extraction.
# enums.D2_ATOMIC_PREDICATE_IDS / D6_ATOMIC_PREDICATE_IDS (the runtime
# single source of truth the pipeline's composer reads) by comment
# cross-reference AND by direct import/list() below, so this cannot
# silently drift the way the D2/D6 qualifier tuples are only kept in sync
# by convention.
_D2_ATOMIC_PREDICATE_ID_SCHEMA = {"type": "string", "enum": list(D2_ATOMIC_PREDICATE_IDS)}
_D6_ATOMIC_PREDICATE_ID_SCHEMA = {"type": "string", "enum": list(D6_ATOMIC_PREDICATE_IDS)}


def _atomic_predicate_schema(predicate_id_schema: dict) -> dict:
    """Milestone 4B v3. An atomic predicate item is deliberately NOT
    span-grounded the same way a top-level observation is: it carries its
    own `evidence_text` (a claimed exact substring of the single
    observation's own source_span.text, checked by extraction.validation
    before the object is ever constructed — see extraction.schemas.
    AtomicPredicateEvidence's docstring), but there is no
    supporting_observation_ref field here at all. Unlike CANDIDATE_D2_
    QUALIFIER_SCHEMA / CANDIDATE_D6_QUALIFIER_SCHEMA (which keep a
    supporting_observation_ref for schema/pipeline continuity even though
    an isolated call only ever has one possible index), an atomic
    predicate item has no need for one: there is exactly one observation
    in scope for the whole isolated call, and the pipeline attaches
    resolved_observation_id/dimension from its own call context, never
    from a model-supplied reference — a deliberately minimal model-facing
    surface for this new item type."""
    return {
        "type": "object",
        "properties": {
            "predicate_id": predicate_id_schema,
            "evidence_text": {"type": "string", "minLength": 1},
            "basis": BASIS_SCHEMA,
        },
        "required": ["predicate_id", "evidence_text", "basis"],
        "additionalProperties": False,
    }


CANDIDATE_D2_ATOMIC_PREDICATE_SCHEMA = _atomic_predicate_schema(_D2_ATOMIC_PREDICATE_ID_SCHEMA)
CANDIDATE_D6_ATOMIC_PREDICATE_SCHEMA = _atomic_predicate_schema(_D6_ATOMIC_PREDICATE_ID_SCHEMA)

# Per-channel restricted supporting-observation-ref schemas: D2 may only
# reference an ADOPTION_OBSERVATION; D6 may only reference a
# STAKEHOLDER_OBSERVATION. Structurally narrower than
# _SUPPORTING_OBSERVATION_REF_SCHEMA above (which allows all 7 semantic
# types, for CandidateRiskSignal/CandidateEvidenceClassification) — this
# is the schema-level half of grounding prohibition (c), "references a
# disallowed observation type" (see extraction.enums.RejectionReason.
# DIMENSION_QUALIFIER_REFERENCES_DISALLOWED_TYPE for the defense-in-depth
# runtime counterpart).
_D2_SUPPORTING_OBSERVATION_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "observation_type": {"type": "string", "enum": [ObservationType.ADOPTION_OBSERVATION.value]},
        "index": {"type": "integer", "minimum": 0},
    },
    "required": ["observation_type", "index"],
    "additionalProperties": False,
}
_D6_SUPPORTING_OBSERVATION_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "observation_type": {"type": "string", "enum": [ObservationType.STAKEHOLDER_OBSERVATION.value]},
        "index": {"type": "integer", "minimum": 0},
    },
    "required": ["observation_type", "index"],
    "additionalProperties": False,
}

# Milestone 4B: unlike stage-1's supporting_observation_ref.index (a
# position into the raw, un-deduped, mixed-type stage-1 output array,
# resolved via pipeline._resolve_array_ref's (array_key, idx) dict), a
# stage-2 index here means "position within the channel-homogeneous list
# of already-accepted observations handed to that stage-2 call" —
# resolved via a deliberately simpler, separate mechanism (pipeline.
# _resolve_dimension_qualifier_reference). Documented here, at the schema
# level, to avoid confusing future maintainers about which resolution
# mechanism applies where; the JSON shape (observation_type + index) is
# intentionally identical so the model-facing contract stays familiar.
CANDIDATE_D2_QUALIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "qualifier": D2_QUALIFIER_SCHEMA,
        "basis": BASIS_SCHEMA,
        "supporting_observation_ref": _D2_SUPPORTING_OBSERVATION_REF_SCHEMA,
    },
    "required": ["qualifier", "basis", "supporting_observation_ref"],
    "additionalProperties": False,
}
CANDIDATE_D6_QUALIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "qualifier": D6_QUALIFIER_SCHEMA,
        "basis": BASIS_SCHEMA,
        "supporting_observation_ref": _D6_SUPPORTING_OBSERVATION_REF_SCHEMA,
    },
    "required": ["qualifier", "basis", "supporting_observation_ref"],
    "additionalProperties": False,
}

# Kept separate from CANDIDATE_CLASSIFICATION_TYPE_TO_SCHEMA above for the
# same reason extraction.enums.DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY is
# kept separate from CANDIDATE_CLASSIFICATION_TYPE_TO_ARRAY_KEY: these
# items are validated by the stage-2 classifier's own pass
# (validation.validate_dimension_qualifier_shape), never the stage-1
# per-item loop or the 2C candidate-classification pass.
DIMENSION_QUALIFIER_TYPE_TO_SCHEMA = {
    ObservationType.CANDIDATE_D2_QUALIFIER: CANDIDATE_D2_QUALIFIER_SCHEMA,
    ObservationType.CANDIDATE_D6_QUALIFIER: CANDIDATE_D6_QUALIFIER_SCHEMA,
}

# Milestone 4B v3: the atomic-predicate item schema counterpart to
# DIMENSION_QUALIFIER_TYPE_TO_SCHEMA above, validated by validation.
# validate_atomic_predicate_shape — never the stage-1 per-item loop, the
# 2C candidate-classification pass, or DIMENSION_QUALIFIER_TYPE_TO_SCHEMA
# itself (that map's per-item schemas no longer accept the 2 compound
# qualifier values at all; this is a structurally separate item shape).
DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_SCHEMA = {
    ObservationType.CANDIDATE_D2_QUALIFIER: CANDIDATE_D2_ATOMIC_PREDICATE_SCHEMA,
    ObservationType.CANDIDATE_D6_QUALIFIER: CANDIDATE_D6_ATOMIC_PREDICATE_SCHEMA,
}

# Milestone 4B isolated-classifier architecture checkpoint: REPLACES the
# prior batched stage-2 envelope (a single call classifying every
# eligible observation in the run at once, both channels together) --
# removed outright, not kept as a second active path (approved
# architecture item C: "Replace the batched qualifier-provider method
# with the proposed isolated method rather than retaining two active
# classifier paths"). Each isolated call classifies exactly ONE already-
# accepted observation and sees only that one dimension's vocabulary, so
# its envelope holds AT MOST ONE item in exactly ONE (dimension-specific)
# array. Reuses the existing, UNCHANGED per-item schemas above
# (CANDIDATE_D2_QUALIFIER_SCHEMA / CANDIDATE_D6_QUALIFIER_SCHEMA) --
# neither the per-item shape, the qualifier vocabularies, nor
# supporting_observation_ref changed; only the envelope's cardinality and
# channel-count are new. `maxItems: 1` is a real, enforced schema
# constraint handed to the model as part of its forced-tool input_schema
# -- not just documentation -- so the model structurally cannot return
# two qualifiers for the single observation it was shown.
ISOLATED_D2_QUALIFIER_TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_d2_qualifiers": {"type": "array", "items": CANDIDATE_D2_QUALIFIER_SCHEMA, "maxItems": 1},
        # Milestone 4B v3: sibling channel for the 3 D2 atomic predicates
        # (RELIABLE_AUTOMATION_OPERATION / LOW_LOGIN_OR_MANUAL_ACTIVITY /
        # LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION). maxItems: 3 is a real,
        # enforced schema constraint (one slot per required predicate ID)
        # handed to the model as part of its forced-tool input_schema —
        # the model cannot return more items than there are predicates to
        # propose, though it may legitimately propose duplicates of the
        # same ID within that limit (rejected explicitly by the pipeline's
        # composer, never silently deduplicated at the schema level).
        "candidate_d2_atomic_predicates": {
            "type": "array", "items": CANDIDATE_D2_ATOMIC_PREDICATE_SCHEMA, "maxItems": 3,
        },
    },
    "required": [],
    "additionalProperties": False,
}
ISOLATED_D6_QUALIFIER_TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_d6_qualifiers": {"type": "array", "items": CANDIDATE_D6_QUALIFIER_SCHEMA, "maxItems": 1},
        # Milestone 4B v3: sibling channel for the 2 D6 atomic predicates
        # (CONFIRMED_CHAMPION_DEPARTURE / NO_SUCCESSOR_OR_CONTINUING_
        # COVERAGE). See candidate_d2_atomic_predicates above for the
        # maxItems rationale, mirrored here at D6's own required count.
        "candidate_d6_atomic_predicates": {
            "type": "array", "items": CANDIDATE_D6_ATOMIC_PREDICATE_SCHEMA, "maxItems": 2,
        },
    },
    "required": [],
    "additionalProperties": False,
}
# Keyed by the SAME ObservationType used throughout this module (mirrors
# DIMENSION_QUALIFIER_TYPE_TO_SCHEMA immediately above) so pipeline.py/
# provider.py can select the right envelope for a given isolated call by
# the same `obs_type`/dimension parameter they already thread through
# _build_dimension_qualifier, with no new lookup convention introduced.
ISOLATED_DIMENSION_QUALIFIER_TOP_LEVEL_SCHEMA = {
    ObservationType.CANDIDATE_D2_QUALIFIER: ISOLATED_D2_QUALIFIER_TOP_LEVEL_SCHEMA,
    ObservationType.CANDIDATE_D6_QUALIFIER: ISOLATED_D6_QUALIFIER_TOP_LEVEL_SCHEMA,
}

# LOOSE variants, mirroring LOOSE_TOP_LEVEL_SCHEMA's own rationale
# exactly: the stage-2 one-retry gate checks only the outer envelope
# (right key name, a list of at most one object), never the item's full
# per-type schema, so a malformed ITEM is handled by the ordinary
# per-item validation path (validate_dimension_qualifier_shape, itself
# unchanged), never by forcing a whole-call retry for a well-shaped
# envelope with one imperfect item.
ISOLATED_D2_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_d2_qualifiers": {"type": "array", "items": {"type": "object"}, "maxItems": 1},
        "candidate_d2_atomic_predicates": {"type": "array", "items": {"type": "object"}, "maxItems": 3},
    },
    "required": [],
    "additionalProperties": False,
}
ISOLATED_D6_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_d6_qualifiers": {"type": "array", "items": {"type": "object"}, "maxItems": 1},
        "candidate_d6_atomic_predicates": {"type": "array", "items": {"type": "object"}, "maxItems": 2},
    },
    "required": [],
    "additionalProperties": False,
}
ISOLATED_DIMENSION_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA = {
    ObservationType.CANDIDATE_D2_QUALIFIER: ISOLATED_D2_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA,
    ObservationType.CANDIDATE_D6_QUALIFIER: ISOLATED_D6_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA,
}


# Full top-level schema (recursively validates every item's own type
# schema too) — this is what's handed to the model as its forced tool
# `input_schema` (provider.py), so the provider itself is constrained to
# emit fully-conformant items wherever possible.
TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "objective_candidates": {"type": "array", "items": OBJECTIVE_CANDIDATE_SCHEMA},
        "stakeholder_observations": {"type": "array", "items": STAKEHOLDER_OBSERVATION_SCHEMA},
        "adoption_observations": {"type": "array", "items": ADOPTION_OBSERVATION_SCHEMA},
        "service_observations": {"type": "array", "items": SERVICE_OBSERVATION_SCHEMA},
        "commercial_observations": {"type": "array", "items": COMMERCIAL_OBSERVATION_SCHEMA},
        "experience_observations": {"type": "array", "items": EXPERIENCE_OBSERVATION_SCHEMA},
        "strategic_observations": {"type": "array", "items": STRATEGIC_OBSERVATION_SCHEMA},
        "missing_information_candidates": {"type": "array", "items": MISSING_INFORMATION_CANDIDATE_SCHEMA},
        "candidate_contradictions": {"type": "array", "items": CANDIDATE_CONTRADICTION_SCHEMA},
        "candidate_risk_signals": {"type": "array", "items": CANDIDATE_RISK_SIGNAL_SCHEMA},
        "candidate_evidence_classifications": {"type": "array", "items": CANDIDATE_EVIDENCE_CLASSIFICATION_SCHEMA},
    },
    "required": [],
    "additionalProperties": False,
}

# LOOSE top-level shape check — used by validation.validate_top_level_shape
# for the pipeline's step-6 "is this even a well-formed response" / one-
# retry gate. Deliberately checks ONLY the outer envelope (right key
# names, each a list of objects), NOT each item's full per-type schema —
# that recursive depth belongs to the per-item validation loop (step 7-8),
# so that ONE malformed item never triggers a whole-response retry/failure
# for observations that were otherwise perfectly valid (spec §4/§13:
# reject the offending claim, not the whole request).
_GENERIC_OBJECT_ARRAY = {"type": "array", "items": {"type": "object"}}
LOOSE_TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {key: _GENERIC_OBJECT_ARRAY for key in (
        "objective_candidates", "stakeholder_observations", "adoption_observations",
        "service_observations", "commercial_observations", "experience_observations",
        "strategic_observations", "missing_information_candidates", "candidate_contradictions",
        # Milestone 2C additions — see CANDIDATE_RISK_SIGNAL_SCHEMA /
        # CANDIDATE_EVIDENCE_CLASSIFICATION_SCHEMA above for the full
        # per-item shape; this envelope check stays intentionally loose.
        "candidate_risk_signals", "candidate_evidence_classifications",
    )},
    "required": [],
    "additionalProperties": False,
}

# Denylist for the recursive defense-in-depth scan (validation.py). Keys
# that would, if accepted, look like an attempt to set a governed CHDM
# conclusion directly. Checked case-insensitively against every dict key
# anywhere in the raw parsed response, BEFORE schema validation, so a
# violation can be classified and counted precisely (spec §15 "Boundary
# violations") rather than just surfacing as a generic schema error.
PROHIBITED_KEY_DENYLIST = frozenset({
    # Governed CHDM conclusions (spec §2 "AI must never set").
    "confirmed", "current_confirmed", "is_confirmed",
    "activated_severity", "activated", "risk_severity", "potential_severity",
    "dimension_state", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8a", "d8b",
    "objective_outcome", "value_evidence_basis",
    "dmeg", "decision_material_evidence_gap",
    "assessment_reliability", "reliability_level", "reliability",
    "operational_priority", "op", "op1", "op2", "op3", "opu",
    "evidence_review", "er", "er0", "er1",
    "disputed",
    "churn_probability", "renewal_probability", "churn_risk", "prediction",
    "recommended_action", "action_recommendation", "recommendation",
    "risk_mechanism_hint", "risk_mechanism", "cr_code",
    # System-generated metadata (Checkpoint 2A refinement 1 — the model
    # must have no field through which it can set or request these).
    "model_provider", "model_version", "extracted_at", "trace_id",
    "evidence_state", "observation_id",
})
