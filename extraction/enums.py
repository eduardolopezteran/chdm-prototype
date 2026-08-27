"""
Build Milestone 2 — extraction-local controlled vocabularies.

Deliberately separate from `domain.enums` (the Milestone 1 CHDM-governed
enum set, unmodified by Milestone 2). Nothing here is a CHDM methodology
object; these are extraction-pipeline concepts only (explicit-vs-inferred
provenance, the extraction record types, and why a candidate was
rejected). None of these values are consumed by any deterministic CHDM
rule.
"""

from enum import Enum


class InferenceBasis(str, Enum):
    """Milestone 2 spec §9 — explicit-vs-inferred distinction. Applies to
    every positive observation type. NOT used by MissingInformationCandidate,
    which has its own fixed basis (see MissingInformationBasis below) since
    "not found in reviewed evidence" is a different kind of claim than
    "explicitly stated" or "reasonably inferred from stated evidence"."""
    EXPLICIT = "EXPLICIT"
    INFERRED_CANDIDATE = "INFERRED_CANDIDATE"


class MissingInformationBasis(str, Enum):
    """Milestone 2A refinement 4: a MissingInformationCandidate must never
    claim information is absent from the customer account globally — only
    that it was not found in the specific evidence reviewed by this
    extraction request. This is the only permitted value; modeled as an
    enum (rather than a bare string) so the scope-relative framing is
    structurally enforced, not just a comment."""
    NOT_FOUND_IN_REVIEWED_EVIDENCE = "NOT_FOUND_IN_REVIEWED_EVIDENCE"


class ObservationType(str, Enum):
    """The extraction record types this milestone implements (spec §5).
    Used both as a discriminator and as the `observation_type` value inside
    a CandidateContradiction's cross-reference (see schemas.ObservationRef)."""
    OBJECTIVE_CANDIDATE = "OBJECTIVE_CANDIDATE"
    STAKEHOLDER_OBSERVATION = "STAKEHOLDER_OBSERVATION"
    ADOPTION_OBSERVATION = "ADOPTION_OBSERVATION"
    SERVICE_OBSERVATION = "SERVICE_OBSERVATION"
    COMMERCIAL_OBSERVATION = "COMMERCIAL_OBSERVATION"
    EXPERIENCE_OBSERVATION = "EXPERIENCE_OBSERVATION"
    STRATEGIC_OBSERVATION = "STRATEGIC_OBSERVATION"
    MISSING_INFORMATION_CANDIDATE = "MISSING_INFORMATION_CANDIDATE"
    CANDIDATE_CONTRADICTION = "CANDIDATE_CONTRADICTION"
    # Milestone 2C: candidate CHDM classification extraction (upstream of
    # Milestone 3A human confirmation). These are span-grounded like the
    # 7 positive types, but ALSO carry a required reference to the
    # semantic observation they interpret (see schemas.ObservationRef /
    # supporting_observation_ref) — a candidate classification never
    # stands alone as a "fact about the account," only as an
    # interpretation of an already-extracted observation. Never
    # confused with an activated CHDM conclusion: both remain
    # Current+Unverified and are never consumed by engine/evaluate.py.
    CANDIDATE_RISK_SIGNAL = "CANDIDATE_RISK_SIGNAL"
    CANDIDATE_EVIDENCE_CLASSIFICATION = "CANDIDATE_EVIDENCE_CLASSIFICATION"
    # Milestone 4B: D2/D6 candidate qualifier extraction. A SEPARATE,
    # second-stage classification pass (extraction.pipeline.
    # run_dimension_qualifier_classification), never part of the
    # Milestone 2/2C monolithic extraction prompt or run_extraction()
    # itself. Unlike CANDIDATE_RISK_SIGNAL / CANDIDATE_EVIDENCE_
    # CLASSIFICATION, these two types use INHERITED grounding: the model
    # never independently reproduces source_span, because it only ever
    # classifies already-accepted, already-exactly-grounded stage-1
    # semantic observations (Checkpoint 2A refinement 2 grounding
    # discipline is unchanged for every other type -- this is a narrow,
    # explicitly-documented exception scoped to these two types only; see
    # extraction.schemas.CandidateDimensionQualifier). Two channels
    # because D2 and D6 each expose only their own canonical CHDM
    # qualifier vocabulary (domain.signals.DIMENSION_QUALIFIERS) --
    # `dimension` itself is application-derived from which channel an
    # item came from and is never model-authored or model-facing.
    CANDIDATE_D2_QUALIFIER = "CANDIDATE_D2_QUALIFIER"
    CANDIDATE_D6_QUALIFIER = "CANDIDATE_D6_QUALIFIER"


# The model-facing JSON top-level key each positive observation type is
# emitted under (arrays), shared by json_schemas.py, validation.py and
# pipeline.py so there is exactly one place this mapping is declared.
OBSERVATION_TYPE_TO_ARRAY_KEY = {
    ObservationType.OBJECTIVE_CANDIDATE: "objective_candidates",
    ObservationType.STAKEHOLDER_OBSERVATION: "stakeholder_observations",
    ObservationType.ADOPTION_OBSERVATION: "adoption_observations",
    ObservationType.SERVICE_OBSERVATION: "service_observations",
    ObservationType.COMMERCIAL_OBSERVATION: "commercial_observations",
    ObservationType.EXPERIENCE_OBSERVATION: "experience_observations",
    ObservationType.STRATEGIC_OBSERVATION: "strategic_observations",
    ObservationType.MISSING_INFORMATION_CANDIDATE: "missing_information_candidates",
}
ARRAY_KEY_TO_OBSERVATION_TYPE = {v: k for k, v in OBSERVATION_TYPE_TO_ARRAY_KEY.items()}

# The 8 "positive" (span-grounded) types, i.e. everything except
# MissingInformationCandidate and CandidateContradiction, which have their
# own shapes (spec §4 vs §10/§11).
SPAN_GROUNDED_OBSERVATION_TYPES = frozenset(
    t for t in OBSERVATION_TYPE_TO_ARRAY_KEY if t != ObservationType.MISSING_INFORMATION_CANDIDATE
)

# Milestone 2C: the 7 span-grounded SEMANTIC types (excludes
# MissingInformationCandidate) that a CandidateRiskSignal /
# CandidateEvidenceClassification is permitted to cite as its
# supporting_observation_ref. Enforced structurally at the JSON-schema
# enum level (json_schemas.py's _SUPPORTING_OBSERVATION_REF_SCHEMA), not
# just here — this set is the single source of truth both schemas build
# their allowed-enum from, so it cannot drift. A candidate classification
# may never reference another candidate classification, a contradiction,
# or a missing-information candidate (absence is not evidence to
# interpret) — only a genuine positive semantic observation.
SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES = frozenset(
    t for t in OBSERVATION_TYPE_TO_ARRAY_KEY if t != ObservationType.MISSING_INFORMATION_CANDIDATE
)

# Milestone 2C: the two new candidate-classification array keys, kept in a
# map separate from OBSERVATION_TYPE_TO_ARRAY_KEY (rather than merged into
# it) because these two types are NOT resolvable supporting-observation
# targets themselves (see SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES above)
# and are not processed by the main per-item loop in pipeline.py — they
# require the full accepted-item index first, like CandidateContradiction.
CANDIDATE_CLASSIFICATION_TYPE_TO_ARRAY_KEY = {
    ObservationType.CANDIDATE_RISK_SIGNAL: "candidate_risk_signals",
    ObservationType.CANDIDATE_EVIDENCE_CLASSIFICATION: "candidate_evidence_classifications",
}

# Milestone 4B: the two D2/D6 candidate-qualifier channels, kept in their
# own map (mirroring CANDIDATE_CLASSIFICATION_TYPE_TO_ARRAY_KEY's own
# precedent) rather than merged into either map above, because these two
# types are produced by an entirely separate stage-2 classifier call
# (extraction.pipeline.run_dimension_qualifier_classification) operating
# on a different input shape (already-accepted AdoptionObservation /
# StakeholderObservation lists, not the raw stage-1 model output array)
# and resolved via a deliberately simpler, list-position-based mechanism
# -- NOT extraction.pipeline._resolve_array_ref's (array_key, idx) dict.
# See extraction.pipeline.run_dimension_qualifier_classification for the
# resolution mechanism itself.
DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY = {
    ObservationType.CANDIDATE_D2_QUALIFIER: "candidate_d2_qualifiers",
    ObservationType.CANDIDATE_D6_QUALIFIER: "candidate_d6_qualifiers",
}

# Milestone 4B v3 — atomic-predicate + deterministic composition
# architecture. For exactly the two compound qualifiers scoped to this
# architecture (AUTOMATION_RELIABLE_LOW_LOGIN_OK on D2, CHAMPION_LOST_NO_
# SUCCESSOR on D6), the model is structurally barred from proposing the
# compound qualifier name directly (removed from D2_QUALIFIER_SCHEMA /
# D6_QUALIFIER_SCHEMA's enum in json_schemas.py) and may instead emit
# 0-or-more independently-grounded ATOMIC PREDICATE items on a SIBLING
# envelope key. This map is the array-key counterpart to
# DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY immediately above, kept separate
# (never merged into it) for the same reason CANDIDATE_CLASSIFICATION_
# TYPE_TO_ARRAY_KEY / DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY are kept apart
# from each other: these are a structurally distinct model-output
# channel, resolved by extraction.pipeline.run_dimension_qualifier_
# classification's own composer logic, never by _build_dimension_
# qualifier's per-item loop directly.
DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_ARRAY_KEY = {
    ObservationType.CANDIDATE_D2_QUALIFIER: "candidate_d2_atomic_predicates",
    ObservationType.CANDIDATE_D6_QUALIFIER: "candidate_d6_atomic_predicates",
}

# The atomic predicate ID vocabulary for each channel — ALL required, no
# partial credit (approved architecture). Deliberately plain string
# tuples, not a formal Enum class, mirroring extraction.schemas.
# _CANDIDATE_D2_QUALIFIERS / _CANDIDATE_D6_QUALIFIERS's own literal-tuple
# convention for extraction-internal (non-CHDM-governed) vocabularies.
# These predicate IDs are NEVER part of CandidateDimensionQualifier.
# qualifier's governed vocabulary (schemas.py's _CANDIDATE_D2_QUALIFIERS/
# _CANDIDATE_D6_QUALIFIERS, unchanged) — they are a strictly upstream,
# extraction-internal concept that only ever feeds the deterministic
# composer below.
D2_ATOMIC_PREDICATE_IDS = (
    "RELIABLE_AUTOMATION_OPERATION",
    "LOW_LOGIN_OR_MANUAL_ACTIVITY",
    "LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION",
)
D6_ATOMIC_PREDICATE_IDS = (
    "CONFIRMED_CHAMPION_DEPARTURE",
    "NO_SUCCESSOR_OR_CONTINUING_COVERAGE",
)
DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_IDS = {
    ObservationType.CANDIDATE_D2_QUALIFIER: D2_ATOMIC_PREDICATE_IDS,
    ObservationType.CANDIDATE_D6_QUALIFIER: D6_ATOMIC_PREDICATE_IDS,
}

# The single, deterministic composition target for each channel's full
# atomic-predicate set. Composition is authoritative and mechanical: if
# and only if every ID in DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_
# IDS[channel] is present as independently-grounded evidence for the SAME
# observation in the SAME isolated call, the pipeline synthesizes exactly
# this qualifier value and routes it through the existing, unmodified
# _build_dimension_qualifier reference-resolution logic — never proposed
# by the model itself (see D2_QUALIFIER_SCHEMA / D6_QUALIFIER_SCHEMA's
# now-narrowed enum in json_schemas.py, which structurally excludes both
# of these two values from the direct-proposal path).
DIMENSION_QUALIFIER_TYPE_TO_COMPOSED_QUALIFIER = {
    ObservationType.CANDIDATE_D2_QUALIFIER: "AUTOMATION_RELIABLE_LOW_LOGIN_OK",
    ObservationType.CANDIDATE_D6_QUALIFIER: "CHAMPION_LOST_NO_SUCCESSOR",
}

# Milestone 4B D2 EXPLICIT-basis composition gate (architecture checkpoint,
# approved after Prompt v3.2 fixed its two targeted RELIABLE_AUTOMATION_
# OPERATION false positives -- Case 23's cadence-alone gap and Case 35
# Observation B's workflow-ownership gap -- but a live probe then showed
# Case 35 Observation A still completing the required 3-predicate D2 set
# via a DIFFERENT semantic-completion path: RELIABLE_AUTOMATION_OPERATION
# correctly EXPLICIT, but LOW_LOGIN_OR_MANUAL_ACTIVITY and LOW_ACTIVITY_
# EXPLAINED_BY_AUTOMATION both grounded to real substrings of the SAME
# reliability-only sentence, self-labeled INFERRED_CANDIDATE by the model,
# and genuinely not stated by that observation's own text. Two rounds of
# prompt-only calibration (v3.1, v3.2) each closed one specific instance
# of this shared-automation-vocabulary force-fit pattern without
# eliminating the underlying class -- direct empirical evidence that
# wording alone has a diminishing, non-convergent return here. Across all
# 3 live probes analyzed to date, every atomic predicate identified as
# semantically unsupported by its cited text was ALREADY self-labeled
# INFERRED_CANDIDATE by the model (5 for 5, zero observed counterexamples
# of a bad predicate mislabeled EXPLICIT) -- an empirical, replicated (but
# not proven-universal) signal already present in every model response,
# making a structural admission gate on that signal a natural extension
# of this architecture's existing exact-substring-grounding philosophy:
# convert a probabilistic behavior into a deterministic, application-
# enforced rule wherever the signal to do so is already reliably
# available, rather than continuing to chase it via prompt wording.
#
# Maps each compound-qualifier channel to whether ALL required atomic
# predicates must have basis == EXPLICIT (never INFERRED_CANDIDATE) to be
# ELIGIBLE for deterministic composition -- checked in addition to, not
# instead of, the existing completeness requirement (every required
# predicate ID present) and the existing exact-substring grounding
# requirement (unchanged). Scoped to D2 ONLY, per explicit architecture
# decision -- D6's CHAMPION_LOST_NO_SUCCESSOR is unaffected and continues
# composing from a complete grounded set regardless of basis, exactly as
# before. This gate governs COMPOSITION ELIGIBILITY only: it does NOT
# affect evidence collection -- every grounded atomic predicate, EXPLICIT
# or INFERRED_CANDIDATE, is still unconditionally recorded in
# ExtractionResult.dimension_qualifier_predicate_evidence (and therefore
# still visible via eval.metrics.dimension_qualifier_atomic_predicate_
# detail) even when the gate blocks composition -- collection and
# composition are, and remain, two separate steps in the composer.
#
# Residual risk (explicitly acknowledged, not eliminated): this control
# is only as reliable as the model's own self-reported basis field.
# Nothing here independently verifies that a predicate labeled EXPLICIT
# truly reflects a literal textual statement rather than a confidently
# mislabeled inference -- exact-substring grounding proves evidence_text
# is real, never that basis was honestly self-reported. This is a
# probabilistic risk REDUCTION grounded in the empirical pattern above,
# not a deterministic guarantee, and should continue to be monitored
# against future live results exactly like every other residual risk
# already documented in this architecture (see extraction/pipeline.py's
# v2 -> v3 history comment on grounding-fabrication risk).
#
# D6 extension (Milestone 4B normalized full-suite live run,
# prompt_v4_4b_dimqual_v3_2_normalized_full_eval1, disposed FAIL): this
# gate was originally scoped to D2 ONLY because, at approval time, D6's
# CHAMPION_LOST_NO_SUCCESSOR had zero observed instances of composing
# from a semantically-unsupported predicate -- D6 was left composing from
# a complete grounded set "regardless of basis" as a deliberate,
# evidence-based scoping decision, not a permanent architectural
# distinction between the two dimensions. The first full 42-case run
# against the normalized benchmark found the SAME failure class in D6:
# Case 24 composed CHAMPION_LOST_NO_SUCCESSOR from CONFIRMED_CHAMPION_
# DEPARTURE (EXPLICIT) + NO_SUCCESSOR_OR_CONTINUING_COVERAGE (INFERRED_
# CANDIDATE, grounded to "Marcus, our only point of contact" -- a
# plausible inference about sole-contact concentration, not an explicit
# statement that no successor or continuing coverage exists). This is
# structurally the identical shape as the D2 finding that motivated the
# original gate: a complete, genuinely-grounded predicate set, one leg of
# which the model itself flagged as an inference rather than a literal
# statement. There is no principled reason the two compound qualifiers
# should be governed differently once both have exhibited the same
# failure mode -- the scoping was always about observed evidence, not
# about D2 and D6 being architecturally different in this respect. D6 is
# therefore extended to require EXPLICIT basis for composition, on the
# same terms as D2: this governs COMPOSITION ELIGIBILITY only, and does
# NOT touch evidence collection, Prompt v3.2, atomic predicate
# definitions, schemas, exact-substring grounding, provenance, the
# "simple" (non-compound) D6 qualifiers, stage-1 extraction, or the
# benchmark -- all unchanged.
DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION = {
    ObservationType.CANDIDATE_D2_QUALIFIER: True,
    ObservationType.CANDIDATE_D6_QUALIFIER: True,
}


class RejectionReason(str, Enum):
    """Why a candidate never entered the accepted/governed-adjacent path
    (spec §13 failure handling). Used on ExtractionValidationFailure
    records — nothing is silently dropped; every rejection is recorded."""
    SCHEMA_INVALID = "SCHEMA_INVALID"
    BOUNDARY_VIOLATION = "BOUNDARY_VIOLATION"
    MISSING_SPAN = "MISSING_SPAN"
    SPAN_NOT_FOUND = "SPAN_NOT_FOUND"
    # Milestone 2B baseline fix: start_char/end_char are no longer model-
    # supplied (see json_schemas.py / validation.py). The application
    # derives them by searching for the model-supplied span text verbatim
    # in the cited evidence. Exactly one exact occurrence is required to
    # resolve deterministically; more than one is a genuine ambiguity the
    # system must not silently guess through (never auto-picks the first
    # match), so it is rejected and recorded distinctly from SPAN_NOT_FOUND.
    SPAN_AMBIGUOUS = "SPAN_AMBIGUOUS"
    UNKNOWN_EVIDENCE_ID = "UNKNOWN_EVIDENCE_ID"
    CONTRADICTION_REFERENCES_REJECTED_ITEM = "CONTRADICTION_REFERENCES_REJECTED_ITEM"
    # Milestone 2B.2 closure: CandidateContradiction integrity rule. A live
    # Prompt v2 run (case 12) exposed the model referencing the SAME
    # accepted observation for both observation_ref_a and
    # observation_ref_b -- the prior code had no check for this and
    # silently accepted it as a "valid" two-sided contradiction, even
    # though only one side was ever actually extracted. Checked in two
    # places (extraction/pipeline.py): immediately after reference
    # resolution (catches a direct same-index reference) and again after
    # deduplication (catches two originally-distinct references that
    # collapsed to the same canonical observation via dedup).
    CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE = "CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE"
    # A CandidateContradiction's two referenced observations must each be
    # traceable to source evidence (source_evidence_id or, for
    # MissingInformationCandidate, reviewed_evidence_ids). Every one of
    # the 8 real observation dataclasses guarantees this structurally, so
    # this should never fire in practice -- it exists as an explicit,
    # testable guard rather than an unchecked assumption.
    CONTRADICTION_OBSERVATION_NOT_TRACEABLE_TO_EVIDENCE = "CONTRADICTION_OBSERVATION_NOT_TRACEABLE_TO_EVIDENCE"
    # Fixed post-live-defect (Case 11, prompt_v4_optionb_2c_eval1 crash): a
    # contradiction's observation_ref_a/b cited an observation_type with
    # no resolvable array key (a Milestone 2C candidate-classification
    # type). json_schemas.py's _OBSERVATION_REF_SCHEMA enum now rejects
    # this at the schema gate (SCHEMA_INVALID, before reference
    # resolution ever runs) -- this reason exists for the defense-in-depth
    # path in pipeline._resolve_array_ref(), which previously raised an
    # uncaught KeyError instead of ever reaching a RejectionReason at all.
    CONTRADICTION_REFERENCES_UNSUPPORTED_TYPE = "CONTRADICTION_REFERENCES_UNSUPPORTED_TYPE"
    MALFORMED_TOP_LEVEL_OUTPUT = "MALFORMED_TOP_LEVEL_OUTPUT"
    # Milestone 2C: a CandidateRiskSignal / CandidateEvidenceClassification's
    # supporting_observation_ref must resolve to an item that both exists
    # and survived extraction validation. Mirrors
    # CONTRADICTION_REFERENCES_REJECTED_ITEM exactly, kept as a distinct
    # code (rather than reused) so 2B and 2C boundary-violation/rejection
    # counts never mix in reporting.
    CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM = "CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM"
    # Defense-in-depth, mirroring CONTRADICTION_OBSERVATION_NOT_TRACEABLE_
    # TO_EVIDENCE: every type in SUPPORTING_OBSERVATION_REF_ALLOWED_TYPES
    # structurally guarantees source_evidence_id, and the JSON schema's
    # enum already prevents the model from citing a disallowed type — so
    # this should never fire in practice. It exists as an explicit,
    # testable guard rather than an unchecked assumption, exactly like
    # its Milestone 2B.2 predecessor.
    CANDIDATE_CLASSIFICATION_REFERENCED_OBSERVATION_NOT_TRACEABLE = (
        "CANDIDATE_CLASSIFICATION_REFERENCED_OBSERVATION_NOT_TRACEABLE"
    )
    # Symmetric defense-in-depth counterpart to
    # CONTRADICTION_REFERENCES_UNSUPPORTED_TYPE above, for
    # supporting_observation_ref (CandidateRiskSignal /
    # CandidateEvidenceClassification). _SUPPORTING_OBSERVATION_REF_SCHEMA
    # already restricts the model-facing enum correctly, so this should
    # never fire in practice -- exists so pipeline._resolve_array_ref()
    # can never surface a raw KeyError from this call site either.
    CANDIDATE_CLASSIFICATION_REFERENCES_UNSUPPORTED_TYPE = "CANDIDATE_CLASSIFICATION_REFERENCES_UNSUPPORTED_TYPE"
    # Milestone 4B: the 4 explicit grounding prohibitions approved for the
    # stage-2 D2/D6 qualifier classifier's inherited-grounding reference
    # resolution (extraction.pipeline.run_dimension_qualifier_
    # classification / _resolve_dimension_qualifier_reference). Kept as 4
    # distinct, individually-raised, individually-testable codes rather
    # than reusing the Milestone 2B/2C codes above, so 4B's inherited-
    # grounding checks never mix with 2B/2C's independent-span-grounding
    # checks in reporting -- these two grounding models are deliberately
    # different (see extraction.schemas.CandidateDimensionQualifier) and
    # must remain distinguishable in every rejection record.
    #
    # (a) supporting_observation_ref cites an observation_type outside
    # the channel's own allowed type (D2 -> AdoptionObservation only, D6
    # -> StakeholderObservation only). Enforced structurally at the JSON-
    # schema enum level already (per-channel _SUPPORTING_OBSERVATION_REF_
    # SCHEMA-equivalent) -- this is the defense-in-depth path, mirroring
    # CANDIDATE_CLASSIFICATION_REFERENCES_UNSUPPORTED_TYPE.
    DIMENSION_QUALIFIER_REFERENCES_DISALLOWED_TYPE = "DIMENSION_QUALIFIER_REFERENCES_DISALLOWED_TYPE"
    # (b) supporting_observation_ref.index does not resolve to any item in
    # the channel-homogeneous accepted-observation list handed to the
    # stage-2 call (out of range, or the list was empty). Mirrors
    # CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM's "not found"
    # half, but kept distinct because stage-2 resolution is list-position
    # based, not the stage-1 (array_key, idx) dict lookup.
    DIMENSION_QUALIFIER_REFERENCE_NOT_FOUND = "DIMENSION_QUALIFIER_REFERENCE_NOT_FOUND"
    # (c) supporting_observation_ref resolves to an item that did NOT
    # survive stage-1 extraction validation (i.e. is not actually present
    # in that run's finished ExtractionResult.accepted). Structurally this
    # should never fire, since stage 2 is only ever handed already-
    # accepted items to index into -- exists as an explicit, testable
    # guard rather than an unchecked assumption, exactly like its
    # Milestone 2B.2/2C predecessors.
    DIMENSION_QUALIFIER_REFERENCES_REJECTED_ITEM = "DIMENSION_QUALIFIER_REFERENCES_REJECTED_ITEM"
    # (d) supporting_observation_ref resolves to an item that is not
    # itself exactly span-grounded (missing/invalid source_span or
    # source_evidence_id) -- i.e. there is nothing valid to inherit
    # grounding FROM. Every real AdoptionObservation/StakeholderObservation
    # structurally guarantees this, so this should never fire in
    # practice; exists as the final explicit, testable guard completing
    # the 4 approved inherited-grounding prohibitions.
    DIMENSION_QUALIFIER_REFERENCES_UNGROUNDED_OBJECT = "DIMENSION_QUALIFIER_REFERENCES_UNGROUNDED_OBJECT"
    # Milestone 4B v3 — atomic-predicate + deterministic composition
    # architecture, approved after the self-attestation design (exact-
    # substring grounding of a directly-proposed compound qualifier) was
    # rejected as insufficient: exact grounding proves a cited span is
    # REAL, never that it actually ESTABLISHES the predicate's specific
    # semantic content. These two new reasons are the atomic-predicate
    # path's own equivalent of the 4 inherited-grounding prohibitions
    # above, kept as distinct, individually-raised, individually-testable
    # codes for the same reporting-hygiene reason those 4 are kept
    # distinct from the 2B/2C codes: an atomic predicate's grounding model
    # (exact substring of the SAME observation's own source_span.text,
    # never a supporting_observation_ref resolution) is different in kind
    # from all of them and must remain distinguishable in every rejection
    # record.
    #
    # (e) An atomic predicate's `evidence_text` does not appear verbatim
    # as an exact substring of the single observation's own
    # source_span.text. No fuzzy/normalized/partial matching — mirrors
    # validation.resolve_source_span's own exact-match discipline. This
    # is a per-predicate rejection only; it never blocks any OTHER
    # predicate proposed in the same call for the same observation, and
    # never triggers the top-level envelope repair-retry (that retry is
    # reserved for malformed/invalid JSON shape only, exactly like the 4
    # existing grounding prohibitions above — semantic insufficiency is
    # never retried).
    DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED = "DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED"
    # (f) The SAME predicate_id was proposed more than once for the SAME
    # observation within the SAME isolated call. Duplicates are never
    # silently deduplicated (explicit user instruction) — the first
    # occurrence is evaluated normally (grounded or rejected on its own
    # merits); every subsequent occurrence of that same predicate_id is
    # rejected under this reason and recorded, so a duplicate attempt is
    # always visible in the audit trail rather than quietly discarded.
    DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE = "DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE"
