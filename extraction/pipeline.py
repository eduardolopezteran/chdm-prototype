"""
Build Milestone 2 — the extraction pipeline (spec §6), implementing all
11 steps:

  1. accept raw evidence            -> caller supplies EvidenceObject tuple
  2. create Evidence Object          -> caller's responsibility (domain.evidence.EvidenceObject)
  3. prepare extraction request       -> prompts.build_system_prompt / build_user_message
  4. invoke model                      -> provider.extract()
  5. receive structured output          -> raw dict
  6. schema-validate                     -> validation.validate_top_level_shape, one retry
  7. validate evidence/source-span linkage -> validation.build_observation
  8. reject malformed/unsupported           -> ExtractionValidationFailure records, nothing silently dropped
  9. deduplicate                              -> dedup.deduplicate
  10. persist/return as Current+Unverified     -> ExtractionSystemFields attachment
  11. create trace records                      -> domain.trace_record.TraceRecord

No human confirmation happens here (out of scope, spec §6). Nothing here
promotes an observation past CURRENT_UNVERIFIED. This function never
raises for provider/schema-level failures (spec §13: "Return
extraction-service failure without changing deterministic assessment
state") — ModelServiceError and a still-malformed-after-retry top-level
response both come back as `ExtractionResult.request_failure`, a plain
description, so callers never need a try/except around normal usage.
Per-item failures (a single claim's bad span, a boundary-violation
attempt, an unknown evidence id, ...) are always recorded in
`ExtractionResult.rejected` — never silently dropped (spec §4, §13).

Milestone 2B.2 closure update: a live Prompt v2 run exposed a gap in
candidate-contradiction handling -- a CandidateContradiction whose
observation_ref_a and observation_ref_b both resolved to the SAME
accepted observation was silently accepted as a "valid" two-sided
contradiction. Reference resolution now additionally enforces that a
contradiction (1) has an existing reference A, (2) has an existing
reference B, (3) A != B, checked both immediately after resolution and
again after deduplication (since dedup can independently collapse two
originally-distinct references onto the same canonical observation),
and (5) both referenced observations are traceable to source evidence.
(4) "both observations survived extraction validation" was already
guaranteed structurally -- `accepted_by_ref` only ever contains items
that passed `build_observation`. A contradiction failing any of these
is rejected with a specific RejectionReason and recorded in
`ExtractionResult.rejected`, exactly like any other rejected item --
never silently retained, never patched by inferring or fabricating a
missing second observation. This is an extraction-object integrity
rule; it does not touch CHDM contradiction logic downstream (D1,
DMEG, Disputed, etc. all remain deterministic/human-governed, per
`CandidateContradiction.__post_init__`'s existing status invariant).

Milestone 2C update: candidate_risk_signals / candidate_evidence_
classifications are handled with the same "resolve after the main loop"
ordering as candidate_contradictions (both need the full accepted-item
index), but additionally derive their own source_evidence_id and
source_span grounding from whichever observation their required
supporting_observation_ref resolves to — never from a model-stated
evidence id (implementation constraint 1 of the Milestone 2C checkpoint).
Unlike contradictions, these two types ARE deduplicated (spec-equivalent
to the 7 positive types), so their supporting_observation_ref is
re-pointed through canonical_map in a dedicated post-dedup step, exactly
mirroring how contradictions are re-pointed. See
`_build_candidate_classification` below for the per-item logic.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional

import jsonschema

from domain.enums import DimensionCode, EvidenceState
from domain.evidence import EvidenceObject
from domain.reason_code import ReasonCode
from domain.trace_record import TraceRecord

from .dedup import DedupAuditRecord, deduplicate
from .enums import (
    ARRAY_KEY_TO_OBSERVATION_TYPE, CANDIDATE_CLASSIFICATION_TYPE_TO_ARRAY_KEY,
    DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY, DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_ARRAY_KEY,
    DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_IDS, DIMENSION_QUALIFIER_TYPE_TO_COMPOSED_QUALIFIER,
    DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION,
    InferenceBasis, OBSERVATION_TYPE_TO_ARRAY_KEY, ObservationType, RejectionReason,
)
from .errors import ItemRejected, ModelServiceError
from .provider import ExtractionProvider
from .schemas import (
    CANDIDATE_CLASSIFICATION_TYPE_TO_DATACLASS, AdoptionObservation, AtomicPredicateEvidence,
    CandidateContradiction, CandidateDimensionQualifier, CandidateRiskSignal, ExtractionSystemFields,
    ObservationRef, StakeholderObservation,
)
from .validation import (
    atomic_predicate_evidence_grounded, build_observation, resolve_source_span,
    validate_atomic_predicate_shape, validate_candidate_classification_shape,
    validate_contradiction_item_shape, validate_dimension_qualifier_shape,
    validate_isolated_dimension_qualifier_top_level_shape, validate_top_level_shape,
)

_id_counter = itertools.count(1)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{next(_id_counter):06d}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class ExtractionValidationFailure:
    """Every rejected candidate, with enough context to audit why (spec
    §4/§13: reject rather than silently retain).

    Milestone 4B v3 evaluator-provenance checkpoint: `resolved_observation_
    id` / `dimension` are OPTIONAL, additive fields (default None), used
    ONLY by the atomic-predicate rejection paths in run_dimension_
    qualifier_classification (ungrounded evidence, duplicate predicate
    ID) -- every other rejection call site in this module leaves both at
    their default. This is pure audit metadata: it does not change which
    items are rejected, why, or when -- only which observation/dimension
    a rejection is attributable to, so a live-eval report can correlate a
    rejected atomic predicate back to the specific observation its
    isolated call was for (needed when a case has more than one eligible
    observation on the same channel)."""
    observation_type: str  # array key, "TOP_LEVEL", or "candidate_contradictions"
    reason: RejectionReason
    detail: str
    raw_item: Any = None
    resolved_observation_id: Optional[str] = None
    dimension: Optional[DimensionCode] = None


@dataclass(frozen=True)
class ExtractionResult:
    accepted: tuple  # the 7 positive types + MissingInformationCandidate, all Current+Unverified
    candidate_contradictions: tuple  # CandidateContradiction, status always "CANDIDATE"
    # Milestone 2C: kept as their own dedicated fields, mirroring
    # candidate_contradictions, rather than folded into `accepted` — that
    # keeps `accepted`'s contract (exactly the 8 Milestone 2B types)
    # unchanged for every existing caller/test.
    candidate_risk_signals: tuple  # CandidateRiskSignal, Current+Unverified, never activated_severity
    candidate_evidence_classifications: tuple  # CandidateEvidenceClassification, Current+Unverified
    rejected: tuple  # ExtractionValidationFailure, one per rejected item
    dedup_audit: tuple  # DedupAuditRecord — audit trail for every collapsed duplicate
    traces: tuple  # TraceRecord, one per accepted observation/contradiction/candidate classification
    request_failure: Optional[str] = None  # set only if the WHOLE request failed (provider or malformed-after-retry)
    # Milestone 4B — D2/D6 Candidate Qualifier Extraction. Populated only
    # by the SEPARATE run_dimension_qualifier_classification() pass below
    # (never by run_extraction() itself, which is completely unchanged —
    # these three fields simply default to "stage 2 has not run yet" for
    # every existing caller/test). CandidateDimensionQualifier,
    # Current+Unverified, never a governed DimensionState/
    # DimensionQualifierSignal.
    candidate_d2_qualifiers: tuple = ()
    candidate_d6_qualifiers: tuple = ()
    # Milestone 4B isolated-classifier architecture checkpoint (item D):
    # per-observation failure detail — one DimensionQualifierFailure per
    # eligible observation whose ISOLATED classifier call did not
    # complete successfully (ModelServiceError, or still malformed after
    # one repair retry), keyed to that observation's own
    # resolved_observation_id. This is the SOLE authoritative failure
    # record. A failure here for one observation never removes or blocks
    # a qualifier successfully produced for any other observation in the
    # same run — every isolated call is independent. Empty tuple means
    # every isolated call in this run either produced a candidate or
    # abstained successfully (or stage 2 has not run / had nothing
    # eligible).
    dimension_qualifier_failures: tuple = ()
    # PRESERVED for backward compatibility (still read the same way by
    # eval/metrics.py's aggregate counter and eval/run_eval.py's per-case
    # report) but its populated state is now DERIVED from
    # dimension_qualifier_failures above, never set independently — see
    # run_dimension_qualifier_classification's docstring. Distinguishes
    # "at least one isolated classifier call in this run failed" (this
    # field set) from "every isolated call in this run either produced a
    # qualifier or abstained successfully" (this field None) — the
    # per-observation detail in dimension_qualifier_failures is always
    # the source of truth; this is a summary view over it, not a second
    # one. None means either stage 2 has not run yet, or it ran and every
    # individual call succeeded (whether or not any of them proposed
    # anything).
    dimension_qualifier_stage_failure: Optional[str] = None
    # Milestone 4B v3 — atomic-predicate + deterministic composition
    # architecture. EVERY grounded AtomicPredicateEvidence proposed across
    # every isolated call in this run, for BOTH channels, whether or not
    # the full required predicate set for its observation was ever
    # complete — an incomplete set's evidence is preserved here for audit
    # exactly like a complete set's is, even though only a complete set
    # additionally produces an entry in candidate_d2_qualifiers /
    # candidate_d6_qualifiers. This is provenance/audit metadata only,
    # never itself a governed object — see extraction.schemas.
    # AtomicPredicateEvidence's own docstring for the full rationale.
    # Empty tuple means either stage 2 has not run, or no isolated call in
    # this run proposed any grounded atomic predicate.
    dimension_qualifier_predicate_evidence: tuple = ()


_EXTRACTION_REASON = ReasonCode(
    code="EXTRACTION-OBSERVATION-ACCEPTED",
    governing_object_id="EXTRACTION-PIPELINE-NOT-A-CHDM-RULE",
    human_readable_text=(
        "AI-extracted candidate observation validated (schema + exact source-span "
        "resolution) and admitted as Current+Unverified. This cites the Milestone 2 "
        "extraction pipeline, not a governed CHDM methodology rule -- nothing "
        "governed has happened yet at this stage."
    ),
)


def _empty_result(rejected: list, request_failure: str) -> ExtractionResult:
    return ExtractionResult(
        accepted=(),
        candidate_contradictions=(),
        candidate_risk_signals=(),
        candidate_evidence_classifications=(),
        rejected=tuple(rejected),
        dedup_audit=(),
        traces=(),
        request_failure=request_failure,
    )


def run_extraction(
    evidence_batch: tuple[EvidenceObject, ...],
    provider: ExtractionProvider,
) -> ExtractionResult:
    evidence_text_by_id = {e.evidence_id: e.indicator_observation for e in evidence_batch}
    rejected: list[ExtractionValidationFailure] = []

    # ---- 3-5. prepare request / invoke model / receive output ----
    try:
        raw = provider.extract(evidence_batch)
    except ModelServiceError as e:
        return _empty_result(rejected, f"Model service failure: {e}")

    # ---- 6. schema-validate, one repair/retry permitted ----
    top_level_error = _check_top_level(raw)
    if top_level_error is not None:
        reason, detail = top_level_error
        rejected.append(ExtractionValidationFailure("TOP_LEVEL", reason, f"initial attempt: {detail}"))
        try:
            raw_retry = provider.extract(evidence_batch, repair_hint="repair")
        except ModelServiceError as e:
            return _empty_result(rejected, f"Model service failure on repair retry: {e}")
        retry_error = _check_top_level(raw_retry)
        if retry_error is not None:
            retry_reason, retry_detail = retry_error
            rejected.append(ExtractionValidationFailure("TOP_LEVEL", retry_reason, f"repair retry: {retry_detail}"))
            return _empty_result(
                rejected,
                f"Top-level extraction output still invalid after one repair retry: {retry_detail}",
            )
        raw = raw_retry

    # ---- 7-8. per-item validation, unsupported/malformed candidates rejected ----
    accepted_by_ref: dict[tuple[str, int], Any] = {}
    for array_key, obs_type in ARRAY_KEY_TO_OBSERVATION_TYPE.items():
        for idx, raw_item in enumerate(raw.get(array_key) or ()):
            try:
                obs = build_observation(obs_type, raw_item, evidence_text_by_id)
                accepted_by_ref[(array_key, idx)] = obs
            except ItemRejected as e:
                rejected.append(ExtractionValidationFailure(array_key, e.reason, e.detail, e.raw_item))

    # ---- candidate contradictions: validate shape, resolve refs ----
    resolved_contradictions: list[tuple[dict, tuple, tuple]] = []
    for raw_contra in raw.get("candidate_contradictions") or ():
        try:
            validate_contradiction_item_shape(raw_contra)
        except ItemRejected as e:
            rejected.append(ExtractionValidationFailure("candidate_contradictions", e.reason, e.detail, e.raw_item))
            continue

        try:
            key_a = _resolve_array_ref(raw_contra["observation_ref_a"])
            key_b = _resolve_array_ref(raw_contra["observation_ref_b"])
        except _UnsupportedReferenceType as e:
            rejected.append(ExtractionValidationFailure(
                "candidate_contradictions", RejectionReason.CONTRADICTION_REFERENCES_UNSUPPORTED_TYPE,
                f"observation_ref cites an unsupported reference type: {e.obs_type.value!r} -- only the "
                "7 span-grounded semantic observation types plus MissingInformationCandidate may be "
                "referenced by a contradiction (never a candidate risk signal, candidate evidence "
                "classification, or another contradiction)",
                raw_contra,
            ))
            continue
        if key_a not in accepted_by_ref or key_b not in accepted_by_ref:
            missing = key_a if key_a not in accepted_by_ref else key_b
            rejected.append(ExtractionValidationFailure(
                "candidate_contradictions", RejectionReason.CONTRADICTION_REFERENCES_REJECTED_ITEM,
                f"references an item that was rejected or does not exist: {missing}",
                raw_contra,
            ))
            continue

        # ---- Milestone 2B.2 closure: CandidateContradiction integrity
        # rule. A contradiction must reference two DISTINCT accepted
        # observations, each traceable to source evidence. Never infer or
        # fabricate a missing second observation -- reject with a
        # specific, auditable reason instead of silently retaining a
        # one-sided "contradiction".
        if key_a == key_b:
            rejected.append(ExtractionValidationFailure(
                "candidate_contradictions", RejectionReason.CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE,
                f"observation_ref_a and observation_ref_b both resolve to the same accepted "
                f"observation ({key_a}) -- a contradiction must reference two distinct observations",
                raw_contra,
            ))
            continue

        untraceable_refs = [
            ref_name for ref_name, key in (("observation_ref_a", key_a), ("observation_ref_b", key_b))
            if _source_ref(accepted_by_ref[key]) == "UNKNOWN_SOURCE"
        ]
        if untraceable_refs:
            rejected.append(ExtractionValidationFailure(
                "candidate_contradictions", RejectionReason.CONTRADICTION_OBSERVATION_NOT_TRACEABLE_TO_EVIDENCE,
                f"referenced observation(s) not traceable to source evidence: {untraceable_refs}",
                raw_contra,
            ))
            continue

        resolved_contradictions.append((raw_contra, key_a, key_b))

    # ---- Milestone 2C: candidate risk signals / evidence classifications.
    # Validate shape, resolve the required supporting_observation_ref
    # against the same accepted_by_ref index contradictions use, and
    # derive BOTH source_evidence_id and the exact span-grounding from
    # that reference (never from a model-stated evidence id — Milestone
    # 2C implementation constraint 1). Like contradictions, this must run
    # after the main per-item loop (needs the full accepted-item index)
    # and before dedup/system-metadata attachment. ----
    resolved_candidate_classifications: list[tuple[Any, tuple]] = []  # (provisional_obj, referenced_key)
    for obs_type, array_key in CANDIDATE_CLASSIFICATION_TYPE_TO_ARRAY_KEY.items():
        for raw_item in raw.get(array_key) or ():
            try:
                obj, ref_key = _build_candidate_classification(
                    obs_type, raw_item, accepted_by_ref, evidence_text_by_id,
                )
                resolved_candidate_classifications.append((obj, ref_key))
            except ItemRejected as e:
                rejected.append(ExtractionValidationFailure(array_key, e.reason, e.detail, e.raw_item))

    # ---- 10. attach system metadata (Current+Unverified, provider/version/timestamp/trace id) ----
    now = datetime.now(timezone.utc)
    finalized_by_ref: dict[tuple[str, int], Any] = {}
    for ref, obs in accepted_by_ref.items():
        system = ExtractionSystemFields(
            observation_id=_new_id("OBS"),
            model_provider=provider.provider_name,
            model_version=provider.model_version,
            extracted_at=now,
            trace_id=_new_id("TRACE"),
            evidence_state=EvidenceState.CURRENT_UNVERIFIED,
        )
        finalized_by_ref[ref] = replace(obs, system=system)

    finalized_candidate_classifications: list[tuple[Any, tuple]] = []
    for obj, ref_key in resolved_candidate_classifications:
        system = ExtractionSystemFields(
            observation_id=_new_id("CAND"),
            model_provider=provider.provider_name,
            model_version=provider.model_version,
            extracted_at=now,
            trace_id=_new_id("TRACE"),
            evidence_state=EvidenceState.CURRENT_UNVERIFIED,
        )
        finalized_candidate_classifications.append((replace(obj, system=system), ref_key))

    # ---- 9. deduplicate (audit-preserving) ----
    # Milestone 2C: the 2 new candidate-classification types are
    # deduplicated together with everything else — dedup.py dispatches by
    # dataclass type internally, so mixing them into one call is safe and
    # keeps a single audit trail (spec §12).
    kept, canonical_map, dedup_audit = deduplicate(
        tuple(finalized_by_ref.values()) + tuple(o for o, _ in finalized_candidate_classifications)
    )

    # ---- Milestone 2C: re-point each surviving candidate classification's
    # supporting_observation_ref to its supporting observation's final
    # canonical id (post-dedup), exactly mirroring how contradictions are
    # re-pointed through canonical_map below. A candidate classification
    # that was ITSELF collapsed as a duplicate is simply dropped here —
    # its dedup_audit record already accounts for it, same as any other
    # deduplicated observation. ----
    kept_by_id = {o.system.observation_id: o for o in kept}
    candidate_risk_signals: list = []
    candidate_evidence_classifications: list = []
    for obj, ref_key in finalized_candidate_classifications:
        survivor = kept_by_id.get(obj.system.observation_id)
        if survivor is None:
            continue
        referenced_finalized = finalized_by_ref[ref_key]
        resolved_id = canonical_map.get(
            referenced_finalized.system.observation_id, referenced_finalized.system.observation_id,
        )
        final_obj = replace(survivor, resolved_observation_id=resolved_id)
        if isinstance(final_obj, CandidateRiskSignal):
            candidate_risk_signals.append(final_obj)
        else:
            candidate_evidence_classifications.append(final_obj)

    # `accepted` keeps its original Milestone 2B contract (exactly the 8
    # positive/missing-information types) — the 2 Milestone 2C types are
    # excluded here and returned via their own dedicated fields instead,
    # even though they shared the same dedup() call above.
    _candidate_classification_ids = {o.system.observation_id for o, _ in finalized_candidate_classifications}
    accepted_kept = tuple(o for o in kept if o.system.observation_id not in _candidate_classification_ids)

    # ---- finalize contradictions, re-pointed through canonical dedup map ----
    contradictions = []
    for raw_contra, key_a, key_b in resolved_contradictions:
        obs_a, obs_b = finalized_by_ref[key_a], finalized_by_ref[key_b]
        canonical_a = canonical_map.get(obs_a.system.observation_id, obs_a.system.observation_id)
        canonical_b = canonical_map.get(obs_b.system.observation_id, obs_b.system.observation_id)

        # Milestone 2B.2 closure: second, structurally distinct place the
        # same integrity rule can be violated. key_a and key_b were
        # confirmed distinct above, but deduplicate() can independently
        # collapse two originally-distinct observations into the same
        # canonical survivor (spec §12) -- if that happens here, this
        # contradiction would end up referencing one real observation
        # twice, exactly the defect this rule exists to prevent.
        if canonical_a == canonical_b:
            rejected.append(ExtractionValidationFailure(
                "candidate_contradictions", RejectionReason.CONTRADICTION_SAME_OBSERVATION_REFERENCED_TWICE,
                f"observation_ref_a and observation_ref_b referenced distinct extracted items, "
                f"but both collapsed to the same canonical observation ({canonical_a}) during "
                f"deduplication -- a contradiction must reference two distinct observations",
                raw_contra,
            ))
            continue

        system = ExtractionSystemFields(
            observation_id=_new_id("CONTRA"),
            model_provider=provider.provider_name,
            model_version=provider.model_version,
            extracted_at=now,
            trace_id=_new_id("TRACE"),
            evidence_state=EvidenceState.CURRENT_UNVERIFIED,
        )
        contradictions.append(CandidateContradiction(
            observation_ref_a=_ref_from_key(key_a),
            observation_ref_b=_ref_from_key(key_b),
            conflict_description=raw_contra["conflict_description"],
            methodology_construct_hint=raw_contra.get("methodology_construct_hint"),
            resolved_observation_id_a=canonical_a,
            resolved_observation_id_b=canonical_b,
            system=system,
        ))

    # ---- 11. trace records ----
    traces = []
    for obs in accepted_kept:
        traces.append(TraceRecord(
            trace_id=obs.system.trace_id,
            subject_object_ref=obs.system.observation_id,
            chain=(_source_ref(obs),),
            methodology_version="0.1",
            reason_code=_EXTRACTION_REASON,
            model_version=f"{obs.system.model_provider}:{obs.system.model_version}",
        ))
    for c in contradictions:
        traces.append(TraceRecord(
            trace_id=c.system.trace_id,
            subject_object_ref=c.system.observation_id,
            chain=(c.resolved_observation_id_a, c.resolved_observation_id_b),
            methodology_version="0.1",
            reason_code=_EXTRACTION_REASON,
            model_version=f"{c.system.model_provider}:{c.system.model_version}",
        ))
    for cc in (*candidate_risk_signals, *candidate_evidence_classifications):
        traces.append(TraceRecord(
            trace_id=cc.system.trace_id,
            subject_object_ref=cc.system.observation_id,
            chain=(cc.source_evidence_id, cc.resolved_observation_id),
            methodology_version="0.1",
            reason_code=_EXTRACTION_REASON,
            model_version=f"{cc.system.model_provider}:{cc.system.model_version}",
        ))

    return ExtractionResult(
        accepted=accepted_kept,
        candidate_contradictions=tuple(contradictions),
        candidate_risk_signals=tuple(candidate_risk_signals),
        candidate_evidence_classifications=tuple(candidate_evidence_classifications),
        rejected=tuple(rejected),
        dedup_audit=dedup_audit,
        traces=tuple(traces),
        request_failure=None,
    )


def _build_candidate_classification(
    observation_type: ObservationType,
    raw_item: dict,
    accepted_by_ref: dict,
    evidence_text_by_id: dict,
):
    """Milestone 2C. Validate shape, resolve the required
    supporting_observation_ref against the already-accepted 7-type index,
    derive source_evidence_id from that reference (never model-stated —
    implementation constraint 1), and resolve the model's own
    source_span.text against that SAME evidence item's text (Checkpoint
    2A refinement 2 grounding discipline, unchanged). Returns a
    provisional (system=pending) dataclass instance plus the
    (array_key, index) key of its supporting observation, for post-dedup
    re-pointing in run_extraction. Raises ItemRejected on any failure —
    nothing is silently retained."""
    validate_candidate_classification_shape(observation_type, raw_item)

    try:
        ref_key = _resolve_array_ref(raw_item["supporting_observation_ref"])
    except _UnsupportedReferenceType as e:
        raise ItemRejected(
            RejectionReason.CANDIDATE_CLASSIFICATION_REFERENCES_UNSUPPORTED_TYPE,
            f"supporting_observation_ref cites an unsupported reference type: {e.obs_type.value!r}",
            raw_item=raw_item,
        ) from e
    if ref_key not in accepted_by_ref:
        raise ItemRejected(
            RejectionReason.CANDIDATE_CLASSIFICATION_REFERENCES_REJECTED_ITEM,
            f"supporting_observation_ref references an item that was rejected or does not exist: {ref_key}",
            raw_item=raw_item,
        )
    referenced_obs = accepted_by_ref[ref_key]
    evidence_id = _source_ref(referenced_obs)
    if evidence_id == "UNKNOWN_SOURCE":
        raise ItemRejected(
            RejectionReason.CANDIDATE_CLASSIFICATION_REFERENCED_OBSERVATION_NOT_TRACEABLE,
            f"supporting_observation_ref target is not traceable to source evidence: {ref_key}",
            raw_item=raw_item,
        )
    span = resolve_source_span(
        evidence_text_by_id[evidence_id], raw_item["source_span"]["text"], raw_item=raw_item,
    )

    dataclass_cls = CANDIDATE_CLASSIFICATION_TYPE_TO_DATACLASS[observation_type]
    common_kwargs = dict(
        source_evidence_id=evidence_id,
        source_span=span,
        basis=InferenceBasis(raw_item["basis"]),
        supporting_observation_ref=_ref_from_key(ref_key),
    )
    try:
        if observation_type == ObservationType.CANDIDATE_RISK_SIGNAL:
            obj = dataclass_cls(
                mechanism=raw_item["mechanism"],
                proposed_severity_tier=raw_item["proposed_severity_tier"],
                **common_kwargs,
            )
        else:
            obj = dataclass_cls(
                proposed_basis=raw_item["proposed_basis"],
                supports=raw_item["supports"],
                **common_kwargs,
            )
    except (ValueError, TypeError) as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"{observation_type.value} failed dataclass construction: {e}",
            raw_item=raw_item,
        ) from e
    return obj, ref_key


def _source_ref(obs) -> str:
    if hasattr(obs, "source_evidence_id"):
        return obs.source_evidence_id
    if hasattr(obs, "reviewed_evidence_ids"):
        return obs.reviewed_evidence_ids[0]
    return "UNKNOWN_SOURCE"


class _UnsupportedReferenceType(Exception):
    """Internal signal only, never surfaced to a caller of run_extraction.
    Raised by _resolve_array_ref() when given an observation_type with no
    corresponding array key -- i.e. one of the two Milestone 2C candidate-
    classification types (CANDIDATE_RISK_SIGNAL, CANDIDATE_EVIDENCE_
    CLASSIFICATION), which are deliberately NOT resolvable reference
    targets (enums.py keeps them in a separate map for exactly this
    reason). Each call site catches this and converts it into a proper
    ExtractionValidationFailure/ItemRejected with call-site-appropriate
    context. This is defense-in-depth: the two call sites already gate on
    a JSON-schema enum that should make this unreachable in practice
    (json_schemas.py's _OBSERVATION_REF_SCHEMA / _SUPPORTING_OBSERVATION_
    REF_SCHEMA) -- fixed post-live-defect (Case 11,
    prompt_v4_optionb_2c_eval1 crash) so a future schema gap can never
    again surface as a raw, uncaught KeyError."""
    def __init__(self, obs_type: ObservationType):
        self.obs_type = obs_type
        super().__init__(f"observation_type {obs_type.value!r} has no resolvable array key")


def _resolve_array_ref(raw_ref: dict) -> tuple[str, int]:
    obs_type = ObservationType(raw_ref["observation_type"])
    array_key = OBSERVATION_TYPE_TO_ARRAY_KEY.get(obs_type)
    if array_key is None:
        raise _UnsupportedReferenceType(obs_type)
    return (array_key, raw_ref["index"])


def _ref_from_key(key: tuple[str, int]):
    from .schemas import ObservationRef
    array_key, idx = key
    return ObservationRef(observation_type=ARRAY_KEY_TO_OBSERVATION_TYPE[array_key], index=idx)


def _check_top_level(raw: Any):
    """Returns (RejectionReason, detail) if `raw` is not schema-valid /
    contains a boundary violation, else None."""
    try:
        validate_top_level_shape(raw)
        return None
    except ItemRejected as e:
        return (e.reason, e.detail)
    except jsonschema.exceptions.ValidationError as e:
        return (RejectionReason.SCHEMA_INVALID, e.message)
    except Exception as e:  # e.g. raw isn't even a dict
        return (RejectionReason.MALFORMED_TOP_LEVEL_OUTPUT, str(e))


# ---------------------------------------------------------------------------
# Milestone 4B — D2/D6 Candidate Qualifier Extraction (resolves M3-OD-01).
#
# A SEPARATE, second-stage pipeline pass. run_extraction() above is
# completely UNCHANGED. Never merged into run_extraction() itself
# (explicit architecture mandate -- see extraction/prompts.py's stage-2
# prompt module docstring).
#
# Isolated-classifier architecture checkpoint (supersedes the original
# Milestone 4B one-call-per-run batched design): this function now
# performs ONE forced-tool-use call PER ELIGIBLE OBSERVATION, never one
# call covering every eligible observation in the run at once. Each
# isolated call receives exactly one already-accepted AdoptionObservation
# (D2) or StakeholderObservation (D6) -- its structured fields, its own
# source_span.text, and only that one dimension's qualifier vocabulary --
# and returns at most one qualifier candidate or none (deliberate
# abstention). This was approved after a live run showed prose-only
# provenance instructions (Prompt v1.1) materially reduced but did not
# eliminate cross-observation semantic leakage when several observations
# were shown to the model in the same call: a departure-only observation
# still occasionally produced CHAMPION_LOST_NO_SUCCESSOR (which requires
# BOTH a departure AND a stated absence of a successor), and an
# automation-only observation still occasionally produced AUTOMATION_
# RELIABLE_LOW_LOGIN_OK (which requires BOTH automation reliability AND a
# stated low-login fact) -- in both cases, the missing half was visible
# only in a SIBLING observation the model could see but should not have
# used. Isolation makes that structurally impossible: there is no sibling
# observation in context to borrow from, in either direction.
# ---------------------------------------------------------------------------

_DIMENSION_QUALIFIER_REASON = ReasonCode(
    code="EXTRACTION-DIMENSION-QUALIFIER-CANDIDATE-PROPOSED",
    governing_object_id="EXTRACTION-PIPELINE-STAGE-2-NOT-A-CHDM-RULE",
    human_readable_text=(
        "AI-proposed candidate D2/D6 qualifier, admitted as Current+Unverified via "
        "inherited grounding from an already-accepted, already-exactly-grounded stage-1 "
        "semantic observation (Milestone 4B stage-2 classifier). This cites the "
        "Milestone 4B stage-2 pipeline pass, not a governed CHDM methodology rule -- "
        "nothing governed has happened yet at this stage, and this reason code is kept "
        "distinct from _EXTRACTION_REASON precisely so stage-1 and stage-2 provenance "
        "are never conflated in an audit trail."
    ),
)


@dataclass(frozen=True)
class DimensionQualifierFailure:
    """Isolated-classifier architecture checkpoint (item D). One isolated
    classifier call's failure, keyed to the SUPPORTING observation it was
    trying to classify -- never a whole-run failure. Recorded when that
    ONE call's provider request raises ModelServiceError, or its response
    is still not well-shaped after one repair retry (mirrors
    run_extraction()'s own one-retry pattern, scoped down to a single
    observation here). This is the third of three structurally distinct
    states a single observation can end up in: (1) candidate produced --
    present in candidate_d2_qualifiers/candidate_d6_qualifiers; (2)
    deliberate abstention -- absent from both that tuple and this one,
    because the classifier successfully ran and returned nothing; (3)
    classifier unavailable for that observation -- present here. A
    grounding-prohibition violation on a successfully-RETURNED answer
    (e.g. an out-of-range reference) is a DIFFERENT thing and still goes
    to ExtractionResult.rejected, unchanged -- the classifier answered,
    it just answered with something ungroundable, which is not the same
    as never answering at all."""
    resolved_observation_id: str
    dimension: DimensionCode
    detail: str


def run_dimension_qualifier_classification(
    extraction_result: ExtractionResult,
    provider: ExtractionProvider,
) -> ExtractionResult:
    """Milestone 4B stage 2, isolated-classifier architecture. Takes an
    already-finished stage-1 ExtractionResult (from run_extraction(),
    unchanged) and classifies D2/D6 candidate qualifiers for its already-
    accepted AdoptionObservation (D2) / StakeholderObservation (D6) items
    -- ONE ISOLATED PROVIDER CALL PER ELIGIBLE OBSERVATION, never one
    call covering the whole run. Returns a NEW ExtractionResult
    (dataclasses.replace) with candidate_d2_qualifiers /
    candidate_d6_qualifiers / dimension_qualifier_failures /
    dimension_qualifier_stage_failure populated -- every stage-1 field
    (accepted, candidate_contradictions, candidate_risk_signals,
    candidate_evidence_classifications, request_failure) is preserved
    untouched; `rejected`, `dedup_audit`, and `traces` gain new stage-2
    entries APPENDED to their stage-1 contents, never replacing any
    existing ones.

    PER-OBSERVATION FAILURE ISOLATION (approved architecture item D): each
    observation's isolated call (and its own single repair/retry, mirroring
    run_extraction()'s own one-retry pattern) is independent. If ONE
    observation's call raises ModelServiceError, or its response is still
    not well-shaped after that one retry, ONLY that observation's failure
    is recorded (a new DimensionQualifierFailure appended to
    dimension_qualifier_failures) -- processing continues to the next
    observation, and any qualifier already produced (or later produced) for
    a different observation in the same run is completely unaffected. This
    replaces the prior batched architecture's all-or-nothing failure mode,
    where one bad call could discard every observation's result for the
    whole run.

    dimension_qualifier_stage_failure is PRESERVED for compatibility (still
    an Optional[str], read the same way by every existing caller) but is
    now DERIVED from dimension_qualifier_failures -- there is exactly one
    authoritative failure record. It is never None while
    dimension_qualifier_failures is non-empty, and vice versa. Neither
    field is ever used to convert the run into `request_failure`, and
    neither is ever confused with "the classifier ran and found nothing"
    (deliberate abstention) -- three structurally distinct states per
    observation, per DimensionQualifierFailure's own docstring.

    Per-item reference-resolution failures on a successfully-RETURNED
    answer (the 4 approved grounding prohibitions -- referencing a
    disallowed observation type, an unresolved reference, a rejected item,
    or an ungrounded object) are recorded as ordinary
    ExtractionValidationFailure entries in the returned `rejected` tuple,
    exactly like any stage-1 per-item failure -- NOT as a
    DimensionQualifierFailure, and NOT reflected in
    dimension_qualifier_stage_failure. _build_dimension_qualifier (the
    function performing this resolution) is completely unchanged from the
    prior batched architecture; only `candidate_list` is different now --
    always a single-item tuple containing the one observation this
    isolated call was about.

    Traceability: a new TraceRecord is created for each accepted
    CandidateDimensionQualifier (chain=(source_evidence_id,
    resolved_observation_id), same shape as CandidateRiskSignal's own
    trace) and APPENDED to extraction_result.traces -- the supporting
    observation's own stage-1 TraceRecord/system fields are never touched.
    Each CandidateDimensionQualifier's own `system` reflects THIS call's
    provider/model/trace_id, distinct from the referenced observation's
    stage-1 `system`. Deduplication runs exactly once, over the full
    cross-call collection of everything every isolated call produced --
    unchanged from the prior architecture (dedup.py needs no special-casing
    either way; see its own CandidateDimensionQualifier branch).
    """
    adoption_observations = tuple(
        o for o in extraction_result.accepted if isinstance(o, AdoptionObservation)
    )
    stakeholder_observations = tuple(
        o for o in extraction_result.accepted if isinstance(o, StakeholderObservation)
    )

    # Nothing eligible to classify -- skip every provider call entirely
    # (nothing to classify is trivial success, not a failure and not a
    # wasted API call). extraction_result's own field defaults already
    # reflect "stage 2 did not need to do anything," so it is returned
    # unchanged.
    if not adoption_observations and not stakeholder_observations:
        return extraction_result

    accepted_ids = {o.system.observation_id for o in extraction_result.accepted}
    new_rejected: list[ExtractionValidationFailure] = []
    new_failures: list[DimensionQualifierFailure] = []
    provisional: list[CandidateDimensionQualifier] = []
    # Milestone 4B v3 — atomic-predicate + deterministic composition
    # architecture. Every grounded, non-duplicate AtomicPredicateEvidence
    # proposed across every isolated call in this run, for BOTH channels
    # — provenance is preserved here regardless of whether its
    # observation's required set was ever complete (see AtomicPredicate
    # Evidence's own docstring). Populates ExtractionResult.
    # dimension_qualifier_predicate_evidence at the end of this function.
    all_predicate_evidence: list[AtomicPredicateEvidence] = []

    channels = (
        (ObservationType.CANDIDATE_D2_QUALIFIER, DimensionCode.D2,
         ObservationType.ADOPTION_OBSERVATION, adoption_observations),
        (ObservationType.CANDIDATE_D6_QUALIFIER, DimensionCode.D6,
         ObservationType.STAKEHOLDER_OBSERVATION, stakeholder_observations),
    )
    for qualifier_type, dimension, expected_ref_type, obs_list in channels:
        array_key = DIMENSION_QUALIFIER_TYPE_TO_ARRAY_KEY[qualifier_type]
        for obs in obs_list:
            # A single-item tuple: `_build_dimension_qualifier` below is
            # completely unchanged from the batched architecture and
            # still resolves supporting_observation_ref.index as a
            # position into whatever `candidate_list` it is given --
            # here that list has exactly one possible position, 0, which
            # is exactly what the isolated prompt instructs the model to
            # always return.
            candidate_list = (obs,)

            # ---- one isolated call, one repair/retry permitted, scoped
            # to THIS observation ONLY. Any failure here is recorded and
            # processing moves on to the next observation -- it never
            # aborts the loop or discards another observation's result
            # (approved architecture item D). ----
            try:
                raw = provider.propose_isolated_dimension_qualifier(dimension, obs)
            except ModelServiceError as e:
                new_failures.append(DimensionQualifierFailure(
                    resolved_observation_id=obs.system.observation_id, dimension=dimension,
                    detail=f"Model service failure: {e}",
                ))
                continue

            top_level_error = _check_isolated_dimension_qualifier_top_level(qualifier_type, raw)
            if top_level_error is not None:
                try:
                    raw_retry = provider.propose_isolated_dimension_qualifier(
                        dimension, obs, repair_hint="repair",
                    )
                except ModelServiceError as e:
                    new_failures.append(DimensionQualifierFailure(
                        resolved_observation_id=obs.system.observation_id, dimension=dimension,
                        detail=f"Model service failure on repair retry: {e}",
                    ))
                    continue
                retry_error = _check_isolated_dimension_qualifier_top_level(qualifier_type, raw_retry)
                if retry_error is not None:
                    _, retry_detail = retry_error
                    new_failures.append(DimensionQualifierFailure(
                        resolved_observation_id=obs.system.observation_id, dimension=dimension,
                        detail=(
                            "Top-level isolated dimension-qualifier output still invalid "
                            f"after one repair retry: {retry_detail}"
                        ),
                    ))
                    continue
                raw = raw_retry

            # ---- 0 or 1 items, per the enforced envelope schema
            # (maxItems: 1). Zero items is deliberate abstention: nothing
            # to do here, no failure, no rejection -- silence is the
            # signal, exactly as in the batched architecture. One item
            # goes through the SAME per-item validation + inherited-
            # grounding resolution as before. This is the SIMPLE-qualifier
            # path only -- the model-facing schema (json_schemas.py)
            # structurally excludes the 2 compound qualifier values here
            # as of v3, so this loop can never produce one directly. ----
            for raw_item in raw.get(array_key) or ():
                try:
                    obj = _build_dimension_qualifier(
                        qualifier_type, dimension, expected_ref_type, raw_item, candidate_list, accepted_ids,
                    )
                    provisional.append(obj)
                except ItemRejected as e:
                    new_rejected.append(ExtractionValidationFailure(array_key, e.reason, e.detail, e.raw_item))

            # ---- Milestone 4B v3: atomic predicates -> grounded evidence
            # collection -> deterministic composition. Each predicate item
            # is independently validated (shape), checked for a duplicate
            # predicate_id within THIS SAME call (rejected explicitly if
            # so -- never silently deduplicated), and grounded (exact
            # substring of THIS SAME observation's own source_span.text --
            # never fuzzy, never resolved against any other observation).
            # A per-item failure here never aborts processing of the other
            # predicate items in the same call, exactly like the simple-
            # qualifier path above. ----
            atomic_array_key = DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_ARRAY_KEY[qualifier_type]
            required_predicate_ids = DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_IDS[qualifier_type]
            composed_qualifier = DIMENSION_QUALIFIER_TYPE_TO_COMPOSED_QUALIFIER[qualifier_type]
            seen_predicate_ids: set = set()
            grounded_predicates: dict[str, AtomicPredicateEvidence] = {}

            for raw_pred in raw.get(atomic_array_key) or ():
                try:
                    validate_atomic_predicate_shape(qualifier_type, raw_pred)
                except ItemRejected as e:
                    new_rejected.append(ExtractionValidationFailure(
                        atomic_array_key, e.reason, e.detail, e.raw_item,
                        resolved_observation_id=obs.system.observation_id, dimension=dimension,
                    ))
                    continue

                predicate_id = raw_pred["predicate_id"]
                if predicate_id in seen_predicate_ids:
                    new_rejected.append(ExtractionValidationFailure(
                        atomic_array_key,
                        RejectionReason.DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE,
                        f"{qualifier_type.value} atomic predicate {predicate_id!r} was proposed more "
                        "than once for this same observation in this same isolated call -- duplicates "
                        "are rejected explicitly, never silently deduplicated",
                        raw_pred,
                        resolved_observation_id=obs.system.observation_id, dimension=dimension,
                    ))
                    continue
                seen_predicate_ids.add(predicate_id)

                evidence_text = raw_pred["evidence_text"]
                if not atomic_predicate_evidence_grounded(evidence_text, obs.source_span.text):
                    new_rejected.append(ExtractionValidationFailure(
                        atomic_array_key,
                        RejectionReason.DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED,
                        f"{qualifier_type.value} atomic predicate {predicate_id!r} evidence_text "
                        f"{evidence_text!r} is not an exact substring of the supporting observation's "
                        "own source_span.text -- atomic-predicate grounding requires an exact match, "
                        "never fuzzy, never resolved against any other observation",
                        raw_pred,
                        resolved_observation_id=obs.system.observation_id, dimension=dimension,
                    ))
                    continue

                evidence = AtomicPredicateEvidence(
                    predicate_id=predicate_id,
                    dimension=dimension,
                    resolved_observation_id=obs.system.observation_id,
                    evidence_text=evidence_text,
                    basis=InferenceBasis(raw_pred["basis"]),
                )
                grounded_predicates[predicate_id] = evidence
                all_predicate_evidence.append(evidence)

            # ---- deterministic composition: the FULL required predicate
            # set is present and grounded -> compose the compound
            # qualifier, routed through the EXISTING, unmodified
            # _build_dimension_qualifier reference-resolution logic
            # (skip_shape_validation=True, since this raw_item is
            # application-constructed, never model-supplied -- the
            # model-facing JSON-schema gate, which now structurally
            # excludes this exact qualifier value, does not apply to it;
            # CandidateDimensionQualifier.__post_init__'s own governed-
            # vocabulary check, unchanged, still applies unconditionally).
            # An INCOMPLETE set is silent abstention -- no candidate, no
            # rejection record, exactly like the ordinary non-force-fit
            # abstention above.
            #
            # Milestone 4B D2 EXPLICIT-basis composition gate (architecture
            # checkpoint, approved after a live probe showed Prompt v3.2's
            # calibration close one RELIABLE_AUTOMATION_OPERATION false-
            # positive path only for the model to complete the SAME
            # 3-predicate set through a different one -- two grounded-but-
            # INFERRED_CANDIDATE predicates on a single-fact observation).
            # For channels where DIMENSION_QUALIFIER_TYPE_REQUIRES_
            # EXPLICIT_BASIS_FOR_COMPOSITION is True (D2 only -- D6 is
            # unaffected and composes exactly as before), completeness
            # ALONE is no longer sufficient: every required predicate must
            # ALSO have basis == EXPLICIT, or the set is INELIGIBLE for
            # composition. This governs composition eligibility only --
            # every grounded predicate was already appended to
            # all_predicate_evidence above, UNCONDITIONALLY, before this
            # check runs, so INFERRED_CANDIDATE evidence remains fully
            # visible in provenance/audit even when this gate blocks
            # composition; an ineligible set abstains exactly like an
            # incomplete one -- no candidate, no rejection record. ----
            predicate_set_complete = all(pid in grounded_predicates for pid in required_predicate_ids)
            requires_explicit_only = DIMENSION_QUALIFIER_TYPE_REQUIRES_EXPLICIT_BASIS_FOR_COMPOSITION[qualifier_type]
            predicate_set_eligible = predicate_set_complete and (
                not requires_explicit_only
                or all(
                    grounded_predicates[pid].basis == InferenceBasis.EXPLICIT
                    for pid in required_predicate_ids
                )
            )
            if predicate_set_eligible:
                composed_basis = (
                    InferenceBasis.EXPLICIT
                    if all(
                        grounded_predicates[pid].basis == InferenceBasis.EXPLICIT
                        for pid in required_predicate_ids
                    )
                    else InferenceBasis.INFERRED_CANDIDATE
                )
                composed_raw_item = {
                    "qualifier": composed_qualifier,
                    "basis": composed_basis.value,
                    "supporting_observation_ref": {"observation_type": expected_ref_type.value, "index": 0},
                }
                try:
                    obj = _build_dimension_qualifier(
                        qualifier_type, dimension, expected_ref_type, composed_raw_item, candidate_list,
                        accepted_ids, skip_shape_validation=True,
                    )
                    provisional.append(obj)
                except ItemRejected as e:
                    new_rejected.append(ExtractionValidationFailure(array_key, e.reason, e.detail, e.raw_item))

    # ---- system-metadata attachment (stage 2's OWN provenance, distinct
    # from the referenced observation's stage-1 `system`) ----
    now = datetime.now(timezone.utc)
    finalized: list[CandidateDimensionQualifier] = []
    for obj in provisional:
        system = ExtractionSystemFields(
            observation_id=_new_id("DIMQ"),
            model_provider=provider.provider_name,
            model_version=provider.model_version,
            extracted_at=now,
            trace_id=_new_id("TRACE"),
            evidence_state=EvidenceState.CURRENT_UNVERIFIED,
        )
        finalized.append(replace(obj, system=system))

    # ---- deduplicate (reuses dedup.py's existing, now-extended machinery
    # unmodified -- see dedup.py's CandidateDimensionQualifier branch;
    # inherited grounding means true duplicates naturally share identical
    # source_evidence_id/source_span, so no special-casing was needed
    # there) ----
    kept, _canonical_map, dq_dedup_audit = deduplicate(tuple(finalized))

    candidate_d2_qualifiers = tuple(o for o in kept if o.dimension == DimensionCode.D2)
    candidate_d6_qualifiers = tuple(o for o in kept if o.dimension == DimensionCode.D6)

    # ---- trace records, APPENDED (never replacing stage-1's own) ----
    new_traces = []
    for cdq in kept:
        new_traces.append(TraceRecord(
            trace_id=cdq.system.trace_id,
            subject_object_ref=cdq.system.observation_id,
            chain=(cdq.source_evidence_id, cdq.resolved_observation_id),
            methodology_version="0.1",
            reason_code=_DIMENSION_QUALIFIER_REASON,
            model_version=f"{cdq.system.model_provider}:{cdq.system.model_version}",
        ))

    # ---- Milestone 4B isolated-classifier architecture checkpoint (item
    # D): dimension_qualifier_stage_failure is DERIVED from new_failures
    # -- there is exactly one authoritative failure record
    # (dimension_qualifier_failures), never a second independent source
    # of truth. None means every isolated call in this run either
    # produced a candidate or abstained successfully. ----
    total_eligible = len(adoption_observations) + len(stakeholder_observations)
    dimension_qualifier_stage_failure = (
        f"{len(new_failures)} of {total_eligible} isolated dimension-qualifier calls failed: "
        + "; ".join(f"{f.resolved_observation_id} ({f.dimension.value}): {f.detail}" for f in new_failures)
        if new_failures else None
    )

    return replace(
        extraction_result,
        candidate_d2_qualifiers=candidate_d2_qualifiers,
        candidate_d6_qualifiers=candidate_d6_qualifiers,
        dimension_qualifier_failures=tuple(new_failures),
        dimension_qualifier_stage_failure=dimension_qualifier_stage_failure,
        rejected=extraction_result.rejected + tuple(new_rejected),
        dedup_audit=extraction_result.dedup_audit + dq_dedup_audit,
        traces=extraction_result.traces + tuple(new_traces),
        # Milestone 4B v3: every grounded atomic predicate proposed across
        # every isolated call in this run, whether or not its observation's
        # required set was ever complete -- see AtomicPredicateEvidence's
        # own docstring. Never filtered down to only the predicates that
        # contributed to a successful composition.
        dimension_qualifier_predicate_evidence=(
            extraction_result.dimension_qualifier_predicate_evidence + tuple(all_predicate_evidence)
        ),
    )


def _build_dimension_qualifier(
    observation_type: ObservationType,
    dimension: DimensionCode,
    expected_ref_type: ObservationType,
    raw_item: dict,
    candidate_list: tuple,
    accepted_ids: set,
    *,
    skip_shape_validation: bool = False,
) -> CandidateDimensionQualifier:
    """Milestone 4B. Validate shape, then resolve supporting_observation_ref
    via the 4 approved grounding prohibitions, in order -- a deliberately
    SIMPLER mechanism than stage-1's _resolve_array_ref/accepted_by_ref
    dict: `candidate_list` here is already the exact, channel-homogeneous,
    already-accepted list the model was shown, so `index` is a direct
    list-position lookup (see extraction.json_schemas.
    CANDIDATE_D2_QUALIFIER_SCHEMA's own docstring for why this differs
    from stage 1's resolution mechanism). Raises ItemRejected on any
    failure -- nothing is silently retained.

    Milestone 4B v3: `skip_shape_validation=True` is used ONLY by the
    atomic-predicate composer (run_dimension_qualifier_classification)
    for its deterministically-constructed, application-authored
    `composed_raw_item` dicts. Those items are NEVER model-supplied, so
    the model-facing JSON-schema gate (validate_dimension_qualifier_shape,
    which as of v3 structurally EXCLUDES the 2 compound qualifier values
    from DIMENSION_QUALIFIER_TYPE_TO_SCHEMA's enum precisely so the model
    cannot propose them directly) does not apply to them -- that gate is a
    model-INPUT boundary control, not a general-purpose value check. The
    qualifier-vocabulary check enforced by CandidateDimensionQualifier.
    __post_init__ below (against schemas._CANDIDATE_D2_QUALIFIERS /
    _CANDIDATE_D6_QUALIFIERS, unchanged and still containing all 5/4
    values including both compound ones) still applies unconditionally on
    every path, direct or composed -- this parameter skips ONLY the
    model-facing shape/denylist gate, never the governed-vocabulary
    guard. Every direct-path call site (the simple-qualifier loop above)
    passes the default (False), so behavior for the other 7 qualifiers is
    completely unchanged."""
    if not skip_shape_validation:
        validate_dimension_qualifier_shape(observation_type, raw_item)

    raw_ref = raw_item["supporting_observation_ref"]

    # (a) disallowed observation type -- defense-in-depth; the per-channel
    # JSON schema (_D2_SUPPORTING_OBSERVATION_REF_SCHEMA /
    # _D6_SUPPORTING_OBSERVATION_REF_SCHEMA) already restricts this at the
    # schema gate, so this should never fire in practice.
    if raw_ref["observation_type"] != expected_ref_type.value:
        raise ItemRejected(
            RejectionReason.DIMENSION_QUALIFIER_REFERENCES_DISALLOWED_TYPE,
            f"{observation_type.value} supporting_observation_ref cites "
            f"{raw_ref['observation_type']!r}, but only {expected_ref_type.value!r} is "
            f"permitted for this channel",
            raw_item=raw_item,
        )

    # (b) unresolved reference -- index out of range of the channel-
    # homogeneous accepted-observation list this call was given.
    idx = raw_ref["index"]
    if idx < 0 or idx >= len(candidate_list):
        raise ItemRejected(
            RejectionReason.DIMENSION_QUALIFIER_REFERENCE_NOT_FOUND,
            f"{observation_type.value} supporting_observation_ref.index={idx} does not "
            f"resolve to any item in the {len(candidate_list)}-item list supplied for "
            "this channel",
            raw_item=raw_item,
        )
    referenced = candidate_list[idx]

    # (c) rejected observation -- defense-in-depth; candidate_list is
    # constructed exclusively from extraction_result.accepted, so any
    # in-range index structurally always resolves to an accepted item.
    # This re-verifies that invariant explicitly rather than assuming it
    # silently holds forever.
    if referenced.system.observation_id not in accepted_ids:
        raise ItemRejected(
            RejectionReason.DIMENSION_QUALIFIER_REFERENCES_REJECTED_ITEM,
            f"{observation_type.value} supporting_observation_ref resolved to an item "
            "that is not part of this run's accepted stage-1 observations",
            raw_item=raw_item,
        )

    # (d) ungrounded object -- defense-in-depth; every real
    # AdoptionObservation/StakeholderObservation structurally guarantees
    # source_evidence_id/source_span, so this should never fire either.
    if not referenced.source_evidence_id or referenced.source_span is None:
        raise ItemRejected(
            RejectionReason.DIMENSION_QUALIFIER_REFERENCES_UNGROUNDED_OBJECT,
            f"{observation_type.value} supporting_observation_ref resolved to an item "
            "with no valid source_evidence_id/source_span to inherit grounding from",
            raw_item=raw_item,
        )

    try:
        return CandidateDimensionQualifier(
            dimension=dimension,
            qualifier=raw_item["qualifier"],
            basis=InferenceBasis(raw_item["basis"]),
            supporting_observation_ref=ObservationRef(observation_type=expected_ref_type, index=idx),
            # Inherited grounding (approved Milestone 4B exception -- see
            # extraction.schemas.CandidateDimensionQualifier's docstring):
            # copied verbatim from the resolved supporting observation,
            # never re-derived or re-validated against evidence text.
            source_evidence_id=referenced.source_evidence_id,
            source_span=referenced.source_span,
            resolved_observation_id=referenced.system.observation_id,
        )
    except (ValueError, TypeError) as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"{observation_type.value} failed dataclass construction: {e}",
            raw_item=raw_item,
        ) from e


def _check_isolated_dimension_qualifier_top_level(qualifier_type: ObservationType, raw: Any):
    """Isolated-classifier architecture checkpoint. REPLACES
    _check_dimension_qualifier_top_level (removed -- that function
    checked the prior batched, both-channel envelope, which no longer
    exists as an active call path). Mirrors _check_top_level's own
    rationale, but against whichever ONE-channel, at-most-one-item
    envelope (ISOLATED_DIMENSION_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA[
    qualifier_type] -- CANDIDATE_D2_QUALIFIER or CANDIDATE_D6_QUALIFIER)
    matches the dimension this particular isolated call was for. Returns
    (RejectionReason, detail) if `raw` is not schema-valid, else None."""
    try:
        validate_isolated_dimension_qualifier_top_level_shape(qualifier_type, raw)
        return None
    except ItemRejected as e:
        return (e.reason, e.detail)
    except jsonschema.exceptions.ValidationError as e:
        return (RejectionReason.SCHEMA_INVALID, e.message)
    except Exception as e:  # e.g. raw isn't even a dict
        return (RejectionReason.MALFORMED_TOP_LEVEL_OUTPUT, str(e))
