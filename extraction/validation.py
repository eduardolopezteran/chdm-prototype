"""
Build Milestone 2 — schema, source-span, and boundary validation
(spec §4, §6 steps 6-8, §13).

Two-layer boundary enforcement, both exercised on every item BEFORE a
dataclass is ever constructed:
  1. `scan_for_prohibited_keys` — recursive denylist scan, run FIRST, so a
     boundary-violation attempt is classified precisely (not just as a
     generic schema error) and can be counted (spec §15).
  2. `jsonschema.validate` against the per-type model-facing schema
     (json_schemas.py) — `additionalProperties: false` structurally
     prevents the model from ever setting model_provider, model_version,
     extracted_at, trace_id, or evidence_state (Checkpoint 2A refinement 1).

Then, independently, every span-grounded item must resolve EXACTLY
against its cited evidence text (Checkpoint 2A refinement 2) — no
fuzzy/bounded fallback. An item that fails any of these checks is
rejected via `ItemRejected` and never reaches dataclass construction;
nothing is silently retained (spec §4, §13).

Milestone 2B baseline-fix update: the live baseline (eval/results/
baseline_v1.json) showed the model cannot reliably self-report exact
start_char/end_char offsets — 68% of rejections were the model's own
offset arithmetic being internally inconsistent, not bad grounding.
`resolve_source_span` below replaces model-authored offsets entirely:
given the model's literal `text` and the cited evidence, the offsets
are derived deterministically by exact substring search. Zero
occurrences -> SPAN_NOT_FOUND (unchanged meaning). More than one exact
occurrence -> SPAN_AMBIGUOUS (new — the system must never silently pick
the first match). Exactly one occurrence -> the canonical SourceSpan is
constructed from the derived offsets, still no fuzzy/normalized/partial
matching anywhere in this path.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from .enums import InferenceBasis, ObservationType
from .errors import ItemRejected, RejectionReason
from .json_schemas import (
    ISOLATED_DIMENSION_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA, LOOSE_TOP_LEVEL_SCHEMA,
    PROHIBITED_KEY_DENYLIST, TYPE_TO_SCHEMA,
)
from .schemas import (
    OBSERVATION_TYPE_TO_DATACLASS, MissingInformationCandidate, SourceSpan,
)


def scan_for_prohibited_keys(obj: Any) -> list[str]:
    """Recursively walk a parsed JSON value and return every dict key
    (case-insensitively) matching the governed-field denylist. Empty list
    means clean. Order-preserving, may contain duplicates (each occurrence
    is a separate attempted violation)."""
    found: list[str] = []

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and k.strip().lower() in PROHIBITED_KEY_DENYLIST:
                    found.append(k)
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for item in o:
                _walk(item)

    _walk(obj)
    return found


def validate_top_level_shape(raw: Any) -> None:
    """Raises jsonschema.exceptions.ValidationError if `raw` is not even a
    well-shaped extraction response (used by pipeline.py's one-retry
    logic, spec §13 "Schema failure").

    Deliberately does NOT recursively denylist-scan item content here —
    that would reject the ENTIRE response for a single boundary-violating
    item deep inside one array, when the correct behavior is to reject
    only THAT item (spec §4/§13: reject the claim, not the whole
    request). Per-item denylist scanning happens in `build_observation`
    and `validate_contradiction_item_shape`. A prohibited key at the
    TOP level (e.g. `{"confirmed": true, "stakeholder_observations": []}`)
    is still caught here structurally, via LOOSE_TOP_LEVEL_SCHEMA's
    `additionalProperties: false` — it is not one of the declared array
    keys, so `jsonschema.validate` rejects it without needing a separate
    denylist pass. Uses the LOOSE envelope schema (array-of-objects only,
    not each item's full per-type schema), so a single malformed ITEM
    doesn't fail the whole response — that's the per-item loop's job."""
    jsonschema.validate(instance=raw, schema=LOOSE_TOP_LEVEL_SCHEMA)


def validate_isolated_dimension_qualifier_top_level_shape(observation_type: ObservationType, raw: Any) -> None:
    """Milestone 4B isolated-classifier architecture checkpoint. Raises
    jsonschema.exceptions.ValidationError if `raw` is not even a
    well-shaped ISOLATED stage-2 response (used by pipeline.
    run_dimension_qualifier_classification's own per-observation one-
    retry logic). REPLACES validate_dimension_qualifier_top_level_shape
    (removed — that function validated the prior batched, both-channel
    envelope, which no longer exists as an active call path). Mirrors
    validate_top_level_shape's own rationale exactly, but against
    whichever ONE-channel, at-most-one-item envelope
    (ISOLATED_DIMENSION_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA[observation_type]
    -- CANDIDATE_D2_QUALIFIER or CANDIDATE_D6_QUALIFIER) matches the
    dimension this particular isolated call was for -- a D2 call's
    response is never checked against the D6 envelope or vice versa."""
    schema = ISOLATED_DIMENSION_QUALIFIER_LOOSE_TOP_LEVEL_SCHEMA[observation_type]
    jsonschema.validate(instance=raw, schema=schema)


def validate_source_span(evidence_text: str, span: SourceSpan) -> bool:
    """Checkpoint 2A refinement 2: the supplied offsets must resolve
    EXACTLY to the supplied span text within the referenced evidence's raw
    text. No normalization, no fuzzy matching, no partial credit."""
    if span.start_char < 0 or span.end_char > len(evidence_text) or span.end_char <= span.start_char:
        return False
    return evidence_text[span.start_char:span.end_char] == span.text


def resolve_source_span(evidence_text: str, span_text: str, *, raw_item: Any = None) -> SourceSpan:
    """Milestone 2B baseline fix: derive `start_char`/`end_char`
    deterministically from the model-supplied literal `span_text`, rather
    than trusting model-authored offsets (which the live baseline showed
    are unreliable). Exact, unambiguous, non-fuzzy resolution only:

      - zero exact occurrences of `span_text` in `evidence_text` -> reject
        SPAN_NOT_FOUND;
      - more than one exact occurrence -> reject SPAN_AMBIGUOUS (never
        silently pick the first match);
      - exactly one exact occurrence -> construct the canonical
        SourceSpan from the derived offsets.

    The resulting SourceSpan is passed through `validate_source_span` as
    a self-check: since the offsets are derived directly from the same
    search that found `span_text`, this can only fail if the derivation
    logic itself has a defect, never because of model input."""
    occurrences = evidence_text.count(span_text)
    if occurrences == 0:
        raise ItemRejected(
            RejectionReason.SPAN_NOT_FOUND,
            f"span text does not appear verbatim in the referenced evidence: {span_text!r}",
            raw_item=raw_item,
        )
    if occurrences > 1:
        raise ItemRejected(
            RejectionReason.SPAN_AMBIGUOUS,
            f"span text appears {occurrences} times verbatim in the referenced evidence "
            f"(ambiguous — not auto-resolved to the first match): {span_text!r}",
            raw_item=raw_item,
        )
    start_char = evidence_text.index(span_text)
    end_char = start_char + len(span_text)
    span = SourceSpan(text=span_text, start_char=start_char, end_char=end_char)
    assert validate_source_span(evidence_text, span), (
        "internal defect: system-derived SourceSpan failed its own grounding self-check"
    )
    return span


def build_observation(
    observation_type: ObservationType,
    raw_item: dict,
    evidence_text_by_id: dict[str, str],
):
    """Validate one raw model-emitted item end-to-end and construct its
    dataclass. Raises ItemRejected (never returns a partially-valid
    object) on any failure. `observation_type` must be one of the 7
    span-grounded types OR MISSING_INFORMATION_CANDIDATE — contradictions
    are handled separately in pipeline.py because resolving their
    cross-references requires the full accepted-item index."""
    denylist_hits = scan_for_prohibited_keys(raw_item)
    if denylist_hits:
        raise ItemRejected(
            RejectionReason.BOUNDARY_VIOLATION,
            f"{observation_type.value} item contains prohibited governed-field key(s): "
            f"{sorted(set(denylist_hits))}",
            raw_item=raw_item,
        )

    schema = TYPE_TO_SCHEMA[observation_type]
    try:
        jsonschema.validate(instance=raw_item, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"{observation_type.value} failed schema validation: {e.message}",
            raw_item=raw_item,
        ) from e

    if observation_type == ObservationType.MISSING_INFORMATION_CANDIDATE:
        reviewed = tuple(raw_item["reviewed_evidence_ids"])
        unknown = [eid for eid in reviewed if eid not in evidence_text_by_id]
        if unknown:
            raise ItemRejected(
                RejectionReason.UNKNOWN_EVIDENCE_ID,
                f"MissingInformationCandidate references unknown evidence id(s): {unknown}",
                raw_item=raw_item,
            )
        return MissingInformationCandidate(
            missing_item=raw_item["missing_item"],
            reviewed_evidence_ids=reviewed,
        )

    # ---- span-grounded types ----
    evidence_id = raw_item["source_evidence_id"]
    if evidence_id not in evidence_text_by_id:
        raise ItemRejected(
            RejectionReason.UNKNOWN_EVIDENCE_ID,
            f"{observation_type.value} references unknown evidence id: {evidence_id!r}",
            raw_item=raw_item,
        )

    span_text = raw_item["source_span"]["text"]
    span = resolve_source_span(evidence_text_by_id[evidence_id], span_text, raw_item=raw_item)

    dataclass_cls = OBSERVATION_TYPE_TO_DATACLASS[observation_type]
    semantic_kwargs = {
        k: v for k, v in raw_item.items()
        if k not in ("source_span",)
    }
    semantic_kwargs["source_span"] = span
    semantic_kwargs["basis"] = InferenceBasis(raw_item["basis"])
    try:
        return dataclass_cls(**semantic_kwargs)
    except (ValueError, TypeError) as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"{observation_type.value} failed dataclass construction: {e}",
            raw_item=raw_item,
        ) from e


def validate_contradiction_item_shape(raw_item: dict) -> None:
    """Schema + denylist validation only, for a raw candidate_contradictions
    entry — reference resolution happens in pipeline.py."""
    from .json_schemas import CANDIDATE_CONTRADICTION_SCHEMA

    denylist_hits = scan_for_prohibited_keys(raw_item)
    if denylist_hits:
        raise ItemRejected(
            RejectionReason.BOUNDARY_VIOLATION,
            f"candidate_contradiction item contains prohibited governed-field key(s): "
            f"{sorted(set(denylist_hits))}",
            raw_item=raw_item,
        )
    try:
        jsonschema.validate(instance=raw_item, schema=CANDIDATE_CONTRADICTION_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"candidate_contradiction failed schema validation: {e.message}",
            raw_item=raw_item,
        ) from e


def validate_dimension_qualifier_shape(observation_type: ObservationType, raw_item: dict) -> None:
    """Milestone 4B. Schema + denylist validation only, for a raw
    candidate_d2_qualifiers / candidate_d6_qualifiers entry — mirrors
    `validate_candidate_classification_shape` exactly. Reference
    resolution AND source_evidence_id/source_span INHERITANCE (not
    re-derivation — see extraction.schemas.CandidateDimensionQualifier's
    docstring) happen in pipeline.py's stage-2 pass
    (run_dimension_qualifier_classification /
    _resolve_dimension_qualifier_reference), because both require the
    channel-homogeneous list of already-accepted observations that stage
    2 was called with, which does not exist at this call site."""
    from .json_schemas import DIMENSION_QUALIFIER_TYPE_TO_SCHEMA

    denylist_hits = scan_for_prohibited_keys(raw_item)
    if denylist_hits:
        raise ItemRejected(
            RejectionReason.BOUNDARY_VIOLATION,
            f"{observation_type.value} item contains prohibited governed-field key(s): "
            f"{sorted(set(denylist_hits))}",
            raw_item=raw_item,
        )
    schema = DIMENSION_QUALIFIER_TYPE_TO_SCHEMA[observation_type]
    try:
        jsonschema.validate(instance=raw_item, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"{observation_type.value} failed schema validation: {e.message}",
            raw_item=raw_item,
        ) from e


def validate_atomic_predicate_shape(observation_type: ObservationType, raw_item: dict) -> None:
    """Milestone 4B v3 (atomic-predicate + deterministic composition
    architecture). Schema + denylist validation only, for a raw
    candidate_d2_atomic_predicates / candidate_d6_atomic_predicates entry
    — mirrors `validate_dimension_qualifier_shape` exactly, against
    DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_SCHEMA instead of
    DIMENSION_QUALIFIER_TYPE_TO_SCHEMA. Grounding (evidence_text must be
    an exact substring of the single observation's own source_span.text)
    and duplicate-predicate-ID detection are NOT checked here — both
    require the single observation this isolated call was about, which
    does not exist at this call site; both happen in pipeline.py's
    composer (extraction.pipeline.run_dimension_qualifier_classification),
    exactly mirroring how reference resolution for the ordinary qualifier
    path happens in pipeline.py rather than here."""
    from .json_schemas import DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_SCHEMA

    denylist_hits = scan_for_prohibited_keys(raw_item)
    if denylist_hits:
        raise ItemRejected(
            RejectionReason.BOUNDARY_VIOLATION,
            f"{observation_type.value} atomic predicate item contains prohibited "
            f"governed-field key(s): {sorted(set(denylist_hits))}",
            raw_item=raw_item,
        )
    schema = DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_SCHEMA[observation_type]
    try:
        jsonschema.validate(instance=raw_item, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID,
            f"{observation_type.value} atomic predicate failed schema validation: {e.message}",
            raw_item=raw_item,
        ) from e


def atomic_predicate_evidence_grounded(evidence_text: str, observation_span_text: str) -> bool:
    """Milestone 4B v3. Pure grounding-check helper for one atomic
    predicate's `evidence_text` against the SAME observation's own
    `source_span.text` — exact, unambiguous substring containment only,
    no fuzzy/normalized/partial matching, mirroring `resolve_source_span`'s
    own discipline. Deliberately does NOT require exactly-one-occurrence
    (unlike resolve_source_span's SPAN_AMBIGUOUS rule): an atomic
    predicate's evidence_text is not used to derive character offsets or
    construct a SourceSpan, only to confirm the claimed substring is
    genuinely present verbatim — multiple occurrences of the same
    sub-phrase within one observation's span are not an ambiguity that
    needs resolving here, since nothing downstream depends on which
    occurrence was meant. Returns False (never raises) so callers can
    decide how to record the rejection with full raw-item context."""
    if not evidence_text or not observation_span_text:
        return False
    return evidence_text in observation_span_text


def validate_candidate_classification_shape(observation_type: ObservationType, raw_item: dict) -> None:
    """Milestone 2C. Schema + denylist validation only, for a raw
    candidate_risk_signals / candidate_evidence_classifications entry —
    mirrors `validate_contradiction_item_shape` exactly. Reference
    resolution AND source_evidence_id/source_span derivation happen in
    pipeline.py, because both require the full accepted-item index
    (`accepted_by_ref`), which does not exist yet at this call site —
    same ordering reason `CandidateContradiction` resolution is not done
    here either."""
    from .json_schemas import CANDIDATE_CLASSIFICATION_TYPE_TO_SCHEMA

    denylist_hits = scan_for_prohibited_keys(raw_item)
    if denylist_hits:
        raise ItemRejected(
            RejectionReason.BOUNDARY_VIOLATION,
            f"{observation_type.value} item contains prohibited governed-field key(s): "
            f"{sorted(set(denylist_hits))}",
            raw_item=raw_item,
        )
    schema = CANDIDATE_CLASSIFICATION_TYPE_TO_SCHEMA[observation_type]
    try:
        jsonschema.validate(instance=raw_item, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ItemRejected(
            RejectionReason.SCHEMA_INVALID, f"{observation_type.value} failed schema validation: {e.message}",
            raw_item=raw_item,
        ) from e
