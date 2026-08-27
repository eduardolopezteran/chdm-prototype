"""
Build Milestone 2B — evaluation metrics (spec §3).
Milestone 2B.1 — evaluation-only refinement (spec §1-§10 of the
Milestone 2B.1 authorization). Nothing in this file touches the
extraction pipeline, schemas, provider, or prompt; it only scores
already-produced ExtractionResult output against eval/labeled_set.yaml.

Why this revision exists: comparing baseline_v1.json against
baseline_v1_fix1.json showed the OLD single-number precision/recall was
being distorted by under-labeling, not by genuine extraction quality --
real, grounded, Customer-Health-relevant facts the model correctly found
had no labeled expectation at all, so they were counted as "unsupported"
even though nothing was actually wrong with them. This module replaces
that with an explicit six-way classification per accepted observation,
and reports three separate precision numbers rather than collapsing them
into one (Milestone 2B.1 spec §6):

  MATCHED_EXPECTED       -- matches a `primary` or `supporting` (i.e.
                            required) labeled expectation, on both type
                            and span.
  VALID_UNLABELED        -- matches an `optional-valid` labeled
                            expectation, OR is a genuinely unanticipated
                            observation that is still grounded and not
                            excluded by any label (flagged
                            `needs_relevance_review` in the per-case dump
                            so a human can reclassify it later -- this
                            module does not silently assume relevance).
  WRONG_TYPE              -- the same fact (span match) as a labeled
                            expectation, but emitted under a type not in
                            that expectation's canonical/alternate types.
  UNSUPPORTED             -- reserved for content that fails byte-exact
                            grounding. Given the Milestone 2B baseline
                            fix's structural guarantee (every accepted
                            observation's span is derived by exact,
                            unambiguous substring search against real
                            evidence -- extraction/validation.py), this
                            should always be 0 here. This module does NOT
                            attempt to detect semantic overreach in
                            free-text fields (e.g. a `stated_outcome`
                            paraphrase that oversteps its own span) --
                            that remains a human-review concern, not an
                            automated one.
  DUPLICATE               -- a second accepted observation (different
                            type, since the pipeline's own dedup already
                            collapses same-type overlapping duplicates)
                            redundantly re-capturing a fact already
                            classified from an earlier observation in the
                            same case.
  IRRELEVANT_BUT_GROUNDED -- matches a case's
                            `grounded_but_irrelevant_permitted` entry:
                            correctly grounded, but not Customer-Health
                            relevant. Never counted as a positive match,
                            never counted as unsupported.

Ontology ambiguity (Milestone 2B.1 Case 12b disposition): a labeled
expectation may carry `alternate_types` + `ontology_ambiguity: true`.
This is an explicit, reviewed EVALUATION ACCOMMODATION for one specific
case (both types are accepted as valid for that fact given the context
the model had) -- it is not a general extraction rule, and is not used
anywhere else in this file except to widen the accepted-type set for
that one labeled slot. Case 05 deliberately has no alternate type: a
model emitting OBJECTIVE_CANDIDATE for its usage-growth fact scores
WRONG_TYPE, per the Milestone 2B.1 Case 05 disposition.

As before: this remains an approximate, 15/20-example engineering
measure, not a validated model-performance claim. Anything flagged
`needs_relevance_review` should be checked against the full per-case
dump this module also produces, not silently trusted either way.

Milestone 2C addition: `classify_candidate_classifications` scores the
two new candidate-classification output types (CandidateRiskSignal,
CandidateEvidenceClassification) against `expected_candidate_risk_
signals` / `expected_candidate_evidence_classifications`. This is
deliberately a SEPARATE, simpler scorer from the six-way classifier
above — candidate classifications are proposals about an already-
classified observation, not observations themselves, so "right fact,
wrong governed detail" (e.g. right span, wrong mechanism) is a
meaningful, distinct outcome worth its own flag rather than being
folded into WRONG_TYPE. Four outcomes per candidate:
  MATCHED_EXPECTED   -- span matches an unclaimed expected slot. Carries
                        `mechanism_correct`/`tier_correct` (risk signals)
                        or `basis_correct`/`supports_correct` (evidence
                        classifications), and `reference_correct` (did
                        resolved_observation_id land on the RIGHT
                        supporting observation, not just any observation)
                        as separate booleans -- a span match with the
                        wrong governed detail is still MATCHED_EXPECTED
                        (the extractor found the right fact) but is
                        flagged, not silently credited as fully correct.
  UNEXPECTED_CANDIDATE -- grounded (guaranteed structurally by the
                        pipeline) but matches no expected slot at all --
                        the false-candidate / over-fabrication signal
                        this module's implementation-constraint-2
                        benchmark cases (32, 33) specifically probe.
A slot that stays unclaimed after this pass is a missed candidate,
reported via `aggregate_metrics`' recall counters exactly like an
unclaimed observation slot.

Milestone 2B.2 closure fix: `classify_accepted_observations` previously
never cross-referenced `inferred_candidates_permitted` (case 16, case
21). A genuinely correct INFERRED_CANDIDATE finding for a permitted
entry could still be scored WRONG_TYPE if its span happened to overlap
an unrelated slot's span_substrings -- exactly what happened to case 21
in the prompt_v2_ontology_eval1 live run, where the model correctly
produced an INFERRED_CANDIDATE ObjectiveCandidate (never EXPLICIT, per
OC-01) and was penalized for it anyway. `_build_inferred_permitted` +
the new step 1.5 in `classify_accepted_observations` now credit a
type+basis+span match against `inferred_candidates_permitted` as
VALID_UNLABELED (flagged `matched_inferred_permitted: true`) BEFORE the
wrong-type fallback runs. This is a metrics-only correction: it changes
nothing about what the extractor produces or how a permitted entry's
EXPLICIT-basis counterpart is scored (still ordinary WRONG_TYPE/
unmatched, unchanged) -- only a basis-correct INFERRED_CANDIDATE match
against a permitted entry is affected.

Milestone 4B addition: `classify_dimension_qualifiers` scores the two
D2/D6 candidate-qualifier output channels (CandidateDimensionQualifier)
against `expected_dimension_d2_qualifiers` / `expected_dimension_d6_
qualifiers`. This is a SEPARATE call from run_extraction()'s own
scoring above -- the D2/D6 classifier is a distinct stage-2 pipeline
pass (extraction.pipeline.run_dimension_qualifier_classification),
never part of run_extraction() itself, so a labeled case exercising it
must run BOTH stages before this scorer can run (see eval/run_eval.py's
run_dimension_qualifier_eval_case). Structural difference from
classify_candidate_classifications above: because CandidateDimension
Qualifier uses INHERITED grounding (its own source_span is copied
verbatim from its supporting observation, never independently model-
supplied -- see extraction.schemas.CandidateDimensionQualifier's
docstring), matching is keyed on the SUPPORTING observation's span (via
resolved_observation_id) rather than the qualifier proposal's own
span_text -- there is no independent span to match on the proposal
itself, by design.
"""

from __future__ import annotations

from typing import Optional


def _obs_summary(obs) -> dict:
    """A JSON-safe, human-readable summary of one accepted observation or
    contradiction — used both for matching and for the report dump."""
    d = {"observation_class": type(obs).__name__}
    if hasattr(obs, "source_evidence_id"):
        d["source_evidence_id"] = obs.source_evidence_id
    if hasattr(obs, "source_span"):
        d["span_text"] = obs.source_span.text
        d["span_offsets"] = (obs.source_span.start_char, obs.source_span.end_char)
    if hasattr(obs, "basis"):
        basis = obs.basis
        d["basis"] = basis.value if hasattr(basis, "value") else basis
    if hasattr(obs, "reviewed_evidence_ids"):
        d["reviewed_evidence_ids"] = list(obs.reviewed_evidence_ids)
        d["missing_item"] = obs.missing_item
    if hasattr(obs, "conflict_description"):
        d["conflict_description"] = obs.conflict_description
        d["methodology_construct_hint"] = obs.methodology_construct_hint
        d["resolved_observation_id_a"] = obs.resolved_observation_id_a
        d["resolved_observation_id_b"] = obs.resolved_observation_id_b
    # Milestone 2C: CandidateRiskSignal / CandidateEvidenceClassification.
    # `resolved_observation_id` deliberately checked via a Milestone-2C-
    # specific field name (not reused from the contradiction's `_a`/`_b`
    # pair above) since each candidate classification references exactly
    # one supporting observation, not two.
    if hasattr(obs, "mechanism"):
        d["mechanism"] = obs.mechanism
        d["proposed_severity_tier"] = obs.proposed_severity_tier
    if hasattr(obs, "proposed_basis"):
        d["proposed_basis"] = obs.proposed_basis
        d["supports"] = obs.supports
    if hasattr(obs, "supporting_observation_ref"):
        ref = obs.supporting_observation_ref
        ref_type = ref.observation_type
        d["supporting_observation_ref"] = {
            "observation_type": ref_type.value if hasattr(ref_type, "value") else ref_type,
            "index": ref.index,
        }
        d["resolved_observation_id"] = obs.resolved_observation_id
    # Milestone 4B: CandidateDimensionQualifier. `qualifier` is checked
    # here rather than folded into the mechanism/proposed_basis checks
    # above -- those two attribute pairs are Milestone 2C-specific field
    # NAMES that happen to be unique to their own dataclasses, but
    # `qualifier` alone is generic enough that gating on `dimension`
    # jointly avoids any accidental false-positive on some future type.
    if hasattr(obs, "qualifier") and hasattr(obs, "dimension"):
        dim = obs.dimension
        d["dimension"] = dim.value if hasattr(dim, "value") else dim
        d["qualifier"] = obs.qualifier
    # Type-specific "primary content" fields, best-effort.
    for attr in (
        "objective_text", "stated_outcome", "person_identifier", "continuity_event",
        "workflow_or_use_case", "observed_behavior", "incident_or_condition",
        "event_type", "description", "statement", "event",
    ):
        if hasattr(obs, attr):
            d[attr] = getattr(obs, attr)
    if hasattr(obs, "system"):
        d["observation_id"] = obs.system.observation_id
        d["evidence_state"] = obs.system.evidence_state.value if obs.system.evidence_state else None
    return d


def _dataclass_matches_type(class_name: str, obs_type_str: str) -> bool:
    mapping = {
        "OBJECTIVE_CANDIDATE": "ObjectiveCandidate",
        "STAKEHOLDER_OBSERVATION": "StakeholderObservation",
        "ADOPTION_OBSERVATION": "AdoptionObservation",
        "SERVICE_OBSERVATION": "ServiceObservation",
        "COMMERCIAL_OBSERVATION": "CommercialObservation",
        "EXPERIENCE_OBSERVATION": "ExperienceObservation",
        "STRATEGIC_OBSERVATION": "StrategicObservation",
        "MISSING_INFORMATION_CANDIDATE": "MissingInformationCandidate",
    }
    return mapping.get(obs_type_str) == class_name


def _span_matches(accepted_span_text: Optional[str], acceptable_substrings: list[str]) -> bool:
    """Milestone 2B.1 spec §7/§8: a label may list more than one exact
    span boundary for the same material fact. Matching stays exact-
    substring (in either containment direction, as the original matcher
    did) — this widens which literal string counts as "the same fact",
    it never introduces fuzzy/normalized matching. Grounding itself
    (whether the accepted span is a real exact quote of the evidence) was
    already enforced upstream by extraction/validation.py; this is
    benchmark matching only.

    Milestone 4B isolated-classifier architecture checkpoint (item F)
    scope note: this general-purpose matcher is used at 6 OTHER call
    sites beyond the D2/D6 qualifier scorer (ordinary observation-slot
    matching, inferred-candidates-permitted matching, wrong-type
    fallback, irrelevant-but-grounded matching, duplicate detection, and
    the 2C candidate-classification scorer) — all of them pre-existing,
    already-approved (non-D2/D6) benchmark accommodations that
    deliberately rely on BOTH containment directions (e.g. the Milestone
    2B.2 case-21 fix, where a shorter accepted excerpt of a longer
    run-on sentence must still match a label's full-sentence substring).
    Item F's own wording ("a qualifier's expected supporting substring")
    is D2/D6-specific, and item H protects everything outside the
    isolated-classifier change from being altered in this build. Rather
    than changing this shared function's behavior for all 7 call sites
    (confirmed via a full-suite run to silently break 3 pre-existing,
    previously-approved test fixtures unrelated to D2/D6 -- see
    `_dimension_qualifier_span_matches` below), the one-directional,
    stricter rule required by item F is implemented as a SEPARATE,
    dedicated function used ONLY by `classify_dimension_qualifiers`.
    This function's own behavior for every other call site is
    unchanged."""
    if accepted_span_text is None:
        return False
    for sub in acceptable_substrings:
        if sub in accepted_span_text or accepted_span_text in sub:
            return True
    return False


def _dimension_qualifier_span_matches(supporting_span_text: Optional[str], acceptable_substrings: list[str]) -> bool:
    """Milestone 4B isolated-classifier architecture checkpoint (item F,
    REQUIRED). Used ONLY by `classify_dimension_qualifiers`, to score a
    CandidateDimensionQualifier's cited SUPPORTING OBSERVATION span
    against a labeled qualifier expectation's
    `supporting_observation_span_substrings`. Unlike `_span_matches`
    above (kept unchanged for its other 6, non-D2/D6 call sites), this
    matcher accepts ONLY the forward containment direction: an
    acceptable substring must be found WITHIN the supporting
    observation's own span text. The reverse direction
    (`supporting_span_text in sub`) is deliberately NOT accepted here,
    because it would let a short, incomplete supporting-observation span
    incorrectly "certify" against a longer, compound expected condition
    that the single cited observation does not, on its own, fully
    support — exactly how live cases 35/40 scored MATCHED_EXPECTED
    despite insufficient single-observation grounding under the prior
    bidirectional matcher. No fuzzy or normalized matching is
    introduced; this remains exact substring search only, restricted to
    the one direction that is sound for "does this single cited
    observation support this qualifier" benchmark matching."""
    if supporting_span_text is None:
        return False
    for sub in acceptable_substrings:
        if sub in supporting_span_text:
            return True
    return False


def _evidence_source_key(obs, case: dict) -> Optional[str]:
    """For two-evidence-item cases (source_text_a/source_text_b), map the
    accepted observation's source_evidence_id ("EA"/"EB") back to the
    label's "a"/"b" disambiguator. Single-evidence cases return None
    (source is not a discriminator there)."""
    if "source_text_a" not in case:
        return None
    eid = getattr(obs, "source_evidence_id", None)
    if eid == "EA":
        return "a"
    if eid == "EB":
        return "b"
    return None


def _build_inferred_permitted(case: dict) -> list:
    """Milestone 2B.2 closure: `inferred_candidates_permitted` entries
    (e.g. case 16, case 21) describe a fact that must NOT surface as an
    EXPLICIT-basis labeled expectation but IS a legitimate, anticipated
    INFERRED_CANDIDATE finding if the model produces it. Prior to this
    fix, `classify_accepted_observations` never consulted this list —
    a matching INFERRED_CANDIDATE observation fell through to the
    wrong-type fallback (step 2 below) whenever its span happened to
    overlap an unrelated slot's span_substrings, exactly what happened
    to case 21 in the prompt_v2_ontology_eval1 live run: the model
    correctly produced an INFERRED_CANDIDATE ObjectiveCandidate (never
    EXPLICIT, per OC-01) and the classifier still penalized it as
    WRONG_TYPE solely because it lacked a normal expected-observation
    slot. Plain dicts with a local `claimed` flag, same pattern as
    `_build_slots`, so a second observation can't double-claim the same
    permitted entry."""
    permitted = []
    for idx, ic in enumerate(case.get("inferred_candidates_permitted", [])):
        if "type" not in ic:
            # Milestone 2B/2B.1 entries (e.g. case 15) describe a free-text
            # inferred claim with no `type`/`span_substrings` at all — not
            # matchable against a span-grounded accepted observation, and
            # not what this fix targets. Skip rather than error.
            continue
        permitted.append({
            "index": idx,
            "type": ic["type"],
            "basis": ic.get("basis"),
            "span_substrings": list(ic.get("span_substrings", [])),
            "claimed": False,
        })
    return permitted


def _build_slots(case: dict) -> list:
    """Slots are plain JSON-safe dicts (not dataclasses) so the exact
    same structure can be used both to compute aggregate metrics and to
    be written straight into the evaluation report — no separate
    serialization step, no risk of a raw object leaking into json.dump."""
    slots = []
    for idx, eo in enumerate(case.get("expected_observations", [])):
        slots.append({
            "index": idx,
            "canonical_type": eo["type"],
            "alternate_types": list(eo.get("alternate_types", [])),
            "role": eo.get("role", "primary"),
            "basis": eo.get("basis"),
            "span_substrings": list(eo.get("span_substrings", [])),
            "source": eo.get("source"),
            "ontology_ambiguity": bool(eo.get("ontology_ambiguity", False)),
            "claimed": False,
        })
    return slots


def classify_accepted_observations(case: dict, accepted: tuple) -> dict:
    """Milestone 2B.1 core scorer. Classifies every span-grounded accepted
    observation (the 7 positive types; MissingInformationCandidate is
    scored separately by `score_missing_information` since it has no
    span and represents a different kind of claim) into exactly one of
    the six buckets described in this module's docstring.

    Returns {"classified": [...], "slots": [...]} — `classified` is one
    entry per span-grounded accepted observation (in original order);
    `slots` is every labeled expectation with its final claimed/unclaimed
    state, used by `aggregate_metrics` to compute recall."""
    slots = _build_slots(case)
    irrelevant = case.get("grounded_but_irrelevant_permitted", [])
    inferred_permitted = _build_inferred_permitted(case)

    span_grounded = [o for o in accepted if hasattr(o, "source_span")]
    classified = []
    # Parallel list of raw observations, same indices as `classified`,
    # used only transiently within this function for the duplicate check
    # below. Never stored on `classified` entries or returned — keeps
    # every entry in `classified` JSON-safe on its own.
    classified_raw_obs = []

    for obs in span_grounded:
        obs_type_name = type(obs).__name__
        obs_span_text = obs.source_span.text
        obs_basis = obs.basis.value if hasattr(obs.basis, "value") else obs.basis
        source_key = _evidence_source_key(obs, case)

        entry = {
            "summary": _obs_summary(obs),
            "classification": None, "matched_slot_index": None,
            "expected_basis": None, "basis_correct": None,
            "ontology_ambiguity_triggered": False, "needs_relevance_review": False,
            "matched_inferred_permitted": False,
        }

        # 1. required/optional-valid slot match: type (canonical or
        #    alternate) + span + source, unclaimed only.
        found_slot = None
        for slot in slots:
            if slot["claimed"]:
                continue
            if slot["source"] is not None and slot["source"] != source_key:
                continue
            acceptable_type_names = {slot["canonical_type"], *slot["alternate_types"]}
            if not any(_dataclass_matches_type(obs_type_name, t) for t in acceptable_type_names):
                continue
            if not _span_matches(obs_span_text, slot["span_substrings"]):
                continue
            found_slot = slot
            break

        if found_slot is not None:
            found_slot["claimed"] = True
            entry["matched_slot_index"] = found_slot["index"]
            entry["expected_basis"] = found_slot["basis"]
            entry["basis_correct"] = (obs_basis == found_slot["basis"]) if found_slot["basis"] else None
            entry["ontology_ambiguity_triggered"] = (
                found_slot["ontology_ambiguity"]
                and not _dataclass_matches_type(obs_type_name, found_slot["canonical_type"])
            )
            entry["classification"] = (
                "MATCHED_EXPECTED" if found_slot["role"] in ("primary", "supporting") else "VALID_UNLABELED"
            )
            classified.append(entry)
            classified_raw_obs.append(obs)
            continue

        # 1.5. Milestone 2B.2 closure: a permitted inferred candidate
        #    (`inferred_candidates_permitted`) — unclaimed, type + basis
        #    (must be INFERRED_CANDIDATE) + span all matching. Checked
        #    BEFORE the wrong-type fallback so a genuinely correct
        #    INFERRED_CANDIDATE finding is never penalized merely because
        #    it lacks a normal expected-observation slot and its span
        #    happens to overlap an unrelated slot's span_substrings (the
        #    exact case 21 defect: an EXPLICIT-basis observation for the
        #    same underlying fact still correctly falls through to
        #    ordinary WRONG_TYPE scoring below — only a basis-correct
        #    INFERRED_CANDIDATE match is credited here).
        found_inferred = None
        for ip in inferred_permitted:
            if ip["claimed"]:
                continue
            if not _dataclass_matches_type(obs_type_name, ip["type"]):
                continue
            if obs_basis != ip["basis"]:
                continue
            if not _span_matches(obs_span_text, ip["span_substrings"]):
                continue
            found_inferred = ip
            break
        if found_inferred is not None:
            found_inferred["claimed"] = True
            entry["classification"] = "VALID_UNLABELED"
            entry["matched_inferred_permitted"] = True
            classified.append(entry)
            classified_raw_obs.append(obs)
            continue

        # 2. right fact (span match), wrong type — unclaimed slots only,
        #    regardless of type this time.
        wrong_type_slot = None
        for slot in slots:
            if slot["claimed"]:
                continue
            if slot["source"] is not None and slot["source"] != source_key:
                continue
            if _span_matches(obs_span_text, slot["span_substrings"]):
                wrong_type_slot = slot
                break
        if wrong_type_slot is not None:
            entry["matched_slot_index"] = wrong_type_slot["index"]
            entry["expected_basis"] = wrong_type_slot["basis"]
            entry["classification"] = "WRONG_TYPE"
            classified.append(entry)
            classified_raw_obs.append(obs)
            continue

        # 3. grounded but explicitly not Customer-Health relevant.
        irrelevant_match = None
        for irr in irrelevant:
            if not _dataclass_matches_type(obs_type_name, irr["type"]):
                continue
            if _span_matches(obs_span_text, irr.get("span_substrings", [])):
                irrelevant_match = irr
                break
        if irrelevant_match is not None:
            entry["classification"] = "IRRELEVANT_BUT_GROUNDED"
            classified.append(entry)
            classified_raw_obs.append(obs)
            continue

        # 4. duplicate of an already-classified fact from this same case
        #    (different type than the earlier one, same evidence item,
        #    overlapping span — same-type overlaps are already collapsed
        #    by the pipeline's own dedup before this ever runs).
        is_duplicate = False
        for prior_entry, prior_obs in zip(classified, classified_raw_obs):
            if prior_entry["classification"] not in ("MATCHED_EXPECTED", "VALID_UNLABELED", "WRONG_TYPE"):
                continue
            if getattr(prior_obs, "source_evidence_id", None) != obs.source_evidence_id:
                continue
            prior_span = prior_obs.source_span.text
            if prior_span == obs_span_text or prior_span in obs_span_text or obs_span_text in prior_span:
                is_duplicate = True
                break
        if is_duplicate:
            entry["classification"] = "DUPLICATE"
            classified.append(entry)
            classified_raw_obs.append(obs)
            continue

        # 5. genuinely unanticipated: grounded (guaranteed by the
        #    pipeline), not excluded, not a duplicate. Credited as
        #    VALID_UNLABELED but flagged for a human relevance check
        #    rather than silently assumed Customer-Health relevant.
        entry["classification"] = "VALID_UNLABELED"
        entry["needs_relevance_review"] = True
        classified.append(entry)
        classified_raw_obs.append(obs)

    return {"classified": classified, "slots": slots}


def score_missing_information(case: dict, accepted: tuple) -> dict:
    """Separate from the six-way classifier above: MissingInformationCandidate
    has no span and represents a different kind of claim ("not found in
    reviewed evidence"), so it is matched on `missing_item` text against
    `missing_information_expected` rather than span-matched."""
    expected = case.get("missing_information_expected", [])
    found_items = [o.missing_item for o in accepted if hasattr(o, "missing_item")]
    matched = [e for e in expected if e["missing_item"] in found_items]
    missed = [e for e in expected if e["missing_item"] not in found_items]
    return {
        "expected": [e["missing_item"] for e in expected],
        "found": found_items,
        "matched_count": len(matched),
        "missed": [e["missing_item"] for e in missed],
    }


def _build_candidate_risk_signal_slots(case: dict) -> list:
    slots = []
    for idx, crs in enumerate(case.get("expected_candidate_risk_signals", [])):
        slots.append({
            "index": idx,
            "mechanism": crs["mechanism"],
            "proposed_severity_tier": crs["proposed_severity_tier"],
            "role": crs.get("role", "primary"),
            "span_substrings": list(crs["span_substrings"]),
            "supporting_observation_span_substrings": list(crs["supporting_observation_span_substrings"]),
            "claimed": False,
        })
    return slots


def _build_candidate_evidence_classification_slots(case: dict) -> list:
    slots = []
    for idx, cec in enumerate(case.get("expected_candidate_evidence_classifications", [])):
        slots.append({
            "index": idx,
            "proposed_basis": cec["proposed_basis"],
            "supports": cec["supports"],
            "role": cec.get("role", "primary"),
            "span_substrings": list(cec["span_substrings"]),
            "supporting_observation_span_substrings": list(cec["supporting_observation_span_substrings"]),
            "claimed": False,
        })
    return slots


def _resolved_reference_matches(candidate, accepted_by_id: dict, expected_span_substrings: list) -> Optional[bool]:
    """Whether `candidate.resolved_observation_id` points at an accepted
    observation whose own span matches the slot's
    `supporting_observation_span_substrings` — i.e. did the candidate
    reference the RIGHT supporting observation, not just A valid one.
    Returns None if the reference cannot be resolved at all (should not
    happen for anything that reached `accepted`/`candidate_risk_signals`,
    since pipeline.py only emits a resolved id for a reference that
    survived validation — checked defensively rather than assumed)."""
    target = accepted_by_id.get(candidate.resolved_observation_id)
    if target is None or not hasattr(target, "source_span"):
        return None
    return _span_matches(target.source_span.text, expected_span_substrings)


def classify_candidate_classifications(case: dict, accepted: tuple, candidate_risk_signals: tuple,
                                        candidate_evidence_classifications: tuple) -> dict:
    """Milestone 2C scorer for the two candidate-classification output
    types — see this module's docstring for the outcome definitions.
    `accepted` is the SAME tuple `classify_accepted_observations` scores
    (the 8 Milestone 2B types), needed here only to resolve what each
    candidate's `resolved_observation_id` actually points at."""
    accepted_by_id = {o.system.observation_id: o for o in accepted if hasattr(o, "system")}

    def _classify_one(candidates: tuple, slots: list, field_a: str, field_b: str) -> list:
        # field_a/field_b name BOTH the slot dict key and the dataclass
        # attribute — true by construction for both candidate types
        # (mechanism/proposed_severity_tier; proposed_basis/supports).
        classified = []
        for cand in candidates:
            span_text = cand.source_span.text
            entry = {
                "summary": _obs_summary(cand),
                "classification": None,
                "matched_slot_index": None,
                f"{field_a}_correct": None,
                f"{field_b}_correct": None,
                "reference_correct": None,
            }
            found_slot = None
            for slot in slots:
                if slot["claimed"]:
                    continue
                if _span_matches(span_text, slot["span_substrings"]):
                    found_slot = slot
                    break
            if found_slot is not None:
                found_slot["claimed"] = True
                entry["matched_slot_index"] = found_slot["index"]
                entry["classification"] = "MATCHED_EXPECTED"
                entry[f"{field_a}_correct"] = (getattr(cand, field_a) == found_slot[field_a])
                entry[f"{field_b}_correct"] = (getattr(cand, field_b) == found_slot[field_b])
                entry["reference_correct"] = _resolved_reference_matches(
                    cand, accepted_by_id, found_slot["supporting_observation_span_substrings"],
                )
            else:
                entry["classification"] = "UNEXPECTED_CANDIDATE"
            classified.append(entry)
        return classified

    risk_slots = _build_candidate_risk_signal_slots(case)
    evidence_slots = _build_candidate_evidence_classification_slots(case)

    return {
        "risk_signals": {
            "classified": _classify_one(candidate_risk_signals, risk_slots, "mechanism", "proposed_severity_tier"),
            "slots": risk_slots,
        },
        "evidence_classifications": {
            "classified": _classify_one(
                candidate_evidence_classifications, evidence_slots, "proposed_basis", "supports",
            ),
            "slots": evidence_slots,
        },
    }


def _build_dimension_qualifier_slots(case: dict, expected_key: str) -> list:
    slots = []
    for idx, dq in enumerate(case.get(expected_key, [])):
        slots.append({
            "index": idx,
            "qualifier": dq["qualifier"],
            "role": dq.get("role", "primary"),
            "supporting_observation_span_substrings": list(dq["supporting_observation_span_substrings"]),
            "claimed": False,
        })
    return slots


def classify_dimension_qualifiers(case: dict, accepted: tuple, candidate_d2_qualifiers: tuple,
                                   candidate_d6_qualifiers: tuple) -> dict:
    """Milestone 4B scorer for the two D2/D6 candidate-qualifier output
    channels — see this module's docstring for the full rationale and how
    this differs structurally from classify_candidate_classifications.
    `accepted` must be the SAME stage-1 tuple the qualifiers' own
    resolved_observation_id values were resolved against."""
    accepted_by_id = {o.system.observation_id: o for o in accepted if hasattr(o, "system")}

    def _classify_one(candidates: tuple, slots: list) -> list:
        classified = []
        for cand in candidates:
            entry = {
                "summary": _obs_summary(cand),
                "classification": None,
                "matched_slot_index": None,
                "qualifier_correct": None,
            }
            supporting = accepted_by_id.get(cand.resolved_observation_id)
            supporting_span_text = (
                supporting.source_span.text
                if supporting is not None and hasattr(supporting, "source_span")
                else None
            )
            found_slot = None
            if supporting_span_text is not None:
                for slot in slots:
                    if slot["claimed"]:
                        continue
                    if _dimension_qualifier_span_matches(
                        supporting_span_text, slot["supporting_observation_span_substrings"],
                    ):
                        found_slot = slot
                        break
            if found_slot is not None:
                found_slot["claimed"] = True
                entry["matched_slot_index"] = found_slot["index"]
                entry["classification"] = "MATCHED_EXPECTED"
                entry["qualifier_correct"] = (cand.qualifier == found_slot["qualifier"])
            else:
                entry["classification"] = "UNEXPECTED_CANDIDATE"
            classified.append(entry)
        return classified

    d2_slots = _build_dimension_qualifier_slots(case, "expected_dimension_d2_qualifiers")
    d6_slots = _build_dimension_qualifier_slots(case, "expected_dimension_d6_qualifiers")

    return {
        "d2_qualifiers": {
            "classified": _classify_one(candidate_d2_qualifiers, d2_slots),
            "slots": d2_slots,
        },
        "d6_qualifiers": {
            "classified": _classify_one(candidate_d6_qualifiers, d6_slots),
            "slots": d6_slots,
        },
    }


# Milestone 4B v3 evaluator-provenance checkpoint: the atomic-predicate
# architecture's own array keys / rejection-reason values, duplicated as
# plain string literals here (never imported from extraction.enums) to
# preserve this module's existing, deliberate discipline of staying
# decoupled from the extraction package and duck-typing every object it
# scores -- the SAME discipline every other function in this file already
# follows (classify_dimension_qualifiers above never imports
# extraction.enums either). Kept in sync with extraction.enums.
# DIMENSION_QUALIFIER_TYPE_TO_ATOMIC_PREDICATE_ARRAY_KEY /
# RejectionReason by comment cross-reference, exactly like json_schemas.
# py's own hardcoded-literal D2/D6 vocabularies.
_ATOMIC_PREDICATE_ARRAY_KEYS = {"candidate_d2_atomic_predicates", "candidate_d6_atomic_predicates"}
_ATOMIC_PREDICATE_REJECTION_REASONS = {
    "DIMENSION_QUALIFIER_COMPOUND_PREDICATE_NOT_GROUNDED",
    "DIMENSION_QUALIFIER_DUPLICATE_ATOMIC_PREDICATE",
}
# The 2 compound qualifiers this architecture composes, by dimension --
# used only to recognize which already-produced CandidateDimensionQualifier
# entries are themselves compositions (never a direct model proposal),
# so `composed` can be set correctly without needing a separate signal
# threaded all the way from the pipeline. Mirrors extraction.enums.
# DIMENSION_QUALIFIER_TYPE_TO_COMPOSED_QUALIFIER's own two values.
_COMPOSED_QUALIFIER_BY_DIMENSION = {
    "D2": "AUTOMATION_RELIABLE_LOW_LOGIN_OK",
    "D6": "CHAMPION_LOST_NO_SUCCESSOR",
}


def dimension_qualifier_atomic_predicate_detail(
    predicate_evidence: tuple, rejected: tuple,
    candidate_d2_qualifiers: tuple, candidate_d6_qualifiers: tuple,
) -> list[dict]:
    """Milestone 4B v3 evaluator-provenance checkpoint. Produces ONE
    JSON-safe, per-predicate audit list merging two structurally distinct
    sources into a single view a live-eval report can inspect directly,
    without needing a code-level rerun to see which specific atomic
    predicate(s) drove — or blocked — a compound-qualifier composition:

      1. `predicate_evidence` — every GROUNDED, non-duplicate atomic
         predicate proposed in the run (extraction.pipeline.
         ExtractionResult.dimension_qualifier_predicate_evidence), for
         BOTH channels, INCLUDING predicates from incomplete sets that
         never composed anything. Each becomes one entry with
         `grounding_passed: true`, `rejection_reason: null`.
      2. `rejected` — the run's full ExtractionValidationFailure tuple,
         filtered down to just the 2 atomic-predicate-specific rejection
         reasons (ungrounded evidence_text, duplicate predicate_id within
         the same call) — every other rejection reason/type in `rejected`
         is ignored here (already covered by the existing generic
         `rejected_detail` report field). Each becomes one entry with
         `grounding_passed: false`, `rejection_reason` set to the specific
         reason. `resolved_observation_id`/`dimension` come from the
         OPTIONAL fields added to ExtractionValidationFailure for exactly
         this purpose (extraction/pipeline.py) — both None only for the
         (structurally unreachable in practice) case where an atomic
         predicate item was rejected before shape validation could even
         run.

    `composed` is true only for grounding_passed entries whose own
    (resolved_observation_id, dimension) pair also appears among the
    run's ALREADY-PRODUCED compound-qualifier candidates (candidate_d2_
    qualifiers / candidate_d6_qualifiers, filtered to the qualifier value
    the composer actually synthesizes for that channel) — i.e. this
    predicate was part of a set that was BOTH complete AND grounded.
    Always false for rejected entries (a rejected predicate can never have
    contributed to a successful composition, by construction — either it
    was a duplicate the composer never counted, or it failed the exact
    grounding check the composer requires).

    Read-only: this function computes a view over already-produced pipeline
    output. It does not re-run grounding, re-run composition, or change
    which candidates the pipeline itself considered composed — it only
    makes that existing, already-final decision visible per predicate."""
    composed_keys = set()
    for cand in candidate_d2_qualifiers:
        if cand.qualifier == _COMPOSED_QUALIFIER_BY_DIMENSION["D2"]:
            composed_keys.add((cand.resolved_observation_id, "D2"))
    for cand in candidate_d6_qualifiers:
        if cand.qualifier == _COMPOSED_QUALIFIER_BY_DIMENSION["D6"]:
            composed_keys.add((cand.resolved_observation_id, "D6"))

    detail: list[dict] = []

    for e in predicate_evidence:
        dim_value = e.dimension.value if hasattr(e.dimension, "value") else e.dimension
        basis_value = e.basis.value if hasattr(e.basis, "value") else e.basis
        detail.append({
            "predicate_id": e.predicate_id,
            "dimension": dim_value,
            "resolved_observation_id": e.resolved_observation_id,
            "evidence_text": e.evidence_text,
            "basis": basis_value,
            "grounding_passed": True,
            "rejection_reason": None,
            "composed": (e.resolved_observation_id, dim_value) in composed_keys,
        })

    for r in rejected:
        reason_value = r.reason.value if hasattr(r.reason, "value") else r.reason
        if r.observation_type not in _ATOMIC_PREDICATE_ARRAY_KEYS:
            continue
        if reason_value not in _ATOMIC_PREDICATE_REJECTION_REASONS:
            continue
        raw_item = r.raw_item or {}
        dim_value = getattr(r, "dimension", None)
        dim_value = dim_value.value if hasattr(dim_value, "value") else dim_value
        detail.append({
            "predicate_id": raw_item.get("predicate_id"),
            "dimension": dim_value,
            "resolved_observation_id": getattr(r, "resolved_observation_id", None),
            "evidence_text": raw_item.get("evidence_text"),
            "basis": raw_item.get("basis"),
            "grounding_passed": False,
            "rejection_reason": reason_value,
            "composed": False,
        })

    return detail


def aggregate_metrics(case_results: list[dict]) -> dict:
    """`case_results` is the list produced by run_eval.py, one dict per
    labeled case. Computes the Milestone 2B.1 metrics framework: three
    separate precision numbers, three separate recall numbers, the
    six-way classification totals, plus contradiction / explicit-vs-
    inferred / grounding / boundary reporting carried forward from
    Milestone 2B. Deliberately does not collapse precision into one
    number (Milestone 2B.1 spec §6)."""
    total_cases = len(case_results)

    calls_total = sum(c["schema"]["calls_made"] for c in case_results)
    valid_first_attempt = sum(1 for c in case_results if c["schema"]["calls_made"] == 1 and c["request_failure"] is None)
    required_retry = sum(1 for c in case_results if c["schema"]["calls_made"] >= 2)
    still_invalid_after_retry = sum(1 for c in case_results if c["request_failure"] is not None)
    schema_valid_rate = (
        sum(1 for c in case_results if c["request_failure"] is None) / total_cases if total_cases else None
    )

    # ---- six-way classification totals ----
    class_counts = {
        "MATCHED_EXPECTED": 0, "VALID_UNLABELED": 0, "WRONG_TYPE": 0,
        "UNSUPPORTED": 0, "DUPLICATE": 0, "IRRELEVANT_BUT_GROUNDED": 0,
    }
    total_accepted_span_grounded = 0
    needs_relevance_review_count = 0
    ontology_ambiguity_triggered_count = 0
    for c in case_results:
        for entry in c["classification"]["classified"]:
            class_counts[entry["classification"]] += 1
            total_accepted_span_grounded += 1
            if entry["needs_relevance_review"]:
                needs_relevance_review_count += 1
            if entry["ontology_ambiguity_triggered"]:
                ontology_ambiguity_triggered_count += 1

    def _rate(numerator: int, denom: int) -> Optional[float]:
        return (numerator / denom) if denom else None

    supported = class_counts["MATCHED_EXPECTED"] + class_counts["VALID_UNLABELED"] + \
        class_counts["WRONG_TYPE"] + class_counts["IRRELEVANT_BUT_GROUNDED"]
    relevant = class_counts["MATCHED_EXPECTED"] + class_counts["VALID_UNLABELED"]

    supported_extraction_precision = _rate(supported, total_accepted_span_grounded)
    label_match_precision = _rate(class_counts["MATCHED_EXPECTED"], total_accepted_span_grounded)
    relevant_extraction_precision = _rate(relevant, total_accepted_span_grounded)

    # ---- recall, by slot role, across all cases ----
    primary_total = primary_matched = 0
    required_total = required_matched = 0
    optional_total = optional_matched = 0
    for c in case_results:
        for slot in c["classification"]["slots"]:
            if slot["role"] == "primary":
                primary_total += 1
                primary_matched += 1 if slot["claimed"] else 0
            if slot["role"] in ("primary", "supporting"):
                required_total += 1
                required_matched += 1 if slot["claimed"] else 0
            if slot["role"] == "optional-valid":
                optional_total += 1
                optional_matched += 1 if slot["claimed"] else 0

    primary_recall = _rate(primary_matched, primary_total)
    required_recall = _rate(required_matched, required_total)
    optional_valid_capture_rate = _rate(optional_matched, optional_total)

    # ---- source grounding (unchanged meaning: structural, from the
    # pipeline's own span-rejection accounting) ----
    span_valid_accepted = sum(c["source_grounding"]["accepted_with_valid_span"] for c in case_results)
    span_rejections = sum(c["source_grounding"]["span_rejections"] for c in case_results)
    unknown_evidence_rejections = sum(c["source_grounding"]["unknown_evidence_rejections"] for c in case_results)

    # ---- explicit vs inferred: scored only on MATCHED_EXPECTED /
    # VALID_UNLABELED entries with a declared expected basis ----
    basis_correct = basis_incorrect = 0
    for c in case_results:
        for entry in c["classification"]["classified"]:
            if entry["classification"] not in ("MATCHED_EXPECTED", "VALID_UNLABELED"):
                continue
            if entry["basis_correct"] is None:
                continue
            if entry["basis_correct"]:
                basis_correct += 1
            else:
                basis_incorrect += 1

    boundary_attempted = sum(c["boundary"]["attempted"] for c in case_results)
    boundary_accepted = sum(c["boundary"]["accepted"] for c in case_results)

    # ---- Milestone 2C: candidate risk signals / evidence classifications ----
    _empty_cc = {"classified": [], "slots": []}

    def _candidate_summary(get_section, report_key: str = "candidate_classification") -> dict:
        matched = unexpected = 0
        reference_correct = reference_total = 0
        slot_total = slot_matched = 0
        for c in case_results:
            section = get_section(c.get(report_key, {}))
            for entry in section.get("classified", _empty_cc["classified"]):
                if entry["classification"] == "MATCHED_EXPECTED":
                    matched += 1
                else:
                    unexpected += 1
                if entry.get("reference_correct") is not None:
                    reference_total += 1
                    reference_correct += 1 if entry["reference_correct"] else 0
            for slot in section.get("slots", _empty_cc["slots"]):
                slot_total += 1
                slot_matched += 1 if slot["claimed"] else 0
        return {
            "matched_expected": matched,
            "unexpected_candidates": unexpected,
            "reference_correct": reference_correct,
            "reference_checked": reference_total,
            "reference_accuracy_rate": _rate(reference_correct, reference_total),
            "expected_total": slot_total,
            "expected_matched": slot_matched,
            "missed_candidate_count": slot_total - slot_matched,
            "recall": _rate(slot_matched, slot_total),
            "false_candidate_rate": _rate(unexpected, matched + unexpected),
        }

    def _field_accuracy(get_section, field: str) -> Optional[float]:
        correct = total = 0
        for c in case_results:
            section = get_section(c.get("candidate_classification", {}))
            for entry in section.get("classified", _empty_cc["classified"]):
                if entry["classification"] != "MATCHED_EXPECTED":
                    continue
                if entry.get(f"{field}_correct") is None:
                    continue
                total += 1
                correct += 1 if entry[f"{field}_correct"] else 0
        return _rate(correct, total)

    candidate_risk_signal_summary = _candidate_summary(lambda cc: cc.get("risk_signals", _empty_cc))
    candidate_risk_signal_summary["mechanism_accuracy_rate"] = _field_accuracy(
        lambda cc: cc.get("risk_signals", _empty_cc), "mechanism",
    )
    candidate_risk_signal_summary["tier_accuracy_rate"] = _field_accuracy(
        lambda cc: cc.get("risk_signals", _empty_cc), "proposed_severity_tier",
    )

    candidate_evidence_classification_summary = _candidate_summary(
        lambda cc: cc.get("evidence_classifications", _empty_cc),
    )
    candidate_evidence_classification_summary["basis_accuracy_rate"] = _field_accuracy(
        lambda cc: cc.get("evidence_classifications", _empty_cc), "proposed_basis",
    )
    candidate_evidence_classification_summary["supports_accuracy_rate"] = _field_accuracy(
        lambda cc: cc.get("evidence_classifications", _empty_cc), "supports",
    )

    # ---- Milestone 4B: D2/D6 candidate qualifiers. Reuses the same
    # _candidate_summary shape as the 2C types above (matched/unexpected/
    # recall/false-candidate-rate), plus a dedicated qualifier_accuracy_
    # rate (there is no separate mechanism/tier pair here -- one field,
    # `qualifier`, per proposal) and a distinct stage-failure counter,
    # since dimension_qualifier_stage_failure is a whole-CALL failure
    # mode with no equivalent in the 2C candidate-classification types
    # above (those failures simply surface as ordinary per-item
    # rejections within run_extraction()'s own request_failure/rejected
    # accounting). ----
    def _qualifier_field_accuracy(get_section) -> Optional[float]:
        correct = total = 0
        for c in case_results:
            section = get_section(c.get("dimension_qualifier_classification", {}))
            for entry in section.get("classified", _empty_cc["classified"]):
                if entry["classification"] != "MATCHED_EXPECTED":
                    continue
                if entry.get("qualifier_correct") is None:
                    continue
                total += 1
                correct += 1 if entry["qualifier_correct"] else 0
        return _rate(correct, total)

    candidate_d2_qualifier_summary = _candidate_summary(
        lambda dq: dq.get("d2_qualifiers", _empty_cc), report_key="dimension_qualifier_classification",
    )
    candidate_d2_qualifier_summary["qualifier_accuracy_rate"] = _qualifier_field_accuracy(
        lambda dq: dq.get("d2_qualifiers", _empty_cc),
    )
    candidate_d6_qualifier_summary = _candidate_summary(
        lambda dq: dq.get("d6_qualifiers", _empty_cc), report_key="dimension_qualifier_classification",
    )
    candidate_d6_qualifier_summary["qualifier_accuracy_rate"] = _qualifier_field_accuracy(
        lambda dq: dq.get("d6_qualifiers", _empty_cc),
    )
    dimension_qualifier_stage_failures = sum(
        1 for c in case_results if c.get("dimension_qualifier_stage_failure") is not None
    )
    # Milestone 4B isolated-classifier architecture checkpoint: a case-
    # level count (above) conflates "1 observation failed" with "5
    # observations failed" -- both show up as exactly one case with a
    # non-None dimension_qualifier_stage_failure. This second counter
    # sums the per-observation detail list instead (eval/run_eval.py's
    # score_case's dimension_qualifier_failures_detail), so a report can
    # distinguish "isolated, single-observation classifier unavailability"
    # from "the classifier was unavailable for most of this case's
    # eligible observations" -- exactly the isolation guarantee the
    # architecture is supposed to provide (one failed call must not look
    # like the whole stage failed).
    dimension_qualifier_failed_observation_count = sum(
        len(c.get("dimension_qualifier_failures_detail", [])) for c in case_results
    )

    # ---- contradictions ----
    contradiction_cases = [c for c in case_results if "contradiction_check" in c]
    contradictions_detected = sum(1 for c in contradiction_cases if c["contradiction_check"]["detected"])
    contradictions_both_resolved = sum(1 for c in contradiction_cases if c["contradiction_check"]["both_sides_resolved"])
    contradictions_all_candidate = sum(1 for c in contradiction_cases if c["contradiction_check"]["status_all_candidate"])

    return {
        "total_cases": total_cases,
        "schema_performance": {
            "total_model_calls": calls_total,
            "valid_first_attempt_cases": valid_first_attempt,
            "cases_requiring_retry": required_retry,
            "cases_still_invalid_after_retry": still_invalid_after_retry,
            "schema_valid_rate": schema_valid_rate,
        },
        "classification_totals": class_counts,
        "total_accepted_span_grounded_observations": total_accepted_span_grounded,
        "needs_relevance_review_count": needs_relevance_review_count,
        "ontology_ambiguity_triggered_count": ontology_ambiguity_triggered_count,
        "precision": {
            "supported_extraction_precision": supported_extraction_precision,
            "label_match_precision": label_match_precision,
            "relevant_extraction_precision": relevant_extraction_precision,
            "note": (
                "Three separate numbers by design (Milestone 2B.1 spec §6) — "
                "not collapsed into one. supported = grounded regardless of "
                "label match; label_match = matches a required benchmark "
                "expectation exactly (the old-style precision number); "
                "relevant = grounded AND either a required or optional-valid "
                "Customer-Health-relevant match. Approximate, engineering-"
                "grade measures, not a validated model-performance claim."
            ),
        },
        "recall": {
            "primary_recall": primary_recall,
            "primary_total": primary_total,
            "required_recall": required_recall,
            "required_total": required_total,
            "optional_valid_capture_rate": optional_valid_capture_rate,
            "optional_valid_total": optional_total,
        },
        "source_grounding": {
            "accepted_with_valid_span": span_valid_accepted,
            "span_validation_failures": span_rejections,
            "unknown_evidence_reference_rejections": unknown_evidence_rejections,
        },
        "explicit_vs_inferred": {
            "correctly_labeled": basis_correct,
            "incorrectly_labeled": basis_incorrect,
        },
        "governance_boundary": {
            "prohibited_content_attempted_by_model": boundary_attempted,
            "prohibited_content_accepted_by_pipeline": boundary_accepted,
            "target_for_accepted": 0,
        },
        "contradictions": {
            "cases_with_expected_contradiction": len(contradiction_cases),
            "detected": contradictions_detected,
            "status_all_candidate": contradictions_all_candidate,
            "both_sides_resolved": contradictions_both_resolved,
        },
        "candidate_risk_signals": candidate_risk_signal_summary,
        "candidate_evidence_classifications": candidate_evidence_classification_summary,
        "candidate_d2_qualifiers": candidate_d2_qualifier_summary,
        "candidate_d6_qualifiers": candidate_d6_qualifier_summary,
        "dimension_qualifier_stage_failures": dimension_qualifier_stage_failures,
        "dimension_qualifier_failed_observation_count": dimension_qualifier_failed_observation_count,
    }
