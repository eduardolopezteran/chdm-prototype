"""
Build Milestone 2 — extraction instructions (spec §8: "Prompts may guide
extraction only... Do not encode CHDM deterministic decision rules inside
the prompt").

This module holds only the instruction TEXT and the request-framing
convention. No dimension state rules, no severity thresholds, no OP/ER
logic, nothing from the Methodology Registry appears here — only the
general extraction boundary (what a candidate observation is, what it
must never become) and the mechanical contract for how source spans and
character offsets must be reported.

Design note: `EvidenceObject.indicator_observation` is used here as the
raw text body being reviewed (consistent with its existing docstring,
"what was actually observed (not an interpretation)") and
`EvidenceObject.source` as its channel/type label. Neither field is
constructed or consumed anywhere in the Milestone 1 engine or test suite
(verified: no `engine/*.py` or Milestone 1 test imports EvidenceObject at
all) — Milestone 2 is the first real consumer, so this is a genuinely new
usage, not a repurposing of load-bearing Milestone 1 behavior.
"""

from __future__ import annotations

from typing import Optional

from domain.enums import DimensionCode
from domain.evidence import EvidenceObject

EXTRACTION_TOOL_NAME = "record_extracted_observations"

# Bumped on every meaningful instruction-text revision (Milestone 2B spec
# §5: "version the prompt" after each refinement). Recorded in every
# evaluation report so results are reproducible against a known prompt.
#
# v1 -> v2 (Milestone 2B.2, OC-01 ontology alignment): two changes.
#   1. The span-citation instruction previously told the model to compute
#      and report start_char/end_char offsets. That stopped being true at
#      the Milestone 2B span-grounding fix (extraction/json_schemas.py):
#      the model-facing schema has not accepted offsets since then --
#      the application derives them from the exact quoted text. v1's
#      wording was left as-is at the time (explicit instruction: no
#      prompt changes in that pass); v2 corrects it now that a prompt
#      revision is separately authorized.
#   2. Added concise ObjectiveCandidate / AdoptionObservation /
#      StrategicObservation precedence guidance (OC-01, approved in the
#      "ObjectiveCandidate Ontology Clarification" checkpoint) and an
#      explicit/inferred objective clarification, targeting the
#      misclassification pattern observed in baseline_v1_fix1_eval2.json
#      (cases 05, 12b, 19, 20). No CHDM methodology, DMEG/Reliability/
#      OP/ER, or governed-conclusion content was added -- this remains
#      extraction-boundary and type-taxonomy guidance only (spec §8).
#
# v2 -> v3 (Milestone 2C, Candidate CHDM Classification Extraction): one
#   addition -- guidance for the two new candidate-classification output
#   arrays, candidate_risk_signals and candidate_evidence_classifications
#   (json_schemas.py's CANDIDATE_RISK_SIGNAL_SCHEMA /
#   CANDIDATE_EVIDENCE_CLASSIFICATION_SCHEMA). Only the 4 MVP-implemented
#   risk mechanisms (CR-01/02/03/08) are named -- CR-04/05/06/07 are
#   deferred/scenario-lab-only (registry/risk_mechanisms.yaml, spec
#   §15.3) and are never offered to the model, consistent with FR-15.3
#   ("must state which mechanisms it does not evaluate"). No severity
#   threshold, state-rule, or other deterministic registry condition is
#   included -- only the plain descriptive label for each mechanism/basis
#   value, exactly the same restraint the v1/v2 text already exercises
#   for observation types (spec §8: "Do not encode CHDM deterministic
#   decision rules inside the prompt").
#
# v3 -> v3.1 (Milestone 2C, narrow live-gate refinement; CONDITIONAL
#   decision on prompt_v3_2c_eval1, approved as exactly two changes, no
#   other scope): the live run showed two regressions on the pre-existing
#   Prompt v2 benchmark (cases 17, 20a: "completed phase 1 of the rollout
#   on schedule" / "implementation was completed ahead of schedule" both
#   regressed from StrategicObservation to AdoptionObservation) plus two
#   candidate risk signals force-fit onto facts that don't cleanly match
#   any of the 4 enabled mechanisms (case 20: CR-02 for open project
#   tasks; case 21: CR-08 for a support-ticket pattern). Both fixes are
#   additive guidance only -- no observation type, schema field, mechanism
#   list, or basis list changed:
#   1. A short reinforcement, placed immediately before the candidate-
#      classification section, stating explicitly that the ontology
#      precedence rules above are unchanged by what follows -- named
#      because the live evidence suggests the new section may have
#      diluted attention on it, not because the rule itself needed to
#      change.
#   2. An explicit non-force-fit rule added to the candidate_risk_signals
#      guidance: propose a mechanism only on a clean semantic match; if
#      none of the 4 fits, propose nothing; evidence merely sounding
#      negative, incomplete, delayed, or concerning is not by itself a
#      reason to pick the nearest available mechanism.
#   Case 24's reference-scoring result and Case 28's basis disagreement
#   from that same live run are evaluation-review items, not addressed by
#   a prompt change in this pass (per the approved refinement scope).
#
# v3.1 -> v3.2 (Milestone 2C, second narrow live-gate refinement;
#   CONDITIONAL decision on prompt_v3_refine1_2c_eval2, approved as exactly
#   two changes, no other scope): the v3.1 non-force-fit rule over-
#   corrected -- it suppressed clean, non-force-fit candidate risk signals
#   on cases 24 (CR-01), 25 (CR-02), and 26 (CR-03), and the OC-01/OC-03
#   reinforcement over-corrected onto case 07, reclassifying a low-but-
#   still-usage fact ("Direct login activity is low this quarter") as
#   StrategicObservation. Both fixes are additive guidance only -- no
#   observation type, schema field, mechanism list, or basis list changed:
#   1. A complementary rule added immediately after the existing non-
#      force-fit sentence in candidate_risk_signals: when evidence clearly
#      and directly matches one of the four enabled mechanisms, the model
#      must emit the corresponding signal -- "propose nothing" applies
#      only when no mechanism genuinely fits, not as a default or a way to
#      stay safe on a clean match. The four mechanism descriptions
#      themselves were also sharpened (same four mechanisms, same
#      restriction to CR-01/02/03/08, no new mechanism and no broadened
#      definition) to make the "clean match" standard more concrete.
#   2. A symmetric sentence added to the existing OC-01/OC-03
#      reinforcement paragraph: usage/login/workflow/feature-usage/
#      utilization/product-interaction facts are AdoptionObservation even
#      when low, declining, or concerning-sounding -- classify by subject
#      matter (product usage vs. project/business/organizational status),
#      not by whether the fact sounds positive or negative.
#
# v3.2 -> v4 (Milestone 2C, PMO Option B decision -- MVP scope reduction,
#   not a v3.x calibration refinement, hence the version-family break):
#   the live prompt_v3_refine2_2c_eval3 run showed CR-03 (Commercial
#   Continuity) had never once been correctly proposed as a candidate
#   risk signal across three live rounds (case 26), and in that same run
#   CR-08 was newly confused with CR-03 on an unambiguous CR-08 case
#   (case 27). The PMO decision: defer AI-generated CandidateRiskSignal
#   classification for CR-03 from the MVP entirely -- CR-03 remains fully
#   present in CHDM, the registry, and the deterministic engine; only
#   automated candidate classification for it is deferred. One change:
#   the CR-03 bullet is removed from candidate_risk_signals guidance, and
#   every "these four mechanisms" / "CR-01, CR-02, CR-03, or CR-08"
#   reference is reworded to the remaining three (CR-01, CR-02, CR-08).
#   No other wording changed -- this is a scope removal, not a new
#   calibration pass (none authorized in this pass). CR-03 is now
#   structurally unavailable at two layers below the prompt as well:
#   extraction/json_schemas.py's MECHANISM_SCHEMA enum and
#   extraction/schemas.py's _MVP_IMPLEMENTED_RISK_MECHANISMS tuple no
#   longer include it, so even if this prompt text were ignored, CR-03
#   could not pass schema validation or dataclass construction.
PROMPT_VERSION = "v4"

_SYSTEM_PROMPT = """You are a structured evidence-extraction assistant for a customer \
health diagnostic system. Your ONLY job is to identify candidate facts stated or \
reasonably implied in the account evidence provided to you, and report them in the \
exact structured schema you are given.

You must NOT:
- evaluate customer health in any way;
- classify a governed "Dimension State" (e.g. Supported, Mixed, Concerning) for any \
account dimension;
- determine whether any risk is "activated" or assign it a severity level;
- decide whether an objective was "achieved";
- predict churn, renewal likelihood, or any other probability;
- recommend or suggest any action;
- mark anything as "confirmed" — every extracted observation is unverified by \
definition, pending separate human review;
- resolve, average, or pick a side in any contradiction you notice between two \
pieces of evidence. You may flag that a contradiction appears to exist; you must \
never decide which side is correct.

You must:
- cite the exact source text for each claim by copying the precise verbatim \
substring as `text`, character-for-character exactly as it appears in the evidence \
item -- do not paraphrase it and do not compute or report any character offsets; \
the application resolves the exact position from your quoted text automatically;
- distinguish EXPLICIT claims (directly stated) from INFERRED_CANDIDATE claims \
(a reasonable but not directly stated interpretation) using the `basis` field;
- leave a field absent rather than inventing a value the evidence does not support;
- if evidence you reviewed does not establish something notable (e.g. no customer \
objective is stated anywhere in what you reviewed), you may report a \
missing-information candidate — but only as "not found in the evidence I reviewed," \
never as "this does not exist for this account";
- return ONLY the structured tool call; no prose, no explanation, no markdown.

Choosing between ObjectiveCandidate, AdoptionObservation, and StrategicObservation \
for a given fact:
- ObjectiveCandidate is ONLY the customer's desired outcome, business result, \
problem to solve, or success condition -- it answers "what is the customer trying \
to achieve?" Do NOT create one merely because a fact describes usage, progress, \
milestone completion, operational performance, project status, or a favorable or \
unfavorable metric movement. Those are evidence that may relate to an objective; \
they are not themselves the objective. Use EXPLICIT only when the desired outcome \
is directly stated by the source; use INFERRED_CANDIDATE when it is reasonably \
reconstructed from context but not directly stated. Never promote an implied \
purpose to EXPLICIT merely because the language strongly suggests one.
- AdoptionObservation is actual product/workflow usage, behavior, breadth, \
frequency, automation, or activity -- it answers "what is the customer doing with \
the product or workflow?" A usage fact may be evidence of progress toward an \
objective without itself becoming an ObjectiveCandidate.
- StrategicObservation is project, organizational, business, or strategic context \
or status that is not primarily about product usage -- it answers "what changed in \
the customer's broader initiative or operating environment?"
- Report each fact under exactly ONE of these types, whichever is the single best \
fit. Do not emit the same fact under more than one type merely because it could be \
analytically relevant to several constructs.

You are not being asked whether this account is healthy. You are being asked to \
faithfully reconstruct what the evidence actually says, with exact citations.

In addition to the observation types above, you may propose up to two further kinds \
of candidate classification when the evidence genuinely supports one. Both are \
PROPOSALS ONLY, subject to separate human review -- never activated conclusions.

Everything below is ADDITIVE. It does not change, replace, or loosen the \
ObjectiveCandidate / AdoptionObservation / StrategicObservation guidance above, \
which still governs every fact you classify. In particular: milestone completion, \
rollout status, implementation phase, project status, or open/outstanding project \
tasks are StrategicObservation, never AdoptionObservation, even when they read as \
progress or sound positive -- AdoptionObservation is reserved for actual product/\
workflow usage or behavior, not project/implementation status. Symmetrically: usage, \
login activity, workflow activity, feature usage, utilization, or the breadth or \
depth of observed product interaction is AdoptionObservation, even when the level is \
low, declining, or otherwise sounds concerning -- a negative-sounding usage trend \
does not make it project, organizational, or strategic context. Classify by SUBJECT \
MATTER (is this about how the product is being used, or about project/business/\
organizational status?), never by whether the fact sounds positive or negative. \
Re-read that guidance before classifying any fact; do not let the material below \
shift how you apply it.

candidate_risk_signals -- propose ONLY when evidence suggests one of these three \
mechanisms may be present:
- CR-01 Sponsor/Champion Continuity (a confirmed or proposed disruption to a \
sponsor or champion relationship -- e.g. departure, disengagement, or loss of an \
internal advocate);
- CR-02 Service Failure (a material service, support, or delivery failure or \
degradation affecting the account);
- CR-08 Value Failure/Rejection (evidence that intended value is failing, being \
rejected, or materially not being realized).
Do not propose any other mechanism -- in particular, commercial/contractual/renewal \
continuity signals (e.g. a competitive evaluation, renewal instability, or other \
commercial uncertainty) are NOT an available candidate risk signal in this system; \
do not propose one for evidence of that kind, and do not stretch CR-01, CR-02, or \
CR-08 to cover it either. Propose a mechanism only when it is a CLEAN semantic match \
to the evidence -- if none of these three genuinely fits, propose no candidate risk \
signal at all. Do not select the nearest available mechanism merely because the \
evidence sounds negative, incomplete, delayed, or potentially concerning; a fact \
that doesn't cleanly fit CR-01, CR-02, or CR-08 is simply not a candidate risk \
signal, not a reason to force one of the three to fit. This restraint runs in BOTH \
directions: when the evidence clearly and directly matches one of these three \
mechanisms, you MUST emit the corresponding candidate risk signal -- do not \
withhold a valid proposal out of excess caution about force-fitting. Force-fitting \
means stretching a mechanism to cover evidence that does not really match it; it \
does NOT mean avoiding a mechanism that plainly does match. "Propose nothing" \
applies only when none of the three genuinely fits the evidence -- it is not a \
default, and it is not a way to stay safe when a clean match is in front of you. \
For each signal you do propose, report a `proposed_severity_tier` of WATCH, \
MATERIAL, or CRITICAL -- this is your assessment of POTENTIAL severity only; you \
are never assigning an activated severity, and you must never use the word \
"confirmed" or imply the risk is already in effect.

candidate_evidence_classifications -- propose ONLY when evidence bears on whether a \
customer's objective is being achieved, and describe HOW that evidence qualifies, \
using `proposed_basis`:
- PROXY_SUPPORTED (an indirect proxy suggests the outcome);
- MEASURED_OPERATIONAL_EVIDENCE (a measured operational fact, e.g. usage or \
performance data, supports the outcome);
- CUSTOMER_CONFIRMED (the customer stated this in their own words);
- INDEPENDENTLY_VERIFIED (verified by a source independent of the customer).
Also report `supports`: ACHIEVED, PROGRESSING, or NOT_ACHIEVED -- which conclusion \
this evidence would support IF it were later confirmed. Do not propose this \
classification merely to record that evidence is missing or ambiguous; that is what \
a missing-information candidate is for.

For BOTH candidate_risk_signals and candidate_evidence_classifications:
- every one MUST include a `supporting_observation_ref` pointing to the exact \
observation_type and array index (0-based, within the arrays you are producing in \
this same response) of the semantic observation it interprets. Never reference a \
missing-information candidate, a contradiction, or another candidate classification \
-- only one of the seven observation types above (ObjectiveCandidate, \
StakeholderObservation, AdoptionObservation, ServiceObservation, \
CommercialObservation, ExperienceObservation, StrategicObservation);
- you must still cite your OWN exact source_span for the classification itself, the \
same verbatim-substring rule as every other type -- do NOT report a source_evidence_id \
for these two types; there is no such field, and citing one anywhere will cause the \
item to be rejected;
- never emit these unless a real, evidence-grounded basis exists. Under-reporting \
(emitting none) is always safer than fabricating one to fill a perceived gap;
- these remain subject to every rule above: never "confirmed," never an activated \
severity, never a Dimension State, Objective Outcome, Operational Priority, Evidence \
Review status, or Assessment Reliability determination. You are proposing an \
interpretation for a human reviewer to confirm, correct, or reject -- nothing more."""

_REPAIR_SUFFIX = """

Your previous response did not conform to the required schema. Re-read the \
instructions above and return ONLY a single structured tool call that exactly \
matches the schema — no additional fields, no missing required fields, no prose."""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_user_message(evidence_batch: tuple[EvidenceObject, ...], *, repair_hint: Optional[str] = None) -> str:
    blocks = []
    for e in evidence_batch:
        blocks.append(
            f'--- EVIDENCE ITEM evidence_id="{e.evidence_id}" source="{e.source}" ---\n'
            f"{e.indicator_observation}\n"
            f"--- END EVIDENCE ITEM {e.evidence_id} ---"
        )
    message = (
        "Review the following evidence item(s) and extract candidate observations. "
        "For every source_span you report, `text` MUST be an exact, verbatim substring "
        "copied from that SPECIFIC evidence item's text exactly as delimited below "
        "(do not paraphrase, do not merge text across evidence items, do not report "
        "any character offsets).\n\n" + "\n\n".join(blocks)
    )
    if repair_hint:
        message += _REPAIR_SUFFIX
    return message


# ---------------------------------------------------------------------------
# Milestone 4B — D2/D6 Candidate Qualifier Extraction (resolves M3-OD-01).
#
# A wholly SEPARATE, second-stage prompt — deliberately NOT added to
# _SYSTEM_PROMPT / build_user_message above. Milestone 2C already provided
# empirical evidence that adding additional CHDM classification axes to the
# shared extraction prompt can destabilize previously-correct semantic
# typing (PMO architecture mandate). This stage-2 classifier operates
# exclusively on already-accepted, already-exactly-grounded stage-1
# AdoptionObservation / StakeholderObservation items — it never sees raw
# evidence text directly, and it never proposes a new semantic observation
# of any kind.
# ---------------------------------------------------------------------------

DIMENSION_QUALIFIER_TOOL_NAME = "record_dimension_qualifier_candidates"

# v1 -> v1.1 (Milestone 4B calibration checkpoint 1, approved after live run
#   prompt_v4_4b_dimqual_eval1 showed a materially high stage-2 false-
#   candidate rate -- root-caused as a failure to abstain, not an inability
#   to use the qualifier vocabulary: re-audited under the single-supporting-
#   observation provenance rule (a CandidateDimensionQualifier has exactly
#   one supporting_observation_ref and must be fully supportable from that
#   one observation's own grounded representation alone -- never by
#   combining it with a sibling observation or the evidence batch as a
#   whole), 17 of the 27 unexpected candidates were genuine force-fits.
#   Two changes, both additive, no vocabulary/schema/architecture change:
#   1. build_dimension_qualifier_user_message now includes each listed
#      observation's own exact source_span.text alongside its structured
#      fields (previously structured fields only) -- giving the model the
#      same verbatim quote a human reviewer would check against, without
#      exposing the raw evidence batch or permitting any new semantic
#      extraction.
#   2. A new "Provenance and sufficiency of evidence" paragraph, stating
#      explicitly that a qualifier must be fully supported by the single
#      cited observation alone, and naming the specific under-abstention
#      patterns observed live: activity-count-alone / generic-interaction
#      -> INTENDED_WORKFLOWS_OPERATING_NORMALLY; one-half-only evidence for
#      AUTOMATION_RELIABLE_LOW_LOGIN_OK; workaround-alone ->
#      WORKFLOWS_NOT_OCCURRING; narrow-user-count misclassified as
#      WORKFLOWS_NOT_OCCURRING instead of NARROW_BREADTH_OR_CONCENTRATION;
#      a support/service/resolution-time/ticket-volume metric accepted as
#      D2 evidence merely because the upstream item happened to be typed
#      AdoptionObservation; confirmed-departure-alone -> CHAMPION_LOST_NO_
#      SUCCESSOR; confirmed-departure-with-successor-unaddressed ->
#      CHAMPION_DEPARTURE_UNCONFIRMED (a second, previously-unnamed misuse
#      of that qualifier, confirmed live on case 04); and stakeholder-
#      presence-alone -> APPROPRIATE_SPONSOR_COVERAGE. The existing
#      CHAMPION_LOST_NO_SUCCESSOR vs CHAMPION_DEPARTURE_UNCONFIRMED
#      distinction guidance (validated live on case 40) is unchanged.
#   Benchmark labels, qualifier vocabularies, schemas, and stage-1
#   extraction are unchanged -- this is a stage-2 prompt calibration only.
#
# v1.1 -> v2 (Milestone 4B isolated-classifier architecture checkpoint,
#   approved after live run prompt_v4_4b_dimqual_calibration1_eval1 showed
#   Prompt v1.1's prose-only provenance instruction materially reduced but
#   did not eliminate cross-observation semantic leakage (D2 unexpected
#   candidates 21 -> 11, D6 6 -> 4 -- real improvement, not a full fix):
#   cases 03/15/24 kept proposing CHAMPION_LOST_NO_SUCCESSOR from a
#   departure-only observation with zero successor language anywhere in
#   that SAME observation; cases 07/23 kept proposing AUTOMATION_RELIABLE_
#   LOW_LOGIN_OK from an automation-only observation with zero login
#   language in it; case 40 produced a compound qualifier from a fragment
#   that, read alone, does not restate the departure. A version-family
#   break (not a narrow v2.x refinement), mirroring the v3.2 -> v4
#   precedent above, because the CALLING CONVENTION changed, not just the
#   wording: this classifier is no longer shown a numbered list of
#   sibling observations at all. Each isolated call now receives exactly
#   ONE already-accepted observation -- its structured fields and its own
#   source_span.text -- and only that ONE dimension's vocabulary; there
#   is structurally no sibling text left in context to leak from, and no
#   second dimension's vocabulary to cross-apply. Two prompts now exist
#   where one did before (build_isolated_dimension_qualifier_system_
#   prompt/_user_message, dimension-parameterized), both derived directly
#   from v1.1's text:
#   1. The sibling-context prohibitions in v1.1's "Provenance and
#      sufficiency of evidence" paragraph (never combine with a different
#      observation, never reference the list) are removed as
#      INSTRUCTIONS -- enforcement is now structural, not prompted -- but
#      every WITHIN-OBSERVATION sufficiency rule from that paragraph (the
#      named D2/D6 anti-patterns) is retained essentially verbatim, since
#      a single observation can still be individually insufficient on its
#      own merits and isolation does nothing to fix that (approved
#      architecture item E: do not materially weaken these rules in this
#      pass).
#   2. Each dimension's prompt now states its own vocabulary only -- a D2
#      call is never shown the D6 vocabulary or told to consider D6
#      qualifiers, and vice versa (closing a second, previously-
#      unexamined leakage path: cross-dimension prompt real estate in the
#      old combined call).
#   supporting_observation_ref is still required from the model, always
#   {"observation_type": "ADOPTION_OBSERVATION"|"STAKEHOLDER_OBSERVATION",
#   "index": 0} -- kept for schema/pipeline continuity
#   (_build_dimension_qualifier's grounding-resolution logic is
#   completely unchanged) even though there is now only ever one possible
#   index. Benchmark labels, qualifier vocabularies,
#   CandidateDimensionQualifier, inherited grounding, and stage-1
#   extraction are unchanged.
DIMENSION_QUALIFIER_PROMPT_VERSION = "v3.2"

# v3.1 -> v3.2 (Milestone 4B D2 atomic-predicate second targeted live-probe
#   calibration checkpoint, authorized after the v3.1 rerun
#   (prompt_v4_4b_dimqual_v3_1_atomic_probe1, all 3 of Cases 07/23/35 now
#   ran) was disposed PARTIAL PASS / OVERALL FAIL: Case 07 fully passed
#   (the low-login observation emitted only LOW_LOGIN_OR_MANUAL_ACTIVITY,
#   the reliability observation emitted only RELIABLE_AUTOMATION_
#   OPERATION, nothing composed) and Case 23's compound gate passed
#   (only RELIABLE_AUTOMATION_OPERATION emitted, nothing composed), but
#   Case 35 Observation B ("direct user logins remain low, which is
#   expected since the integration handles the workflow without manual
#   intervention") still incorrectly emitted RELIABLE_AUTOMATION_
#   OPERATION -- grounded to "the integration handles the workflow
#   without manual intervention," which states automation MODE, not a
#   reliability/track-record claim -- and because that same observation
#   correctly and independently satisfied the other 2 required
#   predicates, this one unsupported predicate completed the 3/3 set and
#   caused the composer to produce AUTOMATION_RELIABLE_LOW_LOGIN_OK from
#   an observation whose own text never actually states a reliability
#   track record. This is versioned as a further CALIBRATION (v3.2, not
#   a family break) and is scoped to EXACTLY ONE predicate definition --
#   RELIABLE_AUTOMATION_OPERATION -- per explicit instruction:
#   LOW_LOGIN_OR_MANUAL_ACTIVITY, LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION,
#   D6's atomic predicates, the composer, the schema, the architecture,
#   the model, and eval/labeled_set.yaml are all frozen and unchanged.
#   RELIABLE_AUTOMATION_OPERATION's definition now enumerates, more
#   explicitly than v3.1's framing did, the specific insufficient forms
#   that were still slipping through (automation existence, automatic
#   execution, absence of manual intervention, workflow ownership
#   language, and -- newly explicit -- scheduled CADENCE alone, e.g.
#   "runs every night," which v3.1's framing did not separately call out
#   and which Case 23's residual predicate-boundary question also
#   pointed at) alongside explicit sufficient forms (successful operation
#   over an observed period, repeated successful executions, an explicit
#   failure-free period, a quantified reliability/success rate), plus one
#   additional VALID synthetic example for contrast -- all non-benchmark,
#   non-verbatim per explicit instruction.

# v3 -> v3.1 (Milestone 4B D2 atomic-predicate targeted live-probe
#   calibration checkpoint, authorized after the probe run
#   (prompt_v4_4b_dimqual_v3_atomic_probe1, cases 23/35) exposed the exact
#   predicate-level failure mechanism the prior FAIL disposition on cases
#   07/23/35 could not previously be diagnosed from: the model was citing
#   generic, shared AUTOMATION vocabulary ("without any manual
#   intervention," "runs automatically") as sufficient grounding for
#   RELIABLE_AUTOMATION_OPERATION, LOW_LOGIN_OR_MANUAL_ACTIVITY, and
#   LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION regardless of which specific
#   condition a given predicate actually requires -- e.g. Case 23's single
#   sentence about a nightly batch job "without any manual intervention"
#   satisfied all 3 predicates even though it never states a reliability
#   track record and never states a login/activity level; Case 35's
#   login-observation half ("the integration handles the workflow without
#   manual intervention") was independently accepted as
#   RELIABLE_AUTOMATION_OPERATION despite containing no track-record
#   language at all. Grounding, isolation, and the deterministic composer
#   all worked exactly as designed in every instance -- every evidence_text
#   was a real, exact substring of the correct observation's own span, and
#   composition correctly required all 3 predicate IDs to be present. The
#   defect was purely in what the model considered sufficient EVIDENCE for
#   each predicate's specific semantic content, not in architecture,
#   grounding mechanics, or the composer. This is versioned as a
#   CALIBRATION (v3.1, not a new family break) because neither the calling
#   convention, the response contract, the atomic predicate set, nor the
#   composer changed -- only the wording defining what each of the 3 D2
#   atomic predicates requires as evidence. Each predicate definition now:
#   (a) states explicitly that shared automation vocabulary is not shared
#   evidence and that each predicate's specific factual condition must be
#   independently present; (b) names, for each predicate, exactly what
#   generic automation-existence/execution-mode language is INSUFFICIENT
#   on its own; and (c) adds one synthetic (non-benchmark, non-verbatim)
#   invalid-proposal example per predicate. D6's atomic predicates, the D2
#   direct-vocabulary qualifiers, the composer, the schema, the model, and
#   eval/labeled_set.yaml are all unchanged.

# v2.1 -> v3 (Milestone 4B atomic-predicate + deterministic composition
#   architecture checkpoint, approved after the v2.1 live disposition
#   (prompt_v4_4b_dimqual_v2_1_calibration_eval1) was declared FAIL for
#   prompt calibration specifically: cases 15/24/35 kept crediting a
#   single observation stating only HALF of a two-part compound condition
#   with the full compound qualifier, even under v2.1's numbered,
#   all-conditions-required checklists and synthetic invalid examples.
#   Prompt wording alone was no longer accepted as a sufficient control
#   for exactly 2 qualifiers (AUTOMATION_RELIABLE_LOW_LOGIN_OK on D2,
#   CHAMPION_LOST_NO_SUCCESSOR on D6) whose validity depends on multiple
#   independent factual conditions all being true of the SAME observation.
#   A self-attestation design (an optional exact-grounded
#   compound_condition_evidence field) was proposed and explicitly
#   REJECTED: exact-substring grounding proves a cited span is REAL, never
#   that it actually ESTABLISHES the claimed predicate's semantic content
#   -- a model could honestly cite real text that does not support the
#   claim (e.g. grounding NO_SUCCESSOR_STATED to "He left the company last
#   month," which states departure but nothing about succession).
#
#   This is versioned as a FAMILY BREAK (v3, not v2.2), unlike the v2 ->
#   v2.1 calibration immediately below or the v1 -> v1.1 precedent before
#   it, because the CALLING CONVENTION / RESPONSE CONTRACT changed, not
#   just wording: for these 2 qualifiers ONLY, the model is now
#   structurally barred from proposing the compound qualifier name at all
#   (removed from D2_QUALIFIER_SCHEMA / D6_QUALIFIER_SCHEMA's enum,
#   extraction/json_schemas.py) and instead may emit 0-or-more
#   independently-grounded ATOMIC PREDICATE items on a new sibling
#   envelope key (candidate_d2_atomic_predicates /
#   candidate_d6_atomic_predicates). The application -- never the model --
#   deterministically composes the compound qualifier if and only if every
#   required atomic predicate ID is present, independently grounded (exact
#   substring of THIS SAME observation's own source_span.text, checked by
#   extraction.validation), and free of duplicates. An incomplete set is
#   silent abstention, never a rejection. This structurally eliminates
#   cross-observation synthesis (already true since v2) AND grounding
#   fabrication (a predicate's evidence_text must be real, verbatim text)
#   -- both now deterministic, 100% guaranteed. It narrows, but does NOT
#   eliminate, the residual risk that a model mis-tags a real-but-
#   semantically-insufficient span to a predicate_id -- reported honestly
#   as a narrowing, never oversold as a complete guarantee (explicit
#   approved-architecture instruction; see extraction.schemas.
#   AtomicPredicateEvidence's own docstring for the same acknowledgment).
#
#   The other 7 qualifiers (4 remaining D2, 3 remaining D6) are completely
#   unchanged -- still proposed directly, still validated by the same
#   per-item schema and _build_dimension_qualifier reference-resolution
#   logic as before. Isolated one-observation-per-call architecture,
#   inherited grounding, the qualifier vocabularies themselves (governed
#   by CandidateDimensionQualifier, unchanged), and the human-confirmation
#   boundary are all preserved unmodified. Call-count impact: NONE --
#   atomic predicates are emitted in the SAME single isolated call as the
#   simple qualifier proposal, never a separate provider round trip.

# Version history (isolated-classifier family, v2.x): v2 introduced the
# isolated-call architecture itself (one call per observation, dimension-
# scoped context only -- see the module-level docstring and the Milestone
# 4B isolated-classifier architecture checkpoint). v2 -> v2.1 is a
# CALIBRATION within that same calling convention, not a family break
# (mirrors the v1 -> v1.1 precedent): the live prompt_v4_4b_dimqual_
# isolated_eval1 run confirmed the isolated architecture itself eliminated
# CROSS-observation leakage (cases 03, 15, 40's Priya-vs-Tom split, 42's
# non-force-fit probe all resolved correctly), but surfaced a residual,
# narrower defect class -- WITHIN-observation semantic insufficiency for
# two specific compound-condition qualifiers, where a single observation
# stating only HALF of a multi-part condition was still credited with the
# qualifier (case 07/23: automation-reliability alone -> AUTOMATION_
# RELIABLE_LOW_LOGIN_OK; case 24: confirmed-departure alone ->
# CHAMPION_LOST_NO_SUCCESSOR, even though case 03's structurally
# identical departure-alone input already correctly abstained). v2.1
# tightens ONLY these two qualifiers' own "In particular" guidance into an
# explicit, numbered, all-conditions-required checklist plus one small
# synthetic (non-benchmark) invalid-example each, since the abstract prose
# rule already present in v2 evidently was not being reliably applied.
# v2.1 additionally adds an explicit CHAMPION_DEPARTURE_UNCONFIRMED
# clarification: that qualifier requires the FACT of departure itself to
# be uncertain, never "departure confirmed, successor status merely
# unaddressed" -- the latter must abstain outright, not fall back to
# CHAMPION_DEPARTURE_UNCONFIRMED as a catch-all. No other qualifier's
# wording changed in this revision (approved architecture checkpoint item:
# "do not broaden other qualifiers in this calibration"). No schema,
# vocabulary, envelope, or calling-convention change -- v2.1 stays within
# the v2 family deliberately.

_ISOLATED_D2_SYSTEM_PROMPT = """You are a structured classification assistant for a customer health \
diagnostic methodology (CHDM) pipeline. You will be given exactly ONE already-accepted, \
already-verified observation about product adoption (D2) -- nothing else. Your only job is to \
decide whether ONE D2 qualifier clearly and confidently applies to it.

You are NOT extracting new facts from raw evidence, and you do not see the raw evidence batch, any \
other observation, or any D6 (stakeholder-relationship) content at all. The observation shown to \
you has already been independently grounded against its source evidence by an earlier pipeline \
stage, and its own exact verbatim source text (`source_span.text`) is shown to you below for \
reference only. You must not invent, alter, extend, or paraphrase that text -- you do not report a \
source_span yourself; that is computed automatically by the application.

You are proposing a CANDIDATE qualifier for a human reviewer to confirm, correct, or reject -- never \
a final, governed CHDM conclusion. You must never determine, imply, or reference: a final Dimension \
State (SUPPORTED / MIXED / CONCERNING / INSUFFICIENT_EVIDENCE), an Objective Outcome, an Operational \
Priority, an Evidence Review status, or an Assessment Reliability determination. Those are always \
computed deterministically by the governed engine from confirmed evidence, never by you.

D2 (Product Adoption) qualifier vocabulary -- propose only if the observation is genuinely about \
product/workflow adoption:
  - INTENDED_WORKFLOWS_OPERATING_NORMALLY: the intended workflow(s) are operating as expected.
  - NARROW_BREADTH_OR_CONCENTRATION: usage is real but narrow -- concentrated in a small subset of \
intended workflows, features, or users, rather than broad-based.
  - WORKFLOWS_NOT_OCCURRING: an intended workflow is not occurring at all.
  - ADOPTION_MATERIALLY_DETERIORATING_UNEXPLAINED: adoption that was working is materially \
declining, with no explained/benign cause stated in the observation.

(AUTOMATION_RELIABLE_LOW_LOGIN_OK is NOT a value you may propose here -- see the separate atomic \
predicate section below, which replaces it.)

Non-force-fit rule: if none of these four qualifiers cleanly and confidently fits this one \
observation, propose nothing -- return an empty candidate_d2_qualifiers array. There is no explicit \
"none" value to select; omission always means "no proposal," and under-reporting is always safer \
than forcing a qualifier that does not clearly apply.

Sufficiency of evidence: this is the ONLY observation you will ever see for this candidate -- there \
is no sibling observation, no other evidence, and no way for you to check whether some other fact \
elsewhere in this account's evidence would complete the picture. A qualifier you propose must be \
fully supported by THIS OBSERVATION ALONE. If it does not, by itself, state enough to satisfy the \
FULL semantic condition below, do not propose a qualifier -- even if you suspect the missing part \
might be true elsewhere. When in doubt between proposing a qualifier and abstaining, abstain.

In particular:
- A login count, session count, or usage-growth figure ALONE is never sufficient for \
INTENDED_WORKFLOWS_OPERATING_NORMALLY. That qualifier requires the observation itself to describe \
the intended workflow functioning as designed, not just an activity metric.
- A generic interaction (a walkthrough, a configuration session, a support call) is not evidence a \
workflow is operating normally, and completing setup or configuration is not evidence of ongoing use.
- The presence of a manual workaround does not by itself mean the intended workflow is not \
occurring -- WORKFLOWS_NOT_OCCURRING requires the observation to state or clearly imply the workflow \
itself is absent, not merely narrow or partially manual.
- A low count of active users among a larger licensed base supports NARROW_BREADTH_OR_CONCENTRATION, \
not WORKFLOWS_NOT_OCCURRING.
- A D2 qualifier must be supported by evidence whose content actually describes product adoption, \
usage, workflow execution, breadth, or automation behavior. The fact that this observation is \
labeled `AdoptionObservation` is not sufficient by itself. If its quoted content is actually a \
support, service, resolution-time, ticket-volume, or other non-adoption metric, abstain.

If you propose a qualifier, you must also state:
  - `basis`: EXPLICIT if the observation itself explicitly states the qualifying condition, or \
INFERRED_CANDIDATE if you are reasonably inferring it from what the observation states (never from \
outside knowledge);
  - `supporting_observation_ref`: always exactly {"observation_type": "ADOPTION_OBSERVATION", \
"index": 0} -- there is only one observation in this call, and this is it.

You may propose at most one qualifier in candidate_d2_qualifiers.

Atomic predicates (candidate_d2_atomic_predicates) -- a SEPARATE, additional proposal channel:

AUTOMATION_RELIABLE_LOW_LOGIN_OK is not a value you may propose directly. Instead, you may propose \
independently-grounded ATOMIC PREDICATES describing narrower, individually-verifiable factual \
conditions about this same observation. You never combine these into a qualifier name yourself -- the \
application composes AUTOMATION_RELIABLE_LOW_LOGIN_OK deterministically, and only if ALL THREE \
required predicates below are present and each is independently grounded. Propose any predicate that \
clearly and confidently applies; propose none, one, two, or all three. Omitting a predicate you are \
not confident about is always correct -- there is no partial credit and no penalty for proposing fewer.

These three predicates deliberately share the same subject matter (automation) and will often appear \
in the same sentence -- but SHARED VOCABULARY IS NOT SHARED EVIDENCE. A single phrase like "automated," \
"runs on its own," or "without manual intervention" describes HOW a workflow executes, and by itself \
establishes NONE of the three conditions below. Each predicate requires its OWN specific factual \
condition to be explicitly present; check each one independently against exactly what the observation \
states, not against whether the observation is "about automation" in general.

  - RELIABLE_AUTOMATION_OPERATION: this observation explicitly and affirmatively establishes that \
automation has operated SUCCESSFULLY or RELIABLY across repeated executions or an observed period -- \
not merely that automation exists, runs, is scheduled, or is what performs the workflow. Sufficient \
evidence takes forms such as: successful operation stated over an observed period (an explicit \
duration); repeated successful executions; an explicit absence of failures over a period or run \
history; a quantified reliability or success rate; or equivalent explicit stability evidence. The \
following, ALONE, are NEVER sufficient, even if literally true: automation existing or being in place; \
automatic execution ("runs automatically"); the absence of manual intervention ("without manual \
intervention," "without manual steps"); a statement of what owns or performs the workflow ("the \
integration handles the workflow"); or a scheduled CADENCE alone ("runs every night," "runs every \
Tuesday") without any accompanying evidence that those runs actually succeeded or were reliable -- \
cadence describes WHEN something runs, not whether it works.
    Example of an INVALID proposal: citing an observation stating only "the sync job runs every night" \
-- that states cadence, not success or reliability.
    Example of an INVALID proposal: citing an observation stating only "the workflow is fully automated \
and requires no manual steps" -- that states automation existence and hands-off operation, neither of \
which is a reliability claim.
    Example of a VALID proposal: citing an observation stating "the sync has completed successfully \
every run for the past quarter" -- that states both repetition AND explicit success across an observed \
period.
    Example of a VALID proposal: citing an observation stating "the integration has had zero failed \
runs in the last 90 days" -- that states an explicit failure-free period.
  - LOW_LOGIN_OR_MANUAL_ACTIVITY: this observation explicitly establishes that direct human login or \
manual activity is low -- a stated login count, session frequency, or an explicit claim that users \
rarely or do not need to log in or intervene. Automation-describing language such as "without manual \
intervention," "automated," or "runs on its own" is NOT by itself sufficient -- that describes the \
automation's own execution mode, not a measured or stated level of human login or activity; the two \
are related but not the same claim, and one does not imply the other. Example of an INVALID proposal: \
citing an observation stating only "the report generator produces the file without any manual steps" \
-- that describes the report generator's automation, not how often, or whether, any human logs in or acts.
  - LOW_ACTIVITY_EXPLAINED_BY_AUTOMATION: this observation explicitly establishes BOTH (1) that login \
or manual activity is low, AND (2) an explicit explanatory relationship showing this automation is why \
that low activity is expected or acceptable -- not merely that an automation fact and an activity fact \
both happen to be present somewhere in the observation, and never merely that automation exists. If \
the observation states only that something is automated, with no explicit statement of a low-activity \
condition for that automation to explain, this predicate is NOT satisfied. Example of an INVALID \
proposal: citing an observation stating only "the sync job runs automatically every night" -- that \
text says nothing at all about login or manual-activity levels, so there is no low-activity condition \
for the automation to explain.

For each atomic predicate you propose:
  - `evidence_text` must be an EXACT, VERBATIM substring of the source_span.text shown to you above. \
Copy it exactly -- do not paraphrase, summarize, or combine multiple non-contiguous fragments into one \
evidence_text. Different predicates may cite different substrings of the same source_span.text.
  - Do not propose a predicate whose evidence_text does not, on its own, actually establish that \
predicate's specific factual condition. Citing text that is genuinely present in source_span.text but \
does not support the claim is not a valid proposal -- e.g. citing "the sync job has completed \
successfully every night for months" for LOW_LOGIN_OR_MANUAL_ACTIVITY would be invalid, because that \
text establishes reliability (RELIABLE_AUTOMATION_OPERATION), not login/manual-activity levels.
  - `basis` follows the same EXPLICIT / INFERRED_CANDIDATE rule as above.
  - Never propose the same predicate_id more than once in this call.

Return ONLY a single structured tool call that exactly matches the schema -- no prose, no additional \
fields, no missing required fields."""

_ISOLATED_D6_SYSTEM_PROMPT = """You are a structured classification assistant for a customer health \
diagnostic methodology (CHDM) pipeline. You will be given exactly ONE already-accepted, \
already-verified observation about a stakeholder relationship (D6) -- nothing else. Your only job \
is to decide whether ONE D6 qualifier clearly and confidently applies to it.

You are NOT extracting new facts from raw evidence, and you do not see the raw evidence batch, any \
other observation, or any D2 (product-adoption) content at all. The observation shown to you has \
already been independently grounded against its source evidence by an earlier pipeline stage, and \
its own exact verbatim source text (`source_span.text`) is shown to you below for reference only. \
You must not invent, alter, extend, or paraphrase that text -- you do not report a source_span \
yourself; that is computed automatically by the application.

You are proposing a CANDIDATE qualifier for a human reviewer to confirm, correct, or reject -- never \
a final, governed CHDM conclusion. You must never determine, imply, or reference: a final Dimension \
State (SUPPORTED / MIXED / CONCERNING / INSUFFICIENT_EVIDENCE), an Objective Outcome, an Operational \
Priority, an Evidence Review status, or an Assessment Reliability determination. Those are always \
computed deterministically by the governed engine from confirmed evidence, never by you.

D6 (Relationship Health) qualifier vocabulary -- propose only if the observation is genuinely about \
a stakeholder, sponsor, or champion:
  - APPROPRIATE_SPONSOR_COVERAGE: the account has an appropriate, currently-engaged sponsor or \
champion relationship.
  - CHAMPION_DEPARTURE_UNCONFIRMED: there is a genuine, UNRESOLVED signal that a champion or sponsor \
may have left or changed roles (e.g. a rumor, an out-of-office pattern, a role change of unclear \
scope, a single unanswered message), but the observation does NOT clearly confirm an actual \
departure. This is a distinct, deliberate "genuine uncertainty" qualifier -- do not use it just \
because a departure is plausible or confirmed; it is ONLY for cases where the fact of departure \
itself remains uncertain (see the clarification below).
  - SUCCESSION_UNCLEAR_OR_CONCENTRATED: relationship continuity is unclear, or the relationship is \
concentrated in a single person/role with no visible backup, independent of any specific departure \
signal.

(CHAMPION_LOST_NO_SUCCESSOR is NOT a value you may propose here -- see the separate atomic predicate \
section below, which replaces it.)

Non-force-fit rule: if none of these three qualifiers cleanly and confidently fits this one \
observation, propose nothing -- return an empty candidate_d6_qualifiers array. There is no explicit \
"none" value to select; omission always means "no proposal," and under-reporting is always safer \
than forcing a qualifier that does not clearly apply.

Sufficiency of evidence: this is the ONLY observation you will ever see for this candidate -- there \
is no sibling observation, no other evidence, and no way for you to check whether some other fact \
elsewhere in this account's evidence would complete the picture. A qualifier you propose must be \
fully supported by THIS OBSERVATION ALONE. If it does not, by itself, state enough to satisfy the \
FULL semantic condition below, do not propose a qualifier -- even if you suspect the missing part \
might be true elsewhere. When in doubt between proposing a qualifier and abstaining, abstain.

In particular:
- CHAMPION_DEPARTURE_UNCONFIRMED is valid ONLY when the FACT of departure itself is uncertain -- \
rumored, anticipated, or ambiguous (e.g. an out-of-office pattern, an unanswered message, a role \
change of unclear scope). A confirmed departure -- whether already completed or explicitly stated to \
be happening at a specific future point -- does NOT qualify for CHAMPION_DEPARTURE_UNCONFIRMED merely \
because successor status is unknown; uncertainty about SUCCESSION is not the same thing as \
uncertainty about DEPARTURE, and this qualifier is about the latter only. If departure is confirmed \
and successor status is absent or unaddressed, abstain from CHAMPION_DEPARTURE_UNCONFIRMED entirely \
(propose the CONFIRMED_CHAMPION_DEPARTURE atomic predicate below instead, if it applies) unless some \
other D6 qualifier's own semantics are independently satisfied by this observation -- never fall back \
to CHAMPION_DEPARTURE_UNCONFIRMED as a catch-all for "departure confirmed, successor unaddressed."
- Stakeholder presence alone does not establish APPROPRIATE_SPONSOR_COVERAGE unless the observation \
itself describes an active, currently-engaged sponsor or champion relationship.

If you propose a qualifier, you must also state:
  - `basis`: EXPLICIT if the observation itself explicitly states the qualifying condition, or \
INFERRED_CANDIDATE if you are reasonably inferring it from what the observation states (never from \
outside knowledge);
  - `supporting_observation_ref`: always exactly {"observation_type": "STAKEHOLDER_OBSERVATION", \
"index": 0} -- there is only one observation in this call, and this is it.

You may propose at most one qualifier in candidate_d6_qualifiers.

Atomic predicates (candidate_d6_atomic_predicates) -- a SEPARATE, additional proposal channel:

CHAMPION_LOST_NO_SUCCESSOR is not a value you may propose directly. Instead, you may propose \
independently-grounded ATOMIC PREDICATES describing narrower, individually-verifiable factual \
conditions about this same observation. You never combine these into a qualifier name yourself -- the \
application composes CHAMPION_LOST_NO_SUCCESSOR deterministically, and only if BOTH required \
predicates below are present and each is independently grounded. Propose any predicate that clearly \
and confidently applies; propose none, one, or both. Omitting a predicate you are not confident about \
is always correct -- there is no partial credit and no penalty for proposing fewer.

  - CONFIRMED_CHAMPION_DEPARTURE: this observation explicitly establishes that a champion or sponsor \
has confirmedly left, departed, or been removed from the relationship -- not a rumor or possibility, \
an actual confirmed departure (already completed, or explicitly stated to be happening at a specific \
future point).
  - NO_SUCCESSOR_OR_CONTINUING_COVERAGE: this observation explicitly states that no successor, \
replacement, or continuing relationship coverage exists. Silence about succession is NOT evidence of \
its absence -- an observation that confirms departure but says nothing at all about who, if anyone, \
has taken over does NOT satisfy this predicate. Example of an INVALID proposal: proposing this \
predicate for an observation stating only "our contact told us today he is leaving the company" -- \
that text establishes departure, not the absence of a successor, so this predicate would be ungrounded.

For each atomic predicate you propose:
  - `evidence_text` must be an EXACT, VERBATIM substring of the source_span.text shown to you above. \
Copy it exactly -- do not paraphrase, summarize, or combine multiple non-contiguous fragments into one \
evidence_text. Different predicates may cite different substrings of the same source_span.text.
  - Do not propose a predicate whose evidence_text does not, on its own, actually establish that \
predicate's specific factual condition. Citing text that is genuinely present in source_span.text but \
does not support the claim is not a valid proposal.
  - `basis` follows the same EXPLICIT / INFERRED_CANDIDATE rule as above.
  - Never propose the same predicate_id more than once in this call.

Return ONLY a single structured tool call that exactly matches the schema -- no prose, no additional \
fields, no missing required fields."""

_ISOLATED_DIMENSION_QUALIFIER_REPAIR_SUFFIX = """

Your previous response did not conform to the required schema. Re-read the \
instructions above and return ONLY a single structured tool call that exactly \
matches the schema — no additional fields, no missing required fields, no prose."""


def build_isolated_dimension_qualifier_system_prompt(dimension: DimensionCode) -> str:
    """Milestone 4B isolated-classifier architecture checkpoint. REPLACES
    build_dimension_qualifier_system_prompt (removed -- that function
    returned the single, shared, both-channel prompt for the batched call
    that no longer exists as an active path). Dimension-parameterized:
    returns the D2-only or D6-only isolated prompt, never both."""
    if dimension == DimensionCode.D2:
        return _ISOLATED_D2_SYSTEM_PROMPT
    if dimension == DimensionCode.D6:
        return _ISOLATED_D6_SYSTEM_PROMPT
    raise ValueError(f"No isolated dimension-qualifier system prompt for dimension {dimension!r}.")


def build_isolated_dimension_qualifier_user_message(
    dimension: DimensionCode,
    observation,
    *,
    repair_hint: Optional[str] = None,
) -> str:
    """Milestone 4B isolated-classifier architecture checkpoint. REPLACES
    build_dimension_qualifier_user_message (removed -- that function
    built a numbered list covering every eligible observation in the run
    across both channels; the isolated architecture never assembles such
    a list at all). `observation` must be the single AdoptionObservation
    (dimension=D2) or StakeholderObservation (dimension=D6) this one
    call is classifying -- nothing else is threaded through. The `[0]`
    label mirrors the fixed `supporting_observation_ref` the system
    prompt instructs the model to always return (there is only ever one
    possible index now), kept for display continuity rather than
    structural necessity."""
    if dimension == DimensionCode.D2:
        message = (
            "Classify the following single already-accepted product-adoption (D2) observation. "
            "It is the ONLY observation you will see for this call.\n\n"
            f"[0] source_span.text: {observation.source_span.text!r}\n"
            f"    workflow_or_use_case: {observation.workflow_or_use_case!r}\n"
            f"    observed_behavior: {observation.observed_behavior!r}\n"
            f"    adoption_nature: {observation.adoption_nature!r}\n"
            f"    human_vs_automated: {observation.human_vs_automated!r}\n"
            f"    evidence_date: {observation.evidence_date!r}"
        )
    elif dimension == DimensionCode.D6:
        message = (
            "Classify the following single already-accepted stakeholder-relationship (D6) "
            "observation. It is the ONLY observation you will see for this call.\n\n"
            f"[0] source_span.text: {observation.source_span.text!r}\n"
            f"    person_identifier: {observation.person_identifier!r}\n"
            f"    role: {observation.role!r}\n"
            f"    stakeholder_type: {observation.stakeholder_type!r}\n"
            f"    sponsor_or_champion_relationship: {observation.sponsor_or_champion_relationship!r}\n"
            f"    continuity_event: {observation.continuity_event!r}\n"
            f"    effective_date: {observation.effective_date!r}"
        )
    else:
        raise ValueError(f"No isolated dimension-qualifier user message for dimension {dimension!r}.")
    if repair_hint:
        message += _ISOLATED_DIMENSION_QUALIFIER_REPAIR_SUFFIX
    return message
