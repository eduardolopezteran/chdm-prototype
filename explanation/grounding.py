"""
I5 build block -- grounding-package assembly and deterministic diagnostic-
question subject selection/ranking.

Pure functions only. No LLM participation anywhere in this module -- exact
same discipline as engine/evaluate.py's own module docstring ("No LLM
participates anywhere in this module"). This is the module responsible for
enforcing, structurally, the BUILD AUTHORIZED decision that "deterministic
logic must select and rank the diagnostic-question subjects before the
model call. The LLM may phrase those selected questions but may not choose
which gaps matter." The provider/pipeline layer never sees a DMEG or an
unresolved dimension directly -- it only ever receives the already-ranked,
already-capped GapSubject tuple this module produces.

Two distinct grounding rules (BUILD AUTHORIZED decision), enforced here by
construction:

  1. Explanation factual claims: confirmed_evidence_refs below is built
     ONLY from ActiveEvidenceItem entries whose evidence_state is
     CURRENT_CONFIRMED (engine.evidence_engine.is_current_confirmed),
     mirroring domain/*.py's own invariants that every governed output's
     contributing_evidence_refs already only cites confirmed evidence.
     Unverified evidence never enters GroundingPackage at all.

  2. Diagnostic questions: select_gap_subjects() below builds GapSubject
     entries from DMEG and DimensionState objects directly -- both already
     governed, ID-bearing outputs -- never from raw evidence content. A
     GapSubject never carries an EvidenceObject, an ActiveEvidenceItem, or
     any extraction.schemas dataclass, confirmed or not.
"""
from __future__ import annotations

from domain.dimension_state import DimensionState
from domain.dmeg import DMEG
from domain.enums import (
    DimensionStateValue, DMEGLinkedConclusion, EvidenceReviewStatus, EvidenceState, OperationalPriority,
)
from domain.reason_code import ReasonCode
from engine.evaluate import EvaluationResult

from confirmation.schemas import RecomputeDiagnostic

from .enums import GapKind
from .schemas import GapSubject, GroundingPackage

import re

_CITATION_PATTERN = re.compile(r"\s*\(CHDM[^)]*\)\.?\s*$")

def _public_text(text: str) -> str:
    """Strip internal spec citations (e.g. '(CHDM v0.1 §4.2)') before this
    text is handed to the model. Mirrors ui/diagnostic_panel.py's own
    _public_text() -- duplicated, not imported, since explanation/ must
    not depend on ui/."""
    return _CITATION_PATTERN.sub("", text).rstrip()

# Plain-language labels mirrored from ui/labels.py's OPERATIONAL_PRIORITY_LABEL/
# EVIDENCE_REVIEW_LABEL and diagnostic_panel.py's own dimension label map.
# Duplicated deliberately -- explanation/ cannot import ui/ (see ui/labels.py's
# own docstring: "one direction only... never the reverse"). Keep in sync by
# hand if that wording ever changes.
_DIMENSION_PUBLIC_LABEL = {"D1": "Objective Outcome", "D2": "Product Adoption", "D6": "Relationship Health"}
_OPERATIONAL_PRIORITY_PUBLIC_LABEL = {
    "OP1": "Urgent Review", "OP2": "Review Required", "OP3": "Routine Monitoring", "OPU": "Undetermined",
}
_EVIDENCE_REVIEW_PUBLIC_LABEL = {"ER1": "Evidence Review Required", "ER0": "Evidence Review Not Required"}

def _public_subject_label(subject_construct_ref: str) -> str:
    """Plain-language label for a subject_construct_ref used in reviewer-
    facing stake descriptions. Falls back to a generic phrase for anything
    that isn't a known dimension code (e.g. a risk mechanism code like
    "CR-01") -- this module doesn't own vetted plain wording for those, so
    it never invents any, mirroring ui/labels.py's own fallback philosophy."""
    if subject_construct_ref in _DIMENSION_PUBLIC_LABEL:
        return _DIMENSION_PUBLIC_LABEL[subject_construct_ref]
    if subject_construct_ref == "OBJECTIVE":
        return "the account's objective"
    return "this area of the assessment"
  
_MAX_GAP_SUBJECTS = 5


def _reason_code_dict(reason_code: ReasonCode) -> dict:
    return {
        "code": reason_code.code,
        "governing_object_id": reason_code.governing_object_id,
        "human_readable_text": _public_text(reason_code.human_readable_text),
    }


def _stake_rank_for_dmeg(dmeg: DMEG) -> int:
    """FR-18.2 priority order, 1 = highest stake:
      1. Could change Operational Priority
      2. Could activate or alter a Material/Critical severity
      3. Could change a material dimension conclusion
      4. Could resolve a contradiction
      5. Could improve Assessment Reliability without changing a conclusion
    A DMEG's rank is the BEST (lowest-numbered) priority any of its
    dmeg2_linked_conclusions or subject_construct_ref actually earns --
    never inferred from DMEG existence alone, mirroring
    engine/evaluate.py's own "never assumed... always genuine differential
    re-computation" discipline (this module only reads what evaluate()
    already decided; it invents no new differential test)."""
    linked = dmeg.dmeg2_linked_conclusions
    if DMEGLinkedConclusion.OPERATIONAL_PRIORITY in linked:
        return 1
    if DMEGLinkedConclusion.RISK_SEVERITY in linked:
        return 2
    if DMEGLinkedConclusion.DIMENSION_STATE in linked or DMEGLinkedConclusion.OBJECTIVE_D1 in linked:
        return 3
    if "CONTRADICTION" in dmeg.subject_construct_ref.upper():
        return 4
    return 5


def _stake_description_for_dmeg(dmeg: DMEG, rank: int) -> str:
    subject_label = _public_subject_label(dmeg.subject_construct_ref)
    by_rank = {
        1: f"Resolving this could change Operational Priority (open gap on {subject_label}).",
        2: f"Resolving this could activate or alter a Material/Critical risk severity for {subject_label}.",
        3: f"Resolving this could change the governed conclusion for {subject_label}.",
        4: f"Resolving this could resolve an open contradiction on {subject_label}.",
        5: f"Resolving this could improve Assessment Reliability without changing a conclusion ({subject_label}).",
    }
    return by_rank[rank]


def _dmeg_gap_subjects(result: EvaluationResult) -> list[GapSubject]:
    subjects = []
    for dmeg in result.dmegs:
        rank = _stake_rank_for_dmeg(dmeg)
        subjects.append(GapSubject(
            gap_id=dmeg.dmeg_id,
            kind=GapKind.DMEG,
            subject_construct_ref=dmeg.subject_construct_ref,
            stake_rank=rank,
            stake_description=_stake_description_for_dmeg(dmeg, rank),
            reason_code=dmeg.reason_code,
        ))
    return subjects


def _unresolved_dimension_gap_subjects(result: EvaluationResult) -> list[GapSubject]:
    """Only for dimensions resolved to INSUFFICIENT_EVIDENCE that have NO
    open DMEG of their own (FR-17: "Supporting dimension unresolved, no
    trigger, no DMEG -> Insufficient Evidence remains visible... No ER1").
    A dimension already covered by an open DMEG is never duplicated here
    -- FR-18.1 rule 5 forbids a question that restates information already
    supplied by another selected question."""
    dmeg_subjects = {d.subject_construct_ref for d in result.dmegs}
    subjects = []
    for dim, dim_state in result.dimension_states.items():
        if dim_state.state != DimensionStateValue.INSUFFICIENT_EVIDENCE:
            continue
        if dim.value in dmeg_subjects:
            continue
        subjects.append(GapSubject(
            gap_id=f"DIMENSION:{dim.value}",
            kind=GapKind.UNRESOLVED_DIMENSION,
            subject_construct_ref=dim.value,
            stake_rank=5,
            stake_description=(
              f"{_DIMENSION_PUBLIC_LABEL.get(dim.value, dim.value)} remains Insufficient Evidence; "
              "resolving this would let it reach a governed conclusion."
            ),
            reason_code=None,
        ))
    return subjects


def select_gap_subjects(result: EvaluationResult) -> tuple[GapSubject, ...]:
    """Deterministic selection AND ranking, before any model call (BUILD
    AUTHORIZED decision). Sort key: (stake_rank ascending, DMEG before
    UNRESOLVED_DIMENSION at the same rank, gap_id for a stable tiebreak).
    The DMEG-before-dimension tiebreak is what structurally guarantees
    FR-18.3 ("Where ER1 is active, at least one question must target a
    DMEG"): ER1 implies at least one open DMEG (INV-12), and that DMEG can
    never be crowded out of the top 5 by a same-rank or lower-priority
    unresolved dimension. Returns AT MOST 5; may return fewer than 3 if
    fewer than 3 real gaps exist -- never fabricates a gap to hit a
    quantity floor (FR-18.1 permits questions only for real unresolved
    objects)."""
    subjects = _dmeg_gap_subjects(result) + _unresolved_dimension_gap_subjects(result)
    subjects.sort(key=lambda s: (s.stake_rank, 0 if s.kind == GapKind.DMEG else 1, s.gap_id))
    return tuple(subjects[:_MAX_GAP_SUBJECTS])


def _confirmed_evidence_refs(diagnostic: RecomputeDiagnostic) -> tuple[str, ...]:
    """Grounding rule 1 (explanation): Current+Confirmed identity only,
    never Unverified evidence, never raw observation content."""
    refs = []
    for item in diagnostic.active_evidence.items:
        if item.evidence_state == EvidenceState.CURRENT_CONFIRMED:
            refs.append(item.source_evidence_id)
            refs.append(item.observation_id)
    return tuple(sorted(set(refs)))


def _known_object_ids(result: EvaluationResult, gap_subjects: tuple[GapSubject, ...]) -> frozenset[str]:
    """The full citation-linking universe (explanation/grounding_check.py
    TAC-13 check): every governed-output-derived identifier legal for the
    model to reference, drawn only from what's actually present in
    EvaluationResult plus the deterministically-selected gap subjects --
    never anything the model itself supplies."""
    ids: set[str] = set()

    def _add_reason_code(rc) -> None:
        if rc is None:
            return
        ids.add(rc.code)
        ids.add(rc.governing_object_id)

    if result.objective_outcome is not None:
        ids.add(result.objective_outcome.objective_id)
        ids.add(result.objective_outcome.state.value)
        _add_reason_code(result.objective_outcome.reason_code)

    for dim, dim_state in result.dimension_states.items():
        ids.add(dim.value)
        ids.add(dim_state.state.value)
        _add_reason_code(dim_state.reason_code)

    for mech, risk in result.risk_records.items():
        ids.add(mech.value)
        if risk.potential_severity is not None:
            ids.add(risk.potential_severity.value)
        if risk.activated_severity is not None:
            ids.add(risk.activated_severity.value)
        _add_reason_code(risk.reason_code)

    for dmeg in result.dmegs:
        ids.add(dmeg.dmeg_id)
        ids.add(dmeg.subject_construct_ref)
        ids.add(dmeg.reason_code)

    ids.add(result.reliability.level.value)
    ids.update(result.reliability.limiting_factor_refs)

    # Live I5 validation finding (S2 run): OperationalPriority and
    # EvidenceReviewStatus are NOT per-instance object identifiers the way
    # a dmeg_id or a risk mechanism code is -- they are small, fixed, closed
    # governed-outcome VOCABULARIES (OP1/OP2/OP3/OPU; ER1/ER0), defined once
    # in domain/enums.py, and operational_priority_summary/
    # evidence_review_summary are unconditionally present in every
    # GroundingPackage (never Optional, unlike objective_outcome_summary).
    # A grounded explanation of a DMEG linked to OPERATIONAL_PRIORITY
    # legitimately needs to name what the field COULD move to (FR-18.1 rule
    # 2: "state what its resolution would change") -- that is naming a real
    # governed category the model was already told is at stake
    # (GapSubject.stake_description literally says "could change Operational
    # Priority"), not fabricating a new fact about the account. Only adding
    # the CURRENT value here was over-scoping citation-linking (whose actual
    # job, per TAC-13, is catching invented per-instance IDs) onto closed
    # methodology vocabulary it was never meant to restrict. This does NOT
    # extend to DimensionCode or RiskMechanismCode: which dimensions/
    # mechanisms are even in scope IS instance-specific (lifecycle-dependent
    # / evidence-dependent), so those remain scoped to what's actually
    # present in THIS result -- citation grounding is not weakened there.
    ids.update(v.value for v in OperationalPriority)
    _add_reason_code(result.operational_priority.reason_code)
    ids.update(result.operational_priority.contributing_risk_or_dimension_refs)

    ids.update(v.value for v in EvidenceReviewStatus)
    ids.update(result.evidence_review.dmeg_refs)
    ids.update(result.evidence_review.reason_codes)

    for gap in gap_subjects:
        ids.add(gap.gap_id)
        ids.add(gap.subject_construct_ref)
        if gap.reason_code:
            ids.add(gap.reason_code)

    ids.discard(None)
    return frozenset(ids)


def build_grounding_package(diagnostic: RecomputeDiagnostic, *, assessment_id: str, methodology_version: str) -> GroundingPackage:
    """The single entry point this whole package exposes for turning a
    RecomputeDiagnostic into model-ready input. Pure function; identical
    input always produces an identical GroundingPackage."""
    result = diagnostic.result

    objective_outcome_summary = None
    if result.objective_outcome is not None:
        objective_outcome_summary = {
            "state": result.objective_outcome.state.value,
            **_reason_code_dict(result.objective_outcome.reason_code),
        }

    dimension_state_summaries = tuple(
        {
            "dimension": dim.value,
            "state": dim_state.state.value,
            "dimension_reliability": dim_state.dimension_reliability,
            **_reason_code_dict(dim_state.reason_code),
        }
        for dim, dim_state in sorted(result.dimension_states.items(), key=lambda kv: kv[0].value)
    )

    risk_record_summaries = tuple(
        {
            "mechanism": mech.value,
            "potential_severity": risk.potential_severity.value if risk.potential_severity else None,
            "activated_severity": risk.activated_severity.value if risk.activated_severity else None,
            **(_reason_code_dict(risk.reason_code) if risk.reason_code else {}),
        }
        for mech, risk in sorted(result.risk_records.items(), key=lambda kv: kv[0].value)
    )

    reliability_summary = {
        "level": result.reliability.level.value,
        "limiting_factor_refs": list(result.reliability.limiting_factor_refs),
    }

    operational_priority_summary = {
        "value": result.operational_priority.value.value,
        **_reason_code_dict(result.operational_priority.reason_code),
    }

    evidence_review_summary = {
        "value": result.evidence_review.value.value,
        "dmeg_refs": list(result.evidence_review.dmeg_refs),
        "reason_codes": list(result.evidence_review.reason_codes),
    }

    gap_subjects = select_gap_subjects(result)

    return GroundingPackage(
        assessment_id=assessment_id,
        methodology_version=methodology_version,
        objective_outcome_summary=objective_outcome_summary,
        dimension_state_summaries=dimension_state_summaries,
        risk_record_summaries=risk_record_summaries,
        reliability_summary=reliability_summary,
        operational_priority_summary=operational_priority_summary,
        evidence_review_summary=evidence_review_summary,
        confirmed_evidence_refs=_confirmed_evidence_refs(diagnostic),
        gap_subjects=gap_subjects,
        known_object_ids=_known_object_ids(result, gap_subjects),
    )
