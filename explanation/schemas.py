"""
I5 build block -- explanation-layer dataclasses.

Field names deliberately match the Technical Solution Architecture's own
data-model table (CHDM_Technical_Solution_Architecture_v1.0.md, "Diagnostic
Question" / "AI Explanation" rows) so this implementation is directly
traceable to the already-reconciled design, not a new invention.

Two distinct grounding rules apply to the two objects this package
produces (per BUILD AUTHORIZED decision):

  1. Explanation factual claims may rely only on governed CHDM outputs,
     Reason Codes, methodology object references, and Current+Confirmed
     evidence. GroundingPackage.confirmed_evidence_refs is the ONLY
     evidence-identity information ever handed to the model for
     explanation grounding -- never Unverified evidence.

  2. Diagnostic questions may reference governed DMEGs, unresolved
     dimensions, and other governed uncertainty objects even when the
     underlying evidence is missing or unconfirmed -- that is the whole
     point of a diagnostic question (FR-18.1: it targets a specific
     UNRESOLVED object). This does not authorize treating unverified
     evidence as true: GapSubject below carries only the governed
     methodology-object identity of the gap (a dmeg_id, a dimension code,
     a reason code) -- never a raw unconfirmed observation's content.

Neither grounding rule permits passing arbitrary unconfirmed extraction
content into the model. GroundingPackage never carries anything from
extraction.schemas or confirmation.schemas.ActiveEvidenceItem.original.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .enums import GapKind, ExplanationFailureReason


@dataclass(frozen=True)
class GapSubject:
    """One real, already-governed unresolved methodology object a
    diagnostic question may target (FR-18.1). Never invented: every
    instance is built directly from a real DMEG or a real
    DimensionState(state=INSUFFICIENT_EVIDENCE) already present in
    EvaluationResult -- see explanation/grounding.py select_gap_subjects().
    """
    gap_id: str                      # the real dmeg_id, or "DIMENSION:{code}" for an unresolved dimension
    kind: GapKind
    subject_construct_ref: str       # e.g. "D6", "CR-01", "OBJECTIVE" (DMEG) or the dimension code (unresolved dim)
    stake_rank: int                  # 1 (highest stake) .. 5, per FR-18.2 priority order
    stake_description: str           # deterministic, human-readable -- what resolving this would change
    reason_code: Optional[str] = None  # ER-DMEG-* family code, if kind == DMEG


@dataclass(frozen=True)
class GroundingPackage:
    """The closed grounding package handed to the model on one explanation
    call (Technical Architecture §12). Nothing outside this object is ever
    visible to the model -- no raw account text, no extraction candidates,
    no unconfirmed observation content.
    """
    assessment_id: str
    methodology_version: str
    # -- explanation grounding (Current+Confirmed only) --
    objective_outcome_summary: Optional[dict]      # {"state", "reason_code", "governing_object_id", "human_readable_text"}
    dimension_state_summaries: tuple[dict, ...]     # one per evaluated dimension, same shape as above + "dimension"
    risk_record_summaries: tuple[dict, ...]         # one per present risk mechanism
    reliability_summary: dict                        # {"level", "limiting_factor_refs"}
    operational_priority_summary: dict               # {"value", "reason_code", "governing_object_id", "human_readable_text"}
    evidence_review_summary: dict                     # {"value", "dmeg_refs", "reason_codes"}
    confirmed_evidence_refs: tuple[str, ...]          # Current+Confirmed observation/evidence ids only (identity, not content)
    # -- diagnostic-question grounding (governed uncertainty objects) --
    gap_subjects: tuple[GapSubject, ...]              # already deterministically selected + ranked, <=5
    # -- citation-linking universe: every object ID legal to cite --
    known_object_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.assessment_id:
            raise ValueError("GroundingPackage.assessment_id must not be empty.")
        if not self.methodology_version:
            raise ValueError("GroundingPackage.methodology_version must not be empty.")
        if len(self.gap_subjects) > 5:
            raise ValueError(
                "GroundingPackage.gap_subjects must not exceed 5 (FR-18.2) -- "
                "selection/ranking in explanation/grounding.py must cap before "
                "constructing this object."
            )


@dataclass(frozen=True)
class DiagnosticQuestion:
    question_id: str
    text: str
    source_gap_ref: str       # == some GapSubject.gap_id from the grounding package this was built from
    stake_description: str
    rank: int


@dataclass(frozen=True)
class Explanation:
    explanation_id: str
    grounding_package_ref: str
    generated_text: str
    model_version: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ExplanationResult:
    """What ui/explanation_panel.py actually renders. Exactly one of
    (explanation present) or (fallback_reason set) -- see __post_init__.
    Never written back into any confirmation/domain/engine object; this is
    a pure leaf/display value returned to the caller."""
    explanation: Optional[Explanation]
    questions: tuple[DiagnosticQuestion, ...]
    grounding_package: GroundingPackage
    fallback_reason: Optional[ExplanationFailureReason] = None
    fallback_detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fallback_reason is None and self.explanation is None:
            raise ValueError(
                "ExplanationResult must carry either an Explanation or a "
                "fallback_reason -- a silent empty result is not a valid "
                "outcome (mirrors extraction's own fail-loud contract)."
            )
        if self.fallback_reason is not None and self.explanation is not None:
            raise ValueError(
                "ExplanationResult must not carry both an Explanation and a "
                "fallback_reason -- fallback means no AI-generated content is "
                "shown at all, per the approved fail-closed behavior."
            )
        if self.fallback_reason is not None and self.questions:
            raise ValueError(
                "ExplanationResult with a fallback_reason must carry no "
                "questions -- fallback shows only the existing deterministic "
                "result plus a plain notice, never partial AI content."
            )

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None
