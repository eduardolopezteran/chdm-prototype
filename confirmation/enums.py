"""
Milestone 3A — Human Confirmation Backend: confirmation-local controlled
vocabularies.

Deliberately separate from `domain.enums` (Milestone 1's CHDM-governed
vocabulary — untouched by this milestone) and from `extraction.enums`
(Milestone 2's extraction-pipeline vocabulary — also untouched). These are
confirmation-workflow concepts only: what kind of Milestone 2 extraction
output a HumanConfirmationRecord targets, and why a previously-extracted
item is excluded from the active-evidence set consumed by recompute().
Nothing here is consumed by any deterministic CHDM rule.
"""

from enum import Enum


class ConfirmationTargetKind(str, Enum):
    """What kind of Milestone 2 extraction output a HumanConfirmationRecord
    reviews. A candidate classification (risk signal / evidence
    classification) is ALWAYS reviewed as a decision separate from the
    semantic observation it interprets — confirming the underlying fact
    never implicitly confirms an AI-proposed interpretation of that fact,
    and vice versa (Milestone 3A spec: candidate-classification handling
    is a separate decision)."""
    SEMANTIC_OBSERVATION = "SEMANTIC_OBSERVATION"                        # one of the 7 span-grounded positive types
    MISSING_INFORMATION_CANDIDATE = "MISSING_INFORMATION_CANDIDATE"
    CANDIDATE_RISK_SIGNAL = "CANDIDATE_RISK_SIGNAL"
    CANDIDATE_EVIDENCE_CLASSIFICATION = "CANDIDATE_EVIDENCE_CLASSIFICATION"
    # Milestone 4C (approved architecture checkpoint): a Milestone 4B
    # CandidateDimensionQualifier (D2 or D6, simple or deterministically
    # composed compound — the object's own `.dimension` field distinguishes
    # D2/D6, exactly like CANDIDATE_RISK_SIGNAL's `.mechanism` distinguishes
    # CR-01/02/08 within one target kind; compound-vs-simple is invisible to
    # confirmation, since composition already happened deterministically in
    # extraction/pipeline.py before a human ever reviews the candidate).
    # Same separate-decision rule as above: confirming a D2/D6 qualifier
    # candidate never implicitly confirms its supporting stage-1
    # AdoptionObservation/StakeholderObservation, and vice versa.
    CANDIDATE_DIMENSION_QUALIFIER = "CANDIDATE_DIMENSION_QUALIFIER"


class ObjectiveResolutionStatus(str, Enum):
    """Milestone 3D — confirmation-layer-only integration/audit metadata
    describing whether confirmed extraction evidence currently
    establishes an objective identity for this account. NOT a governed
    CHDM state: domain/objective.py is untouched, and there is no new
    ObjectiveOutcome value anywhere. When CONFLICTING, the deterministic
    engine still simply receives an unresolved/unknown Objective (is_known
    =False) and Objective Outcome renders UNKNOWN under the existing,
    unmodified rules -- this enum exists only so the UI/integration layer
    can say WHY it is Unknown (conflicting confirmed statements, vs.
    simply no evidence yet), which the engine's own UNKNOWN state cannot
    distinguish on its own."""
    ESTABLISHED = "ESTABLISHED"          # exactly one confirmed objective identity (or one already declared)
    NOT_ESTABLISHED = "NOT_ESTABLISHED"  # no confirmed ObjectiveCandidate evidence yet
    CONFLICTING = "CONFLICTING"          # 2+ confirmed statements, normalized text differs -- never auto-picked


class ExclusionReason(str, Enum):
    """Why a previously-extracted item is NOT part of the active-evidence
    set at a given moment (active_evidence.reconstruct_active_evidence()).
    Every exclusion is explicit and auditable via ExclusionRecord — nothing
    is ever silently dropped. Deliberately does NOT include "never
    reviewed" — an unreviewed item stays active as Current+Unverified
    (unchanged pipeline default), it is not excluded."""
    REJECTED = "REJECTED"
    # CANNOT_CONFIRM is deliberately NOT an exclusion reason: the item
    # remains active, Current+Unverified, preserving uncertainty rather
    # than removing evidence a reviewer merely couldn't validate yet
    # (Milestone 3A test scenario: "Cannot Confirm preserves uncertainty").


class ContradictionReviewAction(str, Enum):
    """Milestone 3E — confirmation-layer/UI-layer-only vocabulary for
    reviewing an AI-flagged CandidateContradiction MARKER itself (the
    flagged relationship between two observations), as distinct from
    domain.enums.ConfirmationAction (which governs confirming/correcting/
    rejecting an observation itself). Deliberately NOT added to
    domain/enums.py, and deliberately NOT reusing
    ConfirmationAction.CANNOT_CONFIRM -- that value already has an
    established evidence-review meaning ("reviewed, but not validated one
    way or the other") and must not be overloaded with this different,
    audit-only meaning.

    ACKNOWLEDGE: the reviewer has seen the flagged conflict. Resolution of
    the underlying disagreement still only happens by reviewing the two
    referenced observations themselves (via the normal Confirm / Correct /
    Reject / Cannot Confirm actions on each) -- this action has no effect
    on either observation.

    DISMISS: the reviewer has determined that the AI's flagged pair is not
    actually contradictory. This does not confirm, correct, or reject
    either referenced observation either -- it only records that the
    CONFLICT ITSELF was judged not real.

    Neither value ever sets ObjectiveOutcome, D1, Operational Priority,
    Evidence Review, or Reliability, and neither creates any new governed
    contradiction state -- see confirmation/contradiction_review.py."""
    ACKNOWLEDGE = "ACKNOWLEDGE"
    DISMISS = "DISMISS"
