"""
Milestone 3B — UI session state.

Pure data holder. In-memory only (approved Milestone 3B scope: no
persistence, no multi-user, no auth -- the confirmation journal lives
only for the lifetime of this Streamlit session and resets on reload).
Holds exactly what ui/actions.py needs to call the completed Milestone 3A
confirmation/ backend; no confirmation-state, active-evidence,
consequentiality, or recomputation logic is reproduced here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AppState:
    account: Any
    registry: Any
    extraction_result: Any
    dimensions_to_evaluate: tuple
    # Fixed at scenario setup, never reviewer-editable -- see
    # sample_data.py and MANIFEST.txt's D2/D6 read-only decision. Exposing
    # a picker for this would functionally resolve M3-OD-01 through UI
    # behavior, which the approved checkpoint explicitly forbids.
    dimension_qualifier_overrides: Dict[str, tuple]
    reviewer: str = ""
    confirmation_records: List[Any] = field(default_factory=list)
    # target_observation_id -> (confirmation_history_version_at_computation, ConsequentialityReport)
    consequentiality_cache: Dict[str, Tuple[int, Any]] = field(default_factory=dict)
    current_diagnostic: Optional[Any] = None       # confirmation.schemas.RecomputeDiagnostic
    previous_diagnostic: Optional[Any] = None       # confirmation.schemas.RecomputeDiagnostic, for diffing
    # Informational only (e.g. could drive a "recently viewed" highlight) --
    # more than one card can be expanded at once, so nothing in ui/actions.py
    # or ui/item_card.py gates rendering behind this being a single value.
    opened_target_observation_id: Optional[str] = None
    # Per-target, not a single shared field -- more than one item can be
    # expanded at once, and an error from submitting on one item must never
    # display on a different, unrelated open card.
    errors_by_target: Dict[str, str] = field(default_factory=dict)
    # Milestone 3C. Plain, user-facing description of which extraction
    # provider produced this session's extraction_result (e.g.
    # "Deterministic demo extraction (no AI call, no network)" or
    # "Live AI extraction (claude-haiku-4-5-...)") -- set once by
    # ui/app.py._build_state_from_outcome, from
    # ui/extraction_bridge.ExtractionRunOutcome.provider_label. Never a
    # class name, and never shown anywhere but the sidebar/setup summary.
    provider_label: str = ""
    # Milestone 3E. A SEPARATE append-only journal from confirmation_records
    # above -- confirmation.contradiction_review.ContradictionReviewRecord
    # instances, one per Acknowledge/Dismiss on a flagged CandidateContradiction
    # marker. Deliberately not merged into confirmation_records: a contradiction
    # marker is not a HumanConfirmationRecord target (see
    # confirmation/contradiction_review.py's module docstring for why).
    contradiction_review_records: List[Any] = field(default_factory=list)
    # Per-contradiction_id, same rationale as errors_by_target above: more
    # than one flagged-conflict card can be expanded at once, and an error
    # from submitting on one must never display on a different card.
    contradiction_errors: Dict[str, str] = field(default_factory=dict)
    # I5 build block. On-demand only (approved decision: no synchronous
    # generation) -- (confirmation_history_version_at_generation,
    # explanation.schemas.ExplanationResult), same staleness convention as
    # consequentiality_cache above: a later confirmation action invalidates
    # the cached explanation rather than silently leaving a stale one on
    # screen. None means "never generated at the current version."
    explanation_cache: Optional[Tuple[int, Any]] = None

    @property
    def confirmation_history_version(self) -> int:
        """Monotonic cache-invalidation key. The confirmation journal is
        append-only (Milestone 3A: nothing is ever mutated or deleted), so
        the record count alone is a safe, cheap version number -- any new
        action strictly increases it, and no action ever decreases it."""
        return len(self.confirmation_records)
