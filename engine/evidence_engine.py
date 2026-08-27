"""
CHDM v0.1 §3 — Evidence-state helper queries.

Milestone 1 scope: evidence is hand-authored directly at its target
state (BAR-01 §6) — there is no live extraction/freshness-from-date
pipeline in this milestone (CHDM v0.1 itself defines no universal
staleness threshold; see registry/evidence_states.yaml `staleness`).
This module's job is narrow: apply the §3.3 confirmation boundary
uniformly wherever the engine needs to ask "is this signal usable for an
activation/escalation decision, or only as a potential one."
"""

from __future__ import annotations

from domain.enums import EvidenceState


def is_current_confirmed(evidence_state: EvidenceState) -> bool:
    """CHDM v0.1 §3.3 confirmation boundary: only Current + Confirmed may
    activate severity, establish Achieved, resolve a contradiction, or
    produce a consequential deterministic escalation."""
    return evidence_state == EvidenceState.CURRENT_CONFIRMED


def is_usable_as_potential(evidence_state: EvidenceState) -> bool:
    """Any evidence state except Not Applicable / Unavailable may still
    contribute a *potential* (non-activated) signal — including
    Unverified, Stale, and Contradictory, each of which is diagnostically
    meaningful even though none can activate anything (§3.2)."""
    return evidence_state not in (EvidenceState.NOT_APPLICABLE, EvidenceState.UNAVAILABLE)
