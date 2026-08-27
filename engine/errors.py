"""Engine-wide error types."""


class UnresolvedMethodologyError(Exception):
    """Raised when a code path would require inventing a CHDM rule that
    is explicitly unresolved in the frozen baseline (e.g. UC-01, the
    general D2-D8B contradiction-to-Dimension-State rule). Must never be
    caught and silently worked around — it exists to stop execution and
    surface the gap for methodology governance review, per BAR-01 §7 and
    the executive build-authorization corrections."""

    def __init__(self, methodology_gap_id: str, message: str) -> None:
        self.methodology_gap_id = methodology_gap_id
        super().__init__(f"[{methodology_gap_id}] {message}")


class InvariantViolationError(Exception):
    """Raised when a computed governed output would violate one of CHDM
    v0.1's 22 invariants (§15). Fail-fast: blocks rendering of that
    specific output rather than silently coercing to a nearby valid
    state (Technical Architecture §20)."""

    def __init__(self, invariant_id: str, message: str) -> None:
        self.invariant_id = invariant_id
        super().__init__(f"[{invariant_id}] {message}")


class NotImplementedForMilestone1(Exception):
    """Raised for a genuine CHDM v0.1 code path that is simply out of
    scope for Build Milestone 1 (e.g. a dimension not required by S1-S6,
    or the L1 Not-Yet-Expected sub-case of D1). Distinct from
    UnresolvedMethodologyError: the rule EXISTS and is knowable, it just
    hasn't been built yet — this is an implementation scoping boundary,
    not a methodology gap."""
