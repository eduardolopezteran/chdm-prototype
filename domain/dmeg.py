"""
CHDM v0.1 §8 — Decision-Material Evidence Gap.

A DMEG object only exists once all three gates (DMEG-1, DMEG-2, DMEG-3)
are satisfied — it is not raised speculatively. See
engine/dmeg_engine.py (the Differential Evaluation Primitive) for how the
gates are actually evaluated; this dataclass is the resulting record.

`linked_conclusions` (DMEG-2) is a set, not a single value, because a
DMEG can materially affect more than one governed conclusion at once —
and, critically, is NOT assumed to include OPERATIONAL_PRIORITY merely
because the DMEG exists. Whether a DMEG affects Operational Priority is
determined by whether the OP outcome actually differs under the DMEG-3
differential test (or, for the REQ-05/REQ-06 structural gap case, by
CHDM's own named validation case) — never inferred from DMEG count alone.
"""

from dataclasses import dataclass, field

from .enums import DMEGLinkedConclusion


@dataclass(frozen=True)
class DMEG:
    dmeg_id: str
    subject_construct_ref: str          # e.g. "D6" or "CR-01" or "OBJECTIVE"
    dmeg1_requirement_condition: str    # which requirement class/trigger satisfied DMEG-1
    dmeg2_linked_conclusions: frozenset[DMEGLinkedConclusion]  # which conclusion(s) actually differ
    dmeg3_resolution_state_space: tuple[str, ...]   # the enumerable states actually tried
    dmeg3_outcomes_differ: bool         # must be True — this is what makes it a DMEG
    reason_code: str                     # an ER-DMEG-* family code (registry/reason_codes.yaml)

    def __post_init__(self) -> None:
        if not self.dmeg3_outcomes_differ:
            raise ValueError(
                "DMEG object constructed with dmeg3_outcomes_differ=False — "
                "if plausible resolution cannot materially change a conclusion, "
                "the missing evidence is NOT a DMEG even if useful to obtain "
                "(CHDM v0.1 §8, DMEG-3). Do not construct a DMEG object in this case."
            )
        if len(self.dmeg3_resolution_state_space) < 2:
            raise ValueError(
                "DMEG-3 requires at least two methodologically permissible "
                "resolutions to have been evaluated (§8)."
            )
        if not self.dmeg2_linked_conclusions:
            raise ValueError(
                "DMEG-2 violation: a DMEG must be linked to at least one governed "
                "conclusion (Objective/D1, risk severity, dimension state, or "
                "Operational Priority) — an unlinked DMEG is invalid (CHDM v0.1 §8)."
            )
        if not self.reason_code.startswith("ER-DMEG"):
            raise ValueError(
                "DMEG.reason_code must be from the ER-DMEG-* family "
                "(registry/reason_codes.yaml §13.2)."
            )

    @property
    def affects_operational_priority(self) -> bool:
        return DMEGLinkedConclusion.OPERATIONAL_PRIORITY in self.dmeg2_linked_conclusions
