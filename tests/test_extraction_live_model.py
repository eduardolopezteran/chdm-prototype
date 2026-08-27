"""
Milestone 2B — live-model sanity tests. SEPARATE from the deterministic
suite on purpose (spec §19/§11): a live-model or network failure here
must never make the deterministic CI fail, and these tests must never run
implicitly as part of the deterministic suite's normal green/red signal.

Auto-skips (prints SKIP, does not fail) whenever ANTHROPIC_API_KEY is not
set. This is intentionally a light sanity check, not the full 15-case
evaluation — that is eval/run_eval.py's job, meant to be run standalone
wherever real network egress to api.anthropic.com is available (this
environment's proxy blocks authenticated calls to that host; see the
Milestone 2B checkpoint report for how that was diagnosed and how results
were obtained instead).
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import EvidenceState, Provenance
from domain.evidence import EvidenceObject

from extraction.enums import RejectionReason
from extraction.pipeline import run_extraction


def _skip_if_no_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIP  (no ANTHROPIC_API_KEY in environment — live-model tests do not run)")
        return True
    return False


def test_live_sponsor_departure_extraction_sanity():
    if _skip_if_no_key():
        return
    from extraction.provider import AnthropicExtractionProvider

    text = "Ana told us Roberto left the company in July. No replacement sponsor has been named."
    e = EvidenceObject("E1", None, text, "account_note", Provenance.USER_PROVIDED)
    provider = AnthropicExtractionProvider()
    result = run_extraction((e,), provider)

    assert result.request_failure is None, result.request_failure
    assert len(result.accepted) >= 1
    for obs in result.accepted:
        assert obs.system.evidence_state == EvidenceState.CURRENT_UNVERIFIED
    boundary_hits = [r for r in result.rejected if r.reason == RejectionReason.BOUNDARY_VIOLATION]
    # Not asserted to be zero (the model MAY attempt something), but the
    # governed invariant under test is that nothing prohibited is ever in
    # `accepted` — structurally guaranteed by the dataclasses themselves,
    # reconfirmed here against real output.
    print(f"  boundary attempts this run: {len(boundary_hits)} (must never appear in `accepted`)")


def test_live_contradiction_case_sanity():
    if _skip_if_no_key():
        return
    from extraction.provider import AnthropicExtractionProvider

    text_a = "The customer confirmed the expected reduction in close time was achieved."
    text_b = "Average close time remains unchanged from baseline."
    ea = EvidenceObject("EA", None, text_a, "qbr_note", Provenance.USER_PROVIDED)
    eb = EvidenceObject("EB", None, text_b, "ops_report", Provenance.USER_PROVIDED)
    provider = AnthropicExtractionProvider()
    result = run_extraction((ea, eb), provider)

    assert result.request_failure is None, result.request_failure
    for c in result.candidate_contradictions:
        assert c.status == "CANDIDATE"


TESTS = [
    test_live_sponsor_departure_extraction_sanity,
    test_live_contradiction_case_sanity,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS/SKIP  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} ran (see SKIP markers above for any not actually executed)")
