"""
Milestone 3C — extraction-to-confirmation integration glue.

Bridges raw evidence text to the completed, UNMODIFIED Milestone 2
extraction pipeline (extraction.pipeline.run_extraction), and hands back
a real extraction.pipeline.ExtractionResult for ui/app.py to build an
AppState from. This replaces ui/sample_data.py's hand-authored,
pipeline-bypassing fixture as the thing an AppState is built from
(ui/sample_data.py is retained only as a fallback/reference, no longer
the default path -- see ui/app.py).

This module reproduces no extraction or confirmation logic itself. It
only:
  (a) turns evidence-entry-mode input into an EvidenceObject batch
      (build_evidence_batch);
  (b) selects a provider -- FakeExtractionProvider by default (a
      deterministic test double, no network, no cost -- approved
      Milestone 3C decision 1), AnthropicExtractionProvider only when
      the caller explicitly opts into live mode;
  (c) calls run_extraction() exactly as eval/run_eval.py and every
      extraction test already do.

Nothing here inspects, reinterprets, or filters `accepted` / `rejected` /
`candidate_risk_signals` / `candidate_evidence_classifications` content --
that is ui/review_queue.py, ui/item_card.py, and ui/extraction_review.py's
job downstream of this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from domain.enums import Provenance
from domain.evidence import EvidenceObject

from extraction.pipeline import ExtractionResult, run_dimension_qualifier_classification, run_extraction
from extraction.provider import AnthropicExtractionProvider, ExtractionProvider, FakeExtractionProvider


@dataclass(frozen=True)
class ExtractionRunOutcome:
    extraction_result: ExtractionResult
    provider_label: str  # plain, user-facing -- never a class name (approved checkpoint UX constraint)
    evidence_batch: tuple


def build_evidence_batch(raw_texts: Iterable[str]) -> tuple:
    """One EvidenceObject per non-empty raw text block. `dimension` is
    left unassigned (None) -- deliberately: extraction infers candidate
    observations FROM the text, it does not receive a pre-assigned CHDM
    dimension as input (mirrors eval/run_eval.py's build_evidence_for_case,
    which does the same for every one of the 33 benchmark cases)."""
    texts = [t.strip() for t in raw_texts if t and t.strip()]
    return tuple(
        EvidenceObject(
            evidence_id=f"E{i + 1}", dimension=None, indicator_observation=text,
            source="pasted_customer_evidence", provenance=Provenance.USER_PROVIDED,
        )
        for i, text in enumerate(texts)
    )


def _fake_provider_for_arbitrary_text() -> FakeExtractionProvider:
    """Deterministic demo behavior for "Paste customer evidence" +
    "Deterministic demo extraction" (no canned response supplied).
    FakeExtractionProvider (extraction/provider.py) is a fixed test
    double with no real language understanding -- there is no canned
    response that could be correct in advance for arbitrary,
    unpredictable pasted text. Rather than fake intelligence, this mode
    honestly demonstrates the real pipeline mechanics end to end (schema
    validation, exact span resolution, system-field attachment, trace
    records) by wrapping each evidence item's full text as one
    ExperienceObservation whose span IS that full text -- guaranteed to
    resolve, since the span is drawn verbatim from the input, and it
    makes no claim about WHAT KIND of fact the text contains, since
    nothing here can actually tell. Richer, realistic multi-type
    extractions are what the curated Sample scenarios
    (ui/sample_scenarios.py) are for: those pair real benchmark evidence
    text with a hand-authored canned response representing a correct
    extraction of it."""
    def _responder(evidence_batch, repair_hint=None):
        return {
            "experience_observations": [
                {
                    "source_evidence_id": e.evidence_id,
                    "source_span": {"text": e.indicator_observation},
                    "basis": "EXPLICIT",
                    "statement": e.indicator_observation,
                }
                for e in evidence_batch
            ],
        }
    return FakeExtractionProvider(_responder, model_version="fake-extraction-provider-v1")


def run_pipeline(
    evidence_batch: tuple,
    *,
    use_live: bool,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    fake_response: Optional[dict] = None,
) -> ExtractionRunOutcome:
    """The single call site every evidence-entry mode routes through.

    fake_response, if given, is the exact canned dict a curated sample
    scenario supplies (ui/sample_scenarios.py's SampleScenario.fake_response)
    -- used only when use_live is False. If use_live is False and
    fake_response is None (the "Paste customer evidence" + Fake
    combination), the generic arbitrary-text fallback above is used
    instead. use_live switches only the provider to
    AnthropicExtractionProvider -- run_extraction() itself is never
    altered or bypassed in either mode (approved checkpoint decision 3:
    "scope remains integration-only")."""
    if use_live:
        provider: ExtractionProvider = (
            AnthropicExtractionProvider(api_key=api_key, model=model) if model
            else AnthropicExtractionProvider(api_key=api_key)
        )
        provider_label = f"Live AI extraction ({provider.model_version})"
    else:
        provider = (
            FakeExtractionProvider(fake_response) if fake_response is not None
            else _fake_provider_for_arbitrary_text()
        )
        provider_label = "Deterministic demo extraction (no AI call, no network)"

    result = run_extraction(evidence_batch, provider)
    # Milestone 4C: the SEPARATE stage-2 D2/D6 classifier call, run
    # immediately after stage 1 -- the exact contract already proven in
    # eval/run_eval.py (same provider instance for both stages). Its own
    # skip-if-nothing-eligible logic makes this a safe no-op whenever
    # `result` has no AdoptionObservation/StakeholderObservation (e.g. the
    # generic arbitrary-text fallback above, which only ever produces
    # ExperienceObservations). Before this call, D2/D6 candidate-qualifier
    # confirmation had nothing to review -- stage 2 simply never ran in
    # the UI path (M3-OD-01, now resolved end to end for D2/D6).
    result = run_dimension_qualifier_classification(result, provider)
    return ExtractionRunOutcome(extraction_result=result, provider_label=provider_label, evidence_batch=evidence_batch)
