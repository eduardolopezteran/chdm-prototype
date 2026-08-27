"""
Build Milestone 2 — isolated model-provider boundary (spec §7).

One abstract interface, one deterministic test double (no network,
powers every Checkpoint 2A test), and one concrete Anthropic
implementation. No multi-provider orchestration framework, no LangChain/
LCEL/LangGraph/agents — direct, explicit API invocation only.

Structured-output mechanism (Checkpoint 2A refinement 5): forced tool use.
Anthropic's Messages API supports two paths to schema-constrained output:
(a) forced tool use — a single tool is declared with an `input_schema`
and `tool_choice` forces the model to call exactly that tool, so its
arguments arrive already parsed as JSON conforming to the schema; (b) a
newer structured-output / strict-JSON response mode, gated behind a beta
header and narrower model/version availability. This module uses (a):
- it needs no beta header or model-version-specific feature flag, so it
  is stable across the model family rather than tied to a beta surface;
- the extraction task IS naturally single-shot structured data (one
  object matching TOP_LEVEL_SCHEMA), so a single forced tool call maps
  onto it exactly — there is no multi-turn tool-use loop to build, and no
  "tool" semantics being introduced for their own sake (the tool is never
  actually "called" against a real function; it's used purely as the
  schema-conformance mechanism, its declared purpose in the API);
- the SDK returns the tool-call arguments as already-parsed JSON, so
  there is no second free-text-JSON round trip to parse/repair client-side
  beyond the one repair/retry the pipeline already performs for a
  malformed top-level shape (spec §13).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from domain.enums import DimensionCode
from domain.evidence import EvidenceObject

from .enums import ObservationType
from .errors import ModelServiceError
from .json_schemas import TOP_LEVEL_SCHEMA, ISOLATED_DIMENSION_QUALIFIER_TOP_LEVEL_SCHEMA
from .prompts import (
    EXTRACTION_TOOL_NAME, build_system_prompt, build_user_message,
    DIMENSION_QUALIFIER_TOOL_NAME, build_isolated_dimension_qualifier_system_prompt,
    build_isolated_dimension_qualifier_user_message,
)

# Milestone 4B isolated-classifier architecture checkpoint: the small,
# local DimensionCode -> ObservationType map an isolated call needs to
# select its own envelope schema from ISOLATED_DIMENSION_QUALIFIER_
# TOP_LEVEL_SCHEMA (json_schemas.py, keyed by ObservationType). Kept
# local to this module rather than promoted to enums.py -- nothing else
# needs this exact mapping; extraction.schemas._DIMENSION_TO_CANDIDATE_
# QUALIFIERS is a different map (dimension -> allowed qualifier strings),
# not this one.
_DIMENSION_TO_QUALIFIER_OBSERVATION_TYPE = {
    DimensionCode.D2: ObservationType.CANDIDATE_D2_QUALIFIER,
    DimensionCode.D6: ObservationType.CANDIDATE_D6_QUALIFIER,
}

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class ExtractionProvider(ABC):
    """The entire provider boundary. `extract()` takes the evidence items
    in scope for one extraction request and returns the raw parsed JSON
    dict — schema conformance is attempted by the provider (via whatever
    mechanism it uses) but is NOT assumed; pipeline.py always re-validates
    independently regardless of provider."""

    @abstractmethod
    def extract(self, evidence_batch: tuple[EvidenceObject, ...], *, repair_hint: Optional[str] = None) -> dict:
        """Returns a raw dict intended to conform to TOP_LEVEL_SCHEMA.
        Raises ModelServiceError on any provider/API/network failure —
        never returns a partial or fabricated result on failure."""
        raise NotImplementedError

    @abstractmethod
    def propose_isolated_dimension_qualifier(
        self,
        dimension: DimensionCode,
        observation,
        *,
        repair_hint: Optional[str] = None,
    ) -> dict:
        """Milestone 4B isolated-classifier architecture checkpoint.
        REPLACES the prior batched propose_dimension_qualifiers method
        (removed, not kept as a second active classifier path — approved
        architecture item C). Takes ONE already-accepted AdoptionObservation
        (dimension=D2) or StakeholderObservation (dimension=D6) from a
        finished stage-1 ExtractionResult and returns a raw dict intended
        to conform to ISOLATED_DIMENSION_QUALIFIER_TOP_LEVEL_SCHEMA's
        entry for that dimension (a single-channel envelope holding AT
        MOST ONE qualifier item). Raises ModelServiceError on any
        provider/API/network failure — never returns a partial or
        fabricated result on failure, exactly mirroring extract()'s own
        failure contract. The caller (extraction.pipeline.
        run_dimension_qualifier_classification) invokes this once per
        eligible observation — never once per run — so a provider
        implementation must not assume or require any other observation
        to be visible within a single call."""
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    @property
    def model_version(self) -> str:
        return "unknown"


class FakeExtractionProvider(ExtractionProvider):
    """Deterministic, no-network test double. Construct with either:
      - a single dict/list -> returned (cycled if list) on every call;
      - a callable(evidence_batch, repair_hint) -> dict, for tests that
        need to assert on the exact request or simulate a stateful
        sequence (e.g. first call malformed, retry call valid).
    This is what every Checkpoint 2A deterministic test uses — real
    network calls never occur in the deterministic test suite (spec §19:
    "Model-dependent tests should be separated from deterministic unit
    tests")."""

    def __init__(
        self,
        responses: Union[dict, list, Callable[..., dict]],
        *,
        model_version: str = "fake-extraction-provider-v1",
        raise_service_error: bool = False,
        # Milestone 4B isolated-classifier architecture checkpoint: the
        # stage-2 D2/D6 classifier's own, independently configurable
        # response(s) — same dict/list/callable shape as `responses`
        # above, but deliberately a SEPARATE constructor parameter (never
        # reused from `responses`) since a single test/fake provider
        # instance may need to serve both a stage-1 extract() call and one
        # or more stage-2 propose_isolated_dimension_qualifier() calls
        # with unrelated response bodies. A callable here now receives
        # (dimension, observation, repair_hint) — one isolated call's
        # worth of context, not the old batched (adoption_observations,
        # stakeholder_observations, repair_hint) shape. A list is
        # consumed one-per-isolated-call, in call order, across BOTH
        # dimensions (mirrors `responses`' own list-cycling convention).
        # None (the default) means this fake was not configured for
        # stage-2 calls; calling propose_isolated_dimension_qualifier() on
        # it raises a clear error rather than silently returning something
        # stage-1-shaped.
        dimension_qualifier_responses: Optional[Union[dict, list, Callable[..., dict]]] = None,
        raise_dimension_qualifier_service_error: bool = False,
    ):
        self._responses = responses
        self._call_count = 0
        self._model_version = model_version
        self._raise_service_error = raise_service_error
        # Same shape as AnthropicExtractionProvider.call_log, so
        # eval/run_eval.py's scoring logic can be exercised against this
        # fake provider too (used for pre-flight validation of the
        # evaluation harness itself, without any network dependency).
        self.call_log: list[dict] = []
        # Milestone 4B: stage-2's own call count/log, kept entirely
        # separate from stage-1's `_call_count`/`call_log` above so a
        # test exercising both stages never conflates the two.
        self._dimension_qualifier_responses = dimension_qualifier_responses
        self._raise_dimension_qualifier_service_error = raise_dimension_qualifier_service_error
        self._dq_call_count = 0
        self.dimension_qualifier_call_log: list[dict] = []

    def extract(self, evidence_batch: tuple[EvidenceObject, ...], *, repair_hint: Optional[str] = None) -> dict:
        self._call_count += 1
        if self._raise_service_error:
            self.call_log.append({"latency_ms": 0.0, "usage": None, "success": False, "error": "simulated"})
            raise ModelServiceError("FakeExtractionProvider configured to simulate a service failure.")
        if callable(self._responses) and not isinstance(self._responses, (dict, list)):
            response = self._responses(evidence_batch, repair_hint)
        elif isinstance(self._responses, list):
            idx = min(self._call_count - 1, len(self._responses) - 1)
            response = self._responses[idx]
        else:
            response = self._responses
        self.call_log.append({"latency_ms": 0.0, "usage": None, "success": True, "error": None})
        return response

    def propose_isolated_dimension_qualifier(
        self,
        dimension: DimensionCode,
        observation,
        *,
        repair_hint: Optional[str] = None,
    ) -> dict:
        self._dq_call_count += 1
        if self._raise_dimension_qualifier_service_error:
            self.dimension_qualifier_call_log.append(
                {"latency_ms": 0.0, "usage": None, "success": False, "error": "simulated"}
            )
            raise ModelServiceError(
                "FakeExtractionProvider configured to simulate a stage-2 isolated "
                "dimension-qualifier service failure."
            )
        if self._dimension_qualifier_responses is None:
            raise ModelServiceError(
                "FakeExtractionProvider was not configured with "
                "dimension_qualifier_responses — this fake cannot serve a stage-2 "
                "propose_isolated_dimension_qualifier() call. Pass "
                "dimension_qualifier_responses=... at construction time."
            )
        responses = self._dimension_qualifier_responses
        if callable(responses) and not isinstance(responses, (dict, list)):
            response = responses(dimension, observation, repair_hint)
        elif isinstance(responses, list):
            idx = min(self._dq_call_count - 1, len(responses) - 1)
            response = responses[idx]
        else:
            response = responses
        self.dimension_qualifier_call_log.append(
            {"latency_ms": 0.0, "usage": None, "success": True, "error": None}
        )
        return response

    @property
    def dimension_qualifier_call_count(self) -> int:
        return self._dq_call_count

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def model_version(self) -> str:
        return self._model_version


class AnthropicExtractionProvider(ExtractionProvider):
    """Live provider. Implemented now per Checkpoint 2A refinement 5 ("document
    the choice") but not exercised by the deterministic suite — execution is
    deferred until ANTHROPIC_API_KEY is available (Checkpoint 2A refinement 6),
    exercised only by tests/test_extraction_live_model.py (Checkpoint 2B),
    which auto-skips without a key.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_ANTHROPIC_MODEL):
        self._api_key = api_key
        self._model = model
        self._client = None  # constructed lazily on first extract() call
        # Evaluation instrumentation only (Milestone 2B spec §2: record
        # latency/usage/retries per call) — never includes the API key or
        # any request/response body content, just call-shape metadata.
        # One entry per extract() invocation, in order.
        self.call_log: list[dict] = []
        # Milestone 4B: stage-2's own instrumentation log, kept entirely
        # separate from self.call_log (stage-1's own) — see
        # propose_dimension_qualifiers below.
        self.dimension_qualifier_call_log: list[dict] = []

    def _get_client(self):
        if self._client is None:
            import os
            try:
                import anthropic  # local import: keeps the `anthropic` package an
                                    # optional dependency for everyone who never
                                    # touches the live provider (deterministic tests
                                    # and Checkpoint 2A never import this path).
            except ImportError as e:
                raise ModelServiceError(
                    "The 'anthropic' package is not installed. Install it "
                    "(`pip install anthropic`) to use AnthropicExtractionProvider.",
                    cause=e,
                ) from e
            # Environment adaptation only (not a boundary/schema change):
            # some sandboxed network environments terminate TLS at a proxy
            # presenting a CA not in the default trust store. If
            # ANTHROPIC_CA_BUNDLE is set, use it explicitly; otherwise fall
            # back to the SDK's normal default trust store untouched.
            ca_bundle = os.environ.get("ANTHROPIC_CA_BUNDLE")
            http_client = None
            if ca_bundle:
                import httpx
                http_client = httpx.Client(verify=ca_bundle)
            self._client = anthropic.Anthropic(api_key=self._api_key, http_client=http_client)
        return self._client

    def extract(self, evidence_batch: tuple[EvidenceObject, ...], *, repair_hint: Optional[str] = None) -> dict:
        client = self._get_client()
        system_prompt = build_system_prompt()
        user_message = build_user_message(evidence_batch, repair_hint=repair_hint)
        tool_definition = {
            "name": EXTRACTION_TOOL_NAME,
            "description": (
                "Record the candidate observations extracted from the supplied "
                "account evidence, strictly conforming to the provided schema."
            ),
            "input_schema": TOP_LEVEL_SCHEMA,
        }
        t0 = time.monotonic()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool_definition],
                tool_choice={"type": "tool", "name": EXTRACTION_TOOL_NAME},
            )
        except Exception as e:  # provider/network/API failure of any kind
            self.call_log.append({
                "latency_ms": (time.monotonic() - t0) * 1000.0, "usage": None,
                "success": False, "error": str(e),
            })
            raise ModelServiceError(f"Anthropic extraction call failed: {e}", cause=e) from e

        latency_ms = (time.monotonic() - t0) * 1000.0
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == EXTRACTION_TOOL_NAME:
                self.call_log.append({"latency_ms": latency_ms, "usage": usage, "success": True, "error": None})
                return block.input  # already-parsed dict, per forced tool-use semantics

        self.call_log.append({
            "latency_ms": latency_ms, "usage": usage, "success": False,
            "error": "response did not contain the expected forced tool_use block",
        })
        raise ModelServiceError(
            "Anthropic response did not contain the expected forced tool_use block."
        )

    def propose_isolated_dimension_qualifier(
        self,
        dimension: DimensionCode,
        observation,
        *,
        repair_hint: Optional[str] = None,
    ) -> dict:
        """Milestone 4B isolated-classifier architecture checkpoint. A
        SEPARATE forced-tool-use request from extract() above, invoked
        once per eligible observation (never once per run) — its own
        tool name (DIMENSION_QUALIFIER_TOOL_NAME, reused across both
        dimensions since only the input_schema/prompt differ), its own
        dimension-specific input_schema (ISOLATED_DIMENSION_QUALIFIER_
        TOP_LEVEL_SCHEMA[...]), its own dimension-specific prompt
        (build_isolated_dimension_qualifier_system_prompt/_user_message).
        The request literally cannot contain any sibling observation's
        content — `observation` is the only observation-shaped argument
        this method receives. Logged to self.dimension_qualifier_call_log,
        kept entirely separate from self.call_log (stage-1's own
        instrumentation); one entry is appended per isolated call, so a
        multi-observation run produces multiple log entries here, exactly
        mirroring how many times this method was actually invoked."""
        client = self._get_client()
        qualifier_type = _DIMENSION_TO_QUALIFIER_OBSERVATION_TYPE[dimension]
        system_prompt = build_isolated_dimension_qualifier_system_prompt(dimension)
        user_message = build_isolated_dimension_qualifier_user_message(
            dimension, observation, repair_hint=repair_hint,
        )
        tool_definition = {
            "name": DIMENSION_QUALIFIER_TOOL_NAME,
            "description": (
                "Record at most one candidate D2 or D6 dimension-qualifier proposal, and/or "
                "zero or more independently-grounded atomic predicate proposals, for the "
                "single supplied already-accepted observation, strictly conforming to the "
                "provided schema."
            ),
            "input_schema": ISOLATED_DIMENSION_QUALIFIER_TOP_LEVEL_SCHEMA[qualifier_type],
        }
        t0 = time.monotonic()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool_definition],
                tool_choice={"type": "tool", "name": DIMENSION_QUALIFIER_TOOL_NAME},
            )
        except Exception as e:  # provider/network/API failure of any kind
            self.dimension_qualifier_call_log.append({
                "latency_ms": (time.monotonic() - t0) * 1000.0, "usage": None,
                "success": False, "error": str(e),
            })
            raise ModelServiceError(f"Anthropic isolated dimension-qualifier call failed: {e}", cause=e) from e

        latency_ms = (time.monotonic() - t0) * 1000.0
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == DIMENSION_QUALIFIER_TOOL_NAME:
                self.dimension_qualifier_call_log.append(
                    {"latency_ms": latency_ms, "usage": usage, "success": True, "error": None}
                )
                return block.input  # already-parsed dict, per forced tool-use semantics

        self.dimension_qualifier_call_log.append({
            "latency_ms": latency_ms, "usage": usage, "success": False,
            "error": "response did not contain the expected forced tool_use block",
        })
        raise ModelServiceError(
            "Anthropic isolated dimension-qualifier response did not contain the "
            "expected forced tool_use block."
        )

    @property
    def model_version(self) -> str:
        return self._model
