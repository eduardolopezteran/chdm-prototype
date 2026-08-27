"""
I5 build block -- explanation-layer model-provider boundary. Mirrors
extraction/provider.py's own shape exactly (ABC + FakeExplanationProvider
deterministic test double + AnthropicExplanationProvider live
implementation, forced tool use, ModelServiceError on any failure). One
call generates both the explanation and the phrasing of the already
deterministically-selected/ranked questions (BUILD AUTHORIZED single-call
decision) -- see explanation/prompts.py for how the tool schema
structurally caps question count to the supplied gap-subject count.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from .errors import ModelServiceError
from .prompts import EXPLANATION_TOOL_NAME, build_explanation_tool_schema, build_system_prompt, build_user_message
from .schemas import GroundingPackage

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class ExplanationProvider(ABC):
    """The entire provider boundary for this package. `generate()` takes
    the already-assembled, closed GroundingPackage and returns the raw
    parsed JSON dict -- schema conformance is attempted by the provider but
    NOT assumed; explanation/pipeline.py always re-validates independently
    (grounding_check.py) regardless of provider, exactly mirroring
    extraction/pipeline.py's own "provider returns raw dict, caller
    re-validates independently" split."""

    @abstractmethod
    def generate(self, grounding_package: GroundingPackage) -> dict:
        """Returns a raw dict with explanation_text, explanation_cited_object_ids,
        and question_texts (one per grounding_package.gap_subjects, in order).
        Raises ModelServiceError on any provider/API/network failure -- never
        returns a partial or fabricated result on failure."""
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    @property
    def model_version(self) -> str:
        return "unknown"


class FakeExplanationProvider(ExplanationProvider):
    """Deterministic, no-network test double -- same construction shape as
    extraction.provider.FakeExtractionProvider: a dict/list (cycled) or a
    callable(grounding_package) -> dict. No real network call ever occurs
    in the deterministic test suite."""

    def __init__(
        self,
        responses: Union[dict, list, Callable[[GroundingPackage], dict]],
        *,
        model_version: str = "fake-explanation-provider-v1",
        raise_service_error: bool = False,
    ):
        self._responses = responses
        self._call_count = 0
        self._model_version = model_version
        self._raise_service_error = raise_service_error
        self.call_log: list[dict] = []

    def generate(self, grounding_package: GroundingPackage) -> dict:
        self._call_count += 1
        if self._raise_service_error:
            self.call_log.append({"latency_ms": 0.0, "usage": None, "success": False, "error": "simulated"})
            raise ModelServiceError("FakeExplanationProvider configured to simulate a service failure.")
        if callable(self._responses) and not isinstance(self._responses, (dict, list)):
            response = self._responses(grounding_package)
        elif isinstance(self._responses, list):
            idx = min(self._call_count - 1, len(self._responses) - 1)
            response = self._responses[idx]
        else:
            response = self._responses
        self.call_log.append({"latency_ms": 0.0, "usage": None, "success": True, "error": None})
        return response

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def model_version(self) -> str:
        return self._model_version


class AnthropicExplanationProvider(ExplanationProvider):
    """Live provider. Not exercised by the deterministic suite -- execution
    is deferred until ANTHROPIC_API_KEY is available, exactly mirroring
    extraction.provider.AnthropicExtractionProvider's own deferral
    contract. Not exercised anywhere in this checkpoint; live evaluation is
    explicitly gated behind separate review per BUILD AUTHORIZED."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_ANTHROPIC_MODEL):
        self._api_key = api_key
        self._model = model
        self._client = None
        self.call_log: list[dict] = []

    def _get_client(self):
        if self._client is None:
            import os
            try:
                import anthropic
            except ImportError as e:
                raise ModelServiceError(
                    "The 'anthropic' package is not installed. Install it "
                    "(`pip install anthropic`) to use AnthropicExplanationProvider.",
                    cause=e,
                ) from e
            ca_bundle = os.environ.get("ANTHROPIC_CA_BUNDLE")
            http_client = None
            if ca_bundle:
                import httpx
                http_client = httpx.Client(verify=ca_bundle)
            self._client = anthropic.Anthropic(api_key=self._api_key, http_client=http_client)
        return self._client

    def generate(self, grounding_package: GroundingPackage) -> dict:
        client = self._get_client()
        system_prompt = build_system_prompt()
        user_message = build_user_message(grounding_package)
        tool_definition = {
            "name": EXPLANATION_TOOL_NAME,
            "description": (
                "Record the grounded explanation and the phrasing of the "
                "pre-selected diagnostic questions, strictly conforming to "
                "the provided schema."
            ),
            "input_schema": build_explanation_tool_schema(len(grounding_package.gap_subjects)),
        }
        t0 = time.monotonic()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool_definition],
                tool_choice={"type": "tool", "name": EXPLANATION_TOOL_NAME},
            )
        except Exception as e:
            self.call_log.append({
                "latency_ms": (time.monotonic() - t0) * 1000.0, "usage": None,
                "success": False, "error": str(e),
            })
            raise ModelServiceError(f"Anthropic explanation call failed: {e}", cause=e) from e

        latency_ms = (time.monotonic() - t0) * 1000.0
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == EXPLANATION_TOOL_NAME:
                self.call_log.append({"latency_ms": latency_ms, "usage": usage, "success": True, "error": None})
                return block.input

        self.call_log.append({
            "latency_ms": latency_ms, "usage": usage, "success": False,
            "error": "response did not contain the expected forced tool_use block",
        })
        raise ModelServiceError(
            "Anthropic response did not contain the expected forced tool_use block."
        )

    @property
    def model_version(self) -> str:
        return self._model
