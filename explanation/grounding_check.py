"""
I5 build block -- post-generation validation (Technical Architecture §12
safeguard table; TAC-13). Two independent checks, both defense-in-depth
over the schema (the PRIMARY boundary mechanism is
prompts.build_explanation_tool_schema()'s additionalProperties:false +
exact question-count cap -- exactly the same "schema is primary, denylist
scan is defense-in-depth" discipline as extraction/json_schemas.py and
extraction/validation.py.scan_for_prohibited_keys()):

1. Citation-linking (TAC-13): every object ID the model declares it relied
   on, AND every ID-shaped token found anywhere in its free text, must
   already exist in the GroundingPackage's known_object_ids. Anything else
   fails closed -- explanation/pipeline.py routes to the deterministic
   fallback rather than show an uncited or invented claim.

2. Prohibited-content scan (FR-19.1): pattern-based rejection of
   predictive/causal, prescriptive, contradiction-resolving, and
   confirmation-status-assigning language. Necessarily imperfect (the
   Technical Architecture's own §15 already flags this evaluation track as
   "necessarily imperfect... supplemented by periodic human spot-review")
   -- this is a pattern scan, not a semantic classifier, and is not
   proposed as a complete solution, only a bounded, testable, fail-closed
   gate consistent with what was authorized.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import GroundingPackage

# Matches this codebase's own governed-ID surface forms, observed directly
# in domain/*.py and the live 4C validation output: DMEG-0002, CR-08,
# CHDM-DIM-ADOPTION-001, OP1/OP2/OP3/OPU, ER1/ER0, D1..D8B,
# ER-DMEG-RISK-MATERIAL, ACHIEVED/DISPUTED/etc. state tokens are handled
# separately by the exact-set membership check, not this regex -- this
# regex only needs to catch structured ID-LIKE tokens (dash-joined,
# uppercase-led), the shape governed IDs actually take.
_ID_LIKE_TOKEN = re.compile(
    r"\b(?:"
    r"DMEG-\d+"
    r"|CR-0[1-8]"
    r"|OP[123U]\b"
    r"|ER[01]\b"
    r"|D[1-8][AB]?\b"
    r"|CHDM-[A-Z0-9]+(?:-[A-Z0-9]+)+"
    r"|ER-DMEG-[A-Z-]+"
    r")"
)

_PROHIBITED_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("prescriptive_recommendation", re.compile(r"\b(should|recommend(?:s|ed|ation)?|must (?:act|reach out|escalate|intervene))\b", re.IGNORECASE)),
    ("predictive_language", re.compile(r"\b(predict(?:s|ed|ion)?|forecast(?:s|ed)?|probability|likelihood|likely to (?:churn|renew|leave|stay)|going to churn|will churn|expected to churn)\b", re.IGNORECASE)),
    ("contradiction_resolution", re.compile(r"\b(resolves? the contradiction|contradiction is resolved|correctly resolv\w*|the correct (?:answer|version|source) is)\b", re.IGNORECASE)),
    ("confirmation_status_assignment", re.compile(r"\b(is now confirmed|we confirm|has been confirmed|has been rejected|i confirm|i reject)\b", re.IGNORECASE)),
    ("severity_characterization", re.compile(r"\b(more severe than|less severe than|actually (?:critical|material|watch))\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class CitationLinkingResult:
    ungrounded_declared_ids: tuple[str, ...]
    ungrounded_text_tokens: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.ungrounded_declared_ids and not self.ungrounded_text_tokens


@dataclass(frozen=True)
class ProhibitedContentResult:
    matches: tuple[tuple[str, str], ...]   # (pattern_name, matched_text)

    @property
    def is_clean(self) -> bool:
        return not self.matches


def check_citation_linking(
    text: str,
    declared_cited_ids: list[str],
    grounding_package: GroundingPackage,
) -> CitationLinkingResult:
    known = grounding_package.known_object_ids
    ungrounded_declared = tuple(sorted({i for i in declared_cited_ids if i not in known}))
    found_tokens = set(_ID_LIKE_TOKEN.findall(text))
    ungrounded_tokens = tuple(sorted(t for t in found_tokens if t not in known))
    return CitationLinkingResult(
        ungrounded_declared_ids=ungrounded_declared,
        ungrounded_text_tokens=ungrounded_tokens,
    )


def check_prohibited_content(text: str) -> ProhibitedContentResult:
    matches = []
    for name, pattern in _PROHIBITED_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((name, m.group(0)))
    return ProhibitedContentResult(matches=tuple(matches))
