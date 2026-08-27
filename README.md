# CHDM Engine — Build Milestone 1 + Milestone 2A

Deterministic core of the Evidence-Grounded Customer Health Diagnostic
(Milestone 1), plus a bounded AI extraction layer (Milestone 2) that
converts raw account text into traceable, schema-valid,
Current + Unverified candidate observations. The deterministic core has
zero LLM dependency; Milestone 2's LLM layer never computes a governed
CHDM conclusion — see `extraction/README` notes below and each module's
docstring.

## Governing baselines (authority order)

1. **CHDM v0.1** — Frozen Governing Methodology Baseline (canonical reconstruction, as supplied to the build conversation)
2. **MVP Product Functionality Specification v1.0 + OFD-01**
3. **UX/Product Design Baseline v1.0 + UXD-01**
4. **Technical Solution Architecture v1.0**
5. **BAR-01** — UX–Technical Architecture Reconciliation & Build Authorization Record

Do not redesign these baselines while coding. If implementation exposes a
contradiction or missing governing rule, stop that specific code path and
report it — do not infer.

## Known non-blocking constraint: UC-01

CHDM v0.1 §12.3/§19 leaves the general deterministic effect of
contradictory Current+Confirmed evidence on **dimensions other than D1**
unresolved (`registry/contradiction_rules.yaml` →
`unresolved_general_contradiction_rule`). This does not block Milestone 1:
none of S1–S6 requires it. The engine must raise an explicit
`UnresolvedMethodologyError` if that code path is ever reached rather than
guess a Dimension State.

## Status

- [x] Methodology Registry (`registry/*.yaml`) — 16 files, CHDM v0.1 §1–§19
- [x] Canonical domain objects/enums (`domain/`)
- [x] Registry loader + structural/exhaustiveness validation (`engine/registry_loader.py`)
- [x] Deterministic rules engine — Objective / Dimensions (D1, D2, D6) / Risks (CR-01,02,03,06,08) / DMEG / Reliability / Operational Priority / Evidence Review / invariants / top-level `evaluate()` / basic trace generation
- [x] DMEG priority-elevation refinement — `DMEGLinkedConclusion` decision-dependency metadata; OP evaluator consumes `DMEG.affects_operational_priority` instead of `len(dmegs)>0`
- [x] Scenario Lab fixtures S1–S6 (`scenarios/*.yaml`) — static data, declared expected outputs
- [x] Checkpoint 3 correction 1 — D1 contradiction (Disputed) now correctly evaluated for DMEG-1/2/3 via genuine differential re-computation (`engine/evaluate.py` step 4c), not assumed resolved by the D1 rule alone
- [x] Checkpoint 3 correction 2 — confirmation state is authoritative once per evidence item (`scenarios/*.yaml` `evidence:` block); loader rejects signal/claim-level `evidence_state` duplication
- [x] Full end-to-end regression suite — 97/97 tests passing

## Build Milestone 1 Exit Criteria: PASS

## Build Milestone 2A — Structured AI Extraction Layer

Additive only — zero modifications to `registry/`, `domain/`, `engine/`,
or `scenarios/`. The AI extraction layer may identify candidate facts; it
can never determine a CHDM outcome. Every accepted observation enters as
`Current + Unverified`; system-generated metadata (provider, model
version, timestamp, trace id, evidence state) is attached exclusively by
`extraction/pipeline.py` — the model itself has no field through which it
can set or request any of that, or any governed CHDM value.

- [x] Extraction enums + typed schemas (`extraction/enums.py`, `extraction/schemas.py`) — 7 span-grounded observation types + `MissingInformationCandidate` (evidence-scope-relative) + `CandidateContradiction` (never resolved)
- [x] Model-facing JSON Schemas, semantic fields only (`extraction/json_schemas.py`) — `additionalProperties: false` structurally blocks system-metadata and governed-CHDM-field injection; explicit denylist scan classifies boundary-violation attempts precisely
- [x] Deterministic source-span validation (`extraction/validation.py`) — exact character-offset resolution against the referenced Evidence Object, no fuzzy fallback
- [x] Deterministic, audit-preserving deduplication (`extraction/dedup.py`)
- [x] Provider boundary (`extraction/provider.py`) — `ExtractionProvider` ABC, `FakeExtractionProvider` (deterministic, powers all Checkpoint 2A tests), `AnthropicExtractionProvider` (forced tool-use; implemented, execution deferred until an API key is available)
- [x] Extraction pipeline (`extraction/pipeline.py`) — all 11 steps, one repair/retry on malformed top-level output, per-item rejection never silently dropped
- [x] Test-only Milestone 1 bridge (`extraction/bridge_to_milestone1.py`) — mechanical field copy only, `evidence_state` always taken from the observation itself, never confirmable through this path
- [x] Synthetic 15-case labeled evaluation set (`eval/labeled_set.yaml`)
- [x] Deterministic extraction test suite — 61 tests, no network calls
- [x] Full regression (Milestone 1 + Milestone 2A) — 158/158 passing

Deferred to Checkpoint 2B: live model execution, `eval/metrics.py` /
`eval/run_eval.py`, `tests/test_extraction_live_model.py`.

## Layout

```
registry/    Canonical CHDM v0.1 methodology, as versioned YAML config (data, not code)
domain/      Canonical objects/enums the engine consumes and produces
engine/      Deterministic evaluation logic (pure functions) — sole authority for governed outputs
scenarios/   Synthetic Scenario Lab fixtures (YAML data only, no executable logic)
extraction/  Milestone 2 AI extraction layer (schemas, validation, provider boundary, pipeline)
eval/        Milestone 2 synthetic labeled evaluation set
tests/       Automated regression suite (Milestone 1 + Milestone 2A)
```
