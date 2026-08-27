"""
Build Milestone 2 — Structured AI Extraction Layer.

Converts raw customer evidence text into traceable, schema-valid,
Current + Unverified candidate observations. This package NEVER computes
a governed CHDM conclusion — that remains the exclusive responsibility of
`engine.evaluate` (Milestone 1, unchanged). See each module's docstring
for its specific role in the pipeline:

  enums.py                  extraction-local controlled vocabularies
  schemas.py                 typed candidate-observation dataclasses
  json_schemas.py              model-facing JSON Schemas (semantic fields only)
  errors.py                     extraction-specific exception types
  validation.py                  schema / source-span / boundary validation
  dedup.py                        deterministic, audit-preserving deduplication
  provider.py                      isolated model-provider boundary
  prompts.py                        extraction instructions (data, not CHDM rules)
  pipeline.py                        the 11-step orchestration (CHDM v0.1 Milestone 2 spec §6)
  bridge_to_milestone1.py             TEST-ONLY mechanical adapter to Milestone 1 signals
"""
