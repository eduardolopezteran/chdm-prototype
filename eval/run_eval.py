#!/usr/bin/env python3
"""
Build Milestone 2B — standalone live-model evaluation runner.

RUN THIS OUTSIDE any network-restricted sandbox, wherever
ANTHROPIC_API_KEY has real egress to api.anthropic.com. It needs nothing
beyond this repository plus three PyPI packages.

Usage:
    cd chdm-engine
    pip install anthropic pyyaml jsonschema
    export ANTHROPIC_API_KEY=sk-ant-...      # never pasted into any file by this script
    python3 eval/run_eval.py \\
        --model claude-haiku-4-5-20251001 \\
        --label prompt_v4_optionb_2c_eval2 \\
        --out eval/results/prompt_v4_optionb_2c_eval2.json

Milestone 2B.2 closure note: this script has no per-case subset flag, so
the closure validation (Case 12 CandidateContradiction integrity, Case
20 as a second contradiction control, Case 21 inference-boundary
verification) runs against the FULL 23-case set, exactly like every
prior run, to preserve comparability. Nothing about Prompt v2, the
model, or the benchmark changed in this closure pass -- only the
CandidateContradiction integrity rule (extraction/pipeline.py) and the
Case 21 evaluator cross-reference fix (eval/metrics.py) did.

What it does (spec §1-§3, §10; scoring per Milestone 2B.1; benchmark and
prompt per Milestone 2B.2 OC-01 ontology clarification):
  1. Loads eval/labeled_set.yaml (23 synthetic cases as of Milestone
     2B.2 -- unmodified by this script).
  2. For each case, builds the EvidenceObject(s) and runs the SAME,
     UNCHANGED extraction.pipeline.run_extraction() used everywhere else
     in this repo, against a real AnthropicExtractionProvider. This
     script never alters model, prompt, schema, or extraction behavior
     -- it only changed how the SAME output is scored (Milestone 2B.1
     spec §11).
  3. Records per-case: provider, model, prompt version, call count
     (retries), latency, token usage (if the SDK returns it), validation
     failures, accepted/rejected counts, and the Milestone 2B.1 six-way
     classification + three-way precision + three-way recall
     (eval/metrics.py).
  4. Writes ONE JSON report to --out. No API key or other secret is ever
     written to that file or printed to stdout.

This script does not modify eval/labeled_set.yaml, does not modify any
Milestone 1 or Milestone 2A/2B file, and does not touch the deterministic
test suite. It is intentionally a thin, direct script — no LangChain, no
agent loop, one bounded request per case.

Milestone 2C update: labeled_set.yaml grew to 33 cases (24-33 exercise
candidate_risk_signals / candidate_evidence_classifications); Prompt v3
adds the corresponding guidance; score_case additionally scores both new
output types via eval.metrics.classify_candidate_classifications and
records their counts/details in the per-case report. Nothing about how
the 8 Milestone 2B types or contradictions are scored changed.

Milestone 4B update: labeled_set.yaml grew to 42 cases (34-42 exercise
the D2/D6 candidate-qualifier channels). For EVERY case, this script now
also runs extraction.pipeline.run_dimension_qualifier_classification
immediately after run_extraction -- a SEPARATE stage-2 call, mirroring
production exactly (stage 2 always runs once stage 1 finishes, whenever
there is something eligible to classify; its own skip-if-nothing-
eligible logic makes this a no-op for cases with no AdoptionObservation/
StakeholderObservation). score_case additionally scores
candidate_d2_qualifiers / candidate_d6_qualifiers via eval.metrics.
classify_dimension_qualifiers and records dimension_qualifier_stage_
failure, counts, and details in the per-case report -- kept structurally
distinct from request_failure throughout (see extraction.pipeline.
run_dimension_qualifier_classification's own docstring for the full
abstention-vs-unavailable contract this preserves end to end into the
evaluation report). Nothing about how the 8 Milestone 2B types,
contradictions, or the 2 Milestone 2C candidate-classification types are
scored changed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from domain.enums import Provenance
from domain.evidence import EvidenceObject

from extraction.enums import RejectionReason
from extraction.pipeline import run_dimension_qualifier_classification, run_extraction
from extraction.prompts import DIMENSION_QUALIFIER_PROMPT_VERSION, PROMPT_VERSION
from extraction.provider import AnthropicExtractionProvider

from eval.metrics import (
    aggregate_metrics, classify_accepted_observations, classify_candidate_classifications,
    classify_dimension_qualifiers, dimension_qualifier_atomic_predicate_detail,
    score_missing_information, _obs_summary,
)

LABELED_SET_PATH = ROOT / "eval" / "labeled_set.yaml"


def load_cases() -> list[dict]:
    with open(LABELED_SET_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["cases"]


def build_evidence_for_case(case: dict) -> tuple[EvidenceObject, ...]:
    if "source_text_a" in case:
        return (
            EvidenceObject("EA", None, case["source_text_a"].strip(), "synthetic_note", Provenance.USER_PROVIDED),
            EvidenceObject("EB", None, case["source_text_b"].strip(), "synthetic_note", Provenance.USER_PROVIDED),
        )
    return (EvidenceObject("E1", None, case["source_text"].strip(), "synthetic_note", Provenance.USER_PROVIDED),)


def score_case(
    case: dict, result, provider, calls_before: int,
    *, dq_calls_before: int = 0,
) -> dict:
    calls_made = len(provider.call_log) - calls_before
    call_slice = provider.call_log[calls_before:calls_before + calls_made] if calls_made else []
    # Milestone 4B: the stage-2 D2/D6 classifier's OWN call instrumentation,
    # kept entirely separate from stage-1's calls_made/call_slice above --
    # see extraction.provider.FakeExtractionProvider/AnthropicExtractionProvider's
    # own dimension_qualifier_call_log, which never mixes with call_log.
    dq_calls_made = len(provider.dimension_qualifier_call_log) - dq_calls_before
    dq_call_slice = (
        provider.dimension_qualifier_call_log[dq_calls_before:dq_calls_before + dq_calls_made]
        if dq_calls_made else []
    )

    # Milestone 2B.1: six-way classification replaces the old single
    # matched/missed/unsupported_extra split (eval/metrics.py docstring
    # has the full rationale). `classification["slots"]` also carries the
    # final claimed/unclaimed state of every labeled expectation, which
    # aggregate_metrics uses to compute primary/required/optional-valid
    # recall separately.
    classification = classify_accepted_observations(case, result.accepted)
    missing_information = score_missing_information(case, result.accepted)
    # Milestone 2C: separate scorer for the two candidate-classification
    # output types (eval/metrics.py docstring has the full rationale for
    # why this is not folded into the six-way classifier above).
    candidate_classification = classify_candidate_classifications(
        case, result.accepted, result.candidate_risk_signals, result.candidate_evidence_classifications,
    )
    # Milestone 4B: scores the SEPARATE stage-2 D2/D6 classifier output.
    # `result` here is expected to already be the POST-stage-2 result
    # (main()'s loop runs run_dimension_qualifier_classification before
    # calling score_case) -- candidate_d2_qualifiers/candidate_d6_
    # qualifiers/dimension_qualifier_stage_failure all default to "stage
    # 2 did not run" if a caller passes a stage-1-only result, so this
    # never raises even against historical stage-1-only call sites.
    dimension_qualifier_classification = classify_dimension_qualifiers(
        case, result.accepted, result.candidate_d2_qualifiers, result.candidate_d6_qualifiers,
    )

    span_rejections = sum(1 for r in result.rejected if r.reason == RejectionReason.SPAN_NOT_FOUND)
    unknown_evidence_rejections = sum(1 for r in result.rejected if r.reason == RejectionReason.UNKNOWN_EVIDENCE_ID)
    boundary_attempted = sum(1 for r in result.rejected if r.reason == RejectionReason.BOUNDARY_VIOLATION)
    # Structural guarantee (Checkpoint 2A): nothing in `accepted` can carry
    # a governed-CHDM field or a model-set system field — there is no such
    # attribute on any accepted dataclass to even check. Recorded as 0 by
    # construction, not by scanning (there is nothing to scan for).
    boundary_accepted = 0

    span_grounded_accepted = [o for o in result.accepted if hasattr(o, "source_span")]

    case_result = {
        "case_id": case["id"],
        "request_failure": result.request_failure,
        # Milestone 4B: distinct from request_failure -- see
        # extraction.pipeline.run_dimension_qualifier_classification's
        # docstring for the full graceful-degradation/abstention-vs-
        # unavailable contract this field preserves.
        "dimension_qualifier_stage_failure": result.dimension_qualifier_stage_failure,
        # Milestone 4B isolated-classifier architecture checkpoint (item
        # D): the single authoritative, per-observation failure record --
        # dimension_qualifier_stage_failure above is a derived, human-
        # readable summary of the SAME data, kept for compatibility. This
        # list is what makes the isolated architecture's failure
        # isolation actually auditable from a report: which SPECIFIC
        # observation(s) the classifier was unavailable for, never just
        # "the stage-2 call failed" undifferentiated -- one entry per
        # DimensionQualifierFailure, never conflated with a rejected
        # candidate (a successfully-returned-but-ungroundable answer is a
        # different thing and stays in rejected_detail, unchanged).
        "dimension_qualifier_failures_detail": [
            {
                "resolved_observation_id": f.resolved_observation_id,
                "dimension": f.dimension.value if hasattr(f.dimension, "value") else f.dimension,
                "detail": f.detail,
            }
            for f in result.dimension_qualifier_failures
        ],
        # Milestone 4B v3 evaluator-provenance checkpoint: per-predicate
        # audit detail for the atomic-predicate + deterministic-
        # composition architecture -- see eval.metrics.
        # dimension_qualifier_atomic_predicate_detail's own docstring for
        # the full field-by-field contract. Read-only over already-
        # produced pipeline output; does not change model behavior, the
        # composer, or which candidates end up in candidate_d2_qualifiers
        # / candidate_d6_qualifiers -- only makes that existing decision
        # inspectable per predicate instead of only at the final,
        # composed-candidate level.
        "dimension_qualifier_atomic_predicate_detail": dimension_qualifier_atomic_predicate_detail(
            result.dimension_qualifier_predicate_evidence, result.rejected,
            result.candidate_d2_qualifiers, result.candidate_d6_qualifiers,
        ),
        "schema": {
            "calls_made": calls_made,
            "call_log": call_slice,
        },
        "dimension_qualifier_schema": {
            "calls_made": dq_calls_made,
            "call_log": dq_call_slice,
        },
        "counts": {
            "accepted": len(result.accepted),
            "rejected": len(result.rejected),
            "candidate_contradictions": len(result.candidate_contradictions),
            "candidate_risk_signals": len(result.candidate_risk_signals),
            "candidate_evidence_classifications": len(result.candidate_evidence_classifications),
            "candidate_d2_qualifiers": len(result.candidate_d2_qualifiers),
            "candidate_d6_qualifiers": len(result.candidate_d6_qualifiers),
            "dedup_collapsed": len(result.dedup_audit),
        },
        "classification": classification,
        "missing_information": missing_information,
        "candidate_classification": candidate_classification,
        "dimension_qualifier_classification": dimension_qualifier_classification,
        "source_grounding": {
            "accepted_with_valid_span": len(span_grounded_accepted),
            "span_rejections": span_rejections,
            "unknown_evidence_rejections": unknown_evidence_rejections,
        },
        "boundary": {
            "attempted": boundary_attempted,
            "accepted": boundary_accepted,
        },
        "rejected_detail": [
            {
                "observation_type": r.observation_type, "reason": r.reason.value, "detail": r.detail,
                # Milestone 2B: raw_item is the candidate JSON the model
                # actually emitted for this rejected entry (no secrets —
                # it is only ever extraction-candidate content). Included
                # so grounding/schema failures can be diagnosed directly
                # from the report instead of needing a code-level rerun.
                "raw_item": r.raw_item,
            }
            for r in result.rejected
        ],
        "accepted_detail": [_obs_summary(o) for o in result.accepted],
        "candidate_contradictions_detail": [_obs_summary(c) for c in result.candidate_contradictions],
        "candidate_risk_signals_detail": [_obs_summary(c) for c in result.candidate_risk_signals],
        "candidate_evidence_classifications_detail": [_obs_summary(c) for c in result.candidate_evidence_classifications],
        "candidate_d2_qualifiers_detail": [_obs_summary(c) for c in result.candidate_d2_qualifiers],
        "candidate_d6_qualifiers_detail": [_obs_summary(c) for c in result.candidate_d6_qualifiers],
        "dedup_audit_detail": [
            {"duplicate_observation_id": a.duplicate_observation_id,
             "canonical_observation_id": a.canonical_observation_id,
             "observation_type": a.observation_type, "reason": a.reason}
            for a in result.dedup_audit
        ],
    }

    # Case-specific checks, triggered by what the label actually declares
    # rather than a hardcoded case-id prefix, so cases 16-20 (Milestone
    # 2B.1) are covered automatically without editing this function again.
    if case.get("missing_information_expected"):
        case_result["missing_information_check"] = missing_information
    if case.get("expected_contradiction", {}).get("required"):
        c = result.candidate_contradictions
        case_result["contradiction_check"] = {
            "detected": len(c) >= 1,
            "count": len(c),
            "status_all_candidate": all(x.status == "CANDIDATE" for x in c),
            "both_sides_resolved": all(x.resolved_observation_id_a and x.resolved_observation_id_b for x in c),
            "both_observations_extracted": sum(
                1 for entry in classification["classified"]
                if entry["classification"] in ("MATCHED_EXPECTED", "VALID_UNLABELED")
            ) >= 2,
        }
    if case.get("expected_observations") == [] and not case.get("missing_information_expected") \
            and not case.get("inferred_candidates_permitted"):
        case_result["empty_extraction_check"] = {
            "accepted_count": len(result.accepted), "expected": 0,
            "grounded_but_irrelevant_permitted_count": len(case.get("grounded_but_irrelevant_permitted", [])),
        }
    # Milestone 2C anti-fabrication check (case 32): triggered whenever a
    # case explicitly declares BOTH candidate-classification lists empty,
    # not by a hardcoded case-id, so any future negative case is covered
    # automatically.
    if case.get("expected_candidate_risk_signals") == [] and case.get("expected_candidate_evidence_classifications") == []:
        case_result["candidate_classification_negative_check"] = {
            "candidate_risk_signals_found": len(result.candidate_risk_signals),
            "candidate_evidence_classifications_found": len(result.candidate_evidence_classifications),
            "expected": 0,
        }
    if case.get("expected_dimension_d2_qualifiers") == [] and case.get("expected_dimension_d6_qualifiers") == []:
        case_result["dimension_qualifier_negative_check"] = {
            "candidate_d2_qualifiers_found": len(result.candidate_d2_qualifiers),
            "candidate_d6_qualifiers_found": len(result.candidate_d6_qualifiers),
            "expected": 0,
        }
    if case.get("inferred_candidates_permitted"):
        inferred = [o for o in result.accepted if getattr(o, "basis", None) and o.basis.value == "INFERRED_CANDIDATE"]
        explicit = [o for o in result.accepted if getattr(o, "basis", None) and o.basis.value == "EXPLICIT"]
        case_result["inference_boundary_check"] = {
            "inferred_candidates_found": [_obs_summary(o) for o in inferred],
            "explicit_observations_found": [_obs_summary(o) for o in explicit],
        }

    return case_result


# Milestone 2B.1 spec §9: baseline_v1 and baseline_v1_fix1 are historical
# evaluation artifacts, scored against the PRIOR single-span labeled set.
# Re-running under those exact labels would silently produce a
# same-named-but-not-comparable report; refuse rather than risk that.
#
# baseline_v1_fix1_eval2 was added here in Milestone 2B.2: it is now also
# a completed historical run (scored against the PRIOR 20-case labeled
# set and Prompt v1) that this milestone's decision gate compares Prompt
# v2 against. This addition follows the same principle as the two labels
# above but was not itself explicitly named for protection in the
# Milestone 2B.2 authorization -- flagged here and in the accompanying
# report rather than done silently.
#
# prompt_v2_ontology_eval1 added during Milestone 2B.2 closure: it is now
# the completed run this closure's CandidateContradiction-integrity /
# Case 21 evaluator fixes are compared against. Same principle, same
# disclosure-not-silence approach as baseline_v1_fix1_eval2 above.
#
# prompt_v2_closure_eval2 added at Milestone 2B final closure: the
# successful re-run after the MissingInformationCandidate dedup crash fix
# (the prior failed attempt was never counted under
# prompt_v2_closure_eval1, per explicit instruction at the time). This is
# the run Milestone 2B's PASS decision gate is based on and must not be
# silently overwritten by a Milestone 2C label collision.
#
# prompt_v3_2c_eval1 added at the Milestone 2C live decision gate: the
# run the CONDITIONAL verdict (and the resulting Prompt v3.1 narrow
# refinement -- OC-01/OC-03 reinforcement + risk-mechanism non-force-fit
# rule) is based on. Must not be silently overwritten by a future run.
#
# prompt_v3_refine1_2c_eval2 added at the second Milestone 2C live decision
# gate: the run the second CONDITIONAL verdict (and the resulting Prompt
# v3.2 narrow refinement -- risk-signal non-suppression complement +
# AdoptionObservation/StrategicObservation symmetric protection) is based
# on. Same disclosure-not-silence approach as every prior addition to this
# set. Must not be silently overwritten by a future run.
#
# prompt_v3_refine2_2c_eval3 added at the third Milestone 2C live decision
# gate: the run the third CONDITIONAL verdict (case 26 CR-03 never once
# correct across three rounds; case 27 newly confused CR-08 with CR-03) is
# based on, and the empirical basis for the PMO Option B decision (defer
# CR-03 from AI-automated candidate classification; Prompt v4). Same
# disclosure-not-silence approach as every prior addition to this set.
# Must not be silently overwritten by a future run.
#
# prompt_v4_4b_dimqual_eval1 added at the Milestone 4B live decision gate:
# the run the CONDITIONAL verdict (stage-2 D2/D6 over-classification;
# 0.8077 D2 / 0.6667 D6 false-candidate rate) is based on, and the
# empirical basis for the Prompt v1.1 calibration (single-supporting-
# observation provenance rule + named abstention patterns). Same
# disclosure-not-silence approach as every prior addition to this set.
# Must not be silently overwritten by a future run.
_PROTECTED_LABELS = {
    "baseline_v1", "baseline_v1_fix1", "baseline_v1_fix1_eval2", "prompt_v2_ontology_eval1",
    "prompt_v2_closure_eval2", "prompt_v3_2c_eval1", "prompt_v3_refine1_2c_eval2",
    "prompt_v3_refine2_2c_eval3", "prompt_v4_4b_dimqual_eval1",
}


def select_cases(cases: list[dict], case_ids_arg: Optional[str]) -> tuple[list[dict], Optional[list[str]]]:
    """Milestone 4B D2 atomic-predicate targeted live probe: pure,
    directly-testable filter logic for --case-ids, factored out of main()
    so it can be exercised deterministically without constructing an
    AnthropicExtractionProvider or making any network call. Returns
    (selected_cases, case_id_prefixes) -- case_id_prefixes is None when
    case_ids_arg is None/empty (full-benchmark run, every case returned
    unfiltered, in original order).

    Bugfix (Milestone 4B D2 targeted-probe harness defect, found via the
    real prompt_v4_4b_dimqual_v3_atomic_probe1 run): the original version
    of this function only raised SystemExit when the ENTIRE filter
    matched nothing. That let a single mistyped prefix (e.g. "7" instead
    of the zero-padded "07" every case id actually uses) silently match
    zero cases while OTHER prefixes in the same --case-ids argument still
    matched something -- Case 07 was silently dropped from what was
    supposed to be a 3-case probe (07,23,35), and the run proceeded as a
    2-case run with no error or warning anywhere. This function now
    validates EACH prefix independently and raises SystemExit naming
    every prefix that matched zero cases, before returning anything, so
    a typo/zero-padding mistake fails loudly and immediately instead of
    silently narrowing the run. Case-id matching itself (startswith, in
    original order, comma-split, whitespace-tolerant) is unchanged --
    "07" already correctly resolved case 07 before this fix and still
    does; only the validation gate is new."""
    if not case_ids_arg:
        return cases, None
    prefixes = [p.strip() for p in case_ids_arg.split(",") if p.strip()]
    unmatched = [p for p in prefixes if not any(c["id"].startswith(p) for c in cases)]
    if unmatched:
        raise SystemExit(
            f"--case-ids {case_ids_arg!r}: prefix(es) {unmatched!r} matched NO case in "
            f"{LABELED_SET_PATH}. Case ids are zero-padded (e.g. '07_...', not '7_...') -- "
            f"check for a missing leading zero. Refusing to run a silently-narrowed subset."
        )
    selected = [c for c in cases if any(c["id"].startswith(p) for p in prefixes)]
    if not selected:
        raise SystemExit(f"--case-ids {case_ids_arg!r} matched no cases in {LABELED_SET_PATH}.")
    return selected, prefixes


def _print_atomic_predicate_table(case_id: str, case_result: dict) -> None:
    """Milestone 4B D2 atomic-predicate targeted live probe (diagnostic
    checkpoint): human-readable console table over the SAME
    dimension_qualifier_atomic_predicate_detail data already written to
    the JSON report -- nothing computed here that isn't already in the
    report; this is purely a readability aid for a --case-ids diagnostic
    run, where reading raw JSON for 1-3 cases is slower than a table.
    Groups entries by (resolved_observation_id, dimension) and calls out
    explicitly which complete, grounded predicate set (if any) is what
    caused a compound qualifier to compose for that observation."""
    entries = case_result.get("dimension_qualifier_atomic_predicate_detail") or []
    if not entries:
        print(f"    (no atomic predicate activity for {case_id})")
        return
    by_obs: dict = {}
    for e in entries:
        key = (e["resolved_observation_id"], e["dimension"])
        by_obs.setdefault(key, []).append(e)
    for (obs_id, dim), group in by_obs.items():
        print(f"    observation {obs_id} (dimension {dim}):")
        for e in group:
            status = "GROUNDED" if e["grounding_passed"] else f"REJECTED ({e['rejection_reason']})"
            composed_flag = " [PART OF COMPOSED SET]" if e["composed"] else ""
            print(f"      - {e['predicate_id']}: {status}{composed_flag}")
            print(f"          evidence_text: {e['evidence_text']!r}")
            print(f"          basis: {e['basis']}")
        if any(e["composed"] for e in group):
            composing_ids = [e["predicate_id"] for e in group if e["composed"]]
            print(f"      => COMPLETE GROUNDED SET that composed the compound qualifier: {composing_ids}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Factored out of main() (Milestone 4B D2 atomic-predicate targeted
    live-probe follow-up) so the REAL, PRODUCTION argparse configuration
    -- the exact same parser main() uses, not a reimplementation or a
    fixture -- can be exercised end-to-end in a deterministic test:
    argv -> parser.parse_args(argv) -> args.case_ids -> select_cases(),
    with no network call and no AnthropicExtractionProvider construction
    in the way. This exists specifically to let a test prove that
    argparse itself never touches, casts, or normalizes --case-ids in
    any way (no type=int, no int(...), no lstrip("0"), no zfill -- see
    the --case-ids argument below, defined with no `type=` at all, so
    argparse stores the raw string exactly as received in sys.argv).

    Diagnostic note (case-id zero-padding follow-up report): a full
    inspection of this module found exactly ONE definition of
    --case-ids (this one), no other argparse configuration for it
    anywhere in the repo, no wrapper .bat/.ps1/.cmd/.sh scripts that
    invoke this file, and no int()/lstrip("0")/zfill/str(int(...)) call
    anywhere near case-id handling. select_cases() itself (below) also
    never casts to int -- it only calls .split(",") and .strip() on the
    string. This function's own test (test_run_eval_cli.py) proves
    argv=["--case-ids", "07,23,35"] produces args.case_ids ==
    "07,23,35" exactly, byte-for-byte, and that select_cases() then
    resolves it to prefixes == ["07", "23", "35"]. If a leading zero is
    still being lost before python3 ever sees it, the loss is happening
    BEFORE argv reaches this process -- most likely the invoking shell
    itself. On Windows, PowerShell in particular can parse an UNQUOTED
    comma-separated list of digit-like tokens (e.g. 07,23,35) as an
    expression that builds an array of numbers rather than passing it
    through as one literal string, which silently drops a leading zero
    on any token PowerShell reads as a decimal integer (07 -> 7) before
    the external python3 process is even launched -- no Python code can
    recover a character the calling shell already discarded. The
    workaround is to quote the value at the call site so the shell
    cannot reinterpret it: --case-ids "07,23,35" (PowerShell/cmd.exe) or
    --case-ids '07,23,35' (bash/zsh); this is a shell-quoting concern at
    the invocation site, not a defect in this file."""
    parser = argparse.ArgumentParser(description="Milestone 2B/2C/4B standalone live-model evaluation runner.")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--label", default="prompt_v4_4b_dimqual_calibration1_eval1", help="Short label for this run, e.g. prompt_v4_4b_dimqual_calibration1_eval1")
    parser.add_argument("--out", default=None, help="Output JSON path (default: eval/results/<label>.json)")
    # Milestone 4B D2 atomic-predicate targeted live probe (diagnostic
    # checkpoint, explicitly authorized): a SMALL, ADDITIVE, opt-in filter
    # -- default None means every existing invocation of this script runs
    # the full, unmodified 42-case benchmark exactly as before (Milestone
    # 2B.1 spec §9 comparability discipline, unchanged). When supplied,
    # ONLY cases whose id starts with one of the given comma-separated
    # prefixes are run -- e.g. --case-ids 07,23,35 runs cases 07/23/35
    # only. This is a DIAGNOSTIC tool, never a scoring/decision-gate run:
    # aggregate_metrics' denominators are computed over whatever subset
    # was actually run, so they are NOT comparable to a full-benchmark
    # aggregate and the report is marked accordingly (see `targeted_run`
    # below). eval/labeled_set.yaml itself is never modified or filtered
    # on disk -- only which of its already-existing cases this particular
    # invocation chooses to run. Deliberately no `type=` argument here --
    # argparse must store the raw comma-separated string exactly as
    # given; select_cases() (not argparse) owns all splitting/parsing.
    parser.add_argument(
        "--case-ids", default=None,
        help="Comma-separated case-id PREFIXES to run only (diagnostic subset run), e.g. --case-ids 07,23,35. "
             "On Windows PowerShell, quote the value (--case-ids \"07,23,35\") -- an unquoted comma list of "
             "digit-like tokens can be reinterpreted by PowerShell's own parser before this script ever sees "
             "it, silently dropping leading zeros. "
             "Default: run the full benchmark (unchanged behavior).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.label in _PROTECTED_LABELS:
        raise SystemExit(
            f"--label {args.label!r} is a protected historical run label (Milestone 2B.1 spec §9: "
            f"baseline_v1.json and baseline_v1_fix1.json must not be modified). Use a new label, "
            f"e.g. baseline_v1_fix1_eval2."
        )

    out_path = pathlib.Path(args.out) if args.out else (ROOT / "eval" / "results" / f"{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provider = AnthropicExtractionProvider(model=args.model)
    cases = load_cases()

    full_case_count = len(cases)
    cases, case_id_prefixes = select_cases(cases, args.case_ids)
    if case_id_prefixes is not None:
        print(f"Diagnostic targeted run: {len(cases)} case(s) matching prefixes {case_id_prefixes} "
              f"(full benchmark has {full_case_count} cases -- aggregate_metrics below is NOT comparable "
              f"to a full-benchmark run).")

    case_results = []
    for case in cases:
        evidence_batch = build_evidence_for_case(case)
        calls_before = len(provider.call_log)
        result = run_extraction(evidence_batch, provider)
        # Milestone 4B: the SEPARATE stage-2 D2/D6 classifier call, run
        # immediately after stage 1 for every case -- mirrors real usage
        # exactly (production always attempts stage 2 once stage 1
        # finishes, whenever there is something eligible to classify;
        # run_dimension_qualifier_classification's own skip-if-nothing-
        # eligible logic means this is a no-op, zero-call pass for cases
        # 1-33, which never touch AdoptionObservation/StakeholderObservation
        # eligibility in a way that matters here -- it also usefully
        # exercises anti-fabrication across the WHOLE benchmark, not just
        # cases 34-42).
        dq_calls_before = len(provider.dimension_qualifier_call_log)
        result = run_dimension_qualifier_classification(result, provider)
        case_result = score_case(case, result, provider, calls_before, dq_calls_before=dq_calls_before)
        case_results.append(case_result)
        print(f"  ran {case['id']}: accepted={len(result.accepted)} rejected={len(result.rejected)} "
              f"contradictions={len(result.candidate_contradictions)} "
              f"risk_signals={len(result.candidate_risk_signals)} "
              f"evidence_classifications={len(result.candidate_evidence_classifications)} "
              f"d2_qualifiers={len(result.candidate_d2_qualifiers)} "
              f"d6_qualifiers={len(result.candidate_d6_qualifiers)} "
              f"dq_stage_failure={result.dimension_qualifier_stage_failure} "
              f"failure={result.request_failure}")
        if case_id_prefixes is not None:
            _print_atomic_predicate_table(case["id"], case_result)

    aggregate = aggregate_metrics(case_results)

    report = {
        "milestone": "4B",
        "run_label": args.label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "anthropic",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        # Milestone 4B: the SEPARATE stage-2 prompt's own version, kept
        # distinct from prompt_version (stage 1's) so a report is always
        # traceable to the exact text of BOTH prompts that produced it.
        "dimension_qualifier_prompt_version": DIMENSION_QUALIFIER_PROMPT_VERSION,
        "evaluation_set_path": str(LABELED_SET_PATH.relative_to(ROOT)),
        "evaluation_set_case_count": len(cases),
        # Milestone 4B D2 atomic-predicate targeted live probe: explicit,
        # non-silent marker distinguishing a diagnostic subset run from a
        # full-benchmark scoring/decision-gate run -- None means this is a
        # normal full-benchmark run, unchanged from every prior report.
        "targeted_run_case_id_prefixes": case_id_prefixes,
        "aggregate_metrics": aggregate,
        "case_results": case_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote report to {out_path}")
    print(json.dumps(aggregate, indent=2, default=str))


if __name__ == "__main__":
    main()
