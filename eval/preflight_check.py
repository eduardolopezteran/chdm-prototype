#!/usr/bin/env python3
"""Milestone 4B runtime/version-trace preflight gate.

WHY THIS EXISTS
---------------
A targeted live probe was labeled as a v3.2 run, but the generated report
serialized dimension_qualifier_prompt_version = "v3.1" -- meaning the
process that made the (paid) Anthropic calls had loaded a v3.1
extraction/prompts.py, not the intended v3.2 one. The behavioral result
was therefore invalid for v3.2 gating, and the calls were wasted.

This script answers, BEFORE any Anthropic call is made, the only question
that matters: "which prompts.py will this Python process actually load,
and what version/prompt text does it contain?" It makes zero network
calls, needs no ANTHROPIC_API_KEY, and costs nothing to run.

It deliberately resolves modules exactly the way eval/run_eval.py does
(same ROOT computation, same sys.path insertion), so what it reports is
what run_eval.py will get -- not an approximation.

STRUCTURAL GUARANTEE THIS SCRIPT RELIES ON
------------------------------------------
eval/run_eval.py imports DIMENSION_QUALIFIER_PROMPT_VERSION from
extraction.prompts, and extraction/provider.py imports the prompt
BUILDER (build_isolated_dimension_qualifier_system_prompt) from that same
module. Python caches modules in sys.modules, so within one process both
resolve to the SAME module object loaded from the SAME file. The reported
version metadata and the prompt text actually sent to the model therefore
cannot diverge: if the report says v3.1, the model was genuinely prompted
with v3.1 text. This script verifies that identity explicitly rather than
assuming it, and additionally fingerprints the prompt text itself so
metadata and behavior are provably tied to one artifact.

USAGE
-----
    cd chdm-engine
    python3 eval/preflight_check.py --expect-version v3.2

Exits 0 if everything matches; exits 1 (loudly) on any mismatch, so it
can gate a live run:

    python3 eval/preflight_check.py --expect-version v3.2 && \\
      python3 eval/run_eval.py --model claude-haiku-4-5-20251001 \\
        --label <label> --case-ids "07,23,35"

This script reads only. It never writes to the repository, never touches
eval/labeled_set.yaml, never modifies prompt content, the composer,
schemas, D6, the benchmark, or the architecture.
"""

import argparse
import hashlib
import pathlib
import sys

# Resolve ROOT exactly as eval/run_eval.py does (its own lines:
# ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))).
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    """Import the real modules and report what actually resolved. Kept
    separate from printing so a deterministic test can assert on it
    without parsing stdout."""
    import extraction.prompts as prompts_module
    import extraction.provider as provider_module

    from extraction.prompts import (
        DIMENSION_QUALIFIER_PROMPT_VERSION,
        PROMPT_VERSION,
        build_isolated_dimension_qualifier_system_prompt,
    )
    from domain.enums import DimensionCode

    d2_prompt = build_isolated_dimension_qualifier_system_prompt(DimensionCode.D2)
    d6_prompt = build_isolated_dimension_qualifier_system_prompt(DimensionCode.D6)

    prompts_path = pathlib.Path(prompts_module.__file__).resolve()
    run_eval_path = (ROOT / "eval" / "run_eval.py").resolve()

    # Identity check: the module the REPORT METADATA comes from must be
    # the same module object the PROMPT BUILDER (and therefore the actual
    # model call) comes from. If these ever differ, metadata and behavior
    # could describe different artifacts.
    same_module = (
        sys.modules.get("extraction.prompts") is prompts_module
        and build_isolated_dimension_qualifier_system_prompt.__module__ == "extraction.prompts"
    )

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": str(pathlib.Path.cwd()),
        "root": str(ROOT),
        "sys_path_head": sys.path[:3],
        "pythonpath_env": __import__("os").environ.get("PYTHONPATH", ""),
        "prompts_module_file": str(prompts_path),
        "provider_module_file": str(pathlib.Path(provider_module.__file__).resolve()),
        "evaluator_source_path": str(run_eval_path),
        "prompts_file_sha256": _file_sha256(prompts_path),
        "prompts_file_mtime": prompts_path.stat().st_mtime,
        "stage1_prompt_version": PROMPT_VERSION,
        "dimension_qualifier_prompt_version": DIMENSION_QUALIFIER_PROMPT_VERSION,
        "d2_prompt_sha256": _sha256(d2_prompt),
        "d6_prompt_sha256": _sha256(d6_prompt),
        "d2_prompt_len": len(d2_prompt),
        "metadata_and_prompt_same_module": same_module,
    }


# Marker phrases that exist ONLY in the v3.2 RELIABLE_AUTOMATION_OPERATION
# calibration. These tie the loaded PROMPT TEXT (behavior) to the version
# STRING (metadata) -- a version constant alone could in principle be
# edited without the prompt body, so both are checked. Read-only content
# checks; nothing here changes prompt semantics.
_V3_2_D2_MARKERS = (
    "scheduled CADENCE alone",
    "cadence describes WHEN something runs, not whether it works",
    "an explicit absence of failures over a period or run history",
)


def verify(info: dict, expect_version: str) -> list:
    """Return a list of human-readable problems; empty list == clean."""
    problems = []
    actual = info["dimension_qualifier_prompt_version"]
    if actual != expect_version:
        problems.append(
            f"VERSION MISMATCH: runtime resolved DIMENSION_QUALIFIER_PROMPT_VERSION == "
            f"{actual!r}, but {expect_version!r} was expected. The file actually loaded is "
            f"{info['prompts_module_file']} -- that is the wrong copy of the repository. "
            f"Do NOT spend an Anthropic call from this directory."
        )
    if not info["metadata_and_prompt_same_module"]:
        problems.append(
            "MODULE IDENTITY FAILURE: the version metadata and the prompt builder did not "
            "resolve to the same extraction.prompts module object. Report metadata could "
            "describe a different artifact than the one actually sent to the model."
        )
    if expect_version == "v3.2":
        import extraction.prompts as pm
        d2 = pm._ISOLATED_D2_SYSTEM_PROMPT
        missing = [m for m in _V3_2_D2_MARKERS if m not in d2]
        if missing:
            problems.append(
                f"PROMPT-TEXT MISMATCH: DIMENSION_QUALIFIER_PROMPT_VERSION says {actual!r}, but "
                f"the loaded D2 prompt text is missing v3.2 marker phrase(s): {missing!r}. "
                f"The version constant and the prompt body disagree."
            )
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Preflight runtime/version verification for eval/run_eval.py (no network calls)."
    )
    parser.add_argument(
        "--expect-version", default=None,
        help='Required dimension-qualifier prompt version, e.g. --expect-version v3.2. '
             'If supplied and the runtime does not match, this exits 1 so a live run can be gated on it.',
    )
    args = parser.parse_args()

    info = collect()

    print("=" * 78)
    print("MILESTONE 4B PREFLIGHT - runtime/version verification (NO Anthropic calls)")
    print("=" * 78)
    print(f"  Python executable        : {info['python_executable']}")
    print(f"  Python version           : {info['python_version']}")
    print(f"  Current working directory: {info['cwd']}")
    print(f"  Resolved repo ROOT       : {info['root']}")
    print(f"  sys.path[:3]             : {info['sys_path_head']}")
    print(f"  PYTHONPATH env           : {info['pythonpath_env'] or '(empty)'}")
    print("-" * 78)
    print(f"  evaluator source path    : {info['evaluator_source_path']}")
    print(f"  extraction.prompts file  : {info['prompts_module_file']}")
    print(f"  extraction.provider file : {info['provider_module_file']}")
    print(f"  prompts.py sha256        : {info['prompts_file_sha256']}")
    print("-" * 78)
    print(f"  PROMPT_VERSION (stage 1)             : {info['stage1_prompt_version']}")
    print(f"  DIMENSION_QUALIFIER_PROMPT_VERSION   : {info['dimension_qualifier_prompt_version']}")
    print(f"  D2 prompt fingerprint (sha256)       : {info['d2_prompt_sha256']}")
    print(f"  D2 prompt length (chars)             : {info['d2_prompt_len']}")
    print(f"  D6 prompt fingerprint (sha256)       : {info['d6_prompt_sha256']}")
    print(f"  metadata & prompt same module object : {info['metadata_and_prompt_same_module']}")
    print("=" * 78)

    if args.expect_version is None:
        print("No --expect-version supplied; reporting only, no pass/fail gate applied.")
        return

    problems = verify(info, args.expect_version)
    if problems:
        print(f"PREFLIGHT FAILED (expected {args.expect_version!r}):")
        for p in problems:
            print(f"  * {p}")
        print("=" * 78)
        raise SystemExit(1)

    print(f"PREFLIGHT PASSED: runtime resolves {args.expect_version!r} from")
    print(f"  {info['prompts_module_file']}")
    print("  and the loaded prompt text matches that version. Safe to run the live evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
