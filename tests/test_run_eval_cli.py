import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import build_arg_parser, load_cases, select_cases

# ===========================================================================
# Milestone 4B D2 atomic-predicate targeted live probe -- follow-up
# regression: end-to-end CLI argument-path test.
#
# Context: the live probe (prompt_v4_4b_dimqual_v3_atomic_probe1) and a
# subsequent rerun both reported --case-ids resolving to prefix "7" instead
# of "07", even though the user reports typing --case-ids 07,23,35 exactly.
# A full inspection of this repository (eval/run_eval.py, select_cases,
# argparse configuration) found exactly ONE --case-ids definition, no
# `type=` casting of any kind on it, no int()/lstrip("0")/zfill/
# str(int(...)) anywhere near case-id handling, and no wrapper .bat/.ps1/
# .cmd/.sh scripts anywhere in the repo that could pre-process the argument
# before Python sees it. These tests exercise the REAL, PRODUCTION argparse
# configuration end-to-end -- build_arg_parser() is the exact function
# main() itself calls, not a reimplementation -- proving conclusively that
# argv -> args.case_ids -> select_cases() never touches, casts, or
# normalizes the string anywhere inside this codebase. If a leading zero is
# still lost before python3 sees it, the loss is happening in the invoking
# shell (most likely PowerShell on Windows, which can parse an UNQUOTED
# comma-separated list of digit-like tokens as an array-construction
# expression, silently converting "07" to the integer 7 before argv is
# built) -- not in this file. See build_arg_parser()'s own docstring for
# the full diagnostic note and the quoting workaround.
# ===========================================================================


def test_cli_end_to_end_zero_padded_case_ids_reach_select_cases_unmodified():
    """argv=["--case-ids", "07,23,35"] must produce args.case_ids ==
    "07,23,35" byte-for-byte (argparse must never cast/strip/normalize
    it), and select_cases() must then resolve it to prefixes ==
    ["07", "23", "35"], selecting exactly Cases 07, 23, and 35 out of the
    REAL eval/labeled_set.yaml (not a test fixture) -- end-to-end proof
    that the full CLI argument path inside this codebase preserves
    zero-padded case ids exactly as typed."""
    parser = build_arg_parser()
    args = parser.parse_args(["--case-ids", "07,23,35"])
    assert args.case_ids == "07,23,35"

    cases = load_cases()
    selected, prefixes = select_cases(cases, args.case_ids)
    assert prefixes == ["07", "23", "35"]
    assert [c["id"] for c in selected] == [
        "07_healthy_automated_workflow_low_human_activity",
        "23_pure_adoption_milestone_no_goal_language",
        "35_d2_qualifier_automation_reliable_low_login_ok",
    ]


def test_cli_end_to_end_unpadded_seven_fails_loudly():
    """argv=["--case-ids", "7,23,35"] (the exact real-world defect
    string, missing the leading zero) must produce args.case_ids ==
    "7,23,35" unmodified by argparse, and select_cases() must then raise
    SystemExit naming "7" specifically -- proving the negative case is
    caught at the same real, end-to-end CLI entry point main() itself
    uses, not just via a direct select_cases() call."""
    parser = build_arg_parser()
    args = parser.parse_args(["--case-ids", "7,23,35"])
    assert args.case_ids == "7,23,35"

    cases = load_cases()
    try:
        select_cases(cases, args.case_ids)
        assert False, "expected SystemExit when '7' (unpadded) matches no case"
    except SystemExit as exc:
        message = str(exc)
        assert "['7']" in message
        assert "zero-padded" in message


def test_cli_case_ids_argument_has_no_type_conversion():
    """Structural guard: --case-ids must never be declared with a
    `type=` argument (e.g. type=int) that could silently coerce or
    normalize the value before select_cases() ever sees it. argparse
    only applies `type` conversion when explicitly configured; this test
    pins that this argument stays a plain string."""
    parser = build_arg_parser()
    case_ids_action = next(a for a in parser._actions if a.option_strings == ["--case-ids"])
    assert case_ids_action.type is None


TESTS = [
    test_cli_end_to_end_zero_padded_case_ids_reach_select_cases_unmodified,
    test_cli_end_to_end_unpadded_seven_fails_loudly,
    test_cli_case_ids_argument_has_no_type_conversion,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
