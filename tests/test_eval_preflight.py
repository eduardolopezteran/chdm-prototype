import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.preflight_check import collect, verify
from extraction.prompts import DIMENSION_QUALIFIER_PROMPT_VERSION

# ===========================================================================
# Milestone 4B runtime/version-trace preflight gate.
#
# Context: a targeted live probe was labeled v3.2 but its report serialized
# dimension_qualifier_prompt_version = "v3.1" -- the process making the paid
# Anthropic calls had loaded an older extraction/prompts.py. eval/
# preflight_check.py exists to catch exactly that BEFORE any call is made.
# These tests verify the gate itself works, in both directions: it must
# pass on a correct tree and it must FAIL LOUDLY on the specific
# divergences that caused the wasted run.
#
# These tests do not change prompt semantics, D2 predicate definitions, the
# composer, schemas, D6, the benchmark, or the architecture.
# ===========================================================================


def test_preflight_collect_reports_the_real_runtime_module_paths():
    """collect() must report the ACTUAL file Python imported -- not a
    guess or a hardcoded path -- so the operator can see which copy of the
    repository is really about to run."""
    info = collect()
    prompts_path = pathlib.Path(info["prompts_module_file"])
    assert prompts_path.is_absolute()
    assert prompts_path.exists()
    assert prompts_path.name == "prompts.py"
    # It must be THIS repository's copy, not some other extracted tree.
    assert prompts_path == (ROOT / "extraction" / "prompts.py").resolve()
    assert pathlib.Path(info["evaluator_source_path"]) == (ROOT / "eval" / "run_eval.py").resolve()


def test_preflight_reports_version_matching_the_imported_constant():
    """The version the preflight reports must be the same value run_eval.py
    will serialize into its report -- both import it from the same module."""
    info = collect()
    assert info["dimension_qualifier_prompt_version"] == DIMENSION_QUALIFIER_PROMPT_VERSION


def test_preflight_confirms_metadata_and_prompt_share_one_module_object():
    """The core structural guarantee: report metadata (the version string)
    and actual model behavior (the prompt builder) must resolve to the SAME
    extraction.prompts module object. If this were ever false, a report
    could describe a different artifact than the one sent to the model."""
    info = collect()
    assert info["metadata_and_prompt_same_module"] is True


def test_preflight_emits_a_prompt_text_fingerprint():
    """A version constant alone can be edited without the prompt body, so
    the preflight must fingerprint the actual loaded prompt text too."""
    info = collect()
    for key in ("d2_prompt_sha256", "d6_prompt_sha256", "prompts_file_sha256"):
        assert isinstance(info[key], str) and len(info[key]) == 64
    assert info["d2_prompt_len"] > 0
    # D2 and D6 are different prompts; their fingerprints must differ.
    assert info["d2_prompt_sha256"] != info["d6_prompt_sha256"]


def test_preflight_passes_clean_on_the_current_tree():
    """Against this repository at its real current version, verify() must
    report zero problems."""
    info = collect()
    assert verify(info, DIMENSION_QUALIFIER_PROMPT_VERSION) == []


def test_preflight_fails_loudly_on_version_mismatch():
    """The exact real-world failure: the operator intends v3.2 but the
    runtime resolved something older. verify() must report it, naming the
    file actually loaded so the wrong directory can be identified."""
    info = collect()
    info = dict(info, dimension_qualifier_prompt_version="v3.1")
    problems = verify(info, "v3.2")
    assert len(problems) >= 1
    joined = " ".join(problems)
    assert "VERSION MISMATCH" in joined
    assert "'v3.1'" in joined and "'v3.2'" in joined
    # Must name the offending file path so the stale tree is identifiable.
    assert info["prompts_module_file"] in joined


def test_preflight_fails_when_version_string_and_prompt_text_disagree():
    """Defense in depth: if someone bumped the version constant to v3.2 but
    the loaded D2 prompt body lacks the v3.2 calibration markers, the gate
    must catch that the metadata and the behavior disagree."""
    import extraction.prompts as pm

    original = pm._ISOLATED_D2_SYSTEM_PROMPT
    try:
        # Simulate a version constant that claims v3.2 over a v3.1 body.
        pm._ISOLATED_D2_SYSTEM_PROMPT = "a D2 prompt body with none of the v3.2 calibration wording"
        info = dict(collect(), dimension_qualifier_prompt_version="v3.2")
        problems = verify(info, "v3.2")
        joined = " ".join(problems)
        assert "PROMPT-TEXT MISMATCH" in joined
        assert "scheduled CADENCE alone" in joined
    finally:
        pm._ISOLATED_D2_SYSTEM_PROMPT = original
    # Confirm the module was restored, so no other test is affected.
    assert pm._ISOLATED_D2_SYSTEM_PROMPT == original


def test_preflight_detects_module_identity_failure():
    """If metadata and prompt ever resolved to different module objects,
    verify() must refuse regardless of the version string matching."""
    info = dict(collect(), metadata_and_prompt_same_module=False)
    problems = verify(info, DIMENSION_QUALIFIER_PROMPT_VERSION)
    assert any("MODULE IDENTITY FAILURE" in p for p in problems)


TESTS = [
    test_preflight_collect_reports_the_real_runtime_module_paths,
    test_preflight_reports_version_matching_the_imported_constant,
    test_preflight_confirms_metadata_and_prompt_share_one_module_object,
    test_preflight_emits_a_prompt_text_fingerprint,
    test_preflight_passes_clean_on_the_current_tree,
    test_preflight_fails_loudly_on_version_mismatch,
    test_preflight_fails_when_version_string_and_prompt_text_disagree,
    test_preflight_detects_module_identity_failure,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
