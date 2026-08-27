"""
Milestone 3B — ui/review_queue.py tests (pure function, no Streamlit, no
live consequentiality evaluation -- lookup is stubbed).
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import ConfirmationAction
from confirmation.enums import ConfirmationTargetKind
from confirmation.schemas import create_confirmation_record

from ui.review_queue import build_review_queue, find_item
from ui.sample_data import build_sample_scenario

_ALL_IDS = {
    "OBS-STAKE-1", "OBS-OBJ-1", "OBS-SERVICE-1", "OBS-MISSING-RENEWAL",
    "OBS-EXPERIENCE-BYSTANDER", "OBS-RISK-CHAMPION", "OBS-RISK-SERVICE", "OBS-EVIDCLASS-1",
}


def _extraction_result():
    _, _, extraction_result, _, _ = build_sample_scenario()
    return extraction_result


def _no_consequentiality(_observation_id):
    return None


def test_all_unreviewed_items_appear_in_queue():
    queue = build_review_queue(_extraction_result(), (), _no_consequentiality)
    assert {e.target_observation_id for e in queue} == _ALL_IDS
    assert all(e.status == "unreviewed" for e in queue)


def test_confirmed_item_drops_out_of_queue():
    er = _extraction_result()
    confirm = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-CHAMPION",
        action=ConfirmationAction.CONFIRM, reviewer="a.reviewer@example.com",
    )
    queue = build_review_queue(er, (confirm,), _no_consequentiality)
    assert "OBS-RISK-CHAMPION" not in {e.target_observation_id for e in queue}
    assert len(queue) == len(_ALL_IDS) - 1


def test_rejected_item_drops_out_of_queue():
    er = _extraction_result()
    reject = create_confirmation_record(
        target_kind=ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, target_observation_id="OBS-RISK-SERVICE",
        action=ConfirmationAction.REJECT, reviewer="a.reviewer@example.com",
        reason="Evidence does not support the proposed statement.",
    )
    queue = build_review_queue(er, (reject,), _no_consequentiality)
    assert "OBS-RISK-SERVICE" not in {e.target_observation_id for e in queue}


def test_cannot_confirm_item_remains_in_queue_with_status():
    er = _extraction_result()
    cannot_confirm = create_confirmation_record(
        target_kind=ConfirmationTargetKind.SEMANTIC_OBSERVATION, target_observation_id="OBS-STAKE-1",
        action=ConfirmationAction.CANNOT_CONFIRM, reviewer="a.reviewer@example.com",
        reason="Source is ambiguous.",
    )
    queue = build_review_queue(er, (cannot_confirm,), _no_consequentiality)
    entry = next(e for e in queue if e.target_observation_id == "OBS-STAKE-1")
    assert entry.status == "cannot_confirm"


def test_ordering_consequential_before_unevaluated_before_not_consequential():
    er = _extraction_result()
    lookup_values = {"OBS-RISK-CHAMPION": True, "OBS-RISK-SERVICE": False}

    def lookup(observation_id):
        return lookup_values.get(observation_id, None)

    queue = build_review_queue(er, (), lookup)
    order = [e.target_observation_id for e in queue]
    assert order.index("OBS-RISK-CHAMPION") < order.index("OBS-STAKE-1")  # consequential before unevaluated
    assert order.index("OBS-STAKE-1") < order.index("OBS-RISK-SERVICE")   # unevaluated before not-consequential


def test_unevaluated_never_sorts_as_non_consequential():
    er = _extraction_result()
    lookup_values = {"OBS-RISK-SERVICE": False}

    def lookup(observation_id):
        return lookup_values.get(observation_id, None)

    queue = build_review_queue(er, (), lookup)
    unevaluated_indices = [i for i, e in enumerate(queue) if e.consequentiality is None]
    not_consequential_indices = [i for i, e in enumerate(queue) if e.consequentiality is False]
    assert unevaluated_indices and not_consequential_indices
    assert max(unevaluated_indices) < min(not_consequential_indices)


def test_find_item_returns_none_for_unknown_target():
    er = _extraction_result()
    assert find_item(er, ConfirmationTargetKind.SEMANTIC_OBSERVATION, "OBS-DOES-NOT-EXIST") is None


def test_find_item_returns_correct_object():
    er = _extraction_result()
    obs = find_item(er, ConfirmationTargetKind.CANDIDATE_RISK_SIGNAL, "OBS-RISK-CHAMPION")
    assert obs is not None
    assert obs.mechanism == "CR-01"


TESTS = [
    test_all_unreviewed_items_appear_in_queue,
    test_confirmed_item_drops_out_of_queue,
    test_rejected_item_drops_out_of_queue,
    test_cannot_confirm_item_remains_in_queue_with_status,
    test_ordering_consequential_before_unevaluated_before_not_consequential,
    test_unevaluated_never_sorts_as_non_consequential,
    test_find_item_returns_none_for_unknown_target,
    test_find_item_returns_correct_object,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(TESTS)}/{len(TESTS)} passed")
