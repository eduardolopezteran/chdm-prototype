import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.enums import EvidenceState
from engine.evidence_engine import is_current_confirmed, is_usable_as_potential


def test_is_current_confirmed():
    assert is_current_confirmed(EvidenceState.CURRENT_CONFIRMED) is True
    for s in (EvidenceState.CURRENT_UNVERIFIED, EvidenceState.STALE,
              EvidenceState.CONTRADICTORY, EvidenceState.UNAVAILABLE,
              EvidenceState.NOT_APPLICABLE):
        assert is_current_confirmed(s) is False


def test_is_usable_as_potential():
    assert is_usable_as_potential(EvidenceState.CURRENT_CONFIRMED) is True
    assert is_usable_as_potential(EvidenceState.CURRENT_UNVERIFIED) is True
    assert is_usable_as_potential(EvidenceState.STALE) is True
    assert is_usable_as_potential(EvidenceState.CONTRADICTORY) is True
    assert is_usable_as_potential(EvidenceState.UNAVAILABLE) is False
    assert is_usable_as_potential(EvidenceState.NOT_APPLICABLE) is False


if __name__ == "__main__":
    test_is_current_confirmed()
    test_is_usable_as_potential()
    print("PASS  test_evidence_engine (2/2)")
