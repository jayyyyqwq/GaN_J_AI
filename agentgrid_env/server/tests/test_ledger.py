"""
Unit tests for CommitmentLedger.
Verifies hash chain integrity and verification logic.
Run: pytest agentgrid_env/server/tests/test_ledger.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from agentgrid_env.server.ledger import CommitmentLedger, VOLTS_PER_ENERGY_UNIT, TOLERANCE


def _make_ledger() -> CommitmentLedger:
    return CommitmentLedger(":memory:")


def test_append_returns_incrementing_ids():
    ledger = _make_ledger()
    id1 = ledger.append(step=1, offerer="A", accepter="B",
                        give_type="energy", give_amount=0.1,
                        want_type="compute", want_amount=1.0)
    id2 = ledger.append(step=2, offerer="B", accepter="C",
                        give_type="energy", give_amount=0.05,
                        want_type="compute", want_amount=0.5)
    assert id2 > id1


def test_hash_chain_links_entries():
    ledger = _make_ledger()
    ledger.append(step=1, offerer="A", accepter="B",
                  give_type="energy", give_amount=0.1,
                  want_type="compute", want_amount=1.0)
    ledger.append(step=2, offerer="B", accepter="C",
                  give_type="energy", give_amount=0.1,
                  want_type="compute", want_amount=1.0)
    entries = ledger.recent(2)
    # Second entry's prev_hash equals first entry's this_hash
    assert entries[1]["prev_hash"] == entries[0]["this_hash"]


def test_hash_chain_is_non_trivial():
    ledger = _make_ledger()
    ledger.append(step=1, offerer="A", accepter="B",
                  give_type="energy", give_amount=0.1,
                  want_type="compute", want_amount=1.0)
    entries = ledger.recent(1)
    assert entries[0]["this_hash"] != "0" * 64
    assert len(entries[0]["this_hash"]) == 64


def test_verify_sim_kept_within_tolerance():
    ledger = _make_ledger()
    entry_id = ledger.append(step=1, offerer="A", accepter="B",
                              give_type="energy", give_amount=0.1,
                              want_type="compute", want_amount=1.0)
    expected = 0.1 * VOLTS_PER_ENERGY_UNIT  # e.g. 0.008
    status = ledger.verify_sim(entry_id, expected)
    assert status == "verified_kept"


def test_verify_sim_broken_outside_tolerance():
    ledger = _make_ledger()
    entry_id = ledger.append(step=1, offerer="A", accepter="B",
                              give_type="energy", give_amount=0.1,
                              want_type="compute", want_amount=1.0)
    status = ledger.verify_sim(entry_id, 0.0)  # no voltage drop at all
    assert status == "verified_broken"


def test_kept_ratio_all_kept():
    ledger = _make_ledger()
    for _ in range(3):
        eid = ledger.append(step=1, offerer="A", accepter="B",
                            give_type="energy", give_amount=0.1,
                            want_type="compute", want_amount=1.0)
        ledger.verify_sim(eid, 0.1 * VOLTS_PER_ENERGY_UNIT)
    assert ledger.kept_ratio("A") == 1.0


def test_kept_ratio_mixed():
    ledger = _make_ledger()
    for i in range(4):
        eid = ledger.append(step=i, offerer="A", accepter="B",
                            give_type="energy", give_amount=0.1,
                            want_type="compute", want_amount=1.0)
        delta = 0.1 * VOLTS_PER_ENERGY_UNIT if i < 3 else 0.0
        ledger.verify_sim(eid, delta)
    assert ledger.kept_ratio("A") == 0.75


def test_kept_ratio_no_entries_returns_default():
    ledger = _make_ledger()
    assert ledger.kept_ratio("A") == 0.5


def test_pending_for_agent():
    ledger = _make_ledger()
    ledger.append(step=1, offerer="A", accepter="B",
                  give_type="energy", give_amount=0.1,
                  want_type="compute", want_amount=1.0)
    pending = ledger.pending_for("B")
    assert len(pending) == 1
    assert pending[0]["accepter"] == "B"


def test_update_status():
    ledger = _make_ledger()
    eid = ledger.append(step=1, offerer="A", accepter="B",
                        give_type="energy", give_amount=0.1,
                        want_type="compute", want_amount=1.0)
    ledger.update_status(eid, "reneged")
    entries = ledger.recent(1)
    assert entries[0]["status"] == "reneged"
