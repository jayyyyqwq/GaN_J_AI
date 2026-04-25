"""
Phase 2 step 5 — CommitmentLedger hash-chain verification.

Appends two entries to an in-memory ledger and asserts that
entry 2's prev_hash == entry 1's this_hash.
"""
from __future__ import annotations

from agentgrid_env.server.ledger import CommitmentLedger


def main() -> None:
    ledger = CommitmentLedger(db_path=":memory:")

    id1 = ledger.append(
        step=0, offerer="A", accepter="B",
        give_type="energy", give_amount=0.10,
        want_type="compute", want_amount=3.0,
    )
    id2 = ledger.append(
        step=1, offerer="B", accepter="C",
        give_type="compute", give_amount=2.0,
        want_type="energy", want_amount=0.05,
    )

    rows = ledger.recent(n=2)
    assert len(rows) == 2, f"expected 2 entries, got {len(rows)}"
    e1, e2 = rows[0], rows[1]

    print(f"entry 1 id={e1['id']}  this_hash={e1['this_hash'][:16]}...")
    print(f"entry 2 id={e2['id']}  prev_hash={e2['prev_hash'][:16]}...")
    print(f"entry 2 id={e2['id']}  this_hash={e2['this_hash'][:16]}...")

    assert e1["prev_hash"] == "0" * 64, "first entry must chain from genesis (64 zeros)"
    assert e2["prev_hash"] == e1["this_hash"], "chain link broken"
    assert e1["this_hash"] != e2["this_hash"], "hashes must differ"

    print("\nPASS: entry 1 chains from genesis, entry 2.prev_hash == entry 1.this_hash")


if __name__ == "__main__":
    main()
