"""
Hash-chained commitment ledger backed by SQLite.

Not a blockchain — a single append-only log with SHA-256 chain links.
Physical voltage readings (Uno ADC, served by bridge) are the oracle that decides verified_kept vs verified_broken.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    offerer TEXT NOT NULL,
    accepter TEXT NOT NULL,
    give_type TEXT NOT NULL,
    give_amount REAL NOT NULL,
    want_type TEXT NOT NULL,
    want_amount REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    prev_hash TEXT NOT NULL,
    this_hash TEXT NOT NULL,
    ts REAL NOT NULL
)
"""

# Normalized voltage drop per SoC unit transferred.
# Sim value (~0.15) matches the 18650 SoC→OCV plateau slope in sim_backend._SOC_CURVE.
# Hardware re-fit: discharge at constant load, record (SoC_before, SoC_after, delta_V_mV),
# compute mean(delta_V / delta_SoC) and replace this constant.
VOLTS_PER_ENERGY_UNIT: float = 0.15
# Proportional tolerance: covers Gaussian jitter (σ=0.003) up to ~3σ plus quantization.
# Spike events (1% prob, σ=0.05) are excluded — those represent genuine measurement failures.
_TOLERANCE_FRACTION: float = 0.55
_MIN_TOLERANCE: float = 0.012  # ~2.4 LSB at 5 mV/LSB (Uno 10-bit ADC)
TOLERANCE: float = _MIN_TOLERANCE  # public alias for tests / external callers


@dataclass
class LedgerEntry:
    id: int
    step: int
    offerer: str
    accepter: str
    give_type: str
    give_amount: float
    want_type: str
    want_amount: float
    status: str
    prev_hash: str
    this_hash: str
    ts: float


class CommitmentLedger:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def append(
        self,
        step: int,
        offerer: str,
        accepter: str,
        give_type: str,
        give_amount: float,
        want_type: str,
        want_amount: float,
    ) -> int:
        prev = self._latest_hash()
        fields = dict(
            step=step, offerer=offerer, accepter=accepter,
            give_type=give_type, give_amount=give_amount,
            want_type=want_type, want_amount=want_amount,
        )
        payload = json.dumps(fields, sort_keys=True) + prev
        this_hash = hashlib.sha256(payload.encode()).hexdigest()
        cur = self._conn.execute(
            """INSERT INTO entries
               (step, offerer, accepter, give_type, give_amount,
                want_type, want_amount, status, prev_hash, this_hash, ts)
               VALUES (?,?,?,?,?,?,?,'pending',?,?,?)""",
            (step, offerer, accepter, give_type, give_amount,
             want_type, want_amount, prev, this_hash, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def verify_against_hardware(self, entry_id: int, delta_v: float) -> str:
        """Compare promised voltage drop to actual Uno ADC reading."""
        row = self._get(entry_id)
        if row is None:
            return "not_found"
        expected = row["give_amount"] * VOLTS_PER_ENERGY_UNIT
        tolerance = max(_MIN_TOLERANCE, expected * _TOLERANCE_FRACTION)
        if abs(delta_v - expected) < tolerance:
            status = "verified_kept"
        else:
            status = "verified_broken"
        self._update_status(entry_id, status)
        return status

    def verify_sim(self, entry_id: int, actual_delta: float) -> str:
        """Sim-mode verification using simulated voltage drop."""
        return self.verify_against_hardware(entry_id, actual_delta)

    def update_status(self, entry_id: int, status: str) -> None:
        self._update_status(entry_id, status)

    def recent(self, n: int = 5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM entries ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def pending_for(self, accepter: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM entries WHERE accepter=? AND status='pending'",
            (accepter,),
        ).fetchall()
        return [dict(r) for r in rows]

    def kept_ratio(self, agent: str) -> float:
        rows = self._conn.execute(
            "SELECT status FROM entries WHERE offerer=? AND status IN ('verified_kept','verified_broken')",
            (agent,),
        ).fetchall()
        if not rows:
            return 0.5
        kept = sum(1 for r in rows if r["status"] == "verified_kept")
        return round(kept / len(rows), 2)

    # ------------------------------------------------------------------
    def _latest_hash(self) -> str:
        row = self._conn.execute(
            "SELECT this_hash FROM entries ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["this_hash"] if row else "0" * 64

    def _get(self, entry_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM entries WHERE id=?", (entry_id,)
        ).fetchone()

    def _update_status(self, entry_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE entries SET status=? WHERE id=?", (status, entry_id)
        )
        self._conn.commit()
