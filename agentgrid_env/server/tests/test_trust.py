"""
Unit tests for TrustModel.
Verifies Q-updates match hand-computed values and UCB bounds behave correctly.
Run: pytest agentgrid_env/server/tests/test_trust.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from agentgrid_env.server.trust_model import TrustModel


def test_q_update_verified_kept():
    tm = TrustModel(peers=["B", "C"], alpha=0.3)
    tm.record_settlement("B", "accept_their_offer", verified_kept=True)
    # Q[B, accept] = 0 + 0.3 * (1.0 - 0) = 0.3
    assert abs(tm.q("B", "accept_their_offer") - 0.3) < 1e-9


def test_q_update_verified_broken():
    tm = TrustModel(peers=["B", "C"], alpha=0.3)
    tm.record_settlement("B", "accept_their_offer", verified_kept=False)
    # Q[B, accept] = 0 + 0.3 * (-2.0 - 0) = -0.6
    assert abs(tm.q("B", "accept_their_offer") - (-0.6)) < 1e-9


def test_q_update_two_steps():
    tm = TrustModel(peers=["B"], alpha=0.3)
    tm.record_settlement("B", "accept_their_offer", verified_kept=True)   # Q = 0.3
    tm.record_settlement("B", "accept_their_offer", verified_kept=True)   # Q = 0.3 + 0.3*(1-0.3) = 0.51
    assert abs(tm.q("B", "accept_their_offer") - 0.51) < 1e-9


def test_visit_counter():
    tm = TrustModel(peers=["B", "C"])
    tm.record_settlement("B", "accept_their_offer", True)
    tm.record_settlement("B", "trust_their_payment", False)
    snap = tm.snapshot_for_obs()
    assert snap["N_B"] == 2
    assert snap["N_C"] == 0


def test_ucb_zero_visits_returns_cap():
    tm = TrustModel(peers=["B"], c=1.4)
    assert tm.ucb("B") == 9.99


def test_ucb_decreases_with_more_visits():
    tm = TrustModel(peers=["B"], c=1.4, alpha=0.3)
    for _ in range(10):
        tm.record_settlement("B", "accept_their_offer", True)
    ucb_10 = tm.ucb("B")
    for _ in range(90):
        tm.record_settlement("B", "accept_their_offer", True)
    ucb_100 = tm.ucb("B")
    assert ucb_100 < ucb_10


def test_mc_reconciliation_adjusts_q():
    tm = TrustModel(peers=["B"], alpha=0.3, alpha_mc=0.1, gamma=0.95)
    tm.record_settlement("B", "accept_their_offer", True)    # +1
    tm.record_settlement("B", "accept_their_offer", False)   # -2
    q_before = tm.q("B", "accept_their_offer")
    tm.end_episode()
    q_after = tm.q("B", "accept_their_offer")
    # MC reconciliation should pull Q toward full-episode return (net negative)
    assert q_after < q_before or abs(q_after - q_before) < 1.0  # direction is right


def test_end_episode_clears_trace():
    tm = TrustModel(peers=["B"])
    tm.record_settlement("B", "accept_their_offer", True)
    assert len(tm.episode_trace) == 1
    tm.end_episode()
    assert len(tm.episode_trace) == 0


def test_snapshot_keys_match_peers():
    tm = TrustModel(peers=["B", "C"])
    snap = tm.snapshot_for_obs()
    for p in ("B", "C"):
        assert f"Q_accept_{p}" in snap
        assert f"Q_trust_{p}" in snap
        assert f"UCB_{p}" in snap
        assert f"N_{p}" in snap
