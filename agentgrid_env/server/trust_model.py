"""
Per-agent tabular trust learner.

Q-learning with online averaging + end-of-episode MC reconciliation.
UCB1 confidence bounds surfaced into the LLM observation.

No gradients. No second training loop. No extra hyperparameter sweep.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

ACTIONS = ("accept_their_offer", "trust_their_payment")


@dataclass
class TrustModel:
    peers: list[str]
    alpha: float = 0.3      # in-episode learning rate
    alpha_mc: float = 0.1   # end-of-episode MC reconciliation rate
    gamma: float = 0.95     # MC discount
    c: float = 1.4          # UCB exploration constant
    Q: dict = field(default_factory=lambda: defaultdict(float))
    N: dict = field(default_factory=lambda: defaultdict(int))
    t: int = 0
    episode_trace: list = field(default_factory=list)

    def record_settlement(self, peer: str, action: str, verified_kept: bool) -> None:
        """Call each time the ledger resolves a promise (hardware-verified or sim-verified)."""
        r = 1.0 if verified_kept else -2.0
        key = (peer, action)
        self.Q[key] += self.alpha * (r - self.Q[key])
        self.N[key] += 1
        self.t += 1
        self.episode_trace.append((peer, action, r))

    def end_episode(self) -> None:
        """First-visit MC reconciliation pass at episode boundary."""
        G = 0.0
        seen: set = set()
        for peer, action, r in reversed(self.episode_trace):
            G = r + self.gamma * G
            key = (peer, action)
            if key not in seen:
                seen.add(key)
                self.Q[key] += self.alpha_mc * (G - self.Q[key])
        self.episode_trace.clear()

    def q(self, peer: str, action: str = "accept_their_offer") -> float:
        return self.Q[(peer, action)]

    def ucb(self, peer: str, action: str = "accept_their_offer") -> float:
        key = (peer, action)
        n = self.N[key]
        if n == 0:
            return 9.99  # cap instead of inf so it serializes cleanly
        return self.Q[key] + self.c * math.sqrt(math.log(max(self.t, 1)) / n)

    def snapshot_for_obs(self) -> dict:
        out: dict = {}
        for p in self.peers:
            out[f"Q_accept_{p}"] = round(self.q(p, "accept_their_offer"), 2)
            out[f"Q_trust_{p}"] = round(self.q(p, "trust_their_payment"), 2)
            out[f"UCB_{p}"] = round(min(self.ucb(p), 9.99), 2)
            out[f"N_{p}"] = sum(self.N[(p, a)] for a in ACTIONS)
        return out
