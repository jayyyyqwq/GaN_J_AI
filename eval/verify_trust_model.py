"""
Phase 2 step 4 — TrustModel verification.

Feeds 3x verified_kept=True, then 2x verified_kept=False for one peer
and prints Q-value evolution. Confirms Q tracks toward the running
mean of rewards (r=+1 for kept, r=-2 for broken).
"""
from __future__ import annotations

from agentgrid_env.server.trust_model import TrustModel

PEER = "B"
ACTION = "accept_their_offer"


def main() -> None:
    tm = TrustModel(peers=[PEER])
    print(f"Initial Q({PEER},{ACTION}) = {tm.q(PEER, ACTION):.4f}")
    print(f"Initial UCB                = {tm.ucb(PEER, ACTION):.4f}\n")

    sequence: list[bool] = [True, True, True, False, False]
    for i, kept in enumerate(sequence, start=1):
        tm.record_settlement(PEER, ACTION, verified_kept=kept)
        r = 1.0 if kept else -2.0
        print(
            f"step {i}: kept={kept!s:<5} r={r:+.1f}  "
            f"Q={tm.q(PEER, ACTION):+.4f}  "
            f"N={tm.N[(PEER, ACTION)]}  "
            f"UCB={tm.ucb(PEER, ACTION):+.4f}"
        )

    print("\nSnapshot block (what the LLM sees):")
    for k, v in tm.snapshot_for_obs().items():
        print(f"  {k}: {v}")

    tm.end_episode()
    print(f"\nAfter end_episode MC pass: Q = {tm.q(PEER, ACTION):+.4f}")
    print(f"episode_trace cleared: len={len(tm.episode_trace)}")


if __name__ == "__main__":
    main()
