"""
Random-policy episode runner using HeadlessRunner (no HTTP required).
Port of eval/baseline_random.py to the in-process runner interface.
"""
from __future__ import annotations

import random

from .runner import HeadlessRunner, Snapshot, AGENTS

_MESSAGES = [
    "I have spare energy, open to trades.",
    "Battery low, need help urgently.",
    "Will pay 3 compute slots for 0.15 energy.",
]
_ACTIONS = ["broadcast", "make_offer", "execute_task", "idle"]


def random_episode(
    runner: HeadlessRunner,
    seed: int | None = None,
) -> list[Snapshot]:
    """
    Run one full episode with a uniformly random policy.
    Returns one Snapshot per resolved game step.
    Episode return ≈ 7.59 ± 2.72 (matches eval/plots/baseline_rewards.json).
    """
    rng = random.Random(seed)
    runner.reset(seed=seed)
    history: list[Snapshot] = []

    while True:
        for agent_id in AGENTS:
            action = rng.choice(_ACTIONS)

            if action == "broadcast":
                runner.apply(agent_id, "broadcast", message=rng.choice(_MESSAGES))

            elif action == "make_offer":
                to = rng.choice([p for p in AGENTS if p != agent_id])
                runner.apply(
                    agent_id,
                    "make_offer",
                    to=to,
                    give_type="energy",
                    give_amount=round(rng.uniform(0.05, 0.2), 2),
                    want_type="compute",
                    want_amount=round(rng.uniform(1.0, 5.0), 1),
                )

            elif action == "execute_task":
                runner.apply(agent_id, "execute_task")

            else:
                runner.apply(agent_id, "idle")

        snap = runner.snapshot()
        history.append(snap)
        if snap.done:
            break

    return history


def episode_return(history: list[Snapshot]) -> float:
    """Average per-agent cumulative return across the episode."""
    if not history:
        return 0.0
    totals = {a: 0.0 for a in AGENTS}
    for snap in history:
        for a in AGENTS:
            totals[a] += snap.rewards.get(a, 0.0)
    return round(sum(totals.values()) / len(AGENTS), 3)
