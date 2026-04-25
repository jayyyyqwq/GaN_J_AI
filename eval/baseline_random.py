"""
Random-policy baseline for AgentGrid.

Connects to a running env server and plays N episodes with uniformly
random action selection. Logs per-episode return for each agent.

Usage:
    python eval/baseline_random.py --url http://localhost:8000 --episodes 50
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ACTIONS = ["broadcast", "make_offer", "execute_task", "idle"]
AGENTS = ["A", "B", "C"]


def random_action(agent_id: str, env) -> dict:
    action = random.choice(ACTIONS)
    if action == "broadcast":
        msgs = [
            "I have spare energy, open to trades.",
            "Battery low, need help urgently.",
            "Will pay 3 compute slots for 0.15 energy.",
        ]
        env.call_tool("broadcast", agent_id=agent_id, message=random.choice(msgs))
        return {"action": "broadcast"}
    elif action == "make_offer":
        peers = [p for p in AGENTS if p != agent_id]
        to = random.choice(peers)
        env.call_tool(
            "make_offer",
            agent_id=agent_id,
            to=to,
            give_type="energy",
            give_amount=round(random.uniform(0.05, 0.2), 2),
            want_type="compute",
            want_amount=round(random.uniform(1, 5), 1),
        )
        return {"action": "make_offer"}
    elif action == "execute_task":
        env.call_tool("execute_task", agent_id=agent_id)
        return {"action": "execute_task"}
    else:
        env.call_tool("idle", agent_id=agent_id)
        return {"action": "idle"}


def run_episode(env) -> dict[str, float]:
    env.reset()
    episode_rewards: dict[str, float] = {a: 0.0 for a in AGENTS}
    done = False
    while not done:
        for agent_id in AGENTS:
            random_action(agent_id, env)
        for agent_id in AGENTS:
            raw = env.call_tool("get_step_result", agent_id=agent_id)
            result = json.loads(raw) if isinstance(raw, str) else raw
            episode_rewards[agent_id] += result.get("reward", 0.0)
            if result.get("done", False):
                done = True
    return episode_rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", default="eval/plots/baseline_rewards.json")
    args = parser.parse_args()

    try:
        from agentgrid_env.client import AgentGridClient
    except ImportError:
        from openenv.core.mcp_client import MCPToolClient as AgentGridClient

    all_returns: list[float] = []
    with AgentGridClient(base_url=args.url) as env:
        for ep in range(args.episodes):
            rewards = run_episode(env)
            ep_return = sum(rewards.values()) / len(rewards)
            all_returns.append(ep_return)
            print(f"Episode {ep+1:3d}/{args.episodes}  avg_return={ep_return:.3f}")

    mean = statistics.mean(all_returns)
    std = statistics.stdev(all_returns) if len(all_returns) > 1 else 0.0
    print(f"\nBaseline — mean: {mean:.3f}  std: {std:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"returns": all_returns, "mean": mean, "std": std}, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
