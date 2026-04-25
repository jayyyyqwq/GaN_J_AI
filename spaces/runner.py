"""
HeadlessRunner — drives AgentGridEnvironment in-process without HTTP.

Replicates each MCP tool closure's logic by calling env private state directly.
Zero changes to agentgrid_env package required.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentgrid_env.server.agentgrid_environment import AgentGridEnvironment, AGENTS

# Must match IDLE_DRAIN in agentgrid_environment.py
_IDLE_DRAIN: float = 0.015


@dataclass
class Snapshot:
    game_step: int
    done: bool
    batteries: dict[str, float]
    reputation: dict[str, float]
    rewards: dict[str, float]          # per-step rubric scores for each agent
    recent_ledger: list[dict]          # last 5 entries from CommitmentLedger
    step_messages: list[dict]          # broadcasts: [{"from": str, "message": str}]
    step_actions: list[dict]           # all actions: [{"agent": str, "action": str, "detail": str}]
    promise_keep_ratio: float          # aggregate kept_ratio across all agents


class HeadlessRunner:
    """
    In-process driver for AgentGridEnvironment. Construct one per Gradio session.
    Callers drive the env by calling apply() for each agent in turn.
    When all 3 agents have submitted, the step resolves automatically inside apply().
    """

    def __init__(self, episode_steps: int = 50) -> None:
        self._episode_steps = episode_steps
        self._env: AgentGridEnvironment | None = None
        # Per-step accumulators (cleared on step resolution)
        self._pending_messages: list[dict] = []
        self._step_action_log: list[dict] = []
        # Preserved from the last resolved step (for snapshot())
        self._resolved_messages: list[dict] = []
        self._resolved_action_log: list[dict] = []

    def reset(self, seed: int | None = None) -> Snapshot:
        self._env = AgentGridEnvironment(
            hardware_url=None,
            episode_steps=self._episode_steps,
        )
        self._env.reset(seed=seed)
        self._pending_messages = []
        self._step_action_log = []
        self._resolved_messages = []
        self._resolved_action_log = []
        return self.snapshot()

    def apply(self, agent_id: str, action: str, **kwargs) -> str:
        """
        Submit one agent's action for the current step.
        Returns the tool's response string (offer_id for make_offer, otherwise descriptive).
        When the 3rd agent acts, the step resolves automatically.
        """
        assert self._env is not None, "call reset() first"
        env = self._env
        step_before = env._game_step

        if action == "broadcast":
            self._pending_messages.append({
                "from": agent_id,
                "message": kwargs.get("message", "")[:200],
            })

        ret = self._dispatch(agent_id, action, **kwargs)

        self._step_action_log.append({
            "agent": agent_id,
            "action": action,
            "detail": ret,
            "kwargs": {k: v for k, v in kwargs.items() if k != "message"},
        })

        if env._game_step > step_before:
            self._resolved_messages = self._pending_messages[:]
            self._resolved_action_log = self._step_action_log[:]
            self._pending_messages = []
            self._step_action_log = []

        return ret

    def snapshot(self) -> Snapshot:
        env = self._env
        assert env is not None
        done = env._done or env._game_step >= env._episode_steps
        ratios = [env._ledger.kept_ratio(a) for a in AGENTS]
        return Snapshot(
            game_step=env._game_step,
            done=done,
            batteries=dict(env._batteries),
            reputation=dict(env._reputation),
            rewards=dict(env._episode_rewards),
            recent_ledger=env._ledger.recent(5),
            step_messages=list(self._resolved_messages),
            step_actions=list(self._resolved_action_log),
            promise_keep_ratio=round(sum(ratios) / len(ratios), 3),
        )

    # ── Action dispatch — mirrors each MCP tool closure exactly ───────────────

    def _dispatch(self, agent_id: str, action: str, **kwargs) -> str:
        handlers = {
            "broadcast": self._broadcast,
            "make_offer": self._make_offer,
            "accept_offer": self._accept_offer,
            "execute_task": self._execute_task,
            "renege": self._renege,
            "idle": self._idle,
        }
        fn = handlers.get(action)
        if fn is None:
            self._env._pending_actions[agent_id] = {"action": "idle"}
            self._env._maybe_resolve_step()
            return f"Unknown action '{action}', fell back to idle"
        return fn(agent_id, **kwargs)

    def _broadcast(self, agent_id: str, message: str = "") -> str:
        env = self._env
        msg = message[:200]
        env._pending_actions[agent_id] = {"action": "broadcast", "message": msg}
        env._message_bus.append({"from": agent_id, "message": msg})
        env._broadcast_log.append({"from": agent_id, "step": env._game_step})
        env._maybe_resolve_step()
        return f"[{agent_id}]: {msg}"

    def _make_offer(
        self,
        agent_id: str,
        to: str,
        give_type: str = "energy",
        give_amount: float = 0.1,
        want_type: str = "compute",
        want_amount: float = 1.0,
    ) -> str:
        env = self._env
        offer_id = f"OFR-{uuid.uuid4().hex[:6].upper()}"
        env._offers[offer_id] = {
            "from": agent_id,
            "to": to,
            "give_type": give_type,
            "give_amount": round(float(give_amount), 3),
            "want_type": want_type,
            "want_amount": round(float(want_amount), 3),
            "step_created": env._game_step,
            "status": "pending",
        }
        env._pending_actions[agent_id] = {"action": "offer", "offer_id": offer_id}
        env._maybe_resolve_step()
        # Return offer_id so scripted_player.py can pass it to accept_offer
        return offer_id

    def _accept_offer(self, agent_id: str, offer_id: str = "") -> str:
        env = self._env
        offer = env._offers.get(offer_id)
        if offer is None or offer["to"] != agent_id or offer["status"] != "pending":
            env._pending_actions[agent_id] = {"action": "idle"}
            env._maybe_resolve_step()
            return f"Error: cannot accept '{offer_id}'"

        offer["status"] = "locked"
        env._pending_actions[agent_id] = {"action": "accept", "offer_id": offer_id}

        delta_v = env._execute_energy_transfer(offer)

        entry_id = env._ledger.append(
            step=env._game_step,
            offerer=offer["from"],
            accepter=agent_id,
            give_type=offer["give_type"],
            give_amount=offer["give_amount"],
            want_type=offer["want_type"],
            want_amount=offer["want_amount"],
        )
        verified = env._ledger.verify_sim(entry_id, delta_v)
        offer["entry_id"] = entry_id
        offer["verified_status"] = verified

        env._step_settlements.append({
            "offerer": offer["from"],
            "accepter": agent_id,
            "status": verified,
        })

        kept = verified == "verified_kept"
        env._trust[agent_id].record_settlement(offer["from"], "accept_their_offer", kept)
        env._trust[offer["from"]].record_settlement(agent_id, "trust_their_payment", kept)
        env._reputation[offer["from"]] = env._ledger.kept_ratio(offer["from"])

        for bc in env._broadcast_log:
            if bc["from"] == offer["from"] and (env._game_step - bc["step"]) <= 3:
                env._message_attribution[offer["from"]] += 1
                break

        env._maybe_resolve_step()
        return f"Accepted {offer_id}: {verified}"

    def _execute_task(self, agent_id: str) -> str:
        env = self._env
        task = env._tasks[agent_id]
        cost = task["energy_cost"]
        if env._batteries[agent_id] < cost:
            env._pending_actions[agent_id] = {"action": "execute_task", "success": False}
            env._maybe_resolve_step()
            return f"Failed: battery {env._batteries[agent_id]:.2f} < cost {cost:.2f}"
        new_soc, delta_v = env._sim.compute_drain_delta_v(env._batteries[agent_id], cost)
        env._batteries[agent_id] = new_soc
        task["completed_this_step"] = True
        env._pending_actions[agent_id] = {"action": "execute_task", "success": True}
        env._maybe_resolve_step()
        return f"Task done. Drained {cost:.2f} (Δv={delta_v:.3f}). Reward={task['reward_if_done']}"

    def _renege(self, agent_id: str, offer_id: str = "") -> str:
        env = self._env
        offer = env._offers.get(offer_id)
        if offer and offer["from"] == agent_id:
            offer["status"] = "reneged"
            if "entry_id" in offer:
                env._ledger.update_status(offer["entry_id"], "reneged")
            env._step_settlements.append({
                "offerer": agent_id,
                "accepter": offer["to"],
                "status": "reneged",
            })
            env._trust[offer["to"]].record_settlement(agent_id, "accept_their_offer", False)
            env._reputation[agent_id] = max(0.0, env._reputation[agent_id] - 0.15)
        env._pending_actions[agent_id] = {"action": "renege", "offer_id": offer_id}
        env._maybe_resolve_step()
        return f"Reneged on {offer_id}. Rep={env._reputation.get(agent_id, 0.5):.2f}"

    def _idle(self, agent_id: str) -> str:
        env = self._env
        env._batteries[agent_id] = max(0.0, env._batteries[agent_id] - _IDLE_DRAIN)
        env._pending_actions[agent_id] = {"action": "idle"}
        env._maybe_resolve_step()
        return f"[{agent_id}] idle. Battery={env._batteries[agent_id]:.3f}"
