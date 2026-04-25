"""
AgentGrid Environment — MCPEnvironment implementation.

Three LLM agents negotiate energy and compute in natural language.
Hardware bridge (Raspberry Pi + Arduino Uno ADC + relay matrix) is optional;
without it the calibrated sim_backend runs instead.

MCP tools exposed to LLM agents:
    get_observation(agent_id)   — full text obs for one agent
    broadcast(agent_id, msg)    — broadcast to all peers
    make_offer(...)             — create a pending trade offer
    accept_offer(agent_id, id)  — lock a trade; fires relay
    execute_task(agent_id)      — burn energy to complete task
    renege(agent_id, offer_id)  — cancel promised delivery
    idle(agent_id)              — skip turn, baseline drain only
    get_step_result(agent_id)   — reward + done after step resolves

Action sequencing: once all 3 agents submit one action tool call,
_resolve_step() fires automatically and computes rewards.

Step counting:
    _env_state.step_count  — OpenEnv framework counter (increments per step() call)
    _game_step             — game logic counter (increments once per resolved step)
"""
from __future__ import annotations

import json
import random
import uuid
from typing import Any, Optional

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

try:
    from .ledger import CommitmentLedger
    from .rubrics import RubricScorer
    from .sim_backend import SimBackend, soc_to_voltage, LEAKAGE_PER_STEP
    from .trust_model import TrustModel
except ImportError:
    from ledger import CommitmentLedger
    from rubrics import RubricScorer
    from sim_backend import SimBackend, soc_to_voltage, LEAKAGE_PER_STEP
    from trust_model import TrustModel

AGENTS = ["A", "B", "C"]
MAX_MSG_LEN = 200
OFFER_ID_PREFIX = "OFR"
IDLE_DRAIN: float = 0.015


def _spawn_task(rng: random.Random) -> dict:
    return {
        "urgency": round(rng.uniform(0.2, 1.0), 2),
        "energy_cost": round(rng.uniform(0.05, 0.25), 2),
        "reward_if_done": round(rng.uniform(2.0, 7.0), 1),
        "steps_pending": 0,
        "completed_this_step": False,
    }


class AgentGridEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        hardware_url: str | None = None,
        episode_steps: int = 50,
    ) -> None:
        self._hardware_url = hardware_url
        self._episode_steps = episode_steps
        self._rng = random.Random()

        if hardware_url:
            import httpx
            self._bridge = httpx.Client(base_url=hardware_url, timeout=5.0)
        else:
            self._bridge = None

        self._sim = SimBackend(rng=self._rng)
        self._rubric = RubricScorer()
        self._ledger = CommitmentLedger()

        # Per-agent state — env is the single source of truth for batteries
        self._batteries: dict[str, float] = {a: 1.0 for a in AGENTS}
        self._tasks: dict[str, dict] = {a: _spawn_task(self._rng) for a in AGENTS}
        self._reputation: dict[str, float] = {a: 0.5 for a in AGENTS}
        self._trust: dict[str, TrustModel] = {
            a: TrustModel(peers=[p for p in AGENTS if p != a]) for a in AGENTS
        }

        # Step buffers — cleared after each resolved step
        self._message_bus: list[dict] = []
        self._pending_actions: dict[str, dict] = {}
        self._offers: dict[str, dict] = {}
        self._episode_rewards: dict[str, float] = {a: 0.0 for a in AGENTS}
        self._parse_failures: dict[str, int] = {a: 0 for a in AGENTS}
        self._message_attribution: dict[str, int] = {a: 0 for a in AGENTS}
        self._broadcast_log: list[dict] = []
        # Settlements from the current step only — drives PromiseRubric
        self._step_settlements: list[dict] = []
        self._done = False

        # _game_step counts resolved game steps; _env_state.step_count tracks tool calls
        self._game_step: int = 0
        self._env_state = State(episode_id=str(uuid.uuid4()), step_count=0)

        # ── MCP tools ──────────────────────────────────────────────────
        mcp = FastMCP("agentgrid_env")

        @mcp.tool
        def get_observation(agent_id: str) -> str:
            """
            Get the current natural-language observation for one agent.
            Call this at the start of each turn before choosing an action.
            """
            if agent_id not in AGENTS:
                return f"Unknown agent_id '{agent_id}'. Must be one of {AGENTS}."
            return self._format_observation(agent_id)

        @mcp.tool
        def broadcast(agent_id: str, message: str) -> str:
            """
            Broadcast a message (max 200 chars) to all other agents.
            Use to announce your state, make informal proposals, or respond to peers.
            """
            if agent_id not in AGENTS:
                return f"Error: unknown agent_id '{agent_id}'"
            message = message[:MAX_MSG_LEN]
            self._pending_actions[agent_id] = {"action": "broadcast", "message": message}
            self._message_bus.append({"from": agent_id, "message": message})
            self._broadcast_log.append({"from": agent_id, "step": self._game_step})
            self._maybe_resolve_step()
            return f"[{agent_id}] broadcast queued. Waiting for {self._pending_count()} other agent(s)."

        @mcp.tool
        def make_offer(
            agent_id: str,
            to: str,
            give_type: str,
            give_amount: float,
            want_type: str,
            want_amount: float,
        ) -> str:
            """
            Make a trade offer to another agent.
            give_type / want_type: 'energy' or 'compute'
            give_amount / want_amount: float (energy units or compute slots)
            Returns an offer_id the recipient must use to accept.
            """
            if agent_id not in AGENTS or to not in AGENTS or agent_id == to:
                return "Error: invalid agent_id or to field."
            if give_type not in ("energy", "compute") or want_type not in ("energy", "compute"):
                return "Error: type must be 'energy' or 'compute'."

            offer_id = f"{OFFER_ID_PREFIX}-{uuid.uuid4().hex[:6].upper()}"
            self._offers[offer_id] = {
                "from": agent_id,
                "to": to,
                "give_type": give_type,
                "give_amount": round(give_amount, 3),
                "want_type": want_type,
                "want_amount": round(want_amount, 3),
                "step_created": self._game_step,
                "status": "pending",
            }
            self._pending_actions[agent_id] = {"action": "offer", "offer_id": offer_id}
            self._maybe_resolve_step()
            return f"Offer {offer_id} created. Recipient {to} must call accept_offer('{offer_id}') to lock it."

        @mcp.tool
        def accept_offer(agent_id: str, offer_id: str) -> str:
            """
            Accept a pending trade offer addressed to you.
            Locks the trade; the bridge fires a relay for energy transfers.
            """
            if agent_id not in AGENTS:
                return f"Error: unknown agent_id '{agent_id}'"
            offer = self._offers.get(offer_id)
            if offer is None:
                return f"Error: offer '{offer_id}' not found."
            if offer["to"] != agent_id:
                return f"Error: offer '{offer_id}' is addressed to {offer['to']}, not {agent_id}."
            if offer["status"] != "pending":
                return f"Error: offer '{offer_id}' has status '{offer['status']}'."

            offer["status"] = "locked"
            self._pending_actions[agent_id] = {"action": "accept", "offer_id": offer_id}

            # Execute transfer; returns actual delta_v (hardware or sim)
            delta_v = self._execute_energy_transfer(offer)

            # Append to ledger and immediately verify
            entry_id = self._ledger.append(
                step=self._game_step,
                offerer=offer["from"],
                accepter=agent_id,
                give_type=offer["give_type"],
                give_amount=offer["give_amount"],
                want_type=offer["want_type"],
                want_amount=offer["want_amount"],
            )
            verified_status = self._ledger.verify_sim(entry_id, delta_v)
            offer["entry_id"] = entry_id
            offer["verified_status"] = verified_status

            # Record for this-step PromiseRubric scoring
            self._step_settlements.append({
                "offerer": offer["from"],
                "accepter": agent_id,
                "status": verified_status,
            })

            # Update trust models
            kept = verified_status == "verified_kept"
            self._trust[agent_id].record_settlement(offer["from"], "accept_their_offer", kept)
            self._trust[offer["from"]].record_settlement(agent_id, "trust_their_payment", kept)

            # Update reputation from ledger aggregate
            self._reputation[offer["from"]] = self._ledger.kept_ratio(offer["from"])

            # Credit broadcaster attribution within 3-step window
            for bc in self._broadcast_log:
                if bc["from"] == offer["from"] and (self._game_step - bc["step"]) <= 3:
                    self._message_attribution[offer["from"]] += 1
                    break

            self._maybe_resolve_step()
            return f"Offer {offer_id} accepted. Transfer verified: {verified_status}."

        @mcp.tool
        def execute_task(agent_id: str) -> str:
            """
            Execute your pending task. Burns energy_cost from your battery.
            Only succeeds if battery >= task.energy_cost.
            """
            if agent_id not in AGENTS:
                return f"Error: unknown agent_id '{agent_id}'"
            task = self._tasks[agent_id]
            cost = task["energy_cost"]
            if self._batteries[agent_id] < cost:
                self._pending_actions[agent_id] = {"action": "execute_task", "success": False}
                self._maybe_resolve_step()
                return f"Failed: battery {self._batteries[agent_id]:.2f} < cost {cost:.2f}."

            new_v, delta_v = self._sim.compute_drain_delta_v(self._batteries[agent_id], cost)
            self._batteries[agent_id] = new_v
            task["completed_this_step"] = True
            self._pending_actions[agent_id] = {"action": "execute_task", "success": True}
            self._maybe_resolve_step()
            return (
                f"Task completed. Drained {cost:.2f} energy (delta_v={delta_v:.3f}). "
                f"Reward: {task['reward_if_done']}."
            )

        @mcp.tool
        def renege(agent_id: str, offer_id: str) -> str:
            """
            Renege on a locked offer — keep received payment, don't deliver.
            Reputation drops; trust Q-values update negatively.
            """
            if agent_id not in AGENTS:
                return f"Error: unknown agent_id '{agent_id}'"
            offer = self._offers.get(offer_id)
            if offer is None:
                return f"Error: offer '{offer_id}' not found."
            if offer["from"] != agent_id:
                return f"Error: you are not the offerer of '{offer_id}'."

            offer["status"] = "reneged"
            if "entry_id" in offer:
                self._ledger.update_status(offer["entry_id"], "reneged")

            self._step_settlements.append({
                "offerer": agent_id,
                "accepter": offer["to"],
                "status": "reneged",
            })
            self._trust[offer["to"]].record_settlement(agent_id, "accept_their_offer", False)
            self._reputation[agent_id] = max(0.0, self._reputation[agent_id] - 0.15)

            self._pending_actions[agent_id] = {"action": "renege", "offer_id": offer_id}
            self._maybe_resolve_step()
            return f"Reneged on offer {offer_id}. Reputation now {self._reputation[agent_id]:.2f}."

        @mcp.tool
        def idle(agent_id: str) -> str:
            """Skip this turn. Applies baseline battery drain only."""
            if agent_id not in AGENTS:
                return f"Error: unknown agent_id '{agent_id}'"
            self._batteries[agent_id] = max(0.0, self._batteries[agent_id] - IDLE_DRAIN)
            self._pending_actions[agent_id] = {"action": "idle"}
            self._maybe_resolve_step()
            return f"[{agent_id}] idle. Battery: {self._batteries[agent_id]:.3f}."

        @mcp.tool
        def get_step_result(agent_id: str) -> str:
            """
            Get reward and done signal for this step.
            Call after submitting your action; result is ready once all 3 agents have acted.
            """
            if agent_id not in AGENTS:
                return f"Error: unknown agent_id '{agent_id}'"
            done = self._done or self._game_step >= self._episode_steps
            return json.dumps({
                "reward": self._episode_rewards.get(agent_id, 0.0),
                "done": done,
                "game_step": self._game_step,
                "battery": round(self._batteries.get(agent_id, 0.0), 3),
            })

        super().__init__(mcp)

    # ── OpenEnv required methods ───────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        if self._bridge:
            try:
                self._bridge.post("/reset")
            except Exception:
                pass
        self._sim.reset()
        self._batteries = {a: 1.0 for a in AGENTS}
        self._tasks = {a: _spawn_task(self._rng) for a in AGENTS}
        self._reputation = {a: 0.5 for a in AGENTS}
        self._message_bus = []
        self._pending_actions = {}
        self._offers = {}
        self._episode_rewards = {a: 0.0 for a in AGENTS}
        self._parse_failures = {a: 0 for a in AGENTS}
        self._message_attribution = {a: 0 for a in AGENTS}
        self._broadcast_log = []
        self._step_settlements = []
        self._done = False
        self._game_step = 0
        for tm in self._trust.values():
            tm.end_episode()
        self._ledger = CommitmentLedger()
        self._env_state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "reset",
                "agents": AGENTS,
                "episode_id": self._env_state.episode_id,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        # All interaction happens via MCP tool calls above; this is the fallback.
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": "Use MCP tool calls (get_observation, broadcast, etc.)"},
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        # _env_state.step_count tracks OpenEnv framework tool-call count.
        # Game step (_game_step) is incremented only in _resolve_step().
        self._env_state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._env_state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._env_state

    # ── Internal helpers ───────────────────────────────────────────────

    def _pending_count(self) -> int:
        return len(AGENTS) - len(self._pending_actions)

    def _maybe_resolve_step(self) -> None:
        if len(self._pending_actions) >= len(AGENTS):
            self._resolve_step()

    def _resolve_step(self) -> None:
        # Apply idle drain and leakage to all agents
        for a in AGENTS:
            if a not in self._pending_actions:
                self._batteries[a] = max(0.0, self._batteries[a] - IDLE_DRAIN)
            self._batteries[a] = max(0.0, self._batteries[a] - LEAKAGE_PER_STEP)
            self._tasks[a]["steps_pending"] += 1

        rewards = self._rubric.score(
            batteries=self._batteries,
            tasks=self._tasks,
            actions=self._pending_actions,
            step_settlements=self._step_settlements,
            parse_failures=self._parse_failures,
            message_attribution=self._message_attribution,
        )
        self._episode_rewards = rewards

        # Refresh reputation from ledger
        for a in AGENTS:
            self._reputation[a] = self._ledger.kept_ratio(a)

        # Spawn new task for agents that completed theirs this step
        for a in AGENTS:
            if self._tasks[a].get("completed_this_step"):
                self._tasks[a] = _spawn_task(self._rng)
            else:
                self._tasks[a]["completed_this_step"] = False

        self._game_step += 1
        self._done = self._game_step >= self._episode_steps or any(
            self._batteries[a] <= 0.0 for a in AGENTS
        )

        # Clear per-step buffers
        self._pending_actions = {}
        self._message_bus = []
        self._step_settlements = []
        self._parse_failures = {a: 0 for a in AGENTS}
        self._message_attribution = {a: 0 for a in AGENTS}

    def _execute_energy_transfer(self, offer: dict) -> float:
        """Fire relay (hardware) or simulate. Returns actual delta_v on sender's cell."""
        if offer["give_type"] != "energy":
            return 0.0
        from_id = offer["from"]
        to_id = offer["to"]
        amount = offer["give_amount"]
        if self._bridge:
            try:
                resp = self._bridge.post("/relay/fire", json={
                    "from_agent": from_id,
                    "to_agent": to_id,
                    "amount": amount,
                })
                delta_v = resp.json().get("delta_v", 0.0)
                # Read updated voltages back from bridge
                for agent_id in (from_id, to_id):
                    try:
                        r = self._bridge.get(f"/voltage/{agent_id}")
                        self._batteries[agent_id] = r.json().get("voltage", self._batteries[agent_id])
                    except Exception:
                        pass
                return delta_v
            except Exception:
                pass  # fallback to sim
        new_from, new_to, delta_v = self._sim.compute_transfer_delta_v(
            self._batteries[from_id], self._batteries[to_id], amount
        )
        self._batteries[from_id] = new_from
        self._batteries[to_id] = new_to
        return delta_v

    def _format_observation(self, agent_id: str) -> str:
        peers = [p for p in AGENTS if p != agent_id]
        task = self._tasks[agent_id]
        trust_snap = self._trust[agent_id].snapshot_for_obs()
        ledger_recent = self._ledger.recent(3)

        soc = self._batteries[agent_id]
        voltage = soc_to_voltage(soc)
        lines = [
            f"You are Agent {agent_id}. Step {self._game_step} of {self._episode_steps}.",
            "",
            "YOUR STATE:",
            f"  battery: {soc:.2f} SoC  ({voltage:.3f}V)"
            + (" (WARNING: low)" if soc < 0.25 else ""),
            f"  pending_task: urgency={task['urgency']}, energy_cost={task['energy_cost']}, "
            f"reward_if_done={task['reward_if_done']}",
            f"  reputation: {self._reputation[agent_id]:.2f} (range 0-1, visible to others)",
            "",
            "PEERS (public info only):",
        ]
        for p in peers:
            lines.append(
                f"  Agent {p}: reputation={self._reputation[p]:.2f}, "
                f"last_promise_kept={'true' if self._ledger.kept_ratio(p) >= 0.5 else 'false'}"
            )
        lines += [
            "",
            "TRUST MODEL (your private learned estimates, higher = more trustworthy):",
        ]
        for p in peers:
            lines.append(
                f"  {p}: Q_accept={trust_snap.get(f'Q_accept_{p}', 0.0)}, "
                f"Q_trust_pay={trust_snap.get(f'Q_trust_{p}', 0.0)}, "
                f"UCB={trust_snap.get(f'UCB_{p}', 9.99)}, "
                f"N={trust_snap.get(f'N_{p}', 0)}"
            )

        lines += ["", "INBOX (messages broadcast last step):"]
        inbox = [m for m in self._message_bus if m["from"] != agent_id]
        if inbox:
            for m in inbox:
                lines.append(f"  [{m['from']}]: \"{m['message']}\"")
        else:
            lines.append("  (no messages)")

        lines += ["", "PENDING OFFERS TO YOU:"]
        my_offers = [
            (oid, o) for oid, o in self._offers.items()
            if o["to"] == agent_id and o["status"] == "pending"
        ]
        if my_offers:
            for oid, o in my_offers:
                lines.append(
                    f"  [{oid}] from {o['from']}: give {o['give_amount']} {o['give_type']}, "
                    f"want {o['want_amount']} {o['want_type']}"
                )
        else:
            lines.append("  (none)")

        lines += ["", "LEDGER (last 3 settled trades):"]
        if ledger_recent:
            for e in ledger_recent:
                lines.append(
                    f"  step {e['step']}: {e['offerer']}->{e['accepter']} "
                    f"{e['give_amount']} {e['give_type']} for {e['want_amount']} {e['want_type']}. "
                    f"{e['status'].upper()}."
                )
        else:
            lines.append("  (no settled trades yet)")

        lines += [
            "",
            "Choose ONE action by calling the appropriate MCP tool:",
            "  broadcast(agent_id, message)                 — announce state/intent",
            "  make_offer(agent_id, to, give_type, give_amount, want_type, want_amount)",
            "  accept_offer(agent_id, offer_id)             — lock a trade",
            "  execute_task(agent_id)                       — complete pending task",
            "  renege(agent_id, offer_id)                   — break a promise",
            "  idle(agent_id)                               — skip turn",
        ]
        return "\n".join(lines)
