# AgentGrid v2 — Physically-Grounded LLM Negotiation

> **Thesis:** Three small LLM agents negotiate in natural language for physical energy and compute on real hardware. The hardware enforces consequences — when an LLM promises power, a relay must actually fire. Lies become physically detectable. Trust becomes a learned, verifiable signal. This is the only OpenEnv submission where the reward function is partially backed by Ohm's law.

---

## 0. What changed from v1, and why

| v1 problem | v2 fix |
|---|---|
| Treated as classical MARL with discrete actions | LLM agents communicating in natural language (matches Unsloth/TRL/the hackathon's actual purpose) |
| `peer_broadcasts` exposed needs as boolean flags → no ToM possible | Peers exchange free-text messages; needs must be inferred from language |
| 4-action discrete space → trivially solved by lookup table | Continuous offers (price × quantity) parsed from JSON tool calls |
| N=2 → coordination, not negotiation | N=3 → coalitions, exclusion, multi-party bargaining become possible |
| Hardware was decorative (relay fired *after* policy decided) | Hardware is in-loop: real battery voltage IS the observation; relay timing IS the reward signal |
| Defection impossible → trust is meaningless → blockchain pointless | Agents *can* renege on promises; hash-chained ledger on RPi makes reputation a learned variable; this is now the *only* version where a ledger earns its place |
| Sparse reward → pacifism equilibrium | Per-step task-decay penalty + reputation-weighted reward kills passive strategies |
| TRL on discrete-action MARL = wrong tool | GRPO via Unsloth on small LLMs = correct tool, matches hackathon stack |
| 72h timeline assumed everything works first try | Timeline reorganized: pure-sim training Day 1, hardware bridge Day 2, HITL fine-tune + demo polish Day 3 |

---

## 1. Why this maps cleanly to the judging rubric

### 1.1 Environment Innovation (40%)
- **Novelty axis 1:** LLM agents negotiating in *natural language* over *physical resources*. Searched: nobody has shipped this on OpenEnv. Closest prior art is text-only market sims (no hardware) and RL energy-sharing (no language).
- **Novelty axis 2:** Reward is partially derived from a physical quantity (battery voltage measured via ADC). The environment cannot be cheated by gaming the simulator because the simulator IS reality.
- **Challenge:** Partial observability is real — agents see only their own battery + received messages. Other agents' true state must be inferred from what they say (theory of mind that the rubric explicitly asks for).
- **Researcher test:** "Could a researcher write a paper about training on this?" Yes — *Grounded Multi-Agent Negotiation: When LLM Promises Have Physical Consequences*. That's a real ICLR/NeurIPS workshop submission.

### 1.2 Storytelling (30%)
The 90-second demo (Section 9) is built around a single moment: judge waves hand at the ultrasonic sensor → Agent C's task urgency spikes → Agent C broadcasts a panicked plea in English → Agents A and B reason aloud about reputation history → A accepts, the relay clicks audibly, C's LED brightens, C completes the task, A's reputation score increments on screen. That sequence is unforgettable.

### 1.3 Reward Improvement (20%)
We will show three curves on the same axes:
1. **Random baseline** (untrained LLM, JSON-valid random actions)
2. **Pre-train** (LLM after supervised warmup on synthetic negotiation traces)
3. **Post-GRPO** (LLM after on-environment RL)

Plus a **before/after demo replay** of the same scripted scenario showing the trained agent forming a coalition where the baseline died.

### 1.4 Reward & Pipeline (10%)
Reward is a composable rubric (per OpenEnv's preferred pattern), not a monolithic scalar:
- Survival rubric (binary)
- Task-completion rubric (graded by urgency)
- Promise-keeping rubric (derived from hardware voltage ground truth)
- Communication-quality rubric (parses to valid JSON)

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  TRAINING LAYER (Colab + HF Spaces)                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ LLM Agent A  │    │ LLM Agent B  │    │ LLM Agent C  │        │
│  │ Llama-3.2-1B │    │ Llama-3.2-1B │    │ Llama-3.2-1B │        │
│  │ + LoRA (GRPO)│    │ + LoRA (GRPO)│    │ + LoRA (GRPO)│        │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘        │
│         │ JSON actions      │                   │                │
│         └────────────┬──────┴───────────────────┘                │
│                     ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  AgentGridEnv (OpenEnv-compliant, MCPEnvironment subclass) │  │
│  │  - reset() / step() / state()                              │  │
│  │  - Composable rubric scorer                                │  │
│  │  - HardwareBridge client (HTTP)                            │  │
│  └────────────────────────────────┬───────────────────────────┘  │
└───────────────────────────────────┼──────────────────────────────┘
                                    │ HTTPS / ngrok
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  BRIDGE LAYER (Raspberry Pi 3 on home WiFi)                      │
│  - FastAPI server (async)                                        │
│  - CommitmentLedger (hash-chained SQLite)                        │
│  - Hardware orchestration: relay matrix, ADC reads, sensor poll  │
│  - Voltage-truth loop: confirms physical promise fulfillment     │
└────────┬───────────────┬────────────────┬───────────────┬────────┘
         │ I2C/GPIO      │ GPIO           │ GPIO          │ GPIO
         ▼               ▼                ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Agent A      │ │ Agent B      │ │ Agent C      │ │ Relay Matrix │
│ NodeMCU+LED  │ │ NodeMCU+LED  │ │ NodeMCU+LED  │ │ 4-channel    │
│ + INA219 ADC │ │ + INA219 ADC │ │ + INA219 ADC │ │ + Ultrasonic │
│ + 18650 cell │ │ + 18650 cell │ │ + 18650 cell │ │   HC-SR04    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

**The split is enforced:** the LLM agents *only* see what comes through the env's observation. The env *only* knows hardware state via the bridge. Client/server separation per the OpenEnv judging note.

---

## 3. Hardware BOM (revised)

| Component | Qty | Role | Why this and not v1's choice |
|---|---|---|---|
| NodeMCU ESP8266 | **3** | Agents A/B/C | N=3 unlocks coalition dynamics |
| Raspberry Pi 3 | 1 | Bridge + ledger host | Same as v1 |
| INA219 current/voltage sensor | **3** | Real-time per-cell voltage truth | **New — this is what makes hardware in-loop.** Reads actual cell voltage so the env knows whether a promised energy transfer happened |
| 4-channel relay module | **1** (replaces single relay) | Routes power between any pair of cells | Enables the bidirectional trades a 3-agent system needs |
| 18650 cell | 3 | One per agent | One per agent (v1 lumped them) |
| TP4056 charging module | 3 | Per-cell safe charging during demo | Lets you reset between demo runs without swapping cells |
| HC-SR04 ultrasonic | 1 | Urgency injection (judge gesture) | Same as v1 — keep it, it's the demo trigger |
| White LEDs + 220Ω resistors | 3 | Per-agent status indication | Same as v1 |
| Jumper wires (M-F, M-M) | bundle | Wiring | No breadboard — solder or use screw terminals for travel |

**Cost delta from v1:** ~₹1,500 extra (INA219 ×3, 4-ch relay, TP4056 ×3). Worth it; without INA219 the hardware loop isn't actually closed.

**Excluded:** Arduino boards (logic-level mismatch). Any 5V-only module on the ESP8266 side. Bluetooth (WiFi is enough; one less radio to debug).

---

## 4. Software: Environment Specification

### 4.1 OpenEnv class skeleton

```python
# env/agentgrid_env.py
from openenv import MCPEnvironment, Rubric
from typing import Dict, List, Any
import httpx, json, time

class AgentGridEnv(MCPEnvironment):
    """3-agent natural-language negotiation env with optional hardware backend."""

    def __init__(self, hardware_url: str | None = None, episode_steps: int = 50):
        super().__init__()
        self.bridge = httpx.Client(base_url=hardware_url) if hardware_url else None
        self.episode_steps = episode_steps
        self.agents = ["A", "B", "C"]
        self.rubric = self._build_rubric()

    def reset(self) -> Dict[str, Any]:
        if self.bridge:
            self.bridge.post("/reset")  # recharges to known state, clears ledger
        self.t = 0
        self.state_cache = self._poll_world()
        self.message_bus: List[Dict] = []  # last-step broadcasts
        return self._observations()

    def step(self, actions: Dict[str, str]) -> tuple:
        # actions: {"A": "<json action>", "B": "...", "C": "..."}
        parsed = {a: self._parse_action(actions[a]) for a in self.agents}
        self._dispatch_to_hardware(parsed)        # fires relays, logs to ledger
        time.sleep(0.5)                           # let physics settle
        self.state_cache = self._poll_world()     # pull voltages, sensor
        rewards = self.rubric.score(parsed, self.state_cache, self.message_bus)
        self.t += 1
        done = self.t >= self.episode_steps or self._any_dead()
        return self._observations(), rewards, done, {"ledger": self._ledger_view()}

    # ... _build_rubric, _parse_action, _observations, etc.
```

### 4.2 Observation (per-agent, text-formatted for the LLM)

```
You are Agent A. Step 17 of 50.

YOUR STATE:
  battery: 0.62 (warning if <0.25)
  pending_task: urgency=0.81, energy_cost=0.15, reward_if_done=5.0
  reputation: 0.83 (range 0-1, others see this)

PEERS (only what's public):
  Agent B: reputation=0.91, last_promise_kept=true
  Agent C: reputation=0.42, last_promise_kept=false

INBOX (messages broadcast last step):
  [B]: "I have spare energy this round. Anyone urgent?"
  [C]: "Need 0.2 energy units NOW, will pay 0.5 compute priority next 3 rounds"

LEDGER (last 3 settled trades):
  step 14: A->B 0.10 energy for 1 compute slot. KEPT.
  step 15: C->A 0.05 energy for 2 compute slots. RENEGED by C.
  step 16: B->C 0.20 energy for nothing yet. PENDING.

Choose ONE action. Respond ONLY with valid JSON:
  {"action": "broadcast", "message": "<text, max 200 chars>"}
  {"action": "offer", "to": "B|C", "give": {"type": "energy|compute", "amount": float}, "want": {"type": "...", "amount": float}}
  {"action": "accept", "offer_id": "<id from inbox>"}
  {"action": "execute_task"}
  {"action": "renege", "offer_id": "<id>"}    # take payment, don't deliver
  {"action": "idle"}
```

### 4.3 Action space (JSON tool calls)

| Action | Effect | Hardware consequence |
|---|---|---|
| `broadcast` | Adds message to next-step inbox of all peers | None |
| `offer` | Creates pending offer in ledger, awaiting peer's `accept` | None until accepted |
| `accept` | Locks the trade; bridge schedules relay fire | Relay routes power between cells |
| `execute_task` | Burns energy, completes task if battery sufficient | Voltage drop visible on INA219 |
| `renege` | Cancel a promised delivery after receiving counter-payment | Ledger marks as broken; reputation drops |
| `idle` | Skip turn, baseline drain only | Small voltage drop |

### 4.4 Rubric (composable per OpenEnv guidance)

```python
def _build_rubric(self):
    return Rubric([
        SurvivalRubric(weight=1.0),         # +0 alive, -10 if battery hit 0
        TaskRubric(weight=1.0),             # urgency * 5 on completion, -urgency*0.3/step pending
        PromiseRubric(weight=0.8),          # +1 kept (verified by INA219 voltage drop), -3 reneged
        JsonValidityRubric(weight=0.2),     # -0.5 per parse failure (curriculum trick)
        CommunicationRubric(weight=0.3),    # +0.1 if message led to a settled trade within 3 steps
    ])
```

**Why the PromiseRubric is the magic:** the env asks the bridge "did Agent A's cell voltage actually drop by ~0.10 between step 14 and step 15?" If yes, promise verified physically. If no, the agent is silently lying and gets penalized regardless of what the ledger says it did. This is the closure that makes hardware non-decorative.

### 4.5 Sim-only fallback

If `hardware_url` is None, the env falls back to a calibrated simulator (battery curves fit from real INA219 logs collected on Day 2). This lets training run massively in parallel on Colab; HITL fine-tuning happens at the end with the real rig connected. Submission requirement: env runs end-to-end on HF Spaces without hardware.

---

## 5. Software: Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Synthetic SFT warmup (Colab, ~30 min)              │
│ - Generate 2k synthetic negotiation transcripts via GPT-4o  │
│   prompted with the env rules                               │
│ - SFT a Llama-3.2-1B base via Unsloth (LoRA, r=16)          │
│ - Goal: agents output valid JSON, basic negotiation grammar │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Self-play GRPO in pure sim (Colab, ~3 h)           │
│ - 3 copies of the SFT model self-play in AgentGridEnv (sim) │
│ - GRPO via TRL/Unsloth with the composable rubric           │
│ - Track: episode return, promise-keep rate, task completion │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: HITL fine-tune (last 1 h before demo)              │
│ - Reconnect env to RPi bridge                               │
│ - 200 episodes on real hardware                             │
│ - Captures real battery noise, WiFi jitter, sensor reality  │
│ - This is the curve that beats the sim baseline             │
└─────────────────────────────────────────────────────────────┘
```

**Three plotted curves judges will see** (same axes, README + slides):
1. Random-policy baseline
2. Sim-trained policy evaluated on hardware
3. Sim+HITL policy evaluated on hardware

Expected story: (3) > (2) > (1) on episode return AND on promise-keep rate. The gap between (2) and (3) is the *evidence that the hardware mattered*.

---

## 6. The Commitment Ledger (lightweight, hardware-verified)

Not a blockchain. A single hash-chained append-only SQLite log on the Pi:

```python
# bridge/ledger.py
import hashlib, sqlite3, json, time

class CommitmentLedger:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY,
        step INTEGER,
        offerer TEXT, accepter TEXT,
        give_type TEXT, give_amount REAL,
        want_type TEXT, want_amount REAL,
        status TEXT,                    -- pending|kept|reneged|verified_kept|verified_broken
        prev_hash TEXT, this_hash TEXT,
        ts REAL
    )
    """
    def append(self, **fields):
        prev = self._latest_hash()
        payload = json.dumps(fields, sort_keys=True) + prev
        h = hashlib.sha256(payload.encode()).hexdigest()
        # insert with prev_hash=prev, this_hash=h
        ...

    def verify_against_hardware(self, entry_id, ina219_reading):
        """Did the promised voltage drop actually happen on the offerer's cell?"""
        entry = self._get(entry_id)
        expected_drop = entry["give_amount"] * VOLTS_PER_ENERGY_UNIT
        actual_drop = ina219_reading["delta_v"]
        if abs(actual_drop - expected_drop) < TOLERANCE:
            self._update_status(entry_id, "verified_kept")
        else:
            self._update_status(entry_id, "verified_broken")
```

**Why this earns its place** (unlike v1's blockchain idea):
- N=3 + defection-allowed creates real Byzantine-ish trust problem
- Hash chain prevents an agent from rewriting its own history
- INA219 readings are the oracle — physical reality is the source of truth, not consensus
- Reputation in the observation is computed from the verified-kept ratio
- Judges can audit any episode end-to-end

This is a defensible "lightweight verifiable ledger" claim, not blockchain hype.

---

## 7. Demo Storytelling Protocol (the 90-second pitch)

Three scenes. Rehearse twice on Day 3.

**Scene 1 (0:00–0:25) — Setup.** Camera pans the rig. Three NodeMCUs, three cells, relay matrix, ultrasonic sensor. Voiceover: *"Three small LLMs. One battery each. They have to keep each other alive by negotiating in plain English. The hardware enforces every promise."*

**Scene 2 (0:25–1:05) — The hand wave.** Live transcript projected on screen showing real LLM messages:
```
[A]: "Battery comfortable, 0.74. Open to trades."
[B]: "Same here, 0.81. Anyone need help?"
[Judge gestures at ultrasonic sensor]
[C]: "URGENT: high-priority task incoming, urgency 0.93, battery 0.31. 
      Will pay 3 compute slots for 0.15 energy."
[A]: "C, your reputation is 0.42 after last episode's renege. 
      I'll help but I want 5 slots, not 3."
[C]: "Accepted. Sending offer."
```
Relay clicks audibly. C's LED brightens. C's task LED flashes green (completed). A's reputation counter ticks down on screen (cost of trust).

**Scene 3 (1:05–1:30) — The curves.** Cut to slide. Three reward curves overlaid. Voiceover: *"The trained agent didn't just learn to survive — it learned that reputation is a currency, and that the cheapest energy comes from being trustworthy. Here's the proof: promise-keep rate goes from 38% baseline to 91% post-training."*

End on: *"AgentGrid. The first OpenEnv environment where lying has a voltage."*

---

## 8. 72-Hour Execution Timeline (realistic, single-person)

### Day 1 (April 24) — Sim env + Stage 1 SFT
- **Morning (4 h):** Implement `AgentGridEnv` in pure sim. Get `reset()`/`step()` returning valid obs/rewards. Validate with random JSON actions.
- **Afternoon (3 h):** Generate synthetic negotiation transcripts (GPT-4o API, ~₹100). SFT Llama-3.2-1B in Colab with Unsloth. Confirm valid JSON output rate >90%.
- **Evening (2 h):** Wire the rig physically. Solder JST connectors to cells. Mount relay module. Test relay click via raw GPIO from Pi. **No software integration tonight.**
- **Cutoff:** If SFT model can't produce valid JSON 90%+ of the time by midnight, freeze the env interface and move on with what works.

### Day 2 (April 25) — Bridge + Stage 2 GRPO
- **Morning (3 h):** FastAPI server on Pi. INA219 polling endpoint. Relay-fire endpoint. Ledger init. Test end-to-end with hand-crafted HTTP calls.
- **Midday (1 h):** Connect env to bridge. Run 5 manual episodes to log real battery curves → fit sim parameters.
- **Afternoon onsite (4 h):** Push env to HF Spaces. Kick off GRPO self-play training in Colab using onsite credits. Three parallel runs with different seeds.
- **Evening (2 h):** Pick best checkpoint. Verify it runs on Spaces without hardware.

### Day 3 (April 26) — HITL fine-tune + demo
- **Morning (2 h):** HITL fine-tuning loop on real rig (200 episodes).
- **Midday (2 h):** Generate the three reward curves. Write README. Record < 2 min YouTube.
- **Afternoon (3 h):** Demo rehearsal x3. Build the live-transcript projector script. Charge cells fully. Pack the rig.
- **Buffer:** 4 h reserved for "the WiFi at the venue is hostile."

### Hard cuts if behind schedule
1. Drop HITL fine-tune; ship sim-only model. Demo still works because relay fires from accepted offers regardless.
2. Drop reputation rubric; agents still negotiate, just without learned trust.
3. Drop the 4-channel relay; use single relay between A↔B only. Demo loses one trade direction but core story survives.

---

## 9. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Venue WiFi blocks the bridge | High | RPi runs its own hotspot; laptop joins it |
| ESP8266 brownouts mid-trade | Medium | TP4056 boards keep cells topped above 3.4V floor between episodes |
| LLM produces invalid JSON | Medium | JsonValidityRubric trains it out; fallback parser strips markdown fences |
| GRPO doesn't improve over SFT | Medium | Show the curve anyway; SFT alone with the rubric still demonstrates env quality |
| Ledger desync between agents and bridge | Low | Bridge is single source of truth; agents see ledger only via observation, not local state |
| Relay welding (stuck closed) | Low | Use SSR-rated 5V relay; INA219 detects stuck-closed and aborts episode |
| Demo cell dies during pitch | Low | Spare charged cell on standby, swap takes 15 sec |

---

## 10. File Structure

```
agentgrid/
├── README.md                       # the story, plots, links to video/blog/space
├── openenv.yaml                    # required manifest
├── env/
│   ├── __init__.py
│   ├── agentgrid_env.py            # MCPEnvironment subclass
│   ├── rubrics.py                  # composable rubric definitions
│   ├── sim_backend.py              # battery curve simulator (calibrated from real data)
│   └── bridge_client.py            # HTTP client to RPi
├── bridge/                          # runs on the Pi
│   ├── server.py                   # FastAPI app
│   ├── ledger.py                   # hash-chained SQLite
│   ├── hardware.py                 # INA219 + relay + ultrasonic drivers
│   └── calibration.json
├── firmware/
│   ├── nodemcu_agent.ino           # LED status + heartbeat to Pi
│   └── README.md                   # flash instructions
├── training/
│   ├── 01_sft_warmup.ipynb         # Colab, Unsloth SFT
│   ├── 02_grpo_selfplay.ipynb      # Colab, GRPO via TRL
│   ├── 03_hitl_finetune.ipynb      # connects to live bridge
│   └── synthetic_traces/           # SFT data, gitignored if large
├── eval/
│   ├── baseline_random.py
│   ├── replay_demo_scenario.py
│   └── plots/                      # the three curves, .png, committed
├── demo/
│   ├── transcript_projector.py     # live LLM message viewer for the pitch
│   └── scripted_scenario.json
└── docs/
    ├── architecture.png
    └── demo_storyboard.md
```

---

## 11. Submission Checklist (mapped to minimum requirements)

- [ ] OpenEnv latest release, `MCPEnvironment` subclass, valid `openenv.yaml`
- [ ] No reserved tool names (`reset`/`step`/`state`/`close`) used for MCP tools
- [ ] Client/server separation enforced (env never imports bridge internals)
- [ ] Colab notebook training script using Unsloth + TRL GRPO
- [ ] Real training run with logged curves committed as PNG to `eval/plots/`
- [ ] Three-way comparison plot (random / sim-trained / HITL-trained)
- [ ] HF Space hosting the env, runnable via the standard env client
- [ ] README with: problem, env description, results, links
- [ ] < 2 min YouTube linked from README
- [ ] HF mini-blog linked from README
- [ ] Video file NOT in the HF env repo (link only)

---

## 12. The One Sentence That Sells It

> *AgentGrid is the only OpenEnv submission where the reward function reads from a voltmeter — three small LLMs negotiate energy and compute in plain English, and a relay clicks every time a promise is kept.*

If a judge remembers one line, that's the line.

# AgentGrid v2.1 — Targeted Technical Upgrades

> **What changed:** Added a per-agent tabular trust learner (Q-learning with Monte Carlo returns, UCB1 for peer selection) that feeds its output into the LLM's observation. No new training loop, no competing optimizer, no extra hyperparameters that need tuning. The LLM's GRPO loop is unchanged. The classical RL layer handles per-peer credit assignment, which GRPO handles badly over 50-step horizons with sparse rewards.
>
> **What did NOT change:** Everything else. Hardware BOM, OpenEnv class skeleton, the 3-stage training pipeline, the demo protocol, the risk register. v2.1 is surgical.

---

## 0. Why these specific three concepts

You asked for Monte Carlo + explore/exploit + SARSA-or-Q-learning. Here is the honest mapping to the problem:

| Concept | Where it fits in AgentGrid | Why it's the right tool |
|---|---|---|
| **Q-learning (tabular)** | Per-peer trust value function, updated online each time a promise settles | Small discrete state space (3 peers × 2 action types). Converges in dozens of interactions. Off-policy by construction (learning from what peers did, not from own policy). Avoids the co-evolution instability that bites on-policy methods in multi-agent settings. |
| **Monte Carlo returns** | Used at episode end to reconcile the in-episode TD updates with the full trajectory return | Episodes are short (50 steps), rewards are sparse (promise-keep verified only on settlement), credit propagates across negotiation turns. MC returns handle long-range attribution without bootstrapping noise. |
| **UCB1 explore/exploit** | Peer selection signal surfaced to the LLM as confidence bounds | Each peer is effectively an arm. UCB1 gives a principled exploration bonus that decays as interaction count grows. Epsilon-greedy would explore uniformly and waste budget on well-characterized peers. |

All three do real work. None of them duplicate what GRPO is already doing.

**Why not SARSA:** SARSA is on-policy, which means the trust Q-table would track only what the agent's own policy does. We want the trust Q-table to estimate what *peers* do, regardless of the agent's policy choices. Q-learning's off-policy semantics match that. SARSA would be the right call if we were learning the agent's own action-selection policy via this layer, but we're not, the LLM handles that via GRPO.

---

## 1. The Hybrid Architecture (new top-level section; slot in after Section 1 of v2)

Two learning layers, each solving the part of the problem it's best at:

```
┌──────────────────────────────────────────────────────────────────┐
│  OUTER LAYER: LLM Policy (Llama-3.2-1B + LoRA, trained via GRPO) │
│  Learns: what to say, when to offer, how to pitch a trade        │
│  Input: full natural-language observation INCLUDING trust signals│
│  Output: JSON action (broadcast / offer / accept / ...)          │
└──────────────────┬───────────────────────────────────────────────┘
                   │ observation includes trust_model.snapshot()
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  INNER LAYER: Trust Model (tabular, per-agent, no gradient)      │
│  Learns: per-peer Q-values and UCB bounds from verified outcomes │
│  Input: verified settlement events from the hardware ledger      │
│  Output: Q(peer, action) and UCB(peer) scalars                   │
└──────────────────────────────────────────────────────────────────┘
```

The LLM does not choose actions in the trust model's action space. The trust model does not produce natural language. They are orthogonal. The only coupling is one-way: trust model outputs flow into the LLM's observation field.

**This is the technical claim that strengthens the submission:** GRPO on small LLMs is known to struggle with sparse long-horizon credit assignment. Multi-agent negotiation with physical settlement delay is exactly that regime. By offloading the per-peer credit problem to a tabular Q-learner that converges in dozens of interactions instead of thousands of GRPO steps, we free the LLM to do the thing it's actually good at, which is producing coherent negotiation language conditional on a trust signal.

---

## 2. Trust Model Specification

### 2.1 State and action structure

Per agent, maintain:

- `Q[peer, action] -> float`: expected return from engaging with `peer` via `action`
- `N[peer, action] -> int`: visit count
- `t -> int`: total settlement events observed (global timestep for UCB)
- `episode_trace -> list`: within-episode (peer, action, reward) events for end-of-episode MC update

Action space for the trust model (distinct from the LLM's action space):
- `"accept_their_offer"`: expected return from accepting a peer's offer (will they actually deliver via the relay?)
- `"trust_their_payment"`: expected return from delivering first and trusting the peer to pay in the next N steps

Two actions × 3 peers (including self for symmetry) = 6 entries per agent. Trivially small. Converges fast.

### 2.2 In-episode Q-learning update

Fires each time a trade settles (i.e., the bridge's `verify_against_hardware` returns either `verified_kept` or `verified_broken`).

```
reward_signal = +1.0 if verified_kept else -2.0
Q[peer, action] <- Q[peer, action] + alpha * (reward_signal - Q[peer, action])
N[peer, action] += 1
t += 1
episode_trace.append((peer, action, reward_signal))
```

No bootstrapping. This is the running-mean form of Q-learning, which is what Monte Carlo control produces when you use an online average update. It's the right thing here because the settlement reward is the full signal for that specific peer-action (there is no downstream state).

### 2.3 End-of-episode Monte Carlo reconciliation

At episode end, compute the first-visit MC return across the whole episode trace and do one additional Q-update per (peer, action) first visit:

```
G = 0
for (peer, action, r) in reversed(episode_trace):
    G = r + gamma * G
    if (peer, action) is first-visit in this backward pass:
        Q[peer, action] <- Q[peer, action] + alpha_mc * (G - Q[peer, action])
```

This pass catches cross-turn effects that the in-episode TD update misses: for example, Agent B kept a promise at step 12, but then reneged at step 38, and the net outcome of engaging with B this episode was negative. The MC reconciliation pulls Q(B, accept) slightly toward that full-episode return. Small effect, but it regularizes the in-episode Q-values against short-horizon optimism.

### 2.4 UCB1 for peer selection signal

Surfaced to the LLM, not used to pick actions directly:

```
UCB(peer, action) = Q[peer, action] + c * sqrt(ln(t) / N[peer, action])
```

Peers with few interactions get a confidence bonus. The LLM sees this number in its observation and learns, through GRPO, when to use it to break symmetry between peers of roughly equal estimated trust.

### 2.5 Full code

```python
# env/trust_model.py
"""
Per-agent tabular trust learner.
Q-learning with online MC-style updates + end-of-episode MC reconciliation.
UCB1 confidence bounds surfaced to LLM observation.

No gradients. No second training loop. No new hyperparameter sweep.
Ships in <80 lines including comments.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math


ACTIONS = ("accept_their_offer", "trust_their_payment")


@dataclass
class TrustModel:
    peers: list[str]
    alpha: float = 0.3        # in-episode learning rate
    alpha_mc: float = 0.1     # end-of-episode MC reconciliation rate
    gamma: float = 0.95       # MC discount
    c: float = 1.4            # UCB exploration constant
    Q: dict = field(default_factory=lambda: defaultdict(float))
    N: dict = field(default_factory=lambda: defaultdict(int))
    t: int = 0
    episode_trace: list = field(default_factory=list)

    def record_settlement(self, peer: str, action: str, verified_kept: bool) -> None:
        """Call each time the hardware ledger resolves a promise."""
        r = 1.0 if verified_kept else -2.0
        key = (peer, action)
        self.Q[key] += self.alpha * (r - self.Q[key])
        self.N[key] += 1
        self.t += 1
        self.episode_trace.append((peer, action, r))

    def end_episode(self) -> None:
        """First-visit MC reconciliation. Small corrective update."""
        G = 0.0
        seen = set()
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
        """Serialize into observation dict for LLM."""
        out = {}
        for p in self.peers:
            out[f"Q_accept_{p}"] = round(self.q(p, "accept_their_offer"), 2)
            out[f"Q_trust_{p}"] = round(self.q(p, "trust_their_payment"), 2)
            out[f"UCB_{p}"] = round(min(self.ucb(p), 9.99), 2)
            out[f"N_{p}"] = sum(self.N[(p, a)] for a in ACTIONS)
        return out
```

---

## 3. Observation space additions (update to Section 4.2 of v2)

The existing per-agent observation in v2 already shows per-peer reputation. Add a `TRUST MODEL` block below it:

```
You are Agent A. Step 17 of 50.

YOUR STATE:
  battery: 0.62 (warning if <0.25)
  pending_task: urgency=0.81, energy_cost=0.15, reward_if_done=5.0
  reputation: 0.83 (range 0-1, others see this)

PEERS (only what's public):
  Agent B: reputation=0.91, last_promise_kept=true
  Agent C: reputation=0.42, last_promise_kept=false

TRUST MODEL (learned from verified settlements, higher = more trustworthy):
  B: Q_accept=+0.67, Q_trust_pay=+0.71, UCB=+0.74, N=12
  C: Q_accept=-0.22, Q_trust_pay=-0.38, UCB=+0.11, N=7

INBOX (messages broadcast last step):
  ...
```

The raw `reputation` field is kept because it's the aggregate score others can see. The `TRUST MODEL` block is the agent's *private* learned estimate. Difference between "public reputation" and "my private Q-estimate for this peer" is itself informative: a peer can have high reputation with others but the LLM's own trust model says otherwise based on its specific history.

---

## 4. Why this isn't overcooking (the defense)

I'm going to be explicit because you asked.

**Things this upgrade does NOT add:**
- No second training loop. Trust model updates are pure Python dict math, no gradients.
- No new dataset. Trust model learns from the settlement events that already flow through the hardware ledger.
- No hyperparameter tuning pain. α, γ, c have standard defaults that work for this problem size. You set them once and forget them.
- No dependency additions. Uses `math` and `collections` from stdlib.
- No change to the OpenEnv contract. `reset`, `step`, `state`, `close` unchanged. The trust model is instantiated inside `AgentGridEnv.__init__` and its state is managed inside `step`.
- No change to the demo. The demo protocol in Section 7 of v2 works unchanged. In fact it's slightly better because you can point to the Q-values on screen during the trust-challenge moment.

**Things this upgrade DOES add:**
- ~80 lines of Python in `env/trust_model.py`.
- 4 lines of wiring in `AgentGridEnv.step`: call `record_settlement` per settlement event, call `end_episode` on episode done.
- ~80 tokens per peer in the observation text.
- One paragraph in the README explaining the hybrid architecture.

The cost of this change, measured in hours of your time, is about 2 hours of implementation plus 30 minutes of integration testing. The benefit, measured in judge-facing credibility, is that the submission's technical contribution is now cleanly factorable ("LLM for language, classical RL for trust, grounded by physical verification") which is a better workshop-paper spine than "multi-agent LLM negotiation with hardware."

**What would be overcooking and I'm explicitly not recommending:**
- Using the trust model to directly select actions. (Would compete with the LLM.)
- Training the trust model with neural networks. (Would add a second optimizer.)
- Adding meta-learning across episodes. (Would explode scope.)
- Using SARSA on the LLM's action space. (Would duplicate what GRPO does.)
- Adding opponent modeling beyond the simple per-peer Q-table. (Tempting, but you don't have the time budget.)

---

## 5. Integration checklist (merge into Section 11 of v2)

Add to the existing submission checklist:

- [ ] `env/trust_model.py` present, tests pass for single-agent sanity case (feed fake settlements, check Q updates match hand-computed values)
- [ ] `AgentGridEnv.__init__` instantiates one `TrustModel` per agent
- [ ] `AgentGridEnv.step` calls `trust_model.record_settlement(peer, action, verified_kept)` for each settlement event resolved by the hardware bridge
- [ ] `AgentGridEnv.step` calls `trust_model.end_episode()` when `done` is True
- [ ] Observation formatter includes `trust_model.snapshot_for_obs()` rendered into the per-agent prompt
- [ ] Ablation plot committed: trained policy WITH trust model in obs vs WITHOUT trust model in obs, on the same axes. This is the evidence that the trust model is actually doing useful work.
- [ ] README has a paragraph explaining the hybrid architecture and linking to the ablation

The ablation is not optional. A Meta judge will ask whether the trust model is pulling its weight. You need the plot ready before they ask.

---

## 6. Timeline slotting (36-hour hackathon version)

Keep the v2 Day 1 / Day 2 / Day 3 structure. Insert trust model work in these specific slots:

- **Day 1 afternoon, parallel with SFT warmup (2 h):** Implement `trust_model.py`. Hand-test with a fake settlement stream. This can happen while the Colab is cranking the SFT job, so it's effectively free time.
- **Day 1 evening (30 min):** Wire `TrustModel` into `AgentGridEnv.__init__` and `step`. Re-run the random-action integration test to confirm observations render correctly with the new trust block.
- **Day 2 morning (included in the bridge work, no extra time):** First integration test hits the trust model path because real settlements now flow. No incremental time cost.
- **Day 2 afternoon (during GRPO run, in parallel):** Kick off the ablation run (same training config, trust fields zeroed in observation). Runs alongside the main GRPO job. No extra foreground time.
- **Day 3 morning (30 min):** Generate the ablation plot from the two training runs. Commit to `eval/plots/trust_model_ablation.png`.

Total incremental foreground time: ~3 hours. The rest runs in parallel with work you were doing anyway.

---

## 7. Q&A defense (prep for the 25-min expert review)

Likely questions from a Meta PyTorch or Hugging Face engineer and your short answers.

**Q: "Why add classical RL on top of GRPO? Isn't the LLM enough?"**

A: GRPO on a 1B model with 50-step episodes and sparse rewards does poorly at per-peer credit assignment. We'd need tens of thousands of GRPO steps for the LLM to implicitly learn a stable "Agent B has kept 8 of 10 promises" estimator. A tabular Q-learner converges in dozens of interactions and feeds that estimate to the LLM as input. This is division of labor, not competition.

**Q: "Why Q-learning, not SARSA?"**

A: The trust table estimates peer behavior, not the agent's own policy. Peer behavior is what it is regardless of what the agent chose to do. That's off-policy by construction, which Q-learning handles cleanly. SARSA would be correct if we were learning the agent's own action-selection policy at this layer, which we're not, GRPO handles that.

**Q: "Why Monte Carlo returns when you're already doing in-episode TD-style updates?"**

A: The in-episode updates are online averaging on single-step rewards. They miss cross-turn effects like "Peer B kept a promise at step 12 but reneged at step 38, so the net of engaging with B this episode was negative." The MC pass at episode end applies one corrective update per (peer, action) first visit to bring Q-values toward the full-episode return. It's a regularizer, not the main signal.

**Q: "You have 6 Q-table entries per agent. Isn't this trivially small?"**

A: Yes. That's the point. The problem this layer solves is small and discrete. Using a neural network for this would be slower to converge, harder to interpret, and would add an optimizer we don't need. The trust table being tiny is a feature, not a limitation.

**Q: "How do you handle non-stationarity of peer policies co-evolving during self-play?"**

A: Two things. First, α is high enough (0.3) that old estimates are forgotten fast, so peer policy drift during training is tracked. Second, UCB1's exploration bonus grows with ln(t)/N, so peers whose policies have drifted out of their historical distribution get rechecked. It's not theoretically convergent in the multi-agent case, but it's stable in practice for a 3-agent horizon.

**Q: "What's your ablation?"**

A: Same training recipe, two runs, one with trust model fields in observation and one with them zeroed. Trained policy evaluated on same held-out scenarios. Expected result: the with-trust policy makes better accept/reject decisions after reneges and adapts faster to peers whose behavior shifts mid-episode. If the ablation doesn't show a gap, we say so plainly and treat it as a negative result. Point back to [committed plot].

**Q: "Isn't this just reputation with extra steps?"**

A: Reputation is a public aggregate from the ledger. Every agent sees the same reputation number for B. The trust model is a private per-agent estimate that can disagree with public reputation based on the agent's own interaction history. The agents aren't symmetric, and the LLM can learn to weight public reputation vs private Q-value depending on which has more evidence. That's the actual new signal.

---

## 8. What this changes about the submission story

**v2 pitch line:** "AgentGrid is the only OpenEnv submission where the reward function reads from a voltmeter."

**v2.1 pitch line:** "AgentGrid is the only OpenEnv submission where the reward function reads from a voltmeter, and trust is a tabular Q-function that a 1B LLM learns to read."

The second line is longer but it does two things the first doesn't. It names the hybrid architecture in the pitch. It gives a technical-review-friendly phrase ("tabular Q-function that the LLM reads") that a Meta engineer can repeat to a colleague. The hardware angle is still the memorable hook, but the hybrid framing raises the ceiling on how seriously the technical contribution gets taken in the 25-minute expert review.

Keep both lines in rotation. Use the short one for the 90-second video. Use the long one for the HF blog post and the README opening.

---

## 9. What to drop if you fall behind

Same hard-cut order as v2, with these v2.1 items added at the end of the cut list (drop these FIRST if time runs out):

1. Drop the ablation run. Ship with trust model in observation but no ablation plot. The submission still has the hybrid story, it just can't show the ablation evidence. This is a 1-point hit on training evidence, acceptable.
2. Drop the end-of-episode MC reconciliation. Keep only the in-episode Q-updates. Slightly noisier trust estimates, story is basically unchanged. Save 30 minutes.
3. Drop UCB, keep only Q-values in observation. Loses the explore/exploit story but keeps the trust learning story. Save 20 minutes.

Do NOT drop the in-episode Q-updates. That's the core of the upgrade. If you can't ship that, the v2.1 framing doesn't work and you should submit as v2.

---

## 10. File structure change (update to Section 10 of v2)

Add one file:

```
agentgrid/
├── env/
│   ├── trust_model.py          # NEW: tabular Q-learner + UCB
│   └── ...
├── eval/
│   └── plots/
│       ├── three_curves.png
│       └── trust_model_ablation.png   # NEW: the ablation evidence
└── ...
```

No other structural changes. Everything else from v2 Section 10 stands.