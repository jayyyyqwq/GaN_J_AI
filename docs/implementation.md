# AgentGrid — Implementation Guide

> Full developer guide: clone → understand → build each layer → test → submit.
> Written for someone new to the repo. Read this before opening any source file.

---

## 0. Prerequisites

### Hardware (for full run)
| Component | Qty | Purpose |
|---|---|---|
| Raspberry Pi 3 | 1 | Bridge host + ledger |
| NodeMCU ESP8266 | 3 | One per agent (A/B/C) |
| INA219 voltage sensor | 3 | Per-cell voltage truth oracle |
| 4-channel relay module | 1 | Routes power between cells |
| 18650 Li-ion cell | 3 | One per agent |
| TP4056 charging module | 3 | Per-cell charging between episodes |
| HC-SR04 ultrasonic | 1 | Urgency injection via judge gesture |
| White LEDs + 220Ω resistors | 3 | Agent status indicators |

**No hardware?** Pure-sim mode works end-to-end. Skip Sections 4, 5, and any step marked `[HARDWARE]`. The env runs identically with a calibrated battery simulator.

### Software
```bash
Python 3.11+
Arduino IDE 2.x  (for firmware, Section 5)
pip install -e .          # pure sim
pip install -e .[bridge]  # + RPi smbus2/GPIO
```

### Accounts
- Google Colab (or HF Spaces with GPU) — for training notebooks (Section 6)
- Hugging Face account — for hosting the env (submission requirement)

---

## 1. Project Overview

AgentGrid is an OpenEnv-compliant multi-agent negotiation environment. Three Llama-3.2-1B agents negotiate energy and compute resources in natural language. On real hardware, a relay matrix physically routes power between 18650 cells, and INA219 sensors read actual voltage — making it impossible for agents to fake promise fulfillment.

**The core claim:** reward is partially derived from a voltmeter, not just a simulator.

### Two-layer architecture

```
TRAINING LAYER (Colab / HF Spaces)
  LLM Agent A/B/C  ──JSON actions──►  AgentGridEnv (MCPEnvironment)
                                             │ HTTP
                                             ▼
BRIDGE LAYER (Raspberry Pi)
  FastAPI server → INA219 reads → relay fire → voltage delta → truth signal
```

The env never imports bridge internals. The bridge never knows about the LLM. HTTP is the only coupling — enforcing the OpenEnv client/server separation requirement.

### Hybrid learning (v2.1)

Two learning loops, no overlap:
- **GRPO (outer):** LLM learns negotiation language and action selection
- **Q-learning (inner):** Tabular trust model learns per-peer reliability from verified settlements

Trust model outputs flow into the LLM's observation as additional fields. The LLM does not update the Q-table; the Q-table does not produce actions.

---

## 2. Codebase Map

| Path | Purpose | Status |
|---|---|---|
| `agentgrid_env/server/agentgrid_environment.py` | Main MCPEnvironment subclass, 8 MCP tools, step resolution | **~95% done** |
| `agentgrid_env/server/rubrics.py` | 5 composable rubric scorers → per-agent scalar reward | **Done** |
| `agentgrid_env/server/sim_backend.py` | Calibrated battery simulator (used when bridge is absent) | **Done** |
| `agentgrid_env/server/trust_model.py` | Tabular Q-learner + MC returns + UCB1 per agent | **Done** |
| `agentgrid_env/server/ledger.py` | Hash-chained SQLite append-only commitment log | **Done** |
| `agentgrid_env/server/app.py` | FastAPI entry point, reads env vars | **Done** |
| `agentgrid_env/client.py` | `AgentGridClient` (thin MCPToolClient subclass) | **Done** |
| `bridge/server.py` | FastAPI on RPi: relay, voltage, sensor endpoints | **Done** |
| `bridge/hardware.py` | INA219/relay/ultrasonic drivers (graceful no-op off-Pi) | **~90% done** — relay timing needs Day 2 calibration |
| `bridge/calibration.json` | Hardware parameters (volts/unit, noise, GPIO pins) | **Done** — update after Day 2 INA219 logs |
| `firmware/nodemcu_agent.ino` | ESP8266 sketch: INA219 read + LED + heartbeat | **MISSING** (~150 lines) |
| `training/01_sft_warmup.ipynb` | Colab: synthetic traces → Unsloth SFT (LoRA r=16) | **MISSING** |
| `training/02_grpo_selfplay.ipynb` | Colab: GRPO self-play in pure sim | **MISSING** |
| `training/03_hitl_finetune.ipynb` | Colab: 200 episodes on real hardware | **MISSING** |
| `eval/baseline_random.py` | Random policy baseline runner (runnable now) | **Done** |
| `eval/replay_demo_scenario.py` | Scripted scenario replayer (runnable now) | **Done** |
| `demo/scripted_scenario.json` | 5-step demo script for 90-second pitch | **Done** |
| `README.md` | Story, results, links to HF Space + video + blog | **MISSING** |
| `eval/plots/` | `three_curves.png`, `trust_model_ablation.png` | **MISSING** (Day 3 output) |

---

## 3. Understanding the Environment Core

Read these files in this order before writing any code. Each one depends on the previous.

### 3.1 Read first: `rubrics.py`

Five composable scorers. Understand the reward structure before anything else.

| Rubric | Signal | Weight |
|---|---|---|
| `SurvivalRubric` | `+0` alive, `-10` battery dead | 1.0 |
| `TaskRubric` | urgency × reward on completion; −urgency×0.3/step pending | 1.0 |
| `PromiseRubric` | `+1` kept (INA219-verified), `-3` reneged | 0.8 |
| `JsonValidityRubric` | `-0.5` per parse failure (curriculum pressure) | 0.2 |
| `CommunicationRubric` | `+0.1` if broadcast led to settled trade within 3 steps | 0.3 |

The `PromiseRubric` is the key: reward depends on whether the INA219 voltage drop actually matches the promised transfer, not just on what the ledger says.

### 3.2 Read second: `sim_backend.py`

Calibrated battery simulator. Provides the same interface as the hardware bridge when `hardware_url` is `None`. Methods: `drain()`, `transfer()`, `idle_drain()`, `reset()`, `is_alive()`, `get_urgency_from_sensor()`.

### 3.3 Read third: `trust_model.py`

Per-agent tabular Q-learner. State: `Q[peer, action]`, `N[peer, action]`. Two trust actions: `accept_their_offer` and `trust_their_payment`.

Key calls (already wired into `agentgrid_environment.py`):
- `record_settlement(peer, action, verified_kept)` — fires on each resolved trade
- `end_episode()` — MC reconciliation, call on episode reset
- `snapshot_for_obs()` — returns dict of Q-values and UCB bounds for the LLM observation

### 3.4 Read fourth: `ledger.py`

Hash-chained SQLite log. `append()` stores each trade with SHA-256 prev/this hash. `verify_sim(entry_id, delta_v)` checks whether the actual voltage drop matches the promised amount within tolerance. `kept_ratio(agent_id)` drives the public reputation field in observations.

### 3.5 The main class: `agentgrid_environment.py`

`AgentGridEnvironment` subclasses `MCPEnvironment`. The 8 MCP tools exposed to LLM agents are defined as closures inside `__init__` using FastMCP's `@mcp.tool` decorator.

**Action sequencing:** each agent calls exactly one action tool per step. After all 3 agents submit, `_maybe_resolve_step()` detects this and fires `_resolve_step()` automatically.

**Key internal methods (all implemented):**

| Method | Location | What it does |
|---|---|---|
| `_pending_count()` | line 374 | `len(AGENTS) - len(_pending_actions)` |
| `_maybe_resolve_step()` | line 377 | Fires `_resolve_step()` when all 3 have acted |
| `_resolve_step()` | line 381 | Scores rubrics, advances step, resets buffers |
| `_execute_energy_transfer()` | line 423 | Routes to bridge HTTP or sim, returns `delta_v` |
| `_format_observation()` | line 439 | Builds full natural-language obs string per agent |

The observation format (line 445–516) renders: agent state, peer reputation, trust model Q-values, message inbox, pending offers to you, and last 3 ledger entries. Each field is a direct string interpolation from live state — no external template.

---

## 4. Layer 1 — Completing the Environment (Day 1 Morning)

The environment core is ~95% done. There is no structural work remaining — just integration testing.

### Step 1: Run the smoke test

```bash
pip install -e .
python eval/baseline_random.py
```

This runs a random-policy episode against the live env. If it completes without exceptions, the environment is functional.

### Step 2: Verify observation format

Start the env server and call `get_observation` manually:

```bash
uvicorn agentgrid_env.server.app:app --reload --port 8000
```

In another terminal:
```bash
python - <<'EOF'
from agentgrid_env.client import AgentGridClient
client = AgentGridClient("http://localhost:8000")
client.reset()
print(client.call_tool("get_observation", {"agent_id": "A"}))
EOF
```

Expected output: multi-line observation with YOUR STATE, PEERS, TRUST MODEL, INBOX, PENDING OFFERS, LEDGER, and action menu.

### Step 3: Verify rubric scores are non-zero

After the smoke test, check that at least one of Survival/Task/Promise rubrics fired a non-zero reward. A clean run where all rewards are 0.0 every step indicates a scoring bug.

### Step 4: Verify ledger hash chain

```python
from agentgrid_env.server.ledger import CommitmentLedger
ledger = CommitmentLedger(":memory:")
id1 = ledger.append(step=1, offerer="A", accepter="B", give_type="energy",
                    give_amount=0.1, want_type="compute", want_amount=1.0)
id2 = ledger.append(step=2, offerer="B", accepter="C", give_type="compute",
                    give_amount=1.0, want_type="energy", want_amount=0.05)
entries = ledger.recent(2)
assert entries[0]["prev_hash"] != entries[1]["prev_hash"]
print("Hash chain OK")
```

---

## 5. Layer 2 — Hardware Bridge (`bridge/`) [HARDWARE]

### 5.1 Wiring (Day 1 Evening)

Do this before any software integration. Get the hardware working at the GPIO level first.

INA219 I2C addresses (set via A0/A1 solder bridges on the board):
- Agent A: `0x40` (default, both open)
- Agent B: `0x41` (A0 closed)
- Agent C: `0x44` (A1 closed)

Relay GPIO (BCM numbering, already set in `bridge/hardware.py`):
- A↔B: GPIO 17
- A↔C: GPIO 27
- B↔C: GPIO 22

HC-SR04: TRIG=GPIO 23, ECHO=GPIO 24

Test relays manually before running any server:
```python
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT, initial=GPIO.HIGH)
GPIO.output(17, GPIO.LOW)   # click — relay closed
import time; time.sleep(0.5)
GPIO.output(17, GPIO.HIGH)  # click — relay open
GPIO.cleanup()
```

### 5.2 Run the bridge server

On the Raspberry Pi:
```bash
pip install -e .[bridge]
uvicorn bridge.server:app --host 0.0.0.0 --port 7000
```

Test it from the laptop:
```bash
curl http://<pi-ip>:7000/health
curl http://<pi-ip>:7000/voltage/A
curl -X POST http://<pi-ip>:7000/relay/fire \
     -H "Content-Type: application/json" \
     -d '{"from": "A", "to": "B", "amount": 0.1}'
```

### 5.3 Calibrate relay timing (Day 2)

The current value in `bridge/hardware.py` line 91 is a placeholder:
```python
duration = amount * 2.5  # seconds per energy unit (calibrate Day 2)
```

Calibration procedure:
1. Run 10 relay fires at `amount=0.1` with a 250ms duration
2. Read `delta_v` from INA219 after each fire
3. Compute `volts_per_energy_unit = mean(delta_v) / 0.1`
4. Update `bridge/calibration.json` → `"volts_per_energy_unit"` field
5. Update the relay duration formula to `duration = amount / volts_per_energy_unit * 0.08`

### 5.4 Connect env to bridge

Set the env var before starting the env server:
```bash
export HARDWARE_BRIDGE_URL=http://<pi-ip>:7000
uvicorn agentgrid_env.server.app:app --reload
```

Run `eval/baseline_random.py` again. This time, relay clicks should be audible when trades execute.

### 5.5 Create `bridge/ledger_bridge.py` stub

`bridge/server.py` attempts to import `CommitmentLedger` from this file. A fallback exists but produces a warning. Create a minimal stub:

```python
# bridge/ledger_bridge.py
from agentgrid_env.server.ledger import CommitmentLedger

__all__ = ["CommitmentLedger"]
```

---

## 6. Layer 3 — Firmware (`firmware/`) [HARDWARE]

### 6.1 Create `firmware/nodemcu_agent.ino`

Target: ~150 lines. Required behaviour:
- Read INA219 voltage via I2C (Adafruit INA219 library)
- Drive status LED on D2 (brightness proportional to battery level)
- POST heartbeat to RPi `/health` every 5 seconds
- Receive LED commands from RPi via HTTP GET `/led/{agent_id}/{brightness}`

Libraries needed (install via Arduino IDE → Library Manager):
- `Adafruit INA219`
- `ESP8266WiFi` (built-in with ESP8266 board package)
- `ESP8266HTTPClient` (built-in)

Flash instructions are in `firmware/README.md`. Edit `AGENT_ID` per board before flashing.

Wiring per NodeMCU:
- INA219 SDA → D2, SCL → D1
- LED anode → D4, cathode → GND via 220Ω resistor

### 6.2 Test firmware independently

After flashing, open Arduino Serial Monitor at 115200 baud. Expected output:
```
AgentGrid Firmware — Agent A
WiFi connected: 192.168.x.x
INA219 init OK
Heartbeat sent: 200
Battery: 4.12V
```

If voltage reads 0.00V, check I2C address solder bridges and SDA/SCL wiring.

---

## 7. Layer 4 — Training Pipeline (`training/`)

Three Colab notebooks in sequence. Estimated runtimes on free-tier Colab T4.

### 7.1 `training/01_sft_warmup.ipynb` (~30 min)

**Goal:** teach the LLM to output valid JSON actions and basic negotiation grammar.

Structure:
1. Generate 2000 synthetic negotiation transcripts using an LLM API (GPT-4o or Claude). Prompt: "You are playing Agent A in a 3-agent energy negotiation. Generate a realistic 10-step transcript where agents negotiate using these JSON actions: [paste action schema from env spec]."
2. Save transcripts to `training/synthetic_traces/sft_data.jsonl`
3. SFT with Unsloth:
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       "meta-llama/Llama-3.2-1B", max_seq_length=2048, load_in_4bit=True
   )
   model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
       target_modules=["q_proj", "v_proj"])
   # SFT trainer via TRL SFTTrainer
   ```
4. Evaluate: JSON validity rate should be >90% on held-out traces.
5. Save checkpoint to HF Hub.

### 7.2 `training/02_grpo_selfplay.ipynb` (~3 hours)

**Goal:** improve on the composable rubric reward via on-environment RL.

Structure:
1. Load SFT checkpoint
2. Start `AgentGridEnv` in pure sim mode (no `HARDWARE_BRIDGE_URL`)
3. Run GRPO via TRL:
   ```python
   from trl import GRPOTrainer, GRPOConfig
   config = GRPOConfig(num_train_epochs=3, learning_rate=1e-5,
                       reward_model=None)  # env provides rewards directly
   ```
4. Three parallel model copies, one per agent
5. Track per-episode: total return, promise-keep rate, task completion rate
6. Save best checkpoint (by promise-keep rate, not just total return)

### 7.3 `training/03_hitl_finetune.ipynb` (~1 hour) [HARDWARE]

**Goal:** close the sim-to-real gap with 200 real hardware episodes.

Prerequisites: bridge server running on Pi, env server set to `HARDWARE_BRIDGE_URL`.

Structure:
1. Load GRPO checkpoint
2. Run 200 episodes against the live bridge
3. Capture INA219 readings into `training/synthetic_traces/hitl_curves.json`
4. Fine-tune for 1 epoch on the HITL replay buffer
5. Generate the three reward curves (random / sim-GRPO / HITL-GRPO) and save to `eval/plots/three_curves.png`

---

## 8. Integration Testing Sequence

Run these in order. Don't proceed to the next until the current one passes.

| # | Test | Command | Pass condition |
|---|---|---|---|
| 1 | Unit: rubric scorers | `pytest agentgrid_env/server/test_rubrics.py` | All pass |
| 2 | Unit: trust model | `pytest agentgrid_env/server/test_trust.py` | Q updates match hand-computed values |
| 3 | Unit: ledger hash chain | `pytest agentgrid_env/server/test_ledger.py` | Hash chain verifies |
| 4 | Smoke: random policy vs sim env | `python eval/baseline_random.py` | Completes N episodes, logs non-zero rewards |
| 5 | Demo replay: scripted scenario | `python eval/replay_demo_scenario.py` | 5-step sequence executes, relay fires logged |
| 6 | `[HARDWARE]` Bridge health | `curl http://<pi-ip>:7000/health` | `{"status": "ok"}` |
| 7 | `[HARDWARE]` E2E: env + bridge | `HARDWARE_BRIDGE_URL=http://<pi-ip>:7000 python eval/baseline_random.py` | Relay clicks audible, delta_v non-zero |
| 8 | `[HARDWARE]` E2E: trained agent | Run notebook `03_hitl_finetune.ipynb` first episode | Reward > random baseline |

---

## 9. Running Locally (Pure Sim, No Hardware)

```bash
# 1. Install
git clone <repo> && cd AgentGrid_V1
pip install -e .

# 2. Start env server
uvicorn agentgrid_env.server.app:app --reload --port 8000

# 3. Sanity check
python eval/baseline_random.py

# 4. Demo replay
python eval/replay_demo_scenario.py
```

Environment variables (all optional for sim mode):
```bash
HARDWARE_BRIDGE_URL=   # empty = pure sim
EPISODE_STEPS=50       # default
MAX_CONCURRENT_ENVS=4  # default
```

---

## 10. Running with Hardware (RPi + NodeMCU) [HARDWARE]

```bash
# On the Raspberry Pi
pip install -e .[bridge]
uvicorn bridge.server:app --host 0.0.0.0 --port 7000

# On the laptop (env server)
export HARDWARE_BRIDGE_URL=http://<pi-ip>:7000
uvicorn agentgrid_env.server.app:app --reload --port 8000

# From any machine on the same network
python eval/baseline_random.py
```

If the Pi is the hotspot (venue WiFi fallback), laptop joins the Pi's network:
```bash
# Pi: enable hotspot (Network Manager)
nmcli device wifi hotspot ifname wlan0 ssid AgentGrid password agentgrid2026

# Laptop: join it, then use 192.168.x.1 as Pi IP
export HARDWARE_BRIDGE_URL=http://192.168.4.1:7000
```

---

## 11. Submission Checklist

Mapped directly to the judging criteria from `docs/problem statements.md`.

### Non-negotiable minimums
- [ ] `openenv.yaml` present and valid (done)
- [ ] `MCPEnvironment` subclass, no reserved tool names used (done — tools are `get_observation`, `broadcast`, `make_offer`, `accept_offer`, `execute_task`, `renege`, `idle`, `get_step_result`)
- [ ] Client/server separation enforced (done — env → bridge via HTTP only)
- [ ] Training notebook runs in Colab using Unsloth or TRL (`01_sft_warmup.ipynb` + `02_grpo_selfplay.ipynb`)
- [ ] Real training run with committed loss/reward plots in `eval/plots/`
- [ ] HF Space hosting the env (push `agentgrid_env/` + `openenv.yaml`)
- [ ] README with: problem, env description, results, links
- [ ] < 2 min YouTube video linked from README
- [ ] HF mini-blog linked from README
- [ ] Video file NOT committed to repo (link only)

### Innovation evidence (40% of score)
- [ ] Three-way reward comparison plot (`eval/plots/three_curves.png`): random baseline / sim-GRPO / HITL-GRPO
- [ ] Trust model ablation plot (`eval/plots/trust_model_ablation.png`): with vs without trust fields in observation
- [ ] Demo video showing relay click + LED brightening on accepted trade

### Training evidence (20% of score)
- [ ] Promise-keep rate tracked separately from episode return
- [ ] Before/after behavior comparison: same scripted scenario with random agent vs trained agent
- [ ] Plots committed as `.png` with labeled axes

### Pipeline evidence (10% of score)
- [ ] `RubricScorer` wires all 5 rubrics with documented weights
- [ ] `verify_sim(entry_id, delta_v)` called for every settled energy trade
- [ ] `trust_model.end_episode()` called on every reset

### Hard cuts if behind schedule (in order)
1. Drop `03_hitl_finetune.ipynb` — ship sim-only. Demo still works, relay still fires.
2. Drop trust model ablation plot — ship with trust model but no ablation evidence.
3. Drop MC reconciliation from trust model — keep in-episode Q-updates only.
4. Drop the 4-channel relay — use one relay (A↔B only), one less trade direction.

---

## 12. Demo Preparation (Day 3)

### The 90-second pitch sequence

1. **Setup (0:00–0:25):** Pan camera across rig. Three NodeMCUs, three cells, relay module, ultrasonic sensor. Voiceover: *"Three small LLMs. One battery each. They negotiate in plain English. The hardware enforces every promise."*

2. **The hand wave (0:25–1:05):** Project live transcript on screen. Judge waves hand at HC-SR04 sensor. Agent C's urgency spikes. Show negotiation messages in real time. Relay clicks. C's LED brightens. C's task completes (LED flash). A's Q-value for C visible on screen.

3. **The curves (1:05–1:30):** Cut to three-curves plot. Voiceover: *"Promise-keep rate: 38% baseline → 91% post-training. The trained agent didn't just learn to survive — it learned that reputation is a currency."*

### The one-line pitch

> *AgentGrid is the only OpenEnv submission where the reward function reads from a voltmeter — and trust is a tabular Q-function that a 1B LLM learns to read.*

### Scripted scenario replay

`demo/scripted_scenario.json` has the 5-step sequence already written. Run it:
```bash
# env server must be running
python eval/replay_demo_scenario.py
```

Rehearse 3× on Day 3 before the actual demo. Know which LED corresponds to which agent.
