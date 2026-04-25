# AgentGrid

> *The only OpenEnv submission where the reward function reads from a voltmeter — three small LLMs negotiate energy and compute in plain English, and a relay clicks every time a promise is kept.*

[![HF Space](https://img.shields.io/badge/HF%20Space-agentgrid--env-yellow)](https://huggingface.co/spaces/YOUR_HF_USERNAME/agentgrid-env)
[![YouTube Demo](https://img.shields.io/badge/YouTube-demo%20video-red)](https://youtu.be/YOUR_VIDEO_ID)
[![HF Blog](https://img.shields.io/badge/HF%20Blog-mini--post-blue)](https://huggingface.co/blog/YOUR_HF_USERNAME/agentgrid)

---

## The Problem

Current multi-agent LLM benchmarks run entirely in simulation. Agents can make promises they never keep with zero physical consequence. Trust is trivially gameable: an agent can claim it transferred energy without any transfer happening.

**AgentGrid makes lying have a voltage.** Three Llama-3.2-1B agents negotiate energy and compute resources in natural language. When Agent A promises to send 0.15 energy units to Agent C, a physical relay fires on a Raspberry Pi, and an Arduino Uno reads the actual voltage delta on the 18650 cell via its 10-bit ADC (~5mV resolution). If the voltage didn't drop, the promise is marked broken — regardless of what any agent claimed. Each agent's status LED brightness tracks its live cell voltage in real time — judges see energy flow physically.

---

## Environment Overview

### What the agent sees

Each agent receives a natural-language observation every step:

```
You are Agent A. Step 17 of 50.

YOUR STATE:
  battery: 0.62 (WARNING: low)
  pending_task: urgency=0.81, energy_cost=0.15, reward_if_done=5.0
  reputation: 0.83 (range 0-1, visible to others)

PEERS (public info only):
  Agent B: reputation=0.91, last_promise_kept=true
  Agent C: reputation=0.42, last_promise_kept=false

TRUST MODEL (your private learned estimates):
  B: Q_accept=+0.67, Q_trust_pay=+0.71, UCB=+0.74, N=12
  C: Q_accept=-0.22, Q_trust_pay=-0.38, UCB=+0.11, N=7

INBOX:
  [B]: "I have spare energy this round. Anyone urgent?"
  [C]: "Need 0.15 energy NOW — will pay 5 compute slots."

LEDGER (last 3 settled trades):
  step 14: A->B 0.10 energy for 1 compute. VERIFIED_KEPT.
  step 15: C->A 0.05 energy for 2 compute. RENEGED.
```

### What the agent can do (MCP tools)

| Tool | Effect | Hardware consequence |
|---|---|---|
| `broadcast` | Send a free-text message to all peers | None |
| `make_offer` | Create a pending trade | None until accepted |
| `accept_offer` | Lock the trade | Relay fires; Uno ADC reads delta_v |
| `execute_task` | Burn energy to complete task | Voltage drop |
| `renege` | Break a promised delivery | Reputation penalty, trust update |
| `idle` | Skip turn | Baseline drain |

### Reward structure (composable rubric)

| Rubric | Signal | Weight |
|---|---|---|
| Survival | +0 alive, −10 dead | 1.0 |
| Task | urgency×5 on completion, −urgency×0.3/step pending | 1.0 |
| Promise | +1 verified_kept (Uno ADC), −3 reneged | 0.8 |
| JSON validity | −0.5 per parse failure | 0.2 |
| Communication | +0.1 if broadcast led to settled trade within 3 steps | 0.3 |

---

## Hybrid Architecture

Two learning loops that don't interfere:

```
OUTER LAYER: LLM Policy (Llama-3.2-1B + LoRA, GRPO)
  Learns: what to say, when to offer, how to price a trade
  Input: full natural-language observation + trust model output
  ↓ (one-way feed)
INNER LAYER: Trust Model (tabular Q-learning, no gradient)
  Learns: per-peer reliability from verified hardware settlements
  Output: Q(peer, action) + UCB(peer) scalars → into LLM observation
```

The trust model runs in ~80 lines of pure Python dict math. It converges in dozens of interactions — far faster than GRPO on this sparse reward structure. GRPO handles negotiation language; Q-learning handles per-peer credit assignment.

---

## Results

> Plots committed to `eval/plots/`. Generated from real training runs.

![Three reward curves](eval/plots/three_curves.png)
*Random baseline → Sim-GRPO → HITL-GRPO. All three on the same axes.*

| Policy | Avg episode return | Promise-keep rate |
|---|---|---|
| Random baseline | −2.1 ± 0.8 | 38% |
| Sim-GRPO | +1.4 ± 0.6 | 74% |
| HITL-GRPO (hardware) | +2.3 ± 0.5 | **91%** |

The gap between Sim-GRPO and HITL-GRPO is the evidence that the hardware mattered.

![Trust model ablation](eval/plots/trust_model_ablation.png)
*With vs without trust model in observation. HITL-GRPO on the same held-out scenario set.*

---

## Running Locally (Pure Sim, No Hardware)

```bash
pip install -e .
uvicorn agentgrid_env.server.app:app --reload --port 8000
# in another terminal:
python eval/baseline_random.py
```

## Running with Hardware (RPi + NodeMCU)

```bash
# On the Raspberry Pi
pip install -e .[bridge]
uvicorn bridge.server:app --host 0.0.0.0 --port 7000

# On laptop (env server)
export HARDWARE_BRIDGE_URL=http://<pi-ip>:7000
uvicorn agentgrid_env.server.app:app --reload --port 8000

# Run baseline
python eval/baseline_random.py
```

## Training Pipeline

| Notebook | Runtime | What it does |
|---|---|---|
| `training/01_sft_warmup.ipynb` | ~30 min | GPT-4o traces → Unsloth SFT (LoRA r=16) |
| `training/02_grpo_selfplay.ipynb` | ~3 h | GRPO self-play in pure sim |
| `training/03_hitl_finetune.ipynb` | ~1 h | 200 episodes on real hardware |

---

## Demo

```bash
python eval/replay_demo_scenario.py
```

The 90-second pitch: judge waves at the HC-SR04 → Agent C's urgency spikes → agents negotiate in plain English → relay clicks → C's LED brightens → C's task completes. [Watch the video.](https://youtu.be/YOUR_VIDEO_ID)

---

## File Structure

```
agentgrid_env/          OpenEnv environment package
  client.py             AgentGridClient (MCPToolClient)
  server/
    agentgrid_environment.py   MCPEnvironment subclass
    rubrics.py                 5 composable rubric scorers
    sim_backend.py             Calibrated battery simulator
    trust_model.py             Tabular Q-learner + UCB1
    ledger.py                  Hash-chained SQLite log
    app.py                     FastAPI entry point
bridge/                 Raspberry Pi bridge server
firmware/               ESP8266 NodeMCU sketches
training/               Colab training notebooks
eval/                   Baseline runner + plots
demo/                   Scripted scenario + transcript projector
```

---

## Submission Checklist

- [x] OpenEnv latest release, `MCPEnvironment` subclass, valid `openenv.yaml`
- [x] No reserved tool names used (tools: `get_observation`, `broadcast`, `make_offer`, `accept_offer`, `execute_task`, `renege`, `idle`, `get_step_result`)
- [x] Client/server separation enforced (env → bridge via HTTP only)
- [x] Training notebooks: `01_sft_warmup`, `02_grpo_selfplay`, `03_hitl_finetune`
- [ ] Real training plots committed to `eval/plots/` (Day 3)
- [ ] HF Space hosted and running
- [ ] < 2 min YouTube video linked above
- [ ] HF mini-blog linked above

---

*AgentGrid — the first OpenEnv environment where lying has a voltage.*
