# Gradio HF Spaces Implementation Plan

## Goal

Ship a Gradio Space that demonstrates AgentGrid V1 entirely in sim mode — judges land on the URL and within 5 seconds see live agent negotiation, battery curves, and ledger entries. No hardware, no live LLM inference. Pre-recorded scripted episodes for instant cold-start; on-demand random baseline runs for "this is reproducible" credibility.

## Constraints & decisions

| Decision | Choice | Why |
|---|---|---|
| SDK | Gradio (CPU tier) | Default for HF Spaces, free tier OK, no GPU needed for replay |
| Env hosting | **In-process** (no FastAPI/uvicorn inside Space) | One container, cold-start <10s. The MCP/HTTP layer is unnecessary when caller and env live in the same Python process |
| LLM inference | None at runtime | Pre-record agent actions; SFT/GRPO stays in Colab notebooks |
| State scope | `gr.State` per Gradio session | Multiple users get isolated `HeadlessRunner` + `CommitmentLedger(:memory:)` |
| Ledger DB | `:memory:` SQLite per session | Free tier has no persistent volume guarantees; transient is fine for demo |
| Default mode on load | Pre-recorded scripted scenario, auto-play | Zero compute on cold-start, judge sees motion immediately |

## File map

### New files

```
c:\AgentGrid_V1\
├── app.py                                 # Gradio Spaces entrypoint
├── requirements.txt                       # HF Spaces deps (Spaces ignores pyproject)
├── spaces\
│   ├── __init__.py
│   ├── runner.py                          # HeadlessRunner — drives env in-process
│   ├── scripted_player.py                 # Replays demo\scripted_scenario.json
│   ├── random_player.py                   # Seeded random policy (port of eval\baseline_random.py loop)
│   ├── recorder.py                        # Captures snapshots → JSON
│   ├── plots.py                           # Plotly figure builders
│   ├── transcript.py                      # HTML/Markdown rendering (port of demo\transcript_projector.py)
│   └── prerecorded\
│       ├── scripted_demo.json             # Pre-baked replay of scripted scenario
│       ├── baseline_seed_42.json          # Pre-baked random episode
│       └── baseline_seed_123.json
```

### Modified files

```
README.md                                  # Add HF Spaces YAML header at top
docs\implementation.md                     # Add "Phase 6: HF Spaces demo" section (1 paragraph)
```

### Untouched (zero changes)

`agentgrid_env\server\*` — env, sim_backend, ledger, trust_model, rubrics. The whole point of `HeadlessRunner` is to be a thin wrapper, not a refactor.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  app.py (Gradio Blocks)                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Mode toggle │  │  Controls    │  │  Plot panels     │  │
│  │  Scripted /  │  │  Reset, Step,│  │  - SoC/agent     │  │
│  │  Random /    │  │  Play, Pause │  │  - Reputation    │  │
│  │  Manual      │  │  Seed slider │  │  - Promise-keep  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────▲─────────┘  │
│         │ callbacks        │                    │            │
│         ▼                  ▼                    │            │
│  ┌─────────────────────────────────────────────┴─────────┐  │
│  │  gr.State { runner, history, mode, step_idx }        │  │
│  └─────┬──────────────────────────────────────────────────┘  │
└────────┼────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────┐    ┌──────────────────────────┐
│  spaces\runner.py      │    │  spaces\scripted_player  │
│  HeadlessRunner        │◄───┤  spaces\random_player    │
│  - reset(seed)         │    │  spaces\prerecorded\*    │
│  - apply(agent, action)│    └──────────────────────────┘
│  - snapshot() → dict   │
└────────┬───────────────┘
         │ wraps
         ▼
┌────────────────────────────────────────────────────────────┐
│  agentgrid_env.server.AgentGridEnvironment(hardware_url=None)│
│  └─ SimBackend, CommitmentLedger, TrustModel, RubricScorer   │
└────────────────────────────────────────────────────────────┘
```

---

## Component contracts

### `spaces\runner.py` — `HeadlessRunner`

In-process driver. Bypasses MCP/HTTP, calls env tool closures directly.

```python
class HeadlessRunner:
    def __init__(self, episode_steps: int = 50) -> None: ...
    def reset(self, seed: int | None = None) -> Snapshot: ...
    def apply(self, agent_id: str, action: str, **kwargs) -> str:
        """action ∈ {broadcast, make_offer, accept_offer, execute_task, renege, idle}"""
    def snapshot(self) -> Snapshot:
        """Captured state after most recent step resolution."""
```

**Snapshot dataclass** (`@dataclass(frozen=True)`):

```python
@dataclass(frozen=True)
class Snapshot:
    game_step: int
    done: bool
    batteries: dict[str, float]              # {"A": 0.74, ...}
    reputation: dict[str, float]
    rewards: dict[str, float]                # cumulative episode reward per agent
    pending_offers: list[OfferView]
    recent_ledger: list[LedgerView]          # last 5 entries with status
    last_messages: list[MessageView]         # broadcasts from this step
    last_action: ActionView | None           # what just happened (for transcript)
```

**Implementation notes**:
- The env's MCP tools are closures inside `__init__`. Two ways to call them:
  - **Path A (preferred)**: bind references at construction — store the closures in a private dict at the end of `_register_tools`, expose via a method. *Requires one tiny addition to env or a runner-side hack to access via `mcp._tools`.*
  - **Path B (fallback)**: spin up the FastMCP server in a background thread on `127.0.0.1:8000` and use `AgentGridClient`. Adds ~1s startup, no env changes.
- **Verification step at the start of Phase 1**: spike both paths in 10 minutes, pick whichever is cleaner. Default to B if A is messy.

### `spaces\scripted_player.py`

Loads `demo\scripted_scenario.json`, walks steps, resolves `__LAST_OFFER__` placeholder by tracking the most recent offer_id returned from `make_offer`.

```python
def scripted_episode(runner: HeadlessRunner, scenario_path: Path) -> list[Snapshot]: ...
def scripted_step(runner: HeadlessRunner, step_def: dict, last_offer_id: str | None) -> tuple[Snapshot, str | None]: ...
```

### `spaces\random_player.py`

Direct port of [eval/baseline_random.py:22-66](../eval/baseline_random.py#L22-L66) but using `runner.apply` instead of `env.call_tool`. Single-episode entry: `random_episode(runner, seed) -> list[Snapshot]`.

### `spaces\plots.py`

Pure functions, no side effects. Each takes `list[Snapshot]` and returns a Plotly figure.

```python
def battery_curve(history: list[Snapshot]) -> go.Figure: ...      # 3 lines, x=step, y=SoC
def reputation_curve(history: list[Snapshot]) -> go.Figure: ...   # 3 lines, x=step, y=rep
def promise_keep_curve(history: list[Snapshot]) -> go.Figure: ... # 1 line, cumulative kept-ratio across all agents
def ledger_table(history: list[Snapshot]) -> pd.DataFrame: ...    # for gr.Dataframe
```

### `spaces\transcript.py`

Markdown renderer (Gradio's `gr.Markdown` supports HTML colors via spans). Port the agent-color logic from [demo/transcript_projector.py:22-30](../demo/transcript_projector.py#L22-L30):

```python
AGENT_COLORS = {"A": "#3B82F6", "B": "#10B981", "C": "#EF4444"}

def render_step_md(snapshots: list[Snapshot]) -> str:
    """Build a Markdown blob with colored agent prefixes, action verbs, and ledger callouts."""
```

### `app.py` — Gradio Blocks

```python
import gradio as gr
from spaces.runner import HeadlessRunner
from spaces.scripted_player import scripted_episode
from spaces.random_player import random_episode
from spaces.plots import battery_curve, reputation_curve, promise_keep_curve, ledger_table
from spaces.transcript import render_step_md

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AgentGrid V1 — Live Demo") as ui:
        gr.Markdown("# AgentGrid V1\n3 LLM agents negotiate energy. Battery physics calibrated; ledger hash-chained.")

        # Session state
        runner_state = gr.State()           # HeadlessRunner | None
        history_state = gr.State([])        # list[Snapshot]
        step_idx_state = gr.State(0)

        with gr.Row():
            mode = gr.Radio(["Scripted demo", "Random baseline (seeded)", "Manual step"], value="Scripted demo")
            seed_slider = gr.Slider(0, 999, value=42, step=1, label="Seed (random mode)")
            reset_btn = gr.Button("Reset")

        with gr.Row():
            with gr.Column(scale=1):
                transcript_md = gr.Markdown(label="Negotiation transcript")
                ledger_df = gr.Dataframe(label="Ledger (last 5)", headers=["step", "from→to", "give", "want", "status", "hash"])
            with gr.Column(scale=1):
                battery_plot = gr.Plot(label="Battery SoC per agent")
                reputation_plot = gr.Plot(label="Reputation per agent")
                promise_plot = gr.Plot(label="Cumulative promise-keep rate")

        with gr.Row():
            step_btn = gr.Button("Step ▶")
            play_btn = gr.Button("Play ▶▶")
            pause_btn = gr.Button("Pause ⏸")

        # callbacks: reset_btn, step_btn, play_btn, pause_btn → update state + render

    return ui

if __name__ == "__main__":
    build_ui().launch()
```

`Play` uses `gr.Timer(0.8)` (Gradio 4.36+) ticking every 800ms for smooth replay.

---

## Implementation order

| Phase | Hours | Deliverable | Verification |
|---|---|---|---|
| 0 — Scaffold | 0.5 | `requirements.txt`, README YAML header, empty `app.py` boots | `python app.py` opens localhost:7860 |
| 1 — Runner | 1.5 | `HeadlessRunner` reset/apply/snapshot works | Unit test: 5 manual actions → snapshot fields populate |
| 2 — Scripted player | 1.0 | Run `scripted_scenario.json` end-to-end | Final snapshot has 1 ledger entry, status=verified_kept |
| 3 — Random player | 0.5 | Run 1 baseline episode | Episode return ≈ 7.59 ± 2.72 (matches `eval/plots/baseline_rewards.json`) |
| 4 — Plots | 1.5 | 3 Plotly figures + ledger DF | Manual eyeball: SoC curves drop monotonically, reputation reacts to renege |
| 5 — Gradio UI | 1.5 | `app.py` wires everything | All buttons functional, no crash on edge cases (reset mid-play, switch modes mid-episode) |
| 6 — Pre-records | 0.5 | 3 episodes saved as JSON in `spaces/prerecorded/` | App auto-loads scripted episode on cold-start <2s |
| 7 — Deploy | 0.5 | HF Space live | Public URL works, RAM under 1.5GB, cold-start <30s |

**Total: ~7 hours.** Allow 1h buffer for HF deployment quirks (build cache, gradio version pinning).

---

## `requirements.txt`

```
openenv-core
fastmcp
fastapi
uvicorn[standard]
httpx
pydantic>=2.0
gradio>=4.36
plotly>=5.20
pandas>=2.0
```

(HF Spaces installs from `requirements.txt`; pyproject is ignored at deploy time. Keep both in sync.)

## `README.md` YAML header (top of file)

```yaml
---
title: AgentGrid V1
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
python_version: "3.11"
pinned: false
---
```

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Env's MCP tools are hard to call in-process | Fallback Path B (background uvicorn + AgentGridClient), adds 1s cold-start, zero env changes |
| `gr.Timer` not available in chosen Gradio version | Pin `gradio>=4.36`; alternative is JS polling via `gr.Interface(every=0.8)` |
| HF free tier sleeps after inactivity | Document in README; first request after sleep takes ~10s. Acceptable for demo |
| Multiple concurrent sessions exhaust RAM | `MAX_CONCURRENT_ENVS=4` already set in app.py env config; Gradio queues beyond that |
| Plotly figures slow on long episodes (50 steps × 3 agents) | Sub-30 datapoints — well within Plotly's smooth range. No mitigation needed |
| Scripted scenario hardcodes `__LAST_OFFER__` | Resolved in `scripted_player.py` by tracking last `make_offer` return. Already handled in plan |

---

## Verification (end-to-end)

1. **Local**: `python app.py` → localhost:7860 → click "Step" 5 times in scripted mode → transcript shows colored agent messages, ledger gets 1 entry, all 3 plots render.
2. **Reproducibility**: switch to "Random baseline", seed=42, click "Play" → final episode return matches `eval/plots/baseline_rewards.json` first entry within rounding.
3. **Audit**: pipe `runner.snapshot().recent_ledger` into `eval/verify_ledger_chain.py` after a full episode → "Hash chain OK".
4. **Trust model**: after reneging in manual mode, reputation plot for the offender drops by ≥0.10.
5. **HF deploy**: push, watch build, hit public URL, run scripted demo without errors, check RAM <1.5GB in Settings → Logs.

## Out of scope (explicitly)

- Live LLM inference (Llama-3.2-1B SFT model) — stays in Colab.
- HITL training mode — requires hardware, not relevant for HF.
- Persistent ledger across sessions — `:memory:` is sufficient.
- Authentication / private Space — public demo.
- Multi-episode sweep dashboard — see `docs/AgentGrid_v2.md` Section 7 if requested later.

---

## Cut list (if time runs short, cut in this order)

1. **Random baseline mode** — keep scripted only (saves ~2h)
2. **Manual step mode** — Play/Step buttons only on scripted (saves ~1h)
3. **Promise-keep curve** — keep just battery + reputation (saves ~30m)
4. **Pre-recorded JSONs** — fall back to live runs every cold-start (saves ~30m, costs 5s on first request)

The minimum viable Space = scripted_player + battery_curve + transcript + ledger table. ~3.5h floor.
