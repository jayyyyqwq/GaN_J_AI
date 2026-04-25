"""
AgentGrid V1 — Hugging Face Spaces demo.

Three LLM agents negotiate energy in plain English.
Battery physics: 18650-style SoC→OCV curve + Uno 10-bit ADC noise model.
Commitment ledger: SHA-256 hash-chained SQLite (in-memory per session).
No hardware, no live LLM inference — sim mode only.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from agentgrid_spaces.runner import HeadlessRunner
from agentgrid_spaces.scripted_player import scripted_episode
from agentgrid_spaces.random_player import random_episode, episode_return
from agentgrid_spaces.plots import battery_curve, reputation_curve, promise_keep_curve, ledger_table
from agentgrid_spaces.transcript import render_transcript
from agentgrid_spaces.recorder import load_prerecorded, save_episode

# ── Constants ──────────────────────────────────────────────────────────────────

_MODES = ["Scripted demo", "Random baseline (seeded)"]
_EPISODE_STEPS = 50

_INFO_MD = """
## AgentGrid V1

> *The only OpenEnv submission where the reward function reads from a voltmeter.*

**3 Llama-3.2-1B agents** negotiate energy and compute in plain English.
A hash-chained ledger records every promise.
In hardware mode, a relay clicks and an Arduino Uno ADC reads the actual voltage delta — lying has a voltage.
This Space runs in **sim mode** (calibrated 18650-style SoC→OCV curve, Uno ADC noise model).

| Rubric | Signal | Weight |
|--------|--------|--------|
| Survival | +0 alive, −10 dead | 1.0 |
| Task | urgency×5 on completion, −urgency×0.3/step | 1.0 |
| Promise | +1 verified_kept, −3 reneged | 0.8 |
| Communication | +0.1 if broadcast led to trade within 3 steps | 0.3 |

**Scripted demo** — 5-step scenario: judge waves at HC-SR04, C's urgency spikes, A negotiates, relay fires, C completes task.
**Random baseline** — seeded random policy. Mean return ≈ 7–10 ± 2.7.
"""


# ── State helpers ──────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {"history": [], "display_idx": 0, "mode": _MODES[0]}


def _render_all(history: list, display_idx: int):
    """Return all 5 display outputs for the given history slice."""
    visible = history[:display_idx]
    return (
        render_transcript(visible),
        ledger_table(visible),
        battery_curve(visible),
        reputation_curve(visible),
        promise_keep_curve(visible),
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def cb_reset(mode: str, seed: int, state: dict) -> tuple:
    """Compute full episode for the selected mode and seed, reset display to step 0."""
    if mode == "Scripted demo":
        # Try pre-recorded first for instant load; fall back to live compute
        history = load_prerecorded("scripted_demo") if seed == 42 else None
        if history is None:
            try:
                runner = HeadlessRunner(episode_steps=_EPISODE_STEPS)
                history = scripted_episode(runner, seed=seed)
            except Exception:
                history = load_prerecorded("scripted_demo") or []
    else:
        prekey = f"baseline_seed_{seed}" if seed in (42, 123, 777) else None
        history = (load_prerecorded(prekey) if prekey else None)
        if history is None:
            try:
                runner = HeadlessRunner(episode_steps=_EPISODE_STEPS)
                history = random_episode(runner, seed=seed)
            except Exception:
                history = load_prerecorded("baseline_seed_42") or []

    new_state = {"history": history, "display_idx": 0, "mode": mode}
    step_label = f"Step 1/{len(history)}"
    return (new_state, step_label) + _render_all(history, 0)


def cb_step(state: dict) -> tuple:
    """Advance display by one game step."""
    if not state or not state.get("history"):
        return (state, "No episode loaded") + _render_all([], 0)
    history = state["history"]
    idx = min(state["display_idx"] + 1, len(history))
    new_state = {**state, "display_idx": idx}
    label = f"Step {idx}/{len(history)}"
    if idx == len(history) and history[-1].done:
        label += " — Episode complete"
    return (new_state, label) + _render_all(history, idx)


def cb_play_all(state: dict) -> tuple:
    """Jump display to the end of the episode."""
    if not state or not state.get("history"):
        return (state, "No episode loaded") + _render_all([], 0)
    history = state["history"]
    idx = len(history)
    new_state = {**state, "display_idx": idx}
    label = f"Step {idx}/{idx} — Episode complete"
    return (new_state, label) + _render_all(history, idx)


def cb_timer_tick(state: dict, playing: bool) -> tuple:
    """Called by gr.Timer when auto-play is active. Steps forward or stops."""
    if not playing or not state or not state.get("history"):
        return (state, gr.update(), False, "▶ Play") + _render_all(
            state.get("history", []), state.get("display_idx", 0) if state else 0
        )
    history = state["history"]
    idx = state["display_idx"]
    if idx >= len(history):
        # Reached end — stop timer
        return (state, gr.update(active=False), False, "▶ Play") + _render_all(history, idx)
    # Advance one step
    new_idx = idx + 1
    new_state = {**state, "display_idx": new_idx}
    label = f"Step {new_idx}/{len(history)}"
    still_playing = new_idx < len(history)
    timer_update = gr.update(active=still_playing)
    return (new_state, timer_update, still_playing, "⏸ Pause" if still_playing else "▶ Play") + _render_all(history, new_idx)


def cb_toggle_play(state: dict, playing: bool) -> tuple:
    """Toggle auto-play on/off."""
    if not state or not state.get("history"):
        return state, gr.update(active=False), False, "▶ Play"
    new_playing = not playing
    return state, gr.update(active=new_playing), new_playing, "⏸ Pause" if new_playing else "▶ Play"


# ── UI Layout ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AgentGrid V1 — Live Demo", theme=gr.themes.Soft()) as ui:

        # ── Info panel ──────────────────────────────────────────
        with gr.Accordion("About AgentGrid V1", open=False):
            gr.Markdown(_INFO_MD)

        # ── Session state ────────────────────────────────────────
        state = gr.State(_empty_state())
        playing_state = gr.State(False)

        # ── Controls row ─────────────────────────────────────────
        gr.Markdown("### Controls")
        with gr.Row():
            mode_radio = gr.Radio(
                choices=_MODES,
                value=_MODES[0],
                label="Mode",
                scale=2,
            )
            seed_slider = gr.Slider(
                minimum=0, maximum=999, value=42, step=1,
                label="Seed",
                scale=1,
            )
            reset_btn = gr.Button("⟳ Reset", variant="primary", scale=1)

        with gr.Row():
            step_btn = gr.Button("Step ▶", scale=1)
            play_btn = gr.Button("▶ Play", scale=1, variant="secondary")
            play_all_btn = gr.Button("▶▶ Play All", scale=1)
            step_label = gr.Textbox(
                value="Press Reset to load episode",
                label="Progress",
                interactive=False,
                scale=2,
            )

        # Timer for auto-play (fires every 900ms when active)
        timer = gr.Timer(value=0.9, active=False)

        # ── Main content ─────────────────────────────────────────
        with gr.Row():
            # Left: transcript + ledger
            with gr.Column(scale=1):
                transcript_md = gr.Markdown(
                    value="_Press Reset to begin._",
                    label="Negotiation Transcript",
                    elem_id="transcript-panel",
                )
                gr.Markdown("**Commitment Ledger** (last 5 entries)")
                ledger_df = gr.Dataframe(
                    headers=["step", "trade", "give", "want", "status", "hash"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    row_count=5,
                    interactive=False,
                )

            # Right: plots
            with gr.Column(scale=1):
                batt_plot = gr.Plot(label="Battery SoC per Agent")
                rep_plot = gr.Plot(label="Reputation per Agent")
                promise_plot = gr.Plot(label="Cumulative Promise-Keep Rate")

        # ── Wire callbacks ────────────────────────────────────────

        _display_outputs = [transcript_md, ledger_df, batt_plot, rep_plot, promise_plot]
        _reset_outputs = [state, step_label] + _display_outputs
        _step_outputs = [state, step_label] + _display_outputs
        _timer_outputs = [state, timer, playing_state, play_btn] + _display_outputs
        _toggle_outputs = [state, timer, playing_state, play_btn]

        reset_btn.click(
            fn=cb_reset,
            inputs=[mode_radio, seed_slider, state],
            outputs=_reset_outputs,
        )

        step_btn.click(
            fn=cb_step,
            inputs=[state],
            outputs=_step_outputs,
        )

        play_all_btn.click(
            fn=cb_play_all,
            inputs=[state],
            outputs=_step_outputs,
        )

        play_btn.click(
            fn=cb_toggle_play,
            inputs=[state, playing_state],
            outputs=_toggle_outputs,
        )

        timer.tick(
            fn=cb_timer_tick,
            inputs=[state, playing_state],
            outputs=_timer_outputs,
        )

        # Auto-reset on mode change to prevent stale state
        mode_radio.change(
            fn=cb_reset,
            inputs=[mode_radio, seed_slider, state],
            outputs=_reset_outputs,
        )

    return ui


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
