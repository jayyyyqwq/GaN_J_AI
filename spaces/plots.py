"""
Plotly figure builders for the Gradio UI.
All functions are pure: list[Snapshot] → go.Figure or pd.DataFrame.
No side effects, no state.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .runner import Snapshot, AGENTS

AGENT_COLORS: dict[str, str] = {
    "A": "#3B82F6",   # blue
    "B": "#10B981",   # green
    "C": "#EF4444",   # red
}

_LAYOUT_BASE = dict(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=48, r=20, t=52, b=40),
    height=270,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
)


def battery_curve(history: list[Snapshot]) -> go.Figure:
    """SoC per agent over episode steps."""
    fig = go.Figure()
    if not history:
        return _empty_fig("Battery SoC per Agent", "SoC (0–1)")
    steps = [s.game_step for s in history]
    for agent in AGENTS:
        fig.add_trace(go.Scatter(
            x=steps,
            y=[s.batteries[agent] for s in history],
            name=f"Agent {agent}",
            line=dict(color=AGENT_COLORS[agent], width=2),
            mode="lines+markers",
            marker=dict(size=4),
        ))
    fig.update_layout(
        title="Battery SoC per Agent",
        xaxis_title="Game Step",
        yaxis=dict(title="SoC (0–1)", range=[0, 1.05]),
        **_LAYOUT_BASE,
    )
    return fig


def reputation_curve(history: list[Snapshot]) -> go.Figure:
    """Reputation score per agent over episode steps."""
    fig = go.Figure()
    if not history:
        return _empty_fig("Reputation per Agent", "Reputation (0–1)")
    steps = [s.game_step for s in history]
    for agent in AGENTS:
        fig.add_trace(go.Scatter(
            x=steps,
            y=[s.reputation[agent] for s in history],
            name=f"Agent {agent}",
            line=dict(color=AGENT_COLORS[agent], width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(size=4, symbol="diamond"),
        ))
    fig.update_layout(
        title="Reputation per Agent",
        xaxis_title="Game Step",
        yaxis=dict(title="Reputation (0–1)", range=[0, 1.05]),
        **_LAYOUT_BASE,
    )
    return fig


def promise_keep_curve(history: list[Snapshot]) -> go.Figure:
    """Aggregate promise-keep ratio across all agents over episode steps."""
    fig = go.Figure()
    if not history:
        return _empty_fig("Promise-Keep Rate", "Ratio (0–1)")
    steps = [s.game_step for s in history]
    fig.add_trace(go.Scatter(
        x=steps,
        y=[s.promise_keep_ratio for s in history],
        name="Promise-keep ratio",
        line=dict(color="#8B5CF6", width=2),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.12)",
        mode="lines",
    ))
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(150,150,150,0.5)",
        annotation_text="50%",
        annotation_position="right",
    )
    fig.update_layout(
        title="Cumulative Promise-Keep Rate",
        xaxis_title="Game Step",
        yaxis=dict(title="Ratio (0–1)", range=[0, 1.05]),
        **_LAYOUT_BASE,
    )
    return fig


def ledger_table(history: list[Snapshot]) -> pd.DataFrame:
    """DataFrame of the 5 most recent ledger entries from the latest snapshot."""
    cols = ["step", "trade", "give", "want", "status", "hash"]
    if not history:
        return pd.DataFrame(columns=cols)
    entries = history[-1].recent_ledger
    if not entries:
        return pd.DataFrame(columns=cols)
    rows = [
        {
            "step": e.get("step", ""),
            "trade": f"{e.get('offerer', '')}→{e.get('accepter', '')}",
            "give": f"{e.get('give_amount', '')} {e.get('give_type', '')}",
            "want": f"{e.get('want_amount', '')} {e.get('want_type', '')}",
            "status": _status_label(e.get("status", "")),
            "hash": (e.get("this_hash", "") or "")[:12] + "…",
        }
        for e in entries
    ]
    return pd.DataFrame(rows)


def _status_label(status: str) -> str:
    return {
        "verified_kept": "✅ kept",
        "verified_broken": "❌ broken",
        "reneged": "⚠️ reneged",
        "pending": "⏳ pending",
    }.get(status, status)


def _empty_fig(title: str, yaxis_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Game Step",
        yaxis=dict(title=yaxis_title, range=[0, 1.05]),
        **_LAYOUT_BASE,
    )
    fig.add_annotation(
        text="No data yet — press Reset to begin",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#888"),
    )
    return fig
