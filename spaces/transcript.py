"""
Markdown transcript renderer for the Gradio negotiation panel.
Ports the ANSI color logic from demo/transcript_projector.py to HTML spans.
"""
from __future__ import annotations

from .runner import Snapshot

AGENT_COLORS: dict[str, str] = {
    "A": "#3B82F6",
    "B": "#10B981",
    "C": "#EF4444",
}


def _agent_span(agent: str, text: str) -> str:
    color = AGENT_COLORS.get(agent, "#888888")
    return f'<span style="color:{color};font-weight:bold">{text}</span>'


def render_transcript(history: list[Snapshot]) -> str:
    """
    Build a Markdown/HTML string covering all resolved steps in history.
    Broadcasts are shown with colored agent names.
    Task completions and relay fires get callout lines.
    """
    if not history:
        return "_Episode not started. Press **Reset** to begin._"

    lines: list[str] = []

    for snap in history:
        lines.append(f"**── Step {snap.game_step} ──**")

        # Broadcasts first
        for msg in snap.step_messages:
            agent = msg["from"]
            lines.append(f'{_agent_span(agent, f"[{agent}]")} {msg["message"]}')

        # Non-broadcast actions
        for act in snap.step_actions:
            agent = act["agent"]
            action = act["action"]
            detail = act["detail"]

            if action == "broadcast":
                continue  # already shown above

            elif action == "make_offer":
                kw = act.get("kwargs", {})
                to = kw.get("to", "?")
                give = f'{kw.get("give_amount", "?")} {kw.get("give_type", "")}'
                want = f'{kw.get("want_amount", "?")} {kw.get("want_type", "")}'
                lines.append(
                    f'{_agent_span(agent, f"[{agent}]")} '
                    f'→ proposes to {_agent_span(to, f"[{to}]")}: '
                    f'give **{give}** for **{want}**'
                )

            elif action == "accept_offer":
                if "verified_kept" in detail:
                    lines.append(
                        f'&nbsp;&nbsp;⚡ **RELAY FIRED** — '
                        f'{_agent_span(agent, f"[{agent}]")} accepted. Promise **KEPT**'
                    )
                elif "verified_broken" in detail:
                    lines.append(
                        f'&nbsp;&nbsp;✗ Relay checked — '
                        f'{_agent_span(agent, f"[{agent}]")} accepted. Promise **BROKEN**'
                    )
                else:
                    lines.append(
                        f'&nbsp;&nbsp;{_agent_span(agent, f"[{agent}]")} accepted offer ({detail})'
                    )

            elif action == "execute_task":
                if "done" in detail.lower():
                    lines.append(
                        f'{_agent_span(agent, f"[{agent}]")} ✓ **TASK COMPLETE** — {detail}'
                    )
                else:
                    lines.append(
                        f'{_agent_span(agent, f"[{agent}]")} ✗ task failed — {detail}'
                    )

            elif action == "renege":
                lines.append(
                    f'{_agent_span(agent, f"[{agent}]")} ✗ **RENEGED** — {detail}'
                )

            # idle is intentionally omitted — visual noise

        # Battery summary line at end of each step
        batt_parts = [
            f'{_agent_span(a, a)}:{snap.batteries[a]:.2f}'
            for a in ("A", "B", "C")
        ]
        lines.append(f'<span style="color:#888;font-size:0.85em">  SoC: {" | ".join(batt_parts)}</span>')
        lines.append("")

    if history and history[-1].done:
        lines.append("---")
        lines.append("**Episode complete.**")

    return "\n\n".join(lines)
