"""
Live transcript projector for the 90-second demo pitch.

Connects to a running env server and streams agent messages to the terminal
in real time. Run this on the presentation laptop; project the terminal.

Usage:
    python demo/transcript_projector.py --url http://localhost:8000

Keyboard shortcuts during demo:
    Ctrl-C  — stop
    s       — trigger urgency spike on Agent C (simulates judge hand wave)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

ANSI = {
    "A": "\033[94m",   # blue
    "B": "\033[92m",   # green
    "C": "\033[91m",   # red
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "YELLOW": "\033[93m",
}


def _color(agent: str, text: str) -> str:
    return f"{ANSI.get(agent, '')}{text}{ANSI['RESET']}"


def _header() -> None:
    print(f"\n{ANSI['BOLD']}{'='*60}{ANSI['RESET']}")
    print(f"{ANSI['BOLD']}  AgentGrid — Live Negotiation Transcript{ANSI['RESET']}")
    print(f"{ANSI['DIM']}  Agents: A (blue)  B (green)  C (red){ANSI['RESET']}")
    print(f"{ANSI['BOLD']}{'='*60}{ANSI['RESET']}\n")


def _print_event(event: dict) -> None:
    step = event.get("game_step", "?")
    agent = event.get("agent", "?")
    action = event.get("action", "?")
    detail = event.get("detail", "")
    prefix = f"{ANSI['DIM']}[step {step:>2}]{ANSI['RESET']} "

    if action == "broadcast":
        print(f"{prefix}{_color(agent, f'[{agent}]')} {detail}")
    elif action in ("make_offer", "accept_offer"):
        print(f"{prefix}{_color(agent, f'[{agent}]')} {ANSI['YELLOW']}{detail}{ANSI['RESET']}")
    elif action == "execute_task":
        print(f"{prefix}{_color(agent, f'[{agent}]')} ✓ TASK COMPLETE — {detail}")
    elif action == "renege":
        print(f"{prefix}{_color(agent, f'[{agent}]')} ✗ RENEGED — {detail}")
    elif action == "verified_kept":
        print(f"{prefix}  → {ANSI['BOLD']}RELAY FIRED{ANSI['RESET']} — promise KEPT  {detail}")
    elif action == "verified_broken":
        print(f"{prefix}  → RELAY checked — promise BROKEN  {detail}")
    else:
        print(f"{prefix}{_color(agent, f'[{agent}]')} {detail}")


def _poll_loop(env, stop: threading.Event) -> None:
    """Poll get_step_result for all agents and print state changes."""
    last_step = -1
    while not stop.is_set():
        try:
            for agent_id in ("A", "B", "C"):
                raw = env.call_tool("get_step_result", agent_id=agent_id)
                result = json.loads(raw) if isinstance(raw, str) else raw
                game_step = result.get("game_step", 0)
                if game_step != last_step:
                    last_step = game_step
                    battery = result.get("battery", 0)
                    reward = result.get("reward", 0)
                    print(
                        f"{ANSI['DIM']}  Agent {agent_id}: "
                        f"battery={battery:.2f}  reward={reward:+.3f}{ANSI['RESET']}"
                    )
                if result.get("done"):
                    print(f"\n{ANSI['BOLD']}Episode complete at step {game_step}.{ANSI['RESET']}\n")
                    stop.set()
                    return
        except Exception:
            pass
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    try:
        from agentgrid_env.client import AgentGridClient
    except ImportError:
        from openenv.core.mcp_client import MCPToolClient as AgentGridClient  # type: ignore

    _header()

    with AgentGridClient(base_url=args.url) as env:
        for ep in range(args.episodes):
            print(f"{ANSI['BOLD']}─── Episode {ep + 1} ───{ANSI['RESET']}\n")
            env.reset()

            stop = threading.Event()
            poller = threading.Thread(target=_poll_loop, args=(env, stop), daemon=True)
            poller.start()

            # Give each agent its observation and let it act (manual demo mode)
            try:
                while not stop.is_set():
                    for agent_id in ("A", "B", "C"):
                        if stop.is_set():
                            break
                        obs = env.call_tool("get_observation", agent_id=agent_id)
                        # In demo mode, print observation for the presenter only
                        # (not projected — too verbose)
                        _ = obs  # actual LLM would read this
                        time.sleep(0.2)
            except KeyboardInterrupt:
                stop.set()
                print("\nStopped by user.")
                sys.exit(0)

            poller.join(timeout=2)
            print()


if __name__ == "__main__":
    main()
