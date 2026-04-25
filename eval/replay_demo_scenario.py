"""
Replay the scripted demo scenario and log the transcript.

Loads demo/scripted_scenario.json, steps through it against a running env,
and prints the live transcript matching the 90-second pitch (Section 7 of v2.md).

Usage:
    python eval/replay_demo_scenario.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--scenario", default="demo/scripted_scenario.json")
    args = parser.parse_args()

    scenario = json.loads(Path(args.scenario).read_text())

    try:
        from agentgrid_env.client import AgentGridClient
    except ImportError:
        from openenv.core.mcp_client import MCPToolClient as AgentGridClient

    with AgentGridClient(base_url=args.url) as env:
        env.reset()
        print("\n=== AgentGrid Demo Replay ===\n")
        for step in scenario["steps"]:
            print(f"--- Step {step['step']} ---")
            for action in step["actions"]:
                agent = action["agent"]
                tool = action["tool"]
                kwargs = action.get("kwargs", {})
                kwargs["agent_id"] = agent
                result = env.call_tool(tool, **kwargs)
                print(f"[{agent}] {tool}: {result}")
                time.sleep(0.3)
            print()

        print("=== Replay complete ===")


if __name__ == "__main__":
    main()
