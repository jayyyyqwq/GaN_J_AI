"""
Scripted episode runner.
Replays demo/scripted_scenario.json step-by-step through HeadlessRunner.
"""
from __future__ import annotations

import json
from pathlib import Path

from .runner import HeadlessRunner, Snapshot

_DEFAULT_SCENARIO = Path(__file__).parent.parent / "demo" / "scripted_scenario.json"


def scripted_episode(
    runner: HeadlessRunner,
    scenario_path: Path = _DEFAULT_SCENARIO,
    seed: int | None = 42,
) -> list[Snapshot]:
    """
    Run the scripted scenario end-to-end.
    Returns one Snapshot per resolved game step.
    """
    scenario = json.loads(scenario_path.read_text())
    runner.reset(seed=seed)
    history: list[Snapshot] = []
    last_offer_id: str | None = None

    for step_def in scenario["steps"]:
        for action_def in step_def["actions"]:
            agent = action_def["agent"]
            tool = action_def["tool"]
            kwargs: dict = dict(action_def.get("kwargs", {}))

            # Resolve __LAST_OFFER__ placeholder with the most recent offer_id
            if kwargs.get("offer_id") == "__LAST_OFFER__":
                kwargs["offer_id"] = last_offer_id or ""

            ret = runner.apply(agent, tool, **kwargs)

            if tool == "make_offer" and ret.startswith("OFR-"):
                last_offer_id = ret

        history.append(runner.snapshot())
        if history[-1].done:
            break

    return history


def scripted_step(
    runner: HeadlessRunner,
    step_def: dict,
    last_offer_id: str | None = None,
) -> tuple[Snapshot, str | None]:
    """
    Execute a single step definition from the scenario.
    Returns (snapshot_after_step, updated_last_offer_id).
    Used by the Gradio step-by-step UI.
    """
    for action_def in step_def["actions"]:
        agent = action_def["agent"]
        tool = action_def["tool"]
        kwargs: dict = dict(action_def.get("kwargs", {}))

        if kwargs.get("offer_id") == "__LAST_OFFER__":
            kwargs["offer_id"] = last_offer_id or ""

        ret = runner.apply(agent, tool, **kwargs)

        if tool == "make_offer" and ret.startswith("OFR-"):
            last_offer_id = ret

    return runner.snapshot(), last_offer_id
