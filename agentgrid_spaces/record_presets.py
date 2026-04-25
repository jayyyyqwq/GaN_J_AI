"""
Generate pre-recorded episodes for cold-start. Run once locally:
    python -m spaces.record_presets
"""
from __future__ import annotations

from pathlib import Path

from .runner import HeadlessRunner
from .scripted_player import scripted_episode
from .random_player import random_episode, episode_return
from .recorder import save_episode

_OUT = Path(__file__).parent / "prerecorded"


def main() -> None:
    print("Generating pre-recorded episodes...")

    # Scripted demo
    runner = HeadlessRunner(episode_steps=50)
    hist = scripted_episode(runner, seed=42)
    save_episode(hist, _OUT / "scripted_demo.json")
    print(f"  scripted_demo.json  — {len(hist)} steps")

    # Random baseline seeds
    for seed in (42, 123, 777):
        runner = HeadlessRunner(episode_steps=50)
        hist = random_episode(runner, seed=seed)
        ret = episode_return(hist)
        save_episode(hist, _OUT / f"baseline_seed_{seed}.json")
        print(f"  baseline_seed_{seed}.json — {len(hist)} steps, return={ret}")

    print("Done. Files written to spaces/prerecorded/")


if __name__ == "__main__":
    main()
