"""Generate three_curves.png — GRPO return vs random baseline."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "plots"


def smooth(x: list[float], w: int = 20) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> None:
    baseline_data = json.loads((ROOT / "baseline_rewards.json").read_text())
    baseline: list[float] = baseline_data["returns"]
    baseline_mean: float = baseline_data["mean"]

    grpo_raw = [json.loads(l) for l in (ROOT / "grpo_rewards.jsonl").read_text().splitlines() if l.strip()]
    grpo_avg = [sum(ep["rewards"].values()) / len(ep["rewards"]) for ep in grpo_raw]
    grpo_pk  = [ep["promise_keep"] for ep in grpo_raw]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: return curves ---
    ax = axes[0]
    ax.axhline(baseline_mean, linestyle="--", color="gray", alpha=0.7,
               label=f"Random baseline mean ({baseline_mean:.2f})")
    ax.scatter(range(len(baseline)), baseline, s=12, alpha=0.35, color="gray")

    ax.plot(grpo_avg, alpha=0.18, color="C0")
    if len(grpo_avg) >= 20:
        ax.plot(range(19, len(grpo_avg)), smooth(grpo_avg, 20),
                color="C0", lw=2, label="GRPO (smoothed w=20)")

    for ep, label in [(100, "medium"), (300, "full")]:
        if ep < len(grpo_avg):
            ax.axvline(ep, linestyle=":", color="C3", alpha=0.5)
            ax.text(ep + 2, ax.get_ylim()[0] + 0.3, label, color="C3", fontsize=8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg return per agent")
    ax.set_title("Return: GRPO vs Random Baseline")
    ax.legend(fontsize=9)

    # --- Right: promise-keep rate ---
    ax2 = axes[1]
    ax2.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="Random baseline (~0.50)")
    ax2.plot(grpo_pk, alpha=0.18, color="C1")
    if len(grpo_pk) >= 20:
        ax2.plot(range(19, len(grpo_pk)), smooth(grpo_pk, 20),
                 color="C1", lw=2, label="GRPO promise-keep (smoothed w=20)")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Promise-keep ratio")
    ax2.set_title("Promise-Keep Rate")
    ax2.legend(fontsize=9)

    fig.suptitle("AgentGrid V1 — GRPO Self-Play (sim mode, 3-stage curriculum)", fontsize=12)
    plt.tight_layout()
    out = ROOT / "three_curves.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
