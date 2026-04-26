"""Generate trust_correlation.png — Q(chosen partner) vs Q(alternatives) over training."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "plots"


def main() -> None:
    path = ROOT / "trust_decisions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run GRPO training first")

    events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not events:
        raise ValueError("trust_decisions.jsonl is empty — no accept_offer events captured")

    xs: list[float] = []
    ys: list[float] = []
    for ev in events:
        alts = list(ev.get("Q_alternatives", {}).values())
        if not alts:
            continue
        delta = float(ev["Q_chosen"]) - float(np.mean(alts))
        xs.append(float(ev["step"]))
        ys.append(delta)

    if not xs:
        raise ValueError("No events have Q_alternatives — check env version")

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    order = np.argsort(xs_arr)
    xs_s, ys_s = xs_arr[order], ys_arr[order]

    window = max(30, len(ys_s) // 50)
    smoothed = np.convolve(ys_s, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(11, 4.8))
    plt.scatter(xs_s, ys_s, s=5, alpha=0.13, color="C0", label="Per-accept Δ Q")
    plt.plot(xs_s[window - 1:], smoothed, color="C1", lw=2,
             label=f"Smoothed (w={window})")
    plt.axhline(0.0, color="gray", linestyle="--", alpha=0.5,
                label="Expected Δ under random partner selection")

    # Annotate trend direction
    if len(smoothed) >= 2:
        trend = smoothed[-1] - smoothed[0]
        direction = "↑ LLM prefers higher-trust partners" if trend > 0.01 else \
                    "→ No clear trust-signal pickup" if abs(trend) <= 0.01 else \
                    "↓ Unexpected — check training"
        plt.annotate(direction, xy=(0.98, 0.92), xycoords="axes fraction",
                     ha="right", fontsize=10, color="C1")

    plt.xlabel("Training step")
    plt.ylabel("Q(chosen) − mean(Q(alternatives))")
    plt.title("AgentGrid V1 — LLM Trust-Signal Pickup During GRPO Training")
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = ROOT / "trust_correlation.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"Events plotted: {len(xs)}")
    print(f"Smoothed trend: {smoothed[-1] - smoothed[0]:+.4f}")


if __name__ == "__main__":
    main()
