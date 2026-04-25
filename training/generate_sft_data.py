"""
Generate SFT training data for AgentGrid via HF Inference Providers (DeepSeek V3).

Produces (observation -> action_json) pairs covering all six action types:
broadcast, make_offer, accept_offer, execute_task, renege, idle.

Output: training/synthetic_traces/sft_data.jsonl
Each line: {"prompt": "<rendered obs>", "completion": "<json action>"}

Resumes from existing rows if interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient

AGENTS = ("A", "B", "C")
ACTIONS = ("broadcast", "make_offer", "accept_offer", "execute_task", "renege", "idle")
MODEL = "deepseek-ai/DeepSeek-V3"
PROVIDER = "novita"

OUT_PATH = Path("training/synthetic_traces/sft_data.jsonl")


SYSTEM_PROMPT = """You are an agent in a 3-agent energy negotiation environment.
Each turn you choose ONE action. You must reply with ONLY a single JSON object,
no prose, no markdown fences, no commentary.

Allowed action shapes:
  {"action": "broadcast",     "agent_id": "A|B|C", "message": "<short text>"}
  {"action": "make_offer",    "agent_id": "A|B|C", "to": "A|B|C",
                              "give_type": "energy|compute", "give_amount": <float>,
                              "want_type": "energy|compute", "want_amount": <float>}
  {"action": "accept_offer",  "agent_id": "A|B|C", "offer_id": "<id>"}
  {"action": "execute_task",  "agent_id": "A|B|C"}
  {"action": "renege",        "agent_id": "A|B|C", "offer_id": "<id>"}
  {"action": "idle",           "agent_id": "A|B|C"}

Rules:
- agent_id must equal YOUR_AGENT in the prompt.
- accept_offer / renege only apply if a matching offer exists in PENDING OFFERS.
- For make_offer, give_type and want_type MUST differ (energy↔compute, never same).
- Energy amounts are floats in [0.05, 0.30] (battery units, voltmeter-derived).
- Compute amounts are integers in [1, 8] (compute slots).
- Use the observation: low battery → consider asking for energy; high reputation peer → safer to trust.
- Keep broadcast messages under 80 characters and specific (mention amounts or intent).
"""


def render_synthetic_observation(target_agent: str, force_action_hint: str) -> tuple[str, dict]:
    """Build a fake but plausible observation. Returns (prompt_text, scenario_meta)."""
    batteries = {a: round(random.uniform(0.6, 4.1), 2) for a in AGENTS}
    reputations = {a: round(random.uniform(0.3, 1.0), 2) for a in AGENTS}
    task_cost = round(random.uniform(0.15, 0.45), 2)
    task_reward = random.choice([5, 8, 10, 12])

    pending_offers: list[dict] = []
    if force_action_hint in ("accept_offer", "renege"):
        # Fabricate at least one matching offer
        if force_action_hint == "accept_offer":
            offerer = random.choice([a for a in AGENTS if a != target_agent])
            recipient = target_agent
        else:  # renege: target is the offerer
            offerer = target_agent
            recipient = random.choice([a for a in AGENTS if a != target_agent])
        give_type = random.choice(["energy", "compute"])
        want_type = "compute" if give_type == "energy" else "energy"

        def _amount(t: str) -> float:
            if t == "energy":
                return round(random.uniform(0.05, 0.25), 2)
            return float(random.randint(1, 6))

        pending_offers.append({
            "offer_id": f"OF-{random.randint(0, 0xFFFFFF):06X}",
            "from": offerer,
            "to": recipient,
            "give_type": give_type,
            "give_amount": _amount(give_type),
            "want_type": want_type,
            "want_amount": _amount(want_type),
        })

    inbox: list[dict] = []
    if random.random() < 0.5:
        sender = random.choice([a for a in AGENTS if a != target_agent])
        inbox.append({
            "from": sender,
            "message": random.choice([
                "Spare energy if anyone needs.",
                "Battery low, looking for trades.",
                "Will pay 4 compute for 0.15 energy.",
                "Urgency spike incoming.",
            ]),
        })

    trust_block = {
        f"{p}": {
            "Q_accept": round(random.uniform(-1.5, 1.0), 2),
            "Q_trust": round(random.uniform(-1.5, 1.0), 2),
            "UCB": round(random.uniform(0.0, 2.5), 2),
            "N": random.randint(0, 12),
        }
        for p in AGENTS if p != target_agent
    }

    obs = {
        "YOUR_AGENT": target_agent,
        "STEP": random.randint(0, 9),
        "BATTERY": batteries[target_agent],
        "TASK": {"energy_cost": task_cost, "reward_if_done": task_reward},
        "PEERS": {p: {"battery": batteries[p], "reputation": reputations[p]}
                  for p in AGENTS if p != target_agent},
        "TRUST_MODEL": trust_block,
        "INBOX": inbox,
        "PENDING_OFFERS": pending_offers,
    }

    prompt = (
        f"{json.dumps(obs, indent=2)}\n\n"
        f"Hint: a reasonable next action right now is `{force_action_hint}`.\n"
        f"Reply with the JSON action."
    )
    return prompt, {"target_agent": target_agent, "hint": force_action_hint, "obs": obs}


def validate_action(action: dict, scenario: dict) -> bool:
    """Return True if action conforms to schema and matches scenario constraints."""
    if not isinstance(action, dict) or "action" not in action:
        return False
    a = action["action"]
    if a not in ACTIONS:
        return False
    if action.get("agent_id") != scenario["target_agent"]:
        return False
    if a == "broadcast":
        return isinstance(action.get("message"), str) and len(action["message"]) > 0
    if a == "make_offer":
        keys = {"to", "give_type", "give_amount", "want_type", "want_amount"}
        if not keys.issubset(action):
            return False
        if action["to"] == scenario["target_agent"]:
            return False
        if action["give_type"] not in ("energy", "compute"):
            return False
        if action["want_type"] not in ("energy", "compute"):
            return False
        if action["give_type"] == action["want_type"]:
            return False
        return True
    if a in ("accept_offer", "renege"):
        if "offer_id" not in action:
            return False
        offers = scenario["obs"]["PENDING_OFFERS"]
        return any(o["offer_id"] == action["offer_id"] for o in offers)
    if a in ("execute_task", "idle"):
        return True
    return False


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response, tolerating fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    try:
        return json.loads(s.strip())
    except json.JSONDecodeError:
        # Last-ditch: find first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None


def call_model(client: InferenceClient, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.9,
    )
    return resp.choices[0].message.content or ""


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0,
                        help="If >0, only generate this many rows (for quick sanity check).")
    args = parser.parse_args()

    random.seed(args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    target = args.smoke if args.smoke > 0 else args.target
    have = count_existing(OUT_PATH)
    need = max(0, target - have)
    print(f"target={target}  existing={have}  need={need}")
    if need == 0:
        print("Nothing to do.")
        return

    if "HF_TOKEN" not in os.environ:
        raise SystemExit("HF_TOKEN not set in this shell. Reopen PowerShell after setting it.")

    client = InferenceClient(provider=PROVIDER, api_key=os.environ["HF_TOKEN"])

    written = 0
    attempts = 0
    invalid = 0
    api_errors = 0
    t0 = time.time()

    with OUT_PATH.open("a", encoding="utf-8") as f:
        while written < need:
            attempts += 1
            target_agent = random.choice(AGENTS)
            hint = random.choice(ACTIONS)
            prompt, scenario = render_synthetic_observation(target_agent, hint)

            try:
                raw = call_model(client, prompt)
            except Exception as e:  # network / provider hiccup
                api_errors += 1
                print(f"  api error #{api_errors}: {type(e).__name__}: {e}")
                if api_errors > 25:
                    raise SystemExit("Too many API errors — aborting.")
                time.sleep(2.0)
                continue

            action = extract_json(raw)
            if action is None or not validate_action(action, scenario):
                invalid += 1
                continue

            row = {"prompt": prompt, "completion": json.dumps(action)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            written += 1

            if written % 25 == 0 or written == need:
                rate = written / max(1e-3, time.time() - t0)
                validity = written / max(1, attempts)
                print(f"  written={written}/{need}  attempts={attempts}  "
                      f"validity={validity:.1%}  invalid={invalid}  "
                      f"api_errors={api_errors}  rate={rate:.2f}/s")

    elapsed = time.time() - t0
    print(f"\nDone. wrote {written} rows in {elapsed:.1f}s. file: {OUT_PATH}")
    print(f"final validity rate: {written / max(1, attempts):.1%}")


if __name__ == "__main__":
    main()
