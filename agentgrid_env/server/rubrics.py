"""
Composable rubric scorers for AgentGrid.

Per OpenEnv guidance: composable rubrics > monolithic scoring.
Each rubric scores one dimension; RubricScorer sums weighted results.
"""
from __future__ import annotations

from dataclasses import dataclass

AGENTS = ["A", "B", "C"]


@dataclass
class _Score:
    agent: str
    value: float
    label: str


class SurvivalRubric:
    """Binary: agent alive = 0, dead = -10."""
    weight: float = 1.0

    def score(self, batteries: dict[str, float], **_) -> list[_Score]:
        return [
            _Score(a, 0.0 if batteries[a] > 0 else -10.0, "survival")
            for a in AGENTS
        ]


class TaskRubric:
    """
    urgency * 5 for completing a task this step.
    -urgency * 0.3 per step the task remains pending.
    """
    weight: float = 1.0

    def score(self, tasks: dict[str, dict], actions: dict[str, dict], **_) -> list[_Score]:
        scores = []
        for a in AGENTS:
            task = tasks[a]
            if task.get("completed_this_step"):
                v = task["urgency"] * 5.0
            else:
                v = -task["urgency"] * 0.3
            scores.append(_Score(a, v, "task"))
        return scores


class PromiseRubric:
    """
    +1 per verified_kept settlement this step (offerer's cell).
    -3 per verified_broken / reneged settlement this step.
    `step_settlements` is a list of dicts passed from the env for this step only.
    """
    weight: float = 0.8

    def score(
        self,
        step_settlements: list[dict],
        **_,
    ) -> list[_Score]:
        scores = []
        for a in AGENTS:
            kept = sum(1 for e in step_settlements if e["offerer"] == a and e["status"] == "verified_kept")
            broken = sum(
                1 for e in step_settlements
                if e["offerer"] == a and e["status"] in ("verified_broken", "reneged")
            )
            scores.append(_Score(a, kept * 1.0 - broken * 3.0, "promise"))
        return scores


class JsonValidityRubric:
    """
    -0.5 per action that failed to parse as valid JSON (curriculum trick).
    Tracks parse failures injected by the env during action intake.
    """
    weight: float = 0.2

    def score(self, parse_failures: dict[str, int], **_) -> list[_Score]:
        return [
            _Score(a, -0.5 * parse_failures.get(a, 0), "json_validity")
            for a in AGENTS
        ]


class CommunicationRubric:
    """
    +0.1 if a broadcast message from this agent led to a settled trade within 3 steps.
    Tracked via message_attribution dict maintained by the env.
    """
    weight: float = 0.3

    def score(self, message_attribution: dict[str, int], **_) -> list[_Score]:
        return [
            _Score(a, 0.1 * message_attribution.get(a, 0), "communication")
            for a in AGENTS
        ]


class RubricScorer:
    """Aggregate all rubric components into per-agent scalar rewards."""

    def __init__(self) -> None:
        self._survival = SurvivalRubric()
        self._task = TaskRubric()
        self._promise = PromiseRubric()
        self._json = JsonValidityRubric()
        self._comm = CommunicationRubric()

    def score(
        self,
        *,
        batteries: dict[str, float],
        tasks: dict[str, dict],
        actions: dict[str, dict],
        step_settlements: list[dict] | None = None,
        parse_failures: dict[str, int] | None = None,
        message_attribution: dict[str, int] | None = None,
    ) -> dict[str, float]:
        step_settlements = step_settlements or []
        parse_failures = parse_failures or {}
        message_attribution = message_attribution or {}

        components = [
            (self._survival.weight, self._survival.score(batteries=batteries)),
            (self._task.weight, self._task.score(tasks=tasks, actions=actions)),
            (self._promise.weight, self._promise.score(step_settlements=step_settlements)),
            (self._json.weight, self._json.score(parse_failures=parse_failures)),
            (self._comm.weight, self._comm.score(message_attribution=message_attribution)),
        ]

        totals: dict[str, float] = {a: 0.0 for a in AGENTS}
        for weight, scores in components:
            for s in scores:
                totals[s.agent] += weight * s.value
        return {a: round(totals[a], 4) for a in AGENTS}
