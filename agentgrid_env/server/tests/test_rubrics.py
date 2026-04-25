"""
Unit tests for RubricScorer composable rubrics.
Run: pytest agentgrid_env/server/tests/test_rubrics.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from agentgrid_env.server.rubrics import (
    RubricScorer, SurvivalRubric, TaskRubric,
    PromiseRubric, JsonValidityRubric, CommunicationRubric,
)

AGENTS = ["A", "B", "C"]


def _full_batteries() -> dict:
    return {a: 1.0 for a in AGENTS}


def _tasks_no_completion() -> dict:
    return {
        a: {"urgency": 0.5, "energy_cost": 0.1, "reward_if_done": 3.0,
            "steps_pending": 1, "completed_this_step": False}
        for a in AGENTS
    }


def test_survival_all_alive():
    r = SurvivalRubric()
    scores = r.score(batteries=_full_batteries())
    assert all(s.value == 0.0 for s in scores)


def test_survival_one_dead():
    r = SurvivalRubric()
    batteries = _full_batteries()
    batteries["B"] = 0.0
    scores = {s.agent: s.value for s in r.score(batteries=batteries)}
    assert scores["B"] == -10.0
    assert scores["A"] == 0.0


def test_task_completion_reward():
    r = TaskRubric()
    tasks = _tasks_no_completion()
    tasks["A"]["completed_this_step"] = True
    tasks["A"]["urgency"] = 0.8
    scores = {s.agent: s.value for s in r.score(tasks=tasks, actions={})}
    assert abs(scores["A"] - 0.8 * 5.0) < 1e-9


def test_task_pending_penalty():
    r = TaskRubric()
    tasks = _tasks_no_completion()
    scores = {s.agent: s.value for s in r.score(tasks=tasks, actions={})}
    for a in AGENTS:
        assert abs(scores[a] - (-0.5 * 0.3)) < 1e-9


def test_promise_kept_reward():
    r = PromiseRubric()
    settlements = [{"offerer": "A", "accepter": "B", "status": "verified_kept"}]
    scores = {s.agent: s.value for s in r.score(step_settlements=settlements)}
    assert abs(scores["A"] - 1.0) < 1e-9
    assert scores["B"] == 0.0


def test_promise_broken_penalty():
    r = PromiseRubric()
    settlements = [{"offerer": "A", "accepter": "B", "status": "reneged"}]
    scores = {s.agent: s.value for s in r.score(step_settlements=settlements)}
    assert abs(scores["A"] - (-3.0)) < 1e-9


def test_promise_empty_settlements():
    r = PromiseRubric()
    scores = r.score(step_settlements=[])
    assert all(s.value == 0.0 for s in scores)


def test_json_validity_penalty():
    r = JsonValidityRubric()
    scores = {s.agent: s.value for s in r.score(parse_failures={"A": 2})}
    assert abs(scores["A"] - (-1.0)) < 1e-9
    assert scores["B"] == 0.0


def test_communication_reward():
    r = CommunicationRubric()
    scores = {s.agent: s.value for s in r.score(message_attribution={"C": 1})}
    assert abs(scores["C"] - 0.1) < 1e-9


def test_rubric_scorer_aggregates_correctly():
    scorer = RubricScorer()
    batteries = _full_batteries()
    tasks = _tasks_no_completion()
    tasks["A"]["completed_this_step"] = True
    tasks["A"]["urgency"] = 1.0

    rewards = scorer.score(
        batteries=batteries,
        tasks=tasks,
        actions={},
        step_settlements=[{"offerer": "A", "accepter": "B", "status": "verified_kept"}],
        parse_failures={},
        message_attribution={},
    )
    # A: survival=0, task=1.0*5.0=5.0, promise=0.8*1.0=0.8, json=0, comm=0 → 5.8
    assert abs(rewards["A"] - 5.8) < 1e-3
    # B: survival=0, task=-1.0*0.5*0.3=-0.15, promise=0, json=0, comm=0 → -0.15
    assert abs(rewards["B"] - (-0.15)) < 1e-3
