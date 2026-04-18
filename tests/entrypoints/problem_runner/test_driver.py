from __future__ import annotations

import json
from pathlib import Path

from slop_code.agent_runner import AgentStateEnum
from slop_code.common import EVALUATION_FILENAME
from slop_code.common import INFERENCE_RESULT_FILENAME
from slop_code.entrypoints.problem_runner.driver import (
    _prepopulate_completed_states,
)
from slop_code.entrypoints.problem_runner.state import ProblemStateTracker
from slop_code.evaluation import GroupType


def _write_checkpoint_artifacts(
    checkpoint_dir: Path,
    *,
    cost: float,
    steps: int,
    core_pass: int,
    core_total: int,
    functionality_pass: int = 0,
    functionality_total: int = 0,
    regression_pass: int = 0,
    regression_total: int = 0,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inference = {
        "usage": {
            "cost": cost,
            "steps": steps,
            "net_tokens": {"input": 100, "output": 50},
        }
    }
    with (checkpoint_dir / INFERENCE_RESULT_FILENAME).open("w") as f:
        json.dump(inference, f)

    evaluation = {
        "schema_version": 1,
        "problem_name": "p",
        "problem_version": 1,
        "checkpoint_name": checkpoint_dir.name,
        "checkpoint_version": 1,
        "duration": 1.0,
        "entrypoint": "python main.py",
        "tests": [],
        "pass_counts": {
            GroupType.CORE.value: core_pass,
            GroupType.FUNCTIONALITY.value: functionality_pass,
            GroupType.REGRESSION.value: regression_pass,
            GroupType.ERROR.value: 0,
        },
        "total_counts": {
            GroupType.CORE.value: core_total,
            GroupType.FUNCTIONALITY.value: functionality_total,
            GroupType.REGRESSION.value: regression_total,
            GroupType.ERROR.value: 0,
        },
        "pytest_exit_code": 0,
        "pytest_collected": core_total
        + functionality_total
        + regression_total,
        "infrastructure_failure": False,
    }
    with (checkpoint_dir / EVALUATION_FILENAME).open("w") as f:
        json.dump(evaluation, f)


def _make_config(run_dir: Path):
    # Minimal duck-typed stand-in: _prepopulate_completed_states only reads
    # config.run_dir, so a tiny object suffices.
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.run_dir = run_dir
    return cfg


def test_prepopulate_loads_cost_and_passed_counts(tmp_path: Path) -> None:
    problem = "p"
    problem_dir = tmp_path / problem
    _write_checkpoint_artifacts(
        problem_dir / "checkpoint_1",
        cost=1.5,
        steps=10,
        core_pass=3,
        core_total=3,
    )
    _write_checkpoint_artifacts(
        problem_dir / "checkpoint_2",
        cost=2.25,
        steps=20,
        core_pass=2,
        core_total=2,
        functionality_pass=1,
        functionality_total=1,
    )

    checkpoint_map = {problem: ["checkpoint_1", "checkpoint_2"]}
    states = ProblemStateTracker([problem], checkpoint_map)

    _prepopulate_completed_states(
        states, [problem], _make_config(tmp_path), checkpoint_map
    )

    state = states[problem]
    assert state.state == AgentStateEnum.COMPLETED
    assert state.overall_usage is not None
    assert state.overall_usage.cost == 3.75
    assert state.overall_usage.steps == 30
    assert state.total_checkpoints_evaluated == 2
    # Both checkpoints are fully-passed including functionality / regression
    assert state.checkpoints_passed == 2
    assert state.checkpoints_iso_passed == 2
    assert state.checkpoints_core_solved == 2


def test_prepopulate_marks_partial_core_pass(tmp_path: Path) -> None:
    problem = "p"
    problem_dir = tmp_path / problem
    _write_checkpoint_artifacts(
        problem_dir / "checkpoint_1",
        cost=1.0,
        steps=5,
        core_pass=2,
        core_total=3,  # one core test failed
    )

    checkpoint_map = {problem: ["checkpoint_1"]}
    states = ProblemStateTracker([problem], checkpoint_map)

    _prepopulate_completed_states(
        states, [problem], _make_config(tmp_path), checkpoint_map
    )

    state = states[problem]
    assert state.total_checkpoints_evaluated == 1
    assert state.checkpoints_passed == 0
    assert state.checkpoints_iso_passed == 0
    assert state.checkpoints_core_solved == 0


def test_prepopulate_handles_missing_evaluation(tmp_path: Path) -> None:
    problem = "p"
    problem_dir = tmp_path / problem
    (problem_dir / "checkpoint_1").mkdir(parents=True)
    with (
        problem_dir / "checkpoint_1" / INFERENCE_RESULT_FILENAME
    ).open("w") as f:
        json.dump({"usage": {"cost": 0.5, "steps": 1}}, f)

    checkpoint_map = {problem: ["checkpoint_1"]}
    states = ProblemStateTracker([problem], checkpoint_map)

    _prepopulate_completed_states(
        states, [problem], _make_config(tmp_path), checkpoint_map
    )

    state = states[problem]
    # Cost still aggregates from inference_result.json
    assert state.overall_usage is not None
    assert state.overall_usage.cost == 0.5
    # Missing evaluation.json -> record_checkpoint_result(None) counts as not-passed
    assert state.total_checkpoints_evaluated == 1
    assert state.checkpoints_passed == 0
    assert state.checkpoints_iso_passed == 0
    assert state.checkpoints_core_solved == 0


def test_prepopulate_empty_completed_list_is_noop(tmp_path: Path) -> None:
    checkpoint_map = {"p": ["checkpoint_1"]}
    states = ProblemStateTracker(["p"], checkpoint_map)

    _prepopulate_completed_states(
        states, [], _make_config(tmp_path), checkpoint_map
    )

    state = states["p"]
    assert state.state == AgentStateEnum.PENDING
    assert state.overall_usage is None
    assert state.total_checkpoints_evaluated == 0
