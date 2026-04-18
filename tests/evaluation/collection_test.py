from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from slop_code import evaluation
from slop_code.evaluation.collection import CheckpointTestCollection
from slop_code.evaluation.collection import check_checkpoint_needs_reevaluation
from slop_code.evaluation.collection import check_problem_needs_reevaluation
from slop_code.evaluation.collection import compute_tc_hash
from slop_code.evaluation.collection import compute_tc_hash_from_report
from slop_code.evaluation.collection import load_stored_test_collection_hash


def test_public_collection_functions_exported() -> None:
    assert callable(evaluation.collect_checkpoint_tc)
    assert callable(evaluation.compute_tc_hash)
    assert callable(evaluation.compute_tc_hash_from_report)


def test_compute_tc_hash_stable_for_reordered_input() -> None:
    hash_a = compute_tc_hash(
        {
            "checkpoint_2-Core": ["test_b", "test_a"],
            "checkpoint_2-Error": ["test_e"],
        }
    )
    hash_b = compute_tc_hash(
        {
            "checkpoint_2-Error": ["test_e"],
            "checkpoint_2-Core": ["test_a", "test_b"],
        }
    )

    assert hash_a == hash_b


def test_compute_tc_hash_from_report_grouped_dict() -> None:
    report = {
        "tests": {
            "checkpoint_2-Core": {
                "passed": ["test_a"],
                "failed": ["test_b"],
                "skipped": [],
            },
            "checkpoint_1-Regression": {
                "passed": ["test_old"],
                "failed": [],
                "skipped": [],
            },
        }
    }

    expected = compute_tc_hash(
        {
            "checkpoint_2-Core": ["test_a", "test_b"],
            "checkpoint_1-Regression": ["test_old"],
        }
    )

    assert compute_tc_hash_from_report(report) == expected


def test_compute_tc_hash_from_report_legacy_list() -> None:
    report = {
        "tests": [
            {
                "id": "test_a",
                "checkpoint": "checkpoint_2",
                "group_type": "Core",
            },
            {
                "id": "test_old",
                "checkpoint": "checkpoint_1",
                "group_type": "Regression",
            },
            {
                "id": "test_old",  # duplicate should be de-duplicated
                "checkpoint": "checkpoint_1",
                "group_type": "Regression",
            },
        ]
    }

    expected = compute_tc_hash(
        {
            "checkpoint_2-Core": ["test_a"],
            "checkpoint_1-Regression": ["test_old"],
        }
    )

    assert compute_tc_hash_from_report(report) == expected


def test_compute_tc_hash_from_report_invalid_shape_returns_none() -> None:
    assert compute_tc_hash_from_report({"tests": 123}) is None


def test_load_stored_test_collection_hash_prefers_explicit_hash(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    eval_data = {
        "test_collection_hash": "stored-hash",
        "tests": {
            "checkpoint_1-Core": {
                "passed": ["test_a"],
                "failed": [],
                "skipped": [],
            }
        },
    }
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(eval_data))

    assert load_stored_test_collection_hash(checkpoint_dir) == "stored-hash"


def test_load_stored_test_collection_hash_falls_back_to_report_hash(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    eval_data = {
        "tests": {
            "checkpoint_1-Core": {
                "passed": ["test_a"],
                "failed": ["test_b"],
                "skipped": [],
            }
        }
    }
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(eval_data))

    expected = compute_tc_hash({"checkpoint_1-Core": ["test_a", "test_b"]})
    assert load_stored_test_collection_hash(checkpoint_dir) == expected


def test_check_checkpoint_needs_reevaluation_when_hash_missing(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "evaluation.json").write_text("{}")

    needs_rerun, patched = check_checkpoint_needs_reevaluation(
        checkpoint_dir=checkpoint_dir,
        problem_config=MagicMock(),
        checkpoint_name="checkpoint_1",
        environment=MagicMock(),
    )

    assert needs_rerun is True
    assert patched is False


def test_check_checkpoint_needs_reevaluation_when_snapshot_missing(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    eval_data = {"test_collection_hash": "stored-hash"}
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(eval_data))

    needs_rerun, patched = check_checkpoint_needs_reevaluation(
        checkpoint_dir=checkpoint_dir,
        problem_config=MagicMock(),
        checkpoint_name="checkpoint_1",
        environment=MagicMock(),
    )

    assert needs_rerun is True
    assert patched is False


def test_check_checkpoint_needs_reevaluation_when_checkpoint_missing(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "snapshot").mkdir()
    eval_data = {"test_collection_hash": "stored-hash"}
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(eval_data))

    problem = MagicMock()
    problem.load_checkpoint.side_effect = KeyError("checkpoint_1")

    needs_rerun, patched = check_checkpoint_needs_reevaluation(
        checkpoint_dir=checkpoint_dir,
        problem_config=problem,
        checkpoint_name="checkpoint_1",
        environment=MagicMock(),
    )

    assert needs_rerun is True
    assert patched is False


def test_check_checkpoint_needs_reevaluation_when_collection_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "snapshot").mkdir()
    eval_data = {"test_collection_hash": "stored-hash"}
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(eval_data))

    monkeypatch.setattr(
        "slop_code.evaluation.collection.collect_checkpoint_tc",
        lambda **_: CheckpointTestCollection(
            tests=[],
            by_nodeid={},
            grouped_test_ids={},
            total_collected=0,
            test_collection_hash="current-hash",
            infrastructure_failure=True,
        ),
    )

    needs_rerun, patched = check_checkpoint_needs_reevaluation(
        checkpoint_dir=checkpoint_dir,
        problem_config=MagicMock(load_checkpoint=MagicMock(return_value=MagicMock())),
        checkpoint_name="checkpoint_1",
        environment=MagicMock(),
    )

    assert needs_rerun is True
    assert patched is False


def test_check_checkpoint_needs_reevaluation_when_hashes_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "snapshot").mkdir()
    eval_data = {"test_collection_hash": "stored-hash"}
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(eval_data))

    monkeypatch.setattr(
        "slop_code.evaluation.collection.collect_checkpoint_tc",
        lambda **_: CheckpointTestCollection(
            tests=[],
            by_nodeid={},
            grouped_test_ids={},
            total_collected=0,
            test_collection_hash="stored-hash",
            infrastructure_failure=False,
        ),
    )

    needs_rerun, patched = check_checkpoint_needs_reevaluation(
        checkpoint_dir=checkpoint_dir,
        problem_config=MagicMock(load_checkpoint=MagicMock(return_value=MagicMock())),
        checkpoint_name="checkpoint_1",
        environment=MagicMock(),
    )

    assert needs_rerun is False
    assert patched is False


def test_check_problem_needs_reevaluation_short_circuits_on_first_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "checkpoint_1").mkdir()
    (problem_dir / "checkpoint_2").mkdir()

    called: list[str] = []

    def _fake_checkpoint_check(**kwargs: object) -> tuple[bool, bool]:
        checkpoint_name = str(kwargs["checkpoint_name"])
        called.append(checkpoint_name)
        if checkpoint_name == "checkpoint_1":
            return True, False
        return False, False

    monkeypatch.setattr(
        "slop_code.evaluation.collection.check_checkpoint_needs_reevaluation",
        _fake_checkpoint_check,
    )

    needs_rerun, patched = check_problem_needs_reevaluation(
        problem_dir=problem_dir,
        source_config=MagicMock(),
        environment=MagicMock(),
    )

    assert needs_rerun is True
    assert patched is False
    assert called == ["checkpoint_1"]


def test_check_problem_needs_reevaluation_when_all_checkpoints_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "checkpoint_1").mkdir()
    (problem_dir / "checkpoint_2").mkdir()

    monkeypatch.setattr(
        "slop_code.evaluation.collection.check_checkpoint_needs_reevaluation",
        lambda **_: (False, False),
    )

    needs_rerun, patched = check_problem_needs_reevaluation(
        problem_dir=problem_dir,
        source_config=MagicMock(),
        environment=MagicMock(),
    )

    assert needs_rerun is False
    assert patched is False
