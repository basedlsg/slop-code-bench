from __future__ import annotations

from slop_code import evaluation
from slop_code.evaluation.collection import compute_tc_hash


def test_public_collection_functions_exported() -> None:
    assert callable(evaluation.collect_checkpoint_tc)
    assert callable(evaluation.compute_tc_hash)


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
