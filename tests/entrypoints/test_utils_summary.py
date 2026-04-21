from __future__ import annotations

import json

import pytest
from rich.console import Console

from slop_code.common import SUMMARY_FILENAME
from slop_code.entrypoints.utils import display_and_save_summary


def test_display_and_save_summary_uses_new_composite_formulas(tmp_path):
    results_file = tmp_path / "checkpoint_results.jsonl"
    rows = [
        {
            "problem": "prob1",
            "idx": 1,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
            "verbosity": 0.6,
            "erosion": 0.6,
        },
        {
            "problem": "prob1",
            "idx": 2,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
            "verbosity": 0.3,
            "erosion": 0.4,
        },
    ]
    results_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    config = {
        "model": {"name": "test-model"},
        "thinking": "none",
        "prompt_path": "test_prompt.jinja",
        "agent": {"type": "test-agent", "version": "v1"},
    }
    console = Console(record=True)

    summary = display_and_save_summary(results_file, tmp_path, config, console)

    assert summary is not None
    expected_first = 0.6
    expected_second = 0.3
    assert summary.verbosity.mean == pytest.approx(
        (expected_first + expected_second) / 2
    )
    assert summary.erosion.mean == pytest.approx(0.5)
    assert summary.erosion.count == 2

    saved = json.loads((tmp_path / SUMMARY_FILENAME).read_text())
    assert saved["verbosity"]["mean"] == summary.verbosity.mean
    assert saved["erosion"]["mean"] == summary.erosion.mean

    rendered = console.export_text()
    assert "Mean verbosity score" in rendered
    assert "Mean erosion score" in rendered


def test_display_and_save_summary_uses_configured_checkpoint_total(tmp_path):
    results_file = tmp_path / "checkpoint_results.jsonl"
    rows = [
        {
            "problem": "prob1",
            "checkpoint": "checkpoint_1",
            "idx": 1,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
        },
        {
            "problem": "prob2",
            "checkpoint": "checkpoint_1",
            "idx": 1,
            "strict_pass_rate": 1.0,
            "isolated_pass_rate": 1.0,
        },
    ]
    results_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    (tmp_path / "prob1").mkdir()
    (tmp_path / "prob1" / "problem.yaml").write_text(
        "name: prob1\n"
        "version: 1\n"
        "checkpoints:\n"
        "  - checkpoint_1\n"
        "  - checkpoint_2\n"
    )
    (tmp_path / "prob2").mkdir()
    (tmp_path / "prob2" / "problem.yaml").write_text(
        "name: prob2\nversion: 1\ncheckpoints:\n  - checkpoint_1\n"
    )

    config = {
        "model": {"name": "test-model"},
        "thinking": "none",
        "prompt_path": "test_prompt.jinja",
        "agent": {"type": "test-agent", "version": "v1"},
    }
    console = Console(record=True)

    summary = display_and_save_summary(results_file, tmp_path, config, console)

    assert summary is not None
    assert summary.num_checkpoints == 3
    assert summary.pct_checkpoints_solved == pytest.approx((2 / 3) * 100)
