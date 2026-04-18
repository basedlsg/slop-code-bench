from __future__ import annotations

from rich.console import Console

from slop_code.entrypoints.evaluation.driver import maybe_progress_bar


def test_maybe_progress_bar_initializes_core_percent_field() -> None:
    console = Console(record=True)

    with maybe_progress_bar(
        description="Evaluating checkpoints",
        num_tasks=3,
        enabled=True,
        console=console,
    ) as bar:
        task = bar.progress.tasks[0]
        assert task.fields["core_pct"] == "0.0%"


def test_maybe_progress_bar_updates_core_percent_field() -> None:
    console = Console(record=True)

    with maybe_progress_bar(
        description="Evaluating checkpoints",
        num_tasks=3,
        enabled=True,
        console=console,
    ) as bar:
        bar.update(current=2, core_pct="66.7%")
        task = bar.progress.tasks[0]
        assert task.completed == 2
        assert task.fields["core_pct"] == "66.7%"


def test_disabled_progress_bar_accepts_core_percent_update() -> None:
    console = Console(record=True)

    with maybe_progress_bar(
        description="Evaluating checkpoints",
        num_tasks=1,
        enabled=False,
        console=console,
    ) as bar:
        bar.update(current=1, core_pct="100.0%")
