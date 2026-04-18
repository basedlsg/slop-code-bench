from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console

from slop_code import evaluation
from slop_code.common import CHECKPOINT_CONFIG_NAME
from slop_code.common import CHECKPOINT_RESULTS_FILENAME
from slop_code.common import CONFIG_FILENAME
from slop_code.common import PROBLEM_CONFIG_NAME
from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.common import serialize_path_dict
from slop_code.entrypoints import evaluation as evaluation_entry
from slop_code.entrypoints.commands import common
from slop_code.entrypoints.config import loader as config_loader
from slop_code.entrypoints.evaluation.metrics import update_results_jsonl
from slop_code.entrypoints.utils import display_and_save_summary
from slop_code.evaluation import EVALUATION_SCHEMA_VERSION
from slop_code.evaluation import ProblemConfig
from slop_code.evaluation import get_available_problems
from slop_code.evaluation.collection import check_problem_needs_reevaluation
from slop_code.logging import get_logger

logger = get_logger(__name__)


REQUIRED_EVAL_FIELDS = [
    "problem_name",
    "problem_version",
    "checkpoint_name",
    "checkpoint_version",
    "tests",
    "pass_counts",
    "total_counts",
    "pytest_exit_code",
    "pytest_collected",
    "infrastructure_failure",
]

REQUIRED_EVAL_DIR_FILES = ["stdout.txt", "stderr.txt", "report.json"]


def _is_evaluation_schema_current(checkpoint_dir: Path) -> bool:
    """Check if evaluation.json has current schema version and evaluation/ dir is valid."""
    eval_path = checkpoint_dir / "evaluation.json"
    eval_dir = checkpoint_dir / "evaluation"

    if not eval_path.exists():
        return False

    try:
        with eval_path.open() as f:
            data = json.load(f)

        # Check schema version
        schema_version = data.get("schema_version")
        if schema_version is None or schema_version < EVALUATION_SCHEMA_VERSION:
            return False

        # Check required fields exist
        if not all(field in data for field in REQUIRED_EVAL_FIELDS):
            return False

        # Check evaluation/ directory has required files
        if not eval_dir.exists() or not eval_dir.is_dir():
            return False

        return all((eval_dir / f).exists() for f in REQUIRED_EVAL_DIR_FILES)
    except (OSError, json.JSONDecodeError):
        return False


def _can_upgrade_evaluation_schema(checkpoint_dir: Path) -> bool:
    """Check if evaluation.json can be upgraded by just adding schema_version.

    Returns True if evaluation.json exists with all required fields but lacks
    schema_version (or has an older version), and the evaluation/ directory
    has all required files.
    """
    eval_path = checkpoint_dir / "evaluation.json"
    eval_dir = checkpoint_dir / "evaluation"

    if not eval_path.exists():
        return False

    try:
        with eval_path.open() as f:
            data = json.load(f)

        # Check required fields exist (excluding schema_version)
        if not all(field in data for field in REQUIRED_EVAL_FIELDS):
            return False

        # Check evaluation/ directory has required files
        if not eval_dir.exists() or not eval_dir.is_dir():
            return False

        if not all((eval_dir / f).exists() for f in REQUIRED_EVAL_DIR_FILES):
            return False

        # Upgradeable if schema_version is missing or outdated
        schema_version = data.get("schema_version")
        return (
            schema_version is None or schema_version < EVALUATION_SCHEMA_VERSION
        )
    except (OSError, json.JSONDecodeError):
        return False


def _upgrade_evaluation_schema(checkpoint_dir: Path) -> bool:
    """Add or update schema_version in evaluation.json.

    Returns True if successful, False otherwise.
    """
    eval_path = checkpoint_dir / "evaluation.json"

    try:
        with eval_path.open() as f:
            data = json.load(f)

        data["schema_version"] = EVALUATION_SCHEMA_VERSION

        with eval_path.open("w") as f:
            json.dump(data, f, indent=2)

        return True
    except (OSError, json.JSONDecodeError):
        return False


def _try_upgrade_problem_evaluations(problem_dir: Path) -> bool:
    """Try to upgrade all checkpoint evaluations in a problem directory.

    Returns True if all checkpoints were successfully upgraded, False otherwise.
    """
    checkpoint_dirs = [
        d
        for d in problem_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_")
    ]
    if not checkpoint_dirs:
        return False

    all_upgraded = True
    for checkpoint_dir in checkpoint_dirs:
        if _can_upgrade_evaluation_schema(checkpoint_dir):
            if not _upgrade_evaluation_schema(checkpoint_dir):
                all_upgraded = False
        elif not _is_evaluation_schema_current(checkpoint_dir):
            # Not upgradeable and not current - can't upgrade this problem
            all_upgraded = False

    return all_upgraded


def _is_problem_fully_evaluated(problem_dir: Path) -> bool:
    """Check if all checkpoints have valid, current-schema evaluation.json."""
    checkpoint_dirs = sorted(
        d
        for d in problem_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_")
    )
    if not checkpoint_dirs:
        return False
    return all(
        (d / "evaluation.json").exists() and _is_evaluation_schema_current(d)
        for d in checkpoint_dirs
    )


def _has_outdated_evaluation_schema(problem_dir: Path) -> bool:
    """Check if problem has evaluation.json files that are outdated.

    Returns True if at least one checkpoint has evaluation.json but with
    outdated schema (missing version or older version number).
    """
    checkpoint_dirs = [
        d
        for d in problem_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_")
    ]
    if not checkpoint_dirs:
        return False

    for checkpoint_dir in checkpoint_dirs:
        eval_path = checkpoint_dir / "evaluation.json"
        if not eval_path.exists():
            continue

        # Has evaluation.json but check if schema is current
        if not _is_evaluation_schema_current(checkpoint_dir):
            return True

    return False


def _extract_pytest_marker_name(node: ast.expr) -> str | None:
    """Extract pytest.mark.X name from a decorator AST node.

    Handles both bare @pytest.mark.X and called @pytest.mark.X(...) forms.
    """
    if isinstance(node, ast.Call):
        return _extract_pytest_marker_name(node.func)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    ):
        return node.attr
    return None


def _parse_test_markers(test_file: Path) -> dict[str, list[str]] | None:
    """AST-parse a test file and return {function_name -> [marker_names]}.

    Only considers top-level test_* functions and test_* methods inside
    Test* classes (with class-level markers inherited). Filters out
    'parametrize' since it is not a group marker.

    Returns None on parse or IO failure.
    """
    try:
        source = test_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(test_file))
    except (SyntaxError, OSError):
        logger.warning(
            "Failed to parse test file for marker extraction",
            test_file=str(test_file),
        )
        return None

    result: dict[str, list[str]] = {}

    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test_"):
            markers = []
            for dec in node.decorator_list:
                m = _extract_pytest_marker_name(dec)
                if m is not None and m != "parametrize":
                    markers.append(m)
            result[node.name] = markers
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_markers = []
            for dec in node.decorator_list:
                m = _extract_pytest_marker_name(dec)
                if m is not None and m != "parametrize":
                    class_markers.append(m)
            for class_node in node.body:
                if isinstance(
                    class_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and class_node.name.startswith("test_"):
                    method_markers = []
                    for dec in class_node.decorator_list:
                        m = _extract_pytest_marker_name(dec)
                        if m is not None and m != "parametrize":
                            method_markers.append(m)
                    result[class_node.name] = class_markers + method_markers

    return result


def _classify_test_group(
    function_name: str,
    markers: list[str],
    *,
    is_prior_checkpoint: bool,
    custom_markers: dict[str, str],
) -> str:
    """Return the GroupType value string for a test function.

    Mirrors the priority order of PytestRunner._determine_group_type.
    custom_markers maps marker name -> GroupType value string.
    """
    if is_prior_checkpoint:
        return "Regression"
    if "error" in markers:
        return "Error"
    if "regression" in markers:
        return "Regression"
    for marker in markers:
        if marker in custom_markers:
            return custom_markers[marker]
    if "functionality" in markers:
        return "Functionality"
    return "Core"


_KNOWN_GROUP_VALUES = {"Core", "Functionality", "Regression", "Error"}


def _collect_expected_test_groups(
    problem_path: Path,
    checkpoint_dir: Path,
    problem_config: ProblemConfig,
) -> dict[str, str] | None:
    """Return {test_function_name -> expected_group_value} for a checkpoint.

    Derives which checkpoint test files to check from the group keys already
    present in evaluation.json, so include_prior_tests=False is handled
    automatically. Returns None if any required file cannot be parsed.
    """
    eval_path = checkpoint_dir / "evaluation.json"
    try:
        with eval_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    current_checkpoint_name: str = data.get("checkpoint_name", "")
    tests: dict = data.get("tests", {})

    referenced_checkpoints: set[str] = set()
    for group_key in tests:
        checkpoint_part, _, group_part = group_key.rpartition("-")
        if group_part in _KNOWN_GROUP_VALUES and checkpoint_part:
            referenced_checkpoints.add(checkpoint_part)

    custom_markers: dict[str, str] = {
        name: config.group for name, config in problem_config.markers.items()
    }

    # Process prior checkpoints first so current checkpoint wins on name conflicts
    ordered = sorted(
        referenced_checkpoints,
        key=lambda cp: (cp == current_checkpoint_name, cp),
    )

    result: dict[str, str] = {}
    for cp_name in ordered:
        test_file = problem_path / "tests" / f"test_{cp_name}.py"
        if not test_file.exists():
            logger.warning(
                "Test file not found for checkpoint",
                test_file=str(test_file),
                checkpoint_name=cp_name,
            )
            return None
        parsed = _parse_test_markers(test_file)
        if parsed is None:
            return None
        is_prior = cp_name != current_checkpoint_name
        for func_name, markers in parsed.items():
            result[func_name] = _classify_test_group(
                func_name,
                markers,
                is_prior_checkpoint=is_prior,
                custom_markers=custom_markers,
            )

    return result


def _find_test_discrepancies(
    checkpoint_dir: Path,
    expected_groups: dict[str, str],
) -> tuple[bool, dict[str, tuple[str, str]]]:
    """Compare evaluation.json against expected test groups.

    Returns (has_missing_tests, reclassifications) where reclassifications
    maps test_id -> (current_group_value, expected_group_value).

    A test is "missing" if its base function name (stripping parametrize
    brackets) does not appear in any test ID in evaluation.json.
    A test is "misclassified" if it is present but under the wrong group.
    """
    eval_path = checkpoint_dir / "evaluation.json"
    try:
        with eval_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, {}

    tests: dict = data.get("tests", {})
    buckets = ("passed", "failed", "skipped")

    eval_test_groups: dict[str, str] = {}
    for group_key, group_data in tests.items():
        group_value = (
            group_key.rsplit("-", 1)[-1] if "-" in group_key else group_key
        )
        for bucket in buckets:
            for test_id in group_data.get(bucket, []):
                eval_test_groups[test_id] = group_value

    eval_base_names = {test_id.split("[")[0] for test_id in eval_test_groups}

    has_missing = any(func not in eval_base_names for func in expected_groups)

    reclassifications: dict[str, tuple[str, str]] = {}
    for test_id, current_group in eval_test_groups.items():
        base_name = test_id.split("[")[0]
        if base_name in expected_groups:
            expected_group = expected_groups[base_name]
            if current_group != expected_group:
                reclassifications[test_id] = (current_group, expected_group)

    return has_missing, reclassifications


def _patch_evaluation_reclassify(
    checkpoint_dir: Path,
    reclassifications: dict[str, tuple[str, str]],
    checkpoint_name: str,
) -> bool:
    """Move tests between groups in evaluation.json and recompute counts.

    Returns True on success, False on any IO or data error.
    """
    eval_path = checkpoint_dir / "evaluation.json"
    try:
        with eval_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    tests: dict = data.get("tests", {})
    buckets = ("passed", "failed", "skipped")

    for test_id, (
        current_group_value,
        new_group_value,
    ) in reclassifications.items():
        source_key = None
        source_bucket = None
        for group_key, group_data in tests.items():
            if group_key.rsplit("-", 1)[-1] != current_group_value:
                continue
            for bucket in buckets:
                if test_id in group_data.get(bucket, []):
                    source_key = group_key
                    source_bucket = bucket
                    break
            if source_key:
                break

        if source_key is None:
            continue

        dest_key = f"{checkpoint_name}-{new_group_value}"
        if dest_key not in tests:
            tests[dest_key] = {"passed": [], "failed": [], "skipped": []}

        tests[source_key][source_bucket].remove(test_id)
        tests[dest_key][source_bucket].append(test_id)

    pass_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for group_key, group_data in tests.items():
        group_value = (
            group_key.rsplit("-", 1)[-1] if "-" in group_key else group_key
        )
        passed = len(group_data.get("passed", []))
        failed = len(group_data.get("failed", []))
        skipped = len(group_data.get("skipped", []))
        pass_counts[group_value] = pass_counts.get(group_value, 0) + passed
        total_counts[group_value] = (
            total_counts.get(group_value, 0) + passed + failed + skipped
        )

    data["pass_counts"] = pass_counts
    data["total_counts"] = total_counts

    try:
        with eval_path.open("w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def _write_problem_and_checkpoint_configs(
    problem_dir: Path, source_config: ProblemConfig
) -> None:
    """Write evaluation configs into the run directory.

    This makes re-evaluation/resume self-contained and ensures future config
    comparisons and report generation reflect the most recent source configs.
    """
    try:
        problem_payload = serialize_path_dict(
            source_config.model_dump(mode="json")
        )
        with (problem_dir / PROBLEM_CONFIG_NAME).open("w") as f:
            yaml.dump(problem_payload, f, indent=2, sort_keys=True)
    except OSError:
        logger.warning(
            "Failed to write problem.yaml into run directory",
            problem_dir=str(problem_dir),
        )

    for checkpoint_name, checkpoint in source_config.iterate_checkpoint_items():
        checkpoint_dir = problem_dir / checkpoint_name
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            continue
        try:
            checkpoint_payload = serialize_path_dict(
                checkpoint.model_dump(mode="json")
            )
            with (checkpoint_dir / CHECKPOINT_CONFIG_NAME).open("w") as f:
                yaml.dump(checkpoint_payload, f, indent=2, sort_keys=True)
        except OSError:
            logger.warning(
                "Failed to write checkpoint.yaml into run directory",
                problem_dir=str(problem_dir),
                checkpoint_name=checkpoint_name,
            )


def _extract_eval_relevant_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Extract fields from a problem config that affect evaluation."""
    checkpoints_summary = {}
    for name, chkpt in config.get("checkpoints", {}).items():
        if isinstance(chkpt, dict):
            checkpoints_summary[name] = {
                "version": chkpt.get("version"),
                "groups": sorted(chkpt.get("groups", {}).keys()),
            }
    return {
        "version": config.get("version"),
        "adapter": config.get("adapter"),
        "static_assets": config.get("static_assets"),
        "checkpoints": checkpoints_summary,
    }


def _has_evaluation_config_changed(
    problem_dir: Path,
    source_config: ProblemConfig,
) -> bool:
    """Check if evaluation-relevant config fields have changed.

    Compares the saved problem.yaml in the agent run directory against
    the current source problem config. Returns True if fields that affect
    evaluation (adapter, static_assets, version, checkpoint groups) differ.
    """
    saved_config_path = problem_dir / PROBLEM_CONFIG_NAME
    if not saved_config_path.exists():
        # No saved config means we can't compare - treat as changed
        return True

    try:
        with saved_config_path.open("r") as f:
            saved_config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        # If we can't load the saved config, treat as changed
        return True

    saved_fields = _extract_eval_relevant_fields(saved_config)
    source_fields = _extract_eval_relevant_fields(
        source_config.model_dump(mode="json")
    )

    return saved_fields != source_fields


def register(app: typer.Typer, name: str) -> None:
    app.command(
        name,
        help=(
            "Evaluate a directory of agent inference results. Must be in the "
            "format <agent_dir>/<problem>/<checkpoint>/<snapshot>."
        ),
    )(evaluate_agent_run)


def evaluate_agent_run(
    ctx: typer.Context,
    agent_run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the inference directory",
            exists=True,
            dir_okay=True,
            file_okay=False,
        ),
    ],
    problem_names: list[str] = typer.Option(
        [],
        "--problem",
        help="Name of the specific problems to run",
    ),
    pass_policy: evaluation.PassPolicy = typer.Option(
        evaluation.PassPolicy.ALL_CASES,
        "--pass-policy",
        help="Policy to determine if the checkpoint passed",
    ),
    env_config: Path | None = typer.Option(
        None,
        "-e",
        "--env-config",
        help="Path to environment specification configuration",
    ),
    live_progress: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--live-progress/--no-live-progress",
        help="Enable live progress display",
    ),
    num_workers: int = typer.Option(
        1,
        "--num-workers",
        "-proc",
        help="Number of parallel evaluation workers (1 for sequential)",
    ),
    overwrite: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--overwrite",
        help="Re-evaluate problems even if they already have evaluation results",
    ),
) -> None:
    """Evaluate a directory of attempts against a problem specification."""

    if not agent_run_dir.exists():
        typer.echo(
            typer.style(
                f"Submission path '{agent_run_dir}' does not exist.",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1)

    if not (agent_run_dir / "environment.yaml").exists() and env_config is None:
        typer.echo(
            typer.style(
                f"Environment configuration file '{agent_run_dir / 'environment.yaml'}' does not exist.",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1)
    env_path = env_config or (agent_run_dir / "environment.yaml")

    environment = config_loader.resolve_environment(env_path)

    console = Console()
    common.setup_command_logging(
        log_dir=agent_run_dir,
        verbosity=ctx.obj.verbosity,
        log_file_name="evaluation.log",
        add_multiproc_info=num_workers > 1,
        console=console,
    )
    logger = get_logger(__name__)
    logger.info(
        "Evaluating a directory of submissions",
        submission_path=str(agent_run_dir),
        problem_names=problem_names,
        env_config=str(env_path),
        pass_policy=pass_policy.value,
    )

    common.ensure_docker_ready(environment)

    valid_problems = get_available_problems(ctx.obj.problem_path)
    problems_to_eval = []
    # Auto-skip evaluated problems unless overwrite requested
    auto_skip_evaluated = not overwrite
    skipped_count = 0
    for problem_name in problem_names or valid_problems.keys():
        if problem_name not in valid_problems:
            logger.warning(
                "Problem not found in available problems",
                problem_name=problem_name,
            )
            continue
        problem_dir = agent_run_dir / problem_name
        if not problem_dir.exists():
            logger.debug(
                "Problem directory does not exist",
                problem_dir=str(problem_dir),
            )
            continue
        if auto_skip_evaluated:
            # Check if fully evaluated with current schema
            if _is_problem_fully_evaluated(problem_dir):
                source_config = valid_problems[problem_name]
                if _has_evaluation_config_changed(problem_dir, source_config):
                    logger.info(
                        "Re-evaluating due to config change",
                        problem_name=problem_name,
                    )
                else:
                    needs_rerun, any_patched = (
                        check_problem_needs_reevaluation(
                            problem_dir=problem_dir,
                            source_config=source_config,
                            environment=environment,
                        )
                    )
                    if needs_rerun:
                        logger.info(
                            "Re-evaluating due to test collection hash change",
                            problem_name=problem_name,
                        )
                    else:
                        if any_patched:
                            logger.info(
                                "Patched misclassified tests, skipping re-evaluation",
                                problem_name=problem_name,
                            )
                        else:
                            logger.info(
                                "Skipping already-evaluated problem",
                                problem_name=problem_name,
                            )
                        skipped_count += 1
                        continue
            # Try to upgrade schema if evaluation exists but is outdated
            elif _has_outdated_evaluation_schema(problem_dir):
                if _try_upgrade_problem_evaluations(problem_dir):
                    # Successfully upgraded - now check if we can skip
                    source_config = valid_problems[problem_name]
                    if _has_evaluation_config_changed(
                        problem_dir, source_config
                    ):
                        logger.info(
                            "Re-evaluating due to config change",
                            problem_name=problem_name,
                        )
                    else:
                        needs_rerun, any_patched = (
                            check_problem_needs_reevaluation(
                                problem_dir=problem_dir,
                                source_config=source_config,
                                environment=environment,
                            )
                        )
                        if needs_rerun:
                            logger.info(
                                "Re-evaluating due to test collection hash change",
                                problem_name=problem_name,
                            )
                        else:
                            if any_patched:
                                logger.info(
                                    "Patched misclassified tests, skipping re-evaluation",
                                    problem_name=problem_name,
                                )
                            else:
                                logger.info(
                                    "Upgraded evaluation schema and skipping",
                                    problem_name=problem_name,
                                )
                            skipped_count += 1
                            continue
                else:
                    logger.info(
                        "Re-evaluating due to incompatible evaluation schema",
                        problem_name=problem_name,
                    )
        logger.info(
            "Adding problem to evaluation",
            problem_name=problem_name,
            problem_dir=str(problem_dir),
        )
        source_problem = valid_problems[problem_name]
        _write_problem_and_checkpoint_configs(problem_dir, source_problem)
        problems_to_eval.append((source_problem, problem_dir))

    if not problems_to_eval:
        if skipped_count > 0:
            logger.info(
                f"No problems to evaluate ({skipped_count} already evaluated, "
                "use --overwrite to re-evaluate)"
            )
        else:
            logger.error("No problems to evaluate")
        raise typer.Exit(1)
    if skipped_count > 0:
        logger.info(
            f"Evaluating {len(problems_to_eval):,} problems "
            f"({skipped_count} skipped as already evaluated)"
        )
    else:
        logger.info(f"Evaluating {len(problems_to_eval):,} problems")

    _, eval_summary = evaluation_entry.evaluate(
        problems=problems_to_eval,
        environment=environment,
        snapshot_dir_name=SNAPSHOT_DIR_NAME,
        live_progress=live_progress,
        console=console,
        num_workers=num_workers,
    )
    logger.info(
        "Evaluation complete",
        successful=eval_summary.successful,
        failed=eval_summary.failed,
    )

    report_file = agent_run_dir / CHECKPOINT_RESULTS_FILENAME
    report_errors: list[tuple[str, str]] = []
    all_reports: list[dict] = []
    for p_dir in agent_run_dir.iterdir():
        if not p_dir.is_dir():
            continue
        typer.echo(f"Processing problem {p_dir}")
        problem_name = p_dir.name
        try:
            problem = ProblemConfig.from_yaml(
                ctx.obj.problem_path / problem_name
            )
        except FileNotFoundError:
            logger.error(
                "Problem configuration not found during report generation",
                problem_name=problem_name,
            )
            report_errors.append((problem_name, "Problem config not found"))
            continue
        except (
            OSError,
            TypeError,
            ValueError,
            KeyError,
            yaml.YAMLError,
        ) as e:
            logger.error(
                "Error loading problem configuration",
                problem_name=problem_name,
                error=str(e),
            )
            report_errors.append((problem_name, str(e)))
            continue

        reports, errors = evaluation_entry.create_problem_reports(
            p_dir, problem
        )
        all_reports.extend(reports)
        for checkpoint_name, error_msg in errors:
            report_errors.append(
                (f"{problem_name}/{checkpoint_name}", error_msg)
            )

    update_results_jsonl(report_file, all_reports)

    typer.echo(f"Reports written to {report_file}")

    # Display evaluation summary at end
    if eval_summary.failed > 0:
        typer.echo(
            typer.style(
                f"\n{eval_summary.format_summary()}",
                fg=typer.colors.YELLOW,
                bold=True,
            )
        )
    else:
        typer.echo(
            typer.style(
                f"\nAll {eval_summary.total_checkpoints} checkpoints evaluated successfully!",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )

    typer.echo(
        typer.style(
            f"\n{len(report_errors)} error(s) during report generation:",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    )
    for identifier, error_msg in report_errors:
        typer.echo(
            typer.style(f"  - {identifier}: {error_msg}", fg=typer.colors.RED)
        )
    with (agent_run_dir / CONFIG_FILENAME).open("r") as f:
        config = yaml.safe_load(f)
    # Display and save summary statistics
    display_and_save_summary(report_file, agent_run_dir, config, console)
