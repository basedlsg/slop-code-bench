"""AST-grep based pattern detection metrics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import TypedDict

import yaml

from slop_code.logging import get_logger
from slop_code.metrics.models import AstGrepMetrics
from slop_code.metrics.models import AstGrepViolation

logger = get_logger(__name__)


class _RuleInfo(TypedDict, total=False):
    category: str
    subcategory: str
    weight: int
    min_file_count: int


RuleLookup = dict[str, _RuleInfo]

AST_GREP_CATEGORY = "slop"
_EMPTY_RULE_INFO: _RuleInfo = {}

# Default rules file (relative to project root).
AST_GREP_RULES_PATH = Path(__file__).parents[5] / "configs" / "slop_rules.yaml"
# Back-compat export; this now points at the single rules file.
AST_GREP_RULES_DIR = AST_GREP_RULES_PATH


def _is_sg_available() -> bool:
    """Check if ast-grep (sg) is available on the system."""
    return shutil.which("sg") is not None


def _get_ast_grep_rules_path() -> Path:
    """Get the AST-grep rules file, with env var override."""
    override = os.environ.get("AST_GREP_RULES_PATH")
    if override:
        return Path(override)
    override = os.environ.get("AST_GREP_RULES_DIR")
    if override:
        return Path(override)
    return AST_GREP_RULES_PATH


def _get_ast_grep_rules_dir() -> Path:
    """Back-compat wrapper returning the configured rules file path."""
    return _get_ast_grep_rules_path()


def _count_rules_in_file(rule_file: Path) -> int:
    """Return the number of AST-grep rules in a (possibly multi-doc) file."""
    try:
        with rule_file.open(encoding="utf-8") as handle:
            return sum(1 for doc in yaml.safe_load_all(handle) if doc)
    except (OSError, yaml.YAMLError) as exc:
        logger.debug(
            "Failed to read AST-grep rule file",
            rule_file=str(rule_file),
            error=str(exc),
        )
        return 0


def calculate_ast_grep_metrics(source: Path) -> AstGrepMetrics:
    """Calculate AST-grep metrics using ast-grep rules.

    Runs ast-grep scan with rules from the configured slop rules file
    and parses the JSON output to produce AstGrepMetrics.

    Args:
        source: Path to the Python source file.

    Returns:
        AstGrepMetrics with violations found, or empty metrics if sg
        unavailable.
    """
    if not _is_sg_available():
        logger.debug("ast-grep (sg) not available, skipping ast-grep metrics")
        return AstGrepMetrics(
            violations=[], total_violations=0, counts={}, rules_checked=0
        )

    rules_path = _get_ast_grep_rules_path()
    if not rules_path.exists():
        logger.warning(
            "AST-grep rules file not found",
            rules_path=str(rules_path),
        )
        return AstGrepMetrics(
            violations=[], total_violations=0, counts={}, rules_checked=0
        )

    rules_checked = _count_rules_in_file(rules_path)

    if rules_checked == 0:
        return AstGrepMetrics(
            violations=[], total_violations=0, counts={}, rules_checked=0
        )

    # Build lookup to get correct category/subcategory/weight from rule files
    # (ast-grep's JSON output doesn't include rule metadata)
    rules_lookup = build_ast_grep_rules_lookup()

    raw_violations: list[AstGrepViolation] = []
    raw_counts: Counter[str] = Counter()
    logger.debug("Running ast-grep rules", rules_path=str(rules_path))
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "sg",
                "scan",
                "--json=stream",
                "-r",
                str(rules_path),
                str(source.absolute()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        logger.debug(
            "Failed to run ast-grep rules",
            rules_path=str(rules_path),
            error=str(e),
        )
        return AstGrepMetrics(
            violations=[],
            total_violations=0,
            counts={},
            rules_checked=rules_checked,
        )
    if result.returncode != 0:
        logger.warning(
            "Failed to run ast-grep rules",
            rules_path=str(rules_path),
            error=result.stderr,
        )
        return AstGrepMetrics(
            violations=[],
            total_violations=0,
            counts={},
            rules_checked=rules_checked,
        )
    # Parse JSON stream output (one JSON object per line)
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            match = json.loads(line)
            rule_id = match.get("ruleId", AST_GREP_CATEGORY)
            rule_info = rules_lookup.get(rule_id, _EMPTY_RULE_INFO)
            violation = AstGrepViolation(
                rule_id=rule_id,
                severity=match.get("severity", "warning"),
                category=rule_info.get("category", AST_GREP_CATEGORY),
                subcategory=rule_info.get("subcategory", AST_GREP_CATEGORY),
                weight=rule_info.get("weight", 1),
                line=match["range"]["start"]["line"],
                column=match["range"]["start"]["column"],
                end_line=match["range"]["end"]["line"],
                end_column=match["range"]["end"]["column"],
            )
            raw_violations.append(violation)
            raw_counts[violation.rule_id] += 1
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Failed to parse ast-grep output",
                line=line,
                error=str(e),
            )
            continue

    violations = [
        violation
        for violation in raw_violations
        if _exceeds_min_file_count(
            raw_counts[violation.rule_id],
            rules_lookup.get(violation.rule_id, _EMPTY_RULE_INFO),
        )
    ]
    counts = Counter(violation.rule_id for violation in violations)

    return AstGrepMetrics(
        violations=violations,
        total_violations=len(violations),
        counts=dict(counts),
        rules_checked=rules_checked,
    )


def _exceeds_min_file_count(count: int, rule_info: _RuleInfo) -> bool:
    """Return whether a rule's file-local matches clear its threshold."""
    min_file_count = rule_info.get("min_file_count")
    if not isinstance(min_file_count, int):
        return True
    return count > min_file_count


def build_ast_grep_rules_lookup() -> RuleLookup:
    """Build lookup table mapping rule_id to category/subcategory/weight.

    Parses the configured slop rules file and extracts metadata for each rule.
    This is used for backfilling ast_grep.jsonl files with correct metadata.

    Returns:
        Dict mapping rule_id to metadata including category, subcategory,
        weight, and optional min_file_count threshold.
    """
    rules_path = _get_ast_grep_rules_path()
    if not rules_path.exists():
        logger.warning(
            "AST-grep rules file not found",
            rules_path=str(rules_path),
        )
        return {}

    lookup: RuleLookup = {}
    try:
        with rules_path.open(encoding="utf-8") as handle:
            for doc in yaml.safe_load_all(handle):
                if not doc:
                    continue
                rule_id = doc.get("id")
                if not rule_id:
                    continue
                metadata = doc.get("metadata", {})
                lookup[rule_id] = {
                    "category": AST_GREP_CATEGORY,
                    "subcategory": metadata.get("category", AST_GREP_CATEGORY),
                    "weight": metadata.get("weight", 1),
                }
                min_file_count = metadata.get("min_file_count")
                if isinstance(min_file_count, int):
                    lookup[rule_id]["min_file_count"] = min_file_count
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Failed to parse AST-grep rules file for lookup",
            rules_path=str(rules_path),
            error=str(exc),
        )
        return {}

    logger.debug(
        "Built AST-grep rules lookup",
        rules_count=len(lookup),
    )
    return lookup
