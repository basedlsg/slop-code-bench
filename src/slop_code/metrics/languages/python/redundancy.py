"""Code clone detection via AST hashing.

Clones are measured in lines.  When clone groups overlap (e.g. a cloned
function contains a cloned if-block), lines are deduplicated via a
line-number union so no line is counted twice.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from token import COMMENT
from token import DEDENT
from token import ENDMARKER
from token import INDENT
from token import NEWLINE
from token import NL
from tokenize import TokenError
from tokenize import generate_tokens
from typing import TYPE_CHECKING

from slop_code.metrics.languages.python.constants import CLONE_NODE_TYPES
from slop_code.metrics.languages.python.line_metrics import (
    calculate_line_metrics,
)
from slop_code.metrics.languages.python.parser import get_python_parser
from slop_code.metrics.languages.python.utils import read_python_code
from slop_code.metrics.models import CodeClone
from slop_code.metrics.models import RedundancyMetrics

if TYPE_CHECKING:
    from tree_sitter import Node


IGNORED_SLOC_TOKEN_TYPES = {
    COMMENT,
    DEDENT,
    ENDMARKER,
    INDENT,
    NEWLINE,
    NL,
}

_LITERAL_TOKENS = {
    "string": "$STR",
    "string_content": "$STR",
    "f_string": "$STR",
    "string_fragment": "$STR",
    "bytes": "$STR",
    "integer": "$INT",
    "float": "$FLOAT",
    "imaginary": "$FLOAT",
    "true": "$BOOL",
    "false": "$BOOL",
    "none": "$NONE",
}


def _normalize_ast(node: Node) -> str:
    """Create normalized string representation of AST subtree."""
    variable_map: dict[str, str] = {}
    variable_counter = 0

    def normalize(current: Node) -> str:
        nonlocal variable_counter
        if current.type == "identifier":
            if current.text is None:
                return "$VAR0"
            name = current.text.decode("utf-8")
            if name not in variable_map:
                variable_counter += 1
                variable_map[name] = f"$VAR{variable_counter}"
            return variable_map[name]

        if current.type in _LITERAL_TOKENS:
            return _LITERAL_TOKENS[current.type]

        children = tuple(
            child
            for child in current.children
            if child.type != "comment" and not _is_plain_string_statement(child)
        )
        if not children:
            return current.type

        child_parts = [normalize(child) for child in children]
        return f"{current.type}({','.join(child_parts)})"

    return normalize(node)


def _hash_ast_subtree(node: Node) -> str:
    """Generate hash of normalized AST subtree."""
    normalized = _normalize_ast(node)
    return hashlib.md5(
        normalized.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]


def _is_type_checking_block(node: Node) -> bool:
    condition = node.child_by_field_name("condition")
    text = condition.text if condition is not None else None
    return text in {b"TYPE_CHECKING", b"typing.TYPE_CHECKING"}


def _is_plain_string_statement(node: Node) -> bool:
    if node.type == "string":
        expression = node
    elif node.type == "expression_statement" and len(node.named_children) == 1:
        expression = node.named_children[0]
    else:
        return False

    if expression.type != "string":
        return False

    text = expression.text
    if text is None:
        return False

    prefix = _string_prefix(text.decode("utf-8"))
    return "b" not in prefix and "f" not in prefix


def _string_prefix(literal: str) -> str:
    prefix_chars: list[str] = []
    for character in literal:
        if character in {'"', "'"}:
            break
        prefix_chars.append(character.lower())
    return "".join(prefix_chars)


def _sloc_line_numbers(source: str, root: Node) -> frozenset[int]:
    source_lines = source.splitlines(keepends=True)
    text_lines = source.splitlines()
    lines: set[int] = set()
    try:
        for token in generate_tokens(iter(source_lines).__next__):
            if token.type not in IGNORED_SLOC_TOKEN_TYPES:
                lines.add(token.start[0])
    except TokenError:
        return frozenset(lines)

    for start, end in _plain_string_statement_ranges(root, text_lines):
        for line_number in range(start, end + 1):
            lines.discard(line_number)
    return frozenset(lines)


def _plain_string_statement_ranges(
    root: Node,
    source_lines: list[str],
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if _is_plain_string_statement(node):
            literal = node if node.type == "string" else node.named_children[0]
            if _owns_line(literal, source_lines):
                ranges.append(
                    (literal.start_point[0] + 1, literal.end_point[0] + 1),
                )
        stack.extend(node.named_children)
    return tuple(ranges)


def _owns_line(node: Node, source_lines: list[str]) -> bool:
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    start_line = (
        source_lines[start_row] if start_row < len(source_lines) else ""
    )
    end_line = source_lines[end_row] if end_row < len(source_lines) else ""
    return not start_line[:start_col].strip() and not end_line[end_col:].strip()


def detect_code_clones(source: Path, min_lines: int = 3) -> RedundancyMetrics:
    """Detect duplicate code blocks via AST hashing."""
    code = read_python_code(source)
    if not code.strip():
        return RedundancyMetrics(
            clones=[], total_clone_instances=0, clone_lines=0, clone_ratio=0.0
        )

    parser = get_python_parser()
    tree = parser.parse(code.encode("utf-8"))
    source_lines = code.splitlines()
    sloc_lines = _sloc_line_numbers(code, tree.root_node)

    groups: dict[str, list[tuple[Node, str, int]]] = {}
    stack = [tree.root_node]

    while stack:
        current = stack.pop()
        if current.type in CLONE_NODE_TYPES and not _is_type_checking_block(
            current
        ):
            start_line = current.start_point[0] + 1
            end_line = current.end_point[0] + 1
            sloc_count = sum(
                1 for line in sloc_lines if start_line <= line <= end_line
            )
            if sloc_count >= min_lines:
                ast_hash = _hash_ast_subtree(current)
                line_count = end_line - start_line + 1
                groups.setdefault(ast_hash, []).append(
                    (current, current.type, line_count)
                )
        stack.extend(current.children)

    clones: list[CodeClone] = []
    total_instances = 0
    clone_line_set: set[int] = set()
    for ast_hash, nodes in groups.items():
        if len(nodes) < 2:
            continue
        locations = [
            (n.start_point[0] + 1, n.end_point[0] + 1) for n, _, _ in nodes
        ]
        node_type = nodes[0][1]
        line_count = nodes[0][2]
        clones.append(
            CodeClone(
                ast_hash=ast_hash,
                locations=locations,
                node_type=node_type,
                line_count=line_count,
            )
        )
        total_instances += len(nodes)
        for n, _, _ in nodes:
            clone_line_set.update(range(n.start_point[0], n.end_point[0] + 1))

    total_lines = calculate_line_metrics(source).loc
    num_clone_lines = len(clone_line_set)
    clone_sloc_lines = sum(
        1
        for line_number in clone_line_set
        if line_number < len(source_lines)
        and (stripped := source_lines[line_number].strip())
        and not stripped.startswith("#")
    )
    clone_ratio = (clone_sloc_lines / total_lines) if total_lines else 0.0

    return RedundancyMetrics(
        clones=clones,
        total_clone_instances=total_instances,
        clone_lines=num_clone_lines,
        clone_ratio=clone_ratio,
    )


def calculate_redundancy_metrics(source: Path) -> RedundancyMetrics:
    """Calculate redundancy metrics for a Python file."""
    return detect_code_clones(source)
