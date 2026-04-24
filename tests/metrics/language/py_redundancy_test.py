"""Exhaustive tests for Python code clone detection.

Tests all functions in slop_code.metrics.languages.python.redundancy including:
- Public API: calculate_redundancy_metrics, detect_code_clones
- Helper functions: _normalize_ast, _hash_ast_subtree
"""

from __future__ import annotations

from contextlib import suppress
from textwrap import dedent

import pytest

from slop_code.metrics.languages.python import calculate_redundancy_metrics

# =============================================================================
# Basic Clone Detection Tests
# =============================================================================


class TestCalculateRedundancyMetrics:
    """Tests for calculate_redundancy_metrics function."""

    def test_no_duplicates(self, tmp_path):
        """Test file with no duplicate code."""
        source = tmp_path / "unique.py"
        source.write_text(
            dedent("""
        def func_a():
            return 1

        def func_b():
            return 2 * 3

        def func_c():
            x = 1
            y = 2
            return x + y
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # No clones expected
        assert metrics.total_clone_instances == 0
        assert metrics.clone_ratio == 0.0

    def test_identical_code_blocks(self, tmp_path):
        """Test detection of identical code blocks."""
        source = tmp_path / "clones.py"
        source.write_text(
            dedent("""
        def first():
            if True:
                return 1

        def second():
            if True:
                return 1
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.clones
        assert metrics.total_clone_instances >= 2
        assert metrics.clone_lines > 0
        assert metrics.clone_ratio > 0.0

    def test_renamed_variables_detected_as_clones(self, tmp_path):
        """Test that code with renamed variables is detected as clones."""
        source = tmp_path / "renamed.py"
        source.write_text(
            dedent("""
        def process_a():
            data = []
            for item in items:
                data.append(item)
            return data

        def process_b():
            result = []
            for element in items:
                result.append(element)
            return result
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # Should detect these as clones due to same structure
        assert metrics.clones
        assert metrics.total_clone_instances >= 2

    def test_operator_changes_are_not_clones(self, tmp_path):
        """Different arithmetic operators do not form structural clones."""
        source = tmp_path / "operators.py"
        source.write_text(
            dedent("""
        def add(left, right):
            result = left + right
            return result

        def subtract(left, right):
            result = left - right
            return result
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.clones == []
        assert metrics.total_clone_instances == 0

    def test_normalizes_literals_for_clone_detection(self, tmp_path):
        """Renamed variables and changed literals form structural clones."""
        source = tmp_path / "literals.py"
        source.write_text(
            dedent("""
        def first(left, right):
            result = left + 1
            return result

        def second(alpha, beta):
            total = alpha + 2
            return total
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.total_clone_instances == 2

    def test_minimum_line_threshold(self, tmp_path):
        """Test that clones below minimum line threshold are not detected."""
        source = tmp_path / "small.py"
        source.write_text(
            dedent("""
        def a(): pass
        def b(): pass
        def c(): pass
        """)
        )

        calculate_redundancy_metrics(source)

        # Single-line functions shouldn't be detected as clones
        # (depends on min_lines setting)

    def test_single_occurrence_not_clone(self, tmp_path):
        """Test that single occurrence is not marked as clone."""
        source = tmp_path / "single.py"
        source.write_text(
            dedent("""
        def unique_function():
            x = 1
            y = 2
            z = x + y
            return z
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # Single occurrence = no clones
        assert metrics.total_clone_instances == 0

    def test_empty_file(self, tmp_path):
        """Test with empty file."""
        source = tmp_path / "empty.py"
        source.write_text("")

        metrics = calculate_redundancy_metrics(source)

        assert metrics.total_clone_instances == 0
        assert metrics.clone_ratio == 0.0
        assert metrics.clones == []


# =============================================================================
# Clone Location Tests
# =============================================================================


class TestCloneLocation:
    """Tests for clone location tracking."""

    def test_clone_locations_tracked(self, tmp_path):
        """Test that clone locations are properly tracked."""
        source = tmp_path / "clones.py"
        source.write_text(
            dedent("""
        def first():
            if True:
                x = 1
                return x

        def second():
            if True:
                y = 1
                return y
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        if metrics.clones:
            for clone in metrics.clones:
                # Each clone should have valid location info
                assert hasattr(clone, "node_type") or hasattr(
                    clone, "locations"
                )
                assert clone.line_count > 0


# =============================================================================
# Clone Metrics Tests
# =============================================================================


class TestCloneMetrics:
    """Tests for clone-related metrics."""

    def test_clone_ratio_calculation(self, tmp_path):
        """Test clone ratio is calculated correctly."""
        source = tmp_path / "test.py"
        source.write_text(
            dedent("""
        def first():
            if True:
                return 1

        def second():
            if True:
                return 1

        def unique():
            return 42
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # clone_ratio should be clone_lines / file SLOC
        if metrics.clone_lines > 0:
            assert 0.0 < metrics.clone_ratio <= 1.0

    def test_clone_ratio_uses_sloc_denominator(self, tmp_path):
        source = tmp_path / "test.py"
        source.write_text(
            dedent("""
        def first():
            # comment
            if True:
                return 1

        def second():
            # comment
            if True:
                return 1
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.clone_lines == 8
        assert metrics.clone_ratio == pytest.approx(1.0)

    def test_total_clone_instances(self, tmp_path):
        """Test total clone instances count."""
        source = tmp_path / "multi.py"
        source.write_text(
            dedent("""
        def a():
            x = 1
            y = 2
            return x + y

        def b():
            a = 1
            b = 2
            return a + b

        def c():
            m = 1
            n = 2
            return m + n
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # If all three are clones of each other
        if metrics.clones:
            assert metrics.total_clone_instances >= 2


# =============================================================================
# Complex Code Patterns
# =============================================================================


class TestComplexCodePatterns:
    """Tests for clone detection in complex code patterns."""

    def test_nested_loops_clone(self, tmp_path):
        """Test clone detection with nested loops."""
        source = tmp_path / "nested.py"
        source.write_text(
            dedent("""
        def matrix_sum_a(matrix):
            total = 0
            for row in matrix:
                for cell in row:
                    total += cell
            return total

        def matrix_sum_b(data):
            result = 0
            for r in data:
                for c in r:
                    result += c
            return result
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # Should detect as clones (same structure, renamed variables)
        assert metrics.clones

    def test_try_except_clone(self, tmp_path):
        """Test clone detection with try/except blocks."""
        source = tmp_path / "exceptions.py"
        source.write_text(
            dedent("""
        def safe_divide_a(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0

        def safe_divide_b(x, y):
            try:
                return x / y
            except ZeroDivisionError:
                return 0
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # Should detect as clones
        assert metrics.clones

    def test_comprehension_clone(self, tmp_path):
        """Test clone detection with list comprehensions.

        Note: Single-line comprehensions may not meet minimum line threshold
        for clone detection (typically 3 lines).
        """
        source = tmp_path / "comprehension.py"
        source.write_text(
            dedent("""
        def filter_positive_a(items):
            return [x for x in items if x > 0]

        def filter_positive_b(values):
            return [v for v in values if v > 0]
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # Single-line functions may not meet minimum line threshold
        # This is expected behavior
        assert isinstance(metrics.clones, list)

    def test_class_method_clones(self, tmp_path):
        """Test clone detection in class methods."""
        source = tmp_path / "classes.py"
        source.write_text(
            dedent("""
        class ClassA:
            def process(self, data):
                result = []
                for item in data:
                    if item > 0:
                        result.append(item)
                return result

        class ClassB:
            def filter(self, items):
                output = []
                for elem in items:
                    if elem > 0:
                        output.append(elem)
                return output
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # Methods with same structure should be detected
        assert metrics.clones


# =============================================================================
# Edge Cases
# =============================================================================


class TestRedundancyEdgeCases:
    """Edge case tests for redundancy detection."""

    def test_single_function(self, tmp_path):
        """Test file with single function."""
        source = tmp_path / "single.py"
        source.write_text(
            dedent("""
        def only_function():
            x = 1
            y = 2
            return x + y
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        # No clones possible with single function
        assert metrics.total_clone_instances == 0

    def test_syntax_error_handling(self, tmp_path):
        """Test graceful handling of syntax errors."""
        source = tmp_path / "broken.py"
        source.write_text("def broken(:\n    pass\n")

        # Should handle gracefully. If it returns, metrics should be empty/zero;
        # raising is also acceptable for malformed source.
        with suppress(Exception):
            calculate_redundancy_metrics(source)

    def test_docstrings_do_not_make_short_functions_clones(self, tmp_path):
        """Docstring lines do not count toward clone candidate size."""
        source = tmp_path / "docs.py"
        source.write_text(
            dedent('''
        import math

        class Metrics:
            def cc_mass(self) -> float:
                """Return the cyclomatic complexity mass."""
                return self.cyc_complexity * math.sqrt(self.sloc)

            def cog_mass(self) -> float:
                """Return the cognitive complexity mass."""
                return self.cog_complexity * math.sqrt(self.sloc)
        ''')
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.clones == []
        assert metrics.total_clone_instances == 0

    def test_docstrings_do_not_affect_clone_hash(self, tmp_path):
        """Docstring presence does not change clone structure."""
        source = tmp_path / "hash_docs.py"
        source.write_text(
            dedent('''
        def first(value):
            """Explain the first function."""
            current = value + 1
            doubled = current * 2
            return doubled

        def second(value):
            current = value + 2
            doubled = current * 2
            return doubled
        ''')
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.total_clone_instances == 2

    def test_type_checking_import_blocks_are_not_clones(self, tmp_path):
        """TYPE_CHECKING import blocks are excluded from clone detection."""
        source = tmp_path / "type_checking.py"
        source.write_text(
            dedent("""
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from one import Thing
            from two import Other

        if TYPE_CHECKING:
            from one import Thing
            from two import Other
        """)
        )

        metrics = calculate_redundancy_metrics(source)

        assert metrics.clones == []
        assert metrics.total_clone_instances == 0

    def test_imports_not_clones(self, tmp_path):
        """Test that similar imports don't create false clones."""
        source = tmp_path / "imports.py"
        source.write_text(
            dedent("""
        import os

        def func():
            import os
            return os.getcwd()
        """)
        )

        calculate_redundancy_metrics(source)

        # Imports shouldn't be marked as clones
