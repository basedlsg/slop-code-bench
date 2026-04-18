from __future__ import annotations

from pathlib import Path

import yaml

from slop_code.entrypoints.commands import inject_canary


def test_get_injector_supports_javascript() -> None:
    injector = inject_canary._get_injector(Path("app.js"))
    assert injector is not None


def test_get_injector_supports_css_and_html() -> None:
    css_injector = inject_canary._get_injector(Path("styles.css"))
    html_injector = inject_canary._get_injector(Path("index.html"))

    assert css_injector is not None
    assert html_injector is not None


def test_javascript_injection_uses_line_comments_and_is_idempotent() -> None:
    injector = inject_canary._get_injector(Path("app.js"))
    assert injector is not None

    canary = "alpha\nbeta"
    content = "const x = 1;\n"

    injected = injector(content, canary)
    assert injected.startswith("// alpha\n// beta\n")
    assert injected.endswith(content)

    reinjected = injector(injected, canary)
    assert reinjected == injected


def test_javascript_injection_preserves_shebang() -> None:
    injector = inject_canary._get_injector(Path("cli.js"))
    assert injector is not None

    canary = "line one\nline two"
    content = "#!/usr/bin/env node\nconsole.log('ok');\n"

    injected = injector(content, canary)

    expected_prefix = "#!/usr/bin/env node\n// line one\n// line two\n"
    assert injected.startswith(expected_prefix)
    assert injected.endswith("console.log('ok');\n")


def test_html_injection_uses_html_comments_and_is_idempotent() -> None:
    injector = inject_canary._get_injector(Path("index.html"))
    assert injector is not None

    canary = "alpha\nbeta"
    content = "<!doctype html>\n<html></html>\n"

    injected = injector(content, canary)
    assert injected.startswith("<!-- alpha\nbeta -->\n")
    assert injected.endswith(content)

    reinjected = injector(injected, canary)
    assert reinjected == injected


def test_css_injection_uses_block_comments_and_is_idempotent() -> None:
    injector = inject_canary._get_injector(Path("styles.css"))
    assert injector is not None

    canary = "alpha\nbeta"
    content = "body { color: black; }\n"

    injected = injector(content, canary)
    assert injected.startswith("/* alpha\nbeta */\n")
    assert injected.endswith(content)

    reinjected = injector(injected, canary)
    assert reinjected == injected


def test_discover_eligible_files_collects_javascript_files(tmp_path: Path) -> None:
    problems_dir = tmp_path / "problems"
    problem_dir = problems_dir / "demo_problem"
    checkpoint_solution_dir = problem_dir / "solutions" / "checkpoint_1"

    checkpoint_solution_dir.mkdir(parents=True)
    (problem_dir / "config.yaml").write_text(
        yaml.safe_dump({"static_assets": {}}),
        encoding="utf-8",
    )
    js_path = checkpoint_solution_dir / "app.js"
    js_path.write_text("console.log('demo');\n", encoding="utf-8")

    eligible = inject_canary._discover_eligible_files(problems_dir)

    assert "javascript" in eligible
    assert js_path in eligible["javascript"]


def test_discover_eligible_files_collects_css_and_html_files(
    tmp_path: Path,
) -> None:
    problems_dir = tmp_path / "problems"
    problem_dir = problems_dir / "demo_problem"
    checkpoint_solution_dir = problem_dir / "solutions" / "checkpoint_1"

    checkpoint_solution_dir.mkdir(parents=True)
    (problem_dir / "config.yaml").write_text(
        yaml.safe_dump({"static_assets": {}}),
        encoding="utf-8",
    )

    html_path = checkpoint_solution_dir / "index.html"
    html_path.write_text("<html></html>\n", encoding="utf-8")

    css_path = checkpoint_solution_dir / "styles.css"
    css_path.write_text("body {}\n", encoding="utf-8")

    eligible = inject_canary._discover_eligible_files(problems_dir)

    assert "html" in eligible
    assert html_path in eligible["html"]
    assert "css" in eligible
    assert css_path in eligible["css"]


def test_discover_eligible_files_excludes_solution_test_data_subtrees(
    tmp_path: Path,
) -> None:
    problems_dir = tmp_path / "problems"
    problem_dir = problems_dir / "demo_problem"
    solution_dir = problem_dir / "solutions" / "checkpoint_1"

    solution_dir.mkdir(parents=True)
    (problem_dir / "config.yaml").write_text(
        yaml.safe_dump({"static_assets": {}}),
        encoding="utf-8",
    )

    included_js = solution_dir / "app.js"
    included_js.write_text("console.log('demo');\n", encoding="utf-8")

    excluded_js = solution_dir / "tests" / "data" / "sample_repo" / "app.js"
    excluded_js.parent.mkdir(parents=True)
    excluded_js.write_text("console.log('fixture');\n", encoding="utf-8")

    excluded_html = solution_dir / "tests" / "assets" / "ui" / "index.html"
    excluded_html.parent.mkdir(parents=True)
    excluded_html.write_text("<html></html>\n", encoding="utf-8")

    excluded_css = solution_dir / "tests" / "data" / "ui" / "styles.css"
    excluded_css.parent.mkdir(parents=True)
    excluded_css.write_text("body {}\n", encoding="utf-8")

    eligible = inject_canary._discover_eligible_files(problems_dir)

    assert included_js in eligible["javascript"]
    assert excluded_js not in eligible["javascript"]
    assert excluded_html not in eligible["html"]
    assert excluded_css not in eligible["css"]
