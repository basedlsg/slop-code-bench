# Problem Review Checklist

Concrete checklist for auditing a problem's specs, config, and tests against
established conventions. Use this after writing a problem and before submitting
for review.

> **See also:** [Design Checklist](checklist.md) for higher-level design
> validation, [Tutorial](../problems/tutorial.md) for authoring guidance.

---

## config.yaml

- [ ] `version` present (currently `1` or `2`)
- [ ] `name` is short snake_case (not a repo URL or long identifier)
- [ ] `description` present — multi-line, 2-4 sentences covering the full arc
- [ ] `category` present (e.g., `cli-tools`, `networking`, `file-systems`, `developer-tools`)
- [ ] `difficulty` present (`Easy`, `Medium`, or `Hard`)
- [ ] `author` present
- [ ] `entry_file` is a short, meaningful name — not `solution` or `main`
- [ ] `timeout` is a reasonable integer for the problem's workload
- [ ] `tags` present — 5-10 descriptive lowercase hyphenated tags
- [ ] Checkpoints have sequential `order` starting at 1
- [ ] All checkpoints have `state: Core Tests` (or `Draft` for WIP)
- [ ] `include_prior_tests` defaults to `true` — only set `false` with clear justification
- [ ] `test_dependencies` lists packages needed by the test suite (not by the solution)
- [ ] `static_assets` entries (if any) have paths that exist on disk

---

## Checkpoint Specs (checkpoint_N.md)

### Structure

- [ ] Opening paragraph is clear, human-friendly, and describes what this checkpoint adds
- [ ] Each checkpoint adds to prior ones — no removal of previously working features
- [ ] "Previously defined behavior is unchanged unless stated here" (or equivalent) appears in CP2+
- [ ] Out of Scope section lists what is deferred to later checkpoints

### Entrypoint

- [ ] `%%%ENTRYPOINT:entry_file%%%` used for all command invocations in code blocks
- [ ] `%%%ENTRYPOINT:entry_file%%%` used for inline code showing commands (e.g., `` `%%%ENTRYPOINT:entry_file%%% ingest` ``)
- [ ] No bare tool name in code blocks — never `mytool export ...`, always the placeholder
- [ ] Prose references to the tool concept (not commands) are OK without placeholder

### Interface

- [ ] CLI flags documented in a table with: flag name, required/optional, type, default, description
- [ ] Data structures and schemas shown in tables or code blocks
- [ ] Output format exactly specified: encoding, line endings, delimiters, header row, column order
- [ ] Determinism requirements called out: sorting, ordering, tie-breaking rules

### Error Handling

- [ ] Error conditions listed with exit codes
- [ ] Error behavior described as "Exit N, error to STDERR" — **not** exact STDERR strings
- [ ] Error examples say "Error message written to STDERR" — **not** prescribed message text
- [ ] Validation ordering specified when it matters (e.g., "resolution validated before provider capability")

### Examples

- [ ] Examples cover common happy-path cases
- [ ] Examples cover at least one edge case (empty input, boundary values, deduplication)
- [ ] Examples cover at least one error case
- [ ] Examples show expected STDOUT, file contents, or "no output" as appropriate
- [ ] Error examples do **not** show exact STDERR text

---

## Language Agnosticity

- [ ] `entry_file` does not assume a language (no `.py`, `.js` extension)
- [ ] Specs describe observable CLI behavior — inputs, outputs, exit codes, file effects
- [ ] No references to language-specific constructs (`package`, `module`, `class`, `import`)
- [ ] No requirement to expose internal API, constants, or programmatic interfaces
- [ ] No prescribed internal architecture, module decomposition, or design patterns
- [ ] No "Design Pressure" paragraphs that hint at how to structure the solution

---

## Leakage

- [ ] Spec describes **what**, not **how** — no algorithm names, library suggestions, or patterns
- [ ] No paragraphs that hint at internal decomposition (e.g., "rewards clean separation between X, Y, Z")
- [ ] No hints about testability or mockability (e.g., "backend should be substitutable")
- [ ] Helper function names, class names, and file layout are not prescribed
- [ ] When the spec mentions a concept (e.g., "cloud storage"), it defines the observable interface, not the implementation strategy

---

## Ambiguity

- [ ] For every observable behavior, could two correct implementations produce different output? If yes, tighten the spec.
- [ ] Validation scope is explicit: which inputs are validated and when (e.g., "ineligible files are not validated for missing keys")
- [ ] Edge cases at boundaries are explicitly called out (e.g., "a date equal to the cutoff is treated as X")
- [ ] When behavior differs between contexts (e.g., different ordering rules), the difference is explicitly documented with rationale or made consistent

---

## Tests (when present)

### conftest.py

- [ ] Defines `pytest_addoption()` with `--entrypoint` and `--checkpoint`
- [ ] Defines `entrypoint_argv` and `checkpoint_name` session-scoped fixtures
- [ ] Custom markers registered in `pytest_configure()` if used

### Test Files

- [ ] Named `test_checkpoint_N.py` matching checkpoint numbering
- [ ] CORE tests are unmarked (no decorator)
- [ ] Optional/nice-to-have tests marked with `@pytest.mark.functionality`
- [ ] Error-handling tests marked with `@pytest.mark.error`
- [ ] Tests use `entrypoint_argv` fixture for subprocess invocation

### Test Data

- [ ] Organized in `data/checkpoint_N/{core,hidden,errors}/`
- [ ] Each case directory has a `case.yaml` and expected output file
- [ ] `{{static:files}}` placeholder used correctly for asset references
- [ ] Edge cases covered: empty input, large input, boundary values, special characters
