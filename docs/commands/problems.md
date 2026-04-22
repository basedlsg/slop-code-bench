---
version: 1.0
last_updated: 2026-04-22
---

# problems

Commands for inspecting problems and checking conversion status.

The CLI reads problems from the managed catalog under `SCBENCH_HOME/problems`
(default: `~/.cache/scbench/problems`). Install or update catalog data with
`slop-code sync`.

If `SCBENCH_PROBLEMS_PATH` is set, these commands read problems from that
flat local directory instead of the managed release catalog.

## Subcommands

| Command | Description |
|---------|-------------|
| [`ls`](#ls) | List all problems |
| [`status`](#status) | Check problem conversion status |

---

## ls

List all problems with their metadata.

### Usage

```bash
slop-code problems ls
```

### Output

Displays a table with:
- Problem name
- Checkpoint count
- Difficulty
- Description (truncated)

### Examples

```bash
# List all problems
slop-code problems ls
```

---

## status

Check if a problem has been converted to the new pytest-based format.

### Usage

```bash
slop-code problems status PROBLEM_NAME
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PROBLEM_NAME` | Yes | Name of the problem to check |

### Output

**Structure Checks:**
- `tests/` directory exists
- `tests/conftest.py` exists
- No `loader.py` (old format)
- No `verifier.py` (old format)

**Checkpoint Files** (for each checkpoint):
- Test file exists (`tests/test_checkpoint_N.py`)
- Solution directory exists (`solutions/checkpoint_N/`)
- Spec file exists (`checkpoint_N.md`)

**Test Counts** (if fully converted):
A table showing test counts by type (Core, Functionality, Error) for each checkpoint.

### Examples

```bash
# Check if file_backup is converted
slop-code problems status file_backup

# Check conversion status of etl_pipeline
slop-code problems status etl_pipeline
```

### Example Output

```
Problem: file_backup
Description: Implement a file backup utility...
Checkpoints: 5

Structure Checks:
  tests/ directory: OK
  tests/conftest.py: OK
  No loader.py (old format): OK
  No verifier.py (old format): OK

Checkpoint Files:
  checkpoint_1:
    test file: OK
    solution dir: OK
    spec file: OK
  ...

Problem is fully converted to new format.

┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Checkpoint   ┃ Core ┃ Functionality ┃ Error ┃ Total ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ checkpoint_1 │    5 │             3 │     2 │    10 │
│ checkpoint_2 │    8 │             4 │     3 │    15 │
└──────────────┴──────┴───────────────┴───────┴───────┘
```

## See Also

- [Problem Tutorial](../problems/tutorial.md)
- [Problem Configuration](../problems/config-schema.md)
- [Pytest Test Patterns](../problems/pytest/README.md)
