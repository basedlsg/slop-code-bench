---
version: 1.0
last_updated: 2026-04-22
---

# sync

Install or update the managed SCBench problem catalog.

## Usage

```bash
slop-code sync [VERSION]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `VERSION` | No | Release tag to install (for example `v0.2`). If omitted, installs the latest release. |

## Behavior

- Catalog data is installed under `SCBENCH_HOME/problems`.
- If `SCBENCH_HOME` is unset, the default home is `~/.cache/scbench`.
- If `SCBENCH_PROBLEMS_PATH` is set, problem-loading commands use that local
  flat directory instead of the managed catalog.
- `manifest.json` stores the installed catalog `version` and `commit`.
- `slop-code sync` is the only update mechanism after initial bootstrap.
- If the requested version is already installed, the command no-ops.

## Examples

```bash
# Install latest release
slop-code sync

# Install a specific release
slop-code sync v0.2
```
