from __future__ import annotations

from typing import Annotated

import typer

from slop_code import problem_catalog


def register(app: typer.Typer, name: str) -> None:
    app.command(
        name,
        help="Install or update the managed SCBench problem catalog.",
    )(sync_catalog)


def sync_catalog(
    ctx: typer.Context,
    version: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Optional release tag (for example: v0.2). "
                "If omitted, installs the latest release."
            )
        ),
    ] = None,
) -> None:
    home = ctx.obj.scbench_home
    before = problem_catalog.load_manifest(home)
    manifest = problem_catalog.sync_catalog(home, version)
    catalog_root = problem_catalog.get_catalog_root(home)

    if before is not None and before == manifest:
        typer.echo(f"Catalog version {manifest.version} is already current.")
        return

    typer.echo(f"Installed catalog version: {manifest.version}")
    typer.echo(f"Resolved commit: {manifest.commit}")
    typer.echo(f"Catalog path: {catalog_root}")
