"""Typer entry point for the `citeformer` command.

For P0 this is a minimal shell exposing `--version`. The real command surface
(generate, verify, render) is populated in later phases.
"""

from __future__ import annotations

import typer

from citeformer import __version__

app = typer.Typer(
    name="citeformer",
    help="Generate verifiably cited text from language models.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed citeformer version."""
    typer.echo(f"citeformer {__version__}")


if __name__ == "__main__":
    app()
