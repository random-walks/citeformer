"""Typer entry point for the `citeformer` command.

Exposes four subcommands that map directly to the library's public surface:

- ``version`` — print the installed version.
- ``styles`` — list the bundled CSL styles.
- ``render`` — render a CSL-JSON item file into a bibliography entry in the
  requested style.
- ``fetch`` — resolve a DOI, arXiv id, or URL to CSL-JSON on stdout.

The ``generate`` / ``verify`` subcommands would require loading a model and
are intentionally omitted from the CLI — at model sizes where generation is
useful, command-line invocation is awkward. Use the Python API for that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from citeformer import Source, __version__
from citeformer.render import render_single_reference
from citeformer.render.formatters import available_formatters

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


@app.command()
def styles() -> None:
    """List the bundled citation styles.

    Any name listed here (or its documented aliases — see
    ``citeformer.render.formatters._REGISTRY``) can be passed as the
    ``style=`` argument to ``Citeformer()`` or ``render_references()``.
    """
    for name in available_formatters():
        typer.echo(name)


@app.command()
def render(
    csl_json: Annotated[
        Path,
        typer.Argument(
            help="Path to a CSL-JSON file — either a single item or a JSON array.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    style: Annotated[
        str,
        typer.Option("--style", "-s", help="Bundled style name (run `citeformer styles`)."),
    ] = "apa-7",
) -> None:
    """Render CSL-JSON items as bibliography entries in the chosen style.

    Useful as a sanity check before feeding sources to the model — preview
    what each reference will look like in the final output.
    """
    items = _load_csl_items(csl_json)
    for position, item in enumerate(items, start=1):
        source = Source(metadata=item, content="")
        reference = render_single_reference(source, style_name=style, number=position)
        typer.echo(reference.rendered)


@app.command()
def fetch(
    identifier: Annotated[
        str,
        typer.Argument(help="DOI (10.xxx/yyy), arXiv id (2305.14627), URL, or PDF path."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write CSL-JSON to this file instead of stdout.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
    include_content: Annotated[
        bool,
        typer.Option(
            "--include-content/--metadata-only",
            help="For URL / PDF inputs, include the extracted body text under `content`.",
        ),
    ] = False,
) -> None:
    """Resolve an identifier to CSL-JSON via the configured metadata adapter.

    Dispatches based on the identifier shape: anything starting with ``10.``
    or containing ``doi.org/`` goes to Crossref; a ``.pdf`` path hits the
    local PDF extractor; anything starting with ``http`` goes through the URL
    extractor; everything else is treated as an arXiv id.
    """
    metadata, body = _resolve_identifier(identifier)
    if include_content and body:
        metadata = {**metadata, "content": body}
    payload = json.dumps(metadata, indent=2, sort_keys=True)
    if output is None:
        typer.echo(payload)
    else:
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"wrote {output}", err=True)


def _load_csl_items(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    raise typer.BadParameter(
        f"Expected a CSL-JSON object or array at {path}, got {type(raw).__name__}"
    )


def _resolve_identifier(identifier: str) -> tuple[dict[str, object], str]:
    """Dispatch an identifier to the right metadata adapter.

    Returns ``(metadata, body)``. ``body`` is empty for DOI / arXiv (those
    adapters only fetch metadata); populated for URL + PDF where the
    adapter extracts the full document text.
    """
    from citeformer.metadata import (  # imported lazily — httpx isn't free on cold start
        extract_pdf,
        extract_url,
        fetch_arxiv,
        fetch_crossref,
    )

    lowered = identifier.strip()
    if lowered.startswith(("http://", "https://")):
        metadata, body = extract_url(lowered)
        return dict(metadata), body
    if lowered.lower().endswith(".pdf") and Path(lowered).exists():
        metadata, body = extract_pdf(Path(lowered))
        return dict(metadata), body
    if lowered.lower().startswith("10.") or "doi.org/" in lowered.lower():
        return dict(fetch_crossref(lowered)), ""
    # Bare arXiv shape: "2305.14627" or "cs.CL/0601024". Accept anything with a slash
    # or the NNNN.NNNNN[NN] pattern.
    return dict(fetch_arxiv(lowered)), ""


if __name__ == "__main__":
    app()
