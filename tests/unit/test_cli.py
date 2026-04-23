"""Smoke tests for the ``citeformer`` CLI.

The CLI is a thin wrapper over the library's public surface. These tests
exercise the command plumbing (typer wiring, argument parsing, output shape)
without touching the network: the ``fetch`` subcommand calls live APIs and
is exercised in ``tests/unit/test_metadata_fetchers.py`` via VCR cassettes
at the library level.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from citeformer import __version__
from citeformer.cli.app import app
from citeformer.render.formatters import available_formatters

runner = CliRunner()


def test_version_prints_installed_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_styles_lists_all_bundled_formatters() -> None:
    result = runner.invoke(app, ["styles"])
    assert result.exit_code == 0
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines == available_formatters()


def test_render_from_csl_json_file(tmp_path: Path) -> None:
    csl_item = {
        "id": "poe-raven",
        "type": "book",
        "title": "The Raven",
        "author": [{"family": "Poe", "given": "Edgar Allan"}],
        "issued": {"date-parts": [[1845]]},
        "publisher": "Putnam",
    }
    path = tmp_path / "one.json"
    path.write_text(json.dumps(csl_item), encoding="utf-8")

    result = runner.invoke(app, ["render", str(path), "--style", "apa-7"])

    assert result.exit_code == 0, result.stdout
    assert "Poe" in result.stdout
    assert "1845" in result.stdout
    assert "The Raven" in result.stdout


def test_render_handles_json_array(tmp_path: Path) -> None:
    items = [
        {
            "id": "poe",
            "type": "book",
            "title": "The Raven",
            "author": [{"family": "Poe", "given": "Edgar Allan"}],
            "issued": {"date-parts": [[1845]]},
        },
        {
            "id": "melville",
            "type": "book",
            "title": "Moby-Dick",
            "author": [{"family": "Melville", "given": "Herman"}],
            "issued": {"date-parts": [[1851]]},
        },
    ]
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(items), encoding="utf-8")

    result = runner.invoke(app, ["render", str(path), "--style", "ieee"])

    assert result.exit_code == 0, result.stdout
    # IEEE is numeric — both entries should appear in order.
    assert "Poe" in result.stdout
    assert "Melville" in result.stdout


def test_render_rejects_unknown_style(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    path.write_text(json.dumps({"id": "x", "type": "book", "title": "x"}), encoding="utf-8")

    result = runner.invoke(app, ["render", str(path), "--style", "not-a-style"])

    assert result.exit_code != 0


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # typer's `no_args_is_help=True` + Click's behaviour exits 2 while still
    # printing help; either 0 or 2 is acceptable here — we care about the help
    # payload, not the exit status.
    assert result.exit_code in (0, 2)
    assert "version" in result.stdout
    assert "styles" in result.stdout
    assert "render" in result.stdout
    assert "fetch" in result.stdout
