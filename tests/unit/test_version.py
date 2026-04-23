"""P0 smoke test — proves the package imports and the version wiring is correct."""

from __future__ import annotations

import re

import citeformer


def test_version_is_pep440_compliant() -> None:
    assert re.match(r"^\d+\.\d+\.\d+([a-z]+\d*)?$", citeformer.__version__)


def test_version_exposed_on_package() -> None:
    assert hasattr(citeformer, "__version__")
    assert isinstance(citeformer.__version__, str)
    assert len(citeformer.__version__) > 0
