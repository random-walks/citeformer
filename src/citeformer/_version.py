"""Single source of truth for the citeformer package version.

Hatch reads this at build time (`[tool.hatch.version] path = ...` in pyproject.toml).
The release workflow bumps this value; never hand-edit it outside a `/bump` slash-command
or the `release-bump` skill ceremony.
"""

__version__ = "0.0.1"
