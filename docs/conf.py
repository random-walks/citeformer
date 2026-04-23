"""Sphinx configuration for citeformer documentation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

# -- Project information -----------------------------------------------------

project = "citeformer"
author = "Blaise Albis-Burdige"
copyright = "2026, Blaise and citeformer contributors"

try:
    release = _dist_version("citeformer")
except PackageNotFoundError:
    release = "0.0.1"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "myst_parser",
    "autodoc2",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.typer",
    "sphinx_llms_txt",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # autodoc2 generates its own index.rst; we use our hand-written docs/reference/index.md instead.
    "apidocs/index.rst",
]

# -- autodoc2 ---------------------------------------------------------------

autodoc2_packages = [
    {
        "path": "../src/citeformer",
        "module": "citeformer",
    },
]
autodoc2_render_plugin = "myst"
autodoc2_output_dir = "apidocs"
autodoc2_hidden_objects = {"dunder", "private", "inherited"}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"citeformer {version}"
html_theme_options = {
    "source_repository": "https://github.com/random-walks/citeformer",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": False,
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# -- sphinx-copybutton -------------------------------------------------------

copybutton_prompt_text = r"^\$ |^> "
copybutton_prompt_is_regexp = True

# -- sphinx-llms-txt ---------------------------------------------------------

llms_txt_title = "citeformer"
llms_txt_summary = (
    "Generate verifiably cited text from language models. Citation markers are "
    "structurally impossible to fabricate when a grammar-level logit-enforcing "
    "backend (HF transformers + XGrammar/llguidance, vLLM, llama.cpp) is used. "
    "Reference lists are rendered deterministically by citeproc-py against any "
    "CSL style — the model never touches the bibliography."
)
# Curated llms.txt excludes auto-generated API dump; llms-full.txt keeps it.
llms_txt_exclude = ["apidocs/**"]

# -- Nitpicky ----------------------------------------------------------------

nitpicky = False  # Flip to True when the API is fully documented.
