---
name: add-citation-format
description: Add a new CSL citation style as a CitationFormatter subclass in src/citeformer/render/formatters/. Covers the template, required test matrix, and registration steps.
---

# Adding a new citation format

A citeformer style is a `CitationFormatter` subclass that turns a CSL-JSON
item (our `Source.metadata` shape) into an inline marker + bibliography
entry. The goal of this skill is to land a new style — e.g. APSA, ACS,
AMA, a journal-specific one — with the same quality bar as the six
built-ins (APA, MLA, Chicago, IEEE, Nature, Vancouver).

## When to use this

- A user asks for a new citation style that isn't in `bundled_style_names()`.
- You're adapting citeformer to a house / journal / school style that
  isn't covered by the six built-ins.
- A bundled style has drifted out of spec and needs a rewrite.

Skip this if the user just wants a one-off citation they can copy-paste
manually — the formatters are for repeated use.

## Steps

### 1. Research the style

Find an authoritative source: the style's own manual (CMOS, MLA Handbook,
APA Publication Manual, ICMJE), a university library guide, or the
official `.csl` file at https://github.com/citation-style-language/styles.
Write down:

- The `citation_format` classification (author-date / author / numeric / note / label).
- The inline marker shape (examples: `(Smith, 2023)`, `[1]`, `(Smith 45)`).
- The bibliography entry shape for **at least these five CSL types**:
  - `book`
  - `article-journal`
  - `chapter`
  - `thesis`
  - `paper-conference`
- The author-name format (`Last, First` / `F. M. Last` / `Last FF`) and
  the `et al.` threshold for the bibliography and for inline cites.
- Page-range conventions: en-dash (`45–67`), hyphen, or `pp. ` prefix.

Three short examples of each type from the style manual, copied verbatim,
are the ground truth for your tests.

### 2. Copy the closest existing formatter

Pick the most similar built-in as a starting point:

- Numeric scientific style → clone `nature.py` or `ieee.py`.
- Author-date social-sciences style → clone `apa.py` or `chicago.py`.
- Author-only / MLA-style → clone `mla.py`.
- Biomedical numeric (NLM / ICMJE-adjacent) → clone `vancouver.py`.

```bash
cp src/citeformer/render/formatters/nature.py \
   src/citeformer/render/formatters/<new-style>.py
```

### 3. Adapt the formatter

Rename the class (`<Name>Formatter`), set the `name` and `citation_format`
class attributes, and rewrite the `inline()` + `_article`, `_book`,
`_chapter`, `_thesis`, `_paper_conference`, `_webpage` methods. Keep
bodies direct — `chunks: list[str] = []` + conditional appends + `" "
.join(...)` is the pattern. Reach for helpers in `_base.py` first:

- `parse_authors(raw)` / `parse_year(issued)` / `get_str(item, key)` /
  `get_title(item)` — robust CSL-JSON accessors.
- `format_page_range(pages, dash="–")` — hyphen/en-dash/em-dash handling.
- `format_doi(doi)` — `10.1/x` → `https://doi.org/10.1/x`.
- `ensure_period(text)` — append a `.` only if one isn't already there
  (critical for author lists that end with initials).

If the style needs a new helper, add it to `_base.py` first (shared), not
inline in the formatter.

### 4. Register the formatter

Edit `src/citeformer/render/formatters/__init__.py`:

1. Import the new class.
2. Add an entry to `_REGISTRY` for each name/alias the style uses (e.g.
   `"apsa"` and `"apsa-2018"` both pointing at `APSAFormatter`).
3. Append the canonical name to `_CANONICAL`.
4. Append the class name to `__all__`.

### 5. Test

Add a test file `tests/unit/test_formatters.py` entries (not a new file —
append to the existing one). Follow the pattern:

```python
def test_<style>_inline_marker_shape() -> None:
    f = get_formatter("<style>")
    assert f.inline(_book(), 1) == "(Expected, 2023)"  # or "[1]" etc.

def test_<style>_bib_starts_correctly() -> None:
    f = get_formatter("<style>")
    out = f.bibliography(_article(), 1)
    assert out.startswith("Expected prefix")
    assert "Journal of X" in out
```

Also add a snapshot in `tests/unit/test_render_csl.py`:

```python
def test_render_snapshot_<style>(data_regression) -> None:
    refs = render_references(_canonical_sources(), _full_citations(), "<style>")
    data_regression.check(_refs_dict(refs))
```

Run the snapshot twice to create + lock it:

```bash
uv run pytest tests/unit/test_render_csl.py -k <style>  # creates the .yml
uv run pytest tests/unit/test_render_csl.py -k <style>  # verifies match
```

Eyeball the generated YAML against the ground-truth examples you wrote
down in step 1. If the output is wrong, fix the formatter *before*
committing the snapshot.

### 6. Document

- `docs/decisions/003-bundle-five-csl-styles.md` — if you're growing the
  bundled set, update the canonical list here. Any addition is an
  additive change (minor bump); a replacement (dropping a bundled style)
  is a §10-adjacent break.
- `CHANGELOG.md` — add an `[Unreleased]` entry under "Added" describing
  the new style and pointing at the skill's invocation.
- `README.md` — bump the "bundled styles" mention if it's hard-coded.

### 7. Run the full suite

```bash
make lint && make test && make docs-build
```

The `test_every_style_renders_canonical_types_without_error` parametrized
test automatically covers the new style once it's registered — make sure
it passes without errors for every fixture type.

## Red flags

- **Double periods** in output (`Smith, E. A..`) — use `ensure_period()`
  on any chunk that might already end in `.`. This bit Nature in the
  initial rewrite.
- **Missing fallbacks**: a formatter that raises on missing `author` or
  `year` is broken. APA uses `"Anon"` and `"n.d."`; other styles should
  follow a similar convention.
- **Silent wrong output**: if `get_str(item, "foo")` returns None for a
  required field, the formatter should still produce *something*. Skip
  the section, don't inject an empty string that screws up the spacing.
- **Adding a style-specific CSL file**: don't. Our value is in the Python
  formatters, not in hand-rolling CSL XML. If a style genuinely needs
  features we don't have, the user can install `citeformer[citeproc-compat]`
  and provide their own `.csl` file.
