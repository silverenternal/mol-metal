"""Top-level pytest configuration for the ``molmetal`` package.

R17 / C3 — keep ``.md`` documents (final.md, status.md, etc.) and other
non-test artefacts out of pytest's collection.  Pytest by default only
collects ``test_*.py`` / ``*_test.py`` files, but a handful of legacy
configurations (``--collect-any-file``, custom python_files globs, IDE
plugins) widen the net.  This conftest adds belt-and-braces defence at
two layers:

  1. ``collect_ignore_glob`` — drop any ``*.md`` from collection so it is
     never even enumerated as a candidate test.
  2. ``pytest_collection_modifyitems`` — if a ``.md`` somehow slips
     through (e.g. user-defined collector), mark it with a skip reason so
     the test runner emits a single, identifiable skip rather than a
     confusing "collected but not run" warning.

The ``reports/`` sub-tree is also added to ``collect_ignore`` to keep the
research write-ups (``molmetal/reports/wf_*/final.md`` etc.) from leaking
into the test discovery scan; the directory contains hundreds of
Markdown reports from past workflow runs that should never be treated as
test files.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# Directories that contain only documentation / research write-ups and
# should never be scanned for tests.  These live alongside the source
# tree to keep workflow outputs co-located with the code that produced
# them — convenient for humans, but a footgun for pytest.
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"
_DOCS_DIR = Path(__file__).resolve().parent / "docs"


collect_ignore = [
    str(_REPORTS_DIR),
    str(_DOCS_DIR),
]
collect_ignore_glob = [
    "**/*.md",
    "**/*.rst",
    "**/*.txt",
    "**/*.ipynb",
]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Defence-in-depth — mark any .md / non-Python items so pytest skips them.

    Pytest's default collector skips non-``test_*.py`` files, but a
    custom ``--collect-any-file`` invocation or a stray hook can pull
    Markdown files into the collected-items list.  Rather than fail with
    a cryptic "could not import test module" traceback, we annotate each
    offending item with a clear skip reason so the test report stays
    self-documenting.

    Additionally applies the ``slow`` marker to every item whose module is
    listed in ``molmetal/tests/_slow_modules.py`` so ``-m "not slow"`` is a
    real fast subset without hand-editing hundreds of files.
    """
    try:
        from molmetal.tests._slow_modules import SLOW_MODULE_STEMS
    except Exception:  # pragma: no cover - list is optional
        SLOW_MODULE_STEMS = frozenset()
    for item in items:
        path = str(getattr(item, "fspath", "") or getattr(item, "location", ""))
        if path.endswith((".md", ".rst", ".txt", ".ipynb")):
            item.add_marker(
                pytest.mark.skip(reason=f"non-test artefact: {Path(path).name}")
            )
        mod = getattr(item, "module", None)
        if getattr(mod, "__name__", "").rsplit(".", 1)[-1] in SLOW_MODULE_STEMS:
            item.add_marker(pytest.mark.slow)