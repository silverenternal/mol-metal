#!/usr/bin/env python3
"""The claim ledger must point at artifacts that exist and resolve.

A ledger entry is a promise: "this number comes from here".  A typo in a
pointer breaks that promise silently, because nothing in the build reads the
markdown.  This test reads every ``<file>.json:<pointer>`` citation out of
``docs/CLAIM_LEDGER.md`` and resolves it with the same resolver the tables use.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from molmetal.scripts.paper_metric_tables import METRICS_DIR, resolve

LEDGER = Path(__file__).resolve().parents[2] / "docs" / "CLAIM_LEDGER.md"

CITATION = re.compile(r"([A-Za-z0-9_]+\.json):(/[^\s`;,)]+)")


def _citations() -> list[tuple[str, str]]:
    text = LEDGER.read_text(encoding="utf-8")
    return sorted(set(CITATION.findall(text)))


def test_the_ledger_exists_and_cites_something():
    assert LEDGER.is_file()
    assert len(_citations()) >= 10


@pytest.mark.parametrize("file,pointer", _citations())
def test_every_citation_resolves(file: str, pointer: str):
    path = METRICS_DIR / file
    assert path.is_file(), f"{file} does not exist under metrics/by_metric"
    import json

    doc = json.loads(path.read_text())
    resolve(doc, pointer)
