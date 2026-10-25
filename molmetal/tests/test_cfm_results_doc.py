#!/usr/bin/env python3
"""Every number in ``docs/CFM_RESULTS.md`` must still match its artifact.

``docs/CFM_RESULTS.md`` is the CFM line's single results document (task R4).
The line has already had a results claim outlive the weights under it
(HONEST_NEGATIVE_CATALOG §6.197: c7's checkpoints were gone while its numbers
were still being quoted), so the document carries its numbers as data, not
prose:

* one fenced block tagged ``json cfm-results``;
* each entry is ``{id, value, artifact, path[, tol]}``;
* this test resolves the dot path against the artifact and asserts the value
  is still there;
* and — the part that makes "every number" true rather than aspirational —
  every decimal number in the surrounding prose must also appear as a value
  in the block.

So there are two ways to break the document: change an artifact without
changing the doc, or write a number in prose that nothing checks. Both are
red tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "CFM_RESULTS.md"

FENCE = re.compile(
    r"^```json\s+cfm-results\s*$(?P<body>.*?)^```$",
    re.MULTILINE | re.DOTALL,
)
#: A number with at least two decimal places.  One decimal is excluded on
#: purpose: section references (``§3.1``) and t-grid points (``t=0.5``) are
#: labels, not measurements, and forcing them into the block would make the
#: block unreadable without making the document any safer.  A ``§`` in front
#: is a section reference and is excluded too.
MEASURED_NUMBER = re.compile(r"(?<![§\w.])-?\d+\.\d{2,}(?![\w.])")
DEFAULT_TOL = 1e-4


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _load_entries() -> list[dict]:
    match = FENCE.search(_text())
    if match is None:
        raise AssertionError(
            "docs/CFM_RESULTS.md has no ```json cfm-results block"
        )
    entries = json.loads(match.group("body"))
    if not isinstance(entries, list) or not entries:
        raise AssertionError("the cfm-results block must be a non-empty list")
    return entries


def _walk(doc, path: str):
    cur = doc
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            raise KeyError(f"{path!r} does not resolve at segment {seg!r}")
    return cur


ENTRIES = _load_entries() if DOC.is_file() else []
PROSE = FENCE.sub("", _text()) if DOC.is_file() else ""

#: The document is task R4 and does not exist yet.  Skipping rather than
#: failing keeps the guard alive without making an unwritten document a red
#: test; the moment the file lands every test below runs.
pytestmark = pytest.mark.skipif(
    not DOC.is_file(), reason="docs/CFM_RESULTS.md not written yet (task R4)"
)


def test_the_block_is_machine_readable():
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids)), f"duplicate ids: {sorted(ids)}"

    for entry in ENTRIES:
        assert {"id", "value", "artifact", "path"} <= set(entry), entry
        assert isinstance(entry["value"], (int, float)), entry
        artifact = REPO_ROOT / entry["artifact"]
        assert artifact.is_file(), f"{entry['id']}: {entry['artifact']} is not on disk"


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["id"] for e in ENTRIES])
def test_indexed_value_matches_the_artifact(entry):
    artifact = REPO_ROOT / entry["artifact"]
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    actual = _walk(doc, entry["path"])
    tol = float(entry.get("tol", DEFAULT_TOL))

    assert isinstance(actual, (int, float)), (
        f"{entry['id']}: {entry['artifact']}:{entry['path']} is "
        f"{type(actual).__name__}, not a number"
    )
    assert abs(float(actual) - float(entry["value"])) <= tol, (
        f"{entry['id']} is stale: the document says {entry['value']}, "
        f"{entry['artifact']}:{entry['path']} says {actual}. "
        "Update docs/CFM_RESULTS.md."
    )


def test_every_artifact_names_the_script_that_wrote_it():
    """An artifact nobody can regenerate is an anecdote, not a measurement."""
    for entry in ENTRIES:
        doc = json.loads((REPO_ROOT / entry["artifact"]).read_text(encoding="utf-8"))
        provenance = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        declared = provenance.get("script") or doc.get("script") or doc.get("source")
        assert declared, f"{entry['artifact']} names no generating script"


def test_no_prose_number_escapes_the_block():
    """The rule that makes 'every number is machine-checked' true."""
    checked = {float(e["value"]) for e in ENTRIES}
    unchecked = []
    for match in MEASURED_NUMBER.finditer(PROSE):
        raw = match.group(0)
        value = float(raw)
        if not any(abs(value - c) <= max(DEFAULT_TOL, abs(c) * 1e-6) for c in checked):
            line = PROSE[: match.start()].count("\n") + 1
            unchecked.append(f"line {line}: {raw}")
    assert not unchecked, (
        "these numbers appear in the prose but not in the cfm-results block, so "
        "nothing checks them:\n  " + "\n  ".join(unchecked)
    )
