#!/usr/bin/env python3
"""Every number in the predictive-line index must still match its artifact.

``docs/PREDICTIVE_LINE_INDEX.md`` exists because the predictive line has
retracted four headline numbers in one week (ligand-grouped R2 0.5828,
delta-sign 0.6193, "0.5826 is close to the 0.65 gate", SA<=4.5 33.7%), and
prose does not notice when the artifact under it moves.

The index carries one fenced block tagged ``json predictive-line-index``.
Each entry is ``{id, value, artifact, path[, tol]}``.  This test resolves the
dot path against the artifact and asserts the value is still there.  A stale
index is therefore a red test, not a reader's problem.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX = REPO_ROOT / "docs" / "PREDICTIVE_LINE_INDEX.md"

FENCE = re.compile(
    r"^```json\s+predictive-line-index\s*$(?P<body>.*?)^```$",
    re.MULTILINE | re.DOTALL,
)
DEFAULT_TOL = 1e-4


def _load_entries() -> list[dict]:
    text = INDEX.read_text(encoding="utf-8")
    match = FENCE.search(text)
    if match is None:
        raise AssertionError(
            "docs/PREDICTIVE_LINE_INDEX.md has no ```json predictive-line-index block"
        )
    entries = json.loads(match.group("body"))
    if not isinstance(entries, list) or not entries:
        raise AssertionError("the predictive-line-index block must be a non-empty list")
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


ENTRIES = _load_entries()


def test_the_index_is_machine_readable():
    """Guard the block itself: unique ids, required keys, artifact on disk."""
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
        f"{entry['id']}: {entry['artifact']}:{entry['path']} is {type(actual).__name__}, "
        "not a number"
    )
    assert abs(float(actual) - float(entry["value"])) <= tol, (
        f"{entry['id']} is stale: the index says {entry['value']}, "
        f"{entry['artifact']}:{entry['path']} says {actual}. "
        "Update docs/PREDICTIVE_LINE_INDEX.md (and re-read its retraction log)."
    )


def test_every_indexed_artifact_declares_the_script_that_wrote_it():
    """An artifact nobody can regenerate is an anecdote, not a measurement."""
    for entry in ENTRIES:
        doc = json.loads((REPO_ROOT / entry["artifact"]).read_text(encoding="utf-8"))
        provenance = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        declared = provenance.get("script") or doc.get("source") or doc.get("script")
        assert declared, f"{entry['artifact']} names no generating script"


def test_the_retraction_log_still_names_the_retracted_numbers():
    """The five retractions are the reason this file exists; losing them loses the warning."""
    text = INDEX.read_text(encoding="utf-8")
    for retracted in ("0.5828", "0.6193", "0.6009", "33.7", "0.5974"):
        assert retracted in text, f"retraction log lost {retracted}"
