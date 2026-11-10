# SPDX-License-Identifier: Apache-2.0
"""Wf-A159 Branch F -- guards the timeout-verdict FRONTIER comparison matrix.

WHAT THIS FILE GUARDS
---------------------
1. The artifact exists on disk and parses as JSON.
2. SCHEMA: every source and every cell carries the required keys.
3. Every cell has a locator + a url + a retrieval date (the brief's
   "every cell has a locator + url" invariant).
4. No cell claims a quote it does not carry: a cell whose verdict is
   AGREES / DISAGREES / DISTINGUISHES MUST carry a non-empty quote AND that
   quote must be a substring of its own quote_context; a cell whose verdict
   is NOT-STATED / NOT-RETRIEVED MUST carry an EMPTY quote.
5. The summary counts are DERIVED from the cells, not typed.
6. Every cell url appears in the retrieval log; every finding cites a
   source that exists.
7. DETERMINISM: re-running the generator into a tmp dir reproduces the
   on-disk artifacts BYTE FOR BYTE (the embedded corpus is a constant, so
   the script is offline and re-runnable).
8. NO-CUDA / NO-TORCH: importing the generator must not pull in torch, and
   the generator's source must not import it.

CPU-ONLY: nothing here spawns a subprocess, loads a checkpoint, or touches a
GPU.  ``CUDA_VISIBLE_DEVICES`` is forced to the empty string at import time,
before anything else runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# Force the CPU-only envelope BEFORE any import that could reach a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfa159_timeout_verdict_frontier.py"
REPORT_DIR = REPO_ROOT / "molmetal" / "reports" / "wfa159_timeout_verdict_frontier"
JSON_PATH = REPORT_DIR / "timeout_verdict_matrix.json"
MD_PATH = REPORT_DIR / "timeout_verdict_matrix.md"

CLAIMING = ("AGREES", "DISAGREES", "DISTINGUISHES")
ABSENCE = ("NOT-STATED", "NOT-RETRIEVED")
CELL_KEYS = (
    "verdict",
    "quote",
    "quote_context",
    "locator",
    "url",
    "retrieved_at",
    "note",
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("wfa159_frontier", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_matrix():
    assert JSON_PATH.exists(), f"missing artifact: {JSON_PATH}"
    with JSON_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_artifact_exists_and_parses():
    assert JSON_PATH.exists(), f"missing artifact: {JSON_PATH}"
    assert MD_PATH.exists(), f"missing artifact: {MD_PATH}"
    matrix = _load_matrix()
    assert matrix["wave"] == "Wf-A159"
    assert matrix["branch"] == "F"
    assert matrix["session_date"] == "2026-11-01"
    assert matrix["sources"], "corpus is empty"
    assert matrix["retrieval_log"], "retrieval log is empty"


def test_schema_every_source_and_cell_has_required_keys():
    matrix = _load_matrix()
    for src in matrix["sources"]:
        for key in ("id", "name", "kind", "url", "retrieved_at", "tool", "cells"):
            assert key in src, f"{src.get('id')} missing {key}"
        for qid, cell in src["cells"].items():
            for key in CELL_KEYS:
                assert key in cell, f"{src['id']}.{qid} missing {key}"
            assert cell["verdict"] in matrix["verdict_vocabulary"], (
                f"{src['id']}.{qid} bad verdict {cell['verdict']!r}"
            )


def test_every_cell_has_locator_and_url():
    matrix = _load_matrix()
    for src in matrix["sources"]:
        for qid, cell in src["cells"].items():
            assert str(cell["locator"]).strip(), f"{src['id']}.{qid} has no locator"
            assert str(cell["url"]).strip().startswith("http"), (
                f"{src['id']}.{qid} has no retrieval url"
            )
            assert str(cell["retrieved_at"]).strip(), (
                f"{src['id']}.{qid} has no retrieval date"
            )


def test_no_cell_claims_a_quote_it_does_not_carry():
    """A claiming cell MUST carry a quote that is present in its own
    context; an absence cell MUST carry no quote at all."""
    matrix = _load_matrix()
    for src in matrix["sources"]:
        for qid, cell in src["cells"].items():
            if cell["verdict"] in CLAIMING:
                assert cell["quote"].strip(), (
                    f"{src['id']}.{qid} claims {cell['verdict']} with no quote"
                )
                assert cell["quote"] in cell["quote_context"], (
                    f"{src['id']}.{qid} quote is not inside its own context"
                )
            elif cell["verdict"] in ABSENCE:
                assert not cell["quote"].strip(), (
                    f"{src['id']}.{qid} claims {cell['verdict']} but carries a quote"
                )
            else:  # pragma: no cover -- guarded by the schema test
                raise AssertionError(f"{src['id']}.{qid} unknown verdict")


def test_summary_counts_are_derived_not_typed():
    matrix = _load_matrix()
    stored = matrix["summary"]["verdict_counts"]
    for qid in stored:
        counts = {v: 0 for v in matrix["verdict_vocabulary"]}
        for src in matrix["sources"]:
            counts[src["cells"][qid]["verdict"]] += 1
        assert stored[qid] == counts, f"summary counts drifted for {qid}"
    assert matrix["summary"]["n_sources"] == len(matrix["sources"])
    assert matrix["summary"]["n_cells"] == len(matrix["sources"]) * len(
        matrix["questions"]
    )


def test_retrieval_log_covers_every_cell_url_and_findings_cite_real_sources():
    matrix = _load_matrix()
    logged = {entry["url"] for entry in matrix["retrieval_log"]}
    for src in matrix["sources"]:
        for qid, cell in src["cells"].items():
            assert cell["url"] in logged, f"{src['id']}.{qid} url not in retrieval log"
    known = {src["id"] for src in matrix["sources"]}
    for finding in matrix["findings"]:
        for sid in finding["supporting"] + finding["contradicting"]:
            assert sid in known, f"finding {finding['id']} cites unknown source {sid}"


def test_corpus_covers_both_directions():
    """The matrix must contain at least one DISAGREES and at least one
    AGREES on Q0 -- a corpus that only agreed with the repo would be a
    cherry-picked matrix, not a frontier."""
    matrix = _load_matrix()
    q0 = "Q0_TIMEOUT_NOT_DEFECT_CLAIM"
    verdicts = {src["cells"][q0]["verdict"] for src in matrix["sources"]}
    assert "DISAGREES" in verdicts, "no source disagrees with the repo's rule"
    assert "AGREES" in verdicts, "no source agrees with the repo's rule"


def test_generator_self_checks_pass():
    module = _load_generator()
    checks = module.self_check(module.build_matrix())
    failed = [(name, detail) for name, ok, detail in checks if not ok]
    assert not failed, f"generator self-checks failed: {failed}"


def test_generator_is_deterministic_and_reproduces_the_on_disk_artifact(tmp_path):
    module = _load_generator()
    rc = module.main(["--out-dir", str(tmp_path)])
    assert rc == 0, f"generator exited {rc}"
    for name in ("timeout_verdict_matrix.json", "timeout_verdict_matrix.md"):
        produced = (tmp_path / name).read_bytes()
        on_disk = (REPORT_DIR / name).read_bytes()
        assert produced == on_disk, f"{name} is not byte-reproducible"


def test_no_cuda_no_torch():
    """Importing the generator must not pull in torch, and its source must
    not import torch at all.

    The invariant is "no NEW torch module": the repo's own pytest conftest
    imports torch (molmetal/_device.py) before any test runs, so a bare
    "'torch' not in sys.modules" assertion would be testing the harness,
    not this branch.
    """
    before = {n for n in sys.modules if n == "torch" or n.startswith("torch.")}
    module = _load_generator()
    after = {n for n in sys.modules if n == "torch" or n.startswith("torch.")}
    assert after - before == set(), f"the generator imported torch: {after - before}"
    assert "torch" not in vars(module), "the generator holds a torch reference"
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import\s+torch|from\s+torch\s+import)", source, re.M), (
        "the generator imports torch"
    )
    assert "nvidia-smi" not in source, "the generator probes the GPU"
    for pattern in (r"\.cuda\(", r"torch\.cuda", r"cuda:[01]", r"device\s*=\s*[\"']cuda"):
        assert not re.search(pattern, source), f"the generator touches CUDA: {pattern}"


def test_generator_writes_only_into_its_own_report_dir(tmp_path):
    """The generator must not write anywhere else.  Asserted by running it
    into a tmp dir and checking the canonical report dir is untouched
    (mtime + size of the two artifacts)."""
    module = _load_generator()
    before = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in (JSON_PATH, MD_PATH)
        if p.exists()
    }
    rc = module.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    after = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in (JSON_PATH, MD_PATH)
        if p.exists()
    }
    assert before == after, "the generator touched the canonical report dir"


def test_markdown_carries_the_matrix_and_the_retrieval_log():
    md = MD_PATH.read_text(encoding="utf-8")
    assert "## 1. The matrix" in md
    assert "## 5. Retrieval log" in md
    assert "## 6. Caveats" in md
    matrix = _load_matrix()
    for src in matrix["sources"]:
        assert src["id"] in md, f"{src['id']} missing from the markdown"
    for entry in matrix["retrieval_log"]:
        assert entry["url"] in md, f"{entry['url']} missing from the retrieval log"
