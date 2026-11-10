#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wf-A163 branch V — regression test for the adversarial c7 re-derivation.

Asserts that an INDEPENDENTLY written re-derivation of the c7 pool
(``molmetal/scripts/wfa163_c7_rederive.py``, which does not import the tool
under test) reproduces the published headline

    metrics/by_metric/cfm_fragmentation_postproc_arm.json
        /raw_axis/single_fragment_rate_over_parsed

to within 1e-12.

The test SKIPS (never fails) when the pool CSV is absent, so a clean checkout
without the generation pool still passes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts import wfa163_c7_rederive as R  # noqa: E402

POOL = Path(os.environ.get("WFA163_POOL", str(R.DEFAULT_POOL)))
PUBLISHED = Path(os.environ.get("WFA163_PUBLISHED", str(R.PUBLISHED_METRIC)))
A162 = R.A162_JSON
SCRIPT = R.REPO / "molmetal" / "scripts" / "wfa163_c7_rederive.py"
TOOL_UNDER_TEST = "wf_a155_ligand_connected_rate"

TOL = 1e-12


@pytest.fixture(scope="module")
def rederived():
    if not POOL.exists():
        pytest.skip(f"pool CSV absent: {POOL}")
    if not PUBLISHED.exists():
        pytest.skip(f"published metric absent: {PUBLISHED}")
    rows = R.read_pool(POOL)
    metals = R.load_vocabularies()["dblock_platinai"]
    res = R.analyze(rows, metals)
    res.pop("_facts", None)
    return res


def test_pool_denominator_is_12668(rederived):
    """12669 CSV rows, one unparseable -> 12668 analysed."""
    d = rederived["denominators"]
    assert d["n_rows_in_csv"] == 12669
    assert d["n_empty_string"] == 0
    assert d["n_unparseable"] == 1
    assert d["n_analyzed"] == 12668


def test_rederived_headline_matches_published(rederived):
    """The re-derived metal-inclusive rate equals the published headline."""
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    published_rate = pub["raw_axis"]["single_fragment_rate_over_parsed"]
    published_n = pub["raw_axis"]["n_single"]
    got_rate = rederived["metal_inclusive"]["single_fragment_rate"]
    got_n = rederived["metal_inclusive"]["n_single_fragment"]
    assert got_n == published_n
    assert abs(got_rate - published_rate) <= TOL


def test_ligand_only_rate_matches_a162(rederived):
    """The re-derived ligand-only rate equals the A162 artifact (if present)."""
    if not A162.exists():
        pytest.skip(f"A162 artifact absent: {A162}")
    a162 = json.loads(A162.read_text(encoding="utf-8"))
    published_rate = a162["ligand_only"]["single_fragment_rate"]
    published_n = a162["ligand_only"]["n_single_fragment"]
    got_rate = rederived["ligand_only"]["single_fragment_rate"]
    got_n = rederived["ligand_only"]["n_single_fragment"]
    assert got_n == published_n
    assert abs(got_rate - published_rate) <= TOL


def test_contingency_matches_a162(rederived):
    if not A162.exists():
        pytest.skip(f"A162 artifact absent: {A162}")
    a162 = json.loads(A162.read_text(encoding="utf-8"))
    c = a162["paired"]["contingency"]
    got = rederived["contingency"]
    assert got["a"] == c["a_both_connected"]
    assert got["b"] == c["b_ligand_only_connected_metal_inclusive_not"]
    assert got["c"] == c["c_metal_inclusive_connected_ligand_only_not"]
    assert got["d"] == c["d_neither_connected"]


def test_mcnemar_p_matches_a162(rederived):
    if not A162.exists():
        pytest.skip(f"A162 artifact absent: {A162}")
    a162 = json.loads(A162.read_text(encoding="utf-8"))
    published_p = a162["paired"]["mcnemar_exact"]["p_value_two_sided"]
    got_p = rederived["mcnemar"]["p_float"]
    assert got_p == published_p  # bit-for-bit: both are exact-rational conversions


def test_independence_contract_holds():
    """The re-derivation must NOT import the tool under test.

    Checked on the PARSED AST, so prose mentions of the forbidden APIs in the
    module docstring do not trip the guard.
    """
    import ast

    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: set = set()
    called: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            for a in node.names:
                imported.add(a.name)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            elif isinstance(fn, ast.Name):
                called.add(fn.id)

    assert not any(TOOL_UNDER_TEST in name for name in imported), (
        f"the re-derivation imports the tool under test: {sorted(imported)}"
    )
    # It must not call RDKit's fragment counter or its atom-removal API: the
    # connected-component count is this file's own BFS.
    assert "GetMolFrags" not in called
    assert "RemoveAtom" not in called
    # It must not copy the d-block metal list into the file.
    assert '"Sc", "Ti", "V", "Cr", "Mn", "Fe"' not in src
