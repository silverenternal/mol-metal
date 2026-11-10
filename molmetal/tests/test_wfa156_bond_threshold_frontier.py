#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`molmetal.scripts.wfa156_bond_threshold_frontier` (Wf-A156).

CPU-only by construction: this file imports no torch and asserts that the
tool imports none either.  Run with::

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \\
        -m pytest molmetal/tests/test_wfa156_bond_threshold_frontier.py -v

The synthetic 3-atom case is the contract the branch brief named: a C-C pair
at 1.5 A must survive ``lambda = 1.0`` and a pair at 3.5 A must not.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from molmetal.scripts import wfa156_bond_threshold_frontier as m  # noqa: E402


# ---------------------------------------------------------------------------
# The tool must be CPU-only
# ---------------------------------------------------------------------------


def test_tool_does_not_import_torch():
    """MEASURED contract: the frontier tool never imports torch.

    Checked in a FRESH subprocess, because this repo's
    ``molmetal/tests/conftest.py`` imports torch itself for its GPU-contract
    gate -- so an in-process ``sys.modules`` check would measure the
    conftest, not the tool.
    """
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r);"
        "import molmetal.scripts.wfa156_bond_threshold_frontier;"
        "print('torch' in sys.modules)" % str(REPO)
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("False"), (
        "importing the tool must not pull torch into sys.modules: " + proc.stdout
    )


def test_no_gpu_calls_in_source():
    """MEASURED contract: no GPU probe or cuda call in the tool's source."""
    src = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in ("nvidia-smi", "torch.cuda", "device('cuda", 'device("cuda'):
        assert forbidden not in src, f"forbidden GPU token in tool source: {forbidden}"
    assert "import torch" not in src, "the tool must not import torch"


# ---------------------------------------------------------------------------
# The threshold contract
# ---------------------------------------------------------------------------


def test_rcov_matches_rdkit_periodic_table():
    """MEASURED: the radii are RDKit's, not a private table."""
    assert m.rcov(6) == pytest.approx(0.76, abs=1e-9)   # C
    assert m.rcov(7) == pytest.approx(0.71, abs=1e-9)   # N
    assert m.rcov(8) == pytest.approx(0.66, abs=1e-9)   # O
    assert m.rcov(78) == pytest.approx(1.36, abs=1e-9)  # Pt
    assert m.rcov(44) == pytest.approx(1.46, abs=1e-9)  # Ru


def test_synthetic_3atom_cc_pair_at_1_5_a_survives_lambda_1_0():
    """The branch brief's contract, under the PAPER's convention.

    ``tau_ij = lambda_POLICY * (r_cov(C) + r_cov(C)) = 1.0 * 1.52 = 1.52 A``
    and ``1.5 A <= 1.52 A`` -> the bond survives.
    """
    tau = m.talfm_tau(6, 6, 1.0, "paper")
    assert tau == pytest.approx(1.52, abs=1e-9)
    assert m.bond_survives(1.5, 6, 6, 1.0, "paper") is True


def test_synthetic_3atom_cc_pair_at_3_5_a_does_not_survive_lambda_1_0():
    """A C-C pair at 3.5 A must NOT survive ``lambda = 1.0`` under the paper."""
    assert m.bond_survives(3.5, 6, 6, 1.0, "paper") is False


def test_synthetic_3atom_cc_pair_at_3_5_a_does_not_survive_under_either_convention_at_1_0():
    """Same pair, both conventions: 3.5 A is past tau for the PAPER's rule.

    Under the repo's ``lambda * pi`` reading the very same pair DOES survive
    (tau = pi * 1.52 = 4.776 A).  That difference is the point of the branch:
    the two readings are not the same rule, and this test pins it.
    """
    assert m.bond_survives(3.5, 6, 6, 1.0, "paper") is False
    assert m.bond_survives(3.5, 6, 6, 1.0, "repo_pi") is True
    assert m.talfm_tau(6, 6, 1.0, "repo_pi") == pytest.approx(
        math.pi * 1.52, abs=1e-9
    )


def test_survival_is_monotone_in_lambda():
    """MEASURED contract: a looser lambda never rejects a bond a tighter one kept."""
    for conv in ("paper", "repo_pi"):
        for z_i, z_j in ((6, 6), (6, 78), (17, 78), (7, 44)):
            prev = None
            for lam in m.LAMBDAS:
                cur = m.bond_survives(2.0, z_i, z_j, lam, conv)
                if prev is not None:
                    assert (not prev) or cur, (
                        f"survival decreased with lambda for {conv} {z_i}-{z_j} at {lam}"
                    )
                prev = cur


def test_unknown_convention_raises():
    with pytest.raises(ValueError):
        m.talfm_tau(6, 6, 1.0, "nonsense")


def test_threshold_is_pair_dependent():
    """A metal pair must NOT get the same tau as a carbon pair."""
    assert m.talfm_tau(6, 6, 1.0, "paper") < m.talfm_tau(6, 78, 1.0, "paper")


# ---------------------------------------------------------------------------
# End-to-end on the REAL pool
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_on_the_real_pool(tmp_path):
    """The tool must run end-to-end on the real 2,112-row pool, CPU-only.

    Writes into ``tmp_path`` so the published tree is untouched.
    """
    out_dir = tmp_path / "wfa156_report"
    rc = m.main(
        [
            "--pool",
            str(m.DEFAULT_POOL),
            "--out-dir",
            str(out_dir),
            "--skip-baseline",
        ]
    )
    assert rc == 0
    for name in ("report.json", "report.md", "per_row.csv", "per_row.json"):
        assert (out_dir / name).is_file(), f"missing {name}"

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    md = (out_dir / "report.md").read_text(encoding="utf-8")

    # -- the comparability caveat must be IN the report file, not just in chat
    assert "COMPARABILITY CAVEAT" in md
    assert "10,000 generated molecules" in md
    assert "92.98" in md
    assert report["comparability_caveat"].startswith("COMPARABILITY CAVEAT")

    # -- the population must be the pool the branch brief named
    assert report["pool"]["n_rows"] == 2112
    assert report["pool"]["n_rows_parsed"] == 2112
    assert report["pool"]["n_bonds_total"] > 0

    pcc = report["pool"]["population_cross_check"]
    assert pcc["rows_with_metal"] == 2112          # metal_ratio 1.0
    assert pcc["metal_atoms_sharing_a_fragment_with_a_nonmetal"] == 2082
    assert pcc["metal_atoms_alone_in_their_fragment"] == 30
    assert pcc["rows_containing_arrow"] == 0
    assert report["pool"]["fragment_count_histogram"]["1"] == 529

    # -- the sweep must be present for every (convention, lambda) cell
    rates = report["pool"]["rates"]
    assert len(rates) == len(m.CONVENTIONS) * len(m.LAMBDAS)
    for key, r in rates.items():
        assert r["convention"] in m.CONVENTIONS
        assert r["lambda"] in m.LAMBDAS
        assert 0.0 <= r["t1_frac_stricter_than_decoder_cutoff"] <= 1.0
        assert 0.0 <= r["t2_frac_guaranteed_survive"] <= 1.0
        assert 0.0 <= r["t2_frac_guaranteed_fail"] <= 1.0
        assert 0.0 <= r["t2_frac_undecidable"] <= 1.0
        # the three T2 buckets must partition the emitted bonds
        t2 = r["t2_n"]
        assert (
            t2["guaranteed_survive"] + t2["guaranteed_fail"] + t2["undecidable"]
            == report["pool"]["n_bonds_total"]
        )

    # -- the per-row files have one line per pool row
    csv_lines = (out_dir / "per_row.csv").read_text(encoding="utf-8").strip().split("\n")
    assert len(csv_lines) == 2113  # header + 2112 rows
    per_row_json = json.loads((out_dir / "per_row.json").read_text(encoding="utf-8"))
    assert len(per_row_json["rows"]) == 2112


@pytest.mark.slow
def test_pool_tier1_is_exact_and_geometry_free():
    """The T1 claim -- "tau vs the decoder's 3.5 A flat gate" -- is exact.

    It must hold for EVERY emitted bond without any geometry, so it is
    re-derived here from the per-element-pair histogram.
    """
    rows = m.read_pool(m.DEFAULT_POOL)
    refs = m.load_ref_lengths(m.DEFAULT_TMQM_ROOT)
    pool = m.analyse_pool(rows, refs)
    for key, r in pool["rates"].items():
        assert r["t1_n_stricter_than_decoder_cutoff"] + r["t1_n_looser_than_decoder_cutoff"] == (
            pool["n_bonds_total"]
        )
