# SPDX-License-Identifier: Apache-2.0
"""Wf-A155 BRANCH LIG — tests for the ligand-only "connected" rate.

WHAT THIS GUARDS
----------------
1. **Per-row metal-stripping maths on synthetic molecules** — a metal that
   bridges two ligand pieces, a metal that is ALONE in its fragment (the
   "metal fell off" case), a genuinely EMPTY ligand (a row that is metal
   and nothing else), a BIMETALLIC row, a free-halide counterion, and a
   molecule with no metal at all.  These pin the exact numbers a reader
   would recompute by hand.
2. **The empty-ligand row is counted, never dropped** — it is NOT
   "connected" in either arm and it is NOT removed from the denominator.
3. **The 2x2 contingency maths** on a synthetic list of paired booleans,
   including the b/c discordant counts that feed the exact McNemar test.
4. **The exact McNemar p-value** — cross-checked against
   ``scipy.stats.binomtest`` so the pure-stdlib implementation cannot
   silently disagree with the standard one.
5. **Both arms go through ONE function** — the metal-inclusive and
   ligand-only numbers must be internally comparable by construction.
6. **The JSON schema** — the emitted artifact carries the denominators, the
   two arms, the contingency, the four comparability breaks, the "what this
   is NOT" block and the premise-correction block.
7. **No CUDA** — the module must import without pulling torch in and with
   ``CUDA_VISIBLE_DEVICES`` forced to ``""``.
8. **The published pool** — an integration test over the real C8 pool that
   pins the headline counts (2112 rows, one metal each, 30 metal-alone
   rows, 0 empty-ligand rows, 529 metal-inclusive single-fragment rows,
   60 ligand-only single-fragment rows) and the 2x2 (53 / 7 / 476 / 1576).

Run:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \\
        molmetal/tests/test_wfa155_ligand_connected_rate.py -q \\
        --no-header -p no:cacheprovider
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import wf_a155_ligand_connected_rate as lig  # noqa: E402

POOL = PROJECT_ROOT / "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
ARTIFACT = PROJECT_ROOT / "metrics/by_metric/wfa155_ligand_connected_rate.json"


# ---------------------------------------------------------------------------
# 1. Per-row metal-stripping maths on synthetic molecules
# ---------------------------------------------------------------------------


def test_metal_bridging_two_ligand_pieces():
    """``C[Ru]C``: the metal is the ONLY thing joining two carbons.

    Metal-inclusive: ONE fragment (the metal bridges them).
    Ligand-only: TWO fragments (the carbons are disconnected without it).
    This is the ``c`` cell -- the metal is HIDING ligand fragmentation.
    """
    facts = lig.analyze_row("C[Ru]C", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["n_metal_atoms"] == 1
    assert facts["metal_symbols"] == ["Ru"]
    assert facts["full"]["n_frags"] == 1
    assert facts["full"]["n_heavy"] == 3
    assert facts["lig"]["n_frags"] == 2
    assert facts["lig"]["n_heavy"] == 2
    assert facts["lig"]["frag_sizes"] == [1, 1]
    assert facts["metal_alone_in_own_fragment"] is False
    assert facts["empty_ligand"] is False


def test_metal_alone_in_its_fragment_is_not_the_empty_ligand_case():
    """``CCO.[Ru]``: the metal sits alone, but the ligand does NOT.

    The brief for this branch asserted that "the metal alone in its fragment"
    rows ARE the empty-ligand rows.  MEASURED, they are not: stripping the
    metal here leaves ``CCO`` -- a NON-empty, CONNECTED ligand.  The two
    counters must stay separate.

    This row is also the ``b`` cell: the metal fell off, the ligand survived.
    """
    facts = lig.analyze_row("CCO.[Ru]", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["metal_alone_in_own_fragment"] is True
    assert facts["empty_ligand"] is False
    assert facts["full"]["n_frags"] == 2
    assert sorted(facts["full"]["frag_sizes"]) == [1, 3]
    assert facts["lig"]["n_heavy"] == 3
    assert facts["lig"]["n_frags"] == 1
    assert facts["lig"]["largest_frag_frac"] == 1.0


def test_free_halide_counterion_is_detected_and_the_metal_is_not_alone():
    """``CC[CH2][Ru].Cl``: the metal is bonded to the CH2 (so it is NOT alone
    in its fragment) and the Cl is a free halide counterion.

    Stripping the metal leaves ``CC[CH2]`` and ``Cl`` -- two fragments, a
    NON-empty ligand.  This is the shape most of the pool's 30 metal-alone
    rows have: the metal is one fragment, but the row also carries
    ligand-bearing fragments, so the ligand is NOT empty.
    """
    facts = lig.analyze_row("CC[CH2][Ru].Cl", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["metal_alone_in_own_fragment"] is False
    assert facts["empty_ligand"] is False
    assert facts["has_free_halide_counterion"] is True
    assert facts["lig"]["n_heavy"] == 4
    assert facts["lig"]["n_frags"] == 2
    assert sorted(facts["lig"]["frag_sizes"]) == [1, 3]


def test_empty_ligand_row_is_counted_not_dropped():
    """``[Ru].[Pt]``: the row IS metal and nothing else.

    Stripping leaves ZERO atoms: the ligand is EMPTY.  Such a row is NOT
    connected in either arm, and it must be visible as its own counter.
    """
    facts = lig.analyze_row("[Ru].[Pt]", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["n_metal_atoms"] == 2
    assert facts["metal_symbols"] == ["Pt", "Ru"]
    assert facts["lig"]["n_heavy"] == 0
    assert facts["lig"]["n_frags"] == 0
    assert facts["lig"]["largest_frag_frac"] is None
    assert facts["empty_ligand"] is True
    # the FULL molecule is two metal fragments -> not connected either
    assert facts["full"]["n_frags"] == 2
    # both metals are alone in their own fragments
    assert facts["metal_alone_in_own_fragment"] is True


def test_bimetallic_row_strips_both_metals():
    """``[Pt].CC[Ru]``: two metals, one ligand chain.

    Metal-inclusive: TWO fragments (the free Pt and the Pt...CC...Ru chain).
    Ligand-only: ONE fragment (``CC``) -- the ligand IS intact.
    This is the ``b`` cell -- the metal fell off and the ligand survived.
    """
    facts = lig.analyze_row("[Pt].CC[Ru]", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["n_metal_atoms"] == 2
    assert facts["full"]["n_frags"] == 2
    assert facts["lig"]["n_frags"] == 1
    assert facts["lig"]["n_heavy"] == 2
    assert facts["empty_ligand"] is False
    assert facts["metal_alone_in_own_fragment"] is True


def test_row_with_no_metal_is_its_own_ligand():
    """No metal -> stripping is a no-op -> both arms agree."""
    facts = lig.analyze_row("CCO", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["n_metal_atoms"] == 0
    assert facts["metal_symbols"] == []
    assert facts["full"]["n_frags"] == 1
    assert facts["lig"]["n_frags"] == 1
    assert facts["full"]["n_heavy"] == facts["lig"]["n_heavy"] == 3
    assert facts["empty_ligand"] is False
    assert facts["metal_alone_in_own_fragment"] is False


def test_strip_metals_helper_reports_what_it_removed():
    """``strip_metals`` returns the stripped mol AND the removed symbols."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C[Ru]C")
    stripped, removed = lig.strip_metals(mol, set(lig.METAL_SYMBOLS))
    assert removed == ["Ru"]
    assert stripped.GetNumHeavyAtoms() == 2
    assert len(Chem.GetMolFrags(stripped, asMols=False, sanitizeFrags=False)) == 2

    mol2 = Chem.MolFromSmiles("CCO")
    stripped2, removed2 = lig.strip_metals(mol2, set(lig.METAL_SYMBOLS))
    assert removed2 == []
    assert stripped2.GetNumHeavyAtoms() == 3


def test_main_group_metal_is_not_stripped_by_the_dblock_rule():
    """The rule is the repo's d-block list, so ``Al`` is NOT a metal here.

    This is stated, not accidental: the artifact records the rule verbatim
    and shows the answer is invariant under the narrower trainer cohort.
    """
    facts = lig.analyze_row("C[Al]C", lig.METAL_SYMBOLS)
    assert facts is not None
    assert facts["n_metal_atoms"] == 0
    assert facts["full"]["n_frags"] == 1
    assert facts["lig"]["n_frags"] == 1


def test_unparseable_smiles_returns_none():
    assert lig.analyze_row("not-a-smiles((", lig.METAL_SYMBOLS) is None


# ---------------------------------------------------------------------------
# 2. Both arms go through ONE function (internal comparability)
# ---------------------------------------------------------------------------


def test_both_arms_use_the_same_arm_stats_function():
    """The two arms must not be two code paths that happen to agree."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C[Ru]C.Cl")
    full = lig.arm_stats(mol)
    stripped, _ = lig.strip_metals(mol, set(lig.METAL_SYMBOLS))
    lig_arm = lig.arm_stats(stripped)
    assert set(full.keys()) == set(lig_arm.keys())
    # largest_frag_frac is the largest component over the molecule's heavy
    # atoms -- same definition, different denominators.  The metal-inclusive
    # arm's largest component INCLUDES the metal atom.
    assert full["largest_frag_frac"] == 3.0 / 4.0
    assert lig_arm["largest_frag_frac"] == 1.0 / 3.0


# ---------------------------------------------------------------------------
# 3. The 2x2 contingency maths
# ---------------------------------------------------------------------------


def test_contingency_on_a_synthetic_list():
    pairs = [
        (True, True),    # a
        (True, False),   # b  -- metal fell off, ligand intact
        (False, True),   # c  -- metal bridged a fragmented ligand
        (False, False),  # d
        (True, True),
        (False, True),
    ]
    cont = lig.contingency(pairs)
    assert cont == {"a": 2, "b": 1, "c": 2, "d": 1}
    assert sum(cont.values()) == len(pairs)


def test_contingency_on_the_empty_list():
    cont = lig.contingency([])
    assert cont == {"a": 0, "b": 0, "c": 0, "d": 0}


# ---------------------------------------------------------------------------
# 4. The exact McNemar p-value
# ---------------------------------------------------------------------------


def test_exact_mcnemar_p_matches_scipy_binomtest():
    """Cross-check the pure-stdlib exact binomial against scipy."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for b, c in [(7, 476), (1, 1), (0, 5), (3, 3), (10, 2), (0, 0)]:
        got = lig.exact_mcnemar_p(b, c)
        assert got["discordant_total"] == b + c
        if b + c == 0:
            assert got["p_value_two_sided"] == 1.0
            continue
        expected = scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue
        assert got["p_value_two_sided"] == pytest.approx(expected, rel=1e-9), (
            b,
            c,
            got["p_value_two_sided"],
            expected,
        )
        # the exact rational must agree with the float
        exact = got["p_value_exact_numerator"] / got["p_value_exact_denominator"]
        assert exact == pytest.approx(got["p_value_two_sided"], rel=1e-9)


def test_exact_mcnemar_p_is_not_washed_out_by_float_arithmetic():
    """b=7 / c=476 gives p ~ 1e-130 -- a float-only implementation that
    computes ``2 * sum(C(n,k) * 0.5**n)`` would underflow.  The exact
    rational must not."""
    got = lig.exact_mcnemar_p(7, 476)
    assert got["p_value_underflow_to_zero"] is False
    assert 0.0 < got["p_value_two_sided"] < 1e-100
    assert got["p_value_log10"] < -100.0


# ---------------------------------------------------------------------------
# 5. The JSON schema
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="published artifact not on disk")
def test_artifact_schema():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for key in (
        "metric",
        "tag",
        "date",
        "wave",
        "purpose",
        "direction",
        "gpu",
        "source_pool",
        "definitions",
        "denominators",
        "metal_inclusive",
        "ligand_only",
        "paired",
        "metal_set_invariance",
        "verdict",
        "premise_checks",
        "not_comparable_to",
        "comparability",
        "caveats",
        # Wf-A162 additions
        "end_to_end_headline",
        "published_cross_check",
        "narrative_self_consistency",
    ):
        assert key in art, f"artifact is missing {key!r}"
    assert art["tag"] == "MEASURED"
    assert art["metric"] == "wfa155_ligand_connected_rate"


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="published artifact not on disk")
def test_artifact_is_internally_consistent():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    den = art["denominators"]
    n = den["n_analyzed"]
    assert n == den["n_rows_in_csv"]

    # the 2x2 must partition the analysed rows
    cont = art["paired"]["contingency"]
    total = (
        cont["a_both_connected"]
        + cont["b_ligand_only_connected_metal_inclusive_not"]
        + cont["c_metal_inclusive_connected_ligand_only_not"]
        + cont["d_neither_connected"]
    )
    assert total == n

    # each histogram must partition the analysed rows
    for arm in ("metal_inclusive", "ligand_only"):
        assert sum(art[arm]["n_frags_histogram"].values()) == n
        assert art[arm]["single_fragment_rate"] == pytest.approx(
            art[arm]["n_single_fragment"] / n
        )
        assert art[arm]["n_frags_histogram"]["1"] == art[arm]["n_single_fragment"]

    # the discordant counts must agree with the contingency
    mcn = art["paired"]["mcnemar_exact"]
    assert mcn["discordant_total"] == (
        cont["b_ligand_only_connected_metal_inclusive_not"]
        + cont["c_metal_inclusive_connected_ligand_only_not"]
    )
    assert mcn["discordant_total"] == 483
    assert mcn["p_value_two_sided"] < 1e-100

    # the two arms must not be a restatement of each other
    assert art["metal_inclusive"]["single_fragment_rate"] != art["ligand_only"][
        "single_fragment_rate"
    ]


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="published artifact not on disk")
def test_artifact_carries_the_four_comparability_breaks():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    breaks = art["comparability"]["breaks"]
    assert len(breaks) == 4
    names = {b["break"] for b in breaks}
    assert names == {"DENOMINATOR", "METRIC", "POST-PROCESSING", "HYDROGENS"}
    for b in breaks:
        assert b["frontier"] and b["ours"]
    assert "92.7-94" in art["comparability"]["frontier_band"]
    assert "92.7-94" in art["comparability"]["statement"]


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="published artifact not on disk")
def test_artifact_states_what_it_is_not():
    """Wf-A162: the two keys were renamed from the pool-specific
    ``8_51_percent_headline`` / ``25_05_percent_single_pool_fragment_count``
    to pool-AGNOSTIC names, because the old names asserted a reproduction
    the tool never verified -- and the old body quoted one pool's row count
    while running against another."""
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    ncmp = art["not_comparable_to"]
    assert "end_to_end_median_headline" in ncmp
    assert "other_pools_single_pool_counts" in ncmp
    # the frontier median itself is a MACHINE field now, not a prose literal
    assert art["end_to_end_headline"]["median"] == 0.08506944444444445
    assert art["end_to_end_headline"]["n_runs"] == 9
    # the row count in the prose must be THIS pool's row count
    n = art["denominators"]["n_analyzed"]
    assert f"ONE pool of {n} rows" in ncmp["end_to_end_median_headline"]


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="published artifact not on disk")
def test_artifact_records_the_premise_correction():
    """The brief asserted the 30 metal-alone rows ARE the empty-ligand rows.
    MEASURED, they are not; the artifact must carry the correction."""
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    pc = art["premise_checks"]
    assert pc["verdict"] == "NOT SUPPORTED BY THE DATA"
    assert pc["measured_metal_alone_in_own_fragment_rows"] == 30
    assert pc["measured_empty_ligand_rows"] == 0


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="published artifact not on disk")
def test_artifact_metal_set_invariance():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    inv = art["metal_set_invariance"]
    assert inv["identical"] is True
    assert (
        inv["ligand_only_single_fragment_rate_dblock"]
        == inv["ligand_only_single_fragment_rate_cohort"]
    )


# ---------------------------------------------------------------------------
# 6. No CUDA
# ---------------------------------------------------------------------------


def test_module_imports_without_torch():
    """STATIC import check: this module's own import graph has no torch."""
    path = PROJECT_ROOT / "molmetal/scripts/wf_a155_ligand_connected_rate.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import inside the package
                continue
            if node.module:
                roots.add(node.module.split(".")[0])

    assert "torch" not in roots, f"module imports torch: {sorted(roots)}"
    assert "numpy" not in roots, f"module imports numpy: {sorted(roots)}"
    assert roots <= {
        "argparse", "csv", "json", "math", "os", "re", "sys", "collections",
        "pathlib", "typing", "__future__", "rdkit", "molmetal",
    }, f"unexpected imports: {sorted(roots)}"

    body = src.replace("CUDA_VISIBLE_DEVICES", "")
    for banned in ("torch.cuda", "cuda(", ".cuda", "device=", "set_device"):
        assert banned not in body, f"module names a CUDA entry point: {banned}"


def test_subprocess_import_forces_cpu_and_sees_no_gpu():
    """A fresh interpreter: CUDA_VISIBLE_DEVICES='' and no GPU is visible."""
    code = (
        "import sys, os;"
        "sys.path.insert(0, %r);"
        "import molmetal.scripts.wf_a155_ligand_connected_rate as m;"
        "print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')));"
        "import torch;"
        "print(bool(torch.cuda.is_available()))"
        % str(PROJECT_ROOT)
    )
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = [ln for ln in proc.stdout.strip().splitlines() if ln]
    assert out[0] == "''", f"CUDA_VISIBLE_DEVICES was not forced empty: {out}"
    assert out[1] == "False", f"a GPU is visible to this module: {out}"


# ---------------------------------------------------------------------------
# 7. The published pool (integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not POOL.is_file(), reason="published pool not on disk")
def test_published_pool_headline_counts():
    """The real C8 pool, end to end through this tool's own code path."""
    res = lig.analyze_pool(POOL, metal_symbols=set(lig.METAL_SYMBOLS))
    den = res["denominators"]
    assert den["n_rows_in_csv"] == 2112
    assert den["n_unparseable"] == 0
    assert den["n_analyzed"] == 2112
    assert den["n_with_no_metal"] == 0
    assert den["n_empty_ligand_after_strip"] == 0
    assert den["n_metal_alone_in_own_fragment"] == 30

    # the metal-inclusive arm must reproduce the pool's own precomputed column
    assert res["csv_cross_check"]["n_frags_raw_mismatches"] == 0
    assert res["csv_cross_check"]["largest_frag_frac_max_abs_delta"] < 1e-4

    assert res["metal_inclusive"]["n_single_fragment"] == 529
    assert res["ligand_only"]["n_single_fragment"] == 60
    assert res["metal_inclusive"]["single_fragment_rate"] == pytest.approx(
        529 / 2112
    )
    assert res["ligand_only"]["single_fragment_rate"] == pytest.approx(60 / 2112)

    cont = res["paired"]["contingency"]
    assert cont["a_both_connected"] == 53
    assert cont["b_ligand_only_connected_metal_inclusive_not"] == 7
    assert cont["c_metal_inclusive_connected_ligand_only_not"] == 476
    assert cont["d_neither_connected"] == 1576

    assert res["paired"]["mcnemar_exact"]["discordant_total"] == 483
    assert res["paired"]["mcnemar_exact"]["p_value_two_sided"] < 1e-100

    # every row carries exactly one metal, and it is one of the six the CFM
    # was trained on
    assert res["n_metal_atoms_per_row"] == {"1": 2112}
    assert set(res["metal_symbol_counts"]) == {"Ru", "Re", "Pt", "Ir", "Os", "Rh"}


@pytest.mark.skipif(not POOL.is_file(), reason="published pool not on disk")
def test_published_pool_metal_set_invariance():
    """The answer does not depend on which of the two repo-canonical metal
    sets is used -- the rule is not pool-fitted."""
    broad = lig.analyze_pool(POOL, metal_symbols=set(lig.METAL_SYMBOLS))
    narrow = lig.analyze_pool(POOL, metal_symbols=set(lig.COHORT_METAL_SYMBOLS))
    assert (
        broad["ligand_only"]["single_fragment_rate"]
        == narrow["ligand_only"]["single_fragment_rate"]
    )
    assert (
        broad["denominators"]["n_empty_ligand_after_strip"]
        == narrow["denominators"]["n_empty_ligand_after_strip"]
    )


@pytest.mark.skipif(not POOL.is_file(), reason="published pool not on disk")
def test_published_pool_ligand_arm_is_not_a_restatement_of_the_headline():
    """Guard against the failure mode the brief names: a 'new' metric that is
    just the 8.51 % headline or the 25.05 % single-pool count again."""
    res = lig.analyze_pool(POOL, metal_symbols=set(lig.METAL_SYMBOLS))
    lig_rate = res["ligand_only"]["single_fragment_rate"]
    full_rate = res["metal_inclusive"]["single_fragment_rate"]
    headline_851 = 0.08506944444444445
    assert lig_rate != pytest.approx(headline_851, abs=1e-9)
    assert lig_rate != pytest.approx(full_rate, abs=1e-9)
    # and the ligand arm really is a different DENOMINATOR DEFINITION, not a
    # re-labelling of the same measurement
    assert res["ligand_only"]["mean_n_frags"] > res["metal_inclusive"]["mean_n_frags"]
