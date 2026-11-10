# SPDX-License-Identifier: Apache-2.0
"""Wf-A155 BRANCH COORD — tests for the metal coordination-sphere metric.

WHAT THIS GUARDS
----------------
1. **Per-row maths on synthetic molecules** — a cisplatin-like complex, a
   metal-only fragment (the EMPTY coordination sphere), a metal with a
   carbanion donor, a bimetallic row, and a fragmented row where the donors
   were split off.  These pin the exact numbers a reader would recompute by
   hand.
2. **Oxidation-state layering** — the three layers are RULE-BASED INFERENCE
   and must never be reported as a measurement; Layer B's scope must hold
   (a charged polyatomic donor takes the row OUT OF SCOPE rather than
   silently producing a number).
3. **Distribution aggregation** — the CN histogram, the coordination-class
   counts, and the donor-motif distribution.
4. **The JSON schema** — the emitted artifact carries the population, the
   definitions, the comparability statement, the "what this is NOT" block and
   the per-row columnar dump.
5. **No CUDA** — the module must import without pulling torch in and with
   ``CUDA_VISIBLE_DEVICES`` forced to ``""``.
6. **The published pool** — an integration test over the real C8 pool that
   pins the headline counts (2112 rows, one metal each, 30 metal-alone rows,
   zero dative bonds, zero fragment-count mismatches).

Run:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \\
        molmetal/tests/test_wfa155_metal_coordination_sphere.py -q \\
        --no-header -p no:cacheprovider
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import wf_a155_metal_coordination_sphere as coord  # noqa: E402

POOL = PROJECT_ROOT / "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"


# ---------------------------------------------------------------------------
# 1. Per-row maths on synthetic molecules
# ---------------------------------------------------------------------------


def test_cisplatin_like_complex():
    """cis-[Pt(NH3)2Cl2]: 4 donors, 2 N + 2 halide, OS +2, at capacity."""
    spheres = coord.analyze_smiles("[NH3][Pt]([NH3])(Cl)Cl")
    assert len(spheres) == 1
    s = spheres[0]
    assert s.metal_symbol == "Pt"
    assert s.metal_z == 78
    assert s.metal_formal_charge == 0
    assert s.coordination_number == 4
    assert s.total_degree == 4  # no implicit H on the metal
    # the motif is the LEXICOGRAPHICALLY sorted multiset of neighbour elements
    assert s.neighbour_motif == "Cl,Cl,N,N"
    assert s.donor_class_counts == {"N_donor": 2, "halide": 2}
    assert s.n_classical_donors == 4
    assert s.metal_alone_in_fragment is False
    assert s.row_n_frags == 1


def test_cisplatin_like_oxidation_state_layers():
    """Two halide donors -> Layer B gives +2 (the ionic approximation)."""
    s = coord.analyze_smiles("[NH3][Pt]([NH3])(Cl)Cl")[0]
    # Layer A: the formal charge of the metal, 0 here
    assert s.os_formal_charge == 0
    # Layer B: Q_row (0) + 2 halide donors (+1 each) = +2, in scope
    assert s.os_charge_balanced == 2
    assert s.os_charge_balanced_status == "determined_in_scope"
    assert s.os_charge_balanced_usable is True
    assert s.os_used == 2
    assert s.os_source.startswith("os_charge_balanced")
    # Layer C: the repo prior for Pt
    assert s.os_prior_default == 2
    # Pt(II) capacity is 4 -> the measured CN is exactly at capacity
    assert s.expected_cn == 4
    assert s.coordination_class == "at_capacity"


def test_metal_only_fragment_has_an_empty_sphere():
    """A metal alone in its fragment: the coordination sphere is EMPTY."""
    s = coord.analyze_smiles("[Pt]")[0]
    assert s.coordination_number == 0
    assert s.total_degree == 0
    assert s.metal_alone_in_fragment is True
    assert s.coordination_class == "empty"
    assert s.neighbour_motif == ""
    assert s.donor_class_counts == {}
    assert s.n_classical_donors == 0
    assert s.expected_cn == 4  # Pt(II) capacity, still reported
    assert s.geometry_class is None


def test_metal_with_a_carbanion_donor():
    """[CH2-][Pt]: the carbanion is a monatomic anionic X-type donor."""
    s = coord.analyze_smiles("[CH2-][Pt]")[0]
    assert s.coordination_number == 1
    assert s.neighbour_motif == "C"
    assert s.donor_class_counts == {"C_donor": 1}
    assert s.row_total_charge == -1
    # the carbanion is IN SCOPE for Layer B: it is the only charged atom in
    # the metal's fragment, so the ligand's OS = -1 is well defined
    assert s.os_charge_balanced_status == "determined_in_scope"
    assert s.os_charge_balanced == 0  # Q_row (-1) + 1 anionic X donor (+1)
    # ... and a non-positive value is NOT usable as an oxidation state, so
    # the fallback (the prior) is used instead
    assert s.os_charge_balanced_usable is False
    assert s.os_used == 2  # Pt prior
    assert s.os_source.startswith("os_prior_default")


def test_charged_polyatomic_donor_takes_the_row_out_of_scope():
    """A carboxylate written [O-] -> Layer B is OUT OF SCOPE, not a number."""
    # Pt is bonded to the neutral O; the carboxylate's charge sits on the
    # other O, so the ligand's oxidation state is not recoverable
    s = coord.analyze_smiles("C(=O)([O-])O[Pt]")[0]
    assert s.coordination_number == 1
    assert s.neighbour_motif == "O"
    assert s.os_charge_balanced is None
    assert s.os_charge_balanced_status == "out_of_scope_charged_polyatomic_donor"
    assert s.os_charge_balanced_usable is False
    assert s.os_used == 2  # falls back to the prior
    assert s.os_source.startswith("os_prior_default")


def test_bimetallic_row_yields_one_sphere_per_metal():
    """Two metals in one row -> two spheres, each with its own neighbourhood."""
    spheres = coord.analyze_smiles("Cl[Pt]CC[Ir]Cl")
    assert len(spheres) == 2
    assert [s.metal_symbol for s in spheres] == ["Pt", "Ir"]
    for s in spheres:
        # each metal sees the bridging carbon AND its own chloride
        assert s.coordination_number == 2
        assert s.neighbour_motif == "C,Cl"
        assert s.donor_class_counts == {"C_donor": 1, "halide": 1}
        assert s.row_n_frags == 1
    # the two spheres are DIFFERENT metals with different capacities
    assert spheres[0].expected_cn == 4  # Pt(II) square planar
    assert spheres[1].expected_cn == 6  # Ir(III) octahedral
    assert spheres[0].coordination_class == "under"
    assert spheres[1].coordination_class == "under"


def test_disconnected_bimetallic_row():
    """Two metals in two fragments -> each is alone in its own fragment."""
    spheres = coord.analyze_smiles("[Pt].[Ru]")
    assert len(spheres) == 2
    assert [s.metal_symbol for s in spheres] == ["Pt", "Ru"]
    assert [s.coordination_number for s in spheres] == [0, 0]
    assert all(s.row_n_frags == 2 for s in spheres)
    # BOTH fragments carry a metal, so there is no metal-free fragment
    assert all(s.n_metal_free_fragments == 0 for s in spheres)
    assert all(s.metal_alone_in_fragment for s in spheres)


def test_fragmented_row_with_split_off_donors():
    """A donor in another fragment is NOT in the sphere, but IS counted."""
    s = coord.analyze_smiles("Cl[Pt]Cl.N")[0]
    assert s.coordination_number == 2
    assert s.neighbour_motif == "Cl,Cl"
    assert s.row_n_frags == 2
    assert s.n_metal_free_fragments == 1
    # the free amine sits in the other fragment: it is a disconnected donor
    assert s.disconnected_donor_atoms == 1
    assert s.metal_fragment_atoms == 3  # Pt + 2 Cl
    assert s.metal_fragment_is_largest is True


def test_double_bond_counts_as_one_neighbour():
    """A metal-oxo DOUBLE bond is one donor, not two."""
    s = coord.analyze_smiles("[Pt]=O")[0]
    assert s.coordination_number == 1
    assert s.neighbour_motif == "O"
    assert s.donor_class_counts == {"O_donor": 1}


def test_no_metal_and_unparseable_smiles():
    """A ligand-only SMILES and a broken SMILES both yield no sphere."""
    assert coord.analyze_smiles("CCO") == []
    assert coord.analyze_smiles("this-is-not-a-smiles") == []
    assert coord.analyze_smiles("") == []


def test_oxygen_is_not_a_metal():
    """Regression guard: an early draft mapped O -> 8 as a 'metal'."""
    assert "O" not in coord.METAL_Z_BY_SYMBOL
    assert coord.analyze_smiles("CCO") == []
    assert coord.analyze_smiles("[OH2]") == []


def test_donor_class_mapping():
    assert coord.donor_class("N") == "N_donor"
    assert coord.donor_class("O") == "O_donor"
    assert coord.donor_class("C") == "C_donor"
    assert coord.donor_class("Cl") == "halide"
    assert coord.donor_class("Br") == "halide"
    assert coord.donor_class("I") == "halide"
    assert coord.donor_class("F") == "halide"
    assert coord.donor_class("S") == "S_donor"
    assert coord.donor_class("B") == "other"


# ---------------------------------------------------------------------------
# 2. Distribution aggregation
# ---------------------------------------------------------------------------


def _synthetic_spheres():
    """A small, hand-checkable pool of spheres."""
    out = []
    for smi in (
        "[Pt]",  # empty sphere
        "[Pt]",  # empty sphere (2nd)
        "Cl[Pt]Cl",  # CN 2, under
        "[NH3][Pt]([NH3])(Cl)Cl",  # CN 4, at capacity
        "Cl[Pt]Cl.N",  # CN 2, under, one split-off donor
    ):
        out.extend(coord.analyze_smiles(smi))
    return out


def test_distribution_aggregation():
    spheres = _synthetic_spheres()
    dist = coord._distribution([s.coordination_number for s in spheres])
    assert dist == {"0": 2, "2": 2, "4": 1}
    counts = coord._coordination_class_counts(spheres)
    assert counts["empty"] == 2
    assert counts["under"] == 2
    assert counts["at_capacity"] == 1
    assert counts["over"] == 0
    # the secondary (structure-derived-OS) classification is reported too
    sec = coord._coordination_class_counts_os(spheres)
    assert sec["empty"] == 2


def test_summary_aggregation():
    spheres = _synthetic_spheres()
    summary = coord.summarize(
        spheres,
        n_rows=5,
        n_rows_parsed=5,
        n_rows_with_metal=5,
        n_rows_multi_metal=0,
        n_rows_unparseable=0,
        n_frags_column_mismatches=0,
    )
    assert summary["population"]["n_metal_atoms"] == 5
    cn = summary["coordination_number"]
    assert cn["n_rows_with_empty_sphere"] == 2
    assert cn["mean_all_rows"] == pytest.approx(8 / 5)
    # the excluding-empty statistics are reported alongside, not instead of
    assert cn["mean_excluding_empty_spheres"] == pytest.approx(8 / 3)
    assert summary["metal_alone_in_fragment"]["n_rows"] == 2
    assert summary["metal_alone_in_fragment"]["fraction"] == pytest.approx(0.4)
    assert summary["donor_motif_distribution"]["n_distinct_motifs"] == 3


def test_cross_tab_and_mechanism_aggregation():
    spheres = _synthetic_spheres()
    frac = [0.0, 0.0, 0.5, 1.0, 0.5]
    xt = coord.cross_tab(spheres, largest_frag_frac=frac)
    assert xt["by_row_n_frags"]["1"]["n_rows"] == 4
    assert xt["by_row_n_frags"]["2"]["n_rows"] == 1
    frag = xt["fragmented_vs_single_fragment"]
    assert frag["single_fragment"]["n_rows"] == 4
    assert frag["fragmented"]["n_rows"] == 1
    assert "two_proportion_z" in frag
    mech = coord.candidate_mechanism(spheres)
    assert mech["fragmented_under_or_empty"]["n_rows"] >= 1
    assert "donor_split_fully_explains_the_missing_coordination" in mech


def test_pearson_and_two_proportion_helpers():
    assert coord._pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert coord._pearson([1, 1, 1], [1, 2, 3]) is None
    assert coord._pearson([1, 2], [1, 2]) is None
    assert coord._pearson_p(0.5, 100) is not None
    assert coord._pearson_p(None, 100) is None
    z, p = coord._two_proportion_z(10, 100, 10, 100)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)
    assert coord._two_proportion_z(0, 0, 1, 1) == (None, None)


def test_median_helper():
    assert coord._median([3, 1, 2]) == 2
    assert coord._median([4, 1, 2, 3]) == pytest.approx(2.5)
    assert coord._median([]) is None


# ---------------------------------------------------------------------------
# 3. JSON schema
# ---------------------------------------------------------------------------

#: The published end-to-end headline that this artefact must NOT restate as
#: a typed-in literal.  The emitter reads it back from disk at run time; the
#: test reads the SAME source independently, so the assertion below is a
#: contract check rather than a restatement of the emitter's own output.
END_TO_END_SOURCE_METRIC = coord.END_TO_END_HEADLINE_METRIC
END_TO_END_SOURCE_PATH = coord.END_TO_END_HEADLINE_PATH


def _read_end_to_end_source():
    """The published source node, read LIVE from disk.

    Returns ``None`` when the source artefact (or the node inside it) is
    not readable, which is the case the emitter is explicitly fail-soft
    about.  The file is read directly here — NOT through the emitter's own
    helper — so that comparing it against the emitted artefact is a real
    check and not a tautology.
    """
    path = (
        PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / f"{END_TO_END_SOURCE_METRIC}.json"
    )
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    node = doc
    for key in END_TO_END_SOURCE_PATH:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, dict) else None


def _tiny_pool(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_pool.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "smiles", "n_frags_raw", "largest_frag_frac"])
        w.writerow([0, "[NH3][Pt]([NH3])(Cl)Cl", 1, 1.0])
        w.writerow([0, "Cl[Pt]Cl.N", 2, 0.75])
        w.writerow([0, "[Pt]", 1, 1.0])
    return p


def test_json_schema(tmp_path):
    pool = _tiny_pool(tmp_path)
    result = coord.analyze_pool(pool)
    artifact = coord.build_artifact(pool, result)

    for key in (
        "metric",
        "tag",
        "date",
        "wave",
        "branch",
        "purpose",
        "population",
        "what_this_is_not",
        "comparability_statement",
        "definitions",
        "summary",
        "findings",
        "cross_tab_fragmentation_vs_coordination",
        "candidate_mechanism",
        "per_row_sphere",
        "reproduce",
        "end_to_end_headline",
    ):
        assert key in artifact, f"missing top-level key: {key}"

    assert artifact["metric"] == "wfa155_metal_coordination_sphere"
    assert artifact["tag"] == "MEASURED"
    assert artifact["branch"] == "COORD"

    # the "what this is NOT" block must DEFER to its source artefacts
    # instead of typing a rate in as a literal
    nots = artifact["what_this_is_not"]
    src_rel = f"metrics/by_metric/{END_TO_END_SOURCE_METRIC}.json"
    src_jsonpath = "/" + "/".join(END_TO_END_SOURCE_PATH)

    source = _read_end_to_end_source()
    if source is None:
        # the emitter is fail-soft here: it names the source and asserts
        # no number at all
        assert artifact["end_to_end_headline"] is None
        assert src_rel in nots["not_8_51_percent"]
        assert src_jsonpath in nots["not_8_51_percent"]
        assert "not readable at run time" in nots["not_8_51_percent"]
    else:
        # the end-to-end rate is carried VERBATIM from the published
        # source, read at run time — never re-derived and never typed in
        head = artifact["end_to_end_headline"]
        assert head["source"] == src_rel
        assert head["jsonpath"] == src_jsonpath
        assert head["median"] == source["median"]
        assert head["n_runs"] == source["n"]
        # the published headline figure is the rounded percent of that same
        # live-read median (8.51 %; MEASURED at
        # metrics/by_metric/cfm_fragmentation_audit.json
        # -> /summary/generation_side/single_molecule_rate_end_to_end/median)
        assert round(source["median"] * 100, 2) == 8.51

        # ... and the sentence itself points at the source and carries NO
        # rate literal of its own
        text = nots["not_8_51_percent"]
        assert src_rel in text
        assert src_jsonpath in text
        assert "end_to_end_headline" in text
        assert not re.search(r"\d+(?:\.\d+)?\s*%", text), text
        assert f"{round(source['median'] * 100, 2)} %" not in text

    # the SINGLE-FRAGMENT fraction is this pool's OWN, re-measured at run
    # time, so the sentence carries THIS pool's rate and not a quoted one
    pop = artifact["population"]
    own_rate = pop["n_single_fragment_rows"] / pop["n_rows_analyzed"]
    assert f"{round(own_rate * 100, 2)} %" in nots["not_25_05_percent"]
    assert "of THIS pool" in nots["not_25_05_percent"]

    # the four comparability breaks must all be present
    comp = artifact["comparability_statement"]
    for key in (
        "summary",
        "break_1_denominator",
        "break_2_metric",
        "break_3_post_processing",
        "break_4_hydrogens",
    ):
        assert key in comp, f"missing comparability break: {key}"

    # the oxidation state must be labelled as an inference, not a measurement
    assert "RULE-BASED INFERENCE" in artifact["summary"]["oxidation_state"]["status"]
    for layer in ("os_formal_charge", "os_charge_balanced", "os_prior_default"):
        assert layer in artifact["summary"]["oxidation_state"]["layers"]

    # the per-row dump is columnar and its length matches the metal count
    prow = artifact["per_row_sphere"]
    assert prow["per_row_columns"] == list(coord.PER_ROW_COLUMNS)
    n_metal = artifact["summary"]["population"]["n_metal_atoms"]
    assert len(prow["per_row"]) == n_metal
    for record in prow["per_row"]:
        assert len(record) == len(coord.PER_ROW_COLUMNS)

    # it must serialise
    text = json.dumps(artifact, indent=2, ensure_ascii=False)
    assert json.loads(text)["metric"] == "wfa155_metal_coordination_sphere"


def test_findings_block_is_computed_not_hard_coded(tmp_path):
    """Every number in the findings block must come from the measurement."""
    pool = _tiny_pool(tmp_path)
    artifact = coord.build_artifact(pool, coord.analyze_pool(pool))
    findings = artifact["findings"]
    assert "findings" in artifact
    for key in (
        "finding_1_no_metal_is_over_coordinated",
        "finding_2_the_oxidation_state_is_not_in_the_smiles",
        "finding_3_fragmentation_does_NOT_significantly_predict_under_coordination",
        "finding_4_the_donor_split_mechanism_is_REFUTED_as_the_general_cause",
        "finding_5_the_metal_alone_rows_ARE_the_donor_split_mechanism",
        "finding_6_the_donor_sets_are_CARBON_DOMINANT",
    ):
        assert key in findings, f"missing finding: {key}"
        assert findings[key]["measured"], f"finding has no measured block: {key}"

    # finding 1 must agree with the summary it was computed from
    f1 = findings["finding_1_no_metal_is_over_coordinated"]["measured"]
    assert f1["over"] == artifact["summary"]["coordination_class"]["counts"]["over"]
    assert f1["n_metal_atoms"] == artifact["summary"]["population"]["n_metal_atoms"]


def test_artifact_is_deterministic(tmp_path):
    pool = _tiny_pool(tmp_path)
    a = coord.build_artifact(pool, coord.analyze_pool(pool))
    b = coord.build_artifact(pool, coord.analyze_pool(pool))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# 4. No CUDA
# ---------------------------------------------------------------------------


def test_module_imports_without_torch():
    """STATIC import check: this module's own import graph has no torch.

    ``molmetal/data/__init__.py`` transitively imports torch (through the
    featuriser stack), so importing ANY ``molmetal.data.*`` submodule pulls
    torch into the process.  That is a repo-wide package property, not
    something this branch owns, and it is NOT a GPU usage.  What this branch
    owns is that ITS OWN module never imports torch and never names a CUDA
    entry point, so the check is done on the module's own import statements
    via the AST rather than on the whole interpreter's sys.modules.
    """
    import ast

    path = PROJECT_ROOT / "molmetal/scripts/wf_a155_metal_coordination_sphere.py"
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
        "argparse",
        "csv",
        "json",
        "math",
        "os",
        "sys",
        "collections",
        "dataclasses",
        "pathlib",
        "typing",
        "__future__",
        "rdkit",
        "molmetal",
    }, f"unexpected imports: {sorted(roots)}"

    # no CUDA entry point is named anywhere in the module body
    body = src.replace("CUDA_VISIBLE_DEVICES", "")
    for banned in ("torch.cuda", "cuda(", ".cuda", "device=", "set_device"):
        assert banned not in body, f"module names a CUDA entry point: {banned}"


def test_subprocess_import_forces_cpu_and_sees_no_gpu():
    """A fresh interpreter: CUDA_VISIBLE_DEVICES='' and no GPU is visible."""
    code = (
        "import sys, os;"
        "sys.path.insert(0, %r);"
        "import molmetal.scripts.wf_a155_metal_coordination_sphere as m;"
        "print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')));"
        "import torch;"
        "print(bool(torch.cuda.is_available()))" % str(PROJECT_ROOT)
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
# 5. The published pool (integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not POOL.is_file(), reason="published pool not on disk")
def test_published_pool_headline_counts():
    result = coord.analyze_pool(POOL)
    summary = result["summary"]
    pop = summary["population"]
    assert pop["n_rows"] == 2112
    assert pop["n_rows_parsed"] == 2112
    assert pop["n_rows_unparseable"] == 0
    assert pop["n_rows_with_at_least_one_metal"] == 2112
    assert pop["n_metal_atoms"] == 2112
    assert pop["n_rows_multi_metal"] == 0
    # the pool's own n_frags_raw column agrees with a fresh RDKit count
    assert pop["n_frags_column_mismatches"] == 0

    cn = summary["coordination_number"]["distribution_all_rows"]
    assert cn == {"0": 30, "1": 95, "2": 117, "3": 111, "4": 287, "5": 147, "6": 1325}
    assert summary["metal_alone_in_fragment"]["n_rows"] == 30

    # every metal carries formal charge 0 and the rows carry no charge
    # information at all -> the oxidation state is a PRIOR, not a measurement
    ox = summary["oxidation_state"]
    assert ox["n_metal_atoms_with_a_nonzero_formal_charge"] == 0
    assert ox["n_rows_where_the_smiles_carries_no_charge_information"] == 2112

    # no metal in the pool is OVER-coordinated relative to its usual
    # oxidation state's capacity
    assert summary["coordination_class"]["counts"]["over"] == 0


@pytest.mark.skipif(not POOL.is_file(), reason="published pool not on disk")
def test_published_pool_has_no_dative_bonds():
    """The pool's metal-incident bonds are covalent; no '->' dative bond."""
    from rdkit import Chem

    with POOL.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_dative = 0
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                n_dative += 1
    assert n_dative == 0
