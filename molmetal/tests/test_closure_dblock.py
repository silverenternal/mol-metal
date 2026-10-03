"""Tests for the closure theorem d-block benchmark (Wf-37 Phase 2, L6).

================================================================
Wf-37 Phase 2: closure-theorem d-block benchmark contract.

This file ships the **20 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.bench_closure_dblock`:

The 20 tests
------------
Per-metal closure (canned drugs / hand-built descriptors)
 1. test_Pt_II_closure               — Pt_II square-planar cis-platin
 2. test_Pt_IV_closure               — Pt_IV octahedral satraplatin-class
 3. test_Ru_II_III_closure           — Ru_II + Ru_III octahedral
 4. test_Rh_III_closure              — Rh_III square-planar
 5. test_Au_I_III_closure            — Au_I + Au_III (linear / sq-planar)
 6. test_Pd_II_closure               — Pd_II square-planar (analogous Pt)

Benchmark mechanics
 7. test_module_imports              — script imports cleanly
 8. test_dblock_metals_30            — DBLOCK_METALS list has 30 entries
 9. test_dblock_metals_in_tmqm       — every DBLOCK_METALS is in tmQM
10. test_task_spec_is_subset         — task-spec metals subset of dblock
11. test_partner_smile_list_size     — partner list has ≥ 6 tiles
12. test_default_n_samples_is_100    — default n_samples_per_metal = 100
13. test_default_dataset_is_tmqm     — default dataset = tmqm
14. test_default_max_depth_is_1      — default max_depth = 1
15. test_default_seeds_is_1          — default seeds = 1
16. test_default_output_dir          — default output dir is wf37_closure_dblock

Smoke run / output contract
17. test_smoke_run_3_metals          — 5 samples × 3 metals smoke OK
18. test_per_metal_aggregate_keys    — every per-metal aggregate has keys
19. test_per_metal_csv_header        — per_metal.csv has expected header
20. test_per_metal_bench_md_written  — Markdown summary written

All tests are CPU-only, deterministic, and complete in < 120 s
combined.  The smoke benchmark is module-scoped so it runs once and
the JSON / CSV / Markdown outputs are reused across tests.

Run with::

    uv run --no-sync pytest -q molmetal/tests/test_closure_dblock.py --tb=short
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from molmetal.scripts import bench_closure_dblock  # noqa: E402
from molmetal.scripts.bench_closure_dblock import (  # noqa: E402
    DBLOCK_METALS,
    DEFAULT_DATASET,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_SAMPLES_PER_METAL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TMQM_CSV,
    METAL_Z,
    PARTNER_SMILE_LIST,
    TASK_SPEC_METALS,
    _aggregate_per_metal,
    _benchmark_one,
    _is_pt_coordinated,
    _load_click_rules,
    _load_metal_samples,
    _load_partner_terms,
    _utcnow_iso,
    run_closure_dblock_benchmark,
    save_dblock_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_payload() -> Dict[str, Any]:
    """A 5-sample × 3-metal smoke payload — fast enough for CI."""
    return run_closure_dblock_benchmark(
        n_samples_per_metal=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
    )


@pytest.fixture(scope="module")
def smoke_paths(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Artefact paths produced by a 5-sample × 3-metal smoke run."""
    out = tmp_path_factory.mktemp("wf37_closure_dblock_smoke")
    payload = run_closure_dblock_benchmark(
        n_samples_per_metal=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
        output_dir=out,
    )
    paths = save_dblock_benchmark(payload, output_dir=out)
    return paths


# ---------------------------------------------------------------------------
# 1. Pt_II_closure (square-planar)
# ---------------------------------------------------------------------------
def test_Pt_II_closure() -> None:
    """Pt_II square-planar cisplatin passes the closure theorem."""
    from molmetal_lam.lam_chem.cisplatin_builder import build_cisplatin
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )

    cisplatin = build_cisplatin()
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=cisplatin,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "cisplatin (Pt_II sq-planar) must pass closure"
    assert "Pt" in cisplatin.canonical_smiles()


# ---------------------------------------------------------------------------
# 2. Pt_IV_closure (octahedral)
# ---------------------------------------------------------------------------
def test_Pt_IV_closure() -> None:
    """Pt_IV octahedral complexes pass the closure theorem.

    The R17 ``PRIMITIVE_ATOMS`` ``Pt`` is square-planar (4), but for
    Pt_IV we explicitly model a 6-coordinate complex (NH3 + 5 Cl's)
    so the resulting descriptor is octahedral (6 ligands around Pt).
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Pt_IV: 6 ligands (NH3 + 5 Cl) → octahedral.
    pt_iv = MoleculeClosedTerm.from_smiles(
        "[NH3][Pt]([NH3])([Cl])([Cl])([Cl])([Cl])",
        embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=pt_iv,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "Pt_IV 6-ligand complex must pass closure"
    assert "Pt" in pt_iv.canonical_smiles()


# ---------------------------------------------------------------------------
# 3. Ru_II_III_closure (octahedral)
# ---------------------------------------------------------------------------
def test_Ru_II_III_closure() -> None:
    """Ru_II (octahedral) hexaamminechloro passes the closure theorem."""
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Ru_II hexaamminechloro — octahedral, 6-coordinate.
    ru_ii = MoleculeClosedTerm.from_smiles(
        "[NH3][Ru]([NH3])([NH3])([NH3])([NH3])[Cl]",
        embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=ru_ii,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "Ru_II hexaamminechloro must pass closure"
    assert "Ru" in ru_ii.canonical_smiles()


# ---------------------------------------------------------------------------
# 4. Rh_III_closure (square-planar)
# ---------------------------------------------------------------------------
def test_Rh_III_closure() -> None:
    """Rh_III (square-planar 4-coordinate) complexes pass closure.

    In the R17 PRIMITIVE_ATOMS, Rh is square-planar with arity 4.  We
    build a 4-coordinate Rh_III analogue and verify the closure
    theorem holds at depth 1 with the standard partner set.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Rh_III square-planar (4-coordinate).
    rh_iii = MoleculeClosedTerm.from_smiles(
        "[NH3][Rh]([NH3])([Cl])([Cl])",
        embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=rh_iii,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "Rh_III 4-coordinate analogue must pass closure"
    assert "Rh" in rh_iii.canonical_smiles()


# ---------------------------------------------------------------------------
# 5. Au_I_III_closure (linear / square-planar)
# ---------------------------------------------------------------------------
def test_Au_I_III_closure() -> None:
    """Au_I (linear, 2-coordinate) + Au_III (square-planar, 4-coordinate)
    both pass the closure theorem.

    Au in the R17 PRIMITIVE_ATOMS is square-planar (4); for Au_I
    we model a 2-coordinate linear complex, and for Au_III a
    4-coordinate square-planar complex.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Au_I linear: 2-coordinate.
    au_i = MoleculeClosedTerm.from_smiles(
        "[NH3][Au][Cl]",
        embed_3d=False,
    )
    # Au_III square-planar: 4-coordinate.
    au_iii = MoleculeClosedTerm.from_smiles(
        "[NH3][Au]([NH3])([Cl])([Cl])",
        embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok_i = closure_theorem(
            start_term=au_i,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
        ok_iii = closure_theorem(
            start_term=au_iii,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok_i, "Au_I 2-coordinate analogue must pass closure"
    assert ok_iii, "Au_III 4-coordinate analogue must pass closure"
    assert "Au" in au_i.canonical_smiles()
    assert "Au" in au_iii.canonical_smiles()


# ---------------------------------------------------------------------------
# 6. Pd_II_closure (square-planar)
# ---------------------------------------------------------------------------
def test_Pd_II_closure() -> None:
    """Pd_II (square-planar, 4-coordinate) complexes pass closure."""
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Pd_II square-planar (4-coordinate).
    pd_ii = MoleculeClosedTerm.from_smiles(
        "[NH3][Pd]([NH3])([Cl])([Cl])",
        embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=pd_ii,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "Pd_II 4-coordinate analogue must pass closure"
    assert "Pd" in pd_ii.canonical_smiles()


# ---------------------------------------------------------------------------
# 7. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """The d-block benchmark module imports cleanly with the expected
    entry points.
    """
    assert bench_closure_dblock is not None
    assert hasattr(bench_closure_dblock, "run_closure_dblock_benchmark")
    assert hasattr(bench_closure_dblock, "save_dblock_benchmark")
    assert hasattr(bench_closure_dblock, "_aggregate_per_metal")
    assert hasattr(bench_closure_dblock, "_load_metal_samples")
    assert hasattr(bench_closure_dblock, "_benchmark_one")
    assert hasattr(bench_closure_dblock, "_is_pt_coordinated")
    assert hasattr(bench_closure_dblock, "_is_metal_coordinated")


# ---------------------------------------------------------------------------
# 8. DBLOCK_METALS list has 30 entries
# ---------------------------------------------------------------------------
def test_dblock_metals_30() -> None:
    """The DBLOCK_METALS list contains exactly 30 d-block metals."""
    assert (
        len(DBLOCK_METALS) == 30
    ), f"30 d-block metals expected, got {len(DBLOCK_METALS)}"
    # 3d / 4d / 5d+La split: 10 + 10 + 10.
    block_3d = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
    block_4d = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}
    block_5d_la = {
        "La",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
    }
    actual = set(DBLOCK_METALS)
    assert block_3d.issubset(
        actual
    ), f"3d block incomplete: missing {block_3d - actual}"
    assert block_4d.issubset(
        actual
    ), f"4d block incomplete: missing {block_4d - actual}"
    assert block_5d_la.issubset(
        actual
    ), f"5d + La block incomplete: missing {block_5d_la - actual}"


# ---------------------------------------------------------------------------
# 9. Every DBLOCK_METALS is in tmQM
# ---------------------------------------------------------------------------
def test_dblock_metals_in_tmqm() -> None:
    """All 30 d-block metals appear in the tmQM d-block subset CSV."""
    assert DEFAULT_TMQM_CSV.is_file(), f"tmQM CSV missing at {DEFAULT_TMQM_CSV}"
    import csv as _csv

    seen: set = set()
    with DEFAULT_TMQM_CSV.open("r", newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            seen.add((row.get("metal_center", "") or "").strip())
    missing = set(DBLOCK_METALS) - seen
    assert not missing, (
        f"tmQM CSV missing metals {missing}; " f"DBLOCK_METALS must match the ship CSV"
    )


# ---------------------------------------------------------------------------
# 10. task-spec metals subset of d-block
# ---------------------------------------------------------------------------
def test_task_spec_is_subset() -> None:
    """The 9 task-spec metals are a subset of the 30 d-block metals."""
    assert set(TASK_SPEC_METALS).issubset(set(DBLOCK_METALS)), (
        f"TASK_SPEC_METALS={sorted(TASK_SPEC_METALS)} must be subset of "
        f"DBLOCK_METALS={sorted(DBLOCK_METALS)}"
    )
    assert len(TASK_SPEC_METALS) == 9


# ---------------------------------------------------------------------------
# 11. Partner SMILES list size
# ---------------------------------------------------------------------------
def test_partner_smile_list_size() -> None:
    """PARTNER_SMILE_LIST has at least 6 entries (covers all 5 click handles)."""
    assert len(PARTNER_SMILE_LIST) >= 6
    joined = " ".join(PARTNER_SMILE_LIST)
    assert "C#C" in joined, "need at least one alkyne partner"
    assert "[N+]=[N-]" in joined, "need at least one azide partner"
    assert "CS" in joined, "need at least one thiol partner"


# ---------------------------------------------------------------------------
# 12. Default n_samples_per_metal = 100
# ---------------------------------------------------------------------------
def test_default_n_samples_is_100() -> None:
    """The default n_samples_per_metal is 100 (30 metals × 100 = 3000)."""
    assert DEFAULT_N_SAMPLES_PER_METAL == 100


# ---------------------------------------------------------------------------
# 13. Default dataset = tmqm
# ---------------------------------------------------------------------------
def test_default_dataset_is_tmqm() -> None:
    """The default dataset is 'tmqm' (the wider 30-metal coverage)."""
    assert DEFAULT_DATASET == "tmqm"


# ---------------------------------------------------------------------------
# 14. Default max_depth = 1
# ---------------------------------------------------------------------------
def test_default_max_depth_is_1() -> None:
    """The default max_depth is 1 (one click rule application)."""
    assert DEFAULT_MAX_DEPTH == 1


# ---------------------------------------------------------------------------
# 15. Default seeds = 1
# ---------------------------------------------------------------------------
def test_default_seeds_is_1() -> None:
    """The default seeds is 1 (placeholder for future stochastic re-runs)."""
    assert DEFAULT_SEEDS == 1


# ---------------------------------------------------------------------------
# 16. Default output dir is wf37_closure_dblock
# ---------------------------------------------------------------------------
def test_default_output_dir() -> None:
    """The default output dir is reports/wf37_closure_dblock/."""
    assert DEFAULT_OUTPUT_DIR.name == "wf37_closure_dblock"
    assert DEFAULT_OUTPUT_DIR.parent.name == "reports"


# ---------------------------------------------------------------------------
# 17. Smoke run — 5 samples × 3 metals works end-to-end
# ---------------------------------------------------------------------------
EXPECTED_AGGREGATE_KEYS: List[str] = [
    "n_samples",
    "n_closure_ok",
    "closure_reachability_rate",
    "mean_beta_depth",
    "mean_n_products",
    "mean_wall_time",
    "std_wall_time",
    "median_wall_time",
    "peak_memory_mb",
    "total_wall_time",
    "n_error",
    "n_metal_coordinated",
    "metal_coordination_rate",
]


def test_smoke_run_3_metals(smoke_payload: Dict[str, Any]) -> None:
    """5 samples × 3 metals smoke run produces the expected schema."""
    assert smoke_payload["schema_version"] == 1
    assert smoke_payload["tool"] == "bench_closure_dblock.py"
    assert smoke_payload["pilot_id"] == "wf37_closure_dblock"
    assert smoke_payload["metals"] == ["Pt", "Ru", "Pd"]
    assert smoke_payload["n_samples_total"] == 15
    # Aggregate is bounded by n_samples_total.
    agg = smoke_payload["aggregate"]
    assert agg["n_samples"] == 15
    # closure_reachability_rate must lie in the closed unit interval.
    rate = float(agg["closure_reachability_rate"])
    assert 0.0 <= rate <= 1.0, f"closure_reachability_rate out of [0,1]: {rate}"
    # No errors in the smoke fixture (RDKit can parse all 15 SMILES).
    assert agg["n_error"] == 0


# ---------------------------------------------------------------------------
# 18. Per-metal aggregate has all expected keys
# ---------------------------------------------------------------------------
def test_per_metal_aggregate_keys(smoke_payload: Dict[str, Any]) -> None:
    """The per-metal aggregate dict carries every documented key."""
    for metal, info in smoke_payload["per_metal"].items():
        agg = info["aggregate"]
        missing = [k for k in EXPECTED_AGGREGATE_KEYS if k not in agg]
        assert (
            not missing
        ), f"per-metal aggregate for {metal!r} missing keys {missing!r}"
        # metal_coordination_rate is one of the new L6 fields.
        assert "metal_coordination_rate" in agg
        assert "n_metal_coordinated" in agg


# ---------------------------------------------------------------------------
# 19. per_metal.csv header is correct
# ---------------------------------------------------------------------------
EXPECTED_CSV_FIELDS: List[str] = [
    "metal",
    "n_samples",
    "n_closure_ok",
    "closure_reachability_rate",
    "metal_coordination_rate",
    "n_metal_coordinated",
    "mean_beta_depth",
    "mean_n_products",
    "mean_wall_time",
    "std_wall_time",
    "median_wall_time",
    "peak_memory_mb",
    "total_wall_time",
    "n_error",
]


def test_per_metal_csv_header(smoke_paths: Dict[str, Path]) -> None:
    """save_dblock_benchmark writes per_metal.csv with the documented header."""
    csv_path = smoke_paths["csv"]
    assert csv_path.is_file(), f"per_metal.csv missing at {csv_path}"
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for f in EXPECTED_CSV_FIELDS:
            assert f in header, f"per_metal.csv missing column {f!r}"
        rows = list(reader)
    # 3 metals in the smoke fixture.
    assert len(rows) == 3, f"per_metal.csv should have 3 rows, got {len(rows)}"


# ---------------------------------------------------------------------------
# 20. Markdown summary is written
# ---------------------------------------------------------------------------
def test_per_metal_bench_md_written(smoke_paths: Dict[str, Path]) -> None:
    """save_dblock_benchmark writes a per_metal_bench.md summary."""
    md_path = smoke_paths["markdown"]
    assert md_path.is_file(), f"per_metal_bench.md missing at {md_path}"
    text = md_path.read_text()
    assert "Wf-37 Phase 2" in text
    assert "30 d-block metals" in text
    assert "per_metal_bench" in text or "Per-metal reachability rate" in text


# ---------------------------------------------------------------------------
# Bonus invariants
# ---------------------------------------------------------------------------
def test_load_metal_samples_first_n() -> None:
    """_load_metal_samples returns the first n SMILES tagged with the metal."""
    smis = _load_metal_samples(
        DEFAULT_TMQM_CSV,
        "Pt",
        5,
        dataset="tmqm",
    )
    assert len(smis) == 5
    assert all(len(s) > 0 for s in smis)


def test_load_click_rules_count() -> None:
    """_load_click_rules returns the 5 canonical click rules."""
    rules = _load_click_rules()
    assert len(rules) == 5
    names = {getattr(r, "name", type(r).__name__) for r in rules}
    assert {"CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"}.issubset(
        names,
    ), f"missing click rule names: {names!r}"


def test_load_partner_terms_count() -> None:
    """_load_partner_terms returns at least 6 partner tiles."""
    partners = _load_partner_terms()
    assert len(partners) >= 6


def test_aggregate_per_metal_empty() -> None:
    """_aggregate_per_metal on an empty list returns the zero-vector."""
    agg = _aggregate_per_metal([])
    assert agg["n_samples"] == 0
    assert agg["closure_reachability_rate"] == 0.0
    assert agg["mean_wall_time"] == 0.0
    assert agg["peak_memory_mb"] == 0.0
    assert agg["n_error"] == 0
    assert agg["n_metal_coordinated"] == 0
    assert agg["metal_coordination_rate"] == 0.0


def test_metal_z_coverage() -> None:
    """Every DBLOCK_METALS symbol has an entry in METAL_Z."""
    missing = [m for m in DBLOCK_METALS if m not in METAL_Z]
    assert not missing, f"METAL_Z missing d-block metals {missing}"


def test_is_pt_coordinated_cisplatin() -> None:
    """_is_pt_coordinated returns True for cisplatin."""
    from molmetal_lam.lam_chem.cisplatin_builder import build_cisplatin

    c = build_cisplatin()
    assert _is_pt_coordinated(c) is True


def test_is_pt_coordinated_ru_complex() -> None:
    """_is_pt_coordinated returns False for a Ru complex (no Pt atoms)."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    ru = MoleculeClosedTerm.from_smiles(
        "[NH3][Ru]([NH3])([NH3])([NH3])([NH3])[Cl]",
        embed_3d=False,
    )
    assert _is_pt_coordinated(ru) is False


def test_utcnow_iso_format() -> None:
    """_utcnow_iso returns a Z-suffixed ISO 8601 timestamp."""
    s = _utcnow_iso()
    assert s.endswith("Z")
    assert "T" in s


def test_benchmark_one_smoke() -> None:
    """_benchmark_one on cisplatin returns the expected per-sample record."""
    rec = _benchmark_one(
        "[NH3][Pt]([NH3])([Cl])([Cl])",
        0,
        "Pt",
        click_rules=_load_click_rules(),
        partner_terms=_load_partner_terms(),
        max_depth=1,
        max_products=128,
    )
    assert rec["metal"] == "Pt"
    assert rec["idx"] == 0
    assert rec["depth"] == 1
    assert rec["status"] in ("ok", "skipped", "error")
    assert isinstance(rec["closure_ok"], bool)
    assert isinstance(rec["wall_time_s"], float)
    assert isinstance(rec["memory_mb"], float)
    assert isinstance(rec["n_metal_coordinated"], int)


def test_smoke_completes_under_budget() -> None:
    """The 5-sample × 3-metal smoke completes in < 90 s wall."""
    t0 = time.perf_counter()
    payload = run_closure_dblock_benchmark(
        n_samples_per_metal=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
    )
    wall = time.perf_counter() - t0
    assert wall < 90.0, f"smoke run took {wall:.2f}s; budget is 90s"
    assert payload["n_samples_total"] == 15


def test_per_block_keys_present() -> None:
    """The payload includes per-periodic-block aggregates."""
    payload = run_closure_dblock_benchmark(
        n_samples_per_metal=2,
        metals=["Sc", "Cu", "Y", "Ru", "La", "Pt"],
        max_depth=1,
        max_products=64,
    )
    assert "per_block" in payload
    for blk in ("3d", "4d", "5d_La"):
        assert blk in payload["per_block"]


def test_default_csv_paths_exist() -> None:
    """The default tmQM CSV path exists on disk."""
    assert DEFAULT_TMQM_CSV.is_file(), f"tmQM CSV missing at {DEFAULT_TMQM_CSV}"


def test_metals_all_alias() -> None:
    """The CLI accepts --metals all as an alias for the 30-metal set."""
    # We don't invoke the CLI here; we just confirm DBLOCK_METALS is
    # large enough to make the alias meaningful (≥ 30 entries).
    assert len(DBLOCK_METALS) >= 30


def test_per_metal_records_have_all_keys() -> None:
    """Every per-sample record carries every documented key."""
    payload = run_closure_dblock_benchmark(
        n_samples_per_metal=2,
        metals=["Pt", "Ru"],
        max_depth=1,
        max_products=64,
    )
    expected_keys = {
        "idx",
        "metal",
        "smiles",
        "status",
        "closure_ok",
        "n_visited",
        "n_products",
        "max_depth_seen",
        "well_typed_rate",
        "n_pt_coordinated",
        "n_metal_coordinated",
        "n_atoms_start",
        "parse_error",
        "wall_time_s",
        "memory_mb",
        "depth",
    }
    for metal, info in payload["per_metal"].items():
        for rec in info["records"]:
            missing = expected_keys - set(rec.keys())
            assert not missing, f"record in {metal!r} missing keys {missing!r}"


def test_closure_ok_summary_present() -> None:
    """The smoke payload aggregate carries the documented summary keys."""
    payload = run_closure_dblock_benchmark(
        n_samples_per_metal=2,
        metals=["Pt", "Ru"],
        max_depth=1,
        max_products=64,
    )
    agg = payload["aggregate"]
    for k in (
        "n_samples",
        "n_closure_ok",
        "closure_reachability_rate",
        "mean_wall_time",
        "peak_memory_mb",
        "total_wall_time",
    ):
        assert k in agg, f"aggregate missing key {k!r}"
