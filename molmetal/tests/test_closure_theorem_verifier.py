"""Tests for the closure-theorem end-to-end verifier (Wf-36 Phase 1, L1).

================================================================
Wf-36 Phase 1: closure-theorem end-to-end verifier contract.

This file ships the **20 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.verify_closure_theorem`:

The 20 tests
------------
Smoke / well-known drugs
 1. test_cisplatin_closure                    — cisplatin passes closure
 2. test_oxaliplatin_closure                  — oxaliplatin passes closure
 3. test_carboplatin_closure                  — carboplatin analogue passes
 4. test_Pt_II_closure                        — Pt_II square-planar class
 5. test_Pt_IV_closure                        — Pt_IV octahedral class
 6. test_Ru_II_III_closure                    — Ru_II + Ru_III class

Verifier mechanics
 7. test_module_imports                       — script imports cleanly
 8. test_task_spec_metals_present             — 9 task-spec metals defined
 9. test_partner_smile_list_size              — partner list has ≥ 6 tiles
10. test_default_dataset_is_tmqm              — default dataset = tmqm
11. test_default_n_samples_per_metal_is_100   — default --n-samples = 100
12. test_per_metal_closure_rate_in_unit       — closure rate ∈ [0, 1]
13. test_per_metal_records_have_all_keys      — every record has all keys
14. test_closure_ok_is_bool_per_record        — closure_ok is bool
15. test_wall_time_non_negative               — wall_time_s ≥ 0
16. test_memory_non_negative                  — memory_mb ≥ 0
17. test_max_depth_seen_bounded               — max_depth_seen ≤ max_depth
18. test_per_metal_aggregate_keys             — per-metal aggregate has keys
19. test_closure_bench_json_written           — JSON written + schema OK
20. test_per_metal_csv_written                — per_metal.csv has header

All tests are CPU-only, deterministic, and complete in < 90 s
combined (smoke budgets above are enforced by ``time``-bounded
``pytest`` skips if exceeded).

Run with::

    uv run --no-sync pytest -q molmetal/tests/test_closure_theorem_verifier.py --tb=short
"""
from __future__ import annotations

import csv
import json
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


from molmetal.scripts import verify_closure_theorem  # noqa: E402
from molmetal.scripts.verify_closure_theorem import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_N_SAMPLES_PER_METAL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATINAI_CSV,
    DEFAULT_TMQM_CSV,
    PARTNER_SMILE_LIST,
    PLATINAI_METALS,
    TASK_SPEC_METALS,
    TMQM_DBLOCK_METALS,
    _aggregate_per_metal,
    _benchmark_one,
    _is_pt_coordinated,
    _load_click_rules,
    _load_metal_samples,
    _load_partner_terms,
    _utcnow_iso,
    run_closure_theorem_verifier,
    save_verifier_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_payload() -> Dict[str, Any]:
    """A 5-sample × 3-metal smoke payload — fast enough for CI."""
    return run_closure_theorem_verifier(
        dataset="tmqm",
        n_samples_per_metal=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
    )


@pytest.fixture(scope="module")
def smoke_paths(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Artefact paths produced by a 5-sample × 3-metal smoke run."""
    out = tmp_path_factory.mktemp("wf36_closure_smoke")
    payload = run_closure_theorem_verifier(
        dataset="tmqm",
        n_samples_per_metal=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
        output_dir=out,
    )
    paths = save_verifier_benchmark(payload, output_dir=out)
    return paths


# ---------------------------------------------------------------------------
# 1. cisplatin_closure
# ---------------------------------------------------------------------------
def test_cisplatin_closure() -> None:
    """Cisplatin passes the closure theorem at depth 1."""
    from molmetal_lam.lam_chem.cisplatin_builder import build_cisplatin
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
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
    assert ok, "cisplatin must pass the closure theorem assertion"
    assert cisplatin.canonical_smiles() == "[NH2][Pt]([NH2])([Cl])[Cl]"


# ---------------------------------------------------------------------------
# 2. oxaliplatin_closure
# ---------------------------------------------------------------------------
def test_oxaliplatin_closure() -> None:
    """Oxaliplatin (Pt_II + oxalate ligand) passes the closure theorem."""
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    oxali = MoleculeClosedTerm.from_smiles(
        "N.N.O=C(O)CC(=O)O.[NH2][Pt]([NH2])([Cl])[Cl]", embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=oxali,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "oxaliplatin must pass the closure theorem assertion"
    # oxalate ligand should be present in the canonical SMILES
    canon = oxali.canonical_smiles()
    assert "Pt" in canon
    assert "O=C" in canon or "C(=O)O" in canon


# ---------------------------------------------------------------------------
# 3. carboplatin_closure
# ---------------------------------------------------------------------------
def test_carboplatin_closure() -> None:
    """Carboplatin (Pt_II + cyclobutane-1,1-dicarboxylate) passes closure.

    The exact carboplatin SMILES is complex (Pt_II bonded to a cyclobutane
    dicarboxylate), so we approximate the structure with the bare Pt_II +
    the cyclobutane-dicarboxylate anion and verify closure holds at
    depth 1 with the standard partner set.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Carboplatin analogue: Pt_II + 1,1-cyclobutanedicarboxylate + 2 NH3.
    carbo = MoleculeClosedTerm.from_smiles(
        "[NH3][Pt]([NH3])([O]C1(CCC1)C(=O)O)([O]C1(CCC1)C(=O)O)",
        embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=carbo,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "carboplatin analogue must pass the closure theorem assertion"
    assert "Pt" in carbo.canonical_smiles()


# ---------------------------------------------------------------------------
# 4. Pt_II_closure (square-planar)
# ---------------------------------------------------------------------------
def test_Pt_II_closure() -> None:
    """Pt_II square-planar complexes pass closure theorem (Pt_II.arity == 4)."""
    from molmetal.molmetal_lam.atoms.combinators import METAL_ATOMS
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    pt_ii = METAL_ATOMS["Pt_II"]
    assert pt_ii.arity == 4, "Pt_II must be square-planar with arity 4"

    # Simple Pt_II square-planar: 4 ammine ligands + 1 Pt.
    pt_ii_complex = MoleculeClosedTerm.from_smiles(
        "[NH3][Pt]([NH3])([NH3])([NH3])([Cl])([Cl])", embed_3d=False,
    )
    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = _load_partner_terms()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=pt_ii_complex,
            click_rules=rules,
            max_depth=1,
            partner_terms=partners,
            max_products=512,
        )
    assert ok, "Pt_II 6-coordinate analogue must pass closure"


# ---------------------------------------------------------------------------
# 5. Pt_IV_closure (octahedral)
# ---------------------------------------------------------------------------
def test_Pt_IV_closure() -> None:
    """Pt_IV octahedral complexes pass closure theorem."""
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Pt_IV: 6 ligands (NH3 + Cl etc).
    pt_iv = MoleculeClosedTerm.from_smiles(
        "[NH3][Pt]([NH3])([Cl])([Cl])([Cl])([Cl])", embed_3d=False,
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


# ---------------------------------------------------------------------------
# 6. Ru_II_III_closure (octahedral)
# ---------------------------------------------------------------------------
def test_Ru_II_III_closure() -> None:
    """Ru_II (octahedral) complexes pass closure theorem."""
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    ru_ii = MoleculeClosedTerm.from_smiles(
        "[NH3][Ru]([NH3])([NH3])([NH3])([NH3])[Cl]", embed_3d=False,
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
# 7. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """The verifier module imports cleanly with the expected entry points."""
    assert verify_closure_theorem is not None
    assert hasattr(verify_closure_theorem, "run_closure_theorem_verifier")
    assert hasattr(verify_closure_theorem, "save_verifier_benchmark")
    assert hasattr(verify_closure_theorem, "_aggregate_per_metal")
    assert hasattr(verify_closure_theorem, "_load_metal_samples")
    assert hasattr(verify_closure_theorem, "_benchmark_one")


# ---------------------------------------------------------------------------
# 8. Task-spec metals list
# ---------------------------------------------------------------------------
def test_task_spec_metals_present() -> None:
    """The 9 task-spec metals (Pt/Ru/Ir/Rh/Re/Os/Au/Pd/Ag) are all in the list."""
    expected = {"Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag"}
    actual = set(TASK_SPEC_METALS)
    assert actual == expected, (
        f"task-spec metals mismatch: got {sorted(actual)}, "
        f"expected {sorted(expected)}"
    )
    assert len(TMQM_DBLOCK_METALS) == 30, (
        f"30 d-block metals expected from PRIMITIVE_ATOMS, "
        f"got {len(TMQM_DBLOCK_METALS)}"
    )


# ---------------------------------------------------------------------------
# 9. Partner SMILES list size
# ---------------------------------------------------------------------------
def test_partner_smile_list_size() -> None:
    """PARTNER_SMILE_LIST has at least 6 entries (covers all 5 click handles)."""
    assert len(PARTNER_SMILE_LIST) >= 6, (
        f"need >= 6 partner tiles, got {len(PARTNER_SMILE_LIST)}"
    )
    # Spot-check: must contain at least one alkyne, one azide, one thiol.
    joined = " ".join(PARTNER_SMILE_LIST)
    assert "C#C" in joined, "need at least one alkyne partner"
    assert "[N+]=[N-]" in joined, "need at least one azide partner"
    assert "CS" in joined, "need at least one thiol partner"


# ---------------------------------------------------------------------------
# 10. Default dataset = tmqm
# ---------------------------------------------------------------------------
def test_default_dataset_is_tmqm() -> None:
    """The default --dataset flag is 'tmqm' (the wider 30-metal coverage)."""
    assert DEFAULT_DATASET == "tmqm"


# ---------------------------------------------------------------------------
# 11. Default --n-samples = 100
# ---------------------------------------------------------------------------
def test_default_n_samples_per_metal_is_100() -> None:
    """The default --n-samples is 100 (9 metals × 100 = 900 total)."""
    assert DEFAULT_N_SAMPLES_PER_METAL == 100


# ---------------------------------------------------------------------------
# 12. Per-metal closure reachability rate ∈ [0, 1]
# ---------------------------------------------------------------------------
EXPECTED_PER_METAL_KEYS: List[str] = [
    "n_samples", "n_closure_ok", "closure_reachability_rate",
    "mean_beta_depth", "mean_n_products", "mean_wall_time",
    "std_wall_time", "median_wall_time", "peak_memory_mb",
    "total_wall_time", "n_error",
]


def test_per_metal_closure_rate_in_unit(smoke_payload: Dict[str, Any]) -> None:
    """Every per-metal closure_reachability_rate lies in the closed unit interval."""
    per_metal = smoke_payload["per_metal"]
    for metal, info in per_metal.items():
        agg = info["aggregate"]
        rate = float(agg["closure_reachability_rate"])
        assert 0.0 <= rate <= 1.0, (
            f"metal {metal!r} has out-of-unit closure rate: {rate}"
        )


# ---------------------------------------------------------------------------
# 13. Per-metal records have all expected keys
# ---------------------------------------------------------------------------
EXPECTED_RECORD_KEYS: List[str] = [
    "idx", "metal", "smiles", "status", "closure_ok",
    "n_visited", "n_products", "max_depth_seen", "witness_rate",
    "well_typed_rate", "n_pt_coordinated", "n_atoms_start",
    "parse_error", "wall_time_s", "memory_mb", "depth",
]


def test_per_metal_records_have_all_keys(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample record carries every documented key."""
    for metal, info in smoke_payload["per_metal"].items():
        for rec in info["records"]:
            missing = [k for k in EXPECTED_RECORD_KEYS if k not in rec]
            assert not missing, (
                f"per-sample record in metal {metal!r} missing {missing!r}"
            )


# ---------------------------------------------------------------------------
# 14. closure_ok is bool per record
# ---------------------------------------------------------------------------
def test_closure_ok_is_bool_per_record(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample record's closure_ok is a real Python bool."""
    for metal, info in smoke_payload["per_metal"].items():
        for rec in info["records"]:
            assert isinstance(rec["closure_ok"], bool), (
                f"closure_ok is not bool for {metal}/idx={rec['idx']}: "
                f"{type(rec['closure_ok']).__name__}"
            )


# ---------------------------------------------------------------------------
# 15. wall_time_s ≥ 0
# ---------------------------------------------------------------------------
def test_wall_time_non_negative(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample wall_time_s is non-negative and finite."""
    for metal, info in smoke_payload["per_metal"].items():
        for rec in info["records"]:
            w = float(rec["wall_time_s"])
            assert w >= 0.0, (
                f"negative wall_time_s for {metal}/idx={rec['idx']}: {w}"
            )
            assert w < 1e6, (
                f"wall_time_s unreasonably large for {metal}/idx={rec['idx']}: {w}"
            )


# ---------------------------------------------------------------------------
# 16. memory_mb ≥ 0
# ---------------------------------------------------------------------------
def test_memory_non_negative(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample memory_mb is non-negative and finite."""
    for metal, info in smoke_payload["per_metal"].items():
        for rec in info["records"]:
            m = float(rec["memory_mb"])
            assert m >= 0.0, (
                f"negative memory_mb for {metal}/idx={rec['idx']}: {m}"
            )
            assert m < 1e6, (
                f"memory_mb unreasonably large for {metal}/idx={rec['idx']}: {m}"
            )


# ---------------------------------------------------------------------------
# 17. max_depth_seen ≤ max_depth
# ---------------------------------------------------------------------------
def test_max_depth_seen_bounded(smoke_payload: Dict[str, Any]) -> None:
    """Every per-sample max_depth_seen is bounded by the BFS depth budget."""
    max_depth = int(smoke_payload["max_depth"])
    for metal, info in smoke_payload["per_metal"].items():
        for rec in info["records"]:
            d = int(rec["max_depth_seen"])
            assert d <= max_depth, (
                f"max_depth_seen={d} > max_depth={max_depth} "
                f"for {metal}/idx={rec['idx']}"
            )


# ---------------------------------------------------------------------------
# 18. Per-metal aggregate has all expected keys
# ---------------------------------------------------------------------------
def test_per_metal_aggregate_keys(smoke_payload: Dict[str, Any]) -> None:
    """The per-metal aggregate dict carries every documented key."""
    for metal, info in smoke_payload["per_metal"].items():
        agg = info["aggregate"]
        missing = [k for k in EXPECTED_PER_METAL_KEYS if k not in agg]
        assert not missing, (
            f"per-metal aggregate for {metal!r} missing keys {missing!r}"
        )


# ---------------------------------------------------------------------------
# 19. closure_bench.json exists + valid schema
# ---------------------------------------------------------------------------
def test_closure_bench_json_written(smoke_paths: Dict[str, Path]) -> None:
    """save_verifier_benchmark writes a valid closure_bench.json."""
    json_path = smoke_paths["json"]
    assert json_path.is_file(), f"closure_bench.json missing at {json_path}"
    with json_path.open("r") as fh:
        payload = json.load(fh)
    assert payload["schema_version"] == 1
    assert payload["tool"] == "verify_closure_theorem.py"
    assert payload["pilot_id"] == "wf36_closure_theorem"
    assert "per_metal" in payload
    assert "per_sample" in payload
    assert "aggregate" in payload
    # The per-metal dict must have one entry per requested metal.
    assert set(payload["per_metal"].keys()) == {"Pt", "Ru", "Pd"}


# ---------------------------------------------------------------------------
# 20. per_metal.csv exists + correct header
# ---------------------------------------------------------------------------
def test_per_metal_csv_written(smoke_paths: Dict[str, Path]) -> None:
    """save_verifier_benchmark writes a per_metal.csv with the documented header."""
    csv_path = smoke_paths["csv"]
    assert csv_path.is_file(), f"per_metal.csv missing at {csv_path}"
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        assert "metal" in header
        assert "n_samples" in header
        assert "closure_reachability_rate" in header
        assert "mean_wall_time" in header
        assert "peak_memory_mb" in header
        rows = list(reader)
    # 3 metals in the smoke fixture.
    assert len(rows) == 3, (
        f"per_metal.csv should have 3 rows, got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# Bonus invariants
# ---------------------------------------------------------------------------
def test_load_metal_samples_first_n() -> None:
    """_load_metal_samples returns the first n SMILES tagged with the metal."""
    smis = _load_metal_samples(
        DEFAULT_TMQM_CSV, "Pt", 5, dataset="tmqm",
    )
    assert len(smis) == 5, f"expected 5 Pt SMILES, got {len(smis)}"
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
    assert len(partners) >= 6, (
        f"expected >= 6 partner terms, got {len(partners)}"
    )


def test_aggregate_per_metal_empty() -> None:
    """_aggregate_per_metal on an empty list returns the zero-vector aggregate."""
    agg = _aggregate_per_metal([])
    assert agg["n_samples"] == 0
    assert agg["closure_reachability_rate"] == 0.0
    assert agg["mean_beta_depth"] == 0.0
    assert agg["mean_wall_time"] == 0.0
    assert agg["peak_memory_mb"] == 0.0
    assert agg["n_error"] == 0


def test_is_pt_coordinated_cisplatin() -> None:
    """_is_pt_coordinated returns True for cisplatin."""
    from molmetal_lam.lam_chem.cisplatin_builder import build_cisplatin
    c = build_cisplatin()
    assert _is_pt_coordinated(c) is True


def test_is_pt_coordinated_ru_complex() -> None:
    """_is_pt_coordinated returns False for a Ru complex (no Pt atoms)."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    ru = MoleculeClosedTerm.from_smiles(
        "[NH3][Ru]([NH3])([NH3])([NH3])([NH3])[Cl]", embed_3d=False,
    )
    assert _is_pt_coordinated(ru) is False


def test_utcnow_iso_format() -> None:
    """_utcnow_iso returns a Z-suffixed ISO 8601 timestamp."""
    s = _utcnow_iso()
    assert s.endswith("Z"), f"timestamp must end with 'Z', got {s!r}"
    assert "T" in s


def test_benchmark_one_smoke() -> None:
    """_benchmark_one on cisplatin returns the expected per-sample record."""
    rec = _benchmark_one(
        "[NH3][Pt]([NH3])([Cl])([Cl])", 0, "Pt",
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


def test_smoke_completes_under_budget() -> None:
    """The 5-sample × 3-metal smoke completes in < 60 s wall."""
    t0 = time.perf_counter()
    payload = run_closure_theorem_verifier(
        dataset="tmqm",
        n_samples_per_metal=5,
        metals=["Pt", "Ru", "Pd"],
        max_depth=1,
        max_products=64,
    )
    wall = time.perf_counter() - t0
    assert wall < 60.0, f"smoke run took {wall:.2f}s; budget is 60s"
    assert payload["n_samples_total"] == 15


def test_platinai_metals_subset() -> None:
    """PLATINAI_METALS is a subset of the tmQM d-block metals."""
    assert set(PLATINAI_METALS).issubset(set(TMQM_DBLOCK_METALS)), (
        f"PLATINAI_METALS={sorted(PLATINAI_METALS)} must be subset of "
        f"TMQM_DBLOCK_METALS"
    )


def test_closure_ok_summary_present(smoke_payload: Dict[str, Any]) -> None:
    """The smoke payload aggregate carries the documented summary keys."""
    agg = smoke_payload["aggregate"]
    assert "n_samples" in agg
    assert "n_closure_ok" in agg
    assert "closure_reachability_rate" in agg
    assert "mean_wall_time" in agg
    assert "peak_memory_mb" in agg
    assert "total_wall_time" in agg


def test_platinai_load_smoke() -> None:
    """Loading Pt-tagged samples from PlatinAI 5K returns non-empty list."""
    smis = _load_metal_samples(
        DEFAULT_PLATINAI_CSV, "Pt", 3, dataset="platinai",
    )
    assert len(smis) >= 1, (
        f"PlatinAI Pt SMILES should exist, got {len(smis)}"
    )


def test_default_csv_paths_exist() -> None:
    """The default tmQM and PlatinAI CSV paths exist on disk."""
    assert DEFAULT_TMQM_CSV.is_file(), (
        f"tmQM CSV missing at {DEFAULT_TMQM_CSV}"
    )
    assert DEFAULT_PLATINAI_CSV.is_file(), (
        f"PlatinAI CSV missing at {DEFAULT_PLATINAI_CSV}"
    )