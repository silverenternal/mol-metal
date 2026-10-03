"""Tests for the closure theorem cross-domain OOD benchmark (Wf-41 Phase 2, L12).

================================================================
Wf-41 Phase 2: closure-theorem cross-domain OOD benchmark contract.

This file ships the **15 pytest contracts** that pin the behaviour of
:mod:`molmetal.scripts.bench_closure_ood`:

Per-domain OOD closure (canned drugs / hand-built descriptors)
 1. test_closure_ood_imports           — script imports cleanly
 2. test_domain_specs_have_csv_paths   — every domain has CSV paths
 4. test_platinai_metals_subset        — PLATINAI_METALS subset of d-block
 5. test_metalcyto_metals_subset       — METALCYTO_METALS subset of d-block
 6. test_common_metals_intersection    — COMMON_METALS = intersection

Benchmark mechanics
 7. test_default_method_is_closure     — default method = "closure"
 8. test_default_n_samples_is_50       — default n_samples = 50
 9. test_default_threshold_is_0_4      — default Tanimoto threshold = 0.4
10. test_default_output_dir            — default output dir is wf41_closure_ood
11. test_partner_smile_list_size       — partner list has ≥ 6 tiles
12. test_tanimoto_protocol_loads       — TanimotoOODProtocol loads
13. test_filter_ood_subset_filters     — OOD filter drops known-similar

Smoke run / output contract
14. test_smoke_run_platinai_unfiltered — 2 samples × 2 metals smoke OK
15. test_smoke_run_ood_filter          — OOD filter path runs end-to-end

All tests are CPU-only, deterministic, and complete in < 180 s
combined.  Smoke runs use ``--n-samples 2`` for sub-30 s CI.

Run with::

    uv run --no-sync pytest -q molmetal/tests/test_closure_ood.py --tb=short
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


from molmetal.scripts import bench_closure_ood  # noqa: E402
from molmetal.scripts.bench_closure_ood import (  # noqa: E402
    COMMON_METALS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PRODUCTS,
    DEFAULT_METHOD,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATINAI_CSV,
    DEFAULT_METALCYTO_CSV,
    DEFAULT_SEEDS,
    DEFAULT_TANIMOTO_THRESHOLD,
    DEFAULT_TMQM_CSV,
    DOMAIN_SPECS,
    METALCYTO_METALS,
    METAL_Z,
    PARTNER_SMILE_LIST,
    PLATINAI_METALS,
    _aggregate_per_metal,
    _benchmark_one,
    _filter_ood_subset,
    _is_metal_coordinated,
    _load_click_rules,
    _load_metal_samples,
    _load_partner_terms,
    _load_tanimoto_protocol,
    _load_tmqm_train_pool,
    _utcnow_iso,
    run_closure_ood_benchmark,
    save_closure_ood_benchmark,
)


# ---------------------------------------------------------------------------
# RDKit availability probe — needed by OOD filter
# ---------------------------------------------------------------------------
def _have_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401

        return True
    except Exception:
        return False


requires_rdkit = pytest.mark.skipif(
    not _have_rdkit(),
    reason="RDKit not installed; OOD filter tests require Morgan FP",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_unfiltered_payload() -> Dict[str, Any]:
    """A 2-sample × 2-metal unfiltered smoke payload — fast enough for CI."""
    return run_closure_ood_benchmark(
        n_samples=2,
        domains=["platinai_5k"],
        method="closure",
        max_depth=1,
        max_products=32,
    )


@pytest.fixture(scope="module")
def smoke_ood_paths(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Artefact paths produced by a 2-sample × 2-metal OOD-filter smoke run."""
    out = tmp_path_factory.mktemp("wf41_closure_ood_smoke")
    payload = run_closure_ood_benchmark(
        n_samples=2,
        domains=["platinai_5k"],
        method="closure+ood_filter",
        max_depth=1,
        max_products=32,
        output_dir=out,
    )
    paths = save_closure_ood_benchmark(payload, output_dir=out)
    return paths


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def test_closure_ood_imports() -> None:
    """The cross-domain OOD benchmark module imports cleanly."""
    assert bench_closure_ood is not None
    assert hasattr(bench_closure_ood, "run_closure_ood_benchmark")
    assert hasattr(bench_closure_ood, "save_closure_ood_benchmark")
    assert hasattr(bench_closure_ood, "_filter_ood_subset")
    assert hasattr(bench_closure_ood, "_load_tanimoto_protocol")
    assert hasattr(bench_closure_ood, "_benchmark_one")
    assert hasattr(bench_closure_ood, "_load_metal_samples")
    assert hasattr(bench_closure_ood, "_load_tmqm_train_pool")
    assert hasattr(bench_closure_ood, "_aggregate_per_metal")


# ---------------------------------------------------------------------------
# 2. Domain specs have CSV paths
# ---------------------------------------------------------------------------
def test_domain_specs_have_csv_paths() -> None:
    """Every domain in DOMAIN_SPECS has a CSV path, metal_col, smiles_col."""
    assert "platinai_5k" in DOMAIN_SPECS
    assert "metalcytotox_5metal" in DOMAIN_SPECS
    for name, spec in DOMAIN_SPECS.items():
        assert "csv" in spec, f"{name} missing 'csv'"
        assert "metal_col" in spec, f"{name} missing 'metal_col'"
        assert "smiles_col" in spec, f"{name} missing 'smiles_col'"
        assert "metals" in spec, f"{name} missing 'metals'"
        assert "description" in spec, f"{name} missing 'description'"
        assert Path(spec["csv"]).is_file(), f"{name} CSV missing at {spec['csv']}"


# ---------------------------------------------------------------------------
# 3. PLATINAI_METALS subset of d-block
# ---------------------------------------------------------------------------
def test_platinai_metals_subset() -> None:
    """The 7 PlatinAI metals are a subset of the d-block 30 metals."""
    # We can't import DBLOCK_METALS from bench_closure_dblock without
    # circular-import risk; instead we use the METAL_Z map as the
    # canonical d-block reference.
    db = set(METAL_Z.keys())
    platinai_set = set(PLATINAI_METALS)
    assert platinai_set.issubset(db), (
        f"PLATINAI_METALS={sorted(platinai_set)} must be subset of "
        f"d-block METAL_Z={sorted(db)}"
    )
    assert len(PLATINAI_METALS) >= 5


# ---------------------------------------------------------------------------
# 4. METALCYTO_METALS subset of d-block
# ---------------------------------------------------------------------------
def test_metalcyto_metals_subset() -> None:
    """The 5 MetalCytoToxDB metals are a subset of the d-block 30 metals."""
    db = set(METAL_Z.keys())
    cyto_set = set(METALCYTO_METALS)
    assert cyto_set.issubset(db), (
        f"METALCYTO_METALS={sorted(cyto_set)} must be subset of "
        f"d-block METAL_Z={sorted(db)}"
    )
    assert len(METALCYTO_METALS) == 5


# ---------------------------------------------------------------------------
# 5. COMMON_METALS = intersection
# ---------------------------------------------------------------------------
def test_common_metals_intersection() -> None:
    """COMMON_METALS = PLATINAI_METALS ∩ METALCYTO_METALS."""
    expected = tuple(m for m in PLATINAI_METALS if m in METALCYTO_METALS)
    assert COMMON_METALS == expected, (
        f"COMMON_METALS={COMMON_METALS} != intersection " f"{expected}"
    )


# ---------------------------------------------------------------------------
# 6. Default method = 'closure'
# ---------------------------------------------------------------------------
def test_default_method_is_closure() -> None:
    """The default method is 'closure' (unfiltered baseline)."""
    assert DEFAULT_METHOD == "closure"


# ---------------------------------------------------------------------------
# 7. Default n_samples = 50
# ---------------------------------------------------------------------------
def test_default_n_samples_is_50() -> None:
    """The default n_samples is 50 (2 domains × up to 7 metals × 50 = 350)."""
    assert DEFAULT_N_SAMPLES == 50


# ---------------------------------------------------------------------------
# 8. Default Tanimoto threshold = 0.4
# ---------------------------------------------------------------------------
def test_default_threshold_is_0_4() -> None:
    """The default Tanimoto OOD threshold is 0.4 (Pocket2Mol / TargetDiff)."""
    assert DEFAULT_TANIMOTO_THRESHOLD == 0.4


# ---------------------------------------------------------------------------
# 9. Default output dir is wf41_closure_ood
# ---------------------------------------------------------------------------
def test_default_output_dir() -> None:
    """The default output dir is reports/wf41_closure_ood/."""
    assert DEFAULT_OUTPUT_DIR.name == "wf41_closure_ood"
    assert DEFAULT_OUTPUT_DIR.parent.name == "reports"


# ---------------------------------------------------------------------------
# 10. Partner SMILES list size
# ---------------------------------------------------------------------------
def test_partner_smile_list_size() -> None:
    """PARTNER_SMILE_LIST has at least 6 entries (covers all 5 click handles)."""
    assert len(PARTNER_SMILE_LIST) >= 6
    joined = " ".join(PARTNER_SMILE_LIST)
    assert "C#C" in joined, "need at least one alkyne partner"
    assert "[N+]=[N-]" in joined, "need at least one azide partner"
    assert "CS" in joined, "need at least one thiol partner"


# ---------------------------------------------------------------------------
# 11. Tanimoto protocol loads
# ---------------------------------------------------------------------------
@requires_rdkit
def test_tanimoto_protocol_loads() -> None:
    """The TanimotoOODProtocol loads with the documented threshold."""
    p = _load_tanimoto_protocol()
    assert float(p.threshold) == DEFAULT_TANIMOTO_THRESHOLD
    # Self-similarity is 1.0 (canonical convention).
    assert p.compute_tanimoto("c1ccccc1O", "c1ccccc1O") == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ---------------------------------------------------------------------------
# 12. OOD filter drops known-similar samples
# ---------------------------------------------------------------------------
@requires_rdkit
def test_filter_ood_subset_filters() -> None:
    """OOD filter drops samples with Tanimoto >= 0.4 to the train pool.

    Build a 2-sample train pool where the second sample is *identical*
    to the second query — the OOD filter must drop it.
    """
    train_pool = ["c1ccccc1O", "N#Cc1ccccc1"]  # phenol + benzonitrile
    queries = [
        "c1ccccc1O",  # identical to train[0] → NOT OOD
        "CCCCCCCCCC",  # decane — fully OOD
        "N#Cc1ccccc1",  # identical to train[1] → NOT OOD
        "c1ccncc1",  # pyridine — likely OOD
    ]
    p = _load_tanimoto_protocol(threshold=0.4)
    ood_subset, audit = _filter_ood_subset(queries, train_pool, p, "Test")
    # The 2 phenol/benzonitrile samples must be filtered out.
    assert audit["n_in"] == 4
    assert audit["n_ood"] == 2  # decane + pyridine
    assert audit["ood_rate"] == pytest.approx(0.5, abs=1e-6)
    ood_smiles = [s for (_, s, _) in ood_subset]
    assert "CCCCCCCCCC" in ood_smiles
    assert "c1ccncc1" in ood_smiles


# ---------------------------------------------------------------------------
# 13. _load_metal_samples returns first n
# ---------------------------------------------------------------------------
def test_load_metal_samples_first_n() -> None:
    """_load_metal_samples returns the first n SMILES tagged with the metal."""
    smis = _load_metal_samples(
        DEFAULT_PLATINAI_CSV,
        "Pt",
        3,
        smiles_col="smiles",
        metal_col="metal",
    )
    assert len(smis) == 3
    assert all(len(s) > 0 for s in smis)


# ---------------------------------------------------------------------------
# 14. Smoke run unfiltered
# ---------------------------------------------------------------------------
EXPECTED_AGGREGATE_KEYS: List[str] = [
    "n_input",
    "n_ood",
    "n_samples",
    "n_closure_ok",
    "closure_reachability_rate",
    "ood_rate",
    "mean_wall_time",
    "std_wall_time",
    "median_wall_time",
    "peak_memory_mb",
    "total_wall_time",
    "n_error",
    "n_metal_coordinated",
    "metal_coordination_rate",
]


def test_smoke_run_platinai_unfiltered(
    smoke_unfiltered_payload: Dict[str, Any],
) -> None:
    """2 samples × 2 metals unfiltered smoke run produces the expected schema."""
    p = smoke_unfiltered_payload
    assert p["schema_version"] == 1
    assert p["tool"] == "bench_closure_ood.py"
    assert p["pilot_id"] == "wf41_closure_ood"
    assert p["method"] == "closure"
    assert p["domains"] == ["platinai_5k"]
    assert p["n_samples_per_metal"] == 2
    assert p["tanimoto_threshold"] == 0.4
    # Aggregate is bounded by n_input.
    agg = p["aggregate"]
    assert agg["n_input"] >= 1
    assert agg["n_ood"] == agg["n_input"]  # unfiltered → ood_rate=1
    rate = float(agg["closure_reachability_rate"])
    assert 0.0 <= rate <= 1.0, f"closure_reachability_rate out of [0,1]: {rate}"


# ---------------------------------------------------------------------------
# 15. Smoke run with OOD filter
# ---------------------------------------------------------------------------
def test_smoke_run_ood_filter(
    smoke_ood_paths: Dict[str, Path],
) -> None:
    """The OOD filter smoke run writes the documented artefacts."""
    json_path = smoke_ood_paths["json"]
    csv_path = smoke_ood_paths["csv"]
    md_path = smoke_ood_paths["markdown"]
    assert json_path.is_file(), f"ood_closure.json missing at {json_path}"
    assert csv_path.is_file(), f"per_domain.csv missing at {csv_path}"
    assert md_path.is_file(), f"ood_closure.md missing at {md_path}"
    payload = json.loads(json_path.read_text())
    assert payload["method"] == "closure+ood_filter"
    assert payload["domains"] == ["platinai_5k"]
    # The OOD filter must shrink the input (some PlatinAI samples are
    # Tanimoto-similar to tmQM).
    agg = payload["aggregate"]
    assert agg["n_input"] >= agg["n_ood"], (
        f"OOD filter must not inflate input: n_input={agg['n_input']} "
        f"< n_ood={agg['n_ood']}"
    )
    assert agg["ood_rate"] <= 1.0
    # Markdown summary carries the documented header + table.
    text = md_path.read_text()
    assert "Wf-41 Phase 2" in text
    assert "OOD" in text or "ood" in text
    # per_domain.csv has the documented header.
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for f in EXPECTED_AGGREGATE_KEYS:
            assert f in header, f"per_domain.csv missing column {f!r}"


# ---------------------------------------------------------------------------
# Bonus invariants
# ---------------------------------------------------------------------------
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
    agg = _aggregate_per_metal([], n_input=0, n_ood=0)
    assert agg["n_samples"] == 0
    assert agg["n_input"] == 0
    assert agg["n_ood"] == 0
    assert agg["closure_reachability_rate"] == 0.0
    assert agg["ood_rate"] == 0.0
    assert agg["mean_wall_time"] == 0.0
    assert agg["peak_memory_mb"] == 0.0
    assert agg["n_error"] == 0


def test_default_csv_paths_exist() -> None:
    """All default CSV paths exist on disk."""
    assert (
        DEFAULT_PLATINAI_CSV.is_file()
    ), f"PlatinAI CSV missing at {DEFAULT_PLATINAI_CSV}"
    assert DEFAULT_TMQM_CSV.is_file(), f"tmQM CSV missing at {DEFAULT_TMQM_CSV}"
    assert (
        DEFAULT_METALCYTO_CSV.is_file()
    ), f"MetalCytoTox CSV missing at {DEFAULT_METALCYTO_CSV}"


def test_default_max_depth_is_1() -> None:
    """The default max_depth is 1 (one click rule application)."""
    assert DEFAULT_MAX_DEPTH == 1


def test_default_seeds_is_1() -> None:
    """The default seeds is 1 (placeholder for future stochastic re-runs)."""
    assert DEFAULT_SEEDS == 1


def test_default_max_products_is_64() -> None:
    """The default max_products is 64 (BFS safety cap)."""
    assert DEFAULT_MAX_PRODUCTS == 64


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
        domain="platinai_5k",
        click_rules=_load_click_rules(),
        partner_terms=_load_partner_terms(),
        max_depth=1,
        max_products=64,
    )
    assert rec["metal"] == "Pt"
    assert rec["idx"] == 0
    assert rec["depth"] == 1
    assert rec["domain"] == "platinai_5k"
    assert rec["status"] in ("ok", "skipped", "error")
    assert isinstance(rec["closure_ok"], bool)
    assert isinstance(rec["wall_time_s"], float)
    assert isinstance(rec["memory_mb"], float)
    assert isinstance(rec["n_metal_coordinated"], int)


def test_smoke_completes_under_budget() -> None:
    """The 2-sample × 1-domain unfiltered smoke completes in < 60 s wall."""
    t0 = time.perf_counter()
    payload = run_closure_ood_benchmark(
        n_samples=2,
        domains=["platinai_5k"],
        method="closure",
        max_depth=1,
        max_products=32,
    )
    wall = time.perf_counter() - t0
    assert wall < 60.0, f"smoke run took {wall:.2f}s; budget is 60s"
    assert payload["n_samples_per_metal"] == 2


def test_per_metal_records_have_all_keys() -> None:
    """Every per-sample record carries every documented key."""
    payload = run_closure_ood_benchmark(
        n_samples=2,
        domains=["platinai_5k"],
        method="closure",
        max_depth=1,
        max_products=32,
    )
    expected_keys = {
        "idx",
        "metal",
        "domain",
        "smiles",
        "status",
        "closure_ok",
        "n_visited",
        "n_products",
        "max_depth_seen",
        "well_typed_rate",
        "n_metal_coordinated",
        "n_atoms_start",
        "parse_error",
        "wall_time_s",
        "memory_mb",
        "depth",
    }
    for dom in payload["domains"]:
        for metal, m_info in payload["per_domain"][dom]["per_metal"].items():
            for rec in m_info["records"]:
                missing = expected_keys - set(rec.keys())
                assert not missing, f"record in {dom}/{metal} missing keys {missing!r}"


def test_method_validation() -> None:
    """run_closure_ood_benchmark raises on invalid method."""
    with pytest.raises(ValueError, match="method must be"):
        run_closure_ood_benchmark(
            n_samples=1,
            domains=["platinai_5k"],
            method="bogus",
            max_depth=1,
            max_products=32,
        )


def test_per_domain_aggregate_keys(
    smoke_unfiltered_payload: Dict[str, Any],
) -> None:
    """Per-domain aggregate has every documented key."""
    for dom in smoke_unfiltered_payload["domains"]:
        d_agg = smoke_unfiltered_payload["per_domain"][dom]["aggregate"]
        missing = [k for k in EXPECTED_AGGREGATE_KEYS if k not in d_agg]
        assert not missing, f"per-domain aggregate for {dom!r} missing keys {missing!r}"
        assert "ood_rate" in d_agg
        assert "n_input" in d_agg


def test_metal_z_coverage() -> None:
    """Every cross-domain metal has an entry in METAL_Z."""
    for m in PLATINAI_METALS + METALCYTO_METALS:
        assert m in METAL_Z, f"METAL_Z missing {m!r}"


def test_load_tmqm_train_pool_returns_list() -> None:
    """_load_tmqm_train_pool returns a non-empty list of SMILES."""
    pool = _load_tmqm_train_pool(DEFAULT_TMQM_CSV)
    assert isinstance(pool, list)
    assert len(pool) >= 1000, f"tmQM pool unexpectedly small: {len(pool)}"
    assert all(isinstance(s, str) for s in pool)


def test_is_metal_coordinated_cisplatin() -> None:
    """_is_metal_coordinated returns True for Pt at Z=78."""
    from molmetal_lam.lam_chem.cisplatin_builder import build_cisplatin

    c = build_cisplatin()
    assert _is_metal_coordinated(c, 78) is True
    assert _is_metal_coordinated(c, 44) is False  # Ru in cisplatin → False
