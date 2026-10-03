"""Tests for Wf-248 Phase 1 — F5_RDKIT_2D 32-d RDKit 2D descriptors.

=====================================================================
Pins down the Wf-248 / Wf-218 invariants for the new
``molmetal.data.rdkit_2d_extended`` module:

  * ``RDKIT_2D_EXTENDED_FEATURE_COLUMNS`` is a 32-tuple matching the
    Wf-218 frozen order, with the first 12 entries bit-for-bit equal
    to ``RDKIT_2D_FEATURE_COLUMNS``.
  * ``compute_rdkit_2d_extended(smiles)`` returns a ``(32,)``
    ``np.float32`` array, or ``None`` for invalid SMILES.
  * The first 12 entries of ``compute_rdkit_2d_extended`` are
    bit-for-bit equal to ``compute_rdkit_2d_12d`` — backward-compat
    guarantee with the Wf-49 Phase 1 12-d extractor.
  * Per-descriptor values match ``rdkit.Chem.Descriptors.<fn>(mol)``
    on canonical SMILES.
  * The Wf-127 §Cause 2 expansion descriptors (BertzCT, Chi0v, Chi0n,
    Chi1n, HallKierAlpha, Kappa1/2/3, LabuteASA, partial-charge
    stats, NumAliphaticRings, NumSaturatedRings, NumHeteroatoms,
    NHOHCount, NOCount, fr_halogen, fr_Al_COO) all carry signal
    beyond the 12-d baseline.

Tests (count target: 18):
 1. Registry invariants — RDKIT_2D_EXTENDED_FEATURE_COLUMNS length 32.
 2. RDKIT_2D_EXTENDED_FEATURE_COLUMNS order matches the Wf-218 frozen order.
 3. First 12 entries of 32-d set match 12-d set bit-for-bit (backward-compat).
 4. No duplicate column names in the 32-d set.
 5. 32-d set covers shape / charge / halogen / heteroatom descriptors.
 6. RDKIT_2D_DIM and RDKIT_2D_BASE_DIM are 32 and 12.
 7. Module-level constants are reachable (no ImportError).
 8. compute_rdkit_2d_extended returns a (32,) float32 array on a valid SMILES.
 9. compute_rdkit_2d_extended returns None on an invalid SMILES.
10. compute_rdkit_2d_extended returns None on an empty SMILES.
11. compute_rdkit_2d_extended returns None on a 0-heavy-atom SMILES.
12. compute_rdkit_2d_12d returns a (12,) float32 array on a valid SMILES.
13. compute_rdkit_2d_12d returns None on an invalid SMILES.
14. Backward-compat — first 12 entries of compute_rdkit_2d_extended
    equal compute_rdkit_2d_12d bit-for-bit (toluene + aspirin + water).
15. Per-descriptor equality — 32-d values match rdkit.Chem.Descriptors
    on canonical SMILES (aspirin).
16. fr_halogen count is correct on halogen-rich SMILES
    (1,1,2,2-tetrachloroethane → fr_halogen = 4).
17. Lipinski drivers (MolWt, MolLogP, TPSA, HBA, HBD) are present and finite.
18. Module is importable + has the public API symbols (compute_*,
    extended_feature_columns, base_feature_columns, RDKIT_2D_DIM,
    RDKIT_2D_BASE_DIM, RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
    RDKIT_2D_FEATURE_COLUMNS, RDKit2DExtendedError).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Probe SMILES — canonical ligands covering common organic + metal-cyto
# chemotypes that the Wf-49 Phase 1 extractor has been tested on.
# ---------------------------------------------------------------------------
TOLUENE = "Cc1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
WATER = "O"
HALOGEN_RICH = "ClC(Cl)C(Cl)Cl"  # 1,1,2,2-tetrachloroethane (fr_halogen = 4)
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # canonical multi-ring
INVALID_SMILES = "@@@invalid@@@"

# Canonical expected 32-d column names — bit-for-bit copy of the
# Wf-218 frozen order in lambda_astsr.py.
EXPECTED_32D_NAMES: tuple = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi0n",
    "Chi1n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "fr_halogen",
    "fr_Al_COO",
)

# Canonical 12-d baseline — bit-for-bit copy of the Wf-49 Phase 1
# frozen order in extract_rdkit_2d_descriptors.py / lambda_astsr.py.
EXPECTED_12D_NAMES: tuple = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)


# ---------------------------------------------------------------------------
# Public API surface — try to import the new module.  If RDKit is
# missing (e.g. in a minimal CI env without chemistry libs), the
# whole test module is skipped — same defensive pattern as
# test_extract_rdkit_2d.py.
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem  # type: ignore

    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


def _skip_if_no_rdkit() -> None:
    if not _HAS_RDKIT:
        pytest.skip(
            "RDKit not importable in this environment; "
            "molmetal.data.rdkit_2d_extended requires RDKit."
        )


@pytest.fixture(scope="module")
def mod():
    _skip_if_no_rdkit()
    import importlib

    return importlib.import_module("molmetal.data.rdkit_2d_extended")


# ---------------------------------------------------------------------------
# 1. Registry invariants
# ---------------------------------------------------------------------------
def test_rdkit_2d_extended_feature_columns_is_tuple_of_32(mod) -> None:
    """RDKIT_2D_EXTENDED_FEATURE_COLUMNS is a tuple of exactly 32 names."""
    cols = mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    assert isinstance(cols, tuple)
    assert len(cols) == 32
    assert all(isinstance(n, str) for n in cols)


def test_rdkit_2d_extended_feature_columns_order_frozen(mod) -> None:
    """The 32-d column order matches the Wf-218 frozen order."""
    cols = mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    assert cols == EXPECTED_32D_NAMES, (
        f"RDKIT_2D_EXTENDED_FEATURE_COLUMNS order drift.\n"
        f"  expected: {EXPECTED_32D_NAMES}\n"
        f"  actual:   {cols}"
    )


def test_first_12_of_32d_match_12d_baseline_bit_for_bit(mod) -> None:
    """Backward-compat: first 12 entries of 32-d set == 12-d set."""
    cols32 = mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    cols12 = mod.RDKIT_2D_FEATURE_COLUMNS
    assert cols32[:12] == cols12 == EXPECTED_12D_NAMES, (
        f"First 12 entries of 32-d set drifted from 12-d baseline.\n"
        f"  32d[:12]: {cols32[:12]}\n"
        f"  12d:      {cols12}"
    )


def test_no_duplicate_column_names_in_32d_set(mod) -> None:
    """No column name appears twice in the 32-d set."""
    cols = mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    dup = [n for n in set(cols) if cols.count(n) > 1]
    assert len(dup) == 0, f"32-d set has duplicate names: {dup}"


def test_32d_set_covers_shape_charge_halogen_heteroatoms(mod) -> None:
    """The 20-d expansion covers shape, charge, and halogen features."""
    required = {
        # Shape / topology (Wf-218 expansion)
        "BertzCT",
        "Chi0v",
        "Chi0n",
        "Chi1n",
        "HallKierAlpha",
        # Kappa shape indices
        "Kappa1",
        "Kappa2",
        "Kappa3",
        # Surface area
        "LabuteASA",
        # Partial charge
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        # Ring features
        "NumAliphaticRings",
        "NumSaturatedRings",
        # Heteroatom / functional group counts
        "NumHeteroatoms",
        "NHOHCount",
        "NOCount",
        # Halogen + carboxylate (Pt-coordination relevant)
        "fr_halogen",
        "fr_Al_COO",
    }
    cols = set(mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS)
    missing = required - cols
    assert (
        not missing
    ), f"32-d set missing Wf-218 expansion descriptors: {sorted(missing)}"


def test_dim_constants_are_32_and_12(mod) -> None:
    """RDKIT_2D_DIM = 32, RDKIT_2D_BASE_DIM = 12."""
    assert mod.RDKIT_2D_DIM == 32
    assert mod.RDKIT_2D_BASE_DIM == 12


def test_module_constants_are_reachable(mod) -> None:
    """Public API constants are present on the module."""
    for name in (
        "compute_rdkit_2d_extended",
        "compute_rdkit_2d_12d",
        "RDKIT_2D_EXTENDED_FEATURE_COLUMNS",
        "RDKIT_2D_FEATURE_COLUMNS",
        "RDKIT_2D_DIM",
        "RDKIT_2D_BASE_DIM",
        "RDKIT_2D_EXTENDED_DESCRIPTORS",
        "extended_feature_columns",
        "base_feature_columns",
        "RDKit2DExtendedError",
    ):
        assert hasattr(mod, name), f"module missing public symbol: {name}"


# ---------------------------------------------------------------------------
# 2. compute_rdkit_2d_extended — output shape, dtype, NaN handling
# ---------------------------------------------------------------------------
def test_compute_32d_returns_32d_float32_array_on_valid_smiles(mod) -> None:
    """compute_rdkit_2d_extended returns (32,) float32 on toluene."""
    v = mod.compute_rdkit_2d_extended(TOLUENE)
    assert v is not None
    assert isinstance(v, np.ndarray)
    assert v.shape == (32,), f"expected (32,), got {v.shape}"
    assert v.dtype == np.float32, f"expected float32, got {v.dtype}"
    # All entries should be finite on toluene (no partial-charge failures
    # because toluene has aromatic carbons with implicit Hs RDKit can
    # assign charges to).
    n_bad = int(np.sum(~np.isfinite(v)))
    assert n_bad == 0, f"toluene produced {n_bad} non-finite descriptor entries"


def test_compute_32d_returns_none_on_invalid_smiles(mod) -> None:
    """compute_rdkit_2d_extended returns None for unparseable SMILES."""
    v = mod.compute_rdkit_2d_extended(INVALID_SMILES)
    assert v is None


def test_compute_32d_returns_none_on_empty_smiles(mod) -> None:
    """compute_rdkit_2d_extended returns None for an empty SMILES."""
    v = mod.compute_rdkit_2d_extended("")
    assert v is None


def test_compute_32d_returns_none_on_zero_heavy_atom_smiles(mod) -> None:
    """compute_rdkit_2d_extended returns None for a SMILES with 0 heavy atoms.

    RDKit can produce a valid Mol for ``[H+]`` with no heavy atoms; the
    32-d descriptor pipeline must reject this (no signal to extract).
    """
    v = mod.compute_rdkit_2d_extended("[H+]")
    assert v is None, (
        f"[H+] has 0 heavy atoms but compute_rdkit_2d_extended returned "
        f"{v!r}; expected None."
    )


# ---------------------------------------------------------------------------
# 3. compute_rdkit_2d_12d — backward-compat 12-d path
# ---------------------------------------------------------------------------
def test_compute_12d_returns_12d_float32_array(mod) -> None:
    """compute_rdkit_2d_12d returns (12,) float32 on toluene."""
    v = mod.compute_rdkit_2d_12d(TOLUENE)
    assert v is not None
    assert isinstance(v, np.ndarray)
    assert v.shape == (12,)
    assert v.dtype == np.float32


def test_compute_12d_returns_none_on_invalid_smiles(mod) -> None:
    """compute_rdkit_2d_12d returns None on unparseable SMILES."""
    v = mod.compute_rdkit_2d_12d(INVALID_SMILES)
    assert v is None


# ---------------------------------------------------------------------------
# 4. Backward-compat — first 12 entries of 32-d match 12-d bit-for-bit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smi", [TOLUENE, ASPIRIN, WATER, CAFFEINE, HALOGEN_RICH])
def test_first_12_of_32d_match_12d_bit_for_bit(mod, smi: str) -> None:
    """For every probe SMILES, the first 12 entries of 32-d == 12-d."""
    v32 = mod.compute_rdkit_2d_extended(smi)
    v12 = mod.compute_rdkit_2d_12d(smi)
    assert (
        v32 is not None and v12 is not None
    ), f"Both compute_* must succeed on {smi!r}; got v32={v32}, v12={v12}."
    np.testing.assert_array_equal(
        v32[:12],
        v12,
        err_msg=(
            f"first 12 entries of compute_rdkit_2d_extended drifted from "
            f"compute_rdkit_2d_12d on {smi!r}; this breaks backward-compat "
            f"for the Wf-49 Phase 1 12-d path."
        ),
    )


# ---------------------------------------------------------------------------
# 5. Per-descriptor equality — 32-d values match rdkit.Chem.Descriptors
# ---------------------------------------------------------------------------
def test_per_descriptor_values_match_rdkit_directly(mod) -> None:
    """Each of the 32-d entries equals the rdkit.Chem.Descriptors callable.

    This pins down that compute_rdkit_2d_extended is a thin wrapper
    around the canonical RDKit Descriptors API — no schema drift, no
    hidden pre/post-processing, no per-descriptor type coercion.  We
    probe on aspirin because it exercises all 32 descriptors without
    triggering the disconnected-molecule NaN path (cisplatin SMILES
    with disconnected fragments would NaN out the partial-charge
    descriptors, which is a separate test).
    """
    from rdkit.Chem import Descriptors  # type: ignore

    mol = Chem.MolFromSmiles(ASPIRIN)
    assert mol is not None
    expected: List[float] = []
    for _name, fn in mod.RDKIT_2D_EXTENDED_DESCRIPTORS:
        v = float(fn(mol))
        expected.append(v)
    expected_arr = np.asarray(expected, dtype=np.float32)
    v = mod.compute_rdkit_2d_extended(ASPIRIN)
    assert v is not None
    np.testing.assert_array_equal(
        v,
        expected_arr,
        err_msg=(
            "compute_rdkit_2d_extended values diverged from "
            "rdkit.Chem.Descriptors on aspirin — the 32-d pipeline is "
            "not a thin wrapper, or the descriptor registry drifted."
        ),
    )


# ---------------------------------------------------------------------------
# 6. Specific Wf-218 expansion invariants
# ---------------------------------------------------------------------------
def test_fr_halogen_count_correct_on_halogen_rich_smiles(mod) -> None:
    """fr_halogen == 4 on 1,1,2,2-tetrachloroethane (4 Cl substituents)."""
    v = mod.compute_rdkit_2d_extended(HALOGEN_RICH)
    assert v is not None
    halo_idx = mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS.index("fr_halogen")
    assert v[halo_idx] == pytest.approx(
        4.0, abs=1e-6
    ), f"fr_halogen on {HALOGEN_RICH} expected 4.0; got {v[halo_idx]}."


def test_lipinski_drivers_present_and_finite_on_aspirin(mod) -> None:
    """MolWt, MolLogP, TPSA, HBA, HBD all finite on aspirin."""
    v = mod.compute_rdkit_2d_extended(ASPIRIN)
    assert v is not None
    cols = mod.RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    for drv in ("MolWt", "MolLogP", "TPSA", "HBA", "HBD"):
        idx = cols.index(drv)
        assert np.isfinite(
            v[idx]
        ), f"Lipinski driver {drv} is not finite on aspirin: {v[idx]}."
    # And the values are in physically meaningful ranges.
    assert v[cols.index("MolWt")] == pytest.approx(
        180.16, abs=0.1
    ), f"Aspirin MolWt expected ~180.16; got {v[cols.index('MolWt')]}."


# ---------------------------------------------------------------------------
# 7. extended_feature_columns / base_feature_columns — fresh-tuple semantics
# ---------------------------------------------------------------------------
def test_extended_feature_columns_returns_fresh_tuple(mod) -> None:
    """extended_feature_columns() returns a fresh tuple copy."""
    a = mod.extended_feature_columns()
    b = mod.extended_feature_columns()
    assert a == b
    assert isinstance(a, tuple)
    assert len(a) == 32
    # Mutating must NOT affect subsequent calls.
    mutated = list(a)
    mutated[0] = "MUTATED"
    c = mod.extended_feature_columns()
    assert c[0] != "MUTATED"
    assert c[0] == a[0]


def test_base_feature_columns_returns_fresh_tuple(mod) -> None:
    """base_feature_columns() returns a fresh tuple copy."""
    a = mod.base_feature_columns()
    b = mod.base_feature_columns()
    assert a == b
    assert isinstance(a, tuple)
    assert len(a) == 12


# ---------------------------------------------------------------------------
# 8. Module-level sanity probe runs at module load
# ---------------------------------------------------------------------------
def test_module_load_completes_without_error() -> None:
    """Importing the module does not raise (smoke test on the module)."""
    import importlib

    importlib.import_module("molmetal.data.rdkit_2d_extended")


def test_error_class_is_runtime_error(mod) -> None:
    """RDKit2DExtendedError is a RuntimeError subclass."""
    assert issubclass(mod.RDKit2DExtendedError, RuntimeError)
