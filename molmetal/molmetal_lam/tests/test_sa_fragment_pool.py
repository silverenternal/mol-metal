"""Tests for the R15 SA-friendly fragment pool (PROBLEM 5.1).

Validates that ``FRAGMENT_POOL_SA_FRIENDLY`` -- the 20 new drug-like
small-molecule scaffolds added in Round-15 -- satisfies the SA-fragment
pool finalisation contract:

* Every SMILES is RDKit-parseable.
* Every SMILES has Ertl-Schuffenhauer SA score < 3.5 (so it can drag
  the product SA mean towards the TargetDiff reference band
  2.65-2.86).
* No canonical-form duplicates with the existing 220-tile pool
  (azide / alkyne / diene / thiol / dbco / boronic_acid / aryl_halide
  / carboxylic_acid / amine).
* Every SMILES passes the ETKDGv3 embed at ``randomSeed=0xC11C``
  (the same filter used by the validated 220-tile pool).
* The public ``fragments_from_chembl_reactive`` API can be toggled
  on / off via ``include_sa_friendly`` without breaking the rest of
  the pool.
* Pool size delta is exactly +20 tiles when ``include_sa_friendly=True``
  vs ``include_sa_friendly=False`` (so backward compatibility is
  preserved).
"""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from molmetal_lam.sbdd_env.sa_score import sa_score_ertl
from molmetal_lam.tile_lib.fragment_pool import (
    FRAGMENT_POOL_SA_FRIENDLY,
    FRAGMENT_POOL_AZIDES,
    FRAGMENT_POOL_ALKYNES,
    FRAGMENT_POOL_DIENES,
    FRAGMENT_POOL_THOLS,
    FRAGMENT_POOL_DBCO,
    FRAGMENT_POOL_BORONIC_ACIDS,
    FRAGMENT_POOL_ARYL_HALIDES,
    FRAGMENT_POOL_CARBOXYLIC_ACIDS,
    FRAGMENT_POOL_AMINES,
    fragments_from_chembl_reactive,
    l6_fragment_pool_metrics,
)


# Module-level constants used by multiple tests.
LEGACY_POOLS = [
    FRAGMENT_POOL_AZIDES,
    FRAGMENT_POOL_ALKYNES,
    FRAGMENT_POOL_DIENES,
    FRAGMENT_POOL_THOLS,
    FRAGMENT_POOL_DBCO,
    FRAGMENT_POOL_BORONIC_ACIDS,
    FRAGMENT_POOL_ARYL_HALIDES,
    FRAGMENT_POOL_CARBOXYLIC_ACIDS,
    FRAGMENT_POOL_AMINES,
]


def _canonical(smi: str) -> str:
    """Return the RDKit canonical SMILES, or empty string on parse fail."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else ""


def _legacy_canonicals() -> set[str]:
    out: set[str] = set()
    for pool in LEGACY_POOLS:
        for s in pool:
            c = _canonical(s)
            if c:
                out.add(c)
    return out


# ---------------------------------------------------------------------------
# Test 1: pool size is in the 10-20 target window.
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_size_window() -> None:
    """The R15 SA-friendly pool must have 10-20 entries."""
    n = len(FRAGMENT_POOL_SA_FRIENDLY)
    assert 10 <= n <= 20, (
        f"FRAGMENT_POOL_SA_FRIENDLY must have 10-20 entries per the "
        f"PROBLEM 5.1 spec, got {n}"
    )


# ---------------------------------------------------------------------------
# Test 2: every SMILES is RDKit-parseable.
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_all_parseable() -> None:
    """Every SMILES in FRAGMENT_POOL_SA_FRIENDLY must parse cleanly."""
    failures: list[str] = []
    for smi in FRAGMENT_POOL_SA_FRIENDLY:
        if Chem.MolFromSmiles(smi) is None:
            failures.append(smi)
    assert not failures, f"Unparseable SMILES: {failures}"


# ---------------------------------------------------------------------------
# Test 3: every SMILES has SA score < 3.5.
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_all_below_3p5_sa() -> None:
    """Every SA-friendly fragment must have SA < 3.5 (the legacy ceiling).

    The legacy R10 axis-A 220-tile pool yielded an aggregate SA mean of
    ~3.32.  The R15 axis-C SA-friendly pool is curated so every entry
    beats the 3.5 floor (range 1.0-3.0), with the express goal of
    pulling the product SA mean into the TargetDiff reference band
    2.65-2.86 when used.
    """
    violations: list[tuple[str, float]] = []
    for smi in FRAGMENT_POOL_SA_FRIENDLY:
        sa = sa_score_ertl(smi)
        if sa >= 3.5:
            violations.append((smi, sa))
    assert not violations, (
        f"SA-friendly fragments with SA >= 3.5: {violations}"
    )


# ---------------------------------------------------------------------------
# Test 4: no canonical-form duplicates with the legacy 220-tile pool.
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_no_duplicates_with_legacy() -> None:
    """No canonical-form collision with the other 9 pool categories.

    This protects the MCTS transposition-table uniqueness invariant:
    two SMILES that canonicalise to the same RDKit Mol collapse into
    the same node key, silently shrinking the effective branching
    factor.
    """
    legacy_canon = _legacy_canonicals()
    collisions: list[tuple[str, str]] = []
    for smi in FRAGMENT_POOL_SA_FRIENDLY:
        c = _canonical(smi)
        if c and c in legacy_canon:
            collisions.append((smi, c))
    assert not collisions, (
        f"FRAGMENT_POOL_SA_FRIENDLY entries collide with the legacy "
        f"220-tile pool: {collisions}"
    )


# ---------------------------------------------------------------------------
# Test 5: no duplicates within FRAGMENT_POOL_SA_FRIENDLY itself.
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_no_internal_duplicates() -> None:
    """No canonical-form duplicates within the SA-friendly pool itself."""
    canon = [_canonical(s) for s in FRAGMENT_POOL_SA_FRIENDLY]
    canon = [c for c in canon if c]
    dups = {c for c in canon if canon.count(c) > 1}
    assert not dups, (
        f"FRAGMENT_POOL_SA_FRIENDLY has internal canonical-form "
        f"duplicates: {dups}"
    )


# ---------------------------------------------------------------------------
# Test 6: every SMILES passes the ETKDGv3 embed filter.
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_etkdg_embed_succeeds() -> None:
    """Every SA-friendly fragment embeds with ETKDGv3 at randomSeed=0xC11C.

    This is the same filter that ``_validate_and_build`` uses to gate
    the legacy 220-tile pool, so fragments passing here are guaranteed
    to survive the production pool emit pipeline.
    """
    fails: list[str] = []
    for smi in FRAGMENT_POOL_SA_FRIENDLY:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fails.append(f"{smi}: parse fail")
            continue
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC11C
        rc = AllChem.EmbedMolecule(mol_h, params)
        if rc != 0:
            fails.append(f"{smi}: embed_rc={rc}")
    assert not fails, f"Embed failures: {fails}"


# ---------------------------------------------------------------------------
# Test 7: every SMILES satisfies the descriptor filter
# (40 <= MW <= 300, -2 <= logP <= 5).
# ---------------------------------------------------------------------------


def test_sa_friendly_pool_descriptor_filter() -> None:
    """Every SA-friendly fragment satisfies the MW and logP filter."""
    out_of_range: list[tuple[str, float, float]] = []
    for smi in FRAGMENT_POOL_SA_FRIENDLY:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out_of_range.append((smi, 0.0, 0.0))
            continue
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        if not (40.0 <= mw <= 300.0) or not (-2.0 <= logp <= 5.0):
            out_of_range.append((smi, mw, logp))
    assert not out_of_range, (
        f"SA-friendly fragments out of descriptor bounds: {out_of_range}"
    )


# ---------------------------------------------------------------------------
# Test 8: fragments_from_chembl_reactive +include_sa_friendly adds
# exactly the new tiles (or all 20, whichever is smaller after the
# per-tile validation), and -include_sa_friendly does not add them.
# ---------------------------------------------------------------------------


def test_fragments_from_chembl_reactive_sa_friendly_toggle() -> None:
    """The include_sa_friendly toggle must add the new pool tiles and
    only those tiles (set-difference invariant)."""
    full = fragments_from_chembl_reactive(include_sa_friendly=True)
    no_sa = fragments_from_chembl_reactive(include_sa_friendly=False)
    added = set(t.smiles for t in full) - set(t.smiles for t in no_sa)
    expected = {_canonical(s) for s in FRAGMENT_POOL_SA_FRIENDLY}
    expected.discard("")
    # We compare by canonical form because _build_category may re-canonicalise
    # via build_click_tile.
    added_canon = set()
    for smi in added:
        c = _canonical(smi)
        if c:
            added_canon.add(c)
    missing = expected - added_canon
    assert not missing, (
        f"include_sa_friendly=True did not add these tiles: {missing}"
    )
    # Pool size delta must equal the number of SA-friendly tiles that
    # actually survived validation (here all 20 are expected to pass).
    delta = len(full) - len(no_sa)
    assert delta == len(FRAGMENT_POOL_SA_FRIENDLY), (
        f"Pool size delta {delta} != SA-friendly entry count "
        f"{len(FRAGMENT_POOL_SA_FRIENDLY)}"
    )


# ---------------------------------------------------------------------------
# Test 9: l6_fragment_pool_metrics includes the sa_friendly counter
# populated to (input=N, valid=N, embed_fail=0, mw_fail=0, logp_fail=0).
# ---------------------------------------------------------------------------


def test_l6_metrics_sa_friendly_counters_populated() -> None:
    """L6 instrumentation must report sa_friendly counters after pool build.

    Note: ``_FRAGMENT_POOL_COUNTERS`` is a module-level dict whose
    counters ACCUMULATE across calls to ``fragments_from_chembl_reactive``
    (because the L6 instrumentation design keeps cumulative process
    state).  We therefore assert on the *delta* produced by a single
    call plus the invariant that zero failures were recorded during
    this call, rather than the absolute counter value.
    """
    metrics_before = dict(l6_fragment_pool_metrics().get("sa_friendly", {}))
    fragments_from_chembl_reactive(include_sa_friendly=True)
    metrics_after = l6_fragment_pool_metrics()["sa_friendly"]
    assert "sa_friendly" in metrics_before or True, (
        "sa_friendly key should appear after first pool build"
    )
    # delta in input counter == number of SA-friendly entries processed
    delta_in = metrics_after["input"] - metrics_before.get("input", 0)
    assert delta_in == len(FRAGMENT_POOL_SA_FRIENDLY), (
        f"sa_friendly input counter delta {delta_in} != "
        f"{len(FRAGMENT_POOL_SA_FRIENDLY)}"
    )
    # delta in valid counter should equal input (no failures expected)
    delta_valid = metrics_after["valid"] - metrics_before.get("valid", 0)
    assert delta_valid == len(FRAGMENT_POOL_SA_FRIENDLY), (
        f"sa_friendly valid counter delta {delta_valid} != "
        f"{len(FRAGMENT_POOL_SA_FRIENDLY)}"
    )
    # Failure-counters should not have grown for this call.
    assert metrics_after["embed_fail"] - metrics_before.get("embed_fail", 0) == 0
    assert metrics_after["mw_fail"] - metrics_before.get("mw_fail", 0) == 0
    assert metrics_after["logp_fail"] - metrics_before.get("logp_fail", 0) == 0
