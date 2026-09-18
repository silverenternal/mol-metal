"""Tests for TODO-25 metallodrug-specific proxy metrics.

Covers :mod:`molmetal_lam.priors.metallodrug_property_proxies` (4
heuristic proxies for reduction potential, trans-effect, LFSE, and
Pt-DNA crosslink propensity).  All values are clipped to ``[0, 1]``.
"""
from __future__ import annotations

import pytest

from molmetal_lam.priors.metallodrug_property_proxies import (
    compute_metallodrug_proxies,
    lfse_proxy,
    pt_dna_crosslink_proxy,
    reduction_potential_proxy,
    trans_effect_proxy,
)


# ---------------------------------------------------------------------------
# Test fixtures: known-good metallodrug SMILES + invalid / non-Pt mols
# ---------------------------------------------------------------------------
CISPLATIN = "[H][N]([H])([H])[Pt]([Cl])([Cl])([N]([H])([H])[H])[N]([H])([H])[H]"
# trans-platin (Cl trans to Cl — unusual but parsable)
TRANSPLATIN = "[H][N]([H])([H])[Pt]([Cl])([N]([H])([H])[H])([N]([H])([H])[H])[Cl]"
# Ferrocene (Fe centre, no Pt) — should return 0.0 for Pt-specific proxies.
FERROCENE = "[Fe][c]1cccc1.[Fe]1ccccc1"
# Free ammine — no metal at all.
AMMONIA = "N"
# Non-Pt organic — aspirin.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
# Pt with CN- ligands (very strong-field) — proxy should rate the
# trans-effect donor strength as high.
PT_CN_COMPLEX = "[H][N]([H])([H])[Pt]([C-]#N)([C-]#N)([N]([H])([H])[H])[N]([H])([H])[H]"
# Pt(IV) high-oxidation-state reference (rare in pool; sanity check).
PT_IV = "[Pt](Cl)(Cl)(Cl)(Cl)([NH3])[NH3]"
# Wildcard / invalid SMILES — must NOT raise.
INVALID_CASES = ["", " ", "not-a-smiles", "C*", "*", None]


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------
def _assert_in_unit_interval(value, label):
    assert isinstance(value, float), f"{label}: expected float, got {type(value).__name__}"
    assert 0.0 <= value <= 1.0, f"{label}: out of [0, 1] — got {value}"


# ---------------------------------------------------------------------------
# 1. reduction_potential_proxy
# ---------------------------------------------------------------------------
def test_reduction_potential_returns_float_in_unit_interval():
    v = reduction_potential_proxy(CISPLATIN)
    _assert_in_unit_interval(v, "reduction_potential[Pt(II)]")


def test_reduction_potential_strong_field_higher_than_weak_field():
    """CN-/CO ligands are strong-field → higher reduction_potential score."""
    # cisplatin uses Cl- (weak-field, score 0.30 in our table)
    weak = reduction_potential_proxy(CISPLATIN)
    # PT_CN_COMPLEX uses CN- (strong-field, score 0.95)
    strong = reduction_potential_proxy(PT_CN_COMPLEX)
    assert strong > weak, (
        f"strong-field ligand (CN-) should yield higher reduction_potential "
        f"than weak-field ligand (Cl-); got strong={strong}, weak={weak}"
    )


def test_reduction_potential_returns_zero_for_non_pt_molecule():
    """Aspirin has no Pt → reduction_potential must be 0.0 (not 0.5)."""
    assert reduction_potential_proxy(ASPIRIN) == 0.0


@pytest.mark.parametrize("bad", INVALID_CASES)
def test_reduction_potential_graceful_for_invalid_smiles(bad):
    """Invalid SMILES must NOT raise — should return 0.0 or 0.5 fallback."""
    try:
        v = reduction_potential_proxy(bad)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"raised on invalid SMILES {bad!r}: {exc}")
    assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 2. trans_effect_proxy
# ---------------------------------------------------------------------------
def test_trans_effect_high_for_cn_ligands():
    """PT_CN_COMPLEX has 2 CN- donors → high trans-effect score."""
    v = trans_effect_proxy(PT_CN_COMPLEX)
    assert v > 0.0, f"trans_effect for CN-rich Pt complex should be > 0; got {v}"
    _assert_in_unit_interval(v, "trans_effect[Pt-CN]")


def test_trans_effect_zero_for_no_pt_molecule():
    """Aspirin (no Pt) → 0.0."""
    assert trans_effect_proxy(ASPIRIN) == 0.0


def test_trans_effect_cl_only_low_score():
    """cisplatin has Cl + NH3 (no high-trans donors) → low / 0 score."""
    v = trans_effect_proxy(CISPLATIN)
    _assert_in_unit_interval(v, "trans_effect[Pt-Cl]")
    # cisplatin has no CN/CO/PR3 — expect 0.0 (no high-trans-effect donors)
    assert v == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. lfse_proxy
# ---------------------------------------------------------------------------
def test_lfse_d8_pt_positive_for_cisplatin():
    """Pt(II) d8 → 6 t2g electrons → positive LFSE."""
    v = lfse_proxy(CISPLATIN, metal="Pt", d_electron_count=8)
    _assert_in_unit_interval(v, "lfse[d8 Pt(II)]")
    assert v > 0.0


def test_lfse_d0_returns_zero():
    """d0 (no t2g electrons) → 0.0 by construction."""
    v = lfse_proxy(CISPLATIN, metal="Pt", d_electron_count=0)
    assert v == 0.0


def test_lfse_d10_returns_zero():
    """d10 → no t2g electrons in the band-filling proxy → 0.0."""
    v = lfse_proxy(CISPLATIN, metal="Pt", d_electron_count=10)
    assert v == 0.0


def test_lfse_returns_zero_for_no_pt_molecule():
    """Non-Pt mols → 0.0."""
    assert lfse_proxy(ASPIRIN) == 0.0


# ---------------------------------------------------------------------------
# 4. pt_dna_crosslink_proxy
# ---------------------------------------------------------------------------
def test_pt_dna_crosslink_cisplatin_positive():
    """cisplatin has 2 labile Pt-Cl bonds → positive score."""
    v = pt_dna_crosslink_proxy(CISPLATIN)
    _assert_in_unit_interval(v, "pt_dna_crosslink[Pt-Cl2]")
    assert v > 0.0, f"cisplatin has 2 labile Pt-Cl bonds; expected > 0, got {v}"


def test_pt_dna_crosslink_zero_for_no_pt_molecule():
    """Aspirin (no Pt) → 0.0."""
    assert pt_dna_crosslink_proxy(ASPIRIN) == 0.0


def test_pt_dna_crosslink_graceful_for_invalid_smiles():
    """Invalid SMILES must NOT raise."""
    for bad in ("not-a-smiles", "", " "):
        try:
            v = pt_dna_crosslink_proxy(bad)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"raised on invalid SMILES {bad!r}: {exc}")
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 5. compute_metallodrug_proxies — aggregator
# ---------------------------------------------------------------------------
def test_compute_metallodrug_proxies_returns_all_four_keys():
    out = compute_metallodrug_proxies(CISPLATIN)
    assert set(out) == {"reduction_potential", "trans_effect", "lfse", "pt_dna_crosslink"}
    for k, v in out.items():
        _assert_in_unit_interval(v, f"agg[{k}]")


def test_compute_metallodrug_proxies_aspirin_returns_zeros():
    """Aspirin has no Pt → all four proxies should be 0.0."""
    out = compute_metallodrug_proxies(ASPIRIN)
    assert all(out[k] == 0.0 for k in out), out


@pytest.mark.parametrize("bad", INVALID_CASES)
def test_compute_metallodrug_proxies_graceful_for_invalid_smiles(bad):
    """The aggregator must NOT raise on invalid SMILES."""
    try:
        out = compute_metallodrug_proxies(bad)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"aggregator raised on invalid SMILES {bad!r}: {exc}")
    assert set(out) == {"reduction_potential", "trans_effect", "lfse", "pt_dna_crosslink"}
    for v in out.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 6. Bounds regression — explicit [0, 1] clip for every proxy × every fixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smi",
    [CISPLATIN, TRANSPLATIN, FERROCENE, ASPIRIN, AMMONIA, PT_CN_COMPLEX, PT_IV],
)
def test_all_proxies_in_unit_interval_for_each_fixture(smi):
    """No proxy exceeds [0, 1] for any of our fixtures."""
    for fn, label in (
        (reduction_potential_proxy, "reduction_potential"),
        (trans_effect_proxy, "trans_effect"),
        (lfse_proxy, "lfse"),
        (pt_dna_crosslink_proxy, "pt_dna_crosslink"),
    ):
        try:
            v = fn(smi)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{label}({smi!r}) raised: {exc}")
        _assert_in_unit_interval(v, f"{label}({smi!r})")