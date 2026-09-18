"""Tests for ``molmetal_lam.sbdd_env.retrosynthesis``.

These tests verify the **measured** synthesis-success metric that
replaces the previous hardcoded ``synthesis_success = 1.0`` for Lambda.

Coverage
--------
1. **test_retrosynthesize_cuaac_product** — A canonical CuAAC
   triazole product (``Cc1cn(C)nn1``) is reversible: at least one
   click rule returns a non-empty educt set.
2. **test_retrosynthesize_random_mol** — A random aliphatic
   (``CCCCCC``, ``benzene``, ``ethanol``) returns ``False`` because
   none of them contain a click-reaction signature.
3. **test_synthesis_success_rate_lambda** — When run on the 12
   standard click tiles, the measured fraction is >= 0.80.  The
   "non-click" partner tiles (cyclopentadiene, maleimide, MVK)
   fail on the strict SMARTS but pass on the structural handlers,
   so we expect a uniform pass rate on the 12 tiles.
"""
from __future__ import annotations

import pytest

from molmetal_lam.sbdd_env.retrosynthesis import (
    retrosynthesize,
    retrosynthesize_with_report,
    synthesis_success_rate,
)


# ---------------------------------------------------------------------------
# 1. A CuAAC product must be reversible.
# ---------------------------------------------------------------------------
def test_retrosynthesize_cuaac_product():
    # 1,4-dimethyl-1,2,3-triazole — canonical CuAAC product of
    # methyl azide + propyne.
    smi = "Cc1cn(C)nn1"
    ok, rules = retrosynthesize_with_report(smi)
    assert ok, f"CuAAC product {smi} should be reversible, got rules={rules}"
    # At least one of the triazole rules must fire.
    assert any(r in {"CuAAC", "SPAAC"} for r in rules), (
        f"expected CuAAC or SPAAC rule, got {rules}"
    )


# ---------------------------------------------------------------------------
# 2. A random aliphatic / aromatic must NOT be reversible.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smi, label",
    [
        ("CCCCCC",     "hexane"),
        ("c1ccccc1",   "benzene"),
        ("CCO",        "ethanol"),
        ("CC(=O)O",    "acetic acid"),
    ],
)
def test_retrosynthesize_random_mol(smi, label):
    """Molecules without a click signature must return False."""
    assert retrosynthesize(smi) is False, (
        f"random molecule {label!r} ({smi}) should not be click-reversible"
    )


# ---------------------------------------------------------------------------
# 3. Lambda's *generated* candidates must pass with high rate.
# ---------------------------------------------------------------------------
def _generate_lambda_products(n_max: int = 12) -> list:
    """Generate ``n_max`` real Lambda candidates by firing CuAAC on
    the azide × alkyne pairs from the standard tile library.

    These are the actual *products* of the Lambda design loop (i.e.
    1,4-disubstituted 1,2,3-triazoles), not the educt tiles.
    """
    from molmetal_lam.tile_lib import AZIDES, ALKYNES, PARTNERS
    from molmetal_lam.reactions.beta_reductions import (
        cuaac, diels_alder, thiol_ene,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    out: list = []
    # CuAAC on every azide × alkyne pair.
    for az_t in AZIDES:
        az = MoleculeClosedTerm.from_smiles(az_t.smiles, embed_3d=False)
        for ak_t in ALKYNES:
            ak = MoleculeClosedTerm.from_smiles(ak_t.smiles, embed_3d=False)
            try:
                prods = cuaac(az, ak)
            except Exception:
                prods = []
            for p in prods:
                out.append(p.canonical_smiles())
                if len(out) >= n_max:
                    return out[:n_max]
    # Diels-Alder on partner pairs (diene + dienophile).
    diene = next((t for t in PARTNERS if "diene" in t.functional_groups), None)
    dien = next((t for t in PARTNERS if "dienophile" in t.functional_groups), None)
    if diene is not None and dien is not None:
        a = MoleculeClosedTerm.from_smiles(diene.smiles, embed_3d=False)
        b = MoleculeClosedTerm.from_smiles(dien.smiles, embed_3d=False)
        try:
            for p in diels_alder(a, b):
                out.append(p.canonical_smiles())
        except Exception:
            pass
    return out[:n_max]


def test_synthesis_success_rate_lambda():
    """Run on Lambda-generated click products; expect >= 80% pass rate.

    Lambda generates *products* of click reactions (1,4-triazoles,
    cyclohexenes, etc.), so we apply the same rule to actual product
    SMILES that the design loop would emit.  A rate >= 0.80 leaves
    room for canonicalisation edge cases.
    """
    smis = _generate_lambda_products(n_max=10)
    assert len(smis) >= 5, f"expected >=5 products, got {len(smis)}"
    rate = synthesis_success_rate(smis)
    assert rate >= 0.80, f"Lambda product rate {rate:.3f} below 0.80"
    failures = [s for s in smis if not retrosynthesize(s)]
    print(f"\n[debug] Lambda retrosynthesis rate={rate:.3f}, "
        f"n_products={len(smis)}, failing={failures}")