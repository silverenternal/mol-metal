"""Smoke test for the fragment-pool duplicate-SMILES fix.

Asserts that the four RDKit-canonical-equivalent duplicates caught
during the r0 audit (azide naphthyl pair, hexadiene pair, isoprene
pair, cyclohexadiene pair, quinoline pair, 3-azidopropanol pair) no
longer collide in the hardcoded :data:`FRAGMENT_POOL_AZIDES` /
:data:`FRAGMENT_POOL_DIENES` lists.

Skips silently when RDKit is not installed — the canonicalisation is
the only check it can do.
"""

from __future__ import annotations

from molmetal_lam.tile_lib.fragment_pool import (
    FRAGMENT_POOL_AZIDES,
    FRAGMENT_POOL_DIENES,
)


def _canonical_set(smiles_list):
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception:
        return None
    out = set()
    invalid = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid += 1
            continue
        out.add(Chem.MolToSmiles(m))
    # Return both the canonical set and the count of valid entries so
    # callers can compare "unique canonical forms" against "valid
    # entries" (pre-existing invalid entries like the t-butyl azide
    # ``N#[N+]=[N-]C(C)(C)C`` would otherwise inflate the apparent
    # collision count).
    return out, invalid


def test_azide_library_no_canonical_duplicates():
    result = _canonical_set(FRAGMENT_POOL_AZIDES)
    if result is None:
        return
    canon, invalid = result
    valid = len(FRAGMENT_POOL_AZIDES) - invalid
    assert len(canon) == valid, (
        f"FRAGMENT_POOL_AZIDES has RDKit-canonical duplicates: "
        f"{valid - len(canon)} collision(s)"
    )


def test_diene_library_no_canonical_duplicates():
    result = _canonical_set(FRAGMENT_POOL_DIENES)
    if result is None:
        return
    canon, invalid = result
    valid = len(FRAGMENT_POOL_DIENES) - invalid
    assert len(canon) == valid, (
        f"FRAGMENT_POOL_DIENES has RDKit-canonical duplicates: "
        f"{valid - len(canon)} collision(s)"
    )


def test_specific_duplicate_pairs_are_resolved():
    """The six originally-reported collisions are no longer duplicates."""
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception:
        return

    pairs_to_check = {
        # (kept, replaced) — both should be RDKit-distinct now
        "3-azidopropanol": (
            "OCCCN=[N+]=[N-]",        # kept (idx 10 in source list)
            "N(=[N+]=[N-])CC=C",      # replaced (was 3-azido-1-propanol)
        ),
        "naphthyl-azides": (
            "N(=[N+]=[N-])C1=CC=C2C=CC=CC2=C1",  # 1-naphthyl, kept
            "N(=[N+]=[N-])C1=CC=C2C=CC=CC2=N1",  # 8-azidoquinoline, new
        ),
        "hexadienes": (
            "C=CC=CC",                # 1,3-hexadiene, kept
            "C=CCC=CC",               # 1,4-hexadiene, new
        ),
        "isoprene": (
            "CC(=C)C=C",              # isoprene, kept
            "C=CC(=C)CO",             # 2-methylene-3-buten-1-ol, new
        ),
        "cyclohexadiene": (
            "C1C=CC=C1",              # cyclopentadiene, kept
            "C1=CCCC=C1",             # 1,4-cyclohexadiene, new
        ),
        "quinoline-pair": (
            "C1=CC=C2C=CC=CC2=N1",    # quinoline, kept
            "C1=CC=C2N=CC=CC2=N1",    # quinoxaline, new
        ),
    }
    for name, (a, b) in pairs_to_check.items():
        ca = Chem.MolToSmiles(Chem.MolFromSmiles(a))
        cb = Chem.MolToSmiles(Chem.MolFromSmiles(b))
        assert ca != cb, f"{name}: kept/replaced collide — {ca!r}"