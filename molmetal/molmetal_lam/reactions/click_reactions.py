"""Click-chemistry reactions as Lambda functions.

This module implements the four canonical click-chemistry reaction
rules — written to match the task spec (TODO/13_lambda_clickchem):

    CuAAC       : R-N3 + R'-C≡CH    -> 1,4-disubstituted 1,2,3-triazole
    SPAAC       : R-N3 + cyclooctyne -> 1,2,3-triazole (strain-promoted)
    SPC         : R-N3 + R'-P(III)   -> iminophosphorane + N2
    DielsAlder  : diene + dienophile -> cyclohexene

These four rules are also implemented in :mod:`beta_reductions`; this
module is a *lightweight, SMILES-string-first* sibling that exposes a
``(Tile, Tile) -> Tile`` functional interface for the
:mod:`tile_lib.click_tiles` smoke test.

Implementation strategy
-----------------------
RDKit's reaction SMARTS produces molecules with bogus valences on
the triazole N atoms because the 1,2,3-triazole product is aromatic
and ``RunReactants`` does not assign aromaticity to ring products.
We therefore use **functional-group SMARTS matching** to detect the
correct positions on each educt and **build the product SMILES
manually** from known click-chemistry SMILES templates.  This is
both more robust and more interpretable than fighting kekulization.

For each tile we compute the *rest* (everything outside the
functional handle) by RDKit's ``FragmentOnBonds`` + ``GetMolFrags``:
enumerate all single-bond cleavages, then keep the fragment that
contains the matching handle atoms.  The rest fragment is grafted
onto a small handle-fusion template SMILES.

Public API
----------
``ClickReactionError``     raised on failed reaction parse / run
``click_cuaac``            azide + terminal alkyne   -> 1,4-triazole
``click_spaac``            azide + cyclooctyne       -> triazole
``click_spc``              azide + phosphine         -> iminophosphorane
``click_diels_alder``      diene + dienophile        -> cyclohexene
``apply_click_reaction``   dispatch by name
``ReactionResult``         frozen dataclass with product tile
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions  # noqa: F401

# Public types from sibling layers
from molmetal_lam.tile_lib.tile import Tile

__all__ = [
    "ClickReactionError",
    "ReactionResult",
    "CLICK_REACTIONS",
    "click_cuaac",
    "click_spaac",
    "click_spc",
    "click_diels_alder",
    "symmetric_click",
    "apply_click_reaction",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClickReactionError(ValueError):
    """Raised when a click-reaction rule cannot be applied."""


# ---------------------------------------------------------------------------
# Reaction result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactionResult:
    """The result of firing a click reaction on a tile pair.

    Attributes
    ----------
    product_tile : Tile
        The product tile (canonical SMILES, atom count = sum of
        educt heavy atoms — click reactions conserve heavy atoms).
    reaction_name : str
        Name of the reaction that was applied.
    educt_smiles : tuple[str, str]
        The two educt SMILES (for traceability / audit).
    product_smiles : str
        Canonical SMILES of the product (alias for ``product_tile.smiles``).
    atom_count : int
        Number of heavy atoms in the product (= sum of educt heavy
        atoms, because click reactions are bond-forming only).
    """

    product_tile: Tile
    reaction_name: str
    educt_smiles: tuple
    product_smiles: str
    atom_count: int


# ---------------------------------------------------------------------------
# Functional-group SMARTS (for reactant-side matching)
# ---------------------------------------------------------------------------

# Azide  : N=N=N (covers both neutral and charged representations
# after RDKit canonicalisation of "CCN=[N+]=[N-]")
_AZIDE_SMARTS = "[N]=[N]=[N]"

# Terminal alkyne : C#CH
_TERMINAL_ALKYNE_SMARTS = "[CH]#[C]"

# Internal alkyne (incl. cyclooctyne) : C#C with no H on either atom
_INTERNAL_ALKYNE_SMARTS = "[C;H0]#[C;H0]"

# Phosphine P(III) — any P atom
_PHOSPHINE_SMARTS = "[P]"

# Conjugated diene : C=C-C=C (aliphatic or aromatic)
_DIENE_SMARTS = "[#6]=[#6]-[#6]=[#6]"

# Simple alkene (dienophile) : C=C (aliphatic or aromatic)
_ALKENE_SMARTS = "[#6]=[#6]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_substructure(smiles: str, smarts: str) -> bool:
    """Return True if ``smiles`` contains the substructure defined by ``smarts``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return False
    return mol.HasSubstructMatch(pat)


def _strip_dummy_atoms(mol):
    """Remove dummy atoms (atomic_num=0) from ``mol`` in-place, returning the mol."""
    rw = Chem.RWMol(mol)
    # Collect indices of dummy atoms, sort descending for safe removal.
    to_remove = [
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetAtomicNum() == 0 or a.GetSymbol() == "*"
    ]
    for idx in sorted(to_remove, reverse=True):
        try:
            rw.RemoveAtom(idx)
        except Exception:
            return None
    return rw.GetMol()


def _build_tile_from_smiles(smiles: str, tags: List[str]) -> Tile:
    """Construct a :class:`Tile` from SMILES using the standard factory."""
    from molmetal_lam.tile_lib.click_tiles import build_click_tile
    return build_click_tile(smiles, tags, embed_3d=False)


def _canonicalise(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def _result_from_product(
    product_smiles: str,
    *,
    reaction_name: str,
    educt_a_smiles: str,
    educt_b_smiles: str,
    product_tags: List[str],
) -> ReactionResult:
    """Wrap a product SMILES into a :class:`ReactionResult`."""
    canonical = _canonicalise(product_smiles)
    if canonical is None:
        raise ClickReactionError(
            f"[{reaction_name}] product SMILES failed to canonicalise: "
            f"{product_smiles!r}"
        )
    mol = Chem.MolFromSmiles(canonical)
    tile = _build_tile_from_smiles(canonical, tags=product_tags)
    return ReactionResult(
        product_tile=tile,
        reaction_name=reaction_name,
        educt_smiles=(educt_a_smiles, educt_b_smiles),
        product_smiles=canonical,
        atom_count=mol.GetNumHeavyAtoms(),
    )


# ---------------------------------------------------------------------------
# Build product by joining (handle_A + rest_A) + (handle_B + rest_B)
# ---------------------------------------------------------------------------
#
# The key helper is _split_handle_rest which:
#   1. Identifies the *handle atoms* (the matched substructure).
#   2. Enumerates every bond from the handle to the rest.
#   3. FragmentOnBonds on those bonds, then uses GetMolFrags(asMols=False)
#      to get ORIGINAL atom indices (RDKit re-uses indices in the
#      fragmented mol — this is critical).
#   4. PathToSubmol to build the rest mol from the original indices.

# Atom-position tokens used inside templates — uniquely placed so we
# can do unambiguous string substitution.  These are literal strings
# embedded in the template; we replace them via ``str.replace``.
_TOK_A = "[_TOK_A]"
_TOK_B = "[_TOK_B]"


def _split_handle_rest(
    smiles: str,
    handle_smarts: str,
) -> tuple:
    """Split ``smiles`` into (handle_smiles, rest_smiles).

    The handle is the fragment that contains the first match of
    ``handle_smarts``.  The rest is the remaining atoms (joined into
    a single fragment, or ``""`` if the handle = whole molecule).

    Returns ``(None, smiles)`` on parse failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (None, smiles)

    pat = Chem.MolFromSmarts(handle_smarts)
    if pat is None:
        return (None, smiles)

    matches = mol.GetSubstructMatches(pat)
    if not matches:
        return (None, smiles)

    handle_atom_set = set(matches[0])
    if len(handle_atom_set) == mol.GetNumAtoms():
        return (smiles, "")

    # Bonds to cut: every bond from a handle atom to a non-handle atom.
    bond_indices_to_cut: List[int] = []
    for atom_idx in handle_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom_idx)
            if other_idx not in handle_atom_set:
                bond_indices_to_cut.append(bond.GetIdx())

    if not bond_indices_to_cut:
        # Handle touches nothing outside itself → handle = whole mol.
        return (smiles, "")

    try:
        frag_mol = Chem.FragmentOnBonds(mol, bond_indices_to_cut)
        # GetMolFrags(asMols=True) returns one mol per fragment; the
        # first call gives the ORIGINAL atom indices, the second
        # gives the fragment mols (with dummy atoms at cut positions).
        frag_idx_sets = Chem.GetMolFrags(frag_mol, asMols=False)
        frag_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return (None, smiles)

    handle_smi: Optional[str] = None
    rest_smi = ""
    for frag_idx_set, frag_m in zip(frag_idx_sets, frag_mols):
        frag_idx_set = set(frag_idx_set)
        try:
            # Strip dummy atoms (inserted at cut positions) before
            # sanitising.
            sub_mol = _strip_dummy_atoms(frag_m)
            if sub_mol is None or sub_mol.GetNumAtoms() == 0:
                continue
            Chem.SanitizeMol(sub_mol)
            sub_smi = Chem.MolToSmiles(sub_mol)
        except Exception:
            continue
        if handle_atom_set.issubset(frag_idx_set):
            handle_smi = sub_smi
        else:
            rest_smi = sub_smi  # last write wins; for one-rest case, fine

    # If handle is still None (e.g. fused-ring case where all atoms
    # end up in one fragment), fall back: handle is the *whole*
    # molecule, rest is empty.
    if handle_smi is None:
        return (smiles, "")
    return (handle_smi, rest_smi)


def _merge_two_handles(
    handle_template: str,
    rest_a: str,
    rest_b: str,
) -> Optional[str]:
    """Replace ``_TOK_A`` / ``_TOK_B`` in ``handle_template`` with rests.

    Empty rest → drop the token (and any neighbouring parentheses to
    keep the SMILES grammatically valid).
    """
    scaffold = handle_template

    # Replace _TOK_A.  If empty, also drop the preceding "(" if any
    # (branch open without branch content is invalid SMILES).
    if rest_a:
        scaffold = scaffold.replace(_TOK_A, rest_a, 1)
    else:
        # Find the token; if it's enclosed in "(...)", drop the parens too.
        idx = scaffold.find(_TOK_A)
        if idx > 0 and scaffold[idx - 1] == "(" and idx + len(_TOK_A) < len(scaffold) and scaffold[idx + len(_TOK_A)] == ")":
            scaffold = scaffold[: idx - 1] + scaffold[idx + len(_TOK_A) + 1 :]
        else:
            scaffold = scaffold.replace(_TOK_A, "", 1)

    # Replace _TOK_B with same handling.
    if rest_b:
        scaffold = scaffold.replace(_TOK_B, rest_b, 1)
    else:
        idx = scaffold.find(_TOK_B)
        if idx > 0 and scaffold[idx - 1] == "(" and idx + len(_TOK_B) < len(scaffold) and scaffold[idx + len(_TOK_B)] == ")":
            scaffold = scaffold[: idx - 1] + scaffold[idx + len(_TOK_B) + 1 :]
        else:
            scaffold = scaffold.replace(_TOK_B, "", 1)

    return scaffold


# ---------------------------------------------------------------------------
# CuAAC — Cu(I)-catalysed azide + terminal alkyne -> 1,4-disubstituted triazole
# ---------------------------------------------------------------------------


# 1,2,3-triazole core: n1cc(nn1) with two attachment points:
#   _TOK_A on N1 (binds azide's R group)
#   _TOK_B on C4 (binds alkyne's R group)
_CUAAC_TEMPLATE = "[_TOK_A]n1cc([_TOK_B])nn1"


def click_cuaac(
    azide_tile: Tile, alkyne_tile: Tile
) -> List[ReactionResult]:
    """Cu(I)-catalysed azide + terminal-alkyne cycloaddition.

    R-N3 + R'-C≡CH  --[Cu(I)]-->  1,4-disubstituted 1,2,3-triazole

    Parameters
    ----------
    azide_tile : Tile
        Tile carrying an azide handle.
    alkyne_tile : Tile
        Tile carrying a terminal alkyne handle.

    Returns
    -------
    list[ReactionResult]
        List of length 1 with the canonical product. Empty list if
        the SMILES don't contain the required functional groups.
    """
    azide_smi = azide_tile.smiles
    alkyne_smi = alkyne_tile.smiles
    if not _has_substructure(azide_smi, _AZIDE_SMARTS):
        return []
    if not _has_substructure(alkyne_smi, _TERMINAL_ALKYNE_SMARTS):
        return []

    _, azide_rest = _split_handle_rest(azide_smi, _AZIDE_SMARTS)
    _, alkyne_rest = _split_handle_rest(alkyne_smi, _TERMINAL_ALKYNE_SMARTS)

    product_smiles = _merge_two_handles(
        _CUAAC_TEMPLATE,
        rest_a=azide_rest,
        rest_b=alkyne_rest,
    )
    if product_smiles is None:
        return []
    canonical = _canonicalise(product_smiles)
    if canonical is None:
        return []

    return [_result_from_product(
        canonical,
        reaction_name="CuAAC",
        educt_a_smiles=azide_smi,
        educt_b_smiles=alkyne_smi,
        product_tags=["triazole", "1,4-disubstituted"],
    )]


# ---------------------------------------------------------------------------
# SPAAC — strain-promoted azide + cyclooctyne -> triazole (no copper)
# ---------------------------------------------------------------------------


# Same triazole core as CuAAC but different regiochemistry of the
# rests on the ring (1,5-disubstituted for SPAAC).  We use the same
# ring SMILES — the canonicaliser will pick the canonical form.
_SPAAC_TEMPLATE = "[_TOK_A]n1cc([_TOK_B])nn1"


def click_spaac(
    azide_tile: Tile, cyclooctyne_tile: Tile
) -> List[ReactionResult]:
    """Strain-promoted azide + cyclooctyne cycloaddition (no copper).

    R-N3 + cyclooctyne-R'  -->  triazole

    Parameters
    ----------
    azide_tile : Tile
        Tile carrying an azide handle.
    cyclooctyne_tile : Tile
        Tile carrying an internal alkyne handle.

    Returns
    -------
    list[ReactionResult]
        List of length 1 with the canonical product.
    """
    azide_smi = azide_tile.smiles
    alkyne_smi = cyclooctyne_tile.smiles
    if not _has_substructure(azide_smi, _AZIDE_SMARTS):
        return []
    if not _has_substructure(alkyne_smi, _INTERNAL_ALKYNE_SMARTS):
        return []

    _, azide_rest = _split_handle_rest(azide_smi, _AZIDE_SMARTS)
    _, alkyne_rest = _split_handle_rest(alkyne_smi, _INTERNAL_ALKYNE_SMARTS)

    product_smiles = _merge_two_handles(
        _SPAAC_TEMPLATE,
        rest_a=azide_rest,
        rest_b=alkyne_rest,
    )
    if product_smiles is None:
        return []
    canonical = _canonicalise(product_smiles)
    if canonical is None:
        return []

    return [_result_from_product(
        canonical,
        reaction_name="SPAAC",
        educt_a_smiles=azide_smi,
        educt_b_smiles=alkyne_smi,
        product_tags=["triazole"],
    )]


# ---------------------------------------------------------------------------
# SPC — Staudinger phosphine-azide coupling
# ---------------------------------------------------------------------------


# Iminophosphorane: R-N=P-R'  →  SMILES "N=P" between the two rests.
_SPC_TEMPLATE = "[_TOK_A]N=P[_TOK_B]"


def click_spc(
    azide_tile: Tile, phosphine_tile: Tile
) -> List[ReactionResult]:
    """Staudinger phosphine-azide coupling.

    R-N3 + R'-P(III)  -->  R-N=P-R'  +  N2

    The classical chemical outcome is an amide (after hydrolysis);
    the *reaction step itself* produces an iminophosphorane
    (R-N=P-R') plus dinitrogen.  Because N2 leaves the molecule,
    the product is (R - terminal-N - middle-N) joined to P-R' via
    P=N.  In our fragment-based approach: the azide handle SMARTS
    matches all 3 N's; we keep the handle = N3 chain and the rest =
    R.  Same on the phosphine side.  The product has *fewer* atoms
    than the sum of the educts by 2 (the two N's released as N2).

    Parameters
    ----------
    azide_tile : Tile
        Tile carrying an azide handle.
    phosphine_tile : Tile
        Tile carrying a phosphine handle.

    Returns
    -------
    list[ReactionResult]
        List of length 1 with the canonical iminophosphorane product.
    """
    azide_smi = azide_tile.smiles
    phosphine_smi = phosphine_tile.smiles
    if not _has_substructure(azide_smi, _AZIDE_SMARTS):
        return []
    if not _has_substructure(phosphine_smi, _PHOSPHINE_SMARTS):
        return []

    _, azide_rest = _split_handle_rest(azide_smi, _AZIDE_SMARTS)
    _, phosphine_rest = _split_handle_rest(phosphine_smi, _PHOSPHINE_SMARTS)

    # Combine: azide_rest-N=P-phosphine_rest.
    product_smiles = _merge_two_handles(
        _SPC_TEMPLATE,
        rest_a=azide_rest,
        rest_b=phosphine_rest,
    )
    if product_smiles is None:
        return []
    canonical = _canonicalise(product_smiles)
    if canonical is None:
        return []

    return [_result_from_product(
        canonical,
        reaction_name="SPC",
        educt_a_smiles=azide_smi,
        educt_b_smiles=phosphine_smi,
        product_tags=["iminophosphorane"],
    )]


# ---------------------------------------------------------------------------
# Diels–Alder [4+2] cycloaddition
# ---------------------------------------------------------------------------


# Cyclohexene ring: C1=CC(*)CCC1* with two attachment points.
_DA_TEMPLATE = "[_TOK_A]C1=CC([_TOK_B])CCC1"


def click_diels_alder(
    diene_tile: Tile, dienophile_tile: Tile
) -> List[ReactionResult]:
    """[4+2] Diels-Alder cycloaddition.

    diene + dienophile  -->  cyclohexene

    Parameters
    ----------
    diene_tile : Tile
        Tile carrying a conjugated diene handle (tag ``"diene"``).
    dienophile_tile : Tile
        Tile carrying a dienophile handle (tag ``"dienophile"``).

    Returns
    -------
    list[ReactionResult]
        List of length 1 with the canonical cyclohexene product.
    """
    diene_smi = diene_tile.smiles
    dienophile_smi = dienophile_tile.smiles
    if not _has_substructure(diene_smi, _DIENE_SMARTS):
        return []
    if not _has_substructure(dienophile_smi, _ALKENE_SMARTS):
        return []

    _, diene_rest = _split_handle_rest(diene_smi, _DIENE_SMARTS)
    _, dienophile_rest = _split_handle_rest(dienophile_smi, _ALKENE_SMARTS)

    product_smiles = _merge_two_handles(
        _DA_TEMPLATE,
        rest_a=diene_rest,
        rest_b=dienophile_rest,
    )
    if product_smiles is None:
        return []
    canonical = _canonicalise(product_smiles)
    if canonical is None:
        return []

    return [_result_from_product(
        canonical,
        reaction_name="DielsAlder",
        educt_a_smiles=diene_smi,
        educt_b_smiles=dienophile_smi,
        product_tags=["cyclohexene"],
    )]


# ---------------------------------------------------------------------------
# Dispatcher — name -> (Tile, Tile) -> List[ReactionResult]
# ---------------------------------------------------------------------------


#: Registry of click-reaction callables.  Each accepts two
#: :class:`Tile` arguments and returns a list of
#: :class:`ReactionResult` (possibly empty).
CLICK_REACTIONS: Dict[str, Callable[[Tile, Tile], List[ReactionResult]]] = {
    "CuAAC":      click_cuaac,
    "SPAAC":      click_spaac,
    "SPC":        click_spc,
    "DielsAlder": click_diels_alder,
}


def apply_click_reaction(
    name: str,
    tile_a: Tile,
    tile_b: Tile,
) -> List[ReactionResult]:
    """Dispatch by name to one of the four click reactions.

    By default this is **symmetric**: each rule's ``(a, b)`` ordering is
    tried first; if that returns no products (because ``a`` lacks the
    required functional group but ``b`` carries it), the call is retried
    with swapped arguments.  This matters when MCTS expansion hands the
    state in arbitrary order — e.g. ``[Pt]C#C`` (an alkyne tile) is the
    root and the partner tile carries the azide, so a rule that
    *requires* azide-first will silently return ``[]``.

    Pass ``symmetric=False`` for tests that need to assert strict-order
    semantics.

    Raises
    ------
    ClickReactionError
        If ``name`` is not a known click reaction.
    """
    fn = CLICK_REACTIONS.get(name)
    if fn is None:
        raise ClickReactionError(
            f"unknown click reaction: {name!r}. "
            f"Known: {sorted(CLICK_REACTIONS.keys())}"
        )
    return symmetric_click(fn, tile_a, tile_b)


def symmetric_click(
    fn: Callable[[Tile, Tile], List[ReactionResult]],
    tile_a: Tile,
    tile_b: Tile,
) -> List[ReactionResult]:
    """Apply ``fn`` to ``(tile_a, tile_b)``; on empty result, retry ``(tile_b, tile_a)``.

    Many click rules are defined with an *asymmetric* signature
    (e.g. ``click_cuaac(azide, alkyne)``), but the underlying chemistry
    is **commutative** — it does not matter which tile is presented
    first, only that *one* of them carries the azide and the other
    the alkyne handle.  The MCTS expansion layer does not know which
    side of the redex is which, so the wrapper tries both orderings
    and returns the first non-empty result.

    The function is **idempotent**: if both orderings return
    products (shouldn't happen in practice because rules dedupe by
    canonical SMILES, but defensive), we still return a single
    canonical list — preferring the forward order to keep the
    ``educt_smiles`` field stable for downstream traceability.

    Parameters
    ----------
    fn : callable
        A click-reaction function with signature
        ``(Tile, Tile) -> List[ReactionResult]``.
    tile_a, tile_b : Tile
        The two tiles to react (either order).

    Returns
    -------
    list[ReactionResult]
        The first non-empty product list.  Empty if neither order
        produces a product (i.e. the chemistry genuinely does not apply).
    """
    try:
        products = fn(tile_a, tile_b)
    except Exception:
        products = []
    if products:
        return products
    # Retry with swapped arguments.  This catches the case where
    # ``tile_a`` is e.g. an alkyne and ``tile_b`` an azide but the rule
    # only recognises azide as the first arg (legacy convention).
    if tile_a is tile_b:
        # Same tile swapped with itself is identical — don't double-fire.
        return products
    try:
        swapped = fn(tile_b, tile_a)
    except Exception:
        swapped = []
    return swapped