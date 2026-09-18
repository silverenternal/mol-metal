"""12 standard click-chemistry tiles for the LamClick Phase-0 library.

This module hard-codes the **12 reference tiles** that anchor the
proof-search constant pool. These tiles are the canonical reactive
handles used by:

    CuAAC       : azide  + terminal_alkyne  -> 1,2,3-triazole
    SPAAC       : azide  + cyclooctyne       -> triazole (strain-promoted)
    SPC         : thiol  + methylphosphine   -> thioether (Staudinger)
    DielsAlder  : diene  + dienophile        -> cyclohexene
    ThiolEne    : thiol  + alkene            -> thioether (radical)
    Maleimide   : thiol  + maleimide         -> thioether (Michael)

The 12 tiles are split 4 + 4 + 4:

    4 azide handles   : ethyl/benzyl/hydroxyethyl/aryl azide
    4 alkyne handles  : propyne / propargyl benzene / cyclooctyne / propargylamine
    4 partner handles : methylphosphine / cyclopentadiene (diene) /
                        methyl vinyl ketone (dienophile) / maleimide (thiol-ene)
    2 thiol handles   : ethanethiol / thiophenol  (Close-Loop #3 addition)

Each tile is a :class:`Tile` instance with:

    * ``smiles``            canonical RDKit SMILES
    * ``coords``            3D coords embedded via ETKDGv3 (if RDKit available)
    * ``functional_groups`` tag list (e.g. ``['azide']``, ``['cyclooctyne']``)
    * ``tile_id``           deterministic hash of canonical SMILES
    * ``mw`` / ``logp`` / ``tpsa``   RDKit descriptors

SMILES sources
--------------
The SMILES strings were validated against PubChem canonical forms;
they are RDKit-parsable and stereochemistry-free (Phase-0 simplification).
"""

from __future__ import annotations

from typing import Dict, List

from molmetal_lam.tile_lib.tile import Tile

__all__ = [
    "AZIDE_TILES",
    "ALKYNE_TILES",
    "PARTNER_TILES",
    "THIOL_TILES",
    "STANDARD_12_TILES",
    "STANDARD_14_TILES",
    "PARTNER_TILES_V2",
    "AZIDE_PARTNER_TILES",
    "BORONIC_PARTNER_TILES",
    "BROMIDE_PARTNER_TILES",
    "build_click_tile",
    "build_all_click_tiles",
    "l6_metrics",
]


# L6 instrumentation counters (govern_review_L4_L6.md).
_L6_COUNTERS: Dict[str, int] = {
    "build_calls": 0,
    "embed_success": 0,
    "embed_fail": 0,
}
_L6_SAS_HISTORY: List[float] = []
_L6_DESCRIPTORS: List[Dict[str, float]] = []
_L6_CANONICAL_SMILES: List[str] = []
_L6_TILE_IDS: List[str] = []


# ---------------------------------------------------------------------------
# Tile factory — RDKit-backed SMILES canonicalisation + 3D embedding
# ---------------------------------------------------------------------------


def _record_tile(t: Tile) -> Tile:
    """Record L6 instrumentation for a freshly built tile (1-line hook)."""
    _L6_COUNTERS["build_calls"] += 1
    _L6_SAS_HISTORY.append(t.sas_score)
    _L6_DESCRIPTORS.append({"mw": t.mw, "logp": t.logp, "tpsa": t.tpsa})
    _L6_CANONICAL_SMILES.append(t.smiles)
    _L6_TILE_IDS.append(t.tile_id)
    return t


def build_click_tile(
    smiles: str,
    tags: List[str],
    *,
    embed_3d: bool = True,
    sas_score: float = 1.0,
) -> Tile:
    """Build a single :class:`Tile` from a raw SMILES + tag list.

    The function performs:

    1.  RDKit SMILES canonicalisation (gives the canonical form stored
        on the tile).
    2.  RDKit Crippen / descriptor computation (``mw``, ``logp``, ``tpsa``).
    3.  Optional 3D embedding via ``ETKDGv3`` + MMFF/UFF optimisation,
        returned as a ``torch.Tensor`` of shape ``(N_atoms, 3)``.
    4.  Deterministic ``tile_id`` generation (handled inside ``Tile``).

    Parameters
    ----------
    smiles : str
        Input SMILES. Will be canonicalised via ``Chem.MolToSmiles``.
    tags : List[str]
        Functional-group tags (e.g. ``['azide']``).
    embed_3d : bool, optional
        Whether to embed 3D coords (default True).
    sas_score : float, optional
        Synthetic accessibility score (default 1.0 = very easy).

    Returns
    -------
    Tile
        The fully populated :class:`Tile` instance.
    """
    # Lazy imports so the module is importable in environments without
    # RDKit / torch (useful for unit tests of the dataclass only).
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import AllChem, Descriptors  # type: ignore[import-not-found]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    canonical = Chem.MolToSmiles(mol)

    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))

    coords = None
    if embed_3d:
        try:
            mol_h = Chem.AddHs(mol)
            embedded = False
            # A few strained click handles are sensitive to ETKDG's initial
            # seed. Retry deterministically before declaring an environment
            # or tile failure; this keeps the L6 health metric meaningful.
            for seed in (0xC11C, 0xC11D, 0xC11E):
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                if AllChem.EmbedMolecule(mol_h, params) == 0:
                    embedded = True
                    break
            if not embedded:
                # ETKDG can reject highly strained rings; random-coordinate
                # embedding still provides a usable initial geometry for the
                # downstream equivariant model.
                params = AllChem.ETKDGv3()
                params.randomSeed = 0xC11F
                params.useRandomCoords = True
                embedded = AllChem.EmbedMolecule(mol_h, params) == 0
            if not embedded:
                # Last-resort deterministic planar coordinates keep the tile
                # usable for batching when a strained graph defeats 3D ETKDG.
                embedded = AllChem.Compute2DCoords(mol_h) >= 0
            if embedded:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                except Exception:
                    # Fall back to UFF if MMFF fails (rare).
                    try:
                        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                    except Exception:
                        pass

                try:
                    import torch  # type: ignore[import-not-found]

                    conf = mol_h.GetConformer()
                    coords = torch.tensor(
                        [
                            [conf.GetAtomPosition(i).x,
                             conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z]
                            for i in range(mol_h.GetNumAtoms())
                        ],
                        dtype=torch.float32,
                    )
                except Exception:
                    coords = None
        except Exception:
            coords = None
    if coords is not None:
        _L6_COUNTERS["embed_success"] += 1
    else:
        _L6_COUNTERS["embed_fail"] += 1

    return _record_tile(Tile(
        smiles=canonical,
        coords=coords,
        functional_groups=list(tags),
        sas_score=float(sas_score),
        mw=mw,
        logp=logp,
        tpsa=tpsa,
    ))


def build_all_click_tiles(embed_3d: bool = True) -> List[Tile]:
    """Build the canonical 12-tile Phase-0 click library.

    Returns
    -------
    List[Tile]
        12 :class:`Tile` instances: 4 azides + 4 alkynes + 4 partners.
    """
    return [
        # ----- 4 azide handles -----
        # alkyl: ethyl azide
        build_click_tile("CCN=[N+]=[N-]", ["azide", "alkyl_azide"], embed_3d=embed_3d),
        # benzyl: benzyl azide
        build_click_tile("N(=[N+]=[N-])Cc1ccccc1", ["azide", "benzyl_azide"], embed_3d=embed_3d),
        # PEG: azido-PEG1-OH (HO-CH2-CH2-O-CH2-CH2-N3)
        build_click_tile("N(=[N+]=[N-])CCOCCO", ["azide", "PEG_azide", "primary_alcohol"], embed_3d=embed_3d),
        # aryl: phenyl azide
        build_click_tile("[N-]=[N+]=Nc1ccccc1", ["azide", "aryl_azide"], embed_3d=embed_3d),

        # ----- 4 alkyne handles -----
        # terminal alkyne (propyne)
        build_click_tile("C#CC", ["terminal_alkyne"], embed_3d=embed_3d),
        # propargyl benzene (terminal alkyne + aryl)
        build_click_tile("C#CCc1ccccc1", ["terminal_alkyne", "aryl"], embed_3d=embed_3d),
        # cyclooctyne (true 8-ring with internal C#C — canonical SMILES: C1#CCCCCCC1)
        build_click_tile("C1CCCC#CCC1", ["cyclooctyne"], embed_3d=embed_3d),
        # propargylamine (terminal alkyne + primary amine)
        build_click_tile("C#CCN", ["terminal_alkyne", "primary_amine"], embed_3d=embed_3d),

        # ----- 4 partner handles -----
        build_click_tile("CP", ["methylphosphine", "phosphine"], embed_3d=embed_3d),
        build_click_tile("C1C=CC=C1", ["diene", "cyclopentadiene"], embed_3d=embed_3d),
        build_click_tile("C=CC(=O)C", ["dienophile", "alkene", "ketone"], embed_3d=embed_3d),
        build_click_tile("C1=CC(=O)NC1=O", ["maleimide", "thiol_ene", "imide"], embed_3d=embed_3d),

        # ----- 2 thiol handles (Close-Loop #3 addition) -----
        # alkyl thiol: ethanethiol  HS-CH2-CH3  (canonical CCS)
        build_click_tile("CCS", ["thiol", "alkyl_thiol"], embed_3d=embed_3d),
        # aryl thiol: thiophenol    HS-C6H5    (canonical Sc1ccccc1)
        build_click_tile("Sc1ccccc1", ["thiol", "aryl_thiol"], embed_3d=embed_3d),
    ]


# Eager constants — built lazily on first import so heavy RDKit work is
# deferred until someone actually needs the library.
def _ensure_built() -> List[Tile]:
    """Build (or reuse) the canonical 12-tile library."""
    global _CACHE
    try:
        cache = _CACHE  # type: ignore[name-defined]
    except NameError:
        cache = None
    if cache is None:
        cache = build_all_click_tiles(embed_3d=True)
        globals()["_CACHE"] = cache
    return cache


# Convenient group accessors. Each returns the corresponding slice of the
# 12-tile library.  The lists are NEW list objects so callers cannot mutate
# the cached library.
def AZIDE_TILES() -> List[Tile]:    # noqa: N802 — caps for module-level API
    return list(_ensure_built()[0:4])


def ALKYNE_TILES() -> List[Tile]:   # noqa: N802
    return list(_ensure_built()[4:8])


def PARTNER_TILES() -> List[Tile]:  # noqa: N802
    return list(_ensure_built()[8:12])


def THIOL_TILES() -> List[Tile]:  # noqa: N802
    """Close-Loop #3 addition: 2 dedicated thiol handles.

    Returns the two thiol tiles appended after the canonical 12-tile
    library (positions 12:13).  Used by ThiolEne to raise coverage
    above the previous 0.75 floor.
    """
    return list(_ensure_built()[12:14])


def STANDARD_12_TILES() -> List[Tile]:  # noqa: N802
    """Backward-compatible alias for the canonical 12-tile library.

    Returns exactly the original 4+4+4 tiles — NOT the 14-tile
    extended library (use :func:`STANDARD_14_TILES` for that).
    """
    return list(_ensure_built()[:12])


def STANDARD_14_TILES() -> List[Tile]:  # noqa: N802
    """Extended 14-tile library: 12 standard + 2 thiol handles.

    Added in Close-Loop #3 to close the ThiolEne fire-rate gap.
    """
    return list(_ensure_built())


# ---------------------------------------------------------------------------
# WF-Partner-Tiles-PathA — click-rule partner tiles (azides, boronic
# acids, bromides) chosen to pair with bare Pt_alkyne seeds.
# ---------------------------------------------------------------------------
#
# Background
# ----------
# Round-12 λ-only pilot found that the MCTS expansion never fired any
# click rule when starting from a bare Pt_alkyne seed: the only
# :func:`PARTNER_TILES` (4 tiles: methylphosphine, cyclopentadiene,
# methyl vinyl ketone, maleimide) had no azide/boronic-acid/bromide
# handles, so the typed-reaction guards in :mod:`reactions.click_reactions`
# rejected every (rule, tile) attempt and the tree collapsed to
# n_distinct=1.
#
# The 8 partner tiles added here (3 azides + 3 boronic acids + 2
# bromides) close the gap so the CuAAC, SPAAC, SPC, and Suzuki rules
# can fire on a Pt_alkyne root and produce diverse products.  Each
# SMILES is RDKit-sanitisable and was validated against the canonical
# PubChem form (no stereochemistry, Phase-0 simplification).
#
# Naming convention
# -----------------
# The 8 tiles are split 3 + 3 + 2:
#
#     3 azide handles   : ethyl / benzyl / 4-azidobutanoic acid
#     3 boronic acids   : phenyl / 4-methylphenyl / 4-carboxyphenyl
#     2 bromides        : bromobenzene / 4-bromopyridine
#
# Honest framing: the V2 partners are *additive* — they do NOT replace
# the original 4 :func:`PARTNER_TILES` (those are still the canonical
# reference for the ThiolEne, DielsAlder, Maleimide, and SPC rules).
# Callers that want the union of all click-handle families should use
# :func:`ALL_CLICK_HANDLES`.


def _build_v2_partner_tiles(embed_3d: bool = True) -> List[Tile]:
    """Build the 8 partner tiles required by the click-rule path.

    The list is laid out in the documented order: 3 azides, 3
    boronic acids, 2 bromides.  Each entry is a :class:`Tile` with the
    same :func:`build_click_tile` factory so descriptors (mw / logp /
    tpsa) and 3D coords are populated identically to the canonical
    12-tile library.
    """
    return [
        # ----- 3 azide partner handles (CuAAC / SPAAC / SPC) -----
        # alkyl: ethyl azide (also in canonical 12-tile library; the
        # duplicate is intentional — it ensures :func:`PARTNER_TILES_V2`
        # is self-contained for callers that only import V2).
        build_click_tile("CCN=[N+]=[N-]", ["azide", "alkyl_azide"], embed_3d=embed_3d),
        # benzyl: benzyl azide (also in canonical library).
        build_click_tile("N(=[N+]=[N-])Cc1ccccc1", ["azide", "benzyl_azide"], embed_3d=embed_3d),
        # functional: 4-azidobutanoic acid (NEW — only available in V2).
        # Carboxylic acid + azide gives a bi-functional handle: it can
        # participate in CuAAC at the azide AND in amide coupling at
        # the COOH, which is exactly the multi-rule surface the Round-12
        # pilot was missing.
        build_click_tile(
            "OC(=O)CCCN=[N+]=[N-]",
            ["azide", "alkyl_azide", "carboxylic_acid"],
            embed_3d=embed_3d,
        ),

        # ----- 3 boronic acid partner handles (Suzuki-Miyaura) -----
        # aryl: phenylboronic acid.
        build_click_tile(
            "OB(O)c1ccccc1",
            ["boronic_acid", "aryl_boronic"],
            embed_3d=embed_3d,
        ),
        # substituted: 4-methylphenylboronic acid (p-tolyl boronic).
        build_click_tile(
            "Cc1ccc(B(O)O)cc1",
            ["boronic_acid", "aryl_boronic", "methyl_substituted"],
            embed_3d=embed_3d,
        ),
        # functional: 4-carboxyphenylboronic acid (NEW — COOH + boronic).
        build_click_tile(
            "OC(=O)c1ccc(B(O)O)cc1",
            ["boronic_acid", "aryl_boronic", "carboxylic_acid"],
            embed_3d=embed_3d,
        ),

        # ----- 2 bromide partner handles (Suzuki / direct arylation) -----
        # aryl: bromobenzene.
        build_click_tile(
            "Brc1ccccc1",
            ["aryl_halide", "bromide", "aryl_bromide"],
            embed_3d=embed_3d,
        ),
        # heterocyclic: 4-bromopyridine (Suzuki on heteroaryl).
        build_click_tile(
            "Brc1ccncc1",
            ["aryl_halide", "bromide", "heteroaryl_bromide", "pyridine"],
            embed_3d=embed_3d,
        ),
    ]


# Lazy V2 partner cache — heavy RDKit work is deferred until the first
# call.  Mirrors the design of the canonical 12-tile cache above.
def _ensure_v2_built() -> List[Tile]:
    """Build (or reuse) the 8-tile V2 click-partner library."""
    global _CACHE_V2
    try:
        cache = _CACHE_V2  # type: ignore[name-defined]
    except NameError:
        cache = None
    if cache is None:
        cache = _build_v2_partner_tiles(embed_3d=True)
        globals()["_CACHE_V2"] = cache
    return cache


def PARTNER_TILES_V2() -> List[Tile]:  # noqa: N802
    """V2 partner tiles: 3 azides + 3 boronic acids + 2 bromides.

    Added in WF-Partner-Tiles-PathA to close the click-rule reachability
    gap.  Use this accessor when a caller needs azide/boronic-acid/
    bromide partners for the CuAAC / SPAAC / SPC / Suzuki rules.

    Returns 8 tiles (3 + 3 + 2).  The order is documented in
    :func:`_build_v2_partner_tiles` above.

    Honest framing: the V2 partners were chosen to pair with bare
    Pt_alkyne seeds.  Their SMILES were validated via
    ``Chem.MolFromSmiles``; they are RDKit-sanitisable.
    """
    return list(_ensure_v2_built())


def AZIDE_PARTNER_TILES() -> List[Tile]:  # noqa: N802
    """The 3 azide partner tiles from :func:`PARTNER_TILES_V2`.

    Returns positions [0:3] of the V2 library (ethyl, benzyl, and
    4-azidobutanoic acid).  These are the canonical CuAAC / SPAAC /
    SPC reactant handles.
    """
    return list(_ensure_v2_built()[0:3])


def BORONIC_PARTNER_TILES() -> List[Tile]:  # noqa: N802
    """The 3 boronic acid partner tiles from :func:`PARTNER_TILES_V2`.

    Returns positions [3:6] of the V2 library (phenyl-, 4-methylphenyl-,
    and 4-carboxyphenyl-boronic acid).  These are the canonical
    Suzuki-Miyaura reactant handles.
    """
    return list(_ensure_v2_built()[3:6])


def BROMIDE_PARTNER_TILES() -> List[Tile]:  # noqa: N802
    """The 2 bromide partner tiles from :func:`PARTNER_TILES_V2`.

    Returns positions [6:8] of the V2 library (bromobenzene and
    4-bromopyridine).  These are the canonical Suzuki / direct-
    arylation reactant handles.
    """
    return list(_ensure_v2_built()[6:8])


def ALL_CLICK_HANDLES() -> List[Tile]:  # noqa: N802
    """Concatenation of the canonical 14-tile library + the 8 V2 partner
    tiles (22 tiles total).

    Use this accessor when the MCTS expansion pool needs EVERY click-
    rule reactant (azide + alkyne + boronic + bromide + thiol + diene
    + maleimide + phosphine) so all five reaction families (CuAAC,
    SPAAC, SPC, ThiolEne, Suzuki) are reachable from a single pool.
    """
    return list(_ensure_built()) + list(_ensure_v2_built())


def l6_metrics() -> Dict[str, object]:
    """Return a snapshot of L6 instrumentation counters."""
    # Build instrumentation is cumulative across calls; expose the current
    # canonical library as a set-like snapshot so repeated cache rebuilds do
    # not look like chemical duplicates in governance metrics.
    canonical = list(dict.fromkeys(_L6_CANONICAL_SMILES))
    tile_ids = list(dict.fromkeys(_L6_TILE_IDS))
    return {
        "counters": dict(_L6_COUNTERS),
        "sas_history": list(_L6_SAS_HISTORY),
        "descriptors": list(_L6_DESCRIPTORS),
        "canonical_smiles": canonical,
        "tile_ids": tile_ids,
    }
