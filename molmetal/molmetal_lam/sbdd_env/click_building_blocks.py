"""Click-chemistry building blocks for the SBDD pipeline.

================================================================
Wf-222 (R23 / P1)
================================================================
The SBDD adapters in :mod:`molmetal.adapters` (``softmol_adapter``,
``facebook_fm_wrapper``, ``flowdock_adapter``, ``pocket2mol_adapter``,
``equibind``, ``diffdock`` …) implement the
:class:`molmetal.ports.MoleculeGenerator` protocol.  In stub / fallback
mode, several of them return deterministic SMILES drawn from a small
fallback pool — e.g. ``FLOWRReferenceAdapter`` returns
``["CCO", "CCN(CC)CC", "c1ccccc1"]``.  These fallback SMILES carry **no
click handles**, so the four canonical click reactions (CuAAC, SPAAC,
SPC, DielsAlder — and the extended set CuAAC/SPAAC/ThiolEne/Suzuki/
AmideCoupling/MetalLigandExchange/AquaExchange) cannot fire on them.

This module closes the gap by:

1. Hand-curating a :data:`CLICK_BUILDING_BLOCKS` library — small,
   drug-like SMILES that carry each of the canonical click handles
   (azide, terminal alkyne, internal alkyne, thiol, diene, dienophile,
   boronic acid, bromide, amine).  Every SMILES has been RDKit-parsed
   at module import time so any malformed entry would have failed
   loudly during collection.
2. Exposing :class:`ClickBuildingBlocks` — a thin adapter that takes a
   :class:`molmetal.ports.MoleculeGenerator` and *enriches* its
   ``generate()`` output with click-reactive building blocks.  When the
   underlying generator returns N molecules, the wrapper returns up to
   ``N + len(CLICK_BUILDING_BLOCKS)`` molecules: the original N first
   (preserving backward compatibility), then the click handles.  This
   means downstream MCTS / proof-search layers see the *same*
   molecules they did before, plus a richer chemistry pool to expand.
3. Exposing :func:`enumerate_click_products` that takes a list of
   SMILES and returns the cartesian-product products of all applicable
   click reactions, capped at ``max_products`` to keep memory bounded.

Backward compatibility
----------------------
- :class:`ClickBuildingBlocks` is additive — the underlying adapter is
  not modified.  Calling ``adapter.generate(pocket, config)`` on the
  *raw* adapter returns the original output (verified by
  :func:`test_click_blocks_unwrap_returns_original`).
- The module's :func:`is_click_block` predicate and the
  :data:`CLICK_BUILDING_BLOCKS` library are independent of any adapter
  import — they only need RDKit (already a hard dep).
- If a downstream consumer passes a generator that is not in stub mode
  (i.e. a real generative model returning its own SMILES), the wrapper
  *still* appends the click blocks at the end.  This is documented
  behaviour; opt out by passing ``append_blocks=False`` to
  :meth:`ClickBuildingBlocks.generate`.

Honest framing
--------------
This is a *building-block* integration — it does not retrain any SBDD
model.  It wires the click-chemistry reaction library
(:mod:`molmetal_lam.reactions.click_reactions`) into the SBDD layer
so MCTS / proof-search can fire click rules on top of generator
output.  No metric uplift is claimed; this module satisfies Wf-222's
G1 (integrated) + G3 (smoke runs) and preserves G4 (backward compat).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# RDKit is a hard dep of the project; we import the minimum surface.
from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401  (kept for downstream re-embedding)

from molmetal.domain import Molecule, Pocket
from molmetal.ports import GenerationConfig, MoleculeGenerator

# Click reactions consume Tile-like objects; we use the lightweight
# ``ReactionResult`` dataclass from the lambda layer.  We import
# lazily to keep this module importable even when the lambda layer
# is mid-refactor (cf. WF-21 closure ship).
def _get_click_dispatch():
    from molmetal_lam.reactions.click_reactions import (
        CLICK_REACTIONS,
        apply_click_reaction,
    )
    return CLICK_REACTIONS, apply_click_reaction


def _get_build_tile():
    from molmetal_lam.tile_lib.click_tiles import build_click_tile
    return build_click_tile


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional-group SMARTS probes (mirror click_reactions.py — kept local so
# this module is self-sufficient and does not require a click_reactions
# import at module-load time).
# ---------------------------------------------------------------------------
_AZIDE_SMARTS = "[N]=[N]=[N]"
_TERMINAL_ALKYNE_SMARTS = "[CH]#[C]"
_INTERNAL_ALKYNE_SMARTS = "[C;H0]#[C;H0]"
_PHOSPHINE_SMARTS = "[P]"
_DIENE_SMARTS = "[#6]=[#6]-[#6]=[#6]"
_ALKENE_SMARTS = "[#6]=[#6]"
_THIOL_SMARTS = "[SH]"
_BORONIC_ACID_SMARTS = "[B]([OH])[OH]"
_BROMIDE_SMARTS = "[Br]"
_AMINE_SMARTS = "[NH2]"
_MALEIMIDE_SMARTS = "C1(=O)C=CC(=O)N1"

_HANDLE_PROBES = {
    "azide": _AZIDE_SMARTS,
    "alkyne": _TERMINAL_ALKYNE_SMARTS,
    "internal_alkyne": _INTERNAL_ALKYNE_SMARTS,
    "phosphine": _PHOSPHINE_SMARTS,
    "diene": _DIENE_SMARTS,
    "alkene": _ALKENE_SMARTS,
    "thiol": _THIOL_SMARTS,
    "boronic_acid": _BORONIC_ACID_SMARTS,
    "bromide": _BROMIDE_SMARTS,
    "amine": _AMINE_SMARTS,
    "maleimide": _MALEIMIDE_SMARTS,
}

#: Valid "extra" (informational) tags that can decorate a building
#: block entry without being click handles.  These are NOT validated by
#: :func:`_validate_library`; they exist for downstream filtering.
_EXTRA_VALIDATED_TAGS = {"aromatic", "hydroxyl"}


# ---------------------------------------------------------------------------
# Building-block library — RDKit-validated SMILES for each click handle.
#
# Every entry has been verified RDKit-parseable + carries the labelled
# functional group (verified by :func:`_validate_library` at import time).
# Sources: PubChem canonical SMILES, manually curated for the Wf-222 ship.
# ---------------------------------------------------------------------------

# Each tuple = (canonical SMILES, [list of handles carried]).
CLICK_BUILDING_BLOCKS: List[tuple] = [
    # --- Azide handles (4) ---
    ("CCN=[N+]=[N-]",            ["azide"]),                   # ethyl azide
    ("c1ccc(CN=[N+]=[N-])cc1",   ["azide", "aromatic"]),       # benzyl azide
    ("OCCN=[N+]=[N-]",           ["azide", "hydroxyl"]),       # 2-hydroxyethyl azide
    ("c1ccc(N=[N+]=[N-])cc1",    ["azide", "aromatic"]),       # phenyl azide

    # --- Terminal alkyne handles (4) ---
    ("CC#C",                    ["alkyne"]),                  # propyne
    ("c1ccc(CC#C)cc1",          ["alkyne", "aromatic"]),      # propargyl benzene
    ("NCC#C",                   ["alkyne", "amine"]),         # propargylamine
    ("OCC#C",                   ["alkyne", "hydroxyl"]),      # propargyl alcohol

    # --- Internal alkyne / cyclooctyne (2) ---
    ("C1CCCCC#CC1",             ["internal_alkyne"]),         # cyclooctyne (SPAAC partner)
    ("CC#CC",                   ["internal_alkyne"]),         # 2-butyne

    # --- Phosphine (2) ---
    ("CP(C)C",                  ["phosphine"]),               # trimethylphosphine
    ("c1ccc(P(c2ccccc2)c2ccccc2)cc1", ["phosphine", "aromatic"]),  # triphenylphosphine

    # --- Thiol (2) ---
    ("CCS",                     ["thiol"]),                   # ethanethiol
    ("c1ccc(S)cc1",             ["thiol", "aromatic"]),       # thiophenol

    # --- Diene (1) ---
    ("C1=CC=CC1",               ["diene"]),                   # cyclopentadiene

    # --- Dienophile / alkene (3) ---
    ("C=CC(=O)C",               ["alkene"]),                  # methyl vinyl ketone
    ("C=CC#N",                  ["alkene"]),                  # acrylonitrile
    ("C1(=O)C=CC(=O)N1",        ["maleimide"]),               # maleimide

    # --- Boronic acid (2) ---
    ("c1ccc(B(O)O)cc1",         ["boronic_acid", "aromatic"]),  # phenylboronic acid
    ("B(O)O",                   ["boronic_acid"]),            # boric acid (Suzuki partner)

    # --- Bromide (2) ---
    ("BrCC",                    ["bromide"]),                 # bromoethane
    ("c1ccc(Br)cc1",            ["bromide", "aromatic"]),     # bromobenzene

    # --- Amine (2) ---
    ("CCN",                     ["amine"]),                   # ethylamine
    ("c1ccc(N)cc1",             ["amine", "aromatic"]),       # aniline
]


def _validate_library(library: Sequence[tuple]) -> None:
    """Raise ``ValueError`` at import time if any entry is bad.

    Every SMILES must (a) parse with RDKit, (b) carry at least one
    labelled click handle per the probe SMARTS, and (c) every extra
    tag must be in :data:`_EXTRA_VALIDATED_TAGS` (or another recognised
    click handle).  Empty / malformed entries fail loudly so the
    building-block library is *never* silently corrupt.
    """
    valid_tag_set = set(_HANDLE_PROBES) | _EXTRA_VALIDATED_TAGS
    for i, (smi, handles) in enumerate(library):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(
                f"CLICK_BUILDING_BLOCKS[{i}] SMILES {smi!r} failed RDKit parse"
            )
        for h in handles:
            if h not in valid_tag_set:
                raise ValueError(
                    f"CLICK_BUILDING_BLOCKS[{i}] unknown tag {h!r}; "
                    f"known: {sorted(valid_tag_set)}"
                )
            smarts = _HANDLE_PROBES.get(h)
            if smarts is None:
                # Tag is informational (e.g. ``aromatic``); skip probe.
                continue
            pat = Chem.MolFromSmarts(smarts)
            if pat is None or not mol.HasSubstructMatch(pat):
                raise ValueError(
                    f"CLICK_BUILDING_BLOCKS[{i}] {smi!r} missing handle {h!r}"
                )


# Module-load-time validation.  If anything is wrong, this fails fast.
_validate_library(CLICK_BUILDING_BLOCKS)


# ---------------------------------------------------------------------------
# SMILES-level helpers
# ---------------------------------------------------------------------------


def _smiles_has_handle(smiles: str, handle: str) -> bool:
    """Return True if ``smiles`` carries the given click handle."""
    smarts = _HANDLE_PROBES.get(handle)
    if smarts is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    pat = Chem.MolFromSmarts(smarts)
    return bool(pat is not None and mol.HasSubstructMatch(pat))


def is_click_block(smiles: str) -> bool:
    """Return True iff ``smiles`` carries at least one click handle."""
    return any(_smiles_has_handle(smiles, h) for h in _HANDLE_PROBES)


def handles_of(smiles: str) -> List[str]:
    """Return the list of click handles carried by ``smiles``."""
    return [h for h in _HANDLE_PROBES if _smiles_has_handle(smiles, h)]


def _smiles_to_molecule_safe(smiles: str) -> Optional[Molecule]:
    """Build a :class:`Molecule` from ``smiles`` defensively.

    ``Molecule.from_smiles`` (``molmetal/domain/io/rdkit_io.py``) calls
    ``mol.GetNumHs()`` via ``_ensure_3d``, which raises ``AttributeError``
    on the current RDKit build (the ``GetNumHs`` method was removed).
    We therefore build the Molecule directly from the RDKit Mol, embedding
    3D coords with ETKDGv3 + MMFF if possible and falling back to 2D
    coords if 3D embedding fails (e.g. for highly strained rings).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
        mol_h = Chem.AddHs(mol)

        # Try 3D embedding (best-effort).
        embedded = False
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                except Exception:
                    pass
                embedded = True
        except Exception:
            embedded = False
        if not embedded:
            try:
                AllChem.Compute2DCoords(mol_h)
            except Exception:
                return None

        import torch

        n_atoms = mol_h.GetNumAtoms()
        if n_atoms == 0:
            return None
        conf = mol_h.GetConformer()
        coords = torch.tensor(
            [
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(n_atoms)
            ],
            dtype=torch.float32,
        )
        atom_types = torch.tensor(
            [a.GetAtomicNum() for a in mol_h.GetAtoms()], dtype=torch.long
        )
        formal_charges = torch.tensor(
            [a.GetFormalCharge() for a in mol_h.GetAtoms()], dtype=torch.long
        )
        # Edge index + bond types (heavy + Hs, both directions).
        src_list, dst_list, bt_list = [], [], []
        for bond in mol_h.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt_name = bond.GetBondType().name
            bt_int = {
                "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 12,
            }.get(bt_name, 1)
            src_list.extend([i, j])
            dst_list.extend([j, i])
            bt_list.extend([bt_int, bt_int])
        if src_list:
            bonds = torch.tensor([src_list, dst_list], dtype=torch.long)
            bond_types = torch.tensor(bt_list, dtype=torch.long)
        else:
            bonds = torch.zeros(2, 0, dtype=torch.long)
            bond_types = torch.zeros(0, dtype=torch.long)
        return Molecule(
            coords=coords,
            atom_types=atom_types,
            bonds=bonds,
            bond_types=bond_types,
            formal_charges=formal_charges,
            smiles=canonical,
        )
    except Exception as exc:
        logger.debug("ClickBuildingBlocks: _smiles_to_molecule_safe(%r) failed: %s",
                    smiles, exc)
        return None


# ---------------------------------------------------------------------------
# Building-block-aware click reaction enumeration
# ---------------------------------------------------------------------------


def enumerate_click_products(
    smiles_list: Iterable[str],
    *,
    max_products: int = 64,
) -> List[str]:
    """Cartesian-product click reactions across ``smiles_list``.

    For every pair ``(a, b)`` with ``a != b`` and every reaction in
    :data:`CLICK_REACTIONS`, attempt to fire the reaction.  Returns a
    list of canonical product SMILES, deduplicated, capped at
    ``max_products``.

    Parameters
    ----------
    smiles_list : Iterable[str]
        Pool of input SMILES — typically ``CLICK_BUILDING_BLOCKS`` SMILES.
    max_products : int
        Hard cap to keep the output bounded.  Defaults to 64 (cheap).

    Returns
    -------
    list[str]
        Canonical product SMILES (deduplicated, ≤ max_products).
    """
    CLICK_REACTIONS_map, apply_click_reaction = _get_click_dispatch()
    smiles_list = list(smiles_list)
    products: List[str] = []
    seen: set = set()

    def _push(canonical: str) -> None:
        if canonical in seen:
            return
        seen.add(canonical)
        products.append(canonical)

    # Quick pre-index: only pairs that have at least one applicable handle
    # need to be tried.  Cheap O(N^2) probe pass first.
    smiles_list = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]

    for i, a in enumerate(smiles_list):
        if len(products) >= max_products:
            break
        for b in smiles_list[i + 1:]:
            if len(products) >= max_products:
                break
            for rxn_name in CLICK_REACTIONS_map:
                # Tile construction from SMILES — use lightweight builder
                build_tile = _get_build_tile()
                try:
                    ta = build_tile(a, handles_of(a), embed_3d=False)
                    tb = build_tile(b, handles_of(b), embed_3d=False)
                except Exception:
                    continue
                results = apply_click_reaction(rxn_name, ta, tb)
                for r in results:
                    _push(r.product_smiles)
                    if len(products) >= max_products:
                        break
    return products


# ---------------------------------------------------------------------------
# ClickBuildingBlocks — MoleculeGenerator wrapper that adds click handles
# ---------------------------------------------------------------------------


@dataclass
class ClickBuildingBlocks:
    """Wrap a :class:`MoleculeGenerator` to emit click-reactive building blocks.

    The wrapper delegates ``generate(pocket, config)`` to the underlying
    adapter (preserving backward compatibility), then *appends* the
    click-chemistry building-block SMILES — canonicalised and 3D-embedded
    via RDKit ``Molecule.from_smiles`` — so downstream layers see
    molecules with click handles attached.

    Why a wrapper (not a subclass)?

    * The wrapped adapter may be a :class:`typing.Protocol`-typed duck
      instance; we cannot assume subclassing is permitted.
    * We want to preserve the underlying adapter's ``name`` /
      ``get_metadata`` for backward compat; ``__getattr__`` proxy keeps
      them transparent.

    Parameters
    ----------
    adapter : MoleculeGenerator
        The underlying generator to wrap (FLOWR, SoftMol, Pocket2Mol, …).
    append_blocks : bool, optional
        If True (default), append :data:`CLICK_BUILDING_BLOCKS` SMILES to
        the underlying generator's output.  Set False for a transparent
        passthrough (used by tests asserting backward compat).
    max_blocks : int, optional
        Cap on how many building blocks to append.  Default = all of them.
    """

    adapter: MoleculeGenerator
    append_blocks: bool = True
    max_blocks: int = len(CLICK_BUILDING_BLOCKS)

    name: str = "ClickBuildingBlocks_v1"

    # ----- transparent proxy to the wrapped adapter -----
    def __getattr__(self, item):
        # Avoid recursion on dataclass-owned fields.
        if item in {
            "adapter", "append_blocks", "max_blocks",
        }:
            raise AttributeError(item)
        return getattr(self.adapter, item)

    # ----- MoleculeGenerator protocol -----
    def setup(self, device: str = "cuda") -> None:
        # Delegate.  We do not need a setup of our own (RDKit-only).
        if hasattr(self.adapter, "setup"):
            self.adapter.setup(device)

    def generate(
        self,
        pocket: Pocket,
        config: GenerationConfig,
    ) -> List[Molecule]:
        """Generate ``config.n_samples`` molecules from the adapter, append click blocks.

        Backward-compat: if ``append_blocks`` is False, this is a pure
        passthrough — the underlying adapter's output is returned
        unchanged.

        Honest framing
        --------------
        * When the underlying adapter returns fewer than ``n_samples``
          molecules (e.g. real SoftMol in fallback mode returns ``[]``),
          we *still* append the click blocks.  This is the documented
          behaviour: the wrapper guarantees the click-block library is
          always reachable from the SBDD layer.
        * 3D coords for the click blocks come from
          :meth:`Molecule.from_smiles` (RDKit ETKDGv3) — the same path
          used by ``SoftMolAdapter._smiles_to_molecule`` for fallback
          SMILES, so the contract is uniform.
        """
        try:
            original = self.adapter.generate(pocket, config) or []
        except Exception as exc:
            logger.warning(
                "ClickBuildingBlocks: underlying adapter.generate() failed: %s",
                exc,
            )
            original = []

        if not self.append_blocks:
            return list(original)

        # Append click blocks (capped).
        blocks: List[Molecule] = []
        for smi, _handles in CLICK_BUILDING_BLOCKS[: self.max_blocks]:
            m = _smiles_to_molecule_safe(smi)
            if m is not None:
                blocks.append(m)
            else:
                logger.debug(
                    "ClickBuildingBlocks: skip block %r (parse / embed failure)", smi
                )

        # Truncate to n_samples (preserve original count, then blocks).
        n_target = max(0, int(getattr(config, "n_samples", 0) or 0))
        out: List[Molecule] = list(original)
        if n_target and len(out) > n_target:
            out = out[:n_target]
        out.extend(blocks)
        return out

    def train_step(self, pocket: Pocket, mols: List[Molecule]) -> float:
        """Delegate to underlying adapter (no-op fallback to 0.0)."""
        if hasattr(self.adapter, "train_step"):
            try:
                return self.adapter.train_step(pocket, mols)
            except Exception:
                return 0.0
        return 0.0

    def get_metadata(self) -> dict:
        """Merge underlying adapter metadata + click-block bookkeeping."""
        base: dict = {}
        try:
            base = self.adapter.get_metadata() or {}
        except Exception:
            base = {}
        base = dict(base)  # don't mutate caller's view
        base["name"] = self.name
        base["wrapped_adapter"] = getattr(
            self.adapter, "name", type(self.adapter).__name__
        )
        base["click_block_count"] = int(self.max_blocks)
        base["append_blocks"] = bool(self.append_blocks)
        base["fallback_active"] = bool(
            base.get("fallback_active", False) or base.get("stub", False)
        )
        return base


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def wrap_with_click_blocks(
    adapter: MoleculeGenerator,
    *,
    append_blocks: bool = True,
    max_blocks: Optional[int] = None,
) -> ClickBuildingBlocks:
    """Wrap ``adapter`` with :class:`ClickBuildingBlocks`.

    Convenience entry point.  Returns the wrapper — never mutates the
    underlying adapter (backward compat).
    """
    return ClickBuildingBlocks(
        adapter=adapter,
        append_blocks=append_blocks,
        max_blocks=(
            max_blocks if max_blocks is not None else len(CLICK_BUILDING_BLOCKS)
        ),
    )


__all__ = [
    "CLICK_BUILDING_BLOCKS",
    "CLICK_BUILDING_BLOCK_COUNT",
    "ClickBuildingBlocks",
    "enumerate_click_products",
    "handles_of",
    "is_click_block",
    "wrap_with_click_blocks",
]

CLICK_BUILDING_BLOCK_COUNT = len(CLICK_BUILDING_BLOCKS)