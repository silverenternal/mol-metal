"""Pocket2Mol adapter — wraps ``molmetal/references/Pocket2Mol`` as a
:class:`MoleculeGenerator`-compatible object that returns a list of
:class:`molmetal.domain.Complex` objects for a given :class:`Pocket`.

================================================================
Checkpoint status (documented)
================================================================
The official Pocket2Mol pretrained weights
(``pretrained_Pocket2Mol.pt``, ~165 MB) live behind a Google Drive
folder that we **do not** download automatically — task policy
forbids pulling external pretrained weights into this environment.
The repo at ``molmetal/references/Pocket2Mol/ckpt/`` therefore only
contains ``README.md`` with the download URL.

This adapter has two paths, picked at runtime:

* :meth:`_Pocket2MolLiveAdapter.sample` — calls the real Pocket2Mol
  ``sample_for_pdb.py`` machinery (``MaskFillModelVN`` + ``AtomComposer``
  + ``get_init`` / ``get_next`` / ``reconstruct_from_generated_with_edges``)
  if the checkpoint is on disk AND PyTorch Geometric is importable.
* :meth:`_Pocket2MolFallbackAdapter.sample` — SMARTS-only baseline that
  emits 100 drug-like SMILES sampled from a built-in pool, mimicking
  the Pocket2Mol output contract (``List[Complex]`` with
  ``molecule.smiles`` populated).

The public :class:`Pocket2MolAdapter` wraps whichever backend is
available and exposes a single ``sample(pocket, n_samples)`` method
implementing the MoleculeGenerator port.

Reference
---------
Peng, X.; Ran, T.; Guan, J.; et al. *Pocket2Mol: Efficient Molecular
Sampling Based on 3D Protein Pockets.*  ICML 2022.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import DockingConfig, GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Locate the cloned Pocket2Mol repo + the (missing) checkpoint
# ---------------------------------------------------------------------------
POCKET2MOL_REPO = (
    Path(__file__).resolve().parent.parent.parent
    / "references"
    / "Pocket2Mol"
)
POCKET2MOL_CKPT = POCKET2MOL_REPO / "ckpt" / "pretrained_Pocket2Mol.pt"
POCKET2MOL_CKPT_ALT = POCKET2MOL_REPO / "ckpt" / "pretrained.pt"
POCKET2MOL_CONFIG = (
    POCKET2MOL_REPO / "configs" / "sample_for_pdb.yml"
)


def _checkpoint_available() -> bool:
    """True iff a pretrained Pocket2Mol ``.pt`` file is on disk."""
    return POCKET2MOL_CKPT.is_file() or POCKET2MOL_CKPT_ALT.is_file()


def _checkpoint_path() -> Optional[Path]:
    if POCKET2MOL_CKPT.is_file():
        return POCKET2MOL_CKPT
    if POCKET2MOL_CKPT_ALT.is_file():
        return POCKET2MOL_CKPT_ALT
    return None


def _pyg_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Built-in SMILES pool — fallback when the Pocket2Mol checkpoint is missing.
#
# This pool mimics the "drug-like, ≤30 heavy atoms" universe that
# Pocket2Mol generates from CrossDocked2020.  It is intentionally
# small (~120 SMILES) and hand-curated so that the same set can be
# reproducibly re-emitted across runs.
# ---------------------------------------------------------------------------
_FALLBACK_SMILES_POOL: Tuple[str, ...] = (
    # carboxylic acids / NSAIDs
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",    # ibuprofen
    "OC(=O)c1ccccc1O",                # salicylic acid
    "OC(=O)c1ccc(N)cc1",              # 4-aminobenzoic acid
    "OC(=O)c1ccc(O)cc1",              # 4-hydroxybenzoic acid
    # hydroxamic acids (MMP-ish)
    "ONC(=O)CCC(=O)O",
    "ONC(=O)CCCC(=O)O",
    "ONC(=O)c1ccccc1",
    "ONC(=O)c1ccc(N)cc1",
    "ONC(=O)c1ccc(O)cc1",
    "ONC(=O)c1ccc(F)cc1",
    "ONC(=O)c1ccc(Cl)cc1",
    "ONC(=O)c1ccc(C)cc1",
    "ONC(=O)c1ccccn1",
    "ONC(=O)c1ccncc1",
    # sulfonamides
    "NS(=O)(=O)c1ccccc1",
    "NS(=O)(=O)c1ccc(N)cc1",
    "Cc1ccc(S(=O)(=O)N)cc1",
    "NS(=O)(=O)c1ccc(F)cc1",
    "NS(=O)(=O)c1ccc(Cl)cc1",
    # triazoles / click-chem
    "Cc1ccnn1C",
    "Cc1ccc(-c2cn[nH]n2)cc1",
    "c1ccc(-c2cn[nH]n2)cc1",
    "OCc1cn[nH]n1",
    "NCc1cn[nH]n1",
    "Cc1ncc(N)nc1",
    # pyrimidines / heterocycles
    "c1cncnc1",
    "Cc1ccncn1",
    "Nc1ncncn1",
    "Oc1ncncn1",
    "Clc1ncncn1",
    # amides
    "CC(=O)Nc1ccccc1",                # acetanilide
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "CC(=O)Nc1ccc(N)cc1",
    "CCNC(=O)c1ccccc1",
    "CCN(C)C(=O)c1ccccc1",
    # amines / piperidines
    "NCCO",
    "NCCN",
    "N1CCOCC1",
    "N1CCNCC1",
    "c1ccc2[nH]ccc2c1",
    "C1CC2CCC1C2",
    # aromatics
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "Fc1ccccc1",
    "Clc1ccccc1",
    "Brc1ccccc1",
    "Ic1ccccc1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]nnc2c1",
    "c1ccc2nccnc2c1",
    # esters
    "CCOC(=O)C",
    "CCOC(=O)c1ccccc1",
    "CCCCOC(=O)C",
    "CCCCOC(=O)c1ccccc1",
    # alcohols
    "CCO",
    "CCCO",
    "CC(O)C",
    "CC(O)CO",
    "OCc1ccccc1",
    "OCCc1ccccc1",
    # nitriles
    "N#Cc1ccccc1",
    "N#CCc1ccccc1",
    "N#CCCO",
    # aldehydes / ketones
    "O=Cc1ccccc1",
    "CC(=O)c1ccccc1",
    "CCC(=O)c1ccccc1",
    # ethers
    "COc1ccccc1",
    "CCOc1ccccc1",
    "c1ccc(Oc2ccccc2)cc1",
    # biphenyls / extended aromatics
    "c1ccc(-c2ccccc2)cc1",
    "c1ccc(-c2ccncc2)cc1",
    "c1ccc(-c2ccnnc2)cc1",
    # heteroaromatics
    "c1ccc2nccnc2c1",
    "c1ccc2nc[nH]c2c1",
    "c1ccc2[nH]cnc2c1",
    "c1ccc2ncnnc2c1",
    "c1ccsc1",
    "c1ccoc1",
    "c1ccnc1",
    "c1ncc[nH]1",
    "c1nc[nH]n1",
    # trifluoromethyl
    "FC(F)(F)c1ccccc1",
    "FC(F)(F)c1ccc(C(F)(F)F)cc1",
    "FC(F)(F)c1ccc(N)cc1",
    "FC(F)(F)c1ccc(O)cc1",
    "OC(=O)c1ccc(C(F)(F)F)cc1",
    # nitro
    "O=[N+]([O-])c1ccccc1",
    "O=[N+]([O-])c1ccc(N)cc1",
    "O=[N+]([O-])c1ccc(O)cc1",
    # azides
    "N=[N+]=[N-]CC",
    "N=[N+]=[N-]Cc1ccccc1",
    "N=[N+]=[N-]CCC",
    # alkynes (click partners)
    "C#CC",
    "C#Cc1ccccc1",
    "C#Cc1ccncc1",
    "C#Cc1ccnnc1",
    "C#Cc1ccc(O)cc1",
    "C#Cc1ccc(N)cc1",
    # bigger drug-like
    "CN(C)c1ccccc1",
    "CN1CCCC1",
    "O=C(NCCO)c1ccccc1",
    "O=C(NCc1ccccc1)c1ccccc1",
    "c1ccc2sc(N)nc2c1",
    "c1ccc2[nH]c(=O)[nH]c2c1",
    "c1ccc2c(c1)oc1ccccc12",
    "c1ccc2c(c1)nc1ccccc12",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "c1ccc2c(c1)oc(=O)cc2",
    "c1ccc2c(c1)OCCO2",
    "Cc1cc(=O)[nH]c2ccccc12",
    # more extended
    "c1cc(-c2ccccc2)cnc1",
    "c1cc(-c2ccnnc2)cnc1",
    "c1cc(-c2cn[nH]n2)cnc1",
    "c1cc(-c2ccoc2)cnc1",
    "c1cc(-c2ccsc2)cnc1",
    "c1cc(-c2ccnc2)cnc1",
    # nitrile-containing
    "N#Cc1ccncc1",
    "N#Cc1ccnnc1",
    "N#Cc1ccc(N)cc1",
    "N#Cc1ccc(O)cc1",
    # indole-like
    "c1ccc2[nH]cc(CCN)c2c1",
    "c1ccc2[nH]cc(CCO)c2c1",
    "c1ccc2[nH]cc(CCN(C)C)c2c1",
)


# ---------------------------------------------------------------------------
# Backend 1 — live Pocket2Mol (only usable when the checkpoint is present)
# ---------------------------------------------------------------------------
class _Pocket2MolLiveAdapter:
    """Run the real Pocket2Mol sampling loop.

    On instantiation we:

        * patch ``sys.path`` so we can import
          ``molmetal/references/Pocket2Mol/{models,utils,...}``;
        * lazy-load ``MaskFillModelVN`` (Pocket2Mol's equivariant
          transformer) and load ``ckpt['model']`` weights;
        * cache a single ``AtomComposer`` transform.

    The :meth:`sample` method then runs Pocket2Mol's own loop
    (``get_init`` → ``get_next`` → ``reconstruct_from_generated_with_edges``)
    for ``n_samples`` finished molecules, exactly as
    ``sample_for_pdb.py`` does it for the configured pocket.
    """

    def __init__(self, device: str = "cpu") -> None:
        if not _checkpoint_available():
            raise RuntimeError(
                "Pocket2Mol checkpoint not found at "
                f"{POCKET2MOL_CKPT}. Download pretrained_Pocket2Mol.pt from "
                "the URL in molmetal/references/Pocket2Mol/ckpt/README.md "
                "and place it there."
            )
        if not _pyg_available():
            raise RuntimeError(
                "torch_geometric is required for the live Pocket2Mol adapter."
            )

        # --- Lazy imports so the fallback path doesn't pull heavy deps.
        sys.path.insert(0, str(POCKET2MOL_REPO))
        try:
            import yaml  # noqa: F401
            from models.maskfill import MaskFillModelVN  # noqa: F401
            from utils.reconstruct import (  # noqa: F401
                reconstruct_from_generated_with_edges,
            )
            from utils.transforms import (  # noqa: F401
                AtomComposer,
                Compose,
                FeaturizeLigandAtom,
                FeaturizeProteinAtom,
                LigandCountNeighbors,
                LigandMaskAll,
                RefineData,
            )
            from sample import (  # noqa: F401
                STATUS_FINISHED,
                STATUS_RUNNING,
                get_init,
                get_next,
                print_pool_status,
                transform_data,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import Pocket2Mol internals — "
                "check that PyTorch Geometric 2.x is installed and "
                f"the repo is at {POCKET2MOL_REPO}. ({exc})"
            )

        # --- Load YAML config
        with open(POCKET2MOL_CONFIG) as fh:
            cfg = yaml.safe_load(fh)
        self._cfg = cfg

        # --- Load checkpoint
        ckpt_path = _checkpoint_path()
        ckpt = torch.load(str(ckpt_path), map_location=device)
        self._ckpt = ckpt
        self._device = device

        # --- Build model
        protein_featurizer = FeaturizeProteinAtom()
        ligand_featurizer = FeaturizeLigandAtom()
        contrastive_sampler = ContrastiveSample(num_real=0, num_fake=0)
        self._model = MaskFillModelVN(
            ckpt["config"].model,
            num_classes=contrastive_sampler.num_elements,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
            num_bond_types=3,
        ).to(device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()
        self._protein_featurizer = protein_featurizer
        self._ligand_featurizer = ligand_featurizer
        self._atom_composer = AtomComposer(
            protein_featurizer.feature_dim,
            ligand_featurizer.feature_dim,
            self._model.config.encoder.knn,
        )

    def sample(
        self,
        pocket: Pocket,
        n_samples: int,
        bbox_size: float = 23.0,
    ) -> List[Complex]:
        """Run Pocket2Mol's sampling loop on ``pocket`` for ``n_samples``.

        This is essentially ``sample_for_pdb.py`` reduced to a function
        call: we build a pocket data object, transform it through
        Pocket2Mol's featurisers, then iterate ``get_next`` until
        ``n_samples`` finished molecules have been collected.
        """
        from rdkit import Chem  # type: ignore

        from sample import (  # type: ignore
            STATUS_FINISHED,
            STATUS_RUNNING,
            get_init,
            get_next,
            transform_data,
        )
        from utils.protein_ligand import PDBProtein  # type: ignore
        from utils.reconstruct import (  # type: ignore
            MolReconsError,
            reconstruct_from_generated_with_edges,
        )

        # Pocket must carry an original PDB path so we can re-extract
        # residues with proper atom names.  VinaDockingAdapter sets
        # ``pocket._pdb_path``; we honour that here too.
        pdb_path = (
            getattr(pocket, "_pdb_path", None)
            or getattr(pocket, "pdb_path", None)
        )
        if pdb_path is None or not Path(str(pdb_path)).is_file():
            raise ValueError(
                "_Pocket2MolLiveAdapter needs an original PDB on disk; "
                "set ``pocket._pdb_path`` before calling sample()."
            )

        center = pocket.center.cpu()
        # 1) Build pocket dict from PDB (mimics sample_for_pdb.pdb_to_pocket_data)
        warnings.simplefilter("ignore")
        ptable = Chem.GetPeriodicTable()
        from Bio.PDB import PDBParser  # type: ignore
        from Bio.PDB.Selection import unfold_entities  # type: ignore

        model_struct = PDBParser().get_structure(None, str(pdb_path))[0]
        protein_dict = {
            "element": [], "pos": [], "is_backbone": [], "atom_to_aa_type": [],
        }
        for atom in unfold_entities(model_struct, "A"):
            res = atom.get_parent()
            resname = res.get_resname()
            if resname == "MSE":
                resname = "MET"
            if resname not in PDBProtein.AA_NAME_NUMBER:
                continue
            element_symb = atom.element.capitalize()
            if element_symb == "H":
                continue
            x, y, z = atom.get_coord()
            pos = torch.FloatTensor([x, y, z])
            if (pos - center).abs().max() > (bbox_size / 2.0):
                continue
            protein_dict["element"].append(ptable.GetAtomicNumber(element_symb))
            protein_dict["pos"].append(pos)
            protein_dict["is_backbone"].append(
                atom.get_name() in ("N", "CA", "C", "O")
            )
            protein_dict["atom_to_aa_type"].append(
                PDBProtein.AA_NAME_NUMBER[resname]
            )
        protein_dict["element"] = torch.LongTensor(protein_dict["element"])
        protein_dict["pos"] = torch.stack(protein_dict["pos"], dim=0)
        protein_dict["is_backbone"] = torch.BoolTensor(
            protein_dict["is_backbone"]
        )
        protein_dict["atom_to_aa_type"] = torch.LongTensor(
            protein_dict["atom_to_aa_type"]
        )

        from utils.data import ProteinLigandData  # type: ignore
        from utils.transforms import (  # type: ignore
            Compose,
            ContrastiveSample,
            FeaturizeLigandAtom,
            FeaturizeProteinAtom,
            LigandCountNeighbors,
            LigandMaskAll,
            RefineData,
        )

        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=protein_dict,
            ligand_dict={
                "element": torch.empty([0], dtype=torch.long),
                "pos": torch.empty([0, 3], dtype=torch.float),
                "atom_feature": torch.empty([0, 8], dtype=torch.float),
                "bond_index": torch.empty([2, 0], dtype=torch.long),
                "bond_type": torch.empty([0], dtype=torch.long),
            },
        )
        data = Compose([
            RefineData(),
            LigandCountNeighbors(),
            FeaturizeProteinAtom(),
            FeaturizeLigandAtom(),
            LigandMaskAll(),
        ])(data)

        # 2) Sample loop (mirrors sample_for_pdb.py)
        threshold = self._cfg["sample"]["threshold"]
        beam_size = int(self._cfg["sample"]["beam_size"])
        max_steps = int(self._cfg["sample"]["max_steps"])
        smiles_set: set = set()
        complexes: List[Complex] = []
        pool = {"queue": [], "finished": [], "failed": [], "smiles": set()}

        data = transform_data(data, self._atom_composer)
        init_data_list = get_init(
            data.to(self._device),
            model=self._model,
            transform=self._atom_composer,
            threshold=threshold,
        )
        pool["queue"] = init_data_list[:beam_size]
        if len(pool["queue"]) == 0:
            return []

        global_step = 0
        while len(pool["finished"]) < n_samples and global_step < max_steps:
            global_step += 1
            queue_tmp = []
            for parent in pool["queue"]:
                data_next_list = get_next(
                    parent.to(self._device),
                    model=self._model,
                    transform=self._atom_composer,
                    threshold=threshold,
                )
                for dn in data_next_list:
                    if dn.status == STATUS_FINISHED:
                        try:
                            rdmol = reconstruct_from_generated_with_edges(dn)
                            smi = Chem.MolToSmiles(
                                Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
                            )
                            if smi not in smiles_set and "." not in smi:
                                pool["finished"].append(dn)
                                smiles_set.add(smi)
                                # Wrap as Complex
                                atom_types = np.asarray(
                                    [a.GetAtomicNum() for a in rdmol.GetAtoms()],
                                    dtype=np.int64,
                                )
                                coords = np.asarray(
                                    [
                                        list(
                                            rdmol.GetConformer().GetAtomPosition(
                                                a.GetIdx()
                                            )
                                        )
                                        for a in rdmol.GetAtoms()
                                    ],
                                    dtype=np.float32,
                                )
                                n_b = rdmol.GetNumBonds()
                                bonds = np.zeros((2, 0), dtype=np.int64)
                                bond_types = np.zeros(0, dtype=np.int64)
                                mol_obj = Molecule(
                                    coords=torch.from_numpy(coords),
                                    atom_types=torch.from_numpy(atom_types),
                                    bonds=torch.zeros(2, n_b, dtype=torch.long),
                                    bond_types=torch.zeros(n_b, dtype=torch.long),
                                    formal_charges=torch.zeros(
                                        len(atom_types), dtype=torch.long
                                    ),
                                    smiles=smi,
                                )
                                complexes.append(
                                    Complex(pocket=pocket, molecule=mol_obj)
                                )
                        except MolReconsError:
                            pool["failed"].append(dn)
                    elif dn.status == STATUS_RUNNING:
                        queue_tmp.append(dn)
            pool["queue"] = queue_tmp[:beam_size]
            if not pool["queue"]:
                break
        return complexes[:n_samples]


# ---------------------------------------------------------------------------
# Backend 2 — SMARTS-only fallback (used when the checkpoint is missing)
# ---------------------------------------------------------------------------
class _Pocket2MolFallbackAdapter:
    """SMARTS-only baseline that emits ``n_samples`` drug-like SMILES.

    This is what we report on the 1h36 head-to-head **today** because
    the pretrained Pocket2Mol ``.pt`` is not on disk.  It samples
    without replacement from :data:`_FALLBACK_SMILES_POOL`, wraps each
    SMILES in a :class:`Molecule` with empty 3D coords and returns the
    matching :class:`Complex` list — exactly the same shape that the
    live adapter returns, so the downstream VinaDockingAdapter pipeline
    is unchanged.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._pool = list(_FALLBACK_SMILES_POOL)

    def sample(
        self,
        pocket: Pocket,
        n_samples: int,
    ) -> List[Complex]:
        from rdkit import Chem  # type: ignore

        n = min(int(n_samples), len(self._pool))
        idxs = self._rng.choice(len(self._pool), size=n, replace=False)
        chosen = [self._pool[i] for i in idxs]
        complexes: List[Complex] = []
        for smi in chosen:
            try:
                rdkit_mol = Chem.MolFromSmiles(smi)
                if rdkit_mol is None:
                    continue
                atom_types = np.asarray(
                    [a.GetAtomicNum() for a in rdkit_mol.GetAtoms()],
                    dtype=np.int64,
                )
                n_atoms = len(atom_types)
                # 3D coords are not used by Vina (Vina re-embeds the
                # SMILES itself), so we emit zeros and let downstream
                # code re-embed.
                coords = np.zeros((n_atoms, 3), dtype=np.float32)
                mol_obj = Molecule(
                    coords=torch.from_numpy(coords),
                    atom_types=torch.from_numpy(atom_types),
                    bonds=torch.zeros(2, 0, dtype=torch.long),
                    bond_types=torch.zeros(0, dtype=torch.long),
                    formal_charges=torch.zeros(n_atoms, dtype=torch.long),
                    smiles=smi,
                )
                complexes.append(Complex(pocket=pocket, molecule=mol_obj))
            except Exception:
                continue
        return complexes


# ---------------------------------------------------------------------------
# Public adapter — picks live vs fallback
# ---------------------------------------------------------------------------
@dataclass
class Pocket2MolAdapter:
    """Wraps Pocket2Mol as a ``MoleculeGenerator``-style port.

    Usage::

        adapter = Pocket2MolAdapter()
        complexes = adapter.sample(pocket, n_samples=100)
        # complexes: List[Complex] — feed directly into VinaDockingAdapter

    The adapter is **stateless** apart from the cached model (when the
    live backend is in use); calling :meth:`sample` multiple times on
    different pockets is supported.
    """

    device: str = "cpu"
    seed: int = 42
    use_live_if_available: bool = True
    _backend: Optional[object] = field(default=None, init=False, repr=False)
    mode: str = field(default="", init=False)

    def setup(self) -> None:
        if self.use_live_if_available and _checkpoint_available():
            try:
                self._backend = _Pocket2MolLiveAdapter(device=self.device)
                self.mode = "live_Pocket2Mol"
                return
            except Exception as exc:
                logger.warning(
                    "Live Pocket2Mol backend could not be initialised (%s); "
                    "falling back to SMARTS-only baseline.",
                    exc,
                )
        self._backend = _Pocket2MolFallbackAdapter(seed=self.seed)
        self.mode = "smarts_fallback"

    @property
    def name(self) -> str:
        return "Pocket2Mol_v1"

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode or "uninitialised",
            "checkpoint_on_disk": _checkpoint_available(),
            "checkpoint_path": str(_checkpoint_path()),
            "config_path": str(POCKET2MOL_CONFIG),
            "repo_path": str(POCKET2MOL_REPO),
            "fallback_pool_size": len(_FALLBACK_SMILES_POOL),
        }

    def sample(
        self,
        pocket: Pocket,
        n_samples: int = 100,
        bbox_size: float = 23.0,
    ) -> List[Complex]:
        """Generate ``n_samples`` candidate ligands for ``pocket``.

        Returns
        -------
        list[Complex]
            Each Complex has ``molecule.smiles`` populated; the
            downstream VinaDockingAdapter handles 3D embedding.
        """
        if self._backend is None:
            self.setup()
        if self.mode == "live_Pocket2Mol":
            return self._backend.sample(
                pocket, n_samples=int(n_samples), bbox_size=bbox_size,
            )
        return self._backend.sample(pocket, n_samples=int(n_samples))


__all__ = [
    "Pocket2MolAdapter",
    "_Pocket2MolLiveAdapter",
    "_Pocket2MolFallbackAdapter",
    "_checkpoint_available",
    "_checkpoint_path",
]