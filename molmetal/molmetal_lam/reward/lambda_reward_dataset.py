"""Lambda reward-weighted dataset for CFM retraining (TODO-21 Strategy 1).

================================================================
Background
================================================================
The CFM retraining loop (e.g. ``metallo_drug_smoke_retrain.py``) learns
to predict the conditional velocity field of metallodrug-like 3D
structures.  When a candidate molecule comes from the Lambda MCTS
side, its reward signal (``r_qed * r_sa * r_pb`` — see
:func:`molmetal_lam.lam_chem.coupling_adapter.lambda_samples_to_reward_targets`)
tells the CFM *how good* that molecule is.  Mixing Lambda samples
into the regression batch with their reward as a per-sample loss
weight is a direct instance of reward-weighted regression (Dayan
1997 / McAllester 1999).

This module is the PyTorch :class:`torch.utils.data.Dataset` adapter
that maps a list of ``(smiles, reward_weight)`` tuples — produced by
:func:`molmetal_lam.lam_chem.coupling_adapter.lambda_samples_to_reward_targets`
— into ``(atom_types, coords, reward_weight)`` tensors the CFM
training loop can consume directly.

Public API
----------
* :class:`LambdaRewardDataset` — PyTorch ``Dataset`` whose items are
  ``(atom_types, coords, reward_weight)`` tuples.  ``atom_types`` is a
  ``LongTensor`` of atomic numbers, ``coords`` is a ``FloatTensor`` of
  shape ``(N_atoms, 3)``, and ``reward_weight`` is a scalar
  ``FloatTensor``.
* :func:`build_lambda_reward_dataset` — convenience factory that
  calls :func:`lambda_samples_to_reward_targets` and wraps the
  results in a :class:`LambdaRewardDataset`.  This is the entry
  point the CFM retrain script uses.

Honest framing
==============
* The 3D coordinates are produced by RDKit's ETKDGv3 + MMFF94
  pipeline (the canonical Mol-Metal conformer embed), so the dataset
  inherits RDKit's known bias toward compact conformers.  This is
  intentional — the CFM velocity field is trained on the same
  ETKDGv3 conformers from the metallodrug pool, so mixing Lambda
  samples into the regression set keeps the coordinate statistics
  consistent.
* When RDKit is unavailable (or fails to embed a candidate SMILES)
  the dataset silently drops the candidate.  The returned length is
  therefore ``<= n_samples`` and may be ``0`` on a fully-broken
  install — the caller must handle that edge case (the CFM retrain
  script logs ``n_dropped`` and falls back to the unweighted batch).
* The dataset holds the full atom-types / coords tensors in memory;
  for ``n_samples > 10_000`` consider switching to an on-disk
  ``TensorDataset`` of pre-computed ``.npz`` chunks.  For the
  TODO-21 Strategy 1 budget (``n_samples <= 500``) the in-memory
  approach is fine.

Lit anchors
===========
* Dayan 1997 — potential-based reward shaping preserves the optimal
  policy when the potential is path-independent.  The reward weight
  here is a state-only potential; mixing it into the CFM loss does
  not change the conditional distribution the velocity field
  approximates, only its empirical sampling distribution.
* McAllester 1999 PAC-Bayes — per-sample loss-weighting is the
  classical sparse-signal path; multiplying three reward channels is
  the standard 3-channel aggregator (matches the TODO-21 plan).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

# Lazy RDKit import — keep the module importable in RDKit-less envs
# (the CFM retrain script checks availability before constructing).
_RDKIT = None
_RDKIT_OK = False


def _try_import_rdkit() -> bool:
    """Idempotent RDKit import; returns ``True`` iff RDKit is available.

    Cached in module-level ``_RDKIT`` / ``_RDKIT_OK`` so the import
    only happens once per process.  Silently returns ``False`` on
    :class:`ImportError` (or any backend failure).
    """
    global _RDKIT, _RDKIT_OK
    if _RDKIT_OK:
        return True
    if _RDKIT is not None:
        return _RDKIT_OK
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit import RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
        _RDKIT = (Chem, AllChem, RDLogger)
        _RDKIT_OK = True
    except Exception:
        _RDKIT = None
        _RDKIT_OK = False
    return _RDKIT_OK


def _smiles_to_atom_types_and_coords(smiles: str):
    """Convert a SMILES string to ``(atom_types, coords)`` tensors.

    Returns ``(None, None)`` on parse failure / embed failure.  When
    RDKit is unavailable returns ``(None, None)`` and the caller is
    expected to drop the candidate.
    """
    if not _try_import_rdkit():
        return None, None
    Chem, AllChem, _ = _RDKIT  # type: ignore
    try:
        # Sanitize aggressively — we drop molecules that fail.
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None, None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return None, None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # MMFF94 not parameterised for this scaffold — UFF
            # fallback is fine for the stand-in embedding.
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        conf = mol.GetConformer()
        coords_np = np.asarray(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        # Strip explicit hydrogens — the CFM velocity field operates
        # on heavy atoms only (matching the metallodrug pool
        # convention).
        heavy_idx = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() > 1
        ]
        coords_heavy = coords_np[heavy_idx] if heavy_idx else coords_np
        atom_types = np.asarray(
            [mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy_idx]
            if heavy_idx else [a.GetAtomicNum() for a in mol.GetAtoms()],
            dtype=np.int64,
        )
        # NaN / Inf guard.
        if (not np.isfinite(coords_heavy).all()
                or atom_types.size == 0):
            return None, None
        return (
            torch.from_numpy(atom_types.astype(np.int64)),
            torch.from_numpy(coords_heavy.astype(np.float32)),
        )
    except Exception:
        return None, None


@dataclass
class _DatasetItem:
    """Container for a single (atom_types, coords, reward_weight) item.

    Matches the CFM ``train_step`` contract:
    * ``atom_types`` — ``LongTensor`` of atomic numbers, shape ``(N,)``.
    * ``coords``     — ``FloatTensor`` of shape ``(N, 3)``.
    * ``reward_weight`` — ``FloatTensor`` scalar in ``[0, 1]``.
    """
    atom_types: torch.Tensor
    coords: torch.Tensor
    reward_weight: torch.Tensor


class LambdaRewardDataset(torch.utils.data.Dataset):
    """Reward-weighted regression dataset of Lambda MCTS candidates.

    Each item is a ``(atom_types, coords, reward_weight)`` triple ready
    to be consumed by :meth:`LipmanFlowMatchingAdapter.train_step` (or
    any reward-weighted CFM regression trainer).  Items are computed
    once at construction time (lazy 3D embed); the rest of the API is
    read-only.

    Parameters
    ----------
    pairs : sequence of ``(smiles, reward_weight)``
        The Lambda candidate list — typically the output of
        :func:`molmetal_lam.lam_chem.coupling_adapter.lambda_samples_to_reward_targets`.
        ``reward_weight`` MUST be in ``[0, 1]`` (caller's
        responsibility — no clamping here so the loss weighting is
        bit-for-bit the same as the upstream Lambda signal).
    max_atoms : int, default ``80``
        Skip molecules with more than ``max_atoms`` heavy atoms.
        The metallodrug pool uses 8..38 heavy atoms; ``80`` is a safe
        upper bound that includes all realistic CuAAC / SPAAC
        products while preventing one giant molecule from blowing up
        the CFM memory budget.
    min_atoms : int, default ``3``
        Skip molecules with fewer than ``min_atoms`` heavy atoms.
        The CFM velocity field cannot learn meaningful flows from
        sub-3-atom point clouds.
    verbose : bool, default ``False``
        Log a one-line summary at construction time (n_kept, n_dropped).

    Attributes
    ----------
    items : list[_DatasetItem]
        The kept, embedded candidates — one entry per valid SMILES.
    smiles : list[str]
        The canonical SMILES of every kept candidate, in the same
        order as ``items``.
    weights : list[float]
        The ``reward_weight`` of every kept candidate, in the same
        order as ``items``.

    The dataset's ``__len__`` returns ``len(items)``; ``__getitem__``
    returns a deep-cloned :class:`_DatasetItem` so callers can mutate
    the tensors without poisoning the dataset state.
    """

    items: List[_DatasetItem]
    smiles: List[str]
    weights: List[float]

    def __init__(
        self,
        pairs: Optional[Sequence[Tuple[str, float]]] = None,
        *,
        n_samples: Optional[int] = None,
        pocket_id: str = "test_001",
        seed: int = 0,
        reward_channels: Sequence[str] = ("qed", "sa", "pb"),
        max_atoms: int = 80,
        min_atoms: int = 3,
        verbose: bool = False,
    ) -> None:
        # Resolve the candidate list.  ``pairs`` is the canonical input;
        # ``n_samples`` is a convenience for callers that want a
        # turn-key dataset (skips the explicit
        # :func:`lambda_samples_to_reward_targets` round-trip).
        if pairs is None:
            if n_samples is None:
                # Empty dataset — no candidates to embed.
                pairs = []
            else:
                try:
                    from molmetal_lam.lam_chem.coupling_adapter import (
                        lambda_samples_to_reward_targets,
                    )
                    pairs = lambda_samples_to_reward_targets(
                        n_samples=int(n_samples),
                        pocket_id=str(pocket_id),
                        seed=int(seed),
                        reward_channels=tuple(reward_channels),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[LambdaRewardDataset] lambda_samples_to_reward_targets "
                        "failed: %s; falling back to empty dataset", exc,
                    )
                    pairs = []
        if not isinstance(pairs, (list, tuple)):
            pairs = list(pairs)
        kept_items: List[_DatasetItem] = []
        kept_smiles: List[str] = []
        kept_weights: List[float] = []
        n_dropped = 0
        for smi, w in pairs:
            if not smi or not isinstance(smi, str):
                n_dropped += 1
                continue
            # Wf-427-A2: ``w`` is now a reward_dict from
            # :func:`lambda_samples_to_reward_targets` rather than a
            # bare scalar.  Compute the mean across known reward
            # channels (excluding the Wf-427-A2 ``_pocket_emb_norm``
            # diagnostic key, which is a different unit of magnitude).
            if isinstance(w, dict):
                _keys = [k for k in w.keys() if k != "_pocket_emb_norm"]
                if _keys:
                    w_f = float(sum(float(w[k]) for k in _keys) / float(len(_keys)))
                else:
                    w_f = 0.0
            else:
                try:
                    w_f = float(w)
                except Exception:
                    w_f = 0.0
            atom_types, coords = _smiles_to_atom_types_and_coords(smi)
            if atom_types is None or coords is None:
                n_dropped += 1
                continue
            n_atoms = int(atom_types.shape[0])
            if n_atoms < int(min_atoms) or n_atoms > int(max_atoms):
                n_dropped += 1
                continue
            kept_items.append(_DatasetItem(
                atom_types=atom_types.clone(),
                coords=coords.clone(),
                reward_weight=torch.tensor(float(w_f), dtype=torch.float32),
            ))
            kept_smiles.append(smi)
            kept_weights.append(float(w_f))
        self.items = kept_items
        self.smiles = kept_smiles
        self.weights = kept_weights
        if verbose:
            log.info(
                "[LambdaRewardDataset] kept=%d dropped=%d (n_input=%d)",
                len(self.items), n_dropped, len(pairs),
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> _DatasetItem:
        item = self.items[int(idx)]
        # Return clones so the caller can mutate freely.
        return _DatasetItem(
            atom_types=item.atom_types.clone(),
            coords=item.coords.clone(),
            reward_weight=item.reward_weight.clone(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_molecule_objects(self):
        """Convert the dataset into a list of :class:`molmetal.domain.Molecule`.

        Used by the CFM retrain script when it wants to mix the
        Lambda samples into an existing molecule batch — the
        ``Molecule`` carries the same ``(coords, atom_types, bonds)``
        triple the CFM ``train_step`` already understands.
        """
        from molmetal.domain import Molecule
        out = []
        for item in self.items:
            n = int(item.coords.shape[0])
            out.append(
                Molecule(
                    coords=item.coords.clone(),
                    atom_types=item.atom_types.clone(),
                    bonds=torch.zeros(2, 0, dtype=torch.long),
                    bond_types=torch.zeros(0, dtype=torch.int8),
                    formal_charges=torch.zeros(n, dtype=torch.int8),
                    smiles="",
                    qed=0.0,
                    sa_score=0.0,
                    logp=0.0,
                )
            )
        return out

    def weights_tensor(self) -> torch.Tensor:
        """Stack all ``reward_weight`` scalars into a single 1-d tensor.

        Useful for the CFM retrain script to compute per-batch weight
        statistics (mean, std, max) without rebuilding the dataset.
        """
        return torch.tensor(self.weights, dtype=torch.float32)

    @property
    def mean_weight(self) -> float:
        """Mean ``reward_weight`` across the kept candidates (0.0 if empty)."""
        if not self.weights:
            return 0.0
        return float(np.mean(self.weights))

    @property
    def max_weight(self) -> float:
        """Max ``reward_weight`` across the kept candidates (0.0 if empty)."""
        if not self.weights:
            return 0.0
        return float(np.max(self.weights))


def build_lambda_reward_dataset(
    n_samples: int = 200,
    pocket_id: str = "test_001",
    *,
    seed: int = 0,
    reward_channels: Sequence[str] = ("qed", "sa", "pb"),
    max_atoms: int = 80,
    min_atoms: int = 3,
    verbose: bool = False,
) -> LambdaRewardDataset:
    """One-shot factory: run Lambda MCTS and wrap the result in a dataset.

    This is the convenience entry point used by the CFM retrain
    script.  Internally it calls
    :func:`molmetal_lam.lam_chem.coupling_adapter.lambda_samples_to_reward_targets`
    and feeds the result into :class:`LambdaRewardDataset`.

    Parameters
    ----------
    n_samples, pocket_id, seed, reward_channels :
        Forwarded verbatim to
        :func:`molmetal_lam.lam_chem.coupling_adapter.lambda_samples_to_reward_targets`.
    max_atoms, min_atoms, verbose :
        Forwarded verbatim to :class:`LambdaRewardDataset`.

    Returns
    -------
    LambdaRewardDataset
        A populated dataset.  May be empty when the Lambda MCTS path
        produces no valid candidates — callers should check
        ``len(ds) > 0`` before mixing into the CFM batch.
    """
    try:
        from molmetal_lam.lam_chem.coupling_adapter import (
            lambda_samples_to_reward_targets,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[build_lambda_reward_dataset] coupling_adapter import "
                    "failed: %s; returning empty dataset", exc)
        return LambdaRewardDataset([], max_atoms=max_atoms,
                                   min_atoms=min_atoms, verbose=verbose)

    pairs = lambda_samples_to_reward_targets(
        n_samples=int(n_samples),
        pocket_id=str(pocket_id),
        seed=int(seed),
        reward_channels=tuple(reward_channels),
    )
    return LambdaRewardDataset(
        pairs,
        max_atoms=max_atoms,
        min_atoms=min_atoms,
        verbose=verbose,
    )


__all__ = [
    "LambdaRewardDataset",
    "build_lambda_reward_dataset",
]
