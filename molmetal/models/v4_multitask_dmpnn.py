"""V4MultiTaskDMPNN — 5-way multi-task D-MPNN for precious-metal cytotoxicity.

Stage 1 of the R17 AUC-recovery plan
(``TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`` §2): a single D-MPNN
encoder is shared across **five** precious-metal cytotoxicity heads
(Ru / Ir / Rh / Re / Os) plus a coordination-geometry auxiliary head.

The motivation is data efficiency: the per-metal MetalCytoToxDB subsets
are individually small (~3K rows for Ru); pooling all five metals lifts
the joint training set to ~15K rows, and the metal embedding acts as a
``metal-conditional`` switch so each head specialises on its own subset.

Architecture (matches the spec in
``TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`` §2 task 2):

    smiles ─► D-MPNN encoder ─► h            (B, hidden_dim)
    metal_id ─► nn.Embedding  ─► e           (B, 64)
    z = [h ; e]                               (B, hidden_dim + 64)

    per-metal heads (5×):
        Linear(z → 128) → ReLU → Dropout(0.2) → Linear(128 → 2)

    coord head:
        Linear(z → 64) → ReLU → Linear(64 → n_coord_classes)

Loss:

    L = Σ_m CE(logits[m], y_true[m])  +  0.3 · CE(coord_logits, coord_true)

Note on the encoder wrapper:
    ``molmetal.models.dmpnn.DirectedMPNN.forward`` expects already-padded
    graph tensors.  We wrap it in :class:`DMPNNEncoder` that handles SMILES
    featurisation (via :class:`molmetal.data.featurize.GraphFeaturizer`),
    padding to (N_max, E_max), batched message passing, and sum-pool over
    atoms → a (B, hidden_dim) graph embedding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from molmetal.data.featurize import GraphFeaturizer
from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig
from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
    UncertaintyWeightedMultiTaskLoss,
)


# ---------------------------------------------------------------------------
# Constants — keep in sync with the R17 plan
# ---------------------------------------------------------------------------
#: Canonical 5-way metal subset for precious-metal cytotoxicity.
SUPPORTED_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Re", "Os")

#: Coordination-geometry class names (5 classes — matches the
#: ``n_coord_classes=5`` default).
COORD_GEOMETRY_NAMES: Tuple[str, ...] = (
    "linear",  # 0
    "trigonal_planar",  # 1
    "tetrahedral",  # 2
    "square_planar",  # 3
    "octahedral",  # 4
)

#: Loss weight on the coordination-geometry auxiliary task (per spec).
COORD_LOSS_WEIGHT: float = 0.3

#: Default dropout in the per-metal heads.
HEAD_DROPOUT: float = 0.2


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------
class DMPNNEncoder(nn.Module):
    """Directed-MPNN encoder that turns a list of SMILES into a pooled
    (B, hidden_dim) graph embedding.

    Wraps :class:`molmetal.models.dmpnn.DirectedMPNN` with the
    featurisation + padding plumbing so the rest of the multi-task
    model can call ``encoder(smiles_list)`` and get a clean
    (B, hidden_dim) tensor back.

    The encoder is **shared** across all metal heads — this is the
    main efficiency win over per-metal single-task models.
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        n_layers: int = 3,
        atom_feat_dim: int = 39,  # GraphFeaturizer default
        edge_feat_dim: int = 6,  # GraphFeaturizer default
        dropout: float = 0.0,
        strip_metal_atoms: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.atom_feat_dim = int(atom_feat_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        # Wf-ultracode (Defect 1 — silent ligand-only graphs): when
        # ``strip_metal_atoms=True`` (default), the encoder removes any
        # precious-metal atom from the SMILES (Pt, Pd, Ru, Ir, Rh, Os,
        # Re, Au) before RDKit parsing.  The metal id is still passed
        # via ``metal_embed`` / ``metal_ids``, so no information is lost
        # — only the double-counting between the graph path and the
        # metal-id tensor.  This makes platinai (4078/4078 rows carry
        # ``[Pt+2]``) consistent with v4 / NCI60 (0/26632 and 0/7823
        # carry metal tokens): the encoder always sees ligand-only
        # graphs.  Set ``strip_metal_atoms=False`` to restore the
        # legacy encoding for callers that need the metal atom in the
        # graph.
        self.strip_metal_atoms = bool(strip_metal_atoms)
        # Featurizer is stateless and side-effect-free; safe to share.
        self.featurizer = GraphFeaturizer()
        cfg = MPNNConfig(
            atom_feat_dim=self.atom_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=dropout,
        )
        self.dmpnn = DirectedMPNN(cfg)

    # ------------------------------------------------------------------
    # Featurise SMILES → padded batch tensors
    # ------------------------------------------------------------------
    def _featurize_batch(
        self,
        smiles_list: List[str],
        device: torch.device,
        mol_objects: Optional[List] = None,
        cache: Optional[Dict] = None,
        batch_idx_key: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,  # h_atom (B, N_max, atom_feat_dim)
        torch.Tensor,  # edge_index (B, 2, E_max)
        torch.Tensor,  # edge_attr (B, E_max, edge_feat_dim)
        torch.Tensor,  # batch_idx (B, N_max)
        torch.Tensor,  # atom_mask (B, N_max) bool
    ]:
        from rdkit import Chem, RDLogger

        # Local import to avoid a top-level circular dependency with
        # ``molmetal.data.featurize`` (the latter imports nothing from
        # ``molmetal.models`` so this is cheap).
        from molmetal.data.featurize import _strip_metal_atoms

        RDLogger.DisableLog("rdApp.*")
        B = len(smiles_list)
        atom_feats, edge_indices, edge_attrs, n_atoms_list = [], [], [], []
        for i, smi in enumerate(smiles_list):
            # Wf-327-F8 / Wf-328-W10 — cache look-up before RDKit parse.
            # Two cache APIs are supported:
            #   1. DMPNNFeaturizationCache (molmetal.models.dmpnn_cache) —
            #      populated by ``cache.precompute_smiles``; per-SMILES
            #      cache populated before the batch loop.  On a cache hit
            #      we skip the RDKit parse + GraphFeaturizer entirely.
            #   2. Raw dict keyed by (smiles, batch_idx_key) — populated
            #      on the first call, reused on the second (useful for
            #      callers that don't precompute).
            _cache_hit: Optional[Dict] = None
            if cache is not None:
                # API (1): DMPNNFeaturizationCache with a precomputed
                # ``_mols`` dict.  Hit short-circuits the parse path.
                if hasattr(cache, "_mols") and smi in cache._mols:
                    _mol_cache_entry = cache._mols[smi]
                    _cache_hit = (
                        _mol_cache_entry["x"],
                        _mol_cache_entry["edge_index"],
                        _mol_cache_entry["edge_attr"],
                        int(_mol_cache_entry["n_atoms"]),
                    )
                elif isinstance(cache, dict):
                    _cache_key = (smi, batch_idx_key)
                    _cache_hit = cache.get(_cache_key)
            if _cache_hit is None:
                try:
                    if mol_objects is not None and mol_objects[i] is not None:
                        mol = mol_objects[i]
                    else:
                        # Wf-ultracode (Defect 1): strip precious-metal atoms
                        # before RDKit parsing so the encoder sees ligand-only
                        # graphs across all cohorts.  ``_strip_metal_atoms``
                        # is a no-op for SMILES that don't carry a metal
                        # token (the v4 / NCI60 case).
                        parse_smi = (
                            _strip_metal_atoms(str(smi))[0]
                            if self.strip_metal_atoms
                            else str(smi)
                        )
                        mol = Chem.MolFromSmiles(parse_smi)
                    if mol is None:
                        raise ValueError(f"RDKit parse failed for {smi!r}")
                    feat = self.featurizer(mol)
                    atom_feats.append(feat["x"])
                    edge_indices.append(feat["edge_index"])
                    edge_attrs.append(feat["edge_attr"])
                    n_atoms_list.append(int(feat["n_atoms"]))
                    # Wf-327-F8 / Wf-328-W10 — store on cache hit-miss path
                    # so subsequent batch passes reuse the parsed features.
                    if cache is not None and isinstance(cache, dict):
                        cache[(smi, batch_idx_key)] = (
                            feat["x"],
                            feat["edge_index"],
                            feat["edge_attr"],
                            int(feat["n_atoms"]),
                        )
                    # Wf-328-W10 — also feed the per-SMILES tier of
                    # DMPNNFeaturizationCache so subsequent passes hit
                    # via the canonical (per-SMILES) key.
                    if cache is not None and hasattr(cache, "_mols") and smi not in cache._mols:
                        cache._mols[smi] = {
                            "x": feat["x"],
                            "edge_index": feat["edge_index"],
                            "edge_attr": feat["edge_attr"],
                            "n_atoms": int(feat["n_atoms"]),
                            "n_bonds": int(feat.get("n_bonds", 0)),
                            "mol": None,
                        }
                except Exception:
                    atom_feats.append(np.zeros((1, self.atom_feat_dim), dtype=np.float32))
                    edge_indices.append(np.zeros((2, 0), dtype=np.int64))
                    edge_attrs.append(np.zeros((0, self.edge_feat_dim), dtype=np.float32))
                    n_atoms_list.append(1)
            else:
                # Cache hit — restore the previously-featurized tensors.
                _cached_x, _cached_ei, _cached_ea, _cached_n = _cache_hit
                atom_feats.append(_cached_x)
                edge_indices.append(_cached_ei)
                edge_attrs.append(_cached_ea)
                n_atoms_list.append(int(_cached_n))

        N_max = max(max(n_atoms_list), 1)
        E_max = max(max(ei.shape[1] for ei in edge_indices), 1)

        h_atom = torch.zeros(B, N_max, self.atom_feat_dim, device=device)
        edge_index = torch.zeros(B, 2, E_max, dtype=torch.long, device=device)
        edge_attr = torch.zeros(B, E_max, self.edge_feat_dim, device=device)
        batch_idx = torch.full((B, N_max), -1, dtype=torch.long, device=device)
        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            af = torch.from_numpy(atom_feats[b]).float()
            n_a = min(af.size(0), N_max)
            h_atom[b, :n_a] = af[:n_a]
            atom_mask[b, :n_a] = True
            ei = torch.from_numpy(edge_indices[b]).long()
            n_e = min(ei.size(1), E_max)
            edge_index[b, :, :n_e] = ei[:, :n_e]
            ea = torch.from_numpy(edge_attrs[b]).float()
            n_ea = min(ea.size(0), E_max)
            edge_attr[b, :n_ea] = ea[:n_ea]
            batch_idx[b, :n_a] = b

        # Wf-328-W10 — populate the per-batch tier of
        # DMPNNFeaturizationCache so subsequent calls with the same
        # ``batch_idx_key`` hit on the second pass.  Stat accounting:
        # the FIRST store is a miss (no prior entry in ``_batch_cache``);
        # subsequent stores are hits.
        if (
            cache is not None
            and hasattr(cache, "_batch_cache")
            and batch_idx_key is not None
        ):
            _was_hit = batch_idx_key in cache._batch_cache
            cache._batch_cache[batch_idx_key] = {
                "h_atom": h_atom,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "batch_idx": batch_idx,
                "atom_mask": atom_mask,
            }
            if _was_hit:
                cache.stats.n_batch_hit += 1
            else:
                cache.stats.n_batch_miss += 1

        return h_atom, edge_index, edge_attr, batch_idx, atom_mask

    # ------------------------------------------------------------------
    def _build_batch_idx_from_mask(
        self,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a (B, N_max) batch_idx from a (B, N_max) bool atom_mask.

        Each row ``b`` of ``batch_idx`` equals ``b`` at positions where
        ``atom_mask[b]`` is True, and ``-1`` elsewhere.  This mirrors
        what :meth:`_featurize_batch` returns, so the cached and
        non-cached paths produce tensor shapes that match
        :meth:`DirectedMPNN.forward`.
        """
        B, N_max = atom_mask.shape
        batch_idx = torch.full(
            (B, N_max), -1, dtype=torch.long, device=atom_mask.device
        )
        # b-th row filled with b where atom_mask is True.
        # Using arange is cheaper than a python loop on CPU and a single
        # broadcast on GPU.
        row_idx = torch.arange(B, device=atom_mask.device).unsqueeze(1)
        batch_idx = torch.where(atom_mask, row_idx.expand(B, N_max), batch_idx)
        return batch_idx

    # ------------------------------------------------------------------
    def forward_cached(
        self,
        atom_features: torch.Tensor,  # (B, N_max, atom_feat_dim)
        edge_index: torch.Tensor,  # (B, 2, E_max)
        atom_mask: Optional[torch.Tensor] = None,  # (B, N_max) bool
        edge_attr: Optional[torch.Tensor] = None,  # (B, E_max, edge_feat_dim)
    ) -> torch.Tensor:
        """Encode pre-featurized graph tensors → (B, hidden_dim) embedding.

        **Skips RDKit entirely** — caller is responsible for producing
        correctly-shaped ``atom_features`` / ``edge_index`` (e.g. via
        ``molmetal.data.pre_featurized`` offline featurisation).

        Args:
            atom_features: ``(B, N_max, atom_feat_dim)`` float tensor —
                already padded.  Rows with ``atom_mask[b, n] == False``
                are ignored.
            edge_index: ``(B, 2, E_max)`` long tensor — directed edges
                per molecule, padded to a common ``E_max``.
            atom_mask: optional ``(B, N_max)`` bool tensor.  When ``None``,
                every slot is treated as valid (full-batch, no padding).
            edge_attr: optional ``(B, E_max, edge_feat_dim)`` float
                tensor.  When ``None``, zero-features are used (D-MPNN
                still trains but loses bond-order signal).

        Returns:
            ``(B, hidden_dim)`` pooled molecular embedding.
        """
        device = next(self.parameters()).device
        atom_features = atom_features.to(device)
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        else:
            B, _, E_max = edge_index.shape
            edge_attr = torch.zeros(
                B,
                E_max,
                self.edge_feat_dim,
                device=device,
                dtype=atom_features.dtype,
            )

        if atom_mask is None:
            # No padding → every atom slot is valid; build the (B, N_max)
            # batch_idx by tiling arange(B) across N_max slots.
            B, N_max, _ = atom_features.shape
            atom_mask = torch.ones(B, N_max, dtype=torch.bool, device=device)
        else:
            atom_mask = atom_mask.to(device)

        batch_idx = self._build_batch_idx_from_mask(atom_mask)
        h = self.dmpnn(atom_features, edge_index, edge_attr, batch_idx)
        # Zero-out embeddings of fully-padded (invalid) molecules so
        # they can't accidentally dominate downstream heads.
        valid_mol = atom_mask.any(dim=1)  # (B,) any real atom
        h = h * valid_mol.unsqueeze(-1).float()
        return h

    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_list: List[str],
        mol_objects: Optional[List] = None,
    ) -> torch.Tensor:
        """Encode a list of SMILES → (B, hidden_dim) pooled embedding.

        Invalid SMILES are routed through a single-atom dummy graph so
        they contribute zero to the gradient but still produce a
        well-shaped embedding (sum-pool over 1 zero atom = 0 vector).
        """
        if not smiles_list:
            raise ValueError("smiles_list must be non-empty")
        device = next(self.parameters()).device
        h_atom, edge_index, edge_attr, batch_idx, atom_mask = self._featurize_batch(
            smiles_list, device, mol_objects=mol_objects
        )
        # DirectedMPNN.forward returns (B, hidden_dim) — sum-pooled.
        h = self.dmpnn(h_atom, edge_index, edge_attr, batch_idx)
        # Zero-out embeddings of fully-padded (invalid) molecules so
        # they can't accidentally dominate downstream heads.
        valid_mol = atom_mask.any(dim=1)  # (B,) any real atom
        h = h * valid_mol.unsqueeze(-1).float()
        return h


# ---------------------------------------------------------------------------
# Per-metal head factory
# ---------------------------------------------------------------------------
def _make_metal_head(
    in_dim: int,
    hidden_dim: int = 128,
    n_classes: int = 2,
    dropout: float = HEAD_DROPOUT,
) -> nn.Sequential:
    """Build one metal head: Linear → ReLU → Dropout → Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, n_classes),
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class V4MultiTaskDMPNN(nn.Module):
    """5-way multi-task D-MPNN for precious-metal cytotoxicity.

    Args:
        hidden_dim: D-MPNN hidden dimension (default 300 per spec).
        n_layers: D-MPNN message-passing layers (default 3).
        n_metals: number of metal embeddings (default 5; supports the
            canonical Ru/Ir/Rh/Re/Os subset).
        n_coord_classes: number of coordination-geometry classes for
            the auxiliary head (default 5).
        metal_embed_dim: metal-embedding dimension (default 64 per spec).
        metals: tuple of metal names — must have ``n_metals`` entries.
            Used as keys for the per-metal ``ModuleDict`` and as a
            public label set.
        head_hidden_dim: per-metal head hidden width (default 128).
        head_dropout: per-metal head dropout (default 0.2).
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        n_layers: int = 3,
        n_metals: int = 5,
        n_coord_classes: int = 5,
        metal_embed_dim: int = 64,
        metals: Optional[Tuple[str, ...]] = None,
        head_hidden_dim: int = 128,
        head_dropout: float = HEAD_DROPOUT,
        use_cached_encoding: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_metals = int(n_metals)
        self.n_coord_classes = int(n_coord_classes)
        self.metal_embed_dim = int(metal_embed_dim)
        # Cached-encoding flag — settable at init (defaults to False for
        # backward compat with the SMILES-based forward path).
        self.use_cached_encoding: bool = bool(use_cached_encoding)
        # Default to the canonical 5-way precious-metal subset.
        self.metals: Tuple[str, ...] = (
            tuple(metals) if metals is not None else SUPPORTED_METALS[:n_metals]
        )
        if len(self.metals) != self.n_metals:
            raise ValueError(
                f"len(metals)={len(self.metals)} must equal n_metals={self.n_metals}"
            )

        # ----- Shared encoder -----
        self.encoder = DMPNNEncoder(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )

        # ----- Metal embedding (one row per metal id) -----
        self.metal_embed = nn.Embedding(self.n_metals, self.metal_embed_dim)

        # ----- Per-metal heads (ModuleDict so named access mirrors spec) -----
        head_in = self.hidden_dim + self.metal_embed_dim
        self.metal_heads = nn.ModuleDict(
            {
                m: _make_metal_head(
                    in_dim=head_in,
                    hidden_dim=head_hidden_dim,
                    n_classes=2,
                    dropout=head_dropout,
                )
                for m in self.metals
            }
        )

        # ----- Coordination-geometry auxiliary head -----
        self.coord_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_coord_classes),
        )

        # NOTE: ``self.use_cached_encoding`` is set in ``__init__`` above
        # so the constructor argument ``use_cached_encoding=...`` wins.

    # ------------------------------------------------------------------
    def metal_id(self, metal: str) -> int:
        """Return the integer id for a metal name (raises if unknown)."""
        if metal not in self.metals:
            raise KeyError(f"Unknown metal {metal!r}. Supported: {self.metals}")
        return self.metals.index(metal)

    # ------------------------------------------------------------------
    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode SMILES → (B, hidden_dim)."""
        return self.encoder(smiles_list)

    # ------------------------------------------------------------------
    def encode_cached(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode pre-featurized tensors → (B, hidden_dim).

        Skips RDKit entirely.  See
        :meth:`DMPNNEncoder.forward_cached` for arg shapes.
        """
        return self.encoder.forward_cached(
            atom_features,
            edge_index,
            atom_mask=atom_mask,
            edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    def forward_cached(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        metal_ids: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with pre-featurized graph tensors (no RDKit).

        Args:
            atom_features: ``(B, N_max, atom_feat_dim)`` float tensor.
            edge_index: ``(B, 2, E_max)`` long tensor.
            metal_ids: ``(B,)`` long tensor with ids in ``[0, n_metals)``.
            atom_mask: optional ``(B, N_max)`` bool tensor — when ``None``
                all slots are treated as valid.
            edge_attr: optional ``(B, E_max, edge_feat_dim)`` float
                tensor — when ``None`` zeros are used.

        Returns:
            Same dict shape as :meth:`forward` — one ``(B, 2)`` logit
            tensor per metal plus ``"coord" → (B, n_coord_classes)``.
        """
        if metal_ids.dim() != 1:
            raise ValueError(
                f"metal_ids must be 1-D (B,), got shape {tuple(metal_ids.shape)}"
            )
        B = atom_features.size(0)
        if metal_ids.size(0) != B:
            raise ValueError(
                f"atom_features.size(0)={B} != metal_ids.size(0)="
                f"{metal_ids.size(0)}"
            )

        h = self.encode_cached(
            atom_features,
            edge_index,
            atom_mask=atom_mask,
            edge_attr=edge_attr,
        )  # (B, hidden_dim)
        e = self.metal_embed(metal_ids.long())  # (B, metal_embed_dim)
        z = torch.cat([h, e], dim=-1)  # (B, hidden_dim + metal_embed_dim)

        out: Dict[str, torch.Tensor] = {
            m: head(z) for m, head in self.metal_heads.items()
        }
        out["coord"] = self.coord_head(z)
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_or_atoms,
        metal_ids: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Two calling conventions are supported, selected by the type of
        ``smiles_or_atoms``:

        * **SMILES path** (default, backward-compatible)::

              out = model(["CCO", "c1ccccc1"], metal_ids=torch.tensor([0, 1]))

          Featurises SMILES on the fly via RDKit + ``GraphFeaturizer`` —
          this is the original V4 behaviour and the default even when
          ``self.use_cached_encoding`` is set, because SMILES are always
          available.

        * **Cached path** (``self.use_cached_encoding=True`` AND first
          arg is a tensor)::

              out = model(atom_features, metal_ids, edge_index,
                          atom_mask=atom_mask, edge_attr=edge_attr)

          Skips RDKit entirely and feeds pre-featurized tensors straight
          into :class:`DirectedMPNN`.  When ``self.use_cached_encoding``
          is False the cached form is rejected — callers must use
          :meth:`forward_cached` explicitly or flip the flag.

        Args:
            smiles_or_atoms: list of SMILES strings OR a
                ``(B, N_max, atom_feat_dim)`` float tensor.
            metal_ids: ``(B,)`` long tensor with ids in ``[0, n_metals)``.
            edge_index: ``(B, 2, E_max)`` long tensor — required only for
                the cached path.
            atom_mask: optional ``(B, N_max)`` bool tensor — cached path.
            edge_attr: optional ``(B, E_max, edge_feat_dim)`` float
                tensor — cached path.

        Returns:
            dict mapping each metal name → ``(B, 2)`` binary logits,
            plus key ``"coord"`` → ``(B, n_coord_classes)`` logits.
        """
        # ----- Cached dispatch -----
        if isinstance(smiles_or_atoms, torch.Tensor):
            if not self.use_cached_encoding:
                raise RuntimeError(
                    "model(use_cached_encoding=False) was called with a "
                    "tensor first-arg. Either set use_cached_encoding=True "
                    "at construction or call forward_cached() directly."
                )
            if edge_index is None:
                raise ValueError(
                    "edge_index is required when calling forward() with "
                    "pre-featurized tensors."
                )
            return self.forward_cached(
                smiles_or_atoms,
                edge_index,
                metal_ids,
                atom_mask=atom_mask,
                edge_attr=edge_attr,
            )

        # ----- SMILES path (original V4 behaviour) -----
        smiles_list = smiles_or_atoms
        if metal_ids.dim() != 1:
            raise ValueError(
                f"metal_ids must be 1-D (B,), got shape {tuple(metal_ids.shape)}"
            )
        B = len(smiles_list)
        if metal_ids.size(0) != B:
            raise ValueError(
                f"len(smiles_list)={B} != metal_ids.size(0)={metal_ids.size(0)}"
            )

        h = self.encode(smiles_list)  # (B, hidden_dim)
        e = self.metal_embed(metal_ids.long())  # (B, metal_embed_dim)
        z = torch.cat([h, e], dim=-1)  # (B, hidden_dim + metal_embed_dim)

        out: Dict[str, torch.Tensor] = {
            m: head(z) for m, head in self.metal_heads.items()
        }
        out["coord"] = self.coord_head(z)
        return out


# ---------------------------------------------------------------------------
# Pretraining-augmented variant (D: 1:20 params:data ratio plan)
# ---------------------------------------------------------------------------
#: Default atom-vocabulary size — matches ``len(_ATOM_SYMBOLS)`` in
#: :mod:`molmetal.data.featurize` (C/N/O/F/P/S/Cl/Br/I/B/Si/Se/H/Other = 14).
ATOM_VOCAB_SIZE: int = 14


class V4MultiTaskDMPNNPretrain(V4MultiTaskDMPNN):
    """V4MultiTaskDMPNN + pretraining heads for DrugOOD-scale pretraining.

    Implements the architecture side of the 1:20 params:data ratio
    plan (Fix D). The base V4MultiTaskDMPNN has ~0.72M params and is
    trained on 2.67M examples (ratio 1:3.7 — overparameterized). The
    pretraining + fine-tune recipe uses 1.58M DrugOOD samples at the
    pretraining stage, lifting the effective params:data ratio above
    the 1:20 rule's 1:3.7 overparameterization threshold.

    Two pretraining heads are added:

    1. ``self.pretrain_head = nn.Linear(hidden_dim, atom_vocab_size)`` —
       per-atom masked-atom-prediction head (BERT/GraphMAE style).
       Operates on per-atom hidden states from the D-MPNN encoder.
    2. ``self.mol_clf_head = nn.Linear(hidden_dim, n_pretrain_tasks)`` —
       molecular-level multi-task classifier for general-purpose
       pretraining (e.g. DrugOOD downstream tasks).

    The pretrain heads are *added* parameters — the base encoder, metal
    embedding, and per-metal heads are reused unchanged. After pretraining
    on DrugOOD (1.58M rows, scaffold split, multi-task + masked atom),
    the pretrain heads are dropped and the base model is fine-tuned on
    the 5K MetalCytoToxDB subset for the per-metal cytotoxicity task.

    Forward behaviour:
        - **Eval mode OR** ``pretrain_mode=False``: returns exactly the
          same dict as :class:`V4MultiTaskDMPNN` (5 metal logits + coord).
          Backward compatibility is therefore preserved by construction.
        - **Train mode AND** ``pretrain_mode=True``: additionally emits
          ``"mol_clf"`` (B, n_pretrain_tasks) and ``"atom_logits"``
          (sum_N, atom_vocab_size) plus the ``"atom_mask"`` (B, N_max)
          tensor needed to align atom labels with per-atom predictions.
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        n_layers: int = 3,
        n_metals: int = 5,
        n_coord_classes: int = 5,
        metal_embed_dim: int = 64,
        metals: Optional[Tuple[str, ...]] = None,
        head_hidden_dim: int = 128,
        head_dropout: float = HEAD_DROPOUT,
        atom_vocab_size: int = ATOM_VOCAB_SIZE,
        n_pretrain_tasks: int = 2,
        use_cached_encoding: bool = False,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_metals=n_metals,
            n_coord_classes=n_coord_classes,
            metal_embed_dim=metal_embed_dim,
            metals=metals,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            use_cached_encoding=use_cached_encoding,
        )
        self.atom_vocab_size = int(atom_vocab_size)
        self.n_pretrain_tasks = int(n_pretrain_tasks)

        # ----- New pretraining heads -----
        # Atom-level masked-atom-prediction head: per-atom logits over the
        # atom-type vocabulary.  Output dim == atom_vocab_size (14 by
        # default; matches len(_ATOM_SYMBOLS) in GraphFeaturizer).
        self.pretrain_head = nn.Linear(self.hidden_dim, self.atom_vocab_size)
        # Molecular-level multi-task classifier: graph-level logits for
        # DrugOOD-style multi-task pretraining.  Output dim == n_pretrain_tasks.
        self.mol_clf_head = nn.Linear(self.hidden_dim, self.n_pretrain_tasks)

        # Pretrain-mode flag — toggle via ``set_pretrain_mode()``.
        # When False (the default), forward() behaves identically to
        # V4MultiTaskDMPNN, so existing fine-tune code paths still work.
        self.pretrain_mode = False

    # ------------------------------------------------------------------
    def set_pretrain_mode(self, on: bool = True) -> None:
        """Toggle pretrain-mode.

        When ``on=True`` AND the model is in ``.train()`` mode, the
        forward pass emits ``"mol_clf"`` and ``"atom_logits"`` in addition
        to the standard metal/coord logits. When ``on=False``, forward
        returns exactly the V4MultiTaskDMPNN output dict.
        """
        self.pretrain_mode = bool(on)

    # ------------------------------------------------------------------
    def encode_atoms(
        self,
        smiles_list: List[str],
        mol_objects: Optional[List] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode SMILES → concatenated per-atom hidden states + atom mask.

        Used by the masked-atom-prediction head. Reuses the encoder's
        featurisation + the D-MPNN ``forward_per_atom`` path so the
        pretrain head sees the same atom representations that the
        graph-level encoder would sum-pool.

        Args:
            smiles_list: list of ``B`` SMILES strings.
            mol_objects: optional parallel list of pre-parsed RDKit mols.

        Returns:
            ``(h_atoms, atom_mask)`` where:
              * ``h_atoms``: ``(sum_N, hidden_dim)`` — concatenated
                per-atom features across the batch (variable-length).
              * ``atom_mask``: ``(B, N_max)`` bool tensor — True for
                real atoms (matches ``DMPNNEncoder._featurize_batch``).
        """
        if not smiles_list:
            raise ValueError("smiles_list must be non-empty")
        device = next(self.parameters()).device
        h_atom_in, edge_index, edge_attr, batch_idx, atom_mask = (
            self.encoder._featurize_batch(smiles_list, device, mol_objects=mol_objects)
        )
        B, N_max, _ = h_atom_in.shape
        per_atom_list = []
        for b in range(B):
            n_atoms = int((batch_idx[b] == b).sum().item())
            if n_atoms == 0:
                continue
            h_mol = h_atom_in[b, :n_atoms]
            ei_mol = edge_index[b]
            ea_mol = edge_attr[b]
            mask_edges = (ei_mol[0] < n_atoms) & (ei_mol[1] < n_atoms)
            ei_local = ei_mol[:, mask_edges]
            ea_local = ea_mol[mask_edges]
            h_per_atom = self.encoder.dmpnn.forward_per_atom(h_mol, ei_local, ea_local)
            per_atom_list.append(h_per_atom)
        if not per_atom_list:
            return (
                torch.zeros(0, self.hidden_dim, device=device),
                atom_mask,
            )
        h_atoms = torch.cat(per_atom_list, dim=0)  # (sum_N, hidden_dim)
        return h_atoms, atom_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_list: List[str],
        metal_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass — backward compatible with V4MultiTaskDMPNN.

        Args:
            smiles_list: list of ``B`` SMILES strings.
            metal_ids: ``(B,)`` long tensor with ids in ``[0, n_metals)``.

        Returns:
            ``dict`` containing:
              * one entry per metal name → ``(B, 2)`` binary logits
                (always present — same as V4MultiTaskDMPNN).
              * ``"coord"`` → ``(B, n_coord_classes)`` logits (always).
              * ``"mol_clf"`` → ``(B, n_pretrain_tasks)`` logits
                (only when training AND ``pretrain_mode=True``).
              * ``"atom_logits"`` → ``(sum_N, atom_vocab_size)`` logits
                (only when training AND ``pretrain_mode=True``).
              * ``"atom_mask"`` → ``(B, N_max)`` bool
                (only when training AND ``pretrain_mode=True``).
        """
        # Base forward — exactly matches V4MultiTaskDMPNN.forward.
        out: Dict[str, torch.Tensor] = super().forward(smiles_list, metal_ids)
        # Pretrain heads fire only in train mode AND when pretrain_mode is on.
        # Default (eval / pretrain_mode=False) → no change to the dict.
        if self.training and self.pretrain_mode:
            h = self.encode(smiles_list)  # (B, hidden_dim)
            out["mol_clf"] = self.mol_clf_head(h)  # (B, n_pretrain_tasks)
            h_atoms, atom_mask = self.encode_atoms(smiles_list)
            out["atom_logits"] = self.pretrain_head(h_atoms)
            out["atom_mask"] = atom_mask
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class V4MultiTaskLoss(nn.Module):
    """Multi-task loss: per-metal CE + weighted coord-geometry CE.

    Matches the spec:

        L = Σ_m CE(logits[m], y_true[m]) + COORD_LOSS_WEIGHT · CE(coord, coord_true)

    The five metal heads share equal weight by default (1.0 each).  If
    a particular metal head sees many more / fewer training samples, the
    caller can pass per-metal weights via ``metal_weights``.

    If ``use_kendall_loss=True`` the manual weighted sum is replaced by
    :class:`UncertaintyWeightedMultiTaskLoss` (Kendall et al. 2018, CVPR,
    arXiv:1705.07115), which learns a per-task scalar weight (the
    homoscedastic noise precision ``1 / sigma^2``) jointly with the
    model parameters.  ``coord_weight`` is then ignored — the Kendall
    loss learns the weighting end-to-end.  See TODO-143 §2.1 for the
    Mol-Metal rationale and the verify gate ``log_sigmas.std() > 0.1
    after 50 epochs (no collapse)``.
    """

    def __init__(
        self,
        n_coord_classes: int = 5,
        coord_weight: float = COORD_LOSS_WEIGHT,
        metal_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.0,
        use_kendall_loss: bool = False,
        kendall_num_tasks: Optional[int] = None,
    ):
        super().__init__()
        self.n_coord_classes = int(n_coord_classes)
        self.coord_weight = float(coord_weight)
        self.metal_weights: Dict[str, float] = (
            dict(metal_weights) if metal_weights is not None else {}
        )
        self.use_kendall_loss = bool(use_kendall_loss)
        # Cross-entropy supports label_smoothing from PyTorch ≥ 1.10.
        self.ce_metal = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_coord = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Kendall UWMTL is lazily built in ``forward`` if ``num_tasks``
        # is not supplied up front — this lets the caller construct the
        # loss before knowing exactly which metals are in the batch.
        self._kendall: Optional[UncertaintyWeightedMultiTaskLoss] = None
        self._kendall_num_tasks = (
            int(kendall_num_tasks) if kendall_num_tasks is not None else None
        )

    def _get_kendall(self, num_tasks: int) -> UncertaintyWeightedMultiTaskLoss:
        """Return the cached Kendall loss, building it on first use.

        Re-using the same module across forward passes ensures the
        ``log_sigmas`` parameter is the one the optimizer is updating.
        """
        if self._kendall is None or self._kendall.num_tasks != num_tasks:
            self._kendall = UncertaintyWeightedMultiTaskLoss(num_tasks=num_tasks)
            # Move to the same device as the head logits if known.
            self._kendall = self._kendall.to(
                next(iter(self.ce_metal.parameters())).device
                if any(p.requires_grad for p in self.ce_metal.parameters())
                else "cpu"
            )
        return self._kendall

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_true: Dict[str, torch.Tensor],
        coord_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the multi-task loss.

        Args:
            out: model output dict (5 metal logits + ``"coord"``).
            y_true: dict mapping metal name → ``(B,)`` long binary labels.
                Missing metals contribute zero to the loss (useful for
                partial-metal batches).
            coord_true: ``(B,)`` long coordination-geometry labels.

        Returns:
            (total_loss, info) — info has per-metal detached scalars
            and the coord loss for logging.
        """
        info: Dict[str, torch.Tensor] = {}
        total = out["coord"].new_zeros(())
        per_task: List[torch.Tensor] = []

        for metal, logits in out.items():
            if metal == "coord":
                continue
            if metal not in y_true:
                # No labels for this metal — skip so partial batches work.
                info[f"loss_{metal}"] = torch.tensor(0.0, device=logits.device)
                if self.use_kendall_loss:
                    per_task.append(torch.tensor(0.0, device=logits.device))
                continue
            y = y_true[metal].long()
            if y.numel() == 0:
                info[f"loss_{metal}"] = torch.tensor(0.0, device=logits.device)
                if self.use_kendall_loss:
                    per_task.append(torch.tensor(0.0, device=logits.device))
                continue
            w = self.metal_weights.get(metal, 1.0)
            l = self.ce_metal(logits, y) * float(w)
            info[f"loss_{metal}"] = l.detach()
            total = total + l
            if self.use_kendall_loss:
                # Strip the manual metal_weight so Kendall learns the
                # weighting afresh — manual multipliers and learned
                # sigmas would be redundant and confusing.
                per_task.append(self.ce_metal(logits, y))

        # Coord auxiliary term — only if labels provided.
        if coord_true is not None and coord_true.numel() > 0:
            lcrd = self.ce_coord(out["coord"], coord_true.long())
            info["loss_coord"] = lcrd.detach()
            if self.use_kendall_loss:
                per_task.append(lcrd)
            else:
                total = total + self.coord_weight * lcrd
        else:
            info["loss_coord"] = torch.tensor(0.0, device=total.device)
            if self.use_kendall_loss:
                per_task.append(torch.tensor(0.0, device=total.device))

        # Kendall aggregation (per-task list -> scalar with learnable weights).
        if self.use_kendall_loss:
            kendall = self._get_kendall(len(per_task))
            total = kendall(per_task)
            # Expose current sigmas for logging / verify-gate.
            info["kendall_sigmas"] = kendall.get_sigmas()
            info["kendall_log_sigmas"] = kendall.log_sigmas.detach()

        info["loss_total"] = total.detach()
        return total, info


# ---------------------------------------------------------------------------
# Smoke helper (used by tests + Stage-1 dev runs)
# ---------------------------------------------------------------------------
def quick_smoke_v4mt(
    smiles: Optional[List[str]] = None,
    n_metals: int = 5,
    n_coord_classes: int = 5,
    hidden_dim: int = 64,  # smaller for tests / smoke runs
    n_layers: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[
    V4MultiTaskDMPNN, V4MultiTaskLoss, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]:
    """Build a model + loss + fake batch for a quick smoke test.

    Returns ``(model, loss_fn, out, y_true_dict)`` so callers can run
    ``loss, info = loss_fn(out, y_true, coord_true); loss.backward()``
    immediately.  Used by the test suite and the Stage-1 dev script.
    """
    if smiles is None:
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"]
    B = len(smiles)
    device = device or torch.device("cpu")
    model = V4MultiTaskDMPNN(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
    ).to(device)
    loss_fn = V4MultiTaskLoss(n_coord_classes=n_coord_classes)
    metal_ids = torch.zeros(B, dtype=torch.long, device=device)
    out = model(smiles, metal_ids)
    # Fake labels — one binary label per metal + coord
    torch.manual_seed(0)
    y_true = {m: torch.randint(0, 2, (B,), device=device) for m in model.metals}
    coord_true = torch.randint(0, n_coord_classes, (B,), device=device)
    return model, loss_fn, out, {"y_true": y_true, "coord_true": coord_true}


__all__ = [
    "ATOM_VOCAB_SIZE",
    "COORD_GEOMETRY_NAMES",
    "COORD_LOSS_WEIGHT",
    "DMPNNEncoder",
    "HEAD_DROPOUT",
    "SUPPORTED_METALS",
    "UncertaintyWeightedMultiTaskLoss",
    "V4MultiTaskDMPNN",
    "V4MultiTaskDMPNNPretrain",
    "V4MultiTaskLoss",
    "quick_smoke_v4mt",
]
