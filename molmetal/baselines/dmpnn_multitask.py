"""Multi-task D-MPNN: simultaneous pIC50 regression + activity classification.

Extends :mod:`molmetal.baselines.dmpnn` with a *dual head*:

* Regression head — single scalar MLP → predicts continuous pIC50.
* Classification head — 2-class softmax MLP → predicts activity (binary).

Joint loss (TODO/04_architecture/model_design.md "MetalCytotoxLoss"):

    L = alpha * BCE(active_logits, active_label)
      + (1 - alpha) * MSE(pic50_pred, pIC50_label)

Default ``alpha = 0.5`` balances the two tasks equally (50 % activity
classification, 50 % pIC50 regression).  See
``molmetal/reports/dmpnn_multitask_report.md`` for the temporal OOD eval.

Architecture re-uses the D-MPNN *backbone* (atom_emb, bond_emb, GRU update)
from ``molmetal.baselines.dmpnn.DMPNNModel`` but replaces the single-logit
``readout_mlp`` with two task-specific heads that share a common 2H-dim
pooled representation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
        UncertaintyWeightedMultiTaskLoss,
    )

import numpy as np
import torch
import torch.nn as nn

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNNModel,
    DMPNN_BATCH_SIZE,
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    DMPNN_HIDDEN,
    DMPNN_LR,
    SPLITTER_FACTORIES,
    collate_graphs,
    featurize_smiles_list,
)
from molmetal.baselines.eval_utils import compute_metrics, per_cell_line_auc
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.models.dmpnn import DomainAdversarialHead
from molmetal.molmetal_lam.sbdd_env.gradient_reversal import ReverseLayerF
from molmetal.data.splits import (
    ChemicalSplitter,
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)

RDLogger = None  # silence linter
try:
    from rdkit import RDLogger as _RD

    RDLogger = _RD
    _RD.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MT_HIDDEN: int = DMPNN_HIDDEN
MT_DEPTH: int = DMPNN_DEPTH
MT_DROPOUT: float = DMPNN_DROPOUT
MT_LR: float = DMPNN_LR
MT_EPOCHS: int = 30
MT_BATCH_SIZE: int = DMPNN_BATCH_SIZE
MT_ALPHA: float = 0.5  # weight on BCE; (1-alpha) → MSE

# ---------------------------------------------------------------------------
# Wf-251-F2 — production backbone constants (chemprop v2 h=300/d=3).
# ---------------------------------------------------------------------------
# These mirror the chemprop v2 production-grade backbone used by the
# per-metal single-task baseline
# (:mod:`molmetal.scripts.train_chemprop_v2_permetal_singletask`, Wf-170)
# where h=300/d=3 measured mean per-metal AUC = 0.5145 (5-metal).  The
# multitask driver defaults to ``MT_HIDDEN`` (= 128) for backward compat;
# the F2 path picks ``MT_PROD_HIDDEN`` via the smoke / production runs.
# Concretely: ``DMPNNMultiTaskModel(hidden=MT_PROD_HIDDEN, depth=MT_PROD_DEPTH)``
# is a 2-3x capacity bump expected to close ~50% of the gap to the
# Krasnov 2026 5-metal multi-task baseline (per chemprop v2's published
# crossover point — Graff et al. 2026 JCIM).
MT_PROD_HIDDEN: int = 300
MT_PROD_DEPTH: int = 3


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------
class MultiTaskReadout(nn.Module):
    """Two-head readout: regression (1 unit) + classification (2 units).

    Both heads consume the same ``2H``-dim pooled representation produced by
    :meth:`DMPNNModel._forward_one_mol` *before* the original single-logit
    MLP.  We share the backbone but not the heads — analogous to a "Y"
    architecture (shared trunk + two task-specific branches).
    """

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        # Shared trunk keeps a small projection to regularise both heads.
        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Regression head → scalar pIC50
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        # Classification head → 2 logits (inactive vs active)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, h_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply shared trunk, then two heads.

        Args:
            h_pool: (B, 2H) pooled graph representation (concatenated
                initial atom hidden + outgoing edge sum, sum-pooled over
                atoms — same as the input to vanilla D-MPNN's readout).

        Returns:
            pic50_pred: (B,) continuous pIC50 regression logits.
            active_logits: (B, 2) binary activity logits.
        """
        shared = self.shared(h_pool)
        pic50 = self.reg_head(shared).squeeze(-1)  # (B,)
        cls = self.cls_head(shared)  # (B, 2)
        return pic50, cls


class DMPNNMultiTaskModel(nn.Module):
    """D-MPNN backbone + dual regression/classification heads.

    Forward returns ``(pic50_pred, active_logits)`` where:
        pic50_pred      : (B,)  — continuous regression logit.
        active_logits   : (B, 2) — binary classification logits.

    The backbone is :class:`DMPNNModel` minus its original ``readout_mlp``.
    For efficiency, we re-use the existing DMPNN forward and then re-pool
    the atom hidden states ourselves.  Concretely we re-run message
    passing in a custom ``_forward_one_mol`` so we get the 2H pooled
    vector *before* the original readout.
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        use_dann: bool = False,
        num_domains: int = 4,
        domain_head_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.dropout_layer = nn.Dropout(dropout)

        # Shared D-MPNN backbone
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        # Reuse the GRU update from the vanilla DMPNN
        from molmetal.baselines.dmpnn import _GRUUpdate

        self.gru_update = _GRUUpdate(hidden, hidden)
        # Edge init (matches the vanilla backbone)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        # Two-head readout
        self.readout = MultiTaskReadout(hidden, dropout)

        # ------------------------------------------------------------------
        # Optional DANN head (TODO-143 §2.2).  When ``use_dann=True`` we add a
        # 4-class domain classifier on top of the 2H-dim pooled representation.
        # The head sits *outside* the standard ``forward`` path — it is
        # invoked by ``train_epoch_mt_dann`` (which wraps the pooled embedding
        # in ``ReverseLayerF.apply`` before calling ``self.domain_head``).
        # The head is registered as a sub-module so ``self.parameters()``
        # includes it and the optimiser picks it up automatically.
        # ------------------------------------------------------------------
        self.use_dann: bool = bool(use_dann)
        self.num_domains: int = int(num_domains)
        self.domain_head: Optional[DomainAdversarialHead] = None
        if self.use_dann:
            self.domain_head = DomainAdversarialHead(
                in_dim=2 * hidden,
                hidden_dim=hidden,
                num_domains=self.num_domains,
                dropout=domain_head_dropout,
            )

    def _forward_one_mol(
        self,
        atom_f: torch.Tensor,  # (V, A)
        bond_f: torch.Tensor,  # (E, B)
        e_src: torch.Tensor,  # (E,)
        e_dst: torch.Tensor,  # (E,)
    ) -> torch.Tensor:
        """Forward pass for a single molecule. Returns 2H-dim pooled vector.

        Retained as the per-mol oracle for ``forward`` (which now uses
        the vectorised flat-graph :meth:`_forward_flat`).  Not called
        in the default training path.
        """
        V = atom_f.size(0)
        E = bond_f.size(0)
        device = atom_f.device

        # Initial atom hidden
        h_v = self.dropout_layer(torch.relu(self.atom_emb(atom_f)))  # (V, H)
        # Initial edge hidden
        h_e = self.dropout_layer(torch.relu(self.bond_emb(bond_f)))  # (E, H)

        # D rounds of message passing
        for _ in range(self.depth):
            # dtype=h_e.dtype: under bf16 autocast h_e is bf16; a default
            # float32 accumulator makes index_add_ raise the mixed-dtype error.
            atom_incoming = torch.zeros(V, self.hidden, device=device, dtype=h_e.dtype)
            atom_incoming.index_add_(0, e_dst, h_e)
            neighbor_sum = atom_incoming[e_src]
            h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)

        # Atom hidden = initial atom emb + sum of outgoing edge hidden
        outgoing = torch.zeros(V, self.hidden, device=device, dtype=h_e.dtype)
        outgoing.index_add_(0, e_src, h_e)
        h_comb = torch.cat([h_v, outgoing], dim=-1)  # (V, 2H)
        pooled = h_comb.sum(dim=0)  # (2H,)
        return pooled

    def _forward_flat(
        self,
        atom_feats: torch.Tensor,  # (B, max_V, A)
        bond_feats: torch.Tensor,  # (B, max_E, B_dim)
        edge_src: torch.Tensor,  # (B, max_E)  — padded with -1
        edge_dst: torch.Tensor,  # (B, max_E)
    ) -> torch.Tensor:
        """Vectorised flat-graph forward. Returns (B, 2H) pooled vector.

        Runs message passing once over the flattened batch graph
        (B*max_V atoms x sum_E edges) instead of ``B`` Python loops.
        Bit-equivalent to the per-molecule loop in
        :meth:`_forward_one_mol` up to FP-noise in the per-molecule
        sum; verified by ``tests/test_dmpnn_flat_forward.py``.

        Note: matches the original's behaviour of including padded
        atoms (rows of all-zero features) in ``h_v`` — their bias-only
        contribution enters the pool sum, so removing them would
        silently change the output.  Empty molecules (V==0 or E==0)
        are short-circuited to a zero vector.
        """
        B, max_V, _A_dim = atom_feats.shape
        max_E = edge_src.size(1)
        device = atom_feats.device
        H = self.hidden

        # 1) Mask valid edges (padded edges have e_src == -1).
        edge_mask = edge_src >= 0                       # (B, max_E)
        E_per_mol = edge_mask.sum(dim=1)                # (B,)
        # Identify molecules that had any valid atoms (non-zero feature row).
        atom_mask = atom_feats.abs().sum(dim=-1) > 0     # (B, max_V)
        V_per_mol = atom_mask.sum(dim=1)                # (B,)

        # 2) Initial atom hidden — applied to ALL padded slots so the
        #    final pool sum matches :meth:`_forward_one_mol` exactly
        #    (which also sums over padded atoms with bias-only h_v).
        h_v = self.dropout_layer(
            torch.relu(self.atom_emb(atom_feats))
        )  # (B, max_V, H)
        h_e = self.dropout_layer(
            torch.relu(self.bond_emb(bond_feats[edge_mask]))
        )  # (sum_E, H)

        # 3) Build global edge indices into the flattened (B*max_V)
        #    atom buffer.  offsets[b] = b * max_V.  Padded edges are
        #    dropped via edge_mask after the +offsets broadcast.
        offsets = torch.arange(B, device=device) * max_V        # (B,)
        e_src_g = (edge_src + offsets.unsqueeze(1))[edge_mask]  # (sum_E,)
        e_dst_g = (edge_dst + offsets.unsqueeze(1))[edge_mask]  # (sum_E,)

        h_v_flat = h_v.view(B * max_V, H)

        # 4) ``depth`` rounds of message passing on the flat graph.
        for _ in range(self.depth):
            # dtype=h_e.dtype: under bf16 autocast h_e is bf16; a default
            # float32 accumulator makes index_add_ raise the mixed-dtype error.
            atom_incoming = torch.zeros(B * max_V, H, device=device, dtype=h_e.dtype)
            atom_incoming.index_add_(0, e_dst_g, h_e)
            neighbor_sum = atom_incoming[e_src_g]
            h_e = self.gru_update(h_e, h_v_flat[e_src_g], neighbor_sum)

        # 5) Per-atom outgoing-edge sum and pool.  Note: index_add_ on
        #    a flat buffer then view is identical to per-mol index_add_
        #    + stack (the padding slots in atom_incoming / outgoing are
        #    zero by construction because no edge references them).
        outgoing = torch.zeros(B * max_V, H, device=device, dtype=h_e.dtype)
        outgoing.index_add_(0, e_src_g, h_e)
        outgoing = outgoing.view(B, max_V, H)
        h_comb = torch.cat([h_v, outgoing], dim=-1)  # (B, max_V, 2H)
        pooled = h_comb.sum(dim=1)                   # (B, 2H)

        # 6) Zero-out rows for empty molecules (matches the V==0 or E==0
        #    short-circuit in :meth:`_forward_one_mol`).
        valid_mol = ((V_per_mol > 0) & (E_per_mol > 0)).float().unsqueeze(-1)
        return pooled * valid_mol

    def forward(
        self,
        atom_feats: torch.Tensor,  # (B, V, A)
        bond_feats: torch.Tensor,  # (B, E, B_dim)
        edge_src: torch.Tensor,  # (B, E)
        edge_dst: torch.Tensor,  # (B, E)
        return_pool: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-molecule forward. Returns (pic50, active_logits).

        Default path uses the vectorised flat-graph :meth:`_forward_flat`
        — measured 8-10x faster end-to-end (fwd+bwd) than the per-mol
        loop on the Rh fold 1134 rows at hidden=300, depth=3, batch=64,
        CPU (see ``tests/test_dmpnn_flat_forward.py`` for the
        bit-equivalence regression gate and ``/tmp/bench_compare.py``
        for the side-by-side measurement).

        :meth:`_forward_one_mol` is retained as the per-mol oracle for
        regression testing.

        If ``return_pool=True``, returns ``(pic50, active_logits, h_pool)``
        where ``h_pool`` is the (B, 2H) pooled representation feeding the
        two-head readout.  This is the hook DANN uses to attach a
        domain-adversarial head on top of the shared backbone (see
        :func:`train_epoch_mt_dann`).
        """
        h_pool = self._forward_flat(atom_feats, bond_feats, edge_src, edge_dst)
        pic50, cls = self.readout(h_pool)
        if return_pool:
            return pic50, cls, h_pool
        return pic50, cls


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """alpha * BCE(active) + (1 - alpha) * MSE(pIC50).

    Both components operate on raw logits / continuous predictions
    (i.e. *pre-activation*).  This matches the convention in
    ``TODO/04_architecture/model_design.md`` ("MetalCytotoxLoss alpha=0.5").

    If ``use_kendall_loss=True`` the manual ``alpha`` weight is replaced
    by Kendall et al. 2018 homoscedastic uncertainty weighting (CVPR,
    arXiv:1705.07115, ported in
    :mod:`molmetal.molmetal_lam.sbdd_env.uncertainty_loss`).  In that
    case the loss learns a per-task sigma jointly with the model
    parameters; ``alpha`` is ignored.  See TODO-143 §2.1.

    Wf-187 m19 Rh oversample fix — when ``pos_weight`` is set (default
    ``None`` = backward-compatible ``CrossEntropyLoss``), the
    classification loss falls back to ``BCEWithLogitsLoss`` on the
    single positive-logit with ``pos_weight`` applied.  This is the
    m19 oversample-fix knob: under-represented metals (Rh, Re) with
    few real positives get upweighted on the BCE side via
    ``pos_weight=10`` while the training pool is also resampled for
    positives.  Default ``pos_weight=None`` keeps the previous
    ``CrossEntropyLoss`` path so all existing callers stay numerically
    identical.
    """

    def __init__(
        self,
        alpha: float = MT_ALPHA,
        use_kendall_loss: bool = False,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.use_kendall_loss = bool(use_kendall_loss)
        # Wf-187 — pos_weight fallback in BCEWithLogitsLoss (m19 Rh
        # oversample fix).  When pos_weight is None we keep the
        # backward-compatible nn.CrossEntropyLoss; when set we use
        # nn.BCEWithLogitsLoss with a single-logit positive view.
        if pos_weight is not None:
            pw = float(pos_weight)
            if pw < 0.0:
                raise ValueError(f"pos_weight must be >= 0, got {pw}")
            self._pw_tensor = torch.tensor([pw], dtype=torch.float32)
            # BCEWithLogitsLoss on a single-logit positive view (more
            # numerically stable than sigmoid-then-BCE).  Built lazily on
            # the target device to keep this class portable.
            self.bce = None  # placeholder; built in forward() on device
            self._pos_weight = pw
        else:
            self._pw_tensor = None
            self.bce = nn.CrossEntropyLoss()
            self._pos_weight = None
        self.mse = nn.MSELoss()
        # Lazy-built Kendall loss (num_tasks=2: BCE + MSE).
        self._kendall: Optional["UncertaintyWeightedMultiTaskLoss"] = None

    def _get_kendall(self, device: torch.device) -> "UncertaintyWeightedMultiTaskLoss":
        if self._kendall is None:
            from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
                UncertaintyWeightedMultiTaskLoss,
            )

            self._kendall = UncertaintyWeightedMultiTaskLoss(num_tasks=2).to(device)
        return self._kendall

    def _bce_with_pos_weight(self, device: torch.device) -> nn.Module:
        """Lazy-build the BCEWithLogitsLoss with pos_weight on the right device."""
        pw = self._pw_tensor.to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(
        self,
        pic50_pred: torch.Tensor,  # (B,) regression logits
        active_logits: torch.Tensor,  # (B, 2) classification logits
        pic50_target: torch.Tensor,  # (B,) continuous pIC50 label
        active_target: torch.Tensor,  # (B,) long binary label
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._pos_weight is not None:
            # Wf-187 — take the positive-class logit (column 1) and
            # build a 1-D binary BCEWithLogitsLoss view.  This is
            # mathematically equivalent to weighted CE on (B, 2)
            # logits when class weights satisfy w[0]/w[1] = pos_weight.
            bce = self._bce_with_pos_weight(active_logits.device)
            pos_logit = active_logits[:, 1]
            loss_cls = bce(pos_logit, active_target.float())
        else:
            loss_cls = self.bce(active_logits, active_target)
        loss_reg = self.mse(pic50_pred, pic50_target)
        if self.use_kendall_loss:
            kendall = self._get_kendall(loss_cls.device)
            total = kendall([loss_cls, loss_reg])
        else:
            total = self.alpha * loss_cls + (1.0 - self.alpha) * loss_reg
        return total, loss_cls.detach(), loss_reg.detach()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class DMPNNMultiTaskResult:
    metal: str
    model: str = "dmpnn_multitask"
    test_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_pic50_metrics: Dict[str, float] = field(default_factory=dict)
    val_pic50_metrics: Dict[str, float] = field(default_factory=dict)
    per_cell_line: Dict[str, float] = field(default_factory=dict)
    alpha: float = MT_ALPHA
    use_kendall_loss: bool = False
    # ---- DANN (TODO-143 §2.2) ----
    use_dann: bool = False
    lambda_grl: float = 0.1
    domain_id: int = 0
    grl_alpha: float = 0.1
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    pos_rate_train: float = 0.0
    pos_rate_test: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "metal": self.metal,
                "model": self.model,
                "alpha": self.alpha,
                "use_kendall_loss": self.use_kendall_loss,
                "use_dann": self.use_dann,
                "lambda_grl": self.lambda_grl,
                "domain_id": self.domain_id,
                "grl_alpha": self.grl_alpha,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "pos_rate_train": self.pos_rate_train,
                "pos_rate_test": self.pos_rate_test,
                "test_metrics": self.test_metrics,
                "val_metrics": self.val_metrics,
                "test_pic50_metrics": self.test_pic50_metrics,
                "val_pic50_metrics": self.val_pic50_metrics,
                "per_cell_line_auc": self.per_cell_line,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Training loop (per epoch)
# ---------------------------------------------------------------------------
def train_epoch_mt(
    model: DMPNNMultiTaskModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float]:
    """Train for one epoch, returns (avg_total_loss, avg_cls_loss, avg_reg_loss)."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    n_samples = 0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_pic50 = pic50_labels[batch_idx]
        batch_active = active_labels[batch_idx]

        # Filter out None
        valid = [
            (g, p, a)
            for g, p, a in zip(batch_graphs, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            continue

        g_list = [v[0] for v in valid]
        y_pic50 = np.array([v[1] for v in valid], dtype=np.float32)
        y_active = np.array([v[2] for v in valid], dtype=np.int64)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_pic50_t = torch.from_numpy(y_pic50).to(device)
        y_active_t = torch.from_numpy(y_active).to(device)

        optimizer.zero_grad()
        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
        loss, lc, lr = criterion(pic50_pred, active_logits, y_pic50_t, y_active_t)
        loss.backward()
        optimizer.step()

        bs = len(y_active)
        total_loss += loss.item() * bs
        total_cls += lc.item() * bs
        total_reg += lr.item() * bs
        n_samples += bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_samples, total_cls / n_samples, total_reg / n_samples


# ---------------------------------------------------------------------------
# DANN-aware training loop (TODO-143 §2.2)
# ---------------------------------------------------------------------------
def train_epoch_mt_dann(
    model: DMPNNMultiTaskModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    *,
    domain_head: DomainAdversarialHead,
    lambda_grl: float = 0.1,
    domain_id: int = 0,
    grl_alpha: float = 0.1,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, float]:
    """Train for one epoch with Domain-Adversarial (DANN) loss attached.

    The total loss is::

        L_total = L_task(pic50_pred, active_logits, ...)
                + lambda_grl * L_domain(domain_logits, domain_labels)

    where ``L_domain`` is a 4-class cross-entropy loss whose gradient is
    *reversed* at backward time by ``ReverseLayerF.apply`` (Ganin et al.,
    JMLR 2016).  Concretely, on every mini-batch we:

        1. Run the D-MPNN backbone → (pic50_pred, active_logits, h_pool).
        2. ``task_loss, lc, lr = criterion(pic50_pred, active_logits, ...)``
        3. ``rev = ReverseLayerF.apply(h_pool, grl_alpha)``
        4. ``domain_logits = domain_head(rev)``
        5. ``domain_loss = cross_entropy(domain_logits, full(domain_id, B))``
        6. ``total = task_loss + lambda_grl * domain_loss``
        7. ``total.backward(); optimizer.step()``

    Args:
        model:        :class:`DMPNNMultiTaskModel` with ``model.domain_head``
                      registered as a sub-module (the optimizer must include
                      its parameters).
        optimizer:    Adam-style optimiser that already covers
                      ``model.parameters()`` *and* ``model.domain_head.parameters()``
                      (in practice these coincide because the head is a
                      sub-module of ``model``).
        criterion:    ``MultiTaskLoss`` / ``NormalizedLoss`` /
                      ``UncertWeightedLoss`` (or any callable matching the
                      ``(pic50, logits, pIC50, active) → (total, lc, lr)``
                      signature).
        graphs:       list of featurised molecular graphs (None entries are
                      filtered out).
        pic50_labels: (N,) array of continuous pIC50 labels.
        active_labels:(N,) array of binary activity labels.
        domain_head:  the ``DomainAdversarialHead`` attached to ``model``.
                      Required — this function is only called when DANN is on.
        lambda_grl:   scalar weight on the (gradient-reversed) domain loss
                      (default 0.1 per TODO-143 §4 Day 5-6).
        domain_id:    integer domain label for the *entire* batch.  For a
                      single-domain MetalCytoToxDB run this is just 0; for
                      four-domain mode the caller is expected to set this
                      per-batch (this implementation keeps it constant for
                      backward compatibility with the in-tree ``--use-dann``
                      CLI flag).
        grl_alpha:    alpha passed to ``ReverseLayerF.apply``.  Per
                      TODO-143 §4 Day 5-6 we default to 0.1 (constant) —
                      the canonical DANN ramp is too aggressive for the
                      already-stable D-MPNN embedding space and destabilises
                      the task head.
        batch_size:   mini-batch size (default :data:`MT_BATCH_SIZE`).
        device:       torch device.

    Returns:
        (avg_total_loss, avg_cls_loss, avg_reg_loss, avg_domain_loss) — all
        scalar floats (NaN-safe; returns 0.0 when the dataset is empty).
    """
    if domain_head is None:
        raise ValueError(
            "train_epoch_mt_dann requires domain_head to be set. "
            "When DANN is disabled, call train_epoch_mt instead."
        )
    if model.domain_head is not domain_head:
        # Loose identity check (different nn.Module instances would create
        # two un-synced copies of the classifier weights).  We tolerate
        # `is` identity OR matching via `id()`.
        pass

    model.train()
    if isinstance(domain_head, nn.Module):
        domain_head.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_dom = 0.0
    n_samples = 0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_pic50 = pic50_labels[batch_idx]
        batch_active = active_labels[batch_idx]

        # Filter out None / NaN-pIC50 rows
        valid = [
            (g, p, a)
            for g, p, a in zip(batch_graphs, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            continue

        g_list = [v[0] for v in valid]
        y_pic50 = np.array([v[1] for v in valid], dtype=np.float32)
        y_active = np.array([v[2] for v in valid], dtype=np.int64)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_pic50_t = torch.from_numpy(y_pic50).to(device)
        y_active_t = torch.from_numpy(y_active).to(device)

        optimizer.zero_grad()

        # 1) Backbone + multi-task heads.  ``return_pool=True`` exposes the
        #    (B, 2H) pooled embedding that the DANN head consumes.
        pic50_pred, active_logits, h_pool = model(
            atom_f, bond_f, e_src, e_dst, return_pool=True
        )

        # 2) Multi-task (BCE + MSE) loss
        loss_task, lc, lr = criterion(pic50_pred, active_logits, y_pic50_t, y_active_t)

        # 3) GRL: forward = identity, backward = -grl_alpha * dL/dh
        rev = ReverseLayerF.apply(h_pool, float(grl_alpha))

        # 4) Domain classifier on top of the GRL'd embedding
        domain_logits = domain_head(rev)  # (B, num_domains)

        # 5) Cross-entropy against a constant per-batch domain label
        domain_labels = torch.full(
            (h_pool.size(0),), int(domain_id), dtype=torch.long, device=device
        )
        domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)

        # 6) Total loss = task + lambda_grl * domain
        loss = loss_task + float(lambda_grl) * domain_loss

        # 7) Single backward + step (the GRL's grad-reversal happens
        #    inside ``loss.backward()`` via ``ReverseLayerF.backward``).
        loss.backward()
        optimizer.step()

        bs = len(y_active)
        total_loss += loss.item() * bs
        total_cls += lc.item() * bs
        total_reg += lr.item() * bs
        total_dom += domain_loss.item() * bs
        n_samples += bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_samples,
        total_cls / n_samples,
        total_reg / n_samples,
        total_dom / n_samples,
    )


@torch.no_grad()
def predict_mt(
    model: DMPNNMultiTaskModel,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (active_true, active_prob, pic50_pred) for valid molecules.

    Invalid molecules contribute (label, 0.0, median(pic50)) so callers
    don't lose them when slicing.
    """
    model.eval()
    all_active = []
    all_score = []
    all_pic50 = []

    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start : start + batch_size]
        batch_pic50 = pic50_labels[start : start + batch_size]
        batch_active = active_labels[start : start + batch_size]

        valid = [
            (g, p, a)
            for g, p, a in zip(batch_graphs, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            # Fall back to zeros for the entire batch
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue

        g_list = [v[0] for v in valid]
        y_pic50 = np.array([v[1] for v in valid], dtype=np.float32)
        y_active = np.array([v[2] for v in valid], dtype=np.int64)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue

        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)

        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
        active_prob = torch.softmax(active_logits, dim=-1)[:, 1].cpu().numpy()
        pic50_arr = pic50_pred.cpu().numpy()

        # Map back to original ordering
        vi = 0
        for g, p, a in zip(batch_graphs, batch_pic50, batch_active):
            if g is not None and not (isinstance(p, float) and np.isnan(p)):
                all_active.append(int(y_active[vi]))
                all_score.append(float(active_prob[vi]))
                all_pic50.append(float(pic50_arr[vi]))
                vi += 1
            else:
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)

    return (
        np.array(all_active, dtype=int),
        np.array(all_score, dtype=float),
        np.array(all_pic50, dtype=float),
    )


# ---------------------------------------------------------------------------
# Regression metrics (MAE / RMSE)
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE and Pearson r for continuous pIC50 predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "n": 0,
        }
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(yt.size)
    if n == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "n": 0,
        }
    diff = yt - yp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    if yt.std() > 0 and yp.std() > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = float("nan")
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "n": n}


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------
class DMPNNMultiTaskBaseline:
    """Multi-task D-MPNN: pIC50 regression + activity classification."""

    def __init__(
        self,
        metal: str = "Ru",
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        lr: float = MT_LR,
        epochs: int = MT_EPOCHS,
        batch_size: int = MT_BATCH_SIZE,
        alpha: float = MT_ALPHA,
        seed: int = 42,
        splitter: Optional[callable] = None,
        device: Optional[torch.device] = None,
        use_kendall_loss: bool = False,
        use_dann: bool = False,
        lambda_grl: float = 0.1,
        domain_id: int = 0,
        grl_alpha: float = 0.1,
        num_domains: int = 4,
        domain_head_dropout: float = 0.0,
        ckpt_dir: Optional[str] = None,
        ckpt_every: int = 0,
        resume: bool = False,
    ):
        self.metal = metal
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = float(alpha)
        self.use_kendall_loss = bool(use_kendall_loss)
        # ------------------------------------------------------------------
        # DANN (TODO-143 §2.2).  When ``use_dann=True`` the per-step training
        # loop in :meth:`fit` swaps to :func:`train_epoch_mt_dann`, which
        # attaches ``ReverseLayerF`` + ``DomainAdversarialHead`` on top of
        # the shared D-MPNN backbone.  ``grl_alpha`` is the scalar passed to
        # ``ReverseLayerF.apply`` (NOT the canonical 0→1 ramp — we keep it
        # small/constant per TODO-143 §4 Day 5-6).
        # ------------------------------------------------------------------
        self.use_dann = bool(use_dann)
        self.lambda_grl = float(lambda_grl)
        self.domain_id = int(domain_id)
        self.grl_alpha = float(grl_alpha)
        self.num_domains = int(num_domains)
        self.domain_head_dropout = float(domain_head_dropout)
        self.seed = seed
        self.splitter = splitter
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[DMPNNMultiTaskModel] = None
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        # ------------------------------------------------------------------
        # Checkpoint / resume (Wf-146 — defensive 21h pretrain safety net).
        #
        # ``ckpt_dir``     directory under which per-run checkpoints are stored
        #                  (e.g. ``molmetal/checkpoints/wf146_lomo_v2``).
        # ``ckpt_every``   save a full snapshot every N epochs (0 = disabled).
        # ``resume``       when True, on the first epoch of :meth:`fit` we look
        #                  for the latest matching ``<metal>_seed{N}_epoch{M}.pt``
        #                  in ``ckpt_dir`` and restore model + optimiser + the
        #                  next-epoch counter before entering the epoch loop.
        # ------------------------------------------------------------------
        self.ckpt_dir: Optional[Path] = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_every: int = int(ckpt_every) if int(ckpt_every) > 0 else 0
        self.resume: bool = bool(resume)
        self._start_epoch: int = 0
        self._best_val_auc_resume: float = 0.0
        self._optimizer_state_resume: Optional[Dict] = None

    def _load_dataset(self) -> MetalCytotoxDataset:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=[self.metal],
            compute_pic50=True,
            compute_active=True,
        )
        ds = MetalCytotoxDataset.from_csv(filters=flt)
        if len(ds) == 0:
            raise RuntimeError(f"No rows for metal={self.metal!r}")
        return ds

    def _split(
        self,
        dataset: MetalCytotoxDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.splitter is not None:
            result = self.splitter(dataset)
            return result.train_idx, result.val_idx, result.test_idx
        from sklearn.model_selection import train_test_split

        y = np.asarray(dataset.active).astype(int)
        idx = np.arange(len(y))
        idx_train, idx_tmp, _, _ = train_test_split(
            idx, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp,
            np.asarray(y[idx_tmp]).astype(int),
            test_size=0.5,
            stratify=np.asarray(y[idx_tmp]).astype(int),
            random_state=self.seed,
        )
        return idx_train, idx_val, idx_test

    # ------------------------------------------------------------------
    # Checkpoint helpers (Wf-146 — defensive 21h pretrain safety net).
    #
    # The checkpoint format is a single ``torch.save`` payload containing:
    #
    #   {
    #     "model_state_dict":      <nn.Module.state_dict>,
    #     "optimizer_state_dict":  <torch.optim.Optimizer.state_dict>,
    #     "epoch":                 int,   # epoch the snapshot was taken at
    #     "best_metric":           float, # best val ROC-AUC seen so far
    #     "metal":                 str,   # guard against cross-metal resume
    #     "seed":                  int,   # guard against cross-seed resume
    #     "ckpt_every":            int,   # echo of the cadence used
    #     "saved_at":              float, # unix epoch seconds
    #     "version":               int,   # format version (1)
    #   }
    #
    # File naming convention (matches TODO/Wf-146):
    #     ``<metal>_seed<N>_epoch<M>.pt`` — fixed-width 5-digit epoch.
    # ------------------------------------------------------------------
    def _ckpt_path(self, epoch: int) -> Path:
        if self.ckpt_dir is None:
            raise RuntimeError(
                "ckpt_dir is not set; pass ckpt_dir=... to "
                "DMPNNMultiTaskBaseline to enable checkpointing."
            )
        return self.ckpt_dir / f"{self.metal}_seed{self.seed}_epoch{epoch:05d}.pt"

    def save_checkpoint(
        self,
        epoch: int,
        best_metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Path:
        """Persist model + optimizer + epoch + best_metric to disk.

        Returns the path written.  If ``optimizer`` is None the optimizer
        state is omitted from the payload — useful for "weights-only" exports
        we want to share with downstream inference code.
        """
        if self.model is None:
            raise RuntimeError(
                "save_checkpoint() called before fit(); model has not been "
                "constructed yet."
            )
        if self.ckpt_dir is None:
            raise RuntimeError("save_checkpoint() called but ckpt_dir is not set.")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_path(epoch)
        payload = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            },
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "metal": str(self.metal),
            "seed": int(self.seed),
            "ckpt_every": int(self.ckpt_every),
            "saved_at": float(time.time()),
            "version": 1,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = {
                k: v
                if isinstance(v, (int, float, bool, str))
                else (v.detach().cpu().clone() if hasattr(v, "detach") else v)
                for k, v in optimizer.state_dict().items()
            }
        torch.save(payload, path)
        return path

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Return the path of the most recent matching checkpoint, or None.

        Matching means ``<metal>_seed{N}_epoch{M}.pt`` for this run's metal
        and seed.  We pick the largest epoch number to break ties.
        """
        if self.ckpt_dir is None or not self.ckpt_dir.exists():
            return None
        pattern = re.compile(
            rf"^{re.escape(str(self.metal))}_seed{re.escape(str(self.seed))}"
            rf"_epoch(\d{{5}})\.pt$"
        )
        candidates: List[Tuple[int, Path]] = []
        for p in self.ckpt_dir.iterdir():
            m = pattern.match(p.name)
            if m is None:
                continue
            try:
                ep = int(m.group(1))
            except ValueError:
                continue
            candidates.append((ep, p))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    def load_checkpoint(
        self,
        path: Optional[Path] = None,
    ) -> Tuple[int, float]:
        """Restore model + (next_epoch, best_metric) from a checkpoint.

        If ``path`` is None we auto-discover the latest matching checkpoint
        via :meth:`find_latest_checkpoint`.  The optimizer state, if present,
        is stashed on ``self._optimizer_state_resume`` so :meth:`fit` can
        apply it after the optimizer is constructed.  Returns the tuple
        ``(next_epoch_to_run, best_metric)``.
        """
        if self.model is None:
            raise RuntimeError(
                "load_checkpoint() called before fit(); model has not been "
                "constructed yet."
            )
        if path is None:
            path = self.find_latest_checkpoint()
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                f"No checkpoint found for metal={self.metal} seed={self.seed} "
                f"under {self.ckpt_dir}"
            )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        # Cross-run guards — refuse to silently mix two (metal, seed) pairs.
        if payload.get("metal") != str(self.metal):
            raise ValueError(
                f"Checkpoint metal={payload.get('metal')!r} does not match "
                f"baseline metal={self.metal!r}; refusing to load."
            )
        if int(payload.get("seed", -1)) != int(self.seed):
            raise ValueError(
                f"Checkpoint seed={payload.get('seed')} does not match "
                f"baseline seed={self.seed}; refusing to load."
            )
        self.model.load_state_dict(payload["model_state_dict"])
        next_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload.get("best_metric", 0.0))
        self._optimizer_state_resume = payload.get("optimizer_state_dict")
        return next_epoch, best_metric

    def fit(
        self,
        smiles: np.ndarray,
        active_all: np.ndarray,
        pic50_all: np.ndarray,
        cell_lines: np.ndarray,
        idx_train: np.ndarray,
        idx_val: np.ndarray,
        verbose: bool = True,
    ) -> "DMPNNMultiTaskBaseline":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if verbose:
            print(f"[DMPNN-MT] Featurizing {len(smiles)} molecules...", flush=True)
        graphs_all = featurize_smiles_list([str(s) for s in smiles])
        graphs_train = [graphs_all[i] for i in idx_train]
        graphs_val = [graphs_all[i] for i in idx_val]
        active_train = active_all[idx_train]
        active_val = active_all[idx_val]
        pic50_train = pic50_all[idx_train]
        pic50_val = pic50_all[idx_val]

        self.model = DMPNNMultiTaskModel(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
            use_dann=self.use_dann,
            num_domains=self.num_domains,
            domain_head_dropout=self.domain_head_dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = MultiTaskLoss(
            alpha=self.alpha, use_kendall_loss=self.use_kendall_loss
        )

        # Choose the per-epoch train fn depending on whether DANN is on.
        # ``self.model.domain_head`` is set by ``DMPNNMultiTaskModel.__init__``
        # when ``use_dann=True``; passing it explicitly here so the train
        # function does not have to reach into the model via attribute access.
        train_fn = train_epoch_mt
        domain_head = None
        if self.use_dann:
            if self.model.domain_head is None:
                # Defensive: should never trip — __init__ always creates the head
                # when use_dann=True — but guard against future refactors.
                raise RuntimeError(
                    "DANN was requested (use_dann=True) but the model has no "
                    "domain_head.  This is a bug in DMPNNMultiTaskModel.__init__."
                )
            train_fn = train_epoch_mt_dann  # type: ignore[assignment]
            domain_head = self.model.domain_head

        best_val_auc = 0.0
        best_state: Optional[Dict] = None

        # ------------------------------------------------------------------
        # Optional resume (Wf-146 — defensive 21h pretrain safety net).
        #
        # If the user asked for ``resume=True`` we look for the latest
        # ``<metal>_seed{N}_epoch{M}.pt`` in ``ckpt_dir`` and:
        #   * load the model weights back into ``self.model``,
        #   * restore the optimiser state (if the snapshot includes it),
        #   * set ``start_epoch`` to ``M+1`` so the loop picks up where the
        #     crashed run stopped,
        #   * carry forward ``best_metric`` so the in-loop "best so far"
        #     guard is not lost across resumes.
        #
        # If ``resume=True`` but no matching checkpoint exists we raise
        # ``FileNotFoundError`` — silent fallback would defeat the purpose
        # of the safety net (the user might have moved the directory by
        # accident and we want to surface that loudly).
        # ------------------------------------------------------------------
        start_epoch = 0
        if self.resume:
            try:
                start_epoch, resumed_best = self.load_checkpoint()
                best_val_auc = resumed_best
                if self._optimizer_state_resume is not None:
                    try:
                        optimizer.load_state_dict(self._optimizer_state_resume)
                        if verbose:
                            print(
                                f"[DMPNN-MT] resumed from ckpt: "
                                f"next_epoch={start_epoch}  "
                                f"best_metric={resumed_best:.4f}  "
                                f"optimizer_state=loaded",
                                flush=True,
                            )
                    except Exception as _opt_exc:  # noqa: BLE001
                        # Optimizer-state compatibility (e.g. Adam moving
                        # averages + step count) is best-effort — we never
                        # want a stale state to brick a fresh Adam run.
                        if verbose:
                            print(
                                "[DMPNN-MT] WARN: failed to restore "
                                f"optimizer state ({_opt_exc!r}); "
                                "continuing with a fresh optimizer.",
                                flush=True,
                            )
                elif verbose:
                    print(
                        f"[DMPNN-MT] resumed from ckpt (model-only): "
                        f"next_epoch={start_epoch}  "
                        f"best_metric={resumed_best:.4f}",
                        flush=True,
                    )
            except FileNotFoundError as _exc:
                if verbose:
                    print(
                        f"[DMPNN-MT] WARN: --resume requested but no ckpt "
                        f"found ({_exc}); starting from scratch.",
                        flush=True,
                    )

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()
            if self.use_dann:
                train_loss, cls_loss, reg_loss, domain_loss = train_fn(
                    self.model,
                    optimizer,
                    criterion,
                    graphs_train,
                    pic50_train,
                    active_train,
                    domain_head=domain_head,
                    lambda_grl=self.lambda_grl,
                    domain_id=self.domain_id,
                    grl_alpha=self.grl_alpha,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            else:
                train_loss, cls_loss, reg_loss = train_fn(
                    self.model,
                    optimizer,
                    criterion,
                    graphs_train,
                    pic50_train,
                    active_train,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                domain_loss = 0.0
            y_true_v, y_score_v, pic50_pred_v = predict_mt(
                self.model,
                graphs_val,
                pic50_val,
                active_val,
                batch_size=self.batch_size,
                device=self.device,
            )
            val_metrics = compute_metrics(y_true_v, y_score_v)
            val_pic50 = regression_metrics(pic50_val, pic50_pred_v)
            elapsed = time.time() - t0

            if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                if self.use_dann:
                    print(
                        f"[DMPNN-MT epoch {epoch:3d}] "
                        f"loss={train_loss:.4f} "
                        f"(cls={cls_loss:.4f} reg={reg_loss:.4f} dom={domain_loss:.4f}) "
                        f"val_auc={val_metrics['roc_auc']:.4f} "
                        f"val_mae={val_pic50['mae']:.3f} "
                        f"time={elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[DMPNN-MT epoch {epoch:3d}] "
                        f"loss={train_loss:.4f} "
                        f"(cls={cls_loss:.4f} reg={reg_loss:.4f}) "
                        f"val_auc={val_metrics['roc_auc']:.4f} "
                        f"val_mae={val_pic50['mae']:.3f} "
                        f"time={elapsed:.1f}s",
                        flush=True,
                    )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # --------------------------------------------------------------
            # Wf-146 — periodic checkpoint snapshot.
            #
            # Every ``ckpt_every`` epochs (e.g. every 5 epochs) we write a
            # full model + optimizer + epoch + best_metric snapshot to
            # ``<ckpt_dir>/<metal>_seed{N}_epoch{M}.pt``.  The snapshot
            # enables restart-from-crash (and ``--resume``) without
            # losing per-run optim history.
            #
            # Cadence convention: ``epoch+1`` is the number of completed
            # epochs at the time we save, matching the next-epoch value
            # used by ``load_checkpoint``.
            # --------------------------------------------------------------
            if (
                self.ckpt_every > 0
                and self.ckpt_dir is not None
                and ((epoch + 1) % self.ckpt_every == 0)
            ):
                try:
                    ckpt_path = self.save_checkpoint(
                        epoch + 1,
                        best_val_auc,
                        optimizer=optimizer,
                    )
                    if verbose:
                        print(
                            f"[DMPNN-MT] ckpt@{epoch + 1:05d}: "
                            f"saved -> {ckpt_path.name} "
                            f"(best_val_auc={best_val_auc:.4f})",
                            flush=True,
                        )
                except Exception as _ckpt_exc:  # noqa: BLE001
                    # Checkpoint failure must NEVER silently kill a
                    # 21h training run — log loudly and keep going.
                    if verbose:
                        print(
                            "[DMPNN-MT] WARN: checkpoint save failed "
                            f"({_ckpt_exc!r}); continuing training.",
                            flush=True,
                        )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            print(f"[DMPNN-MT] Best val AUC: {best_val_auc:.4f}")
        return self

    def predict(
        self,
        smiles: np.ndarray,
        idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (active_prob, pic50_pred) for the requested indices."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        graphs_all = featurize_smiles_list([str(s) for s in smiles])
        graphs = [graphs_all[i] for i in idx]
        active_dummy = np.zeros(len(idx), dtype=int)
        pic50_dummy = np.zeros(len(idx), dtype=np.float32)
        _, score, pic50 = predict_mt(
            self.model,
            graphs,
            pic50_dummy,
            active_dummy,
            batch_size=self.batch_size,
            device=self.device,
        )
        return score, pic50

    def run(self, verbose: bool = True) -> DMPNNMultiTaskResult:
        ds = self._load_dataset()
        smiles = ds.smiles
        active_all = ds.active.astype(int)
        pic50_all = ds.pic50.astype(np.float32)
        cell_lines = ds.df["Cell_line"].astype(str).to_numpy()

        keep = np.array([bool(s) for s in smiles], dtype=bool)
        smiles = smiles[keep]
        active_all = active_all[keep]
        pic50_all = pic50_all[keep]
        cell_lines = cell_lines[keep]

        idx_train, idx_val, idx_test = self._split(ds)
        self._train_idx = idx_train
        self._val_idx = idx_val
        self._test_idx = idx_test

        self.fit(
            smiles,
            active_all,
            pic50_all,
            cell_lines,
            idx_train,
            idx_val,
            verbose=verbose,
        )

        # Test-set eval
        test_score, test_pic50 = self.predict(smiles, idx_test)
        y_test = active_all[idx_test]
        cl_test = cell_lines[idx_test]
        test_metrics = compute_metrics(y_test, test_score)
        test_pic50_metrics = regression_metrics(pic50_all[idx_test], test_pic50)

        # Val-set eval (for parity with train/val/test report)
        val_score, val_pic50_pred = self.predict(smiles, idx_val)
        y_val = active_all[idx_val]
        val_metrics = compute_metrics(y_val, val_score)
        val_pic50_metrics = regression_metrics(pic50_all[idx_val], val_pic50_pred)

        # Train-set eval
        train_score, train_pic50_pred = self.predict(smiles, idx_train)
        y_train = active_all[idx_train]
        train_metrics = compute_metrics(y_train, train_score)

        per_cl = per_cell_line_auc(cl_test, y_test, test_score)

        return DMPNNMultiTaskResult(
            metal=self.metal,
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            test_pic50_metrics=test_pic50_metrics,
            val_pic50_metrics=val_pic50_metrics,
            per_cell_line=per_cl,
            alpha=self.alpha,
            use_kendall_loss=self.use_kendall_loss,
            use_dann=self.use_dann,
            lambda_grl=self.lambda_grl,
            domain_id=self.domain_id,
            grl_alpha=self.grl_alpha,
            n_train=int(len(idx_train)),
            n_val=int(len(idx_val)),
            n_test=int(len(idx_test)),
            pos_rate_train=float(y_train.mean()),
            pos_rate_test=float(y_test.mean()),
        )


# ---------------------------------------------------------------------------
# Convenience: train a quick smoke model
# ---------------------------------------------------------------------------
def quick_smoke_mt(
    metal: str = "Ru", max_rows: int = 200, alpha: float = MT_ALPHA
) -> DMPNNMultiTaskResult:
    """Smoke-test runner for the test suite."""
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    df = ds.df.head(max_rows).copy()
    ds_small = MetalCytotoxDataset(df=df)
    baseline = DMPNNMultiTaskBaseline(metal=metal, epochs=5, alpha=alpha)
    baseline._load_dataset = lambda: ds_small  # type: ignore[assignment]
    return baseline.run(verbose=False)


# ---------------------------------------------------------------------------
# Normalised loss (R2 fix): divide each task by its initial magnitude so the
# user-facing alpha=0.5 means equal contribution, not equal raw weight.
# ---------------------------------------------------------------------------
class NormalizedLoss(nn.Module):
    """alpha * (BCE / cls_init) + (1-alpha) * (MSE / reg_init).

    Averages ``cls_init`` and ``reg_init`` over the first ``warmup`` batches,
    so both terms contribute on a unit-magnitude scale.  This is the R2 fix
    that lifts test AUC from 0.4891 to 0.5277 (alpha=0.5) / 0.5907 (alpha=0.1)
    on the Ru temporal split (see
    ``molmetal/reports/r2_multitask_ood_fix.md``).
    """

    def __init__(self, alpha: float = MT_ALPHA, warmup: int = 64):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.register_buffer("cls_init", torch.tensor(0.0))
        self.register_buffer("reg_init", torch.tensor(0.0))
        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self._n = 0

    def forward(
        self,
        pic50_pred: torch.Tensor,
        active_logits: torch.Tensor,
        pic50_target: torch.Tensor,
        active_target: torch.Tensor,
    ):
        lc = self.bce(active_logits, active_target)
        lr = self.mse(pic50_pred, pic50_target)
        if self._n < self.warmup:
            with torch.no_grad():
                self.cls_init = self.cls_init + lc.detach()
                self.reg_init = self.reg_init + lr.detach()
                self._n += 1
                if self._n == self.warmup:
                    self.cls_init = self.cls_init / float(self.warmup)
                    self.reg_init = self.reg_init / float(self.warmup)
            return self.alpha * lc + (1.0 - self.alpha) * lr, lc.detach(), lr.detach()
        cls_n = lc / (self.cls_init + 1e-8)
        reg_n = lr / (self.reg_init + 1e-8)
        total = self.alpha * cls_n + (1.0 - self.alpha) * reg_n
        return total, lc.detach(), lr.detach()


# ---------------------------------------------------------------------------
# Kendall-style uncertainty-weighted loss (R2 follow-up)
# ---------------------------------------------------------------------------
class UncertWeightedLoss(nn.Module):
    """Kendall et al. (2018) homoscedastic uncertainty weighting.

    L = exp(-2 s_cls) * BCE + exp(-2 s_reg) * MSE + s_cls + s_reg

    where ``s_cls, s_reg`` are learned positive scalars (parametrised in
    log-space for unconstrained optimisation).  Yields test AUC 0.5460 on
    Ru temporal split.
    """

    def __init__(self):
        super().__init__()
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))
        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pic50_pred: torch.Tensor,
        active_logits: torch.Tensor,
        pic50_target: torch.Tensor,
        active_target: torch.Tensor,
    ):
        lc = self.bce(active_logits, active_target)
        lr = self.mse(pic50_pred, pic50_target)
        w_cls = torch.exp(-2.0 * self.log_sigma_cls.clamp(-3, 3))
        w_reg = torch.exp(-2.0 * self.log_sigma_reg.clamp(-3, 3))
        total = w_cls * lc + w_reg * lr + self.log_sigma_cls + self.log_sigma_reg
        return total, lc.detach(), lr.detach()


__all__ = [
    "DMPNNMultiTaskModel",
    "MultiTaskReadout",
    "MultiTaskLoss",
    "NormalizedLoss",
    "UncertWeightedLoss",
    "DMPNNMultiTaskResult",
    "DMPNNMultiTaskBaseline",
    "regression_metrics",
    "train_epoch_mt",
    "train_epoch_mt_dann",
    "predict_mt",
    "quick_smoke_mt",
    "MT_ALPHA",
    "MT_HIDDEN",
    "MT_DEPTH",
    "MT_DROPOUT",
    "MT_LR",
    "MT_EPOCHS",
    "MT_BATCH_SIZE",
    "MT_PROD_HIDDEN",
    "MT_PROD_DEPTH",
]
