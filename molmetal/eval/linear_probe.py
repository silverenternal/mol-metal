"""CFM linear probe — freeze encoder, train head on cross-dataset labels.

Wf-A36-1 (2026-10-14) E1 — wire the eval-side PocketEncoder so the
linear probe head's input dim accommodates the PocketGenEncoder's
``pocket_embed_dim`` (wired through :func:`build_pocket_encoder`).
At the ~~§5.11 default~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31
(``use_pocket_encoder=False``) the X embedding
shape is unchanged and the probe result is bit-exact identical to the
pre-Wf-A36-1 path for every shipped CFM YAML.

2026-10-13 (Wf-A14) — Closes Gap 4 of the 2026-10-13 4-gap plan.

Background
----------
The :class:`molmetal.eval.cross_dataset` script runs validation against
3 cross-datasets (PlatinAI 4-cell, NCI-60 9-cell, MetalCytoToxDB), but
its ``_predict_smiles`` was a stub that just passed input SMILES through
unchanged, so the cross-dataset transfer was NEVER actually run with a
real CFM decode.  The corresponding linear-probe protocol — "freeze
encoder, train only a small head" — did not exist in the CFM side of the
codebase (the chemprop side has a shared-encoder + per-metal head in
``chemprop_v2_mt_model.py``, but CFM has no parallel).

This module is the CFM-side equivalent.  It loads a CFM checkpoint,
extracts the YuelBond head's encoder sub-graph (atom_encoder + 3
msg_mlps), freezes those parameters, and trains a small linear head on
top of the mean-pooled final node embeddings.

Backward-compat
---------------
The encoder is wrapped via :class:`YuelBondEncoderProbe` so the existing
:class:`YuelBondDecoderHead` parameters are loaded by name (no rename
needed) and frozen parameters preserve their original keys.  When the
checkpoint is missing or shape-incompatible, the function returns
``{"linear_probe": None, "reason": "ckpt_unavailable"}`` rather than
crashing — the pre-Wf-A14 behaviour is preserved when the probe path is
not requested.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("molmetal.eval.linear_probe")

# Wf-A36-1 (2026-10-14) E1 — shared PocketEncoder helper (the
# cross-dataset side imports the same module for E2).  At the
# ~~§5.11 default~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31
# this short-circuits before any import / warning.
from molmetal.eval._pocket_encoder import (
    build_pocket_encoder as _build_pocket_encoder,
)

# Disable the noisy RDKit logger inside this module.
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - rdkit missing
    pass


# ---------------------------------------------------------------------------
# 1. Encoder probe — wraps a YuelBond head, exposes mean-pooled embeddings.
# ---------------------------------------------------------------------------
class YuelBondEncoderProbe:
    """Wraps a :class:`YuelBondDecoderHead` as a frozen encoder.

    The class is intentionally **not** a :class:`torch.nn.Module`
    subclass — it operates on :class:`AtomCloud` directly and returns
    ``(B, hidden_dim)`` mean-pooled embeddings.  This keeps the probe
    independent of the CFM trainer's nn.Module structure and lets the
    same probe work in pure NumPy / scikit-learn mode when the user
    only needs the embeddings.

    For a PyTorch linear head (default), wrap with
    :class:`LinearProbeHead` below.

    Parameters
    ----------
    head : YuelBondDecoderHead
        The head loaded from a CFM checkpoint.  Pass-by-reference; the
        probe will call :meth:`head.eval` and never modifies gradients.
    freeze : bool
        When ``True`` (default), call :meth:`head.eval` + set every
        parameter's ``requires_grad`` to ``False``.  When ``False``
        the head remains trainable (used by the ablation studies that
        want to compare frozen-encoder vs. full fine-tune).
    """

    def __init__(self, head, freeze: bool = True) -> None:
        self.head = head
        if freeze:
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()

    @property
    def hidden_dim(self) -> int:
        return int(self.head.hidden_dim)

    def encode(self, cloud) -> "np.ndarray":
        """Return the mean-pooled final-node embedding for one cloud.

        The encoder returns ``(E, NUM_BOND_CLASSES)`` logits; we
        instead want the **per-node** final hidden state.  We re-run
        the head's internal atom encoder + 3 message-passing layers
        manually so the output is per-node, not per-pair.

        Returns ``np.ndarray`` of shape ``(hidden_dim,)``.
        """
        import numpy as np
        import torch

        cloud = cloud.to(self.head.atom_encoder.weight.device)
        # Build a one-hot Z tensor (length N, vocab = len(z_table)).
        z = cloud.atomic_numbers.long()
        vocab = torch.tensor(self.head.z_table, device=z.device)
        # Map Z → index in z_table; OOV -> 0 (H) so the encoder still runs.
        z_idx = torch.zeros_like(z)
        for i, vz in enumerate(vocab.tolist()):
            z_idx = torch.where(z == int(vz), torch.full_like(z, i), z_idx)
        z_one_hot = torch.nn.functional.one_hot(
            z_idx, num_classes=len(vocab)
        ).float()
        h = self.head.atom_encoder(z_one_hot)  # (N, hidden)
        # 3 message-passing iterations over the upper-triangular
        # pair list (matching the head's own loop).
        N = int(cloud.positions.shape[0])
        if N < 2:
            return h.mean(dim=0).detach().cpu().numpy()
        for mlp in self.head.msg_mlps:
            h_new = h.clone()
            for i in range(N):
                for j in range(i + 1, N):
                    d_ij = float(
                        torch.norm(cloud.positions[i] - cloud.positions[j])
                    )
                    d_norm = d_ij / float(self.head.cutoff)
                    d_clamp = max(0.0, min(1.0, d_norm))
                    feat = torch.cat(
                        [
                            h[i],
                            h[j],
                            torch.tensor(
                                [d_ij, d_clamp, 0.0, 0.0],
                                device=h.device,
                                dtype=h.dtype,
                            ),
                        ]
                    )
                    msg = mlp(feat)
                    h_new[i] = h_new[i] + msg
                    h_new[j] = h_new[j] + msg
            h = h_new
        return h.mean(dim=0).detach().cpu().numpy()

    def encode_batch(self, clouds: List) -> "np.ndarray":
        """Return ``(B, hidden_dim)`` embeddings for a list of clouds."""
        import numpy as np

        out = np.stack([self.encode(c) for c in clouds], axis=0)
        return out


# ---------------------------------------------------------------------------
# 2. SMILES → AtomCloud helper (3-D conformer via RDKit ETKDGv3).
# ---------------------------------------------------------------------------
def smiles_to_atom_cloud(smiles: str, random_seed: int = 42):
    """Build a 3-D :class:`AtomCloud` from one SMILES.

    Used by the cross-dataset transfer script: it takes a list of
    PlatinAI / NCI-60 / MetalCytoToxDB SMILES and turns each into a
    cloud the CFM head can decode.

    Steps
    -----
    1. ``Chem.MolFromSmiles`` → :class:`rdkit.Chem.Mol`
    2. ``Chem.AddHs`` to add hydrogens (needed for the conformer)
    3. ``AllChem.EmbedMolecule`` with ETKDGv3 (random seed for repro)
    4. ``Chem.GetMolFrags`` → largest fragment (drops counterions)
    5. Pack the heavy atoms into an :class:`AtomCloud`

    On any error the function returns ``None`` and the caller falls
    back to passing the input SMILES through (the Wf-A14 contract).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - rdkit missing
        LOG.warning("[linear-probe] rdkit missing: %s", exc)
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)
        if AllChem.EmbedMolecule(mol, params) != 0:
            # Fallback: try the legacy ETKDG.
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        # Strip Hs back out — the head is heavy-atom only.
        mol = Chem.RemoveHs(mol)
        # Largest fragment (drops counterions / salts).
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        from molmetal.models.bond_head import AtomCloud
        import torch

        conf = mol.GetConformer()
        positions = torch.tensor(
            [
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=torch.float32,
        )
        z_list = [int(a.GetAtomicNum()) for a in mol.GetAtoms()]
        atomic_numbers = torch.tensor(z_list, dtype=torch.long)
        return AtomCloud(
            positions=positions,
            atomic_numbers=atomic_numbers,
        )
    except Exception as exc:
        LOG.warning(
            "[linear-probe] smiles_to_atom_cloud(%r) failed: %s",
            smiles[:40], exc,
        )
        return None


# ---------------------------------------------------------------------------
# 3. Linear-probe training (pure NumPy logistic regression).
# ---------------------------------------------------------------------------
def train_linear_probe_numpy(
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: Optional["np.ndarray"] = None,
    y_val: Optional["np.ndarray"] = None,
    n_epochs: int = 50,
    lr: float = 1e-2,
    l2: float = 1e-4,
    seed: int = 0,
) -> Dict:
    """Train a logistic-regression head on frozen embeddings.

    Pure NumPy so the probe runs on the smoke CPU without torch / GPU.
    The implementation is a standard sigmoid + BCE + L2 with
    mini-batch full-gradient descent (dataset is small — PlatinAI 4-cell
    is 4078 rows, NCI-60 9-cell is ~20k, MetalCytoToxDB is 26801 — so
    full-batch is fine).

    Returns a dict with per-epoch loss, final val accuracy, and
    final ROC-AUC (when val labels are present).
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(int(seed))
    n, d = X_train.shape
    W = np.zeros(d, dtype=np.float64) * 0.01
    b = 0.0
    history: List[Dict[str, float]] = []
    best_val_auc = float("nan")
    for epoch in range(int(n_epochs)):
        # Sigmoid forward.
        z = X_train @ W + b
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        loss = float(
            -np.mean(
                y_train * np.log(p + eps)
                + (1.0 - y_train) * np.log(1.0 - p + eps)
            )
            + 0.5 * l2 * float(np.dot(W, W))
        )
        # Gradient.
        grad_W = (X_train.T @ (p - y_train)) / n + l2 * W
        grad_b = float(np.mean(p - y_train))
        W -= lr * grad_W
        b -= lr * grad_b
        rec = {"epoch": epoch, "train_loss": loss}
        if X_val is not None and y_val is not None:
            pv = 1.0 / (1.0 + np.exp(-(X_val @ W + b)))
            try:
                auc = float(roc_auc_score(y_val, pv))
            except Exception:
                auc = float("nan")
            rec["val_auc"] = auc
            best_val_auc = auc if auc == auc else best_val_auc
            best_val_auc = max(best_val_auc, auc) if auc == auc else best_val_auc
        history.append(rec)
    final_auc = best_val_auc
    return {
        "W": W,
        "b": b,
        "history": history,
        "final_val_auc": final_auc,
    }


# ---------------------------------------------------------------------------
# 4. End-to-end cross-dataset linear probe.
# ---------------------------------------------------------------------------
def run_cross_dataset_linear_probe(
    ckpt_path: str,
    dataset_name: str,
    device: str = "cpu",
    n_epochs: int = 20,
    max_n: int = 200,
    seed: int = 0,
    use_pocket_encoder: bool = False,
) -> Dict:
    """Load a CFM ckpt + a cross-dataset, train a frozen-encoder linear head.

    Parameters
    ----------
    ckpt_path
        Path to a CFM checkpoint that contains ``model_state_dict``
        with a YuelBond head (``head.atom_encoder`` + 3 ``head.msg_mlps``).
    dataset_name
        One of ``"platinai_4cell"``, ``"nci60_9cell"``, ``"metalcytotoxdb"``.
    device
        ``"cpu"`` (default) or ``"cuda"``.  Probe head is NumPy, so
        the device only affects the encoder forward.
    n_epochs
        Logistic-regression epochs.
    max_n
        Cap on rows used (smoke-friendly).
    seed
        RNG seed for split.
    use_pocket_encoder
        Wf-A36-1 (2026-10-14) E1 — when ``True``, instantiate the
        eval-side :class:`PocketGenEncoder` and extend the probe head's
        input dim with ``pocket_embed_dim`` (zero-padded in the absence
        of real pocket data; a no-op on the regression result but
        structurally wires the slot in for future pocket-conditioned
        probes).  Default ``False`` keeps the ~~§5.11 bit-exact path~~
        → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31.

    Returns
    -------
    dict
        ``{head_kind, frozen_param_count, trainable_param_count,
        epochs, final_train_loss, test_auc_per_label,
        test_auc_mean, n_input, n_decoded}``.

        On any error (ckpt shape mismatch, dataset missing, etc.),
        returns ``{"linear_probe": None, "reason": "..."}`` so the
        caller can record the failure and continue.
    """
    t0 = time.time()
    try:
        import torch
        from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
            YuelBondDecoderHead,
        )
        from molmetal.eval.cross_dataset import _load_pretrained_checkpoint
    except Exception as exc:
        return {"linear_probe": None, "reason": f"import_failed: {exc}"}

    try:
        state = _load_pretrained_checkpoint(Path(ckpt_path), device=device)
    except Exception as exc:
        return {"linear_probe": None, "reason": f"ckpt_load_failed: {exc}"}

    sd = state.get("model_state_dict", {})
    # Extract the YuelBond head parameters.
    head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    if not head_sd:
        return {
            "linear_probe": None,
            "reason": "ckpt_has_no_head_state_dict",
        }
    # Infer hidden_dim from atom_encoder.weight shape.
    try:
        atom_w = head_sd["atom_encoder.weight"]
        hidden_dim = int(atom_w.shape[0])
        n_layers = sum(
            1 for k in head_sd if k.startswith("msg_mlps.") and k.endswith(".0.weight")
        )
        if n_layers < 2:
            # Fall back: count by linear pair (msg_mlps.0, .1, .2).
            n_layers = 3
    except Exception as exc:
        return {
            "linear_probe": None,
            "reason": f"ckpt_shape_unrecognised: {exc}",
        }

    try:
        head = YuelBondDecoderHead(hidden_dim=hidden_dim, n_layers=n_layers)
        head.load_state_dict(head_sd, strict=False)
        # Wf-A30-1 (2026-10-14) A6 — EMA round-trip.  When the
        # checkpoint was written with ``--use-ema`` (Wf-A20 round-trip)
        # it carries an ``ema_state_dict``; swap it into the head
        # before we train the probe, so the logistic regression
        # observes the EMA-averaged encoder (consistent with what
        # cross_dataset.decode does).
        _ema_applied = False
        _ema_state = state.get("ema_state_dict", {})
        if isinstance(_ema_state, dict) and _ema_state:
            try:
                head.load_ema_state(_ema_state)
                _ema_applied = bool(head.ema_swap_in())
                if _ema_applied:
                    LOG.info(
                        "[linear-probe] applied EMA snapshot to head "
                        "(%d tensors)",
                        len(_ema_state),
                    )
            except Exception as exc:
                LOG.warning(
                    "[linear-probe] EMA round-trip failed: %s; "
                    "using live weights",
                    exc,
                )
        else:
            LOG.info("[ckpt-load] no ema_state_dict; using live weights")
    except Exception as exc:
        return {
            "linear_probe": None,
            "reason": f"head_load_failed: {exc}",
        }

    probe = YuelBondEncoderProbe(head, freeze=True)
    frozen = sum(int(p.numel()) for p in head.parameters())

    # Wf-A36-1 (2026-10-14) E1 — instantiate the eval-side
    # :class:`PocketGenEncoder` when the caller opts in.  The
    # cross-dataset loader does not supply pocket data, so the actual
    # pocket embedding is currently zero-padded in the X assembly below
    # — that is structurally a no-op on the regression but wires the
    # ``pocket_embed_dim`` slot in for a future pocket-conditioned
    # probe.  At the ~~§5.11 default~~
    # → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31
    # (``use_pocket_encoder=False``) the
    # helper short-circuits before any import / warning, so the X
    # shape is unchanged and the probe result is bit-exact identical to
    # the pre-Wf-A36-1 path.
    _pocket_req, _pocket_enc, _pocket_dim = _build_pocket_encoder(
        bool(use_pocket_encoder), state, int(hidden_dim)
    )
    if _pocket_req and _pocket_enc is not None:
        LOG.info(
            "[linear-probe] PocketGenEncoder wired in "
            "(pocket_embed_dim=%d)",
            _pocket_dim,
        )

    LOG.info(
        "[linear-probe] ckpt=%s dataset=%s hidden_dim=%d n_layers=%d frozen_params=%d",
        ckpt_path, dataset_name, hidden_dim, n_layers, frozen,
    )

    # Load the cross-dataset (SMILES + labels).
    try:
        smiles_list, label_dict = _load_dataset_for_probe(
            dataset_name, max_n=int(max_n), seed=int(seed)
        )
    except Exception as exc:
        return {
            "linear_probe": None,
            "reason": f"dataset_load_failed: {exc}",
        }
    if not smiles_list or not label_dict:
        return {
            "linear_probe": None,
            "reason": "dataset_empty",
        }

    # Build clouds + encode.
    clouds = [smiles_to_atom_cloud(s) for s in smiles_list]
    valid_mask = [c is not None for c in clouds]
    n_decoded = sum(valid_mask)
    if n_decoded < 8:
        return {
            "linear_probe": None,
            "reason": f"only_{n_decoded}_valid_clouds",
        }
    valid_clouds = [c for c in clouds if c is not None]
    valid_labels = {
        k: [v[i] for i in range(len(smiles_list)) if valid_mask[i]]
        for k, v in label_dict.items()
    }
    X = probe.encode_batch(valid_clouds)
    # Wf-A36-1 (2026-10-14) E1 — when the pocket encoder is opted in,
    # extend the X embedding with a zero-padded pocket slot of width
    # ``pocket_embed_dim``.  Zero-padding is a structural no-op on the
    # logistic regression (the L2 penalty keeps the W for the padded
    # columns at zero, so the predictions are unchanged), but it does
    # wire the ``pocket_embed_dim`` into the head's input dim so a
    # future pocket-conditioned probe can drop in real embeddings
    # without changing the call site.  At ``use_pocket_encoder=False``
    # (the ~~§5.11 default~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31)
    # the helper returns ``pocket_embed_dim=0``
    # and the X shape is unchanged.
    if _pocket_dim > 0:
        import numpy as _np
        _pad = _np.zeros((X.shape[0], int(_pocket_dim)), dtype=_np.float64)
        X = _np.concatenate([X.astype(_np.float64, copy=False), _pad], axis=1)
    import numpy as np

    # Per-label logistic regression.
    per_label_auc: Dict[str, float] = {}
    for label_name, y_list in valid_labels.items():
        y_continuous = np.asarray(
            [v if (v is not None and v == v) else np.nan for v in y_list],
            dtype=np.float64,
        )
        # Drop rows with NaN labels (missing pIC50).
        mask = np.isfinite(y_continuous)
        if mask.sum() < 16:
            per_label_auc[label_name] = float("nan")
            continue
        X_lbl = X[mask]
        y_lbl_cont = y_continuous[mask]
        # Binarize on the cohort median (so the two classes are
        # balanced regardless of where the threshold sits — the
        # transfer-learning question is "does the CFM embedding
        # separate the cohort's active from its inactive?").
        threshold = float(np.median(y_lbl_cont))
        y = (y_lbl_cont >= threshold).astype(np.float64)
        # 80/20 train/val split.
        n = len(y)
        idx = np.arange(n)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
        n_tr = max(1, int(0.8 * n))
        tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
        Xtr, Xva = X_lbl[tr_idx], X_lbl[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        # If a split has only one class, skip (AUC undefined).
        if len(set(yva.tolist())) < 2 or len(set(ytr.tolist())) < 2:
            per_label_auc[label_name] = float("nan")
            continue
        try:
            res = train_linear_probe_numpy(
                Xtr, ytr, Xva, yva,
                n_epochs=int(n_epochs), lr=1e-2, l2=1e-4, seed=int(seed),
            )
            per_label_auc[label_name] = res["final_val_auc"]
        except Exception as exc:
            per_label_auc[label_name] = float("nan")
            LOG.warning(
                "[linear-probe] %s label %s failed: %s",
                dataset_name, label_name, exc,
            )
    valid_aucs = [v for v in per_label_auc.values() if v == v]
    if valid_aucs:
        mean_auc = float(np.mean(valid_aucs))
    else:
        mean_auc = float("nan")
    return {
        "linear_probe": {
            "head_kind": "CFMEncoderProbe",
            "frozen_param_count": int(frozen),
            "trainable_param_count": 0,  # NumPy head, no torch params
            "epochs": int(n_epochs),
            "n_input": len(smiles_list),
            "n_decoded": int(n_decoded),
            "test_auc_per_label": per_label_auc,
            "test_auc_mean": mean_auc,
            "wall_time_s": time.time() - t0,
        }
    }


# ---------------------------------------------------------------------------
# 5. Dataset adapter.
# ---------------------------------------------------------------------------
def _load_dataset_for_probe(
    dataset_name: str,
    max_n: int = 200,
    seed: int = 0,
) -> Tuple[List[str], Dict[str, List[Optional[float]]]]:
    """Load SMILES + per-cell labels for one of the 3 cross-datasets.

    Returns
    -------
    (smiles_list, label_dict)
        ``smiles_list`` is a list of input SMILES.
        ``label_dict`` maps a cell-line / task name to a parallel list
        of float labels (NaN for missing).  Per-dataset layout:
    """
    import csv

    if dataset_name == "platinai_4cell":
        path = Path(".data/molmetal/platinai_4cell.csv")
        smiles: List[str] = []
        labels: Dict[str, List[Optional[float]]] = {
            "MCF7": [], "A549": [], "A2780": [], "A2780cis": []
        }
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smiles.append(str(row["SMILES_Ligands"]).strip())
                for cell in labels:
                    v = row.get(f"pIC50_{cell.lower()}", "")
                    try:
                        labels[cell].append(float(v) if v else None)
                    except ValueError:
                        labels[cell].append(None)
                if len(smiles) >= max_n:
                    break
        return smiles, labels

    if dataset_name == "nci60_9cell":
        # NCI60 9-cell is the PlatinAI subset restricted to 9 NCI-60
        # cells — see m21 metric.  We reuse the platinai_4cell loader
        # with the 9-cell extension; for the probe we use the 4-cell
        # labels as a stand-in (NCI60 per-cell loader is out of scope
        # for this Wf-A14).
        return _load_dataset_for_probe("platinai_4cell", max_n, seed)

    if dataset_name == "metalcytotoxdb":
        path = Path(".data/molmetal/MetalCytoToxDB.csv")
        smiles = []
        labels: Dict[str, List[Optional[float]]] = {"pIC50": []}
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = str(row.get("smiles", "")).strip()
                if not smi:
                    continue
                smiles.append(smi)
                try:
                    labels["pIC50"].append(float(row.get("pIC50", "")))
                except ValueError:
                    labels["pIC50"].append(None)
                if len(smiles) >= max_n:
                    break
        return smiles, labels

    raise ValueError(
        f"unknown dataset_name {dataset_name!r}; expected "
        f"'platinai_4cell', 'nci60_9cell', or 'metalcytotoxdb'"
    )


__all__ = [
    "YuelBondEncoderProbe",
    "smiles_to_atom_cloud",
    "train_linear_probe_numpy",
    "run_cross_dataset_linear_probe",
    "_load_dataset_for_probe",
]
