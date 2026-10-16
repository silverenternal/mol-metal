#!/usr/bin/env python
"""BLK-F1 — D-MPNN graph representation ⊕ ECFP4 feature fusion, within-metal.

=====================================================================
Why this file exists
=====================================================================
BLK-P1 measured the D-MPNN on the probe's cohort/folds and found it *below*
the fingerprint+XGBoost column on every metal (Ru 0.7041 vs 0.8271, Ir 0.6912
vs 0.8430, …). That is a clean honest negative, but it leaves the paper with
no methodological contribution — every result is a measurement, a finding, or
a negative.

The fusion hypothesis turns that negative into a claim: if the D-MPNN's graph
representation and ECFP4 carry *different* information, concatenating them
should beat ECFP4 alone even though the D-MPNN alone is the weaker scorer.
The theory is falsifiable and pre-registered here:

* ECFP4 (Morgan r=2) is a *bounded-radius* substructure map. An L-layer MPNN
  is at most as expressive as L-WL (Xu et al. 2019; Morris et al. 2019), so a
  depth-3 D-MPNN is in principle *strictly more* expressive than ECFP4-r2 —
  but on n≈500-1250 ligands it cannot realise that capacity, which is exactly
  why it loses as a standalone scorer.
* Ensemble gain equals member ambiguity (Krogh & Vedelsby 1995):
  ``E_ens = E_avg - A``. If the two representations disagree (A > 0) the
  concatenation can win; if A ≈ 0 it provably cannot. Bias-variance-
  covariance (Ueda & Nakano 1996) says the same about the error covariance.

So a *negative* here is not a failure — it is the ambiguity term being small,
reported as a theoretically-anchored negative.

=====================================================================
Arms (pre-registered; selected by --arms)
=====================================================================
* ``1_ecfp4_xgb``                       — ECFP4 → XGB. Reproduces BLK-P1's XGB
  column (sanity anchor; must match ``dmpnn_within_metal_auc.json``).
* ``2_ecfp4_plus_tmqm_emb_xgb``         — [ECFP4 ⊕ frozen tmQM encoder(256)]
  → XGB. **Leak-free by construction**: the tmQM encoder never saw
  MetalCytoToxDB labels. This is the cleanest "graph rep ⊕ fingerprint → GBDT".
* ``3_ecfp4_plus_dmpnn_mlp``            — end-to-end [ECFP4 ⊕ supervised D-MPNN
  pooled(256)] → MLP, trained per fold. Single-level CV, no cross-fit.
* ``4_ecfp4_mlp``                       — ECFP4 → MLP (same head/schedule as
  arm 3). Separates "the embedding's contribution" (3 vs 4) from "changing the
  head" (4 vs 1).
* ``5_ecfp4_plus_dmpnn_oof_emb_xgb``    — [ECFP4 ⊕ supervised D-MPNN OOF
  embedding(256)] → XGB. **Upper bound only, NOT confirmatory**: two-level OOF
  removes the first-order leak but leaves a second-order, *differential*
  fold-bleed (the encoder that produced row i's embedding trained on the outer
  test rows' labels), which inflates precisely the small delta we would claim.
  See --nested-bleed for the quantified correction.
* ``6_ecfp4_plus_noise_xgb``            — [ECFP4 ⊕ 256-d Gaussian noise] → XGB.
  Feature-count negative control: proves a gain comes from the representation,
  not from "256 extra columns".

=====================================================================
Leakage control (the load-bearing part)
=====================================================================
A *supervised* D-MPNN's in-sample embeddings let a downstream GBDT read labels
straight off the features. Only arms 2/3/4/6 are confirmatory; arm 5 is an
upper bound and is reported as such, with the nested bleed check quantifying
how much of its apparent gain is fold-bleed.

=====================================================================
Statistics
=====================================================================
Per split seed: 5-fold mean AUC (folds are shared across arms → deltas are
paired). ``delta(m, s) = AUC_arm - AUC_ecfp4`` over the 5 split seeds. The
family correction uses a **paired one-sided t-test** (df=4) with a t-based 95%
CI; the percentile bootstrap is reported ``*_reference_only`` because at n=5 it
is anti-conservative (MEASURED on the BLK-PA4 Ir arm: bootstrap p=0.012 vs
t p=0.068 vs Wilcoxon p=0.156 on identical deltas — see
``project_five_seed_bootstrap_anticonservative``). "Fusion wins" requires
mean delta > 0 **and** Holm p < 0.05 **and** t-CI lower bound > 0.

Reporting rule: per-metal table only, **no pooled mean** — Ru is 64% of the
deduplicated cohort, so a pooled mean is a Ru mean wearing a 5-metal label.
Rh/Os/Re (n=51/36/74) are below ``MIN_UNIQUE_FOR_GATE=200`` and are reported
as unresolvable, not gated.

Usage
-----
::

    # canonical: hyperparameters live in the YAML (AGENTS.md §5.11)
    .venv/bin/python molmetal/scripts/fuse_dmpnn_gbdt_within_metal.py \
        --config configs/fusion_dmpnn_gbdt_within_metal.yaml

    # cheap arms first (execution order step 2)
    .venv/bin/python molmetal/scripts/fuse_dmpnn_gbdt_within_metal.py \
        --config configs/fusion_dmpnn_gbdt_within_metal.yaml \
        --arms 1_ecfp4_xgb 2_ecfp4_plus_tmqm_emb_xgb 6_ecfp4_plus_noise_xgb

    # smoke (CPU, 2 epochs, 1 metal, 1 split seed, 200 ligands)
    .venv/bin/python molmetal/scripts/fuse_dmpnn_gbdt_within_metal.py \
        --metals Ru --split-seeds 42 --epochs 2 --max-fold-n 200
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CPU thread control (perf, not correctness) — same rationale as BLK-P1: the
# D-MPNN forward loops over molecules in Python and torch's default intra-op
# pool spawns ~90 threads that thrash on sub-ms work. Must precede torch import.
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from molmetal.scripts.metal_activity_transfer_probe import (  # noqa: E402
    METALS,
    MIN_UNIQUE_FOR_GATE,
    _featurise,
    _make_model,
)
from molmetal.scripts.dmpnn_within_metal_auc import (  # noqa: E402
    MaskedMetalCytotoxModel,
    _load_dedup_cohort,
    _resolve_device,
    _same_folds_as_probe,
)
from molmetal.scripts.finetune_metalcytotox_from_tmqm import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    load_tmqm_encoder,
    make_batch,
)
from molmetal.scripts.synthesize_tmqm_pretrain_init_ab_multimetal import (  # noqa: E402
    _holm,
    _paired_stats,
)

# Belt-and-braces with the OMP_* env vars above (torch may already be imported
# transitively by the modules above, making the env vars too late).
import torch as _torch  # noqa: E402

_torch.set_num_threads(1)
try:
    _torch.set_num_interop_threads(1)
except RuntimeError:
    pass

#: Arm keys, in report order. The first four are confirmatory; arm 5 is an
#: upper bound (documented fold-bleed); arm 6 is a negative control.
ALL_ARMS: Tuple[str, ...] = (
    "1_ecfp4_xgb",
    "2_ecfp4_plus_tmqm_emb_xgb",
    "3_ecfp4_plus_dmpnn_mlp",
    "4_ecfp4_mlp",
    "5_ecfp4_plus_dmpnn_oof_emb_xgb",
    "6_ecfp4_plus_noise_xgb",
)
CONFIRMATORY_ARMS: Tuple[str, ...] = (
    "2_ecfp4_plus_tmqm_emb_xgb",
    "3_ecfp4_plus_dmpnn_mlp",
    "4_ecfp4_mlp",
)
UPPER_BOUND_ARMS: Tuple[str, ...] = ("5_ecfp4_plus_dmpnn_oof_emb_xgb",)
CONTROL_ARMS: Tuple[str, ...] = ("6_ecfp4_plus_noise_xgb",)
CONFIRMATORY_METALS: Tuple[str, ...] = ("Ru", "Ir")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class FusionDMPNN(MaskedMetalCytotoxModel):
    """Masked D-MPNN backbone + a head on ``[ECFP4 ⊕ pooled graph vector]``.

    ``embed`` is inherited from ``MaskedMetalCytotoxModel`` (the exact 2*hidden
    pooled vector the D-MPNN's own head consumes), so arms 2/5 read the same
    representation arm 3 fuses — not a re-derivation of it. The inherited
    ``self.head`` is replaced by ``Identity`` so no dead parameters ride along
    in the optimizer.
    """

    def __init__(
        self,
        ecfp_dim: int,
        hidden: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        mlp_hidden: Optional[int] = None,
    ):
        super().__init__(hidden=hidden, depth=depth, dropout=dropout)
        import torch.nn as nn

        self.ecfp_dim = int(ecfp_dim)
        mh = int(mlp_hidden or hidden)
        self.head = nn.Identity()
        self.fusion_head = nn.Sequential(
            nn.Linear(self.ecfp_dim + 2 * hidden, mh),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mh, 2),
        )

    def forward(self, atom_feats, bond_feats, edge_src, edge_dst, ecfp):
        import torch

        h = self.embed(atom_feats, bond_feats, edge_src, edge_dst)
        return self.fusion_head(torch.cat([ecfp, h], dim=-1))


class ECFPMLP:
    """Factory for the ECFP4-only MLP (arm 4) — the head-change control."""

    @staticmethod
    def build(ecfp_dim: int, mlp_hidden: int, dropout: float):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(int(ecfp_dim), int(mlp_hidden)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_hidden), 2),
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _valid_indices(idx, graphs) -> np.ndarray:
    return np.asarray([int(i) for i in idx if graphs[int(i)] is not None], dtype=int)


def _standardise(train_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std from the TRAIN fold only (leak-free standardisation)."""
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _class_weighted_criterion(targets_tr, device):
    import torch
    import torch.nn as nn

    pos = max(1, int(np.asarray(targets_tr).sum()))
    neg = max(1, int(len(targets_tr) - pos))
    w = torch.tensor([1.0, neg / pos], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


def _auc(y_true, prob) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, prob))


def _xgb_fold_aucs_matrix(
    X: np.ndarray, y: np.ndarray, folds, split_seed: int
) -> List[float]:
    """XGBoost on prebuilt features, on the shared folds (paired reference)."""
    aucs: List[float] = []
    for tr, te in folds:
        if len(set(np.asarray(y[te]).tolist())) < 2:
            aucs.append(float("nan"))
            continue
        clf = _make_model(y[tr], split_seed)
        clf.fit(X[tr], y[tr], verbose=False)
        aucs.append(_auc(y[te], clf.predict_proba(X[te])[:, 1]))
    return aucs


# ---------------------------------------------------------------------------
# Embedding extraction (shared by arms 2 and 5)
# ---------------------------------------------------------------------------
@_torch.no_grad()
def _embed_indices(
    model,
    graphs: List,
    idx: np.ndarray,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    device,
) -> np.ndarray:
    """Masked pooled ``2*hidden`` embeddings for ``idx`` (all-valid indices)."""
    model.eval()
    dim = 2 * int(model.hidden)
    if len(idx) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    chunks: List[np.ndarray] = []
    for start in range(0, len(idx), batch_size):
        bi = idx[start : start + batch_size]
        batched = make_batch(
            [graphs[int(i)] for i in bi],
            [0] * len(bi),
            max_atoms,
            max_edges,
            ATOM_FEATURE_DIM,
            BOND_FEATURE_DIM,
        )
        if batched is None:
            continue
        h = model.embed(
            batched.atom_feats.to(device),
            batched.bond_feats.to(device),
            batched.edge_src.to(device),
            batched.edge_dst.to(device),
        )
        chunks.append(h.cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _train_supervised_dmpnn(
    graphs, targets, tr, train_seed, hidden, depth, dropout, lr, epochs,
    batch_size, max_atoms, max_edges, device,
):
    """Train the masked D-MPNN on ``tr`` (BLK-P1's scorer) and return the model.

    Reused verbatim from BLK-P1's protocol (masked pooling, inverse-class-
    frequency loss, Adam lr=1e-3, grad-clip 5.0) so arm 5's embeddings come
    from the *same* scorer the paper already reports.
    """
    import torch

    from molmetal.scripts.finetune_metalcytotox_from_tmqm import _train_one_epoch

    rng = np.random.default_rng(train_seed)
    torch.manual_seed(train_seed)
    model = MaskedMetalCytotoxModel(hidden=hidden, depth=depth, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = _class_weighted_criterion(targets[tr], device)
    for _ in range(epochs):
        _train_one_epoch(
            model, optimizer, criterion, graphs, targets, tr,
            batch_size=batch_size, max_atoms=max_atoms, max_edges=max_edges,
            device=device, rng=rng,
        )
    return model


# ---------------------------------------------------------------------------
# Arm 3 / 4 — neural heads (fused and ECFP-only), identical schedule
# ---------------------------------------------------------------------------
def _train_neural_head_fold(
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    *,
    train_seed: int,
    mlp_hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device,
    dmpnn_hidden: int = 128,
    dmpnn_depth: int = 3,
    graphs: Optional[List] = None,
    max_atoms: int = 120,
    max_edges: int = 240,
) -> Dict[str, float]:
    """Train a fusion model (``graphs`` given) or an ECFP-only MLP (``graphs`` None).

    The two arms share every schedule knob (standardisation, epochs, lr,
    batch, seed, dropout, class weighting) and differ only in whether the
    pooled D-MPNN vector is concatenated onto the fingerprint — that is what
    makes the 3-vs-4 comparison a clean isolation of the embedding.
    """
    import torch

    rng = np.random.default_rng(train_seed)
    torch.manual_seed(train_seed)

    mu, sd = _standardise(X[tr])
    Xs = (X - mu) / sd

    if graphs is not None:
        model = FusionDMPNN(
            ecfp_dim=X.shape[1], hidden=dmpnn_hidden, depth=dmpnn_depth,
            dropout=dropout, mlp_hidden=mlp_hidden,
        ).to(device)
    else:
        model = ECFPMLP.build(X.shape[1], mlp_hidden, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = _class_weighted_criterion(y[tr], device)

    for _ in range(epochs):
        model.train()
        order = np.asarray(tr).copy()
        rng.shuffle(order)
        for start in range(0, order.shape[0], batch_size):
            bi = order[start : start + batch_size]
            ecfp = torch.from_numpy(np.ascontiguousarray(Xs[bi], dtype=np.float32)).to(device)
            if graphs is not None:
                batched = make_batch(
                    [graphs[int(i)] for i in bi],
                    [int(y[int(i)]) for i in bi],
                    max_atoms, max_edges, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                )
                if batched is None:
                    continue
                logits = model(
                    batched.atom_feats.to(device), batched.bond_feats.to(device),
                    batched.edge_src.to(device), batched.edge_dst.to(device), ecfp,
                )
                tgt = batched.target.to(device)
            else:
                logits = model(ecfp)
                tgt = torch.from_numpy(np.asarray([int(y[int(i)]) for i in bi], dtype=np.int64)).to(device)
            loss = criterion(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    probs: List[float] = []
    true: List[int] = []
    with torch.no_grad():
        for start in range(0, len(te), batch_size):
            bi = te[start : start + batch_size]
            ecfp = torch.from_numpy(np.ascontiguousarray(Xs[bi], dtype=np.float32)).to(device)
            if graphs is not None:
                batched = make_batch(
                    [graphs[int(i)] for i in bi],
                    [int(y[int(i)]) for i in bi],
                    max_atoms, max_edges, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                )
                if batched is None:
                    continue
                logits = model(
                    batched.atom_feats.to(device), batched.bond_feats.to(device),
                    batched.edge_src.to(device), batched.edge_dst.to(device), ecfp,
                )
            else:
                logits = model(ecfp)
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
            true.extend([int(y[int(i)]) for i in bi])
    if len(set(true)) < 2:
        return {"auc": float("nan")}
    return {"auc": _auc(np.asarray(true), np.asarray(probs))}


# ---------------------------------------------------------------------------
# Per-metal driver
# ---------------------------------------------------------------------------
def _run_metal(metal: str, args: argparse.Namespace, device) -> Dict[str, object]:
    t0 = time.time()
    arms: Sequence[str] = tuple(args.arms)
    smiles, y, n_raw = _load_dedup_cohort(metal, getattr(args, "cohort_spec", "24h"))
    if args.max_fold_n and len(y) > args.max_fold_n:
        rng = np.random.default_rng(args.split_seeds[0])
        keep = np.sort(rng.choice(len(y), size=args.max_fold_n, replace=False))
        smiles = [smiles[i] for i in keep]
        y = y[keep]
    n_unique = int(len(y))
    entry: Dict[str, object] = {
        "n_rows_raw": n_raw,
        "n_unique": n_unique,
        "pos_rate": float(y.mean()) if n_unique else float("nan"),
        "resolvable": bool(n_unique >= MIN_UNIQUE_FOR_GATE),
        "arms_run": list(arms),
    }
    if n_unique < 2 * args.n_splits or len(set(y.tolist())) < 2:
        entry["status"] = "SKIPPED"
        entry["reason"] = f"n={n_unique} too small or single-class"
        return entry

    from molmetal.baselines.dmpnn import featurize_smiles_list

    graphs = featurize_smiles_list(list(smiles))
    entry["n_invalid_graphs"] = int(sum(g is None for g in graphs))
    X_ecfp = _featurise(list(smiles), args.radius, args.nbits)

    # ---- frozen tmQM embeddings (arm 2), computed once per metal -------------
    tmqm_emb: Optional[np.ndarray] = None
    if "2_ecfp4_plus_tmqm_emb_xgb" in arms:
        enc_path = Path(args.encoder)
        if not enc_path.exists():
            entry["arm2_status"] = f"SKIPPED: encoder not found at {enc_path}"
        else:
            enc = MaskedMetalCytotoxModel(hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
            n_loaded = load_tmqm_encoder(enc, enc_path)
            # Row-aligned full matrix: invalid-graph rows keep a zero embedding
            # (make_batch compacts valid graphs to the front, so it must only
            # ever be handed an all-valid index array — hence the scatter).
            tmqm_emb = np.zeros((n_unique, 2 * args.hidden), dtype=np.float32)
            valid_all = _valid_indices(np.arange(n_unique, dtype=int), graphs)
            tmqm_emb[valid_all] = _embed_indices(
                enc, graphs, valid_all,
                args.batch_size, args.max_atoms, args.max_edges, device,
            )
            entry["arm2_encoder_tensors_loaded"] = int(n_loaded)
            entry["arm2_emb_dim"] = int(tmqm_emb.shape[1])

    # ---- noise matrix (arm 6) ----------------------------------------------
    noise: Optional[np.ndarray] = None
    if "6_ecfp4_plus_noise_xgb" in arms:
        nrng = np.random.default_rng(args.noise_seed)
        noise = nrng.standard_normal((n_unique, args.noise_dim)).astype(np.float32)

    per_split: List[Dict[str, object]] = []
    per_arm_seed_auc: Dict[str, List[float]] = {a: [] for a in arms}

    for split_seed in args.split_seeds:
        folds = _same_folds_as_probe(y, split_seed, args.n_splits)
        # Filter invalid graphs per fold once, reused across arms.
        folds_valid: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr, te in folds:
            trv = _valid_indices(tr, graphs)
            tev = _valid_indices(te, graphs)
            folds_valid.append((trv, tev))

        arm_fold_aucs: Dict[str, List[float]] = {a: [] for a in arms}

        # ---- arm 1: ECFP4 → XGB (anchor) -----------------------------------
        if "1_ecfp4_xgb" in arms:
            arm_fold_aucs["1_ecfp4_xgb"] = _xgb_fold_aucs_matrix(X_ecfp, y, folds, split_seed)

        # ---- arm 2: [ECFP4 ⊕ frozen tmQM emb] → XGB ------------------------
        if "2_ecfp4_plus_tmqm_emb_xgb" in arms and tmqm_emb is not None and tmqm_emb.shape[0] == n_unique:
            X2 = np.hstack([X_ecfp, tmqm_emb]).astype(np.float32)
            arm_fold_aucs["2_ecfp4_plus_tmqm_emb_xgb"] = _xgb_fold_aucs_matrix(X2, y, folds, split_seed)

        # ---- arm 6: [ECFP4 ⊕ noise] → XGB ----------------------------------
        if "6_ecfp4_plus_noise_xgb" in arms and noise is not None:
            X6 = np.hstack([X_ecfp, noise]).astype(np.float32)
            arm_fold_aucs["6_ecfp4_plus_noise_xgb"] = _xgb_fold_aucs_matrix(X6, y, folds, split_seed)

        # ---- arm 5: supervised D-MPNN OOF embedding → XGB (upper bound) ----
        if "5_ecfp4_plus_dmpnn_oof_emb_xgb" in arms:
            Z = np.zeros((n_unique, 2 * args.hidden), dtype=np.float32)
            for k, (trv, tev) in enumerate(folds_valid):
                if len(trv) == 0 or len(tev) == 0 or len(set(y[tev].tolist())) < 2:
                    continue
                model = _train_supervised_dmpnn(
                    graphs, y, trv, args.train_seeds[0], args.hidden, args.depth,
                    args.dropout, args.lr, args.epochs, args.batch_size,
                    args.max_atoms, args.max_edges, device,
                )
                Z[tev] = _embed_indices(
                    model, graphs, tev, args.batch_size, args.max_atoms, args.max_edges, device
                )
            X5 = np.hstack([X_ecfp, Z]).astype(np.float32)
            arm_fold_aucs["5_ecfp4_plus_dmpnn_oof_emb_xgb"] = _xgb_fold_aucs_matrix(X5, y, folds, split_seed)

        # ---- arms 3 & 4: neural heads, per fold ----------------------------
        need_neural = ("3_ecfp4_plus_dmpnn_mlp" in arms) or ("4_ecfp4_mlp" in arms)
        if need_neural:
            a3, a4 = [], []
            for k, (trv, tev) in enumerate(folds_valid):
                if len(trv) == 0 or len(tev) == 0 or len(set(y[tev].tolist())) < 2:
                    a3.append(float("nan"))
                    a4.append(float("nan"))
                    continue
                if "4_ecfp4_mlp" in arms:
                    ev4 = _train_neural_head_fold(
                        X_ecfp, y, trv, tev, train_seed=args.train_seeds[0],
                        mlp_hidden=args.mlp_hidden, dropout=args.dropout, lr=args.lr,
                        epochs=args.epochs, batch_size=args.batch_size, device=device,
                        graphs=None,
                    )
                    a4.append(ev4["auc"])
                if "3_ecfp4_plus_dmpnn_mlp" in arms:
                    ev3 = _train_neural_head_fold(
                        X_ecfp, y, trv, tev, train_seed=args.train_seeds[0],
                        mlp_hidden=args.mlp_hidden, dropout=args.dropout, lr=args.lr,
                        epochs=args.epochs, batch_size=args.batch_size, device=device,
                        dmpnn_hidden=args.hidden, dmpnn_depth=args.depth,
                        graphs=graphs, max_atoms=args.max_atoms, max_edges=args.max_edges,
                    )
                    a3.append(ev3["auc"])
            if "4_ecfp4_mlp" in arms:
                arm_fold_aucs["4_ecfp4_mlp"] = a4
            if "3_ecfp4_plus_dmpnn_mlp" in arms:
                arm_fold_aucs["3_ecfp4_plus_dmpnn_mlp"] = a3

        row: Dict[str, object] = {"split_seed": split_seed, "arms": {}}
        for a in arms:
            vals = [v for v in arm_fold_aucs.get(a, []) if v == v]  # drop nan
            if vals:
                mean_auc = float(np.mean(vals))
                per_arm_seed_auc[a].append(mean_auc)
                row["arms"][a] = {"mean_auc": mean_auc, "per_fold": arm_fold_aucs[a]}
        per_split.append(row)
        summary_bits = " ".join(
            f"{a.split('_')[0]}={row['arms'][a]['mean_auc']:.4f}"
            for a in arms
            if a in row["arms"]
        )
        print(f"[blk-f1]   {metal} seed={split_seed} {summary_bits}", flush=True)

    # ---- per-arm paired deltas vs arm 1 ------------------------------------
    ref = per_arm_seed_auc.get("1_ecfp4_xgb", [])
    deltas: Dict[str, object] = {}
    for a in arms:
        if a == "1_ecfp4_xgb":
            continue
        vals = per_arm_seed_auc.get(a, [])
        if not vals or len(vals) != len(ref):
            deltas[a] = {"status": "NOT_MEASURED"}
            continue
        d = np.asarray(vals, float) - np.asarray(ref, float)
        deltas[a] = _paired_stats(d)

    entry.update(
        {
            "status": "MEASURED",
            "n_splits": args.n_splits,
            "split_seeds": list(args.split_seeds),
            "per_arm_seed_auc": {a: [round(x, 6) for x in v] for a, v in per_arm_seed_auc.items()},
            "per_arm_mean_auc": {a: round(float(np.mean(v)), 6) for a, v in per_arm_seed_auc.items() if v},
            "deltas_vs_ecfp4_xgb": deltas,
            "per_split": per_split,
            "wall_sec": round(time.time() - t0, 1),
        }
    )
    for a in arms:
        if a in entry["per_arm_mean_auc"]:
            d = deltas.get(a, {})
            dtxt = f"delta={d.get('mean'):+.4f}" if isinstance(d, dict) and "mean" in d else "delta=n/a"
            print(
                f"[blk-f1] {metal:>3} {a:<34} auc={entry['per_arm_mean_auc'][a]:.4f} {dtxt}",
                flush=True,
            )
    print(f"[blk-f1] {metal:>3} wall={entry['wall_sec']}s", flush=True)
    return entry


# ---------------------------------------------------------------------------
# Nested bleed check (quantifies arm 5's second-order fold-bleed)
# ---------------------------------------------------------------------------
def _run_nested_bleed(metal: str, args: argparse.Namespace, device) -> Dict[str, object]:
    """For each outer fold: inner-OOF embeddings for the XGB's TRAIN features.

    Arm 5 trains XGB on train rows whose embeddings came from models that saw
    the outer test fold's labels (second-order differential bleed). This
    variant re-derives the train rows' embeddings from an *inner* CV, so the
    XGB's features are also out-of-sample w.r.t. an inner model — the cleanest
    stack we can build at this scale. If nested ≈ arm 5, the upper bound is
    tight; if nested drops, the gap IS the bleed.
    """
    from sklearn.model_selection import StratifiedKFold
    from molmetal.baselines.dmpnn import featurize_smiles_list

    smiles, y, _ = _load_dedup_cohort(metal, getattr(args, "cohort_spec", "24h"))
    if args.max_fold_n and len(y) > args.max_fold_n:
        rng = np.random.default_rng(args.split_seeds[0])
        keep = np.sort(rng.choice(len(y), size=args.max_fold_n, replace=False))
        smiles = [smiles[i] for i in keep]
        y = y[keep]
    n = len(y)
    graphs = featurize_smiles_list(list(smiles))
    X_ecfp = _featurise(list(smiles), args.radius, args.nbits)

    per_seed: List[Dict[str, object]] = []
    for split_seed in args.nested_split_seeds:
        folds = _same_folds_as_probe(y, split_seed, args.n_splits)
        nested_aucs: List[float] = []
        for tr, te in folds:
            trv = _valid_indices(tr, graphs)
            tev = _valid_indices(te, graphs)
            if len(trv) == 0 or len(tev) == 0 or len(set(y[tev].tolist())) < 2:
                continue
            # Inner-OOF embeddings for the TRAIN rows: each inner fold trains a
            # D-MPNN on its own inner-train and embeds its inner-test, so no
            # train row's feature comes from a model that saw it.
            Z_in = np.zeros((n, 2 * args.hidden), dtype=np.float32)
            inner = StratifiedKFold(
                n_splits=args.nested_inner_splits, shuffle=True,
                random_state=split_seed + 1000,
            )
            for itr, ite in inner.split(np.zeros((len(trv), 1)), y[trv]):
                itr_abs = trv[itr]
                ite_abs = trv[ite]
                m = _train_supervised_dmpnn(
                    graphs, y, itr_abs, args.train_seeds[0], args.hidden, args.depth,
                    args.dropout, args.lr, args.epochs, args.batch_size,
                    args.max_atoms, args.max_edges, device,
                )
                Z_in[ite_abs] = _embed_indices(
                    m, graphs, ite_abs, args.batch_size, args.max_atoms, args.max_edges, device
                )
            # A model trained on the WHOLE outer-train fold embeds the outer test.
            m_full = _train_supervised_dmpnn(
                graphs, y, trv, args.train_seeds[0], args.hidden, args.depth,
                args.dropout, args.lr, args.epochs, args.batch_size,
                args.max_atoms, args.max_edges, device,
            )
            Z_te = _embed_indices(
                m_full, graphs, tev, args.batch_size, args.max_atoms, args.max_edges, device
            )
            Xtr = np.hstack([X_ecfp[trv], Z_in[trv]]).astype(np.float32)
            Xte = np.hstack([X_ecfp[tev], Z_te]).astype(np.float32)
            clf = _make_model(y[trv], split_seed)
            clf.fit(Xtr, y[trv], verbose=False)
            nested_aucs.append(_auc(y[tev], clf.predict_proba(Xte)[:, 1]))
        if nested_aucs:
            per_seed.append(
                {
                    "split_seed": split_seed,
                    "nested_auc": float(np.mean(nested_aucs)),
                    "per_fold_nested": nested_aucs,
                }
            )
            print(
                f"[blk-f1] nested {metal} seed={split_seed} "
                f"nested_auc={per_seed[-1]['nested_auc']:.4f}",
                flush=True,
            )
    return {
        "metal": metal,
        "n_unique": n,
        "inner_splits": args.nested_inner_splits,
        "split_seeds": list(args.nested_split_seeds),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _arm_verdict(metal_entry, arm: str, holm_p: Optional[float]) -> str:
    if not metal_entry.get("resolvable", False):
        return "UNRESOLVABLE"
    d = metal_entry.get("deltas_vs_ecfp4_xgb", {}).get(arm, {})
    if not isinstance(d, dict) or "mean" not in d:
        return "NOT_MEASURED"
    ci_lo = d["ci95_ttest"][0]
    if d["mean"] > 0 and ci_lo > 0 and (holm_p is not None and holm_p < 0.05):
        return "FUSION_WINS"
    if ci_lo > 0:
        return "POSITIVE_NOT_HOLM_SIGNIFICANT"
    if d["ci95_ttest"][0] <= 0.0 <= d["ci95_ttest"][1]:
        return "NULL"
    return "NEGATIVE"


def _family_block(per_metal: Dict[str, object], arms: Sequence[str]) -> Dict[str, object]:
    per_arm: Dict[str, object] = {}
    any_win = False
    for arm in arms:
        if arm == "1_ecfp4_xgb":
            continue
        # confirmatory family = {Ru, Ir}
        p_conf = {}
        for m in CONFIRMATORY_METALS:
            d = per_metal.get(m, {}).get("deltas_vs_ecfp4_xgb", {}).get(arm, {})
            if isinstance(d, dict) and "p_one_sided_ttest" in d:
                p_conf[m] = float(d["p_one_sided_ttest"])
        holm_conf = _holm(p_conf) if p_conf else None
        # exploratory family = all metals measured
        p_expl = {}
        for m, me in per_metal.items():
            d = me.get("deltas_vs_ecfp4_xgb", {}).get(arm, {}) if isinstance(me, dict) else {}
            if isinstance(d, dict) and "p_one_sided_ttest" in d:
                p_expl[m] = float(d["p_one_sided_ttest"])
        holm_expl = _holm(p_expl) if p_expl else None

        per_arm[arm] = {
            "category": (
                "confirmatory" if arm in CONFIRMATORY_ARMS
                else ("upper_bound_only" if arm in UPPER_BOUND_ARMS else "control")
            ),
            "holm_confirmatory_ru_ir": holm_conf,
            "holm_exploratory_5metal": holm_expl,
        }
        # verdicts per metal (attach to the metal entry for convenience)
        for m, me in per_metal.items():
            if not isinstance(me, dict):
                continue
            hp = None
            if holm_conf is not None:
                hp = holm_conf["p_one_sided_holm"].get(m)
            v = _arm_verdict(me, arm, hp)
            me.setdefault("verdicts", {})[arm] = v
            if v == "FUSION_WINS":
                any_win = True
    return {
        "confirmatory_metals": list(CONFIRMATORY_METALS),
        "per_arm": per_arm,
        "family_verdict": (
            "AT_LEAST_ONE_CONFIRMATORY_ARM_METAL_FUSION_WINS"
            if any_win
            else "NO_CONFIRMATORY_ARM_METAL_FUSION_WIN"
        ),
    }


def _with_anchor(arms: Sequence[str]) -> List[str]:
    """Ensure the arm-1 ECFP4→XGB reference is present (every delta needs it)."""
    out = list(arms)
    if "1_ecfp4_xgb" not in out:
        out.insert(0, "1_ecfp4_xgb")
    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    # The arm-1 anchor is the reference every delta is taken against; if any
    # other arm is selected, arm 1 must run too (else no delta is defined).
    args.arms = _with_anchor(args.arms)

    if getattr(args, "deterministic", False):
        # cuBLAS needs a fixed workspace for deterministic matmul on CUDA.
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        # warn_only: the D-MPNN scatter/segment reductions have no deterministic
        # CUDA kernel in some torch versions; warn rather than hard-fail.
        _torch.use_deterministic_algorithms(True, warn_only=True)

    device = _resolve_device(args.device)
    report: Dict[str, object] = {
        "metric": "fusion_dmpnn_gbdt_within_metal",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "BLK-F1: within-metal feature fusion of the D-MPNN graph "
            "representation with ECFP4, on the probe's deduplicated cohort and "
            "identical StratifiedKFold folds. Tests whether the two "
            "representations are complementary (a method contribution) or not "
            "(a theoretically-anchored negative)."
        ),
        "unit": "ROC-AUC (0-1, higher better) and paired delta vs ECFP4-only XGBoost",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/fuse_dmpnn_gbdt_within_metal.py",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "cohort": (
            "identical to metal_activity_transfer_probe: CytotoxFilter("
            "time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m]) then "
            "collapse_duplicates(tie_rule=active)"
            if getattr(args, "cohort_spec", "24h") == "24h"
            else (
                "paper_funnel (Krasnov-style): build_paper_cohort(raw, metal=m) "
                "then collapse_duplicates(tie_rule=active); the Table-1 / BLK-P1 "
                "paper cohort"
            )
        ),
        "cohort_spec": getattr(args, "cohort_spec", "24h"),
        "deterministic": bool(getattr(args, "deterministic", False)),
        "fold_protocol": (
            "StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed); "
            "identical to the probe's within_metal and to BLK-P1."
        ),
        "arms": {
            "1_ecfp4_xgb": "ECFP4 -> XGB (anchor; reproduces BLK-P1's XGB column)",
            "2_ecfp4_plus_tmqm_emb_xgb": "[ECFP4 ⊕ frozen tmQM encoder(256)] -> XGB (leak-free)",
            "3_ecfp4_plus_dmpnn_mlp": "end-to-end [ECFP4 ⊕ supervised D-MPNN pooled(256)] -> MLP",
            "4_ecfp4_mlp": "ECFP4 -> MLP (head-change control for arm 3)",
            "5_ecfp4_plus_dmpnn_oof_emb_xgb": "[ECFP4 ⊕ supervised D-MPNN OOF embedding(256)] -> XGB (UPPER BOUND, fold-bleed)",
            "6_ecfp4_plus_noise_xgb": "[ECFP4 ⊕ 256-d Gaussian noise] -> XGB (feature-count negative control)",
        },
        "arms_run": list(args.arms),
        "split_seeds": list(args.split_seeds),
        "device": str(device),
        "encoder": str(args.encoder),
        "mlp_hidden": args.mlp_hidden,
        "noise_dim": args.noise_dim,
        "min_unique_for_gate": MIN_UNIQUE_FOR_GATE,
        "per_metal": {},
    }

    for metal in args.metals:
        report["per_metal"][metal] = _run_metal(metal, args, device)

    report["family"] = _family_block(report["per_metal"], args.arms)

    # ---- cross-check: arm 1 must reproduce BLK-P1's XGB column --------------
    blk_p1_path = PROJECT_ROOT / "metrics" / "by_metric" / "dmpnn_within_metal_auc.json"
    cross: Dict[str, object] = {"source": "metrics/by_metric/dmpnn_within_metal_auc.json"}
    if blk_p1_path.exists():
        p1 = json.loads(blk_p1_path.read_text())
        for m, me in report["per_metal"].items():
            arm1 = me.get("per_arm_mean_auc", {}).get("1_ecfp4_xgb")
            ref = p1.get("per_metal", {}).get(m, {}).get("xgb_auc_same_folds")
            if arm1 is not None and ref is not None:
                cross[m] = {"arm1_auc": arm1, "blk_p1_xgb_auc": ref, "abs_diff": abs(arm1 - ref)}
    report["cross_check_vs_blk_p1"] = cross

    # ---- nested bleed check (opt-in; expensive) -----------------------------
    if args.nested_bleed:
        nested = [
            _run_nested_bleed(m, args, device)
            for m in args.nested_bleed_metals
            if m in args.metals
        ]
        # Quantify the bleed: arm-5 cross-fit mean AUC vs the nested (clean)
        # mean AUC on the same split seeds. A positive gap is the inflation
        # arm 5's upper bound carries.
        arm5 = "5_ecfp4_plus_dmpnn_oof_emb_xgb"
        for rec in nested:
            m = rec["metal"]
            me = report["per_metal"].get(m, {})
            seeds = me.get("split_seeds", [])
            arm5_aucs = me.get("per_arm_seed_auc", {}).get(arm5)
            gaps = []
            for ps in rec.get("per_seed", []):
                s = ps["split_seed"]
                if arm5_aucs and s in seeds:
                    i = seeds.index(s)
                    if i < len(arm5_aucs):
                        gaps.append(arm5_aucs[i] - ps["nested_auc"])
            rec["arm5_minus_nested_auc_per_seed"] = gaps
            rec["mean_bleed_gap"] = round(float(np.mean(gaps)), 6) if gaps else None
        report["nested_bleed"] = nested

    report["honest_negative"] = {
        "pooled_mean_reported": False,
        "why": (
            "Ru is 64% of the deduplicated cohort, so a pooled mean is a Ru "
            "mean wearing a 5-metal label. The per-metal table is the unit."
        ),
        "unresolvable_metals": [
            m for m, v in report["per_metal"].items() if not v.get("resolvable", False)
        ],
        "arm5_is_upper_bound_only": (
            "Arm 5's OOF embeddings carry a second-order differential fold-bleed "
            "(the encoder that produced a train row's embedding trained on the "
            "outer test rows' labels); its delta must not be read as a "
            "confirmatory gain. See nested_bleed when --nested-bleed was run."
        ),
        "fallback_if_no_fusion_wins": (
            "If no confirmatory arm wins, the honest reading is that the two "
            "representations are not complementary at this scale: the D-MPNN's "
            "graph embedding adds no information beyond ECFP4, consistent with "
            "the D-MPNN being a strictly weaker scorer on every metal (BLK-P1). "
            "Reported with the ambiguity (Krogh & Vedelsby 1995) reading: the "
            "ensemble-gain term is ≈0. MEASURED, not PROJECTED."
        ),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D-MPNN ⊕ ECFP4 fusion within-metal (BLK-F1).")
    p.add_argument("--metals", nargs="+", default=list(METALS))
    p.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44, 49, 123])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-edges", type=int, default=240)
    p.add_argument(
        "--mlp-hidden", type=int, default=128,
        help="hidden width of the fusion / ECFP-only MLP heads (arms 3, 4)",
    )
    p.add_argument(
        "--arms", nargs="+", default=list(ALL_ARMS), choices=list(ALL_ARMS),
        help="which fusion arms to run (default: all)",
    )
    p.add_argument(
        "--encoder", type=Path,
        default=Path("models/tmqm_pretrained/tmqm_pretrained_encoder.pt"),
        help="frozen tmQM encoder checkpoint for arm 2",
    )
    p.add_argument("--noise-dim", type=int, default=256)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument(
        "--nested-bleed", action="store_true", default=False,
        help="run the inner-CV bleed check for arm 5 (expensive; Ru+Ir by default)",
    )
    p.add_argument("--nested-bleed-metals", nargs="+", default=list(CONFIRMATORY_METALS))
    p.add_argument("--nested-split-seeds", type=int, nargs="+", default=[42])
    p.add_argument("--nested-inner-splits", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--cohort-spec", choices=["24h", "paper_funnel"], default="24h",
        help=(
            "Cohort definition.  '24h' (default) = the probe's exact 24 h cohort "
            "(CytotoxFilter time_threshold=24, ic50_min=0.01) — the cohort the "
            "original BLK-F1 probe ran on.  'paper_funnel' = the Krasnov-style "
            "funnel (the Table-1 / BLK-P1 paper cohort); it makes Rh/Os/Re "
            "resolvable (n=334/333/250 vs 51/36/74) and is the cohort the "
            "G1 comparison (D-MPNN vs XGB, BLK-P1) is stated on."
        ),
    )
    p.add_argument(
        "--deterministic", action="store_true",
        help=(
            "Force bit-reproducible kernels (torch.use_deterministic_algorithms, "
            "cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG).  Required for CUDA runs "
            "to be reproducible; harmless on CPU."
        ),
    )
    p.add_argument(
        "--max-fold-n", type=int, default=0,
        help="smoke only: truncate each metal to N ligands. 0 = full cohort.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config (AGENTS.md §5.11). Keys resolve to the dests above; "
        "an explicit CLI flag still wins.",
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/fusion_dmpnn_gbdt_within_metal.json",
    )
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Module-level entry point for scripts/_config_binding_gate.py."""
    p = _build_parser()
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        # wfA2_config_loader lives beside this script; script-mode invocation
        # already puts that dir on sys.path, but importing this module (tests,
        # gates) does not — so make both paths work.
        _scripts_dir = str(Path(__file__).resolve().parent)
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rep = run(args)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float, ensure_ascii=False))

    print(f"\n{'metal':>5} {'arm':<34} {'auc':>8} {'delta':>9}  verdict")
    for m in args.metals:
        v = rep["per_metal"].get(m, {})
        if v.get("status") != "MEASURED":
            print(f"{m:>5}  SKIPPED: {v.get('reason', '')}")
            continue
        for a in args.arms:
            if a not in v.get("per_arm_mean_auc", {}):
                continue
            d = v.get("deltas_vs_ecfp4_xgb", {}).get(a, {})
            dtxt = f"{d['mean']:+.4f}" if isinstance(d, dict) and "mean" in d else "n/a"
            verdict = v.get("verdicts", {}).get(a, "")
            print(f"{m:>5} {a:<34} {v['per_arm_mean_auc'][a]:8.4f} {dtxt:>9}  {verdict}")
    print(f"\nfamily verdict: {rep['family']['family_verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
