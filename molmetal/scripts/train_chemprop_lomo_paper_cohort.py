"""PA3 — chemprop v2 (official D-MPNN) cross-metal LOMO on the Paper-A cohort.

WHY THIS SCRIPT EXISTS (BLK-PA3)
--------------------------------
Paper A Claim 1 reports a large within-metal -> cross-metal (LOMO) AUC collapse
measured with a Morgan-FP + XGBoost probe.  The supporting argument that "the
collapse is a property of the metal axis, not a broken backbone" currently rests
on MoleculeNet only, where this repo's D-MPNN loses to the chemprop reference
value on 4/4 tasks (Table 4 of the draft).  A reviewer can therefore argue the
cross-metal 0.44--0.62 column is *underfitting* rather than a metal-axis effect.

The fix (BLK-PA3 Step 1) is to re-run the cross-metal LOMO with the **official
chemprop v2 message-passing backbone** (``chemprop.nn.AtomMessagePassing``,
production ``d_h=300 / depth=3``) as the scorer, on the SAME cohort and the SAME
split the paper's Table 1 uses.  If the collapse reproduces under chemprop, it
cannot be an artefact of a weak backbone.

Same cohort / same split — the part that is easy to get wrong
------------------------------------------------------------
The paper's Table 1 numbers do NOT come from ``load_metal_cytotox()`` (that is
the full 26 801-row MetalCytoToxDB release).  They come from
``metal_activity_transfer_probe.py``, whose cohort is built as:

    MetalCytotoxDataset.from_csv(
        filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01,
                              metal_whitelist=[m]))
    -> collapse_duplicates(...)          # one row per unique ligand, majority
                                          # label, ties -> ACTIVE

and whose LOMO split drops from the training pool any ligand that appears in the
held-out metal's test set.  This script reuses that exact cohort builder
(:func:`collapse_duplicates` imported from the probe) and that exact leakage
rule, so the chemprop numbers are apples-to-apples with Table 1.

Scorer contract — single pooled binary head, no metal features
--------------------------------------------------------------
The XGB probe is a single pooled binary classifier with no metal input: it is
trained on the ligands of four metals and asked to rank the held-out metal's
ligands.  To be comparable, this script uses the chemprop encoder + ONE
``BinaryClassificationFFN`` binary head, and passes no metal embedding.  (The
existing ``train_chemprop_v2_lomo.py`` / ``wf146_chemprop_lomo.py`` use a
*per-metal-head* design in which the held-out head only ever sees zero targets
during LOMO training, which caps the fold AUC and is not the probe's contract.)

Honest framing (AGENTS.md §5.2)
-------------------------------
Every number this script emits is MEASURED on the held-out metal's deduplicated
ligands with ``sklearn.metrics.roc_auc_score`` on the sigmoid of the pooled
head's logit.  No fabricated / aspirational values.  If the chemprop collapse is
smaller than the XGB probe's, that is the finding and it is reported as such —
the honest-negative fallback in BLK-PA3 is to downgrade the model class and let
the XGB probe carry the main claim.

Launch (AGENTS.md §5.11 — YAML only)
------------------------------------
    scripts/launch_pa3_chemprop_lomo.sh [--dry-run]

Every knob lives in ``configs/pa3_chemprop_lomo_v1.yaml``; the wrapper passes
``--config`` (and ``--out-dir``) and nothing else.

Modes
-----
``--mode cross`` (default) reports the leave-one-metal-out column — the paper's
Table-1 *cross* column.  ``--mode within`` reports stratified 5-fold CV inside
each metal (the same ``StratifiedKFold(n_splits, shuffle=True,
random_state=seed)`` protocol the XGB probe uses, reporting the mean of the
per-fold AUCs) — the paper's Table-1 *within* column.  ``--mode both`` runs
each, which is what puts the chemprop within-vs-cross penalty on the same
footing as the probe's pair.

``--n-permutations N`` additionally attaches a label-permutation null to each
cross fold, computed on the ``--perm-seed`` run's fixed ranking via the probe's
own ``permutation_null`` (so the chemprop small-fold verdicts are judged against
the same exchangeability reference as ``metal_activity_transfer_permutation_null
.json``).  The per-row labels/probabilities for that seed are persisted under
``per_fold[*].perm_seed_scores``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_default() -> None:
    """Point ``MOLMETAL_DATA_DIR`` at the in-repo copy before cytotox imports.

    ``molmetal.data.cytotox`` freezes ``DEFAULT_CSV`` at import time from
    ``MOLMETAL_DATA_DIR`` (falling back to a /mnt/storage mount that does not
    exist on this host — MEASURED 2026-10-10 in the probe).  This module's own
    import chain pulls cytotox in, so the env var must be set at the top of the
    file, not inside :func:`build_probe_cohort`.
    """
    import os

    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


_load_env_default()

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from chemprop.data import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from sklearn.metrics import average_precision_score, roc_auc_score

from molmetal.models.chemprop_v2_mt_model import (
    ChempropV2Encoder,
    _make_metal_head,
)

# ---------------------------------------------------------------------------
# Defaults (overridable from the YAML; AGENTS.md §5.11)
# ---------------------------------------------------------------------------
METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44, 49, 123)
DEFAULT_EPOCHS: int = 50
DEFAULT_HIDDEN: int = 300  # chemprop v2 production default
DEFAULT_DEPTH: int = 3  # chemprop v2 production default
DEFAULT_FFN_HIDDEN: int = 300
DEFAULT_FFN_LAYERS: int = 1  # chemprop BinaryClassificationFFN default
DEFAULT_DROPOUT: float = 0.0
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_TIME_THRESHOLD: float = 24.0
DEFAULT_IC50_MIN: float = 0.01
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "pa3_chemprop_lomo"

logger = logging.getLogger("pa3_chemprop_lomo")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Cohort — byte-for-byte the probe's cohort construction.
# ---------------------------------------------------------------------------
def build_probe_cohort(
    metals: List[str],
    time_threshold: float,
    ic50_min: float,
) -> Tuple[Dict[str, Tuple[List[str], np.ndarray]], Dict[str, Dict[str, int]]]:
    """Per-metal deduplicated cohort, identical to the paper's Table-1 probe."""
    _load_env_default()
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    # Reuse the probe's dedup so the cohort is not a look-alike but the same one.
    from molmetal.scripts.metal_activity_transfer_probe import collapse_duplicates

    per_metal: Dict[str, Tuple[List[str], np.ndarray]] = {}
    dup_stats: Dict[str, Dict[str, int]] = {}
    for m in metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(
                time_threshold=float(time_threshold),
                ic50_min=float(ic50_min),
                metal_whitelist=[m],
            )
        )
        s_u, y_u, n_raw = collapse_duplicates(ds.smiles, ds.active)
        per_metal[m] = (list(s_u), np.asarray(y_u, dtype=np.int64))
        dup_stats[m] = {
            "n_rows_raw": int(n_raw),
            "n_unique": int(len(s_u)),
            "pos_rate_dedup": float(np.mean(y_u)) if len(y_u) else 0.0,
        }
    return per_metal, dup_stats


def lomo_split(
    per_metal: Dict[str, Tuple[List[str], np.ndarray]],
    held_out: str,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, int]:
    """LOMO split with the probe's leakage rule.

    A ligand appearing in the held-out metal's test set is dropped from the
    training pool, so no ligand can straddle the split.
    """
    te_s, te_y = per_metal[held_out]
    blocked = set(te_s)
    tr_s: List[str] = []
    tr_y: List[int] = []
    n_leaked = 0
    for m, (s_list, y_arr) in per_metal.items():
        if m == held_out:
            continue
        for s, yv in zip(s_list, y_arr.tolist()):
            if s in blocked:
                n_leaked += 1
                continue
            tr_s.append(s)
            tr_y.append(int(yv))
    return tr_s, np.asarray(tr_y, dtype=np.int64), list(te_s), np.asarray(
        te_y, dtype=np.int64
    ), n_leaked


# ---------------------------------------------------------------------------
# Featurisation cache (chemprop v2 SimpleMoleculeMolGraphFeaturizer).
# ---------------------------------------------------------------------------
def featurize_cache(
    smiles_iter,
) -> Tuple[Dict[str, MolGraph], int, int]:
    """SMILES -> MolGraph, computed once per unique SMILES.

    Invalid SMILES get a 1-atom / 0-bond dummy graph so ``BatchMolGraph`` keeps
    a consistent shape (mirrors ``train_chemprop_v2_lomo.prefeaturize_all_smiles``).
    """
    ftr = SimpleMoleculeMolGraphFeaturizer()
    cache: Dict[str, MolGraph] = {}
    atom_fdim = -1
    bond_fdim = -1
    for smi in smiles_iter:
        if smi in cache:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("RDKit parse failed")
            mg = ftr(mol)
            if atom_fdim < 0:
                atom_fdim = int(mg.V.shape[1])
                bond_fdim = int(mg.E.shape[1])
            cache[smi] = mg
        except Exception:
            if atom_fdim < 0:
                tmp = ftr(Chem.MolFromSmiles("C"))
                atom_fdim = int(tmp.V.shape[1])
                bond_fdim = int(tmp.E.shape[1])
            cache[smi] = MolGraph(
                np.zeros((1, atom_fdim), dtype=np.single),
                np.zeros((0, bond_fdim), dtype=np.single),
                np.zeros((2, 0), dtype=int),
                np.zeros((0,), dtype=int),
            )
    return cache, atom_fdim, bond_fdim


# ---------------------------------------------------------------------------
# Model — chemprop encoder + single pooled binary head (no metal embedding).
# ---------------------------------------------------------------------------
class ChempropBinaryLOMO(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        ffn_hidden: int,
        ffn_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = ChempropV2Encoder(hidden_dim=hidden_dim, depth=depth)
        self.head = _make_metal_head(
            input_dim=hidden_dim,
            hidden_dim=ffn_hidden,
            n_layers=ffn_layers,
            dropout=dropout,
        )

    def logits(self, bmg: BatchMolGraph) -> torch.Tensor:
        h_v = self.encoder.message_passing(bmg)  # (sum_N, hidden_dim)
        h = self.encoder.agg(h_v, bmg.batch)  # (B, hidden_dim)
        return self.head.train_step(h).squeeze(-1)  # (B,)


def _batched(mgs: List[MolGraph], idx: np.ndarray, device: torch.device):
    bmg = BatchMolGraph([mgs[i] for i in idx])
    bmg.to(device)  # in-place; returns None
    return bmg


# ---------------------------------------------------------------------------
# One (fold, seed) run.
# ---------------------------------------------------------------------------
def run_one(
    *,
    model_factory,
    tr_mgs: List[MolGraph],
    tr_y: np.ndarray,
    te_mgs: List[MolGraph],
    te_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    use_amp: bool,
    atom_fdim: int,
    bond_fdim: int,
    return_scores: bool = False,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = model_factory()
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_tr = len(tr_mgs)
    idx = np.arange(n_tr)
    t0 = time.time()
    last_loss = float("nan")
    model.train()
    for _epoch in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        seen = 0
        for start in range(0, n_tr, batch_size):
            bidx = idx[start : start + batch_size]
            bmg = _batched(tr_mgs, bidx, device)
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logit = model.logits(bmg)
            tgt = torch.from_numpy(tr_y[bidx]).float().to(device)
            loss = loss_fn(logit.float(), tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item()) * len(bidx)
            seen += len(bidx)
        last_loss = total / max(seen, 1)

    # Held-out scoring.
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(te_mgs), batch_size):
            bidx = np.arange(start, min(start + batch_size, len(te_mgs)))
            bmg = _batched(te_mgs, bidx, device)
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logit = model.logits(bmg)
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)

    auc = float("nan")
    pr_auc = float("nan")
    if len(p) and len(set(te_y.tolist())) >= 2:
        auc = float(roc_auc_score(te_y, p))
        pr_auc = float(average_precision_score(te_y, p))

    return {
        "seed": int(seed),
        "n_train": int(n_tr),
        "n_test": int(len(te_mgs)),
        "test_auc": auc,
        "test_pr_auc": pr_auc,
        "final_train_loss": float(last_loss),
        "elapsed_s": float(time.time() - t0),
        **(
            {
                "test_labels": [int(v) for v in te_y.tolist()],
                "test_probs": [float(v) for v in p.tolist()],
            }
            if return_scores
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# Within-metal 5-fold CV — the field's own setting, and the paper's Table-1
# "Within" column.  Split protocol mirrors the probe exactly:
# StratifiedKFold(n_splits, shuffle=True, random_state=seed), and the reported
# number is the MEAN of the per-fold AUCs (not the pooled out-of-fold AUC).
# ---------------------------------------------------------------------------
def within_metal_chemprop(
    *,
    model_factory,
    s_u: List[str],
    y_u: np.ndarray,
    cache: Dict[str, MolGraph],
    seeds: List[int],
    n_splits: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_amp: bool,
    atom_fdim: int,
    bond_fdim: int,
) -> Dict[str, object]:
    from sklearn.model_selection import StratifiedKFold

    if len(set(y_u.tolist())) < 2 or len(y_u) < 2 * n_splits:
        return {"status": "SKIPPED", "reason": f"n={len(y_u)} too small or single-class"}

    mgs = [cache[s] for s in s_u]
    per_seed: Dict[str, Dict[str, object]] = {}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        fold_aucs: List[float] = []
        fold_pr: List[float] = []
        for tr_idx, te_idx in skf.split(np.zeros(len(y_u)), y_u):
            rec = run_one(
                model_factory=model_factory,
                tr_mgs=[mgs[i] for i in tr_idx],
                tr_y=y_u[tr_idx],
                te_mgs=[mgs[i] for i in te_idx],
                te_y=y_u[te_idx],
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                seed=int(seed),
                use_amp=use_amp,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
            )
            fold_aucs.append(rec["test_auc"])
            fold_pr.append(rec["test_pr_auc"])
        per_seed[str(seed)] = {
            "fold_aucs": fold_aucs,
            "mean_fold_auc": float(np.mean(fold_aucs)),
            "fold_std": float(np.std(fold_aucs)),
            "mean_fold_pr_auc": float(np.mean(fold_pr)),
        }
    means = [v["mean_fold_auc"] for v in per_seed.values()]
    return {
        "status": "MEASURED",
        "n_unique": int(len(y_u)),
        "n_splits": int(n_splits),
        "pos_rate": float(y_u.mean()),
        "mean_auc": float(np.mean(means)),
        "std": float(np.std(means)),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# CLI (§5.11: --config + run-selecting flags only)
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=str, default=None, help="§5.11 YAML config path")
    p.add_argument("--metals", nargs="+", default=list(METALS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--ffn-hidden", type=int, default=DEFAULT_FFN_HIDDEN)
    p.add_argument("--ffn-layers", type=int, default=DEFAULT_FFN_LAYERS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--time-threshold", type=float, default=DEFAULT_TIME_THRESHOLD)
    p.add_argument("--ic50-min", type=float, default=DEFAULT_IC50_MIN)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--mode",
        choices=("cross", "within", "both"),
        default="cross",
        help=(
            "cross = leave-one-metal-out (the paper's Table-1 cross column); "
            "within = stratified 5-fold CV inside each metal (Table-1 within "
            "column); both = run each.  Default: cross."
        ),
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="StratifiedKFold splits for --mode within (default: 5, matches the probe).",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help=(
            "label-permutation null shuffles per cross fold (0 = skip).  "
            "Run on --perm-seed's fold predictions, matching "
            "metal_activity_transfer_permutation_null.json."
        ),
    )
    p.add_argument(
        "--perm-seed",
        type=int,
        default=42,
        help="Which seed's cross-fold predictions the permutation null uses.",
    )
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BF16 autocast on the chemprop encoder forward (default: True).",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)

    args = p.parse_args(argv)
    if getattr(args, "config", None):
        from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else out_dir / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    use_amp = bool(args.bf16) and device.type == "cuda"

    logger.info(
        "device=%s bf16=%s metals=%s seeds=%s epochs=%d hidden=%d depth=%d "
        "ffn=%dx%d dropout=%.2f batch=%d lr=%.1e cohort=(24h,ic50>=%.2f)",
        device,
        args.bf16,
        args.metals,
        args.seeds,
        args.epochs,
        args.hidden,
        args.depth,
        args.ffn_layers,
        args.ffn_hidden,
        args.dropout,
        args.batch_size,
        args.lr,
        args.ic50_min,
    )

    t_cohort = time.time()
    per_metal, dup_stats = build_probe_cohort(
        list(args.metals), args.time_threshold, args.ic50_min
    )
    logger.info(
        "probe cohort built in %.1fs: %s",
        time.time() - t_cohort,
        {m: dup_stats[m]["n_unique"] for m in per_metal},
    )

    # Featurise every unique SMILES once.
    all_smiles = set()
    for s_list, _ in per_metal.values():
        all_smiles.update(s_list)
    t_feat = time.time()
    cache, atom_fdim, bond_fdim = featurize_cache(sorted(all_smiles))
    logger.info(
        "featurised %d unique SMILES in %.1fs (atom_fdim=%d bond_fdim=%d)",
        len(cache),
        time.time() - t_feat,
        atom_fdim,
        bond_fdim,
    )

    def model_factory():
        return ChempropBinaryLOMO(
            hidden_dim=args.hidden,
            depth=args.depth,
            ffn_hidden=args.ffn_hidden,
            ffn_layers=args.ffn_layers,
            dropout=args.dropout,
        )

    folds: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    sweep_t0 = time.time()
    for held_out in args.metals if args.mode in ("cross", "both") else []:
        if held_out not in per_metal:
            logger.warning("held_out=%s missing from cohort; skip", held_out)
            continue
        tr_s, tr_y, te_s, te_y, n_leaked = lomo_split(per_metal, held_out)
        tr_mgs = [cache[s] for s in tr_s]
        te_mgs = [cache[s] for s in te_s]
        logger.info(
            "fold=%s n_train=%d n_test=%d n_leaked=%d pos_rate_test=%.3f",
            held_out,
            len(tr_s),
            len(te_s),
            n_leaked,
            float(np.mean(te_y)) if len(te_y) else 0.0,
        )
        per_seed: Dict[str, Dict[str, object]] = {}
        for seed in args.seeds:
            want_scores = int(seed) == int(args.perm_seed) and args.n_permutations > 0
            rec = run_one(
                model_factory=model_factory,
                tr_mgs=tr_mgs,
                tr_y=tr_y,
                te_mgs=te_mgs,
                te_y=te_y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=int(seed),
                use_amp=use_amp,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                return_scores=want_scores,
            )
            per_seed[str(seed)] = rec
            runs.append({"metal": held_out, **rec})
            logger.info(
                "fold=%s seed=%d test_auc=%.4f pr_auc=%.4f loss=%.4f %.1fs",
                held_out,
                seed,
                rec["test_auc"],
                rec["test_pr_auc"],
                rec["final_train_loss"],
                rec["elapsed_s"],
            )
        aucs = [
            r["test_auc"]
            for r in per_seed.values()
            if isinstance(r["test_auc"], float) and not np.isnan(r["test_auc"])
        ]
        fold_rec: Dict[str, object] = {
            "metal": held_out,
            "status": "MEASURED" if aucs else "SKIPPED",
            "n_train": int(len(tr_s)),
            "n_test": int(len(te_s)),
            "n_train_rows_dropped_as_leaked": int(n_leaked),
            "pos_rate_test": float(np.mean(te_y)) if len(te_y) else 0.0,
            "per_seed_auc": {k: v["test_auc"] for k, v in per_seed.items()},
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "std": float(np.std(aucs)) if aucs else 0.0,
            "n_seeds_ok": len(aucs),
            "per_seed": per_seed,
        }
        # Label-permutation null on the --perm-seed run's fixed ranking.
        perm_rec = per_seed.get(str(args.perm_seed))
        if args.n_permutations > 0 and perm_rec and "test_probs" in perm_rec:
            from molmetal.scripts.metal_activity_transfer_probe import permutation_null

            fold_rec["permutation_null"] = permutation_null(
                perm_rec["test_labels"],
                perm_rec["test_probs"],
                int(args.n_permutations),
                int(args.perm_seed),
            )
            # Keep the raw scores in the report but out of the fold summary.
            fold_rec["perm_seed_scores"] = {
                "seed": int(args.perm_seed),
                "test_labels": perm_rec["test_labels"],
                "test_probs": perm_rec["test_probs"],
            }
            pn = fold_rec["permutation_null"]
            logger.info(
                "fold=%s permutation-null: obs=%.4f null_sd=%.4f "
                "p_above=%.4f p_below=%.4f",
                held_out,
                pn.get("observed_auc", float("nan")),
                pn.get("null_sd", float("nan")),
                pn.get("p_above_chance", float("nan")),
                pn.get("p_below_chance", float("nan")),
            )
        folds.append(fold_rec)

    mean_across = (
        float(
            np.nanmean(
                [
                    f["mean_auc"]
                    for f in folds
                    if not np.isnan(f["mean_auc"])
                ]
            )
        )
        if folds
        else float("nan")
    )
    total_min = (time.time() - sweep_t0) / 60.0

    # ------------------------------------------------------------------
    # Within-metal 5-fold CV (the paper's Table-1 "Within" column).
    # ------------------------------------------------------------------
    within: Dict[str, object] = {}
    if args.mode in ("within", "both"):
        for held_metal in args.metals:
            if held_metal not in per_metal:
                continue
            s_u, y_u = per_metal[held_metal]
            within[held_metal] = within_metal_chemprop(
                model_factory=model_factory,
                s_u=s_u,
                y_u=y_u,
                cache=cache,
                seeds=[int(s) for s in args.seeds],
                n_splits=int(args.n_splits),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                use_amp=use_amp,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
            )
            w = within[held_metal]
            logger.info(
                "within %s: n=%s mean_auc=%.4f std=%.4f (%d splits)",
                held_metal,
                w.get("n_unique"),
                w.get("mean_auc", float("nan")),
                w.get("std", float("nan")),
                int(args.n_splits),
            )
    within_mean = (
        float(
            np.nanmean(
                [v["mean_auc"] for v in within.values() if v.get("status") == "MEASURED"]
            )
        )
        if within
        else float("nan")
    )

    note = (
        f"PA3 chemprop v2 (AtomMessagePassing d_h={args.hidden}, depth={args.depth}) "
        f"+ single pooled BinaryClassificationFFN ({args.ffn_layers}x{args.ffn_hidden}, "
        f"dropout={args.dropout}) on the Paper-A probe cohort "
        f"(MetalCytotoxDataset + CytotoxFilter(time<={args.time_threshold}h, "
        f"ic50>={args.ic50_min}uM) + collapse_duplicates, leak-dropped train pool). "
        f"mode={args.mode}; {len(runs)} cross (fold, seed) runs in "
        f"{total_min:.2f} min wall.  AUCs are MEASURED with sklearn roc_auc_score "
        f"on the sigmoid of the pooled head's logit.  Same cohort as "
        f"metrics/by_metric/metal_activity_transfer.json (the XGB probe); the "
        f"comparison isolates the scorer, not the cohort."
    )

    report = {
        "metric": "pa3_chemprop_lomo_probe_cohort",
        "mode": str(args.mode),
        "description": (
            "ROC-AUC of the official chemprop v2 message-passing backbone on the "
            "Paper-A probe cohort — the same cohort as the Morgan-FP + XGBoost "
            "probe in metrics/by_metric/metal_activity_transfer.json.  "
            "--mode cross gives the leave-one-metal-out column (the paper's "
            "Table-1 cross), --mode within gives the stratified 5-fold CV inside "
            "each metal (Table-1 within).  Answers BLK-PA3: is the cross-metal "
            "collapse an artefact of a weak backbone?"
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/train_chemprop_lomo_paper_cohort.py",
        "model": (
            f"chemprop v2 AtomMessagePassing (d_h={args.hidden}, depth={args.depth}) "
            f"+ BinaryClassificationFFN (single pooled head, no metal features)"
        ),
        "data_source": (
            "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838) — "
            "probe cohort"
        ),
        "cohort_rule": (
            "per metal: MetalCytotoxDataset.from_csv(CytotoxFilter("
            f"time_threshold={args.time_threshold}, ic50_min={args.ic50_min}, "
            "metal_whitelist=[m])) then collapse_duplicates (one row per unique "
            "ligand, label = majority, ties -> ACTIVE)"
        ),
        "leakage_rule": (
            "a ligand appearing in the held-out metal's test set is dropped from "
            "the LOMO training pool"
        ),
        "split_protocol": (
            "LOMO (leave-one-metal-out), ligand-disjoint"
            if args.mode == "cross"
            else (
                "within-metal stratified 5-fold CV"
                if args.mode == "within"
                else "LOMO (cross) + within-metal stratified 5-fold CV"
            )
        ),
        "comparison_baseline": "metrics/by_metric/metal_activity_transfer.json",
        "seeds": [int(s) for s in args.seeds],
        "duplication": dup_stats,
        "config": {
            "metals": list(args.metals),
            "seeds": [int(s) for s in args.seeds],
            "mode": str(args.mode),
            "n_splits": int(args.n_splits),
            "n_permutations": int(args.n_permutations),
            "perm_seed": int(args.perm_seed),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "ffn_hidden": int(args.ffn_hidden),
            "ffn_layers": int(args.ffn_layers),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "time_threshold": float(args.time_threshold),
            "ic50_min": float(args.ic50_min),
            "device": str(device),
            "bf16_autocast": bool(use_amp),
        },
        "per_fold": folds,
        "mean_auc_across_metals": mean_across,
        "within_metal": within,
        "within_metal_mean_auc": within_mean,
        "total_min": float(total_min),
        "honest_framing_note": note,
        "runs": runs,
    }

    with out_json.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("wrote %s", out_json)
    if folds:
        logger.info("mean cross-metal AUC = %.4f", mean_across)
    if within:
        logger.info("mean within-metal AUC = %.4f", within_mean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
