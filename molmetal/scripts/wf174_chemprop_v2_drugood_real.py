"""Wf-174 BD-11 — chemprop v2 retrain on real DrugOOD lbap_refined_ic50_assay (CPU).

==============================================================================
Wf-174 ultracode follow-up to Wf-170 chemprop v2 baseline.  Builds on BD-1
(drugood_loader canonical 6-col schema) + BD-4 (chemprop_v2.MolGraphCache +
persistent npz) which already shipped (see
``molmetal/scripts/integration_bd4_bd1.py`` integration_report.json).

Scope
-----
* Real DrugOOD ``lbap_refined_ic50_assay`` (266,522 rows × 5 splits:
  train 95,373 + iid_val 31,401 + iid_test 33,164 + ood_val 53,293 +
  ood_test 53,291) — loaded via :func:`molmetal.data.drugood_loader.load_drugood`
  with ``prefer_library=False`` (raw JSON fallback so the .the project .venv can
  consume the data without ``mmcv``/``drugood`` libs installed).
* Train fold (95,373 rows) trains the chemprop v2 backbone for 20 epochs.
* Per-split AUC computed at the end of training for train / iid_val /
  iid_test / ood_val / ood_test (binary classification of
  ``active=True/False`` — DrugOOD's curator-supplied IC50-active flag).
* 5 seeds × 20 epochs, CPU-friendly smaller backbone
  (``hidden_dim=128``, ``depth=2``, ``head_hidden_dim=128``).

Model
-----
:class:`ChempropV2MTModel` is the project's chemprop v2 multi-task
scaffold.  For DrugOOD we feed ``metal_id=0`` for every row and route
the binary target through the ``"Ru"`` head — the other 4 metal heads
get ``metal_weights=0`` so they contribute zero to the loss.  Coord head
is zeroed via ``coord_weight=0`` (DrugOOD has no coordination-geometry
labels).  This makes ChempropV2MTModel behave as a chemprop v2 binary
classifier while still exercising its full forward / loss path
(end-to-end ``.train()`` round-trip).

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* Backbone is smaller than chemprop v2 defaults (``hidden=128/2`` vs
  ``300/3``) — per-step cost ~3-4× lower but capacity also reduced.
* 20 epochs may be insufficient for full convergence at ``hidden=128``.
* No Wasserstein transfer, no L2 guard, no LR scheduler.
* Wall budget 3600s; smaller backbone + 5 seeds × 20 epochs is
  comfortably within budget (~25 min on CPU per integration_bd4_bd1
  G1 measurement: 0.69s / 5K-row epoch × 95K = ~13s/epoch × 20 × 5 ≈
  22 min training + ~2 min eval).
* Cache hit rate from BD-4 is 100% on 2nd pass — first pass on the full
  ~200K unique canonical SMILES takes ~3 min via MolGraphCache with
  n_workers=4 (per integration_bd4_bd1 G1 ratio).

Outputs
-------
* ``molmetal/reports/wf174_chemprop_v2_drugood_real/per_seed/seed<n>.json``
* ``molmetal/reports/wf174_chemprop_v2_drugood_real/summary.json``
* ``molmetal/reports/wf174_chemprop_v2_drugood_real/final.md``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHEMPROP_V2_PATH = PROJECT_ROOT / ".cloned" / "chemprop_v2"
for p in (str(PROJECT_ROOT), str(CHEMPROP_V2_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.data.molgraph import MolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from chemprop.featurizers.molgraph.cache import MolGraphCache  # noqa: E402

from molmetal.data.drugood_loader import load_drugood  # noqa: E402
from molmetal.models.chemprop_v2_mt_model import (  # noqa: E402
    SUPPORTED_METALS,
    ChempropV2MTLoss,
    ChempropV2MTModel,
)

# ---------------------------------------------------------------------------
# Defaults — CPU-friendly smaller backbone per task spec.
# ---------------------------------------------------------------------------
DEFAULT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf174_chemprop_v2_drugood_real"
DEFAULT_CACHE_DIR = DEFAULT_OUT / "cache"
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 20
DEFAULT_HIDDEN: int = 128
DEFAULT_DEPTH: int = 2
DEFAULT_HEAD_HIDDEN: int = 128
DEFAULT_BATCH: int = 128
DEFAULT_LR: float = 1e-3
DEFAULT_WALL_BUDGET_S: float = 3600.0

# DrugOOD splits exposed by load_drugood (raw JSON).
SPLITS = ("train", "iid_val", "iid_test", "ood_val", "ood_test")
SUBSET = "lbap_refined_ic50_assay"

HASH_FN = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()


# --- Wf-190 env guard --------------------------------------------
# Fail fast at import time if --bf16 / CUDA is required but unavailable.
# Mirrors mol-metal-gpu-execution §"MEASURED-vs-PROJECTED label discipline":
# we never silently run on CPU when the script claims GPU.  Auto-fallback to
# CPU is still supported (set MOLMETAL_ALLOW_CPU=1) to keep CPU-only boxes
# working, but explicit is preferred.  The cuda-availability check itself
# lives in main() (after `import torch`) per the molmetal-gpu-execution
# skill convention.
_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"
# --- end Wf-190 env guard ----------------------------------------


def _log(msg: str, **kw) -> None:
    print(f"[wf174] {msg}", flush=True, **kw)


# ---------------------------------------------------------------------------
# Featurization — same pattern as integration_bd4_bd1.G1.featurize_and_cache.
# ---------------------------------------------------------------------------
def _canon_smiles(s: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _molgraph_to_arrays(mg: MolGraph):
    V = np.ascontiguousarray(mg.V.astype(np.float32))
    if mg.E.shape[0] > 0:
        E = np.ascontiguousarray(mg.E.astype(np.float32))
    else:
        E = np.zeros((0, mg.E.shape[1] if mg.E.ndim > 1 else 14), dtype=np.float32)
    edge_index = np.ascontiguousarray(mg.edge_index.astype(np.int64))
    rev = np.ascontiguousarray(mg.rev_edge_index.astype(np.int64))
    return V, E, edge_index, rev


def featurize_and_cache(canon_list: List[str], cache_dir: Path,
                        featurizer: SimpleMoleculeMolGraphFeaturizer,
                        n_workers: int = 4,
                        chunk_size: int = 30000) -> Tuple[Dict[str, dict], int, int]:
    """Persistent MolGraphCache + npz on disk.

    Returns ``(cached_paths, n_skip, n_new)`` where ``cached_paths`` is
    a dict mapping ``canon_smiles -> npz_path_str``.  We deliberately
    store the PATH (lightweight) rather than the npz contents — the
    contents are loaded on demand inside :func:`build_batch_molgraph`.
    This keeps memory bounded for 266K SMILES (the npz contents
    would otherwise be ~1.5 GB if eagerly loaded).

    Chunked: ``MolGraphCache`` (chemprop v2) holds all MolGraphs in
    memory until __init__ returns — for 266K SMILES this OOMs.  We
    therefore process canon_list in chunks of ``chunk_size`` (default
    30K) — each chunk is featurized via MolGraphCache + flushed to npz
    + freed before the next chunk starts.  Already-cached SMILES are
    skipped on each chunk (resume-safe across runs).

    Corrupted-cache recovery: if a previous run was killed mid-write
    the .npz file may be empty / truncated → ``np.load`` raises
    ``EOFError``.  We delete the bad file and re-featurize.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_paths: Dict[str, dict] = {}
    n_skip = 0
    n_new = 0
    todo: List[str] = []
    n_corrupted = 0
    for s in canon_list:
        p = cache_dir / f"{HASH_FN(s)}.npz"
        if p.exists() and p.stat().st_size > 0:
            cached_paths[s] = str(p)
            n_skip += 1
        elif p.exists():
            # Empty / corrupted — delete and re-featurize.
            try:
                p.unlink()
            except OSError:
                pass
            n_corrupted += 1
            todo.append(s)
        else:
            todo.append(s)
    n_new_total = len(todo)
    if n_corrupted:
        _log(f"  recovered {n_corrupted} corrupted cache entries")
    _log(f"  featurize: cache_hits={n_skip}  todo={n_new_total}  "
         f"chunk_size={chunk_size}")
    # Process todo in chunks.
    for chunk_start in range(0, n_new_total, chunk_size):
        chunk = todo[chunk_start:chunk_start + chunk_size]
        _log(f"  featurize chunk [{chunk_start}..{chunk_start + len(chunk)})  "
             f"({len(chunk)} SMILES)")
        mols = [Chem.MolFromSmiles(s) for s in chunk]
        valid = [(s, m) for s, m in zip(chunk, mols) if m is not None]
        if not valid:
            continue
        valid_canon = [s for s, _ in valid]
        valid_mols = [m for _, m in valid]
        cache = MolGraphCache(
            inputs=valid_mols,
            V_fs=[None] * len(valid_mols),
            E_fs=[None] * len(valid_mols),
            featurizer=featurizer,
            n_workers=n_workers,
        )
        n_new += len(valid_canon)
        for s, mg in zip(valid_canon, cache):
            V, E, ei, rev = _molgraph_to_arrays(mg)
            p = cache_dir / f"{HASH_FN(s)}.npz"
            np.savez_compressed(p, V=V, E=E, edge_index=ei, rev_edge_index=rev)
            cached_paths[s] = str(p)
        _log(f"    chunk done  cached_so_far={n_skip + n_new}")
    return cached_paths, n_skip, n_new


def build_batch_molgraph(canon_list: List[str], featurized: Dict[str, dict],
                         atom_fdim: int, bond_fdim: int,
                         device: torch.device) -> BatchMolGraph:
    """Build a chemprop :class:`BatchMolGraph` from canonical SMILES.

    ``featurized`` maps canon_smiles -> ``{"V": ndarray, "E": ndarray,
    "edge_index": ndarray, "rev_edge_index": ndarray}``.  All npz
    arrays are pre-loaded into memory (see
    :func:`load_cached_features_in_memory`) so each batch construction
    is a dict lookup + copy, no disk I/O.
    """
    mgs: List[MolGraph] = []
    for smi in canon_list:
        d = featurized.get(smi)
        if d is None:
            d_V = np.zeros((1, atom_fdim), dtype=np.float32)
            d_E = np.zeros((0, bond_fdim), dtype=np.float32)
            d_ei = np.zeros((2, 0), dtype=np.int64)
            d_rev = np.zeros((0,), dtype=np.int64)
        else:
            d_V = d["V"]
            d_E = d["E"]
            d_ei = d["edge_index"]
            d_rev = d["rev_edge_index"]
        mgs.append(MolGraph(d_V, d_E, d_ei, d_rev))
    bmg = BatchMolGraph(mgs)
    bmg.V = bmg.V.to(device)
    bmg.E = bmg.E.to(device)
    bmg.edge_index = bmg.edge_index.to(device)
    bmg.rev_edge_index = bmg.rev_edge_index.to(device)
    bmg.batch = bmg.batch.to(device)
    return bmg


def load_cached_features_in_memory(cached_paths: Dict[str, dict],
                                   target_canon: set) -> Dict[str, dict]:
    """Load the npz arrays for every SMILES in ``target_canon`` into memory.

    Avoids the per-batch np.load() disk I/O cost during training/eval.
    Memory cost: ~10 KB per molecule × 266K = ~2.7 GB (acceptable).
    """
    out: Dict[str, dict] = {}
    for smi in target_canon:
        path = cached_paths.get(smi)
        if path is None:
            continue
        npz = np.load(path)
        d_V = np.ascontiguousarray(npz["V"].astype(np.float32))
        if npz["E"].ndim > 1 and npz["E"].shape[0] > 0:
            d_E = np.ascontiguousarray(npz["E"].astype(np.float32))
        else:
            d_E = None
        d_ei = np.ascontiguousarray(npz["edge_index"].astype(np.int64))
        d_rev = np.ascontiguousarray(npz["rev_edge_index"].astype(np.int64))
        out[smi] = {"V": d_V, "E": d_E, "edge_index": d_ei, "rev_edge_index": d_rev}
    return out


# ---------------------------------------------------------------------------
# Data loading — load all 5 splits once.
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    canon: List[str]
    y: np.ndarray  # int8 (0/1) — DrugOOD cls_label
    valid_mask: np.ndarray  # bool
    n_total: int
    n_valid: int
    n_pos: int
    load_wall_s: float

    def __repr__(self) -> str:
        return (
            f"SplitData(n_total={self.n_total} n_valid={self.n_valid} "
            f"n_pos={self.n_pos} pos_rate={self.n_pos/max(self.n_valid,1):.3f})"
        )


def load_all_splits(target_root: Optional[Path] = None,
                    n_per_split: Optional[Dict[str, int]] = None,
                    ) -> Dict[str, SplitData]:
    """Load all 5 DrugOOD splits via load_drugood (raw JSON fallback).

    If ``n_per_split`` is supplied, take a head of that many rows after
    canonicalization (useful for smoke runs).
    """
    out: Dict[str, SplitData] = {}
    for sp in SPLITS:
        t0 = time.time()
        df = load_drugood(
            subset=SUBSET, split=sp, prefer_library=False, target_root=target_root,
        )
        # Canonicalize.
        canon = [_canon_smiles(s) for s in df["smiles"].astype(str).tolist()]
        valid = [(c, y) for c, y in zip(canon, df["active"].astype(int).tolist()) if c is not None]
        if n_per_split is not None and sp in n_per_split:
            valid = valid[: n_per_split[sp]]
        canon_list = [c for c, _ in valid]
        y_arr = np.array([y for _, y in valid], dtype=np.int8)
        mask = np.ones(len(canon_list), dtype=bool)
        out[sp] = SplitData(
            canon=canon_list,
            y=y_arr,
            valid_mask=mask,
            n_total=int(len(df)),
            n_valid=int(len(canon_list)),
            n_pos=int(y_arr.sum()),
            load_wall_s=time.time() - t0,
        )
        _log(f"loaded {sp}: {out[sp]} in {out[sp].load_wall_s:.2f}s")
    return out


# ---------------------------------------------------------------------------
# Per-seed training.
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    seed: int
    epochs_done: int
    elapsed_s: float
    train_auc: float
    iid_val_auc: float
    iid_test_auc: float
    ood_val_auc: float
    ood_test_auc: float
    n_train: int
    n_eval: Dict[str, int]
    status: str
    error: Optional[str] = None


def _eval_split_auc(
    model: ChempropV2MTModel,
    canon_list: List[str],
    y_true: np.ndarray,
    featurized: Dict[str, dict],
    atom_fdim: int,
    bond_fdim: int,
    device: torch.device,
    batch_size: int,
    max_eval: Optional[int] = None,
    seed: int = 0,
) -> Tuple[float, int]:
    """Evaluate AUC of metal_id=0 head's sigmoid output on a split.

    Returns ``(auc, n)``.  When the split has only one class, returns
    ``(nan, len(canon_list))`` so we still record what was evaluated.
    """
    model.eval()
    n = len(canon_list)
    if n == 0:
        return float("nan"), 0
    if max_eval is not None and n > max_eval:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_eval, replace=False))
        canon_eval = [canon_list[i] for i in idx]
        y_eval = y_true[idx]
    else:
        canon_eval = canon_list
        y_eval = y_true
    if len(np.unique(y_eval)) < 2:
        return float("nan"), int(len(y_eval))
    scores = np.zeros(len(canon_eval), dtype=np.float32)
    metal_ids = torch.zeros(len(canon_eval), dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, len(canon_eval), batch_size):
            chunk_canon = canon_eval[start:start + batch_size]
            bmg = build_batch_molgraph(chunk_canon, featurized, atom_fdim,
                                       bond_fdim, device)
            H_v = model.encoder.message_passing(bmg)
            H = model.encoder.agg(H_v, bmg.batch)
            e = model.metal_embed(metal_ids[start:start + len(chunk_canon)])
            z = torch.cat([H, e], dim=-1)
            # Use the "Ru" head (index 0) — single binary classifier.
            logit_1 = model.metal_heads["Ru"].train_step(z).squeeze(-1)
            scores[start:start + len(chunk_canon)] = torch.sigmoid(logit_1).cpu().numpy()
    return float(roc_auc_score(y_eval, scores)), int(len(y_eval))


def run_one_seed(
    seed: int,
    splits: Dict[str, SplitData],
    featurized: Dict[str, dict],
    atom_fdim: int,
    bond_fdim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> SeedResult:
    """Train one (seed) and evaluate per-split AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    train_sp = splits["train"]
    canon_train = train_sp.canon
    y_train = train_sp.y.astype(np.float32)

    # Fresh model + loss per seed.
    model = ChempropV2MTModel(
        hidden_dim=args.hidden,
        depth=args.depth,
        n_metals=5,
        n_coord_classes=5,
        head_hidden_dim=args.head_hidden,
        head_n_layers=2,
        head_dropout=0.0,
    )
    # Eagerly build chemprop v2 backbone before .to(device) so lazy-built
    # message_passing / agg submodules get moved too.
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # metal_weights: only "Ru" contributes to the loss; the other 4 metal
    # heads get zero weight — equivalent to single-task binary classifier
    # through the Ru head.  coord_weight=0 zeroes the coord head.
    metal_weights = {m: (1.0 if m == "Ru" else 0.0) for m in SUPPORTED_METALS}
    loss_fn = ChempropV2MTLoss(
        n_coord_classes=5,
        coord_weight=0.0,
        metal_weights=metal_weights,
    ).to(device)

    n_train = len(canon_train)
    perm = np.arange(n_train)
    epoch_done = 0
    timed_out = False
    for epoch in range(1, args.epochs + 1):
        if (time.time() - t0) > args.time_budget_s:
            _log(f"[seed={seed}] wall-budget hit at epoch={epoch - 1}; aborting")
            timed_out = True
            break
        epoch_done = epoch
        np.random.shuffle(perm)
        model.train()
        total_loss = 0.0
        n_batches = 0
        # P1 FIX (Wf-187): BF16 autocast wraps the chemprop v2 message-passing
        # forward pass on CUDA — Blackwell sm_120 runs BF16 matmul ~3x faster
        # than FP32. CPU paths are no-ops (autocast requires CUDA).
        use_amp = args.bf16 and device.type == "cuda"
        for start in range(0, n_train, args.batch):
            chunk_idx = perm[start:start + args.batch]
            chunk_canon = [canon_train[i] for i in chunk_idx]
            chunk_y = torch.from_numpy(y_train[chunk_idx]).to(device)
            bmg = build_batch_molgraph(chunk_canon, featurized, atom_fdim,
                                       bond_fdim, device)
            metal_ids = torch.zeros(len(chunk_canon), dtype=torch.long, device=device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    H_v = model.encoder.message_passing(bmg)
                    H = model.encoder.agg(H_v, bmg.batch)
                    e = model.metal_embed(metal_ids)
                    z = torch.cat([H, e], dim=-1)
            else:
                H_v = model.encoder.message_passing(bmg)
                H = model.encoder.agg(H_v, bmg.batch)
                e = model.metal_embed(metal_ids)
                z = torch.cat([H, e], dim=-1)
            # Per-head logits → assemble V4 (B,2)-shim dict for loss.
            out = {}
            for m, head in model.metal_heads.items():
                logit_1 = head.train_step(z)  # (B, 1) pre-sigmoid
                zeros = torch.zeros_like(logit_1)
                out[m] = torch.cat([logit_1, zeros], dim=-1)  # (B, 2)
            out["coord"] = model.coord_head(z)
            # y_true dict — only "Ru" matters (others have metal_weights=0).
            y_true = {m: chunk_y for m in SUPPORTED_METALS}
            coord_true = torch.zeros(len(chunk_canon), dtype=torch.long, device=device)
            loss, info = loss_fn(out, y_true, coord_true)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            total_loss += float(loss.detach().cpu().item()) * len(chunk_canon)
            n_batches += len(chunk_canon)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            _log(
                f"[seed={seed}] epoch={epoch:3d}/{args.epochs}  "
                f"loss={total_loss / max(n_batches, 1):.4f}  "
                f"elapsed={time.time() - t0:.1f}s",
            )

    # Per-split evaluation.
    eval_results: Dict[str, Tuple[float, int]] = {}
    n_eval: Dict[str, int] = {}
    for sp in SPLITS:
        sd = splits[sp]
        # Subsample eval splits to bound eval wall-time (consistent with
        # wf170_chemprop_drugood RECIPE_EVAL_MAX_SAMPLES=4000).
        max_eval = args.eval_max_samples if sp != "train" else args.eval_train_max_samples
        auc, n = _eval_split_auc(
            model=model,
            canon_list=sd.canon,
            y_true=sd.y.astype(np.int8),
            featurized=featurized,
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            device=device,
            batch_size=args.batch,
            max_eval=max_eval,
            seed=seed,
        )
        eval_results[sp] = (auc, n)
        n_eval[sp] = n
        _log(f"[seed={seed}] {sp}_auc={auc:.4f} n_eval={n}")

    elapsed = time.time() - t0
    status = "ok" if not timed_out else "timeout"
    return SeedResult(
        seed=seed,
        epochs_done=epoch_done,
        elapsed_s=elapsed,
        train_auc=float(eval_results["train"][0]),
        iid_val_auc=float(eval_results["iid_val"][0]),
        iid_test_auc=float(eval_results["iid_test"][0]),
        ood_val_auc=float(eval_results["ood_val"][0]),
        ood_test_auc=float(eval_results["ood_test"][0]),
        n_train=n_train,
        n_eval=n_eval,
        status=status,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    # P0 FIX (Wf-187): default to GPU when CUDA is available; line ~582 below
    # auto-falls-back to CPU if torch.cuda.is_available() is False. This keeps
    # backward compat (CPU box still works) while letting the Blackwell sm_120
    # actually carry the chemprop v2 message-passing compute instead of
    # leaving the GPU idle at 0% with a lazy 550 MiB CUDA context.
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--reset-cache", action="store_true",
                   help="Delete the cache dir before featurizing.")
    p.add_argument("--time-budget-s", type=float, default=DEFAULT_WALL_BUDGET_S)
    p.add_argument("--eval-max-samples", type=int, default=4000,
                   help="Subsample non-train splits to this many rows for eval.")
    p.add_argument("--eval-train-max-samples", type=int, default=8000,
                   help="Subsample the train split to this many rows for eval AUC.")
    p.add_argument("--n-train", type=int, default=None,
                   help="Optional cap on train rows (for smoke runs).")
    p.add_argument("--n-per-split", type=str, default=None,
                   help="Comma-separated split=cap dict, e.g. 'train=2000,iid_val=500'. "
                        "Overrides --n-train.")
    # P1 FIX (Wf-187): add BF16 autocast + DataLoader-style flags so the
    # chemprop v2 retrain scripts can opt into Blackwell sm_120 BF16 matmul
    # (~3x FP32 throughput) and overlap CPU featurize with GPU forward.
    # Mirrors train_metal_lomo_v2.py lines 1187-1209. No-op on CPU.
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wrap model.encoder.message_passing() in "
            "torch.amp.autocast('cuda', dtype=torch.bfloat16) when CUDA is "
            "active (default: True). Pass --no-bf16 to disable. No-op on CPU."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4). Currently logged; the manual "
             "range-based loop at line ~460 is the next-migration target.",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader pin_memory (default: True). Same caveat as --num-workers.",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    if args.reset_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    n_per_split: Optional[Dict[str, int]] = None
    if args.n_per_split:
        n_per_split = {}
        for kv in args.n_per_split.split(","):
            k, v = kv.split("=")
            n_per_split[k.strip()] = int(v.strip())
    elif args.n_train is not None:
        n_per_split = {"train": args.n_train}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        # Wf-190 env guard: only auto-fall-back to CPU if the script-level
        # allow-CPU flag is set OR --bf16 is explicitly disabled (otherwise
        # the user explicitly asked for GPU and we should fail fast rather
        # than silently burn hours on a CPU box).
        if args.bf16 and not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[wf174] CUDA requested (--device cuda) but "
                "torch.cuda.is_available() is False and --bf16 is enabled.  Either:\n"
                "  -1- re-run on a CUDA-enabled host, OR\n"
                "  -2- pass --no-bf16 --device cpu for CPU fallback, OR\n"
                "  -3- set MOLMETAL_ALLOW_CPU=1 to allow silent CPU fallback "
                "(Wf-190 env guard)."
            )
        _log(
            f"CUDA requested but torch.cuda.is_available() is False; "
            f"falling back to CPU.  Set MOLMETAL_ALLOW_CPU=0 + --no-bf16 "
            f"for fail-fast behaviour.",
        )
        device = torch.device("cpu")
    _log(f"device={device}  bf16={args.bf16}  num_workers={args.num_workers}  "
         f"pin_memory={args.pin_memory}  seeds={args.seeds}  epochs={args.epochs}  "
         f"hidden={args.hidden} depth={args.depth} head_hidden={args.head_hidden}  "
         f"batch={args.batch} lr={args.lr}  "
         f"time_budget_s={args.time_budget_s}  "
         f"n_per_split={n_per_split}")

    # ------------------------------------------------------------------
    # Load all 5 splits.
    # ------------------------------------------------------------------
    splits = load_all_splits(n_per_split=n_per_split)
    for sp in SPLITS:
        _log(f"  {sp}: {splits[sp]}")

    # ------------------------------------------------------------------
    # Featurize all unique canonical SMILES across all splits (one-time).
    # ------------------------------------------------------------------
    canon_unique_all = sorted({c for sp in SPLITS for c in splits[sp].canon})
    _log(f"featurizing {len(canon_unique_all)} unique canonical SMILES "
         f"(cache_dir={cache_dir})...")
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    t_fe = time.time()
    cached_paths, n_skip, n_new = featurize_and_cache(
        canon_unique_all, cache_dir, featurizer, n_workers=4,
    )
    fe_wall = time.time() - t_fe
    _log(f"  featurized n_skip={n_skip}  n_new={n_new}  in {fe_wall:.2f}s  "
         f"({100*n_skip/max(len(canon_unique_all),1):.1f}% hit)")

    # Probe fdim from a sample npz.
    sample_path = next(iter(cached_paths.values()))
    sample_npz = np.load(sample_path)
    atom_fdim = int(sample_npz["V"].shape[1])
    bond_fdim = int(sample_npz["E"].shape[1]) if sample_npz["E"].ndim > 1 else 14
    _log(f"  atom_fdim={atom_fdim}  bond_fdim={bond_fdim}")

    # Load all npz arrays into memory (avoids per-batch np.load disk I/O).
    target_canon = {c for sp in SPLITS for c in splits[sp].canon}
    _log(f"loading {len(target_canon)} npz arrays into memory...")
    t_lm = time.time()
    featurized = load_cached_features_in_memory(cached_paths, target_canon)
    _log(f"  loaded in {time.time() - t_lm:.2f}s "
         f"({len(featurized)} / {len(target_canon)} entries)")

    # ------------------------------------------------------------------
    # Per-seed training.
    # ------------------------------------------------------------------
    t_wall_start = time.time()
    results: List[SeedResult] = []
    for seed in args.seeds:
        if (time.time() - t_wall_start) > args.time_budget_s:
            _log(f"wall-budget hit before seed={seed}; stopping")
            break
        try:
            r = run_one_seed(
                seed=seed,
                splits=splits,
                featurized=featurized,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                args=args,
                device=device,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _log(f"[seed={seed}] FAILED: {exc}")
            r = SeedResult(
                seed=seed, epochs_done=0, elapsed_s=0.0,
                train_auc=float("nan"), iid_val_auc=float("nan"),
                iid_test_auc=float("nan"), ood_val_auc=float("nan"),
                ood_test_auc=float("nan"), n_train=0, n_eval={},
                status="error", error=str(exc),
            )
        results.append(r)
        # Persist per-seed JSON.
        per_seed_path = out_dir / "per_seed" / f"seed{seed}.json"
        per_seed_path.write_text(json.dumps({
            "seed": r.seed,
            "epochs_done": r.epochs_done,
            "elapsed_s": r.elapsed_s,
            "train_auc": r.train_auc,
            "iid_val_auc": r.iid_val_auc,
            "iid_test_auc": r.iid_test_auc,
            "ood_val_auc": r.ood_val_auc,
            "ood_test_auc": r.ood_test_auc,
            "n_train": r.n_train,
            "n_eval": r.n_eval,
            "status": r.status,
            "error": r.error,
        }, indent=2))
        _log(f"[seed={seed}] FINAL  train={r.train_auc:.4f}  "
             f"iid_val={r.iid_val_auc:.4f}  iid_test={r.iid_test_auc:.4f}  "
             f"ood_val={r.ood_val_auc:.4f}  ood_test={r.ood_test_auc:.4f}  "
             f"({r.elapsed_s:.1f}s status={r.status})")

    total_wall_min = (time.time() - t_wall_start) / 60.0

    # ------------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------------
    def _mean(xs: List[float]) -> float:
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    def _std(xs: List[float]) -> float:
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    per_split_aucs: Dict[str, List[float]] = {
        "train": [r.train_auc for r in results],
        "iid_val": [r.iid_val_auc for r in results],
        "iid_test": [r.iid_test_auc for r in results],
        "ood_val": [r.ood_val_auc for r in results],
        "ood_test": [r.ood_test_auc for r in results],
    }
    per_split_mean = {k: _mean(v) for k, v in per_split_aucs.items()}
    per_split_std = {k: _std(v) for k, v in per_split_aucs.items()}

    ood_test_mean = per_split_mean["ood_test"]
    passes_ood_gate = bool(not np.isnan(ood_test_mean) and ood_test_mean > 0.55)
    n_ok = sum(1 for r in results if r.status == "ok")
    n_total = len(results)

    honest_framing_note = (
        f"Wf-174 BD-11 chemprop v2 retrain on real DrugOOD "
        f"lbap_refined_ic50_assay ({SUBSET}). "
        f"Train fold size = {splits['train'].n_valid} rows; "
        f"eval splits = iid_val({splits['iid_val'].n_valid}), "
        f"iid_test({splits['iid_test'].n_valid}), "
        f"ood_val({splits['ood_val'].n_valid}), "
        f"ood_test({splits['ood_test'].n_valid}). "
        f"Backbone is hidden_dim={args.hidden}, depth={args.depth}, "
        f"head_hidden_dim={args.head_hidden} (smaller than chemprop v2 "
        f"defaults 300/3 — per-step cost ~3-4x lower but capacity also "
        f"reduced). {args.epochs} epochs x batch {args.batch} x lr {args.lr} "
        f"x {len(args.seeds)} seeds (CPU-friendly, wall "
        f"{total_wall_min:.2f} min). "
        f"{n_ok}/{n_total} seeds ran cleanly. "
        f"AUC is sigmoid(chemprop BinaryClassificationFFN.train_step(z)[:,0]) "
        f"on the metal_heads['Ru'] head; metal_weights zeroed for Ir/Os/Rh/Re, "
        f"coord_weight=0 — ChempropV2MTModel behaves as single-task binary "
        f"classifier through the Ru head. Eval subsampled to "
        f"eval_max_samples={args.eval_max_samples} on non-train splits "
        f"({args.eval_train_max_samples} on train) to bound eval wall-time. "
        f"BD-4 MolGraphCache hit rate = "
        f"{100*n_skip/max(len(canon_unique_all),1):.1f}% on this run "
        f"(n_skip={n_skip}, n_new={n_new}, featurize_wall={fe_wall:.1f}s). "
        f"NO Wasserstein transfer, NO L2 guard, NO LR scheduler — pure "
        f"chemprop v2 BCE baseline. Honest framing: ood_test AUC = "
        f"{ood_test_mean:.4f} (gate ≥0.55 → "
        f"{'PASS' if passes_ood_gate else 'FAIL'}); if PASS, this replaces "
        f"the LOMO v1 'broken like LOMO v1' honest-negative baseline "
        f"for the BD-11 chemprop v2 line on lbap_refined_ic50_assay."
    )

    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "data_source": f"drugood_{SUBSET}",
        "subset": SUBSET,
        "splits": {sp: {
            "n_total": int(splits[sp].n_total),
            "n_valid": int(splits[sp].n_valid),
            "n_pos": int(splits[sp].n_pos),
            "pos_rate": float(splits[sp].n_pos / max(splits[sp].n_valid, 1)),
            "load_wall_s": float(splits[sp].load_wall_s),
        } for sp in SPLITS},
        "config": {
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "head_hidden": int(args.head_hidden),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "device": str(device),
            "time_budget_s": float(args.time_budget_s),
            "eval_max_samples": int(args.eval_max_samples),
            "eval_train_max_samples": int(args.eval_train_max_samples),
            "n_per_split": n_per_split,
        },
        "featurization": {
            "n_unique_canon": int(len(canon_unique_all)),
            "n_skip": int(n_skip),
            "n_new": int(n_new),
            "hit_rate": float(n_skip / max(len(canon_unique_all), 1)),
            "featurize_wall_s": float(fe_wall),
            "atom_fdim": int(atom_fdim),
            "bond_fdim": int(bond_fdim),
        },
        "per_split_auc_mean": per_split_mean,
        "per_split_auc_std": per_split_std,
        "per_split_auc_per_seed": per_split_aucs,
        "ood_test_auc_mean": float(ood_test_mean),
        "passes_ood_gate": passes_ood_gate,
        "n_seeds_ok": int(n_ok),
        "n_seeds_total": int(n_total),
        "total_wall_min": float(total_wall_min),
        "honest_framing_note": honest_framing_note,
        "results": [
            {
                "seed": r.seed, "epochs_done": r.epochs_done,
                "elapsed_s": r.elapsed_s,
                "train_auc": r.train_auc,
                "iid_val_auc": r.iid_val_auc,
                "iid_test_auc": r.iid_test_auc,
                "ood_val_auc": r.ood_val_auc,
                "ood_test_auc": r.ood_test_auc,
                "n_train": r.n_train, "n_eval": r.n_eval,
                "status": r.status, "error": r.error,
            }
            for r in results
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {summary_path}")

    # ------------------------------------------------------------------
    # Markdown report.
    # ------------------------------------------------------------------
    md = [
        "# Wf-174 BD-11 — chemprop v2 retrain on DrugOOD lbap_refined_ic50_assay",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Data:** DrugOOD lbap_refined_ic50_assay (266,522 rows × 5 splits)  ",
        f"**Train rows:** {splits['train'].n_valid}  ",
        f"**Model:** chemprop v2 AtomMessagePassing + SumAggregation + "
        f"BinaryClassificationFFN (via ChempropV2MTModel Ru head)  ",
        f"**Backbone:** hidden={args.hidden}, depth={args.depth}, "
        f"head_hidden={args.head_hidden}  ",
        f"**Recipe:** {args.epochs} epochs × batch {args.batch} × lr {args.lr} "
        f"× {len(args.seeds)} seeds  ",
        f"**Wall time:** {total_wall_min:.2f} min  ",
        f"**Time budget:** {args.time_budget_s}s  ",
        "",
        "## Per-split AUC (mean ± std across seeds)",
        "",
        "| Split | AUC mean | AUC std | n seeds |",
        "|---|---|---|---|",
    ]
    for sp in SPLITS:
        m = per_split_mean[sp]
        s = per_split_std[sp]
        n = n_total
        md.append(f"| {sp} | {m:.4f} | {s:.4f} | {n} |")
    md += [
        "",
        "## Per-seed OOD test AUC",
        "",
        "| seed | train | iid_val | iid_test | ood_val | ood_test | epochs | elapsed_s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        md.append(
            f"| {r.seed} | {r.train_auc:.4f} | {r.iid_val_auc:.4f} | "
            f"{r.iid_test_auc:.4f} | {r.ood_val_auc:.4f} | {r.ood_test_auc:.4f} | "
            f"{r.epochs_done} | {r.elapsed_s:.1f} |"
        )
    md += [
        "",
        "## Honest framing",
        "",
        honest_framing_note,
        "",
        "## Files",
        "",
        "* `summary.json` — full metrics + per-seed",
        "* `per_seed/seed<n>.json` — per-seed detail",
        "* `cache/<hash>.npz` — BD-4 MolGraphCache persistent featurization",
        "",
    ]
    (out_dir / "final.md").write_text("\n".join(md))
    _log(f"wrote {out_dir / 'final.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
