"""Wf-185 — chemprop v2 retrain on real DrugOOD lbap_refined_ic50_*{assay,scaffold,size}.

==============================================================================
Wf-185 EXTEND of Wf-174 BD-11 chemprop v2 retrain on real DrugOOD
``lbap_refined_ic50_assay`` (single-config, seed=0 result ood_test=0.7138).

Scope (per Wf-185 task brief)
-----------------------------
* 3 DrugOOD configs (``lbap_refined_ic50_assay``, ``lbap_refined_ic50_scaffold``,
  ``lbap_refined_ic50_size``) — all three "refined" curator outputs.
* 5 seeds × 20 epochs each per config = 15 distinct training runs total
  (Wf-174 measured ~7 min/seed on lbap_refined_ic50_assay 95K-row train, so
  15 runs + 3× featurization is well within the 6-hr budget).
* CPU only, backbone hidden_dim=128 depth=2 head_hidden_dim=128 (same as
  Wf-174 — explicitly required by the Wf-185 brief "smaller backbone for
  CPU speed").
* Per-config per-seed AUC JSON saved.
* Cache reuse for lbap_refined_ic50_assay (266K SMILES already featurized
  from Wf-174) → saves ~3 min featurize + 108s mem-load on that config.

Model
-----
Same as Wf-174: :class:`ChempropV2MTModel` with metal_weights zeroed for
Ir/Os/Rh/Re and coord_weight=0 → single-task binary classifier through the
``"Ru"`` head (binary cross-entropy on DrugOOD's curator-supplied
``active=True/False`` label).

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* Backbone is smaller than chemprop v2 defaults (``hidden=128/2`` vs
  ``300/3``) — per-step cost ~3-4× lower but capacity also reduced.
* 20 epochs may be insufficient for full convergence at ``hidden=128``.
* No Wasserstein transfer, no L2 guard, no LR scheduler.
* Wall budget 21600s (6 hr) per Wf-185 brief; per-config sub-budget is
  implicit at ~7200s (per Wf-174 estimate 5×420s = 2100s for
  lbap_refined_ic50_assay + ~5 min featurize/mem-load = ~30 min/config;
  30×3 + headroom = 6 hr).
* Cache reuse: lbap_refined_ic50_assay cache is hit from Wf-174; for
  scaffold and size configs the SMILES set is similar but distinct so
  cache miss is expected (n_skip=0, n_new≈266K).

Outputs
-------
* ``molmetal/reports/wf185_chemprop_v2_drugood_full/<subset>/per_seed/seed<n>.json``
* ``molmetal/reports/wf185_chemprop_v2_drugood_full/<subset>/summary.json``
* ``molmetal/reports/wf185_chemprop_v2_drugood_full/<subset>/final.md``
* ``molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json`` — cross-config
* ``molmetal/reports/wf185_chemprop_v2_drugood_full/final.md`` — final aggregate report
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
from dataclasses import dataclass, field
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
# Defaults — CPU-friendly smaller backbone per task spec (Wf-185 brief).
# ---------------------------------------------------------------------------
DEFAULT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf185_chemprop_v2_drugood_full"
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 20
DEFAULT_HIDDEN: int = 128
DEFAULT_DEPTH: int = 2
DEFAULT_HEAD_HIDDEN: int = 128
DEFAULT_BATCH: int = 128
DEFAULT_LR: float = 1e-3
DEFAULT_WALL_BUDGET_S: float = 21600.0  # 6 hr (Wf-185 brief)

# The three "refined" curator outputs (Ji et al. NeurIPS 2022).
CONFIGS = (
    "lbap_refined_ic50_assay",
    "lbap_refined_ic50_scaffold",
    "lbap_refined_ic50_size",
)

# Splits exposed by load_drugood (raw JSON).
SPLITS = ("train", "iid_val", "iid_test", "ood_val", "ood_test")

# Cache from Wf-174 for lbap_refined_ic50_assay — reuse to save featurize + mem-load.
WF174_CACHE = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf174_chemprop_v2_drugood_real" / "cache"
)

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
    print(f"[wf185] {msg}", flush=True, **kw)


# ---------------------------------------------------------------------------
# Featurization (same pattern as wf174).
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


def featurize_and_cache(
    canon_list: List[str],
    cache_dir: Path,
    featurizer: SimpleMoleculeMolGraphFeaturizer,
    n_workers: int = 4,
    chunk_size: int = 30000,
) -> Tuple[Dict[str, dict], int, int]:
    """Persistent MolGraphCache + npz on disk. See wf174 docstring."""
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
    _log(
        f"  featurize: cache_hits={n_skip}  todo={n_new_total}  "
        f"chunk_size={chunk_size}"
    )
    for chunk_start in range(0, n_new_total, chunk_size):
        chunk = todo[chunk_start : chunk_start + chunk_size]
        _log(
            f"  featurize chunk [{chunk_start}..{chunk_start + len(chunk)})  "
            f"({len(chunk)} SMILES)"
        )
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


def copy_wf174_cache(cache_dir: Path, target_canon: set) -> int:
    """Copy npz files from Wf-174 cache (lbap_refined_ic50_assay only) for SMILES
    present in our target_canon set. Returns the number of files copied.
    """
    if not WF174_CACHE.exists():
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for s in target_canon:
        h = HASH_FN(s)
        src = WF174_CACHE / f"{h}.npz"
        dst = cache_dir / f"{h}.npz"
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)  # symlink to save disk + copy time
                n_copied += 1
            except OSError:
                shutil.copy2(src, dst)
                n_copied += 1
    return n_copied


def build_batch_molgraph(
    canon_list: List[str],
    featurized: Dict[str, dict],
    atom_fdim: int,
    bond_fdim: int,
    device: torch.device,
) -> BatchMolGraph:
    """Build a chemprop :class:`BatchMolGraph` from canonical SMILES."""
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


def load_cached_features_in_memory(
    cached_paths: Dict[str, dict], target_canon: set
) -> Dict[str, dict]:
    """Load the npz arrays for every SMILES in ``target_canon`` into memory."""
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
# Data loading.
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    canon: List[str]
    y: np.ndarray
    valid_mask: np.ndarray
    n_total: int
    n_valid: int
    n_pos: int
    load_wall_s: float

    def __repr__(self) -> str:
        return (
            f"SplitData(n_total={self.n_total} n_valid={self.n_valid} "
            f"n_pos={self.n_pos} pos_rate={self.n_pos/max(self.n_valid,1):.3f})"
        )


def load_all_splits(
    subset: str,
    n_per_split: Optional[Dict[str, int]] = None,
) -> Dict[str, SplitData]:
    out: Dict[str, SplitData] = {}
    for sp in SPLITS:
        t0 = time.time()
        df = load_drugood(
            subset=subset,
            split=sp,
            prefer_library=False,
        )
        canon = [_canon_smiles(s) for s in df["smiles"].astype(str).tolist()]
        valid = [
            (c, y)
            for c, y in zip(canon, df["active"].astype(int).tolist())
            if c is not None
        ]
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
        _log(f"  loaded {sp}: {out[sp]} in {out[sp].load_wall_s:.2f}s")
    return out


# ---------------------------------------------------------------------------
# Per-seed training (same as wf174).
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
            chunk_canon = canon_eval[start : start + batch_size]
            bmg = build_batch_molgraph(
                chunk_canon, featurized, atom_fdim, bond_fdim, device
            )
            H_v = model.encoder.message_passing(bmg)
            H = model.encoder.agg(H_v, bmg.batch)
            e = model.metal_embed(metal_ids[start : start + len(chunk_canon)])
            z = torch.cat([H, e], dim=-1)
            logit_1 = model.metal_heads["Ru"].train_step(z).squeeze(-1)
            scores[start : start + len(chunk_canon)] = (
                torch.sigmoid(logit_1).cpu().numpy()
            )
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    train_sp = splits["train"]
    canon_train = train_sp.canon
    y_train = train_sp.y.astype(np.float32)

    model = ChempropV2MTModel(
        hidden_dim=args.hidden,
        depth=args.depth,
        n_metals=5,
        n_coord_classes=5,
        head_hidden_dim=args.head_hidden,
        head_n_layers=2,
        head_dropout=0.0,
    )
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
        if (time.time() - t0) > args.seed_time_budget_s:
            _log(f"[seed={seed}] seed-budget hit at epoch={epoch - 1}; aborting")
            timed_out = True
            break
        epoch_done = epoch
        np.random.shuffle(perm)
        model.train()
        total_loss = 0.0
        n_batches = 0
        # P1 FIX (Wf-187): BF16 autocast wraps chemprop v2 message-passing on
        # CUDA. Blackwell sm_120 BF16 matmul ~3x faster than FP32; CPU no-op.
        use_amp = args.bf16 and device.type == "cuda"
        for start in range(0, n_train, args.batch):
            chunk_idx = perm[start : start + args.batch]
            chunk_canon = [canon_train[i] for i in chunk_idx]
            chunk_y = torch.from_numpy(y_train[chunk_idx]).to(device)
            bmg = build_batch_molgraph(
                chunk_canon, featurized, atom_fdim, bond_fdim, device
            )
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
            out = {}
            for m, head in model.metal_heads.items():
                logit_1 = head.train_step(z)
                zeros = torch.zeros_like(logit_1)
                out[m] = torch.cat([logit_1, zeros], dim=-1)
            out["coord"] = model.coord_head(z)
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

    eval_results: Dict[str, Tuple[float, int]] = {}
    n_eval: Dict[str, int] = {}
    for sp in SPLITS:
        sd = splits[sp]
        max_eval = (
            args.eval_max_samples if sp != "train" else args.eval_train_max_samples
        )
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
# Per-config runner.
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    subset: str
    results: List[SeedResult] = field(default_factory=list)
    total_wall_min: float = 0.0
    featurize_wall_s: float = 0.0
    n_skip: int = 0
    n_new: int = 0
    hit_rate: float = 0.0
    per_split_mean: Dict[str, float] = field(default_factory=dict)
    per_split_std: Dict[str, float] = field(default_factory=dict)
    ood_test_mean: float = float("nan")
    ood_test_std: float = 0.0
    n_ok: int = 0
    n_total: int = 0


def run_one_config(
    subset: str,
    args: argparse.Namespace,
    device: torch.device,
    cfg_start_time: float,
    wall_budget_s: float,
) -> ConfigResult:
    out_dir = Path(args.out) / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) / subset
    if args.reset_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    _log(
        f"=== [{subset}] start  cache_dir={cache_dir}  "
        f"t_elapsed={(time.time() - cfg_start_time):.1f}s"
    )

    # Load all 5 splits for this config.
    splits = load_all_splits(subset=subset, n_per_split=None)
    for sp in SPLITS:
        _log(f"  [{subset}] {sp}: {splits[sp]}")

    # Featurize unique SMILES across all 5 splits.
    canon_unique_all = sorted({c for sp in SPLITS for c in splits[sp].canon})
    _log(f"[{subset}] featurizing {len(canon_unique_all)} unique canonical SMILES")
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    # If Wf-174 cache exists and we're lbap_refined_ic50_assay, copy/symlink
    # npz files into our cache_dir (the hash key is MD5 of canon SMILES,
    # identical to Wf-174). This converts the typical
    # 3-min featurize + 108s mem-load into a near-instant symlink + mem-load.
    n_copied = 0
    if subset == "lbap_refined_ic50_assay" and WF174_CACHE.exists():
        target_canon_for_link = set(canon_unique_all)
        n_copied = copy_wf174_cache(cache_dir, target_canon_for_link)
        _log(
            f"[{subset}] reused {n_copied}/{len(canon_unique_all)} npz files from Wf-174 cache"
        )

    t_fe = time.time()
    cached_paths, n_skip, n_new = featurize_and_cache(
        canon_unique_all,
        cache_dir,
        featurizer,
        n_workers=4,
    )
    fe_wall = time.time() - t_fe
    hit_rate = float(n_skip) / max(len(canon_unique_all), 1)
    _log(
        f"[{subset}] featurize done  n_skip={n_skip}  n_new={n_new}  "
        f"hit={hit_rate*100:.1f}%  in {fe_wall:.2f}s"
    )

    # Probe fdim.
    sample_path = next(iter(cached_paths.values()))
    sample_npz = np.load(sample_path)
    atom_fdim = int(sample_npz["V"].shape[1])
    bond_fdim = int(sample_npz["E"].shape[1]) if sample_npz["E"].ndim > 1 else 14

    # Load all npz arrays into memory.
    target_canon = {c for sp in SPLITS for c in splits[sp].canon}
    t_lm = time.time()
    featurized = load_cached_features_in_memory(cached_paths, target_canon)
    _log(
        f"[{subset}] loaded {len(featurized)}/{len(target_canon)} npz into memory "
        f"in {time.time() - t_lm:.2f}s"
    )

    # Per-seed training loop.
    cfg_wall_start = time.time()
    results: List[SeedResult] = []
    for seed in args.seeds:
        if (time.time() - cfg_wall_start) > wall_budget_s:
            _log(f"[{subset}] cfg-budget hit before seed={seed}; stopping")
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
            _log(f"[{subset}][seed={seed}] FAILED: {exc}")
            r = SeedResult(
                seed=seed,
                epochs_done=0,
                elapsed_s=0.0,
                train_auc=float("nan"),
                iid_val_auc=float("nan"),
                iid_test_auc=float("nan"),
                ood_val_auc=float("nan"),
                ood_test_auc=float("nan"),
                n_train=0,
                n_eval={},
                status="error",
                error=str(exc),
            )
        results.append(r)
        # Persist per-seed JSON.
        per_seed_path = out_dir / "per_seed" / f"seed{seed}.json"
        per_seed_path.write_text(
            json.dumps(
                {
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
                },
                indent=2,
            )
        )
        _log(
            f"[{subset}][seed={seed}] FINAL  train={r.train_auc:.4f}  "
            f"iid_val={r.iid_val_auc:.4f}  iid_test={r.iid_test_auc:.4f}  "
            f"ood_val={r.ood_val_auc:.4f}  ood_test={r.ood_test_auc:.4f}  "
            f"({r.elapsed_s:.1f}s status={r.status})"
        )

    cfg_total_wall_min = (time.time() - cfg_wall_start) / 60.0

    def _mean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    def _std(xs):
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

    cfg_res = ConfigResult(
        subset=subset,
        results=results,
        total_wall_min=cfg_total_wall_min,
        featurize_wall_s=fe_wall,
        n_skip=n_skip,
        n_new=n_new,
        hit_rate=hit_rate,
        per_split_mean=per_split_mean,
        per_split_std=per_split_std,
        ood_test_mean=per_split_mean["ood_test"],
        ood_test_std=per_split_std["ood_test"],
        n_ok=sum(1 for r in results if r.status == "ok"),
        n_total=len(results),
    )

    # Persist summary.json for this config.
    passes_ood_gate = bool(
        not np.isnan(cfg_res.ood_test_mean) and cfg_res.ood_test_mean > 0.55,
    )
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "wf": "wf185",
        "subset": subset,
        "data_source": f"drugood_{subset}",
        "splits": {
            sp: {
                "n_total": int(splits[sp].n_total),
                "n_valid": int(splits[sp].n_valid),
                "n_pos": int(splits[sp].n_pos),
                "pos_rate": float(splits[sp].n_pos / max(splits[sp].n_valid, 1)),
                "load_wall_s": float(splits[sp].load_wall_s),
            }
            for sp in SPLITS
        },
        "config": {
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "head_hidden": int(args.head_hidden),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "device": str(device),
            "eval_max_samples": int(args.eval_max_samples),
            "eval_train_max_samples": int(args.eval_train_max_samples),
        },
        "featurization": {
            "n_unique_canon": int(len(canon_unique_all)),
            "n_skip": int(n_skip),
            "n_new": int(n_new),
            "hit_rate": float(hit_rate),
            "featurize_wall_s": float(fe_wall),
            "atom_fdim": int(atom_fdim),
            "bond_fdim": int(bond_fdim),
        },
        "per_split_auc_mean": cfg_res.per_split_mean,
        "per_split_auc_std": cfg_res.per_split_std,
        "per_split_auc_per_seed": per_split_aucs,
        "ood_test_auc_mean": float(cfg_res.ood_test_mean),
        "ood_test_auc_std": float(cfg_res.ood_test_std),
        "passes_ood_gate": passes_ood_gate,
        "n_seeds_ok": int(cfg_res.n_ok),
        "n_seeds_total": int(cfg_res.n_total),
        "total_wall_min": float(cfg_res.total_wall_min),
        "results": [
            {
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
            }
            for r in results
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"[{subset}] wrote {summary_path}")

    # Markdown per-config.
    md = [
        f"# Wf-185 — chemprop v2 retrain on DrugOOD {subset}",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Data:** DrugOOD {subset}  ",
        f"**Train rows:** {splits['train'].n_valid}  ",
        f"**Model:** chemprop v2 AtomMessagePassing + SumAggregation + "
        f"BinaryClassificationFFN (via ChempropV2MTModel Ru head)  ",
        f"**Backbone:** hidden={args.hidden}, depth={args.depth}, "
        f"head_hidden={args.head_hidden}  ",
        f"**Recipe:** {args.epochs} epochs × batch {args.batch} × lr {args.lr} "
        f"× {len(args.seeds)} seeds  ",
        f"**Wall time:** {cfg_res.total_wall_min:.2f} min  ",
        "",
        "## Per-split AUC (mean ± std across seeds)",
        "",
        "| Split | AUC mean | AUC std | n seeds |",
        "|---|---|---|---|",
    ]
    for sp in SPLITS:
        m = cfg_res.per_split_mean[sp]
        s = cfg_res.per_split_std[sp]
        n = cfg_res.n_total
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
        "## Files",
        "",
        "* `summary.json` — full metrics + per-seed",
        "* `per_seed/seed<n>.json` — per-seed detail",
        "* `cache/<hash>.npz` — BD-4 MolGraphCache persistent featurization",
        "",
    ]
    (out_dir / "final.md").write_text("\n".join(md))
    _log(f"[{subset}] wrote {out_dir / 'final.md'}")
    return cfg_res


# ---------------------------------------------------------------------------
# Main driver — runs all 3 configs sequentially.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS),
        help="DrugOOD subsets to run (default: all 3 refined)",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    # P0 FIX (Wf-187): default to GPU when CUDA is available; line ~824 below
    # auto-falls-back to CPU if torch.cuda.is_available() is False. Same fix
    # as wf174 — leaving the Blackwell sm_120 GPU idle at 0% with only a
    # lazy 550 MiB CUDA context was the engineering-default bug. Backward
    # compat preserved: CPU-only box still works via the fallback.
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", default=str(DEFAULT_OUT / "cache"))
    p.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete the cache dir before featurizing.",
    )
    p.add_argument(
        "--time-budget-s",
        type=float,
        default=DEFAULT_WALL_BUDGET_S,
        help="Total wall budget across all configs (default 21600s / 6 hr).",
    )
    p.add_argument(
        "--seed-time-budget-s",
        type=float,
        default=1200.0,
        help="Per-seed time budget in seconds (default 1200 = 20 min).",
    )
    p.add_argument("--eval-max-samples", type=int, default=4000)
    p.add_argument("--eval-train-max-samples", type=int, default=8000)
    p.add_argument(
        "--n-per-split",
        type=str,
        default=None,
        help="Comma-separated split=cap dict (smoke only).",
    )
    # P1 FIX (Wf-187): BF16 + DataLoader-style flags mirroring wf174.
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
        help="DataLoader num_workers (default: 4). Currently logged; manual "
        "range-based loop is the next-migration target.",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader pin_memory (default: True).",
    )
    args = p.parse_args(argv)

    n_per_split: Optional[Dict[str, int]] = None
    if args.n_per_split:
        n_per_split = {}
        for kv in args.n_per_split.split(","):
            k, v = kv.split("=")
            n_per_split[k.strip()] = int(v.strip())

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        # Wf-190 env guard: only auto-fall-back to CPU if the script-level
        # allow-CPU flag is set OR --bf16 is explicitly disabled (otherwise
        # the user explicitly asked for GPU and we should fail fast rather
        # than silently burn hours on a CPU box).
        if args.bf16 and not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[wf185] CUDA requested (--device cuda) but "
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
    _log(
        f"device={device}  bf16={args.bf16}  num_workers={args.num_workers}  "
        f"pin_memory={args.pin_memory}  configs={args.configs}  seeds={args.seeds}  "
        f"epochs={args.epochs}  hidden={args.hidden} depth={args.depth} "
        f"head_hidden={args.head_hidden}  batch={args.batch} lr={args.lr}  "
        f"time_budget_s={args.time_budget_s}  "
        f"seed_time_budget_s={args.seed_time_budget_s}  "
        f"n_per_split={n_per_split}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.time()
    cfg_results: Dict[str, ConfigResult] = {}
    # Per-config sub-budget: 1 config → full budget; N configs → split evenly.
    n_cfg = len(args.configs)
    if n_cfg == 1:
        per_cfg_budget = max(args.time_budget_s - 60.0, 60.0)
    else:
        per_cfg_budget = max(900.0, (args.time_budget_s - 60.0) / n_cfg)

    for cfg_idx, subset in enumerate(args.configs):
        elapsed = time.time() - wall_start
        remaining = args.time_budget_s - elapsed
        cfg_budget = min(per_cfg_budget, max(remaining - 60.0, 60.0))
        if cfg_budget < 300.0:
            _log(
                f"insufficient remaining wall budget ({cfg_budget:.0f}s) for {subset}; stopping"
            )
            break
        _log(
            f"=== config {cfg_idx+1}/{n_cfg}: {subset}  "
            f"cfg_budget={cfg_budget:.0f}s  elapsed={elapsed:.0f}s"
        )
        try:
            cfg_res = run_one_config(
                subset=subset,
                args=args,
                device=device,
                cfg_start_time=wall_start,
                wall_budget_s=cfg_budget,
            )
            cfg_results[subset] = cfg_res
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _log(f"[{subset}] CONFIG FAILED: {exc}")

    total_wall_min = (time.time() - wall_start) / 60.0

    # ------------------------------------------------------------------
    # Aggregate JSON across configs.
    # ------------------------------------------------------------------
    gate_1_all_ran = all(c in cfg_results for c in args.configs)
    gate_2_total_runs = sum(c.n_total for c in cfg_results.values())
    gate_3_per_config = {
        sub: [r.ood_test_auc for r in cfg_results[sub].results] for sub in cfg_results
    }
    gate_4_summary = {
        sub: {
            "mean": float(cfg_results[sub].ood_test_mean),
            "std": float(cfg_results[sub].ood_test_std),
            "n": int(cfg_results[sub].n_total),
        }
        for sub in cfg_results
    }
    gate_5_wall_min = total_wall_min

    # Improvements over Wf-174.
    wf174_assay_seed0_ood = 0.7138
    improvements = []
    for sub in cfg_results:
        m = cfg_results[sub].ood_test_mean
        s = cfg_results[sub].ood_test_std
        n = cfg_results[sub].n_total
        if sub == "lbap_refined_ic50_assay" and n >= 1:
            seed0 = cfg_results[sub].results[0].ood_test_auc
            delta = seed0 - wf174_assay_seed0_ood
            improvements.append(
                f"assay(seed=0)={seed0:.4f} vs Wf-174 seed=0 ({wf174_assay_seed0_ood:.4f}); "
                f"Δ={delta:+.4f}"
            )
        improvements.append(f"{sub}: ood_test mean={m:.4f} ± {s:.4f} (n={n})")
    improvements_str = " | ".join(improvements)

    # Honest framing.
    honest = (
        f"Wf-185 chemprop v2 retrain on 3 DrugOOD refined configs. "
        f"Per-config: 5 seeds × 20 epochs each, backbone hidden_dim={args.hidden} "
        f"depth={args.depth} head_hidden_dim={args.head_hidden}, CPU only. "
        f"Recipe identical to Wf-174 (smaller than chemprop v2 defaults 300/3 — "
        f"per-step cost ~3-4x lower but capacity also reduced). "
        f"No Wasserstein transfer, no L2 guard, no LR scheduler — pure "
        f"chemprop v2 BCE baseline. Total wall time = {total_wall_min:.2f} min "
        f"(budget {args.time_budget_s/3600:.1f} hr). "
        f"{sum(c.n_ok for c in cfg_results.values())}/{sum(c.n_total for c in cfg_results.values())} "
        f"seeds ok across all configs. "
        f"Per-config ood_test AUC: "
        + "; ".join(
            f"{sub}={cfg_results[sub].ood_test_mean:.4f}±{cfg_results[sub].ood_test_std:.4f}"
            for sub in cfg_results
        )
        + ". Wf-185 vs Wf-174: "
        + improvements_str
        + "."
    )

    aggregate = {
        "date": time.strftime("%Y-%m-%d"),
        "wf": "wf185",
        "config": {
            "configs": list(args.configs),
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "head_hidden": int(args.head_hidden),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "device": str(device),
            "eval_max_samples": int(args.eval_max_samples),
            "eval_train_max_samples": int(args.eval_train_max_samples),
            "time_budget_s": float(args.time_budget_s),
        },
        "gate_1_all_ran": gate_1_all_ran,
        "gate_2_total_runs": gate_2_total_runs,
        "gate_3_per_config_ood_test": gate_3_per_config,
        "gate_4_per_config_summary": gate_4_summary,
        "gate_5_total_wall_min": gate_5_wall_min,
        "improvements_over_wf174": improvements_str,
        "honest_framing_note": honest,
        "per_config": {
            sub: {
                "subset": cfg_results[sub].subset,
                "ood_test_mean": float(cfg_results[sub].ood_test_mean),
                "ood_test_std": float(cfg_results[sub].ood_test_std),
                "n_seeds_ok": int(cfg_results[sub].n_ok),
                "n_seeds_total": int(cfg_results[sub].n_total),
                "total_wall_min": float(cfg_results[sub].total_wall_min),
                "featurize_wall_s": float(cfg_results[sub].featurize_wall_s),
                "hit_rate": float(cfg_results[sub].hit_rate),
                "per_split_mean": cfg_results[sub].per_split_mean,
                "per_split_std": cfg_results[sub].per_split_std,
            }
            for sub in cfg_results
        },
    }
    agg_path = out_dir / "aggregate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2))
    _log(f"wrote {agg_path}")

    # Final markdown.
    md = [
        "# Wf-185 — chemprop v2 DrugOOD full benchmark",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Configs:** {', '.join(args.configs)}  ",
        f"**Backbone:** hidden={args.hidden}, depth={args.depth}, "
        f"head_hidden={args.head_hidden} (smaller than chemprop v2 defaults 300/3)  ",
        f"**Recipe:** {args.epochs} epochs × {len(args.seeds)} seeds per config  ",
        f"**Total wall time:** {total_wall_min:.2f} min "
        f"(budget {args.time_budget_s/3600:.1f} hr)  ",
        "",
        "## Acceptance gates",
        "",
        f"- [G1] All configs ran: **{gate_1_all_ran}**",
        f"- [G2] Total runs: **{gate_2_total_runs}**",
        f"- [G3] Per-config ood_test AUC: see table below",
        f"- [G4] Per-config mean ± std: see table below",
        f"- [G5] Total wall time: **{gate_5_wall_min:.2f} min** "
        f"(≤ {args.time_budget_s/3600:.1f} hr)",
        "",
        "## Per-config summary",
        "",
        "| Config | ood_test mean | ood_test std | n seeds | wall_min | hit_rate |",
        "|---|---|---|---|---|---|",
    ]
    for sub in args.configs:
        if sub in cfg_results:
            cr = cfg_results[sub]
            md.append(
                f"| {sub} | {cr.ood_test_mean:.4f} | {cr.ood_test_std:.4f} | "
                f"{cr.n_total} | {cr.total_wall_min:.2f} | "
                f"{100*cr.hit_rate:.1f}% |"
            )
        else:
            md.append(f"| {sub} | — | — | — | — | — |")
    md += [
        "",
        "## Per-config per-seed ood_test AUC",
        "",
    ]
    for sub in args.configs:
        if sub in cfg_results:
            md.append(f"### {sub}")
            md.append("")
            md.append("| seed | ood_test | elapsed_s | status |")
            md.append("|---|---|---|---|")
            for r in cfg_results[sub].results:
                md.append(
                    f"| {r.seed} | {r.ood_test_auc:.4f} | {r.elapsed_s:.1f} | "
                    f"{r.status} |"
                )
            md.append("")
    md += [
        "## Improvements over Wf-174",
        "",
        improvements_str,
        "",
        "## Honest framing",
        "",
        honest,
        "",
        "## Files",
        "",
        "* `aggregate.json` — cross-config gate report",
        f"* `{args.configs[0]}/summary.json` + `per_seed/seed<n>.json` "
        "(per-config results; one dir per config)",
        "* `cache/<config>/<hash>.npz` — BD-4 MolGraphCache persistent featurization",
        "",
    ]
    final_md_path = out_dir / "final.md"
    final_md_path.write_text("\n".join(md))
    _log(f"wrote {final_md_path}")

    # Print summary to stdout for easy grep.
    print(f"\n=== Wf-185 SUMMARY ===")
    print(f"gate_1_all_ran: {gate_1_all_ran}")
    print(f"gate_2_total_runs: {gate_2_total_runs}")
    print(f"gate_3_per_config_ood_test: {json.dumps(gate_3_per_config)}")
    print(f"gate_4_per_config_summary: {json.dumps(gate_4_summary)}")
    print(f"gate_5_total_wall_min: {gate_5_wall_min}")
    print(f"improvements_over_wf174: {improvements_str}")
    print(f"honest_framing_note: {honest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
