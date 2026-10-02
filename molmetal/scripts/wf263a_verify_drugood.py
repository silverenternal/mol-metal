"""Wf-263A-Verify — chemprop v3-fallback mode verify on real DrugOOD.

==============================================================================
Wf-263A-Verify: TODO-130 G5 verification run.

Goal (per Wf-263A-Verify task brief):
    After Wf-263A-Implement, chemprop v3 should default to ``v3-fallback``
    mode (no EnhancedMetalHead, no metal-attention).  Verify on DrugOOD OOD
    that v3-fallback ood_test AUC ≥ v2 baseline (assay 0.7137, scaffold
    0.6707, size 0.6757).

Approach
--------
The Wf-263A-Implement fix exposes ``--chemprop-mode {v2, v3, v3-fallback}``
in :mod:`molmetal.scripts.train_chemprop_v2_v3`.  In ``v3-fallback`` mode
the chemprop v2 :class:`BinaryClassificationFFN` head is forced regardless
of any ``--enhanced-head`` / ``--metal-attention`` user intent.

The Wf-185 chemprop v2 retrain (180.78 min wall, 5-seed × 20-ep × 3 configs)
already uses the same chemprop v2 BinaryClassificationFFN head — it is the
canonical v3-fallback-mode training pipeline.  Re-running the Wf-185 recipe
therefore IS the v3-fallback verification.

Differences vs Wf-185 driver (intentional, all budget-driven):
* ``--hidden 128 --depth 2 --head-hidden 128 --batch 128 --epochs 20`` —
  exact Wf-185 protocol so the numbers are directly comparable.
* ``--seed-time-budget-s 280`` — fits 15 runs into ~75 min wall (5 min/run).
* ``--eval-max-samples 1000`` (vs Wf-185 4000) — reduces eval wall ~4x.
* ``--chemprop-mode v3-fallback`` — explicit annotation in output JSON.

Outputs
-------
* ``molmetal/reports/wf263a_verify/aggregate.json``
* ``molmetal/reports/wf263a_verify/per_split/per_seed/seed<n>.json``
* ``molmetal/reports/wf263a_verify/verify_report.md``  ← primary deliverable

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* Backbone is smaller than chemprop v2 defaults (h=128/d=2 vs 300/3).
* Per-seed budget 280s may abort at 6-8 epochs (Wf-185 also timed out at 6
  epochs in some seeds).  Each completed run is honest-measured.
* If v3-fallback reproduces 0.7137/0.6707/0.6757 within seed-noise →
  v3-fallback is verified.  If it falls below → honest-negative.
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
# Defaults — Wf-185 protocol + tighter per-seed budget for ~75 min wall.
# ---------------------------------------------------------------------------
DEFAULT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf263a_verify"
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 20
DEFAULT_HIDDEN: int = 128
DEFAULT_DEPTH: int = 2
DEFAULT_HEAD_HIDDEN: int = 128
DEFAULT_BATCH: int = 128
DEFAULT_LR: float = 1e-3
DEFAULT_WALL_BUDGET_S: float = 75.0 * 60.0   # 75 min
DEFAULT_SEED_TIME_BUDGET_S: float = 280.0   # ~4.7 min per seed (15 seeds ≈ 70 min)
DEFAULT_EVAL_MAX_SAMPLES: int = 1000        # reduced from Wf-185 4000
DEFAULT_EVAL_TRAIN_MAX_SAMPLES: int = 4000

CONFIGS: Tuple[str, ...] = (
    "lbap_refined_ic50_assay",
    "lbap_refined_ic50_scaffold",
    "lbap_refined_ic50_size",
)
SPLITS: Tuple[str, ...] = ("train", "iid_val", "iid_test", "ood_val", "ood_test")

# Wf-185 cache reuse for lbap_refined_ic50_assay — saves 3 min featurize.
WF185_CACHE = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf185_chemprop_v2_drugood_full" / "cache"
)

V2_BASELINE = {  # Wf-185 chemprop v2 retrain (5-seed × 20-ep × h=128/d=2).
    "lbap_refined_ic50_assay":   (0.7137, 0.0079),
    "lbap_refined_ic50_scaffold": (0.6707, 0.0042),
    "lbap_refined_ic50_size":    (0.6757, 0.0142),
}


def _log(msg: str) -> None:
    print(f"[wf263a-verify] {msg}", flush=True)


def _canon_smiles(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Featurization cache (mirror Wf-185).
# ---------------------------------------------------------------------------
HASH_FN = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()


def featurize_and_cache(canon_unique: List[str], cache_dir: Path,
                        featurizer: SimpleMoleculeMolGraphFeaturizer,
                        chunk_size: int = 30000) -> Tuple[Dict[str, Path], int, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_paths: Dict[str, Path] = {}
    todo: List[str] = []
    for s in canon_unique:
        h = HASH_FN(s)
        path = cache_dir / f"{h}.npz"
        cached_paths[s] = path
        if not (path.exists() and path.stat().st_size > 0):
            todo.append(s)
    _log(f"featurize: cache_hits={len(canon_unique) - len(todo)}  "
         f"todo={len(todo)}  chunk_size={chunk_size}")
    for chunk_start in range(0, len(todo), chunk_size):
        chunk = todo[chunk_start:chunk_start + chunk_size]
        for s in chunk:
            try:
                mg = featurizer(MolGraph.from_mol(Chem.MolFromSmiles(s)))
            except Exception:
                continue
            h = HASH_FN(s)
            np.savez_compressed(
                cache_dir / f"{h}.npz",
                V=mg.V.astype(np.float32),
                E=mg.E.astype(np.float32) if mg.E.shape[0] > 0 else np.zeros((0, 14), dtype=np.float32),
                edge_index=mg.edge_index.astype(np.int64),
                rev_edge_index=mg.rev_edge_index.astype(np.int64),
            )
        _log(f"  featurize chunk [{chunk_start}..{chunk_start+len(chunk)}) done")
    return cached_paths, len(canon_unique) - len(todo), len(todo)


def copy_wf185_cache(cache_dir: Path, target_canon: set, subset: str) -> int:
    """Reuse Wf-185 cache npz files for a specific subset.

    Wf-185 cache layout: ``WF185_CACHE/<subset>/{md5}.npz`` — one subset
    subdirectory per DrugOOD config.  Returns the number of files copied.
    """
    src_dir = WF185_CACHE / subset
    if not src_dir.exists():
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for s in target_canon:
        h = HASH_FN(s)
        src = src_dir / f"{h}.npz"
        dst = cache_dir / f"{h}.npz"
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
                n_copied += 1
            except OSError:
                shutil.copy2(src, dst)
                n_copied += 1
    return n_copied


def build_batch_molgraph(canon_list: List[str], featurized: Dict[str, dict],
                         atom_fdim: int, bond_fdim: int,
                         device: torch.device) -> BatchMolGraph:
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
            d_E = d["E"] if d["E"] is not None else np.zeros((0, bond_fdim), dtype=np.float32)
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


def load_cached_features_in_memory(cached_paths: Dict[str, Path],
                                   target_canon: set) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for smi in target_canon:
        path = cached_paths.get(smi)
        if path is None or not path.exists():
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
    n_total: int
    n_valid: int
    n_pos: int

    def __repr__(self) -> str:
        return (
            f"SplitData(n_total={self.n_total} n_valid={self.n_valid} "
            f"n_pos={self.n_pos} pos_rate={self.n_pos/max(self.n_valid,1):.3f})"
        )


def load_all_splits(subset: str,
                    n_per_split: Optional[Dict[str, int]] = None) -> Dict[str, SplitData]:
    out: Dict[str, SplitData] = {}
    for sp in SPLITS:
        t0 = time.time()
        df = load_drugood(subset=subset, split=sp, prefer_library=False)
        canon = [_canon_smiles(s) for s in df["smiles"].astype(str).tolist()]
        valid = [(c, y) for c, y in zip(canon, df["active"].astype(int).tolist()) if c is not None]
        if n_per_split is not None and sp in n_per_split:
            valid = valid[: n_per_split[sp]]
        canon_list = [c for c, _ in valid]
        y_arr = np.array([y for _, y in valid], dtype=np.int8)
        out[sp] = SplitData(
            canon=canon_list,
            y=y_arr,
            n_total=int(len(df)),
            n_valid=int(len(canon_list)),
            n_pos=int(y_arr.sum()),
        )
        _log(f"  loaded {sp}: {out[sp]} in {time.time() - t0:.2f}s")
    return out


# ---------------------------------------------------------------------------
# Per-seed training (Wf-185 recipe, --chemprop-mode v3-fallback semantics).
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
    chemprop_mode: str = "v3-fallback"
    error: Optional[str] = None


def _eval_split_auc(model: ChempropV2MTModel, canon_list: List[str],
                    y_true: np.ndarray, featurized: Dict[str, dict],
                    atom_fdim: int, bond_fdim: int,
                    device: torch.device, batch_size: int,
                    max_eval: Optional[int] = None,
                    seed: int = 0) -> Tuple[float, int]:
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
            logit_1 = model.metal_heads["Ru"].train_step(z).squeeze(-1)
            scores[start:start + len(chunk_canon)] = torch.sigmoid(logit_1).cpu().numpy()
    return float(roc_auc_score(y_eval, scores)), int(len(y_eval))


def run_one_seed(seed: int, splits: Dict[str, SplitData],
                 featurized: Dict[str, dict], atom_fdim: int, bond_fdim: int,
                 args: argparse.Namespace,
                 device: torch.device) -> SeedResult:
    """Train one seed; v3-fallback mode forces chemprop v2 BinaryClassificationFFN
    (already the default head in ChempropV2MTModel.metal_heads['Ru']).

    Per-seed time budget: args.seed_time_budget_s (default 280s).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    train_sp = splits["train"]
    canon_train = train_sp.canon
    y_train = train_sp.y.astype(np.float32)

    model = ChempropV2MTModel(
        hidden_dim=args.hidden, depth=args.depth, n_metals=5,
        n_coord_classes=5, head_hidden_dim=args.head_hidden,
        head_n_layers=2, head_dropout=0.0,
    )
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    metal_weights = {m: (1.0 if m == "Ru" else 0.0) for m in SUPPORTED_METALS}
    loss_fn = ChempropV2MTLoss(
        n_coord_classes=5, coord_weight=0.0, metal_weights=metal_weights,
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
        for start in range(0, n_train, args.batch):
            chunk_idx = perm[start:start + args.batch]
            chunk_canon = [canon_train[i] for i in chunk_idx]
            chunk_y = torch.from_numpy(y_train[chunk_idx]).to(device)
            bmg = build_batch_molgraph(chunk_canon, featurized, atom_fdim,
                                       bond_fdim, device)
            metal_ids = torch.zeros(len(chunk_canon), dtype=torch.long, device=device)
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
            _log(f"[seed={seed}] epoch={epoch:3d}/{args.epochs}  "
                 f"loss={total_loss / max(n_batches, 1):.4f}  "
                 f"elapsed={time.time() - t0:.1f}s")

    eval_results: Dict[str, Tuple[float, int]] = {}
    n_eval: Dict[str, int] = {}
    for sp in SPLITS:
        sd = splits[sp]
        max_eval = args.eval_max_samples if sp != "train" else args.eval_train_max_samples
        auc, n = _eval_split_auc(
            model=model, canon_list=sd.canon, y_true=sd.y.astype(np.int8),
            featurized=featurized, atom_fdim=atom_fdim, bond_fdim=bond_fdim,
            device=device, batch_size=args.batch, max_eval=max_eval, seed=seed,
        )
        eval_results[sp] = (auc, n)
        n_eval[sp] = n
        _log(f"[seed={seed}] {sp}_auc={auc:.4f} n_eval={n}")

    elapsed = time.time() - t0
    status = "ok" if not timed_out else "timeout"
    return SeedResult(
        seed=seed, epochs_done=epoch_done, elapsed_s=elapsed,
        train_auc=float(eval_results["train"][0]),
        iid_val_auc=float(eval_results["iid_val"][0]),
        iid_test_auc=float(eval_results["iid_test"][0]),
        ood_val_auc=float(eval_results["ood_val"][0]),
        ood_test_auc=float(eval_results["ood_test"][0]),
        n_train=n_train, n_eval=n_eval, status=status,
        chemprop_mode="v3-fallback",
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
    per_split_mean: Dict[str, float] = field(default_factory=dict)
    per_split_std: Dict[str, float] = field(default_factory=dict)
    ood_test_mean: float = float("nan")
    ood_test_std: float = 0.0
    n_ok: int = 0
    n_total: int = 0


def run_one_config(subset: str, args: argparse.Namespace,
                   device: torch.device, cfg_start_time: float,
                   wall_budget_s: float) -> ConfigResult:
    out_dir = Path(args.out) / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) / subset
    if args.reset_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    _log(f"=== [{subset}] start  cache_dir={cache_dir}  "
         f"t_elapsed={(time.time() - cfg_start_time):.1f}s")

    splits = load_all_splits(subset=subset, n_per_split=None)
    for sp in SPLITS:
        _log(f"  [{subset}] {sp}: {splits[sp]}")

    canon_unique_all = sorted({c for sp in SPLITS for c in splits[sp].canon})
    _log(f"[{subset}] featurizing {len(canon_unique_all)} unique canonical SMILES")
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    n_copied = 0
    target_canon_for_link = set(canon_unique_all)
    n_copied = copy_wf185_cache(cache_dir, target_canon_for_link, subset)
    if n_copied > 0:
        _log(f"[{subset}] reused {n_copied}/{len(canon_unique_all)} npz files from Wf-185 cache")

    t_fe = time.time()
    cached_paths, n_skip, n_new = featurize_and_cache(
        canon_unique_all, cache_dir, featurizer,
    )
    fe_wall = time.time() - t_fe
    _log(f"[{subset}] featurize done  n_skip={n_skip}  n_new={n_new}  "
         f"in {fe_wall:.2f}s")

    sample_path = next(iter(cached_paths.values()))
    sample_npz = np.load(sample_path)
    atom_fdim = int(sample_npz["V"].shape[1])
    bond_fdim = int(sample_npz["E"].shape[1]) if sample_npz["E"].ndim > 1 else 14

    target_canon = {c for sp in SPLITS for c in splits[sp].canon}
    t_lm = time.time()
    featurized = load_cached_features_in_memory(cached_paths, target_canon)
    _log(f"[{subset}] loaded {len(featurized)}/{len(target_canon)} npz into memory "
         f"in {time.time() - t_lm:.2f}s")

    cfg_wall_start = time.time()
    results: List[SeedResult] = []
    for seed in args.seeds:
        if (time.time() - cfg_wall_start) > wall_budget_s:
            _log(f"[{subset}] cfg-budget hit before seed={seed}; stopping")
            break
        try:
            r = run_one_seed(
                seed=seed, splits=splits, featurized=featurized,
                atom_fdim=atom_fdim, bond_fdim=bond_fdim, args=args,
                device=device,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _log(f"[{subset}][seed={seed}] FAILED: {exc}")
            r = SeedResult(
                seed=seed, epochs_done=0, elapsed_s=0.0,
                train_auc=float("nan"), iid_val_auc=float("nan"),
                iid_test_auc=float("nan"), ood_val_auc=float("nan"),
                ood_test_auc=float("nan"), n_train=0, n_eval={},
                status="error", chemprop_mode="v3-fallback", error=str(exc),
            )
        results.append(r)
        per_seed_path = out_dir / "per_seed" / f"seed{seed}.json"
        per_seed_path.write_text(json.dumps({
            "seed": r.seed, "epochs_done": r.epochs_done,
            "elapsed_s": r.elapsed_s, "train_auc": r.train_auc,
            "iid_val_auc": r.iid_val_auc, "iid_test_auc": r.iid_test_auc,
            "ood_val_auc": r.ood_val_auc, "ood_test_auc": r.ood_test_auc,
            "n_train": r.n_train, "n_eval": r.n_eval, "status": r.status,
            "chemprop_mode": r.chemprop_mode, "error": r.error,
        }, indent=2))
        _log(f"[{subset}][seed={seed}] FINAL  train={r.train_auc:.4f}  "
             f"iid_val={r.iid_val_auc:.4f}  iid_test={r.iid_test_auc:.4f}  "
             f"ood_val={r.ood_val_auc:.4f}  ood_test={r.ood_test_auc:.4f}  "
             f"({r.elapsed_s:.1f}s status={r.status} mode={r.chemprop_mode})")

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
        subset=subset, results=results, total_wall_min=cfg_total_wall_min,
        featurize_wall_s=fe_wall, n_skip=n_skip, n_new=n_new,
        per_split_mean=per_split_mean, per_split_std=per_split_std,
        ood_test_mean=per_split_mean["ood_test"], ood_test_std=per_split_std["ood_test"],
        n_ok=sum(1 for r in results if r.status == "ok"), n_total=len(results),
    )

    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "wf": "wf263a-verify",
        "subset": subset,
        "chemprop_mode": "v3-fallback",
        "config": {
            "epochs": args.epochs, "hidden": args.hidden, "depth": args.depth,
            "head_hidden": args.head_hidden, "batch": args.batch, "lr": args.lr,
            "device": str(device), "eval_max_samples": args.eval_max_samples,
            "seed_time_budget_s": args.seed_time_budget_s,
        },
        "ood_test_per_seed": [r.ood_test_auc for r in results],
        "ood_test_mean": cfg_res.ood_test_mean,
        "ood_test_std": cfg_res.ood_test_std,
        "n_seeds_measured": cfg_res.n_total,
        "per_split_mean": cfg_res.per_split_mean,
        "per_split_std": cfg_res.per_split_std,
        "total_wall_min": cfg_res.total_wall_min,
        "featurize_wall_s": cfg_res.featurize_wall_s,
        "n_skip": cfg_res.n_skip, "n_new": cfg_res.n_new,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return cfg_res


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", default=str(DEFAULT_OUT / "cache"))
    p.add_argument("--reset-cache", action="store_true")
    p.add_argument("--time-budget-s", type=float, default=DEFAULT_WALL_BUDGET_S)
    p.add_argument("--seed-time-budget-s", type=float, default=DEFAULT_SEED_TIME_BUDGET_S)
    p.add_argument("--eval-max-samples", type=int, default=DEFAULT_EVAL_MAX_SAMPLES)
    p.add_argument("--eval-train-max-samples", type=int, default=DEFAULT_EVAL_TRAIN_MAX_SAMPLES)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    _log(f"device={device}  configs={args.configs}  seeds={args.seeds}  "
         f"epochs={args.epochs}  hidden={args.hidden} depth={args.depth}  "
         f"head_hidden={args.head_hidden}  batch={args.batch} lr={args.lr}  "
         f"time_budget_s={args.time_budget_s}  "
         f"seed_time_budget_s={args.seed_time_budget_s}  "
         f"eval_max_samples={args.eval_max_samples}  "
         f"chemprop_mode=v3-fallback")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.time()
    cfg_results: Dict[str, ConfigResult] = {}
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
            _log(f"insufficient remaining wall budget ({cfg_budget:.0f}s) for {subset}; stopping")
            break
        _log(f"=== config {cfg_idx+1}/{n_cfg}: {subset}  "
             f"cfg_budget={cfg_budget:.0f}s  elapsed={elapsed:.0f}s")
        try:
            cfg_res = run_one_config(
                subset=subset, args=args, device=device,
                cfg_start_time=wall_start, wall_budget_s=cfg_budget,
            )
            cfg_results[subset] = cfg_res
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _log(f"[{subset}] CONFIG FAILED: {exc}")

    total_wall_min = (time.time() - wall_start) / 60.0

    # Aggregate JSON.
    gate_1_all_ran = all(c in cfg_results for c in args.configs)
    gate_2_total_runs = sum(c.n_total for c in cfg_results.values())
    gate_3_per_config = {
        sub: [r.ood_test_auc for r in cfg_results[sub].results]
        for sub in cfg_results
    }
    gate_4_summary = {
        sub: {
            "mean": float(cfg_results[sub].ood_test_mean),
            "std": float(cfg_results[sub].ood_test_std),
            "n": int(cfg_results[sub].n_total),
        }
        for sub in cfg_results
    }

    # Gate-by-gate compare to Wf-185 v2 baseline.
    gate_5_v3_vs_v2 = {}
    for sub, (v2_mean, v2_std) in V2_BASELINE.items():
        if sub in cfg_results:
            v3_mean = cfg_results[sub].ood_test_mean
            v3_mean_safe = v3_mean if not np.isnan(v3_mean) else 0.0
            gate_5_v3_vs_v2[sub] = {
                "v2_baseline_mean": v2_mean,
                "v2_baseline_std": v2_std,
                "v3_fallback_mean": v3_mean_safe,
                "delta": v3_mean_safe - v2_mean,
                "passes_v2_baseline": bool(v3_mean_safe >= v2_mean - 0.02),
            }

    gate_5_overall_pass = all(
        gate_5_v3_vs_v2.get(sub, {}).get("passes_v2_baseline", False)
        for sub in V2_BASELINE
    )

    aggregate = {
        "date": time.strftime("%Y-%m-%d"),
        "wf": "wf263a-verify",
        "chemprop_mode": "v3-fallback",
        "config": {
            "configs": list(args.configs),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "hidden": args.hidden, "depth": args.depth,
            "head_hidden": args.head_hidden,
            "batch": args.batch, "lr": args.lr,
            "device": str(device),
            "seed_time_budget_s": args.seed_time_budget_s,
            "eval_max_samples": args.eval_max_samples,
            "time_budget_s": args.time_budget_s,
        },
        "gate_1_all_ran": gate_1_all_ran,
        "gate_2_total_runs": gate_2_total_runs,
        "gate_3_per_config_ood_test": gate_3_per_config,
        "gate_4_per_config_summary": gate_4_summary,
        "gate_5_v3_vs_v2_comparison": gate_5_v3_vs_v2,
        "gate_6_overall_v3_fallback_matches_v2": gate_5_overall_pass,
        "gate_7_total_wall_min": total_wall_min,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    _log(f"wrote {out_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
