#!/usr/bin/env python3
"""FIX-M1 — put the in-house D-MPNN (with ``metal_graph``) on the MSRS surface.

Why this script exists
======================
Every ``msrs_*.py`` surface/decision script so far is LightGBM on ECFP4.  That
is a *baseline*, not the project's model.  The audit's DEF-1/DEF-6 fixes made the
representation honest; FIX-L1/L2/L3/L5 measured how far a fixed-capacity tree can
go.  This package asks the next question, on the same rows and the same folds:

    does a message-passing encoder over the *whole complex* — with the metal as a
    connected graph object — carry metal-differential signal the ECFP4 tree
    cannot represent?

Three arms, one controlled variable each (the decomposition mirrors FIX-L2):

    ligand_cell     cell line only, no metal node        (metal-blind control)
    metal_identity  + metal identity one-hot + node/edges (message injection)
    metal_state     + oxidation state / charge / d-electrons / geometry

``metal_identity`` minus ``ligand_cell`` is "the metal exists as a graph object";
``metal_state`` minus ``metal_identity`` is "the metal's state is visible".  The
L2 result (state is the lever, periodic descriptors are not) predicts the second
delta to be the larger one; if it is not, the encoder is not seeing the feature
vector and the result must be reported as an architecture finding, not a data one.

Comparability rules
===================
* Rows, target and cell-line vocabulary are identical to ``msrs_complex_features``.
* Folds are ``GroupKFold(5)`` on the same key, computed by ``folds_for``.
* The LightGBM reference (FIX-L1 ``complex`` arm) is recomputed **inside this
  script on the same folds**, so the paired test compares two models on
  identical rows rather than two reports written at different times.
* Both splits are reported (FIX-E1).  A single-split number here would repeat
  exactly the defect the audit found.

YAML-only launch (§5.11)
========================
All hyperparameters come from ``--config``.  The CLI selects *which run*:
``--config``, ``--out`` (override the config's output path), ``--limit`` (smoke
subset), ``--dry-run`` (no artifact written).  Nothing that changes what is
trained is settable on the command line.

Usage::

    # CPU smoke (400 ligands, minutes)
    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_dmpnn_surface.py \
        --config configs/msrs_dmpnn_surface_cpu0.yaml --dry-run

    # formal run on GPU0 (full cohort; >1 hr wall needs USER_OK)
    .venv/bin/python molmetal/scripts/msrs_dmpnn_surface.py \
        --config configs/msrs_dmpnn_surface_gpu0.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from molmetal.data.metal_graph import (
    METALS5,
    METAL_FEAT_DIM,
    build_metal_graph,
    metal_feature_vector,
)
from molmetal.data.metal_smiles import coordination_capacity
from molmetal.features.ligand import complex_ligand_blocks
from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN
from molmetal.scripts.msrs_code_levers import fit_arm, folds_for, native_codes
from molmetal.scripts.msrs_complex_features import ARMS as LGB_ARMS
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_surface_model import (
    _git_commit,
    _sha256,
    paired_ligand_test,
    score,
)

METAL_MODES = ("blind", "identity", "state")
DEFAULT_ARM = {
    "metal_mode": "state",
    "use_metal_node": True,
    "metal_injection": "message",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: Path) -> Dict[str, object]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")
    for section in ("data", "model", "train", "compute", "eval", "arms", "output"):
        if section not in cfg:
            raise ValueError(f"{path}: missing required section {section!r}")
    for arm, spec in cfg["arms"].items():
        mode = spec.get("metal_mode", "state")
        if mode not in METAL_MODES:
            raise ValueError(f"{path}: arm {arm!r} has metal_mode={mode!r}")
    return cfg


# ---------------------------------------------------------------------------
# Cohort (identical to msrs_complex_features)
# ---------------------------------------------------------------------------
def select_cohort(long_path: Path, endpoint: str, limit: int) -> pd.DataFrame:
    long = pd.read_parquet(long_path)
    sub = long[
        (long["endpoint"] == endpoint)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)
    if limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)
    return sub


def split_groups(frame: pd.DataFrame, split: str) -> np.ndarray:
    if split == SPLIT_FRAGMENT_GROUPED:
        return frame["ligand_main_fragment"].to_numpy()
    if split == SPLIT_CANONICAL_GROUPED:
        return frame["ligand_canonical"].to_numpy()
    raise ValueError(f"unknown split {split!r}")


# ---------------------------------------------------------------------------
# Conditioning vector
# ---------------------------------------------------------------------------
def cell_one_hot(frame: pd.DataFrame, vocabulary: Sequence[str]) -> np.ndarray:
    """(n, len(vocab)+1) one-hot; the last column is "other"."""
    index = {c: i for i, c in enumerate(vocabulary)}
    other = len(vocabulary)
    out = np.zeros((len(frame), len(vocabulary) + 1), dtype=np.float32)
    for i, value in enumerate(frame["cell_line_key"].astype(str)):
        out[i, index.get(value, other)] = 1.0
    return out


def conditioning_matrix(
    frame: pd.DataFrame, metal_mode: str, cell_oh: np.ndarray
) -> np.ndarray:
    """Metal block (per ``metal_mode``) concatenated with the cell-line one-hot.

    ``blind`` zeroes the whole metal block — the control still sees the cell line,
    so the comparison isolates the metal, not "conditioning exists at all".
    """
    n = len(frame)
    metal = np.zeros((n, METAL_FEAT_DIM), dtype=np.float32)
    if metal_mode != "blind":
        for i, row in enumerate(
            frame[["metal", "oxidation_state", "charge_complex"]].itertuples(index=False)
        ):
            metal_name = str(row.metal)
            os_ = int(row.oxidation_state)
            charge = int(row.charge_complex)
            cap = int(coordination_capacity(metal_name, os_))
            vec = metal_feature_vector(metal_name, os_, charge, cap)
            if metal_mode == "identity":
                # Keep the identity one-hot (first len(METALS5) slots), drop state.
                vec[len(METALS5) :] = 0.0
            metal[i] = vec
    return np.concatenate([metal, cell_oh], axis=1)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def graph_key(row) -> Tuple[str, str, int, int]:
    return (
        str(row.ligand_canonical),
        str(row.metal),
        int(row.oxidation_state),
        int(row.charge_complex),
    )


def build_graph_cache(
    frame: pd.DataFrame, use_metal_node: bool
) -> Dict[Tuple[str, str, int, int], Optional[Tuple]]:
    """One graph per unique (complex, metal, os, charge) — 3.5x row reuse."""
    cache: Dict[Tuple[str, str, int, int], Optional[Tuple]] = {}
    cols = ["ligand_canonical", "metal", "oxidation_state", "charge_complex"]
    for row in frame[cols].itertuples(index=False):
        key = graph_key(row)
        if key in cache:
            continue
        cache[key] = build_metal_graph(
            key[0],
            key[1],
            key[2],
            key[3],
            use_metal_node=use_metal_node,
        )
    return cache


def attach_conditioning(
    graph: Optional[Tuple], cond: np.ndarray
) -> Optional[Tuple]:
    """Replace the graph's metal vector with this arm's conditioning vector."""
    if graph is None:
        return None
    af, bf, es, ed, _mf = graph
    return (af, bf, es, ed, cond.astype(np.float32))


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
def _predict(
    model: nn.Module, graphs: Sequence[Optional[Tuple]], device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            chunk = list(graphs[start : start + batch_size])
            out.append(model.forward_graphs(chunk, device=device).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_fold(
    graphs_tr: Sequence[Optional[Tuple]],
    y_tr: np.ndarray,
    graphs_va: Sequence[Optional[Tuple]],
    y_va: np.ndarray,
    *,
    cond_dim: int,
    arm: Mapping[str, object],
    model_cfg: Mapping[str, object],
    train_cfg: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> Tuple[MetalAwareDMPNN, float]:
    """Fit one fold; early-stop on ligand-grouped inner validation MAE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MetalAwareDMPNN(
        metal_feat_dim=cond_dim,
        metal_injection=str(arm.get("metal_injection", "message")),
        hidden=int(model_cfg["hidden"]),
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    bs = int(train_cfg["batch_size"])
    rng = np.random.default_rng(seed)

    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    bad = 0
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        order = rng.permutation(len(graphs_tr))
        for start in range(0, len(order), bs):
            idx = order[start : start + bs]
            batch = [graphs_tr[i] for i in idx]
            pred = model.forward_graphs(batch, device=device)
            loss = loss_fn(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        if len(graphs_va):
            va_pred = _predict(model, graphs_va, device)
            mae = float(np.mean(np.abs(va_pred - y_va)))
            if mae < best_mae - 1e-5:
                best_mae, bad = mae, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= int(train_cfg["patience"]):
                    break

    model.load_state_dict(best_state)
    return model, best_mae


def run_arm_on_split(
    frame: pd.DataFrame,
    y: np.ndarray,
    graphs: Sequence[Optional[Tuple]],
    groups: np.ndarray,
    *,
    arm_name: str,
    arm: Mapping[str, object],
    cond_dim: int,
    cfg: Mapping[str, object],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, object]]:
    fold_id = folds_for(groups)
    seed = int(cfg["train"]["seed"])
    y_mu, y_sd = float(y.mean()), float(y.std(ddof=0)) or 1.0
    y_z = (y - y_mu) / y_sd

    oof = np.full(len(y), np.nan, dtype=np.float64)
    inner_folds: List[float] = []
    t0 = time.time()

    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = np.flatnonzero(fold_id != k)
        te = np.flatnonzero(fold_id == k)

        # Inner validation: hold out whole ligand groups from the training rows.
        rng = np.random.default_rng(seed + k)
        uniq = np.unique(groups[tr])
        n_val = max(1, int(round(len(uniq) * float(cfg["train"]["inner_val_fraction"]))))
        val_groups = set(rng.choice(uniq, size=min(n_val, max(1, len(uniq) - 1)), replace=False))
        is_val = np.array([g in val_groups for g in groups[tr]])
        fit_idx, val_idx = tr[~is_val], tr[is_val]
        if len(val_idx) == 0:
            fit_idx, val_idx = tr, tr

        model, best_mae = train_fold(
            [graphs[i] for i in fit_idx],
            y_z[fit_idx],
            [graphs[i] for i in val_idx],
            y_z[val_idx],
            cond_dim=cond_dim,
            arm=arm,
            model_cfg=cfg["model"],
            train_cfg=cfg["train"],
            device=device,
            seed=seed + k,
        )
        pred_z = _predict(model, [graphs[i] for i in te], device)
        oof[te] = pred_z * y_sd + y_mu
        inner_folds.append(round(best_mae * y_sd, 4))
        print(
            f"    {arm_name:16s} fold={k} inner_val_mae={best_mae * y_sd:.4f} "
            f"({time.time() - t0:.0f}s)"
        )

    entry = score(y, oof)
    entry["inner_val_mae"] = inner_folds
    entry["wall_s"] = round(time.time() - t0, 1)
    entry["cond_dim"] = int(cond_dim)
    return oof, entry


# ---------------------------------------------------------------------------
# Reference bars
# ---------------------------------------------------------------------------
def _best_canonical(path: Path, arm_key: Optional[str] = None) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get("splits", {}).get(SPLIT_CANONICAL_GROUPED, {})
    if not isinstance(table, dict) or not table:
        return None
    if arm_key is None:
        arm_key = payload.get("best_arm_canonical_split") or max(
            table, key=lambda a: table[a]["r2"]
        )
    if arm_key not in table:
        return None
    return {
        "arm": arm_key,
        "r2": float(table[arm_key]["r2"]),
        "path": str(path.relative_to(PROJECT_ROOT)),
    }


def reference_bars() -> Dict[str, object]:
    """The two MEASURED bars this run is judged against, read from disk.

    ``fix_l1_best`` is the literal FIX-M1 acceptance criterion; ``fix_l5_best``
    is the strongest representation the LightGBM line reached, reported so a
    narrow win over L1 is not mistaken for a win over the representation.
    """
    out: Dict[str, object] = {}
    l1 = _best_canonical(PROJECT_ROOT / "metrics/by_metric/msrs_complex_features.json")
    if l1:
        out["fix_l1_best"] = l1
    l5 = _best_canonical(PROJECT_ROOT / "metrics/by_metric/msrs_joint_representation.json")
    if l5:
        out["fix_l5_best"] = l5
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="override output path")
    parser.add_argument(
        "--limit", type=int, default=None, help="override data.limit (smoke subset)"
    )
    parser.add_argument(
        "--oof-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_dmpnn_surface" / "oof.npz",
        help="row-level out-of-fold predictions (path selection only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="do not write the artifact")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg, train_cfg, compute_cfg = cfg["data"], cfg["train"], cfg["compute"]
    limit = int(data_cfg.get("limit", 0)) if args.limit is None else int(args.limit)
    if compute_cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(compute_cfg.get("num_threads", 1)))
    device = torch.device(str(compute_cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("config asks for CUDA but torch.cuda.is_available() is False")

    long_path = PROJECT_ROOT / str(data_cfg["long"])
    frame = select_cohort(long_path, str(data_cfg["endpoint"]), limit)
    print(
        f"[m1] rows={len(frame)} ligands={frame['ligand_main_fragment'].nunique()} "
        f"complexes={frame['ligand_canonical'].nunique()} device={device}"
    )

    # Graphs are shared across arms that use the same node flag.
    t0 = time.time()
    cache_node = build_graph_cache(frame, True)
    cache_plain = build_graph_cache(frame, False)
    n_bad = sum(v is None for v in cache_node.values())
    print(
        f"[m1] graphs={len(cache_node)} (unparseable={n_bad}) "
        f"in {time.time() - t0:.1f}s"
    )

    keys = [
        (str(r.ligand_canonical), str(r.metal), int(r.oxidation_state), int(r.charge_complex))
        for r in frame[["ligand_canonical", "metal", "oxidation_state", "charge_complex"]]
        .itertuples(index=False)
    ]

    # A None graph has no representation.  Drop those rows from *both* models so
    # the paired test still compares identical rows; the drop is reported.
    if n_bad:
        keep_mask = np.array([cache_node[k] is not None for k in keys])
        frame = frame[keep_mask].reset_index(drop=True)
        keys = [k for k, ok in zip(keys, keep_mask) if ok]
        print(f"[m1] dropped {int((~keep_mask).sum())} rows with unparseable graphs")

    y = frame["value_log10"].to_numpy(dtype=float)
    vocab = (
        frame["cell_line_key"].value_counts().head(int(data_cfg["top_cell_lines"])).index.tolist()
    )
    cell_oh = cell_one_hot(frame, vocab)
    print(f"[m1] cells={len(vocab) + 1}")

    arms = cfg["arms"]
    conds = {
        name: conditioning_matrix(frame, str(spec.get("metal_mode", "state")), cell_oh)
        for name, spec in arms.items()
    }
    cond_dim = next(iter(conds.values())).shape[1]

    # LightGBM reference on identical folds (FIX-L1 `complex` arm).
    baseline_arm = str(cfg["eval"].get("baseline_arm", "complex"))
    include, _desc = LGB_ARMS[baseline_arm]
    counterions = frame["counterion"].fillna("").astype(str)
    counterion_vocab = counterions.value_counts().head(32).index.tolist()
    lig_block, _names, _slices = complex_ligand_blocks(
        frame, counterion_vocabulary=counterion_vocab, include=include
    )
    from scipy import sparse

    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    lgb_codes = native_codes(frame, vocab, "full")

    splits_out: Dict[str, object] = {}
    oof_store: Dict[Tuple[str, str], np.ndarray] = {}
    for split in cfg["eval"]["splits"]:
        split = str(split)
        groups = split_groups(frame, split)
        fold_id = folds_for(groups)
        print(f"[m1] split={split}")

        t_lgb = time.time()
        lgb_oof = fit_arm(
            lig_block, empty, empty, None, 0, False, y, fold_id, cat_codes=lgb_codes
        )
        lgb_entry = score(y, lgb_oof)
        lgb_entry["wall_s"] = round(time.time() - t_lgb, 1)
        oof_store[(split, f"lgb_{baseline_arm}")] = lgb_oof
        results: Dict[str, object] = {f"lgb_{baseline_arm}": lgb_entry}
        print(f"    lgb_{baseline_arm:11s} R2={lgb_entry['r2']:.4f} MAE={lgb_entry['mae']:.4f}")

        for arm_name, spec in arms.items():
            use_node = bool(spec.get("use_metal_node", True))
            cache = cache_node if use_node else cache_plain
            cond = conds[arm_name]
            graphs = [
                attach_conditioning(cache.get(k), cond[i]) for i, k in enumerate(keys)
            ]
            oof, entry = run_arm_on_split(
                frame,
                y,
                graphs,
                groups,
                arm_name=arm_name,
                arm=spec,
                cond_dim=cond_dim,
                cfg=cfg,
                device=device,
            )
            entry["vs_lgb"] = paired_ligand_test(frame, np.abs(y - lgb_oof), np.abs(y - oof))
            entry["delta_r2_vs_lgb"] = round(float(entry["r2"] - lgb_entry["r2"]), 4)
            oof_store[(split, arm_name)] = oof
            results[arm_name] = entry
            print(
                f"    {arm_name:16s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
                f"delta_vs_lgb={entry['delta_r2_vs_lgb']:+.4f}"
            )

        # Controlled arm-to-arm deltas (same folds, same rows).
        if "ligand_cell" in results and "metal_identity" in results:
            results["metal_identity"]["vs_ligand_cell"] = paired_ligand_test(
                frame,
                np.abs(y - oof_store[(split, "ligand_cell")]),
                np.abs(y - oof_store[(split, "metal_identity")]),
            )
            results["metal_identity"]["delta_r2_vs_ligand_cell"] = round(
                float(results["metal_identity"]["r2"] - results["ligand_cell"]["r2"]), 4
            )
        if "metal_identity" in results and "metal_state" in results:
            results["metal_state"]["vs_metal_identity"] = paired_ligand_test(
                frame,
                np.abs(y - oof_store[(split, "metal_identity")]),
                np.abs(y - oof_store[(split, "metal_state")]),
            )
            results["metal_state"]["delta_r2_vs_metal_identity"] = round(
                float(results["metal_state"]["r2"] - results["metal_identity"]["r2"]), 4
            )
        splits_out[split] = results

    best_arm = max(
        arms,
        key=lambda a: splits_out[SPLIT_CANONICAL_GROUPED][a]["r2"],
    )
    lgb_canonical = splits_out[SPLIT_CANONICAL_GROUPED][f"lgb_{baseline_arm}"]["r2"]
    best_canonical = splits_out[SPLIT_CANONICAL_GROUPED][best_arm]["r2"]
    delta = float(best_canonical - lgb_canonical)
    bars = reference_bars()
    l1_bar = (bars.get("fix_l1_best") or {}).get("r2")
    l5_bar = (bars.get("fix_l5_best") or {}).get("r2")

    payload = {
        "artifact": "msrs_dmpnn_surface",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "config": str(args.config.resolve().relative_to(PROJECT_ROOT))
            if args.config.resolve().is_relative_to(PROJECT_ROOT)
            else str(args.config),
            "long_parquet": str(long_path),
            "long_sha256": _sha256(long_path),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "rows": int(len(frame)),
            "ligands_main_fragment": int(frame["ligand_main_fragment"].nunique()),
            "complexes_canonical": int(frame["ligand_canonical"].nunique()),
            "unique_graphs": int(len(cache_node)),
            "unparseable_graphs": int(n_bad),
            "device": str(device),
            "folds": "GroupKFold(5) via folds_for; both splits reported",
            "cond_dim": int(cond_dim),
            "metal_feat_dim": int(METAL_FEAT_DIM),
            "cell_vocabulary_size": int(len(vocab) + 1),
            "model": dict(cfg["model"]),
            "train": {k: v for k, v in train_cfg.items()},
            "arms": {name: dict(spec) for name, spec in arms.items()},
            "arm_ligand_source": {name: "ligand_canonical" for name in arms},
            "baseline_arm": f"lgb_{baseline_arm}",
            "baseline_note": (
                "LightGBM reference recomputed in this script on the same folds; "
                "it sees metal and cell line as native categorical codes."
            ),
        },
        "splits": splits_out,
        "reference_bars": bars,
        "acceptance": {
            "criterion": "beat FIX-L1's LightGBM on canonical_grouped",
            "bar": "fix_l1_best",
            "bar_r2": l1_bar,
            "measured_r2": round(float(best_canonical), 4),
            "passed": None if l1_bar is None else bool(best_canonical > float(l1_bar)),
            "strong_bar_r2": l5_bar,
            "beats_strong_bar": None if l5_bar is None else bool(best_canonical > float(l5_bar)),
        },
        "best_arm_canonical_split": best_arm,
        "delta_r2_canonical_split": round(delta, 4),
        "verdict": "POSITIVE" if delta > 0 else "NEGATIVE",
        "reading": (
            f"最佳 D-MPNN 臂 {best_arm} 在 canonical 分组下 R2={best_canonical:.4f}，"
            f"同折 LightGBM({baseline_arm}) R2={lgb_canonical:.4f}，差 {delta:+.4f}。"
            "三臂共享同一批图与折：ligand_cell→metal_identity 是金属作为图对象，"
            "metal_identity→metal_state 是金属态可见。"
        ),
    }

    out = args.out or (PROJECT_ROOT / str(cfg["output"]["path"]))
    oof_path = args.oof_out
    if not args.dry_run:
        # Row-level OOF is the only way to answer the estimand question later
        # (within-cohort Δ ranking, FIX-E2/E3) without re-running the GPU job.
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "y": y,
            # Fixed-width unicode, not pandas object arrays: an object dtype
            # makes the npz unreadable with allow_pickle=False.
            "metal": np.asarray(frame["metal"].to_numpy(), dtype=str),
            "cell_line_key": np.asarray(frame["cell_line_key"].to_numpy(), dtype=str),
            "doi": np.asarray(frame["doi"].to_numpy(), dtype=str),
            "time_h": frame["time_h"].to_numpy(),
            "ligand_canonical": np.asarray(frame["ligand_canonical"].to_numpy(), dtype=str),
            "ligand_main_fragment": np.asarray(
                frame["ligand_main_fragment"].to_numpy(), dtype=str
            ),
        }
        for (split, arm), pred in oof_store.items():
            arrays[f"{split}__{arm}"] = pred
        np.savez_compressed(oof_path, **arrays)
        try:
            payload["design"]["oof_path"] = str(oof_path.relative_to(PROJECT_ROOT))
        except ValueError:
            payload["design"]["oof_path"] = str(oof_path)
        print(f"[m1] wrote {oof_path}")
    if args.dry_run:
        print(json.dumps(payload["splits"], ensure_ascii=False, indent=2)[:4000])
        print(f"[m1] dry-run: verdict={payload['verdict']} best={best_arm} delta={delta:+.4f}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, payload)
    print(f"[m1] verdict={payload['verdict']} best={best_arm} delta={delta:+.4f}")
    print(f"[m1] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
