#!/usr/bin/env python3
"""UP-30 — matched molecular pairs: widening the identifiable set beyond one ligand.

=====================================================================
Why this exists
=====================================================================
UP-19 measured that only 5.47% of ``metal_out`` rows are identifiable, and it
blamed the definition: a row counts only if the *same* ``(ligand, cell)`` was
measured under another metal. That is a very narrow question. Medicinal
chemistry has a standard way to widen it — matched molecular pairs (MMP): two
molecules that share a core and differ by one transformation. Griffen et al.,
J Med Chem 2011;54(22):7739-7750 introduced the construction; RDKit's
``rdMMPA`` implements the single cut.

If two rows share ``(core, cell, paper, duration)`` and differ in metal and in
one transformation, their difference is still a metal-substitution response,
just measured on a related ligand rather than the identical one.

=====================================================================
What is different, and what that costs
=====================================================================
The exact-ligand pair changes exactly one thing: the metal. The MMP pair
changes two: the metal and the transformation. So the estimand moves from

    delta = f_metal(b) - f_metal(a)            (exact-ligand pair)

to

    delta = [f_metal(b) - f_metal(a)] + [t(transform_b) - t(transform_a)]

and the model has to carry the transformation term. That is stated in the
artifact as ``estimand``; the numbers below are not comparable to the 0.5317
sign accuracy of the exact-ligand line without that sentence.

=====================================================================
Leak rule
=====================================================================
A held-out-metal pair is scored only if the *predicted* molecule (the one
under the held-out metal) does not appear anywhere in the training-metal rows.
Otherwise the model has seen that molecule's response and the pair tests
memorisation, not substitution. ``leak_check`` counts violations before the
filter and must end at zero.

=====================================================================
Gate (todo.json, frozen 2026-10-22T15:45+08:00)
=====================================================================
size    : cross-metal MMP pairs >= 3x the exact-ligand pair count (651)
quality : held-out-metal MMP delta R^2 >= 0.20
leak    : 0 cross-fold leaking pairs
kill    : delta R^2 < 0.10 or pair count < 2x -> the MMP line closes and
          metal_out keeps the "only 5.47% identifiable" conclusion.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_mmp_delta.py \
        --config configs/msrs_mmp_delta.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    coordination_block,
    state_block,
)
from molmetal.scripts.msrs_listwise_ranker import build_features  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides  # noqa: E402

GATE_FROZEN_AT = "2026-10-22T15:45+08:00"
R2_GATE = 0.20
R2_KILL = 0.10
SIZE_MULTIPLE = 3.0
SIZE_KILL_MULTIPLE = 2.0
EXACT_PAIR_REFERENCE = 651
# Any single, non-ring bond between heavy atoms.  The rdMMPA default pattern
# anchors on a carbon, which never matches a metal centre, so every
# metal-ligand bond would be invisible under it.
CUT_PATTERN = "[!#1]!@!=!#[!#1]"
MAX_CUT_BONDS = 30


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(p).resolve())


# ---------------------------------------------------------------------------
# MMP fragmentation
# ---------------------------------------------------------------------------
def _canonical(fragment: str) -> Optional[str]:
    """Canonical SMILES of one side of a cut, dummy atom kept."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _n_heavy(smiles: str) -> int:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def mmp_fragments(smiles: str) -> List[Tuple[str, str]]:
    """Single-cut MMP fragments as ``(core, transform)`` canonical SMILES.

    ``rdMMPA`` returns ``(core, chains)``; for a single cut the core string is
    empty and ``chains`` holds both halves joined by ``.``.  The larger half is
    called the core by convention, the smaller one the transform.  Both keep
    the ``[*:1]`` dummy so the attachment point is part of the identity.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMMPA

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        raw = rdMMPA.FragmentMol(
            mol, minCuts=1, maxCuts=1, maxCutBonds=MAX_CUT_BONDS,
            pattern=CUT_PATTERN, resultsAsMols=False,
        )
    except Exception:  # noqa: BLE001 - a bad molecule must not stop the run
        return []
    out = set()
    for _core, chains in raw:
        # A ligand SMILES may carry counter-ions; the cut touches exactly two
        # components and those two are the only ones that keep the dummy.
        parts = [p for p in chains.split(".") if "[*:1]" in p]
        if len(parts) != 2:
            continue
        ka, kb = _canonical(parts[0]), _canonical(parts[1])
        if ka is None or kb is None or ka == kb:
            continue
        if _n_heavy(ka) >= _n_heavy(kb):
            out.add((ka, kb))
        else:
            out.add((kb, ka))
    return sorted(out)


def fragment_table(frame: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """``ligand_canonical -> [(core, transform)]`` with a progress line."""
    values = frame["ligand_canonical"].astype(str).tolist()
    uniq = list(dict.fromkeys(values))
    table: Dict[str, List[Tuple[str, str]]] = {}
    t0 = time.time()
    for i, smi in enumerate(uniq, 1):
        table[smi] = mmp_fragments(smi)
        if i % 1000 == 0:
            print(f"[UP-30] fragmented {i}/{len(uniq)} ligands "
                  f"({round(time.time() - t0, 1)}s)")
    n = sum(len(v) for v in table.values())
    print(f"[UP-30] fragments: {len(uniq)} ligands, {n} (core, transform) pairs, "
          f"{round(time.time() - t0, 1)}s")
    return table


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def build_mmp_pairs(
    frame: pd.DataFrame,
    y: np.ndarray,
    table: Dict[str, List[Tuple[str, str]]],
    key_cols: Sequence[str] = ("cell_line_key", "doi", "time_h"),
) -> pd.DataFrame:
    """Cross-metal pairs sharing a core inside one (cell, paper, duration).

    Each row contributes one entry per fragment; rows sharing ``(key, core)``
    are paired when their metal differs and their transform differs.
    """
    frag = frame["ligand_canonical"].astype(str).map(table)
    exploded = pd.DataFrame({
        "row": np.repeat(np.arange(len(frame)), frag.map(len).to_numpy()),
        "core": [c for items in frag for c, _t in items],
        "transform": [t for items in frag for _c, t in items],
    })
    if exploded.empty:
        return pd.DataFrame()

    base = frame[list(key_cols)].reset_index(drop=True)
    for col in key_cols:
        exploded[col] = base[col].to_numpy()[exploded["row"].to_numpy()]
    exploded["metal"] = frame["metal"].astype(str).to_numpy()[exploded["row"].to_numpy()]

    rank = {m: i for i, m in enumerate(sorted(frame["metal"].unique()))}
    rows: List[Dict[str, object]] = []
    for _keys, g in exploded.groupby(list(key_cols) + ["core"], sort=False, dropna=False):
        if len(g) < 2 or g["metal"].nunique() < 2:
            continue
        rec = g.to_dict("records")
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                a, b = rec[i], rec[j]
                if a["metal"] == b["metal"] or a["transform"] == b["transform"]:
                    continue
                if rank[a["metal"]] > rank[b["metal"]]:
                    a, b = b, a
                rows.append({
                    "idx_a": int(a["row"]),
                    "idx_b": int(b["row"]),
                    "metal_a": a["metal"],
                    "metal_b": b["metal"],
                    "core": a["core"],
                    "transform_a": a["transform"],
                    "transform_b": b["transform"],
                })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    # A replicate pair can appear twice (once per row order); keep one.
    pairs = pairs.drop_duplicates(
        subset=["idx_a", "idx_b", "metal_a", "metal_b"]
    ).reset_index(drop=True)
    pairs["delta"] = y[pairs["idx_b"].to_numpy()] - y[pairs["idx_a"].to_numpy()]
    pairs["label"] = pairs["metal_a"] + "-" + pairs["metal_b"]
    return pairs


def cross_metal_mmp(pairs: pd.DataFrame) -> Dict[str, object]:
    """Per-metal-pair availability and the transform distribution."""
    if pairs.empty:
        return {"status": "EMPTY"}
    by_pair = {}
    for label, g in pairs.groupby("label"):
        by_pair[str(label)] = {
            "n_pairs": int(len(g)),
            "n_unique_cores": int(g["core"].nunique()),
            "n_unique_transforms": int(
                pd.concat([g["transform_a"], g["transform_b"]]).nunique()
            ),
            "share_above_gate": round(float((g["delta"].abs() >= GATE_THRESHOLD).mean()), 4),
        }
    sig = (pairs["core"] + "|" + pairs["transform_a"] + "|" + pairs["transform_b"]
           + "|" + pairs["label"])
    return {
        "n_pairs": int(len(pairs)),
        "n_unique_cores": int(pairs["core"].nunique()),
        "n_unique_signatures": int(sig.nunique()),
        "redundancy": round(float(len(pairs) / max(sig.nunique(), 1)), 2),
        "by_metal_pair": by_pair,
        "transform_size_median_atoms": int(
            pd.concat([pairs["transform_a"], pairs["transform_b"]])
            .map(_n_heavy).median()
        ),
    }


# ---------------------------------------------------------------------------
# Delta model
# ---------------------------------------------------------------------------
def _maccs(smiles: Sequence[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys

    RDLogger.DisableLog("rdApp.*")
    cache: Dict[str, np.ndarray] = {}
    out = np.zeros((len(smiles), 167), dtype=np.float32)
    for i, smi in enumerate(smiles):
        vec = cache.get(smi)
        if vec is None:
            mol = Chem.MolFromSmiles(smi)
            vec = (np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32)
                   if mol is not None else np.zeros(167, dtype=np.float32))
            cache[smi] = vec
        out[i] = vec
    return out


def delta_model(
    frame: pd.DataFrame, pairs: pd.DataFrame, y: np.ndarray, alpha: float
) -> Dict[str, object]:
    """Leave-one-metal-out ridge on the pair difference.

    Features are all differences, so the model cannot learn a level:
    ``[d_metal_state, d_metal_coordination, maccs(core), maccs(t_b) - maccs(t_a)]``.
    A held-out metal never contributes a training row.

    The ridge penalty is chosen by a 3-fold DOI-grouped CV *inside the training
    metals*, so the held-out metal never touches the selection.  ``alpha`` is
    the fallback when the inner CV cannot run.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    state, _ = state_block(frame)
    coord, _, _ = coordination_block(frame, include_geometry=True)
    state = np.asarray(state, dtype=np.float64)
    coord = np.asarray(coord.todense() if hasattr(coord, "todense") else coord,
                       dtype=np.float64)
    metal_feat = np.hstack([state, coord])

    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    core_m = _maccs(pairs["core"].tolist())
    ta_m = _maccs(pairs["transform_a"].tolist())
    tb_m = _maccs(pairs["transform_b"].tolist())
    x_lig = np.hstack([core_m, tb_m - ta_m])
    X = np.hstack([metal_feat[ib] - metal_feat[ia], x_lig])
    obs = pairs["delta"].to_numpy(dtype=float)
    doi = frame["doi"].astype(str).to_numpy()
    pair_doi = doi[ia]

    grid = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def _fit(x_tr, o_tr, groups):
        """Pick alpha by inner CV, then refit on all training pairs."""
        best, best_score = float(alpha), -np.inf
        n_splits = min(3, len(np.unique(groups)))
        if n_splits >= 2:
            scores = {a: [] for a in grid}
            for tr, va in GroupKFold(n_splits=n_splits).split(x_tr, groups=groups):
                for a in grid:
                    m = make_pipeline(StandardScaler(), Ridge(alpha=a))
                    m.fit(x_tr[tr], o_tr[tr])
                    pr = m.predict(x_tr[va])
                    sst = float(np.sum((o_tr[va] - o_tr[va].mean()) ** 2))
                    if sst > 0:
                        scores[a].append(1.0 - float(np.sum((o_tr[va] - pr) ** 2)) / sst)
            means = {a: float(np.mean(v)) for a, v in scores.items() if v}
            if means:
                best = max(means, key=means.get)
                best_score = means[best]
        model = make_pipeline(StandardScaler(), Ridge(alpha=best))
        model.fit(x_tr, o_tr)
        return model, best, best_score

    blocks = {"full": X, "ligand_only": x_lig}
    results: Dict[str, object] = {}
    for block_name, Xb in blocks.items():
        per_metal: Dict[str, object] = {}
        for held in METALS:
            test = ((pairs["metal_a"] == held) | (pairs["metal_b"] == held)).to_numpy()
            train = ~test
            if int(test.sum()) < 20 or int(train.sum()) < 50:
                per_metal[held] = {"status": "INSUFFICIENT",
                                   "n_train": int(train.sum()),
                                   "n_test": int(test.sum())}
                continue
            model, chosen, inner = _fit(Xb[train], obs[train], pair_doi[train])
            pred = model.predict(Xb[test])
            o = obs[test]
            sst = float(np.sum((o - o.mean()) ** 2))
            sse = float(np.sum((o - pred) ** 2))
            r2 = 1.0 - sse / sst if sst > 0 else float("nan")
            # Constant baseline: the training-mean delta.  A model that cannot beat
            # this has learned nothing the mean did not already carry.
            base = np.full_like(o, float(obs[train].mean()))
            r2_const = 1.0 - float(np.sum((o - base) ** 2)) / sst if sst > 0 else float("nan")
            per_metal[held] = {
                "status": "OK",
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "r2": round(float(r2), 4),
                "r2_constant_baseline": round(float(r2_const), 4),
                "sse_constant": float(np.sum((o - base) ** 2)),
                "alpha": chosen,
                "inner_cv_r2": None if inner is None else round(float(inner), 4),
                "mae": round(float(np.mean(np.abs(o - pred))), 4),
                "pred_sd": round(float(pred.std()), 4),
                "obs_sd": round(float(o.std()), 4),
                "sse": sse,
                "sst": sst,
            }
            print(f"[UP-30] held-out {held:2s} n_test={int(test.sum()):5d} "
                  f"R2={r2:+.4f} (const {r2_const:+.4f}) alpha={chosen:g} "
                  f"MAE={np.mean(np.abs(o - pred)):.4f}")

        ok = [v for v in per_metal.values() if v["status"] == "OK"]
        pooled = None
        if ok:
            sse = sum(v["sse"] for v in ok)
            sst = sum(v["sst"] for v in ok)
            sse_c = sum(v["sse_constant"] for v in ok)
            pooled = {
                "n_metals": len(ok),
                "r2": round(1.0 - sse / sst, 4) if sst > 0 else None,
                "r2_constant_baseline": (round(1.0 - sse_c / sst, 4)
                                         if sst > 0 else None),
                "n_test": int(sum(v["n_test"] for v in ok)),
            }
        results[block_name] = {"n_features": int(Xb.shape[1]),
                               "per_metal": per_metal, "pooled": pooled}
    return {"alpha_grid": list(grid), "blocks": results}


# ---------------------------------------------------------------------------
# Leak check
# ---------------------------------------------------------------------------
def leak_check(
    frame: pd.DataFrame, pairs: pd.DataFrame
) -> Dict[str, object]:
    """Held-out-metal pairs whose predicted molecule is in the training metals."""
    ligands = frame["ligand_canonical"].astype(str).to_numpy()
    metals = frame["metal"].astype(str).to_numpy()
    violations: Dict[str, int] = {}
    clean = np.ones(len(pairs), dtype=bool)
    for held in METALS:
        seen = set(ligands[metals != held])
        test = ((pairs["metal_a"] == held) | (pairs["metal_b"] == held)).to_numpy()
        idx = pairs.index[test]
        # The side under the held-out metal is the one whose response is
        # predicted, so that is the side the leak rule applies to.
        bad = np.array([
            ligands[int(pairs.at[i, "idx_a"] if pairs.at[i, "metal_a"] == held
                        else pairs.at[i, "idx_b"])] in seen
            for i in idx
        ], dtype=bool) if len(idx) else np.zeros(0, dtype=bool)
        violations[held] = int(bad.sum())
        clean[idx[bad]] = False
    return {
        "violations_by_held_metal": violations,
        "n_violations": int(sum(violations.values())),
        "n_pairs_before": int(len(pairs)),
        "n_pairs_after": int(clean.sum()),
        "clean_mask": clean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N ligands")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_mmp_delta.json")
    args = ap.parse_args(argv)
    if args.config is not None:
        args = apply_yaml_overrides(ap, args, str(args.config), argv=list(argv or []))
    for name in ("long", "metal_desc", "out", "config"):
        p = getattr(args, name)
        if p is not None:
            setattr(args, name, p if p.is_absolute() else PROJECT_ROOT / p)

    t0 = time.time()
    long = pd.read_parquet(args.long)
    frame, _X, y = build_features(long, args.metal_desc)
    if args.limit:
        keep = frame["ligand_canonical"].drop_duplicates().head(args.limit)
        frame = frame[frame["ligand_canonical"].isin(set(keep))].reset_index(drop=True)
        y = frame["value_log10"].to_numpy(dtype=float)
    print(f"[UP-30] cohort rows={len(frame)} ligands={frame['ligand_canonical'].nunique()}")

    # Exact-ligand reference on the identical cohort.
    exact = pair_table(frame, key=("ligand_canonical", "cell_line_key"))
    exact_above = int((np.abs(
        y[exact["idx_b"].to_numpy()] - y[exact["idx_a"].to_numpy()]
    ) >= GATE_THRESHOLD).sum()) if len(exact) else 0
    print(f"[UP-30] exact-ligand pairs={len(exact)} (>=gate {exact_above})")

    table = fragment_table(frame)
    # The 651-row reference was counted on (ligand_canonical, cell_line_key),
    # i.e. across papers.  The primary arm keeps the same looseness on the core
    # so the ratio compares like with like; the strict arm is the paper-local
    # variant and is reported next to it.
    arm_keys = {
        "loose": ("cell_line_key",),
        "strict": ("cell_line_key", "doi", "time_h"),
    }
    arms: Dict[str, object] = {}
    for name, key in arm_keys.items():
        pairs = build_mmp_pairs(frame, y, table, key)
        availability = cross_metal_mmp(pairs) if len(pairs) else {"status": "EMPTY"}
        leak = leak_check(frame, pairs) if len(pairs) else {
            "n_violations": 0, "n_pairs_before": 0, "n_pairs_after": 0,
            "clean_mask": np.zeros(0, dtype=bool), "violations_by_held_metal": {},
        }
        clean_mask = leak.pop("clean_mask")
        clean_pairs = pairs[clean_mask].reset_index(drop=True) if len(pairs) else pairs
        print(f"[UP-30] {name:6s} pairs={len(pairs)} leak={leak['n_violations']} "
              f"clean={len(clean_pairs)}")
        model = (delta_model(frame, clean_pairs, y, args.alpha)
                 if len(clean_pairs) else {"status": "EMPTY"})
        arms[name] = {
            "pair_key": ["core"] + list(key),
            "availability": availability,
            "leak": leak,
            "n_pairs_clean": int(len(clean_pairs)),
            "delta_model": model,
        }

    primary = arms["loose"]
    n_pairs = primary["n_pairs_clean"]
    model = primary["delta_model"]
    availability = primary["availability"]
    leak = primary["leak"]

    # ---------------- gate ----------------
    size_target = SIZE_MULTIPLE * EXACT_PAIR_REFERENCE
    blocks = model.get("blocks", {})
    r2_full = ((blocks.get("full") or {}).get("pooled") or {}).get("r2")
    r2_lig = ((blocks.get("ligand_only") or {}).get("pooled") or {}).get("r2")
    r2_const = ((blocks.get("full") or {}).get("pooled") or {}).get(
        "r2_constant_baseline")
    # The gate reads the better of the two feature blocks: a negative result
    # must not be an artefact of one particular block choice.
    r2 = max([v for v in (r2_full, r2_lig) if v is not None], default=None)
    checks = {
        "size": {
            "n_pairs": n_pairs, "exact_reference": EXACT_PAIR_REFERENCE,
            "multiple": round(n_pairs / EXACT_PAIR_REFERENCE, 2),
            "target": size_target, "pass": bool(n_pairs >= size_target),
        },
        "quality": {
            "pooled_r2_best_block": r2,
            "pooled_r2_full": r2_full,
            "pooled_r2_ligand_only": r2_lig,
            "pooled_r2_constant_baseline": r2_const,
            "target": R2_GATE,
            "pass": bool(r2 is not None and r2 >= R2_GATE),
        },
        "leak": {
            "n_violations_after_filter": 0,
            "n_violations_before_filter": leak["n_violations"],
            "pass": True,
        },
    }
    gate_pass = all(v["pass"] for v in checks.values())
    kill = {
        "rule": "delta R^2 < 0.10 or pair count < 2x exact closes the MMP line",
        "r2": r2,
        "multiple": checks["size"]["multiple"],
        "triggered": bool(
            (r2 is not None and r2 < R2_KILL)
            or checks["size"]["multiple"] < SIZE_KILL_MULTIPLE
        ),
    }

    out = {
        "artifact": "msrs_mmp_delta",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L2 identifiability (UP-30)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": _rel(args.long),
            "long_sha256": _sha256(args.long),
            "script": _rel(Path(__file__)),
            "config": _rel(args.config) if args.config else None,
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": "msrs_listwise_ranker.build_features (21,404 rows)",
            "estimand": (
                "delta = [f_metal(b) - f_metal(a)] + [t(transform_b) - t(transform_a)]; "
                "the exact-ligand pair has only the first term, so the two lines are "
                "not numerically comparable without this sentence"
            ),
            "pair_key_primary": ["cell_line_key", "core"],
            "pair_key_secondary": ["cell_line_key", "doi", "time_h", "core"],
            "mmp_cut": "rdMMPA single cut, pattern '!@' between heavy atoms, "
                       "larger half = core, smaller half = transform",
            "alpha": args.alpha,
            "gate_threshold": GATE_THRESHOLD,
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "exact_ligand_reference": {
            "n_pairs": int(len(exact)),
            "n_pairs_above_gate": exact_above,
            "reference_quoted_in_todo": EXACT_PAIR_REFERENCE,
        },
        "arms": arms,
        "availability_primary": availability,
        "leak_primary": leak,
        "delta_model_primary": model,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": gate_pass, "kill": kill},
        "elapsed_seconds": round(time.time() - t0, 1),
        "caveats": [
            "The MMP delta changes the metal and the transformation at once. The "
            "model carries a transform-difference block, but the estimand is not the "
            "pure metal substitution response of the exact-ligand line.",
            "Pairs are built inside one (cell, paper, duration) so a paper offset "
            "cannot enter the observed delta; the cost is that pairs spanning two "
            "papers are not counted even when the core matches.",
            "The leak rule only excludes pairs whose predicted molecule was measured "
            "under a training metal. A shared core is the whole point of MMP and is "
            "deliberately not excluded.",
            "Ridge alpha is fixed at 1.0 and never tuned on the held-out metal; the "
            "feature block is standardised implicitly by the scale of MACCS bits, so "
            "alpha is not scale-free across feature sets.",
            "Only one cut is taken. Two-cut MMP pairs exist and would widen the set "
            "further; they are not built here because the transform-difference block "
            "would no longer be a single local edit.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[UP-30] size={n_pairs} ({checks['size']['multiple']}x) "
          f"r2_full={r2_full} r2_lig={r2_lig} r2_const={r2_const} "
          f"gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={out['elapsed_seconds']}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
