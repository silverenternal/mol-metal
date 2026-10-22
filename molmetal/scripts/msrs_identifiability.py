#!/usr/bin/env python3
"""UP-19 — is metal_out a model failure, or an unidentifiable estimand?

=====================================================================
Why this exists
=====================================================================
metal_out R^2 has been negative for two rounds (-0.0734 descriptors). Two
rounds of modelling (UP-1 interaction arms, UP-12 two-stage residual)
treated that as a capacity problem. UP-18 showed the paired-evidence graph
is a tree, and a held-out node has no edge to the training graph, so the
held-out metal's *level* is not identified by the data at all.

If the negative R^2 is mostly a level offset, then the number that has
been optimised is not the number anyone wants. This script splits it.

=====================================================================
The decomposition
=====================================================================
For one held-out metal with test rows (y, p), R^2 as reported by
``sklearn.r2_score`` uses mean(y) as the baseline, so it is already
centred at the metal's own test mean. The error splits exactly:

    SSE_abs  = sum (y - p)^2
             = sum ((y - ybar) - (p - pbar))^2  +  n * (pbar - ybar)^2
             = SSE_centred                       +  SS_bias

    R^2_abs      = 1 - SSE_abs      / SST
    R^2_centred  = 1 - SSE_centred  / SST
    offset_penalty = R^2_centred - R^2_abs = bias^2 / Var(y)

``R^2_centred`` is what the model would score if an oracle supplied the
per-metal offset. It is therefore the part of the task that does not
depend on the unidentifiable level. ``R^2_abs`` is that part minus the
price of not knowing the level.

=====================================================================
Which rows are identifiable
=====================================================================
A test row (ligand, cell, held-out metal) is *identifiable* when the same
(ligand_canonical, cell_line_key) appears in training under another
metal. Then the ligand and cell effects are in the training data and only
the metal level differs. Rows without a partner are still scored (the
ligand features may generalise), but they test ligand extrapolation as
well as metal extrapolation, so they are reported separately.

Centring is an oracle operation: it uses the held-out labels. That is
stated in the artifact rather than hidden, because the point of the
number is to bound what is left once the unidentifiable part is removed,
not to claim a deployable score.

=====================================================================
Gate (todo.json, frozen 2026-10-22T12:42+08:00)
=====================================================================
primary   : centred R^2 on identifiable rows >= 0.20
secondary : >= 3 of 5 metals with centred R^2 > 0.10
kill      : centred R^2 < 0.10 or < 2 of 5 metals > 0.10 closes metal_out
            for this and every later round; the paper then reports
            NOT_IDENTIFIABLE only.

Re has zero identifiable rows (it shares no (ligand, cell) unit with any
other metal), so its centred R^2 is NOT_EVALUABLE. It still counts in the
denominator of the secondary gate, which makes that gate stricter.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_identifiability.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import (  # noqa: E402
    ENDPOINTS,
    _git_commit,
    _sha256,
)
from molmetal.scripts.msrs_lomo_surface import (  # noqa: E402
    DEFAULT_METAL_COLUMNS,
    assemble,
    build_blocks,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    TOP_CELL_LINES,
    fit_predict,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

REPRESENTATIONS = ("descriptors", "one_hot", "none")
PRIMARY_REP = "descriptors"
GATE_FROZEN_AT = "2026-10-22T12:42+08:00"
CENTRED_R2_MIN = 0.20
PER_METAL_MIN = 0.10
PER_METAL_COUNT = 3


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / sst


def decompose(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    """Split absolute R^2 into the identifiable part and the offset penalty."""
    n = int(len(y))
    ybar, pbar = float(y.mean()), float(p.mean())
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - ybar) ** 2))
    bias = pbar - ybar
    ss_bias = n * bias ** 2
    sse_centred = max(0.0, sse - ss_bias)
    r2_abs = 1.0 - sse / sst if sst > 0 else float("nan")
    r2_cent = 1.0 - sse_centred / sst if sst > 0 else float("nan")
    return {
        "n": n,
        "y_mean": round(ybar, 4),
        "pred_mean": round(pbar, 4),
        "bias": round(bias, 4),
        "var_y": round(sst / n, 4),
        "r2_abs": round(r2_abs, 4),
        "r2_centred": round(r2_cent, 4),
        "offset_penalty": round(r2_cent - r2_abs, 4),
        "ss_bias_share_of_sst": round(ss_bias / sst, 4) if sst > 0 else None,
        "sse": round(sse, 4),
        "sst": round(sst, 4),
        "sse_centred": round(sse_centred, 4),
    }


def identifiable_mask(
    df: pd.DataFrame, train: np.ndarray, test: np.ndarray, key: str
) -> np.ndarray:
    """Test rows whose (ligand, cell) partner is measured in training."""
    if key == "unit":
        train_keys = set(zip(df["ligand_canonical"].iloc[train], df["cell_line_key"].iloc[train]))
        return np.array([
            (l, c) in train_keys
            for l, c in zip(df["ligand_canonical"].iloc[test], df["cell_line_key"].iloc[test])
        ])
    if key == "fragment":
        train_frag = set(df["ligand_main_fragment"].iloc[train])
        return df["ligand_main_fragment"].iloc[test].isin(train_frag).to_numpy()
    raise ValueError(key)


def pool(parts: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pooled R^2 from per-metal SSE/SST, not an average of R^2."""
    out: Dict[str, object] = {"n": int(sum(p["n"] for p in parts))}
    sse = sum(p["sse"] for p in parts)
    sst = sum(p["sst"] for p in parts)
    sse_c = sum(p["sse_centred"] for p in parts)
    out["r2_abs"] = round(1.0 - sse / sst, 4) if sst > 0 else None
    out["r2_centred"] = round(1.0 - sse_c / sst, 4) if sst > 0 else None
    out["offset_penalty"] = (
        round(out["r2_centred"] - out["r2_abs"], 4)
        if out["r2_abs"] is not None else None
    )
    out["sst"] = round(sst, 4)
    out["sse"] = round(sse, 4)
    out["sse_centred"] = round(sse_c, 4)
    return out


def run_metal_out(
    df: pd.DataFrame,
    blocks: Dict[str, object],
    y: np.ndarray,
    metal_columns: Sequence[str],
) -> Dict[str, object]:
    metals = df["metal"].to_numpy()
    per_metal: Dict[str, object] = {}
    t0 = time.time()

    for held in METALS:
        train = np.where(metals != held)[0]
        test = np.where(metals == held)[0]
        if len(train) < 200 or len(test) < 20:
            per_metal[held] = {"status": "INSUFFICIENT", "n_train": int(len(train)),
                               "n_test": int(len(test))}
            continue

        preds: Dict[str, np.ndarray] = {}
        for rep in REPRESENTATIONS:
            X_tr, groups = assemble(blocks, rep, train)
            X_te, _ = assemble(blocks, rep, test)
            preds[rep] = fit_predict(X_tr, y[train], X_te, groups)

        y_test = y[test]
        id_unit = identifiable_mask(df, train, test, "unit")
        id_frag = identifiable_mask(df, train, test, "fragment")

        rows: Dict[str, object] = {}
        for rep in REPRESENTATIONS:
            p = preds[rep]
            all_rows = decompose(y_test, p)
            ident = decompose(y_test[id_unit], p[id_unit]) if id_unit.sum() >= 2 else None
            frag = decompose(y_test[id_frag], p[id_frag]) if id_frag.sum() >= 2 else None
            rows[rep] = {
                "all_rows": all_rows,
                "identifiable_unit_rows": ident,
                "identifiable_fragment_rows": frag,
            }

        n_id = int(id_unit.sum())
        per_metal[held] = {
            "status": "OK" if n_id >= 2 else "NO_IDENTIFIABLE_ROWS",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_identifiable_unit_rows": n_id,
            "share_identifiable_unit_rows": round(float(id_unit.mean()), 4),
            "n_identifiable_fragment_rows": int(id_frag.sum()),
            "share_identifiable_fragment_rows": round(float(id_frag.mean()), 4),
            "n_ligands_identifiable": int(
                df["ligand_canonical"].iloc[test][id_unit].nunique()
            ) if n_id else 0,
            "representations": rows,
            "metal_columns": list(metal_columns),
        }
        d = rows[PRIMARY_REP]["all_rows"]
        print(f"[UP-19] {held:2s} n_test={len(test):5d} id={n_id:4d} ({id_unit.mean():.3f}) "
              f"r2_abs={d['r2_abs']:+.4f} bias={d['bias']:+.4f} "
              f"r2_centred={d['r2_centred']:+.4f} penalty={d['offset_penalty']:.4f}"
              + (f"  r2_centred|id={rows[PRIMARY_REP]['identifiable_unit_rows']['r2_centred']:+.4f}"
                 if rows[PRIMARY_REP]["identifiable_unit_rows"] else ""))

    # Pooled numbers. A pooled R^2 is 1 - sum(SSE)/sum(SST); averaging per-metal
    # R^2 would weight a 756-row metal the same as a 14,785-row one.
    pooled = {}
    for rep in REPRESENTATIONS:
        all_parts = [per_metal[m]["representations"][rep]["all_rows"]
                     for m in METALS if per_metal.get(m, {}).get("status") in ("OK", "NO_IDENTIFIABLE_ROWS")]
        id_parts = [per_metal[m]["representations"][rep]["identifiable_unit_rows"]
                    for m in METALS
                    if per_metal.get(m, {}).get("representations", {})
                    .get(rep, {}).get("identifiable_unit_rows")]
        frag_parts = [per_metal[m]["representations"][rep]["identifiable_fragment_rows"]
                      for m in METALS
                      if per_metal.get(m, {}).get("representations", {})
                      .get(rep, {}).get("identifiable_fragment_rows")]
        pooled[rep] = {
            "all_rows": pool(all_parts),
            "identifiable_unit_rows": pool(id_parts) if id_parts else None,
            "identifiable_fragment_rows": pool(frag_parts) if frag_parts else None,
        }

    return {"per_metal": per_metal, "pooled": pooled,
            "elapsed_seconds": round(time.time() - t0, 1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--endpoint", default="potency", choices=list(ENDPOINTS))
    ap.add_argument("--metal-cols", default=",".join(DEFAULT_METAL_COLUMNS))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_identifiability.json")
    args = ap.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)
    metal_columns = tuple(c.strip() for c in str(args.metal_cols).split(",") if c.strip())

    sub = df[df["endpoint"] == args.endpoint]
    n_total = len(sub)
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna()].copy()
    sub = sub.reset_index(drop=True)
    if sub.empty:
        print("FAIL: empty endpoint")
        return 1

    vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    blocks = build_blocks(sub, descriptors, vocabulary, metal_columns=metal_columns)
    y = sub["value_log10"].to_numpy(dtype=float)

    print(f"[UP-19] endpoint={args.endpoint} rows={len(sub)} "
          f"ligands={sub['ligand_canonical'].nunique()}")
    mo = run_metal_out(sub, blocks, y, metal_columns)

    # Cross-check: this script refits the same model, so its all-rows absolute R^2
    # must reproduce msrs_lomo_surface to four decimals.
    lomo_path = PROJECT_ROOT / "metrics" / "by_metric" / "msrs_lomo_surface.json"
    check = {"status": "MISSING"}
    if lomo_path.exists():
        lomo = json.loads(lomo_path.read_text())
        ref = lomo["endpoints"][args.endpoint]["splits"]["metal_out"]["per_metal"]
        deltas = {}
        for m in METALS:
            a = mo["per_metal"].get(m, {})
            if a.get("status") not in ("OK", "NO_IDENTIFIABLE_ROWS"):
                continue
            mine = a["representations"][PRIMARY_REP]["all_rows"]["r2_abs"]
            theirs = ref.get(m, {}).get("models", {}).get(PRIMARY_REP, {}).get("r2")
            if theirs is None:
                continue
            deltas[m] = round(abs(mine - theirs), 4)
        check = {
            "status": "OK",
            "reference": "metrics/by_metric/msrs_lomo_surface.json:"
                         f"/endpoints/{args.endpoint}/splits/metal_out/per_metal/*/models/"
                         f"{PRIMARY_REP}/r2",
            "abs_delta_per_metal": deltas,
            "max_abs_delta": max(deltas.values()) if deltas else None,
            "reproduces": bool(deltas and max(deltas.values()) <= 0.0002),
        }

    primary = mo["pooled"][PRIMARY_REP]["identifiable_unit_rows"]
    per_metal_centred = {}
    for m in METALS:
        r = mo["per_metal"].get(m, {})
        block = r.get("representations", {}).get(PRIMARY_REP, {}).get("identifiable_unit_rows")
        per_metal_centred[m] = {
            "status": r.get("status"),
            "n_identifiable": r.get("n_identifiable_unit_rows"),
            "r2_centred": block["r2_centred"] if block else None,
            "r2_abs": block["r2_abs"] if block else None,
            "bias": block["bias"] if block else None,
        }
    evaluable = [m for m, v in per_metal_centred.items() if v["r2_centred"] is not None]
    n_above = sum(1 for m in evaluable if per_metal_centred[m]["r2_centred"] > PER_METAL_MIN)

    primary_pass = bool(primary and primary["r2_centred"] >= CENTRED_R2_MIN)
    secondary_pass = bool(n_above >= PER_METAL_COUNT)
    kill = bool(
        (primary is None or primary["r2_centred"] < 0.10) or n_above < 2
    )

    gate = {
        "frozen_at": GATE_FROZEN_AT,
        "primary": {
            "metric": "pooled centred R2 on identifiable (ligand, cell) rows",
            "value": primary["r2_centred"] if primary else None,
            "threshold": CENTRED_R2_MIN,
            "pass": primary_pass,
        },
        "primary_reference": {
            "value": 0.2128,
            "source": "metrics/by_metric/msrs_two_stage.json:"
                      "/within_metal/aggregate/additive/weighted_r2",
            "note": "same additive descriptors model with the metal inside the training set",
        },
        "secondary": {
            "metric": f"metals with centred R2 > {PER_METAL_MIN}",
            "value": n_above,
            "n_evaluable": len(evaluable),
            "n_metals": len(METALS),
            "threshold": PER_METAL_COUNT,
            "pass": secondary_pass,
            "not_evaluable": [m for m in METALS if m not in evaluable],
            "note": "Re has zero identifiable rows, so it can never satisfy this "
                    "criterion; it is kept in the denominator.",
        },
        "gate_pass": bool(primary_pass and secondary_pass),
        "kill": {
            "rule": "centred R2 < 0.10, or fewer than 2 of 5 metals above 0.10, "
                    "closes the metal_out line permanently",
            "triggered": kill,
        },
    }

    out = {
        "artifact": "msrs_identifiability",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 metal extrapolation, identifiability split (UP-19)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup.relative_to(PROJECT_ROOT)),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc.relative_to(PROJECT_ROOT)),
            "metal_descriptor_columns": list(metal_columns),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": args.endpoint,
            "n_rows": int(len(sub)),
            "n_rows_before_drop": int(n_total),
            "n_ligands": int(sub["ligand_canonical"].nunique()),
            "learner": "the additive LightGBM used by msrs_lomo_surface (same seed)",
            "representations": list(REPRESENTATIONS),
            "primary_representation": PRIMARY_REP,
            "identifiable_definition": "test row whose (ligand_canonical, cell_line_key) "
                                       "appears in the training metals",
            "centred_r2": "1 - sum((y-ybar)-(p-pbar))^2 / sum((y-ybar)^2) within metal; "
                          "equivalent to R2_abs + bias^2/Var(y), i.e. an oracle offset",
            "pooling": "1 - sum(SSE)/sum(SST) across metals, not a mean of per-metal R2",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "reproduces_lomo": check,
        "metal_out": mo,
        "decomposition_summary": {
            "primary_rep": PRIMARY_REP,
            "identifiable_row_share": (
                round(primary["n"] / mo["pooled"][PRIMARY_REP]["all_rows"]["n"], 4)
                if primary else None
            ),
            "all_rows": mo["pooled"][PRIMARY_REP]["all_rows"],
            "identifiable_unit_rows": mo["pooled"][PRIMARY_REP]["identifiable_unit_rows"],
            "identifiable_fragment_rows": mo["pooled"][PRIMARY_REP]["identifiable_fragment_rows"],
            "per_metal_centred_on_identifiable": per_metal_centred,
        },
        "gate": gate,
        "caveats": [
            "Centring uses the held-out labels, so R2_centred is an oracle-offset bound, "
            "not a deployable score. The claim it supports is about within-metal ordering.",
            "The identifiable set uses ligand_canonical, the same key as UP-10/UP-11. "
            "Keying on ligand_main_fragment (the Morgan input) gives a much larger set and "
            "is reported as a sensitivity, but it conflates 'seen this ligand' with "
            "'seen this scaffold'.",
            "Re has zero identifiable rows: no (ligand, cell) unit measured under Re is also "
            "measured under another metal, so nothing about Re transfers.",
            "Per-metal centred R2 on 177-406 rows has a wide interval; the secondary gate "
            "counts metals, it does not test them individually.",
            "Rows are the deduplicated table (21,045), so a ligand measured many times in one "
            "paper counts once per (ligand, cell, metal) group; n is not independent.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    print(f"[UP-19] pooled all_rows       r2_abs={mo['pooled'][PRIMARY_REP]['all_rows']['r2_abs']} "
          f"r2_centred={mo['pooled'][PRIMARY_REP]['all_rows']['r2_centred']}")
    if mo["pooled"][PRIMARY_REP]["identifiable_unit_rows"]:
        v = mo["pooled"][PRIMARY_REP]["identifiable_unit_rows"]
        print(f"[UP-19] pooled identifiable   n={v['n']} r2_abs={v['r2_abs']} "
              f"r2_centred={v['r2_centred']}")
    print(f"[UP-19] reproduces_lomo={check.get('reproduces')} "
          f"max_delta={check.get('max_abs_delta')}")
    print(f"[UP-19] gate_pass={gate['gate_pass']} primary={gate['primary']['value']} "
          f"secondary={n_above}/{len(evaluable)} kill={kill}")
    print(f"json: {args.out}")
    print("PASS" if gate["gate_pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
