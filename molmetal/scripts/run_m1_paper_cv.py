#!/usr/bin/env python3
"""Run m1 (Ru binary ROC-AUC) under the Krasnov-comparable protocol.

==============================================================================
What "Krasnov-comparable" means here, and what it does NOT mean
==============================================================================

Adopted 2026-10-04 per user decision (option B of the m1 alignment audit):

  * COHORT — the paper's funnel (IC50 present, Ru, measurement-level dedup on
    (SMILES_Ligands, Cell_line, Time(h)), 76-name non-cancer exclusion) PLUS a
    cell-line restriction to the top N by row volume.
  * SPLIT  — DOI-grouped 10-fold CV: every row of an article lands entirely on
    one side.
  * METRIC — binary ROC-AUC (toxic < 10 uM), the paper's cut.

The cell-line restriction exists because the paper's Ru ML set is 23 cell lines
and it does NOT publish which 23.  MEASURED, the top-23 by row volume give
11,517 rows / 4,502 complexes / 23 lines against the paper's
11,294 / 4,399 / 23 — within 2% on all three, which is why this is the best
available reconstruction.  It is still OUR choice, and any number produced here
must be reported with that caveat attached.  See COMPARISON_BASELINES.md §6.

==============================================================================
Hazards this driver exists to avoid
==============================================================================

1. `--resume-from auto` + a SHARED checkpoint directory silently skips
   training: the trainer finds the previous fold's checkpoint and resumes past
   the epoch budget, finishing in ~2 s and scoring the held-out fold with
   weights that saw it.  Every fold therefore gets its OWN `--ckpt-dir`, and
   the log is checked for the cold-start marker.
2. A shared `--out-dir` would let folds overwrite each other's events.

Both are recorded per fold in the summary so a reader can check them rather
than trust them.

Usage::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    python -m molmetal.scripts.run_m1_paper_cv \\
        --top-cell-lines 23 --n-folds 10 --epochs 10 --device cuda \\
        --out-dir /tmp/m1_top23_cv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
TRAINER = "molmetal.scripts.train_dmpnn_multitask_wasserstein"
SPLIT_COLUMNS = ("smiles", "metal", "scaffold_id", "split", "activity")


def build_fold_csvs(
    csv_path: Path, folds_dir: Path, tmp_dir: Path, top_n: int
) -> list[dict]:
    """Write one split-table CSV per fold and return the fold metadata."""
    from molmetal.data import cytotox

    df = pd.read_csv(csv_path)
    _probe = df.loc[cytotox.build_paper_cohort_indices(df, metal="Ru")]
    lines = cytotox.top_cell_lines(_probe, top_n)
    cohort = cytotox.build_paper_cohort_indices(
        df, metal="Ru", cell_lines=lines
    )
    print(
        f"[m1-cv] cohort: {len(cohort)} rows, {len(lines)} cell lines "
        f"(paper: 11,294 / 23)",
        flush=True,
    )

    sub = df.loc[cohort].reset_index(drop=True)
    out: list[dict] = []
    folds_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for k in range(10):
        fp = folds_dir / f"fold_{k}.json"
        if not fp.is_file():
            raise SystemExit(
                f"{fp} missing — generate folds first with doi_split_cv.py "
                "--top-cell-lines and --out <folds_dir>"
            )
        payload = json.loads(fp.read_text())
        # The fold indices are positional into the FULL csv, not into the
        # cohort — `doi_split_cv.py` documents them as "positional into df"
        # and reports cohort_rows separately.  Building the split on `sub`
        # would mislabel most rows; build it on the full frame and then
        # project onto the cohort.
        va = set(payload["val_idx"])
        tr = set(payload["train_idx"])
        # 2026-10-04 FIX — this used to label the held-out fold "val" and
        # write no "test" rows at all.  `build_metal_split_indices` then
        # returned an EMPTY test index, the trainer printed
        # `n_test=0 test_auc=nan`, and the regex below (which cannot match
        # "nan") recorded `test_auc: null` — a fold that trained for 833 s and
        # reported nothing.  MEASURED in /tmp/m1_top23/fold_0/run.log:45.
        #
        # The held-out fold is the EVALUATION set under k-fold CV, so it is
        # labelled "test".  A small DOI-grouped slice OF THE REMAINING rows
        # becomes "val", which is what early stopping selects on — without it
        # the trainer would select its best epoch on the same rows it reports,
        # and the number would be optimistic.
        full = df.copy()
        smi_series = full["SMILES_Ligands"].astype(str)
        doi_series = full.get("DOI", pd.Series(["?"] * len(full))).astype(str)

        # The split table must cover EVERY row the trainer will load, not just
        # the top-N cohort.  `build_metal_split_indices` resolves each dataset
        # row through a SMILES-keyed lookup and DEFAULTS AN UNSEEN SMILES TO
        # "train" — so a table holding only the 11,517 cohort rows would push
        # the other ~7,600 Ru rows into training silently, and the fold would
        # not be the fold we built.
        #
        # Rows outside the cohort therefore get an explicit "train" label: they
        # are legitimately training data (the paper's cohort restriction is an
        # EVALUATION scope, not a training-scope claim we can make), and the
        # held-out fold is exactly the cohort rows of fold k.
        label = np.full(len(full), "train", dtype=object)
        label[np.array([i in va for i in range(len(full))])] = "test"

        # Inner validation slice, grouped so early stopping cannot leak.
        #
        # Groups come from `cytotox.doi_group_ids`, NOT from raw DOI: a
        # complex appearing in two articles links those articles into one
        # group (MEASURED, 6.1% of cohort complexes span >1 DOI, up to 13), so
        # a raw-DOI slice can drop a "val" row into a "test" DOI.  MEASURED
        # with raw DOIs this leaked 36 groups.
        #
        # Grouped over the COHORT rows only, because that is the frame
        # `doi_split_cv.py` grouped when it built the outer fold.  Grouping
        # over the full frame instead merges MORE DOIs (it sees more
        # complex-article links), which disagrees with the outer fold and
        # still leaked 1 group (MEASURED, fold 2).
        groups_full = pd.Series(-1, index=full.index, dtype=np.int64)
        groups_cohort = cytotox.doi_group_ids(
            doi_series.iloc[cohort].reset_index(drop=True),
            smiles=smi_series.iloc[cohort].reset_index(drop=True),
        )
        groups_full.iloc[cohort] = groups_cohort
        train_groups = sorted({int(groups_full.iloc[i]) for i in sorted(tr)})
        rng = np.random.RandomState(1000 + int(k))
        perm = rng.permutation(len(train_groups))
        n_inner = max(1, int(round(0.10 * len(train_groups))))
        val_groups = {train_groups[j] for j in perm[:n_inner]}
        group_in_val = groups_full.isin(list(val_groups)).to_numpy()
        is_inner_pool = np.array(
            [i in tr and label[i] == "train" for i in range(len(full))]
        )
        label[is_inner_pool & group_in_val] = "val"

        # ONE label per SMILES, resolved with priority test > val > train.
        # Rows repeat a SMILES across cell lines, and `build_metal_split_
        # indices` builds its dict with last-write-wins over rows — so a
        # per-row label makes the assignment depend on file order.  Resolving
        # per SMILES makes it deterministic, which is what a published
        # protocol needs.
        priority = {"test": 0, "val": 1, "train": 2}

        def _resolve(labels: np.ndarray) -> np.ndarray:
            best: dict[str, str] = {}
            for smi, lab in zip(smi_series.tolist(), labels.tolist()):
                cur = best.get(smi)
                if cur is None or priority[lab] < priority[cur]:
                    best[smi] = lab
            return smi_series.map(best).to_numpy()

        full_split = _resolve(label)

        table = pd.DataFrame(
            {
                "smiles": smi_series,
                "metal": full["Metal"].astype(str),
                "scaffold_id": doi_series,
                "split": full_split,
                # NOTE: inert on this path.  `build_metal_split_indices` reads
                # only smiles/metal/split, and the labels come from
                # `MetalCytotoxDataset` built off the raw CSV.  Kept because
                # the loader validates the column set.
                "activity": pd.to_numeric(
                    full["IC50_Dark_value"], errors="coerce"
                ).fillna(0.0),
            }
        )
        path = tmp_dir / f"split_fold_{k}.csv"
        table.to_csv(path, index=False)
        cohort_labels = full_split[cohort]
        # THE trainer's contract is SMILES disjointness, not DOI disjointness:
        # `build_metal_split_indices` builds a SMILES-keyed dict and never
        # reads `scaffold_id` (verified — the column is inert on this path).
        # A SMILES on both sides is the thing that actually leaks, and the
        # per-SMILES resolution above makes it impossible by construction;
        # assert it anyway so a future refactor cannot quietly reintroduce it.
        for a, b in (("test", "train"), ("test", "val"), ("val", "train")):
            sa = set(smi_series[full_split == a])
            sb = set(smi_series[full_split == b])
            if sa & sb:
                raise SystemExit(
                    f"fold {k}: {len(sa & sb)} SMILES appear in both {a!r} "
                    f"and {b!r} — the trainer keys its split on SMILES, so "
                    "these rows would be assigned by dict order."
                )
        # DOI overlap is REPORTED, not asserted.  It can be non-zero for a
        # reason that is not a leak: a Ru measurement OUTSIDE the top-N
        # cell-line cohort can share a complex with a held-out cohort row,
        # which forces that whole SMILES (and hence its article) into "test".
        # MEASURED at 1-2 DOIs per fold after the grouping fix, down from 36
        # with raw-DOI grouping.
        doi_test = set(doi_series[full_split == "test"])
        doi_val = set(doi_series[full_split == "val"])
        out.append(
            {
                "fold": k,
                "csv": str(path),
                "n_train": int((cohort_labels == "train").sum()),
                "n_val": int((cohort_labels == "val").sum()),
                "n_test": int((cohort_labels == "test").sum()),
                "doi_overlap": payload.get("doi_overlap_train_val"),
                "doi_test_val_overlap": len(doi_test & doi_val),
            }
        )
    return out


def run_fold(meta: dict, args) -> dict:
    """Train one fold.  Per-fold ckpt AND out dirs (see module docstring)."""
    out = Path(args.out_dir) / f"fold_{meta['fold']}"
    ck = out / "ck"
    log = out / "run.log"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MOLMETAL_DATA_DIR"] = os.environ.get(
        "MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal")
    )
    cmd = [
        PYTHON, "-m", TRAINER,
        "--dataset", "metalcytotox_5metal",
        "--metals", "Ru",
        "--seeds", "0",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--csv", meta["csv"],
        "--out-dir", str(out), "--ckpt-dir", str(ck),
        # --loose-filter disables the exposure-time / IC50 floor, which the
        # paper does not apply (it keeps time as a feature).
        "--loose-filter",
    ]
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    text = log.read_text(errors="ignore")

    # Read the metric off the `done` event, which is the authoritative final
    # value the trainer emits.  The previous regex matched only `[0-9.]+`, so
    # a run whose held-out set was empty reported `test_auc=nan` in the log
    # and the driver recorded `null` — a silently missing result rather than a
    # failure.  Anything non-finite now raises.
    auc = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or '"done"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "test_auc" in ev:
            auc = float(ev["test_auc"])
    if auc is None:
        for m in re.finditer(r"test_auc\s*[=:]\s*([0-9.]+|nan|NaN)", text):
            try:
                auc = float(m.group(1))
            except ValueError:
                auc = float("nan")
    if auc is None or not math.isfinite(auc):
        raise RuntimeError(
            f"fold {meta['fold']}: test_auc came back {auc!r}. "
            "An empty or mislabelled held-out set is a broken fold, not a "
            "result — see the docstring's hazard list. Log: "
            f"{log}\nTail:\n" + "\n".join(text.splitlines()[-15:])
        )

    # Hazard (1): prove this fold actually trained.
    resumed = len(re.findall(r"resume", text, flags=re.I))
    cold = len(re.findall(r"cold.start", text, flags=re.I))
    return {
        "fold": meta["fold"],
        "rc": int(rc),
        "test_auc": auc,
        "wall_s": round(wall, 1),
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": meta["n_test"],
        "doi_overlap": meta["doi_overlap"],
        "doi_test_val_overlap": meta.get("doi_test_val_overlap"),
        "resume_mentions": resumed,
        "cold_start_mentions": cold,
        "log": str(log),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="default: MetalCytoToxDB.csv under MOLMETAL_DATA_DIR")
    ap.add_argument("--folds-dir", default=None, help="default: <out-dir>/folds")
    ap.add_argument("--top-cell-lines", type=int, default=23)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", default="all", help="'all' or comma-separated fold ids")
    ap.add_argument("--out-dir", default="/tmp/m1_paper_cv")
    args = ap.parse_args(argv)

    data_dir = Path(
        os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
    )
    csv_path = Path(args.csv) if args.csv else data_dir / "MetalCytoToxDB.csv"
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    folds_dir = Path(args.folds_dir) if args.folds_dir else out_dir / "folds"
    tmp_dir = out_dir / "split_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = build_fold_csvs(csv_path, folds_dir, tmp_dir, args.top_cell_lines)
    wanted = (
        [m["fold"] for m in metas]
        if args.folds == "all"
        else [int(x) for x in args.folds.split(",")]
    )
    rows = [run_fold(m, args) for m in metas if m["fold"] in wanted]

    aucs = [r["test_auc"] for r in rows if r["test_auc"] is not None]
    n_requested = len(wanted)
    complete = len(aucs) == n_requested and n_requested > 0
    summary = {
        "protocol": {
            "cohort": f"paper funnel + top-{args.top_cell_lines} cell lines by row volume",
            "split": "DOI-grouped 10-fold CV; held-out fold = TEST, inner "
                     "10%-of-DOIs slice of the training rows = VAL (early stop)",
            "metric": "binary ROC-AUC (toxic < 10 uM) on the held-out fold",
            "epochs": args.epochs,
            "note": (
                "The paper does not publish which 23 cell lines it used; this "
                "subset is OUR choice. MEASURED it gives 11,517 rows / 4,502 "
                "complexes / 23 lines vs the paper's 11,294 / 4,399 / 23. "
                "Mean/std over fewer than 10 folds is NOT the paper's number "
                "— check `complete` before quoting a mean."
            ),
        },
        "n_requested_folds": n_requested,
        "n_folds_run": len(aucs),
        "complete": complete,
        "per_fold_auc": aucs,
        "mean": float(np.mean(aucs)) if aucs else None,
        "std": float(np.std(aucs)) if aucs else None,
        "paper_reference": "0.810 +/- 0.024 (LightGBM, DOI-split 10CV)",
        "per_fold": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if aucs:
        print(
            f"[m1-cv] folds={len(aucs)}  mean={np.mean(aucs):.4f} "
            f"+/- {np.std(aucs):.4f}   paper=0.810 +/- 0.024",
            flush=True,
        )
    print(f"[m1-cv] wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
