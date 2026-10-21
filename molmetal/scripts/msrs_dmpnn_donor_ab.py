#!/usr/bin/env python3
"""UP-3 fallback A/B — do the Cp / aryl-anion donors change the metal-conditioned D-MPNN?

``msrs_complex3d_probe.json`` says a 3D coordination graph cannot be built from
this corpus: the metal is absent from every SMILES, 100% of ligands are
disconnected multi-fragment strings, and 93.5% of "complete donor set" graphs are
rejected by RDKit sanitisation.  UP-3 therefore falls back to the 2D graph that
already exists (``molmetal/data/metal_graph.py:303-310``: explicit metal node +
coordination edges).

But the probe also found the one thing that graph gets wrong: ``_DONOR_STRENGTH``
covers N/O/S/P/Se only, while **33.5% of the sample binds through a Cp or aryl
carbanion** that is never scored as a donor.  The builder never sanitises — it
emits edge indices — so those donors can be represented at zero cost.

This script runs the ``metal_state`` arm twice on the same folds, same seed,
same hyper-parameters, changing only ``include_carbanion``, and reports the
paired Δ.  One seed: DP-1 resolved as "shape is not learnable on a held-out
metal", so this is a probe, not a headline run.

Gate (frozen before the run, 2026-10-21):

    PASS  if delta_r2 >= +0.03 on the DOI holdout
    FAIL  otherwise -> the donors that bind the metal are not the missing input
         either, and UP-3 closes with the 3D negative alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

from msrs_dmpnn_surface import (  # noqa: E402
    attach_conditioning,
    cell_one_hot,
    conditioning_matrix,
    graph_key,
    load_config,
    run_arm_on_split,
    score,
    select_cohort,
)
from molmetal.data.metal_graph import build_metal_graph  # noqa: E402

OUT = REPO / "metrics" / "by_metric" / "msrs_dmpnn_donor_ab.json"
GATE = {"delta_r2": 0.03, "frozen_at": "2026-10-21T15:10+08:00"}


def build_cache(frame: pd.DataFrame, include_carbanion: bool):
    cache = {}
    cols = ["ligand_canonical", "metal", "oxidation_state", "charge_complex"]
    for row in frame[cols].itertuples(index=False):
        key = graph_key(row)
        if key in cache:
            continue
        cache[key] = build_metal_graph(
            key[0], key[1], key[2], key[3],
            use_metal_node=True, include_carbanion=include_carbanion,
        )
    return cache


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "msrs_dmpnn_surface_gpu0.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg, compute_cfg = cfg["data"], cfg["compute"]
    if compute_cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(compute_cfg.get("num_threads", 1)))
    device = torch.device(str(compute_cfg.get("device", "cpu")))

    limit = int(data_cfg.get("limit", 0)) if args.limit is None else int(args.limit)
    frame = select_cohort(REPO / str(data_cfg["long"]), str(data_cfg["endpoint"]), limit)
    print(f"[UP-3b] rows={len(frame)} device={device}")

    t0 = time.time()
    cache_hetero = build_cache(frame, include_carbanion=False)
    cache_carb = build_cache(frame, include_carbanion=True)
    print(f"[UP-3b] graphs={len(cache_hetero)} in {time.time() - t0:.1f}s")

    keys = [
        (str(r.ligand_canonical), str(r.metal), int(r.oxidation_state), int(r.charge_complex))
        for r in frame[["ligand_canonical", "metal", "oxidation_state", "charge_complex"]]
        .itertuples(index=False)
    ]
    ok = np.array([cache_hetero[k] is not None for k in keys])
    if not ok.all():
        frame = frame[ok].reset_index(drop=True)
        keys = [k for k, good in zip(keys, ok) if good]
        print(f"[UP-3b] dropped {int((~ok).sum())} unparseable rows")

    # how many graphs actually change?
    changed = sum(
        1 for k in keys
        if not np.array_equal(cache_hetero[k][2], cache_carb[k][2])
    )
    changed_frac = changed / max(1, len(keys))
    print(f"[UP-3b] graphs whose edge set changed: {changed_frac:.3f}")

    y = frame["value_log10"].to_numpy(dtype=float)
    vocab = frame["cell_line_key"].value_counts().head(int(data_cfg["top_cell_lines"])).index.tolist()
    cell_oh = cell_one_hot(frame, vocab)
    spec = cfg["arms"]["metal_state"]
    cond = conditioning_matrix(frame, str(spec.get("metal_mode", "state")), cell_oh)
    cond_dim = cond.shape[1]

    groups = frame["doi"].astype(str).to_numpy()
    print(f"[UP-3b] split=doi groups={len(set(groups))}")

    out: Dict[str, object] = {}
    oofs: Dict[str, np.ndarray] = {}
    for name, cache in (("hetero", cache_hetero), ("hetero_carbanion", cache_carb)):
        graphs = [attach_conditioning(cache.get(k), cond[i]) for i, k in enumerate(keys)]
        oof, entry = run_arm_on_split(
            frame, y, graphs, groups,
            arm_name=name, arm=spec, cond_dim=cond_dim, cfg=cfg, device=device,
        )
        oofs[name] = oof
        out[name] = entry
        print(f"[UP-3b] {name:18s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
              f"wall={entry['wall_s']:.0f}s")

    delta = float(out["hetero_carbanion"]["r2"] - out["hetero"]["r2"])
    verdict = "DONORS_MATTER" if delta >= GATE["delta_r2"] else "DONORS_DO_NOT_MATTER"

    doc = {
        "artifact": "msrs_dmpnn_donor_ab",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 representation / UP-3 fallback",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "config": str(Path(args.config).relative_to(REPO)),
            "script": str(Path(__file__).relative_to(REPO)),
            "seed": int(cfg["train"]["seed"]),
            "device": str(device),
            "probe_artifact": "metrics/by_metric/msrs_complex3d_probe.json",
        },
        "design": {
            "arm": "metal_state",
            "splits": ["doi (GroupKFold on doi)"],
            "arms": {
                "hetero": "donors = N/O/S/P/Se (the shipped graph)",
                "hetero_carbanion": "donors = N/O/S/P/Se + carbanion [c-]/[C-] at strength 3.5",
            },
            "graph_change_share": changed_frac,
            "seeds": 1,
            "gate": GATE,
        },
        "results": out,
        "delta_r2": delta,
        "verdict": verdict,
        "verdict_rule": "PASS iff delta_r2 >= 0.03 on the DOI holdout",
        "caveats": [
            "One seed. The paired folds are identical, but the run-to-run SD of this model is "
            "not characterised on this split.",
            "The DOI split here is GroupKFold(5) on doi, matching the msrs_generalization "
            "protocol but not identical to msrs_dmpnn_surface's fragment/canonical splits, so "
            "these R2 values are not comparable to that artifact's.",
            "A Cp ring contributes several carbanion atoms; the builder caps donors at the "
            "coordination number, so an eta-5 ring consumes up to 5 of the 6 slots. That "
            "over-counts hapticity and is a known simplification.",
        ],
    }
    if not args.dry_run:
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json: {args.out}")
    print(f"[UP-3b] delta_r2={delta:+.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
