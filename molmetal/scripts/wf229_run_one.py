#!/usr/bin/env python3
"""Wf-229 single-config runner — patches CSV path AND runs one train config.

Usage:
    python molmetal/scripts/wf229_run_one.py --code 000 --epochs 5 --seed 0

This is called by wf229_ablation_launcher.py for each of the 8 (or 11)
configurations in the ablation matrix.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(PROJECT_ROOT))

# Patch module-level path defaults BEFORE the train script touches them.
LOCAL_CSV = Path("/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv")
LOCAL_CACHE = Path("/home/hugo/codes/mol-metal/.data/molmetal/3d_cache")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

import molmetal.data.cytotox as _cytotox
_cytotox.DEFAULT_DATA_DIR = LOCAL_CSV.parent
_cytotox.DEFAULT_CSV = LOCAL_CSV
_cytotox.DEFAULT_CACHE_DIR = LOCAL_CACHE

# Default-arg values are bound at function definition time, so patch
# MetalCytotoxDataset.from_csv signature's default AND __init__ defaults.
import molmetal.data.cytotox as _cy
_orig_from_csv = _cy.MetalCytotoxDataset.from_csv
_orig_init = _cy.MetalCytotoxDataset.__init__

@classmethod
def _patched_from_csv(cls, path=LOCAL_CSV, filters=None, **kw):
    return _orig_from_csv.__func__(cls, path=path, filters=filters, **kw)

def _patched_init(self, df, cache_dir=LOCAL_CACHE, conformer_cache_lru=None):
    return _orig_init(self, df=df, cache_dir=cache_dir,
                      conformer_cache_lru=conformer_cache_lru
                      if conformer_cache_lru is not None
                      else _cy.CONFORMER_CACHE_LRU)

_cy.MetalCytotoxDataset.from_csv = _patched_from_csv
_cy.MetalCytotoxDataset.__init__ = _patched_init


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="3-bit config code e.g. 000")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metal", default="Ru")
    p.add_argument("--split", default="random")
    p.add_argument("--use-kendall-loss", action="store_true")
    p.add_argument("--use-dann", action="store_true")
    p.add_argument("--data-config", default="single",
                   choices=["single", "four-domain"])
    p.add_argument("--max-rows-domain", default="0:200,1:200,2:200,3:200",
                   help="Per-domain row caps for fast four-domain runs")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = _parse()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cli = [
        "--metal", args.metal,
        "--split", args.split,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--num-workers", "0",
        "--bf16",
        "--out", str(out_path),
    ]
    if args.use_kendall_loss:
        cli.append("--use-kendall-loss")
    if args.use_dann:
        cli.append("--use-dann")
    if args.data_config != "single":
        cli.extend(["--data-config", args.data_config])
        if args.max_rows_domain:
            cli.extend(["--max-rows-domain", args.max_rows_domain])

    from molmetal.scripts import train_dmpnn_multitask as _train
    t0 = time.time()
    rc = _train.main(cli)
    elapsed = time.time() - t0
    print(
        f"[wf229-run-one] code={args.code} rc={rc} elapsed={elapsed:.1f}s "
        f"out={out_path}",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())