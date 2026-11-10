#!/usr/bin/env python3
"""Wf-229 smoke test — verify train_dmpnn_multitask runs end-to-end with
patched CSV path.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Patch DEFAULT_CSV BEFORE the train script imports MetalCytotoxDataset.from_csv
import molmetal.data.cytotox as _cytotox

_cytotox.DEFAULT_CSV = PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"
_cytotox.DEFAULT_CACHE_DIR = PROJECT_ROOT / ".data" / "molmetal" / "3d_cache"
_cytotox.DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cytotox.DEFAULT_DATA_DIR = PROJECT_ROOT / ".data" / "molmetal"
print(f"[wf229-smoke] patched DEFAULT_CSV -> {_cytotox.DEFAULT_CSV}", flush=True)

from molmetal.scripts import train_dmpnn_multitask as _train

t0 = time.time()
rc = _train.main(
    [
        "--metal",
        "Ru",
        "--split",
        "random",
        "--epochs",
        "2",
        "--seed",
        "0",
        "--num-workers",
        "0",
        "--bf16",
        "--out",
        str(
            PROJECT_ROOT
            / "molmetal"
            / "reports"
            / "wf229_ablation"
            / "per_config"
            / "_smoke.json"
        ),
    ]
)
print(f"[wf229-smoke] rc={rc} elapsed={time.time()-t0:.1f}s", flush=True)
sys.exit(rc)
