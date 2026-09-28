"""Wf-222 — Smoke benchmark for click-chemistry building-block integration.

================================================================
What this benchmark exercises
================================================================
This is a *cheap, CPU-only* end-to-end smoke benchmark for the Wf-222
ship surface.  It exercises the new
:mod:`molmetal.molmetal_lam.sbdd_env.click_building_blocks` module by:

1. Wrapping the real :class:`FLOWRReferenceAdapter` and
   :class:`SoftMolAdapter` with :class:`ClickBuildingBlocks`.
2. Calling ``generate()`` on each wrapper.
3. Counting (a) the original molecules returned and (b) the click-
   building-block molecules appended.
4. Firing the canonical 4 click reactions on a small pool of the
   appended click blocks and recording how many canonical products
   emerge.

This is the *G3* smoke-benchmark gate for Wf-222.  Output is a JSON
dict on stdout + a sibling file under
``molmetal/reports/wf222_sbdd_click_blocks_smoke/`` for the audit trail.

Honest framing
--------------
* CPU-only, no GPU, no upstream clone, no checkpoint download.
* The benchmark does NOT claim any metric uplift — it only verifies
  that (a) the wrapper preserves the original adapter output, (b) the
  click-block library is reachable from the SBDD layer, and (c) the
  four canonical click reactions actually fire on the building-block
  library.
* We never invoke the FLOWR or SoftMol upstream repos; we exercise
  the *fallback* / *stub* paths only — consistent with
  ``test_softmol_adapter.py``.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf222_sbdd_click_blocks_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Repo setup — make `molmetal` importable when invoked as a script.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from molmetal.domain import Pocket  # noqa: E402
from molmetal.ports import GenerationConfig  # noqa: E402


REPORT_DIR = _REPO / "molmetal" / "reports" / "wf222_sbdd_click_blocks_smoke"


def _tiny_pocket() -> Pocket:
    return Pocket(
        pdb_id="wf222_smoke",
        coords=torch.zeros(2, 3, dtype=torch.float32),
        atom_types=torch.tensor([6, 7], dtype=torch.long),
        residue_ids=torch.tensor([0, 0], dtype=torch.long),
        chain_ids=torch.tensor([0, 0], dtype=torch.long),
        mask=torch.tensor([True, True]),
        center=torch.zeros(3, dtype=torch.float32),
        radius=6.0,
    )


def _run_flowr_smoke() -> dict:
    """Wrap FLOWRReferenceAdapter with ClickBuildingBlocks and run."""
    from molmetal.molmetal_lam.sbdd_env.flowr_adapter import (
        FLOWRReferenceAdapter,
    )
    from molmetal.molmetal_lam.sbdd_env.click_building_blocks import (
        CLICK_BUILDING_BLOCKS,
        wrap_with_click_blocks,
    )
    adapter = FLOWRReferenceAdapter()
    adapter.setup(device="cpu")
    wrapped = wrap_with_click_blocks(adapter)

    cfg = GenerationConfig(n_samples=2, seed=42)
    t0 = time.time()
    mols = wrapped.generate(_tiny_pocket(), cfg)
    dt = time.time() - t0

    click_mols = [m for m in mols[cfg.n_samples:] if m.smiles]
    click_smiles = {m.smiles for m in click_mols}
    return {
        "adapter": "FLOWRReferenceAdapter",
        "n_samples_requested": cfg.n_samples,
        "n_total_returned": len(mols),
        "n_click_blocks_appended": len(click_mols),
        "n_unique_click_block_smiles": len(click_smiles),
        "library_size": len(CLICK_BUILDING_BLOCKS),
        "elapsed_seconds": round(dt, 4),
        "fallback_active": wrapped.get_metadata().get("fallback_active", None),
        "wrapped_metadata_name": wrapped.get_metadata().get("name"),
    }


def _run_softmol_smoke() -> dict:
    """Wrap SoftMolAdapter with ClickBuildingBlocks and run."""
    from molmetal.adapters.softmol_adapter import SoftMolAdapter
    from molmetal.molmetal_lam.sbdd_env.click_building_blocks import (
        CLICK_BUILDING_BLOCKS,
        wrap_with_click_blocks,
    )
    adapter = SoftMolAdapter()
    adapter.setup(device="cpu")  # enters fallback mode without upstream
    wrapped = wrap_with_click_blocks(adapter)

    cfg = GenerationConfig(n_samples=3, seed=42)
    t0 = time.time()
    mols = wrapped.generate(_tiny_pocket(), cfg)
    dt = time.time() - t0

    return {
        "adapter": "SoftMolAdapter",
        "n_samples_requested": cfg.n_samples,
        "n_total_returned": len(mols),
        "n_click_blocks_appended": len(mols),
        "library_size": len(CLICK_BUILDING_BLOCKS),
        "elapsed_seconds": round(dt, 4),
        "fallback_active": wrapped.get_metadata().get("fallback_active", None),
        "wrapped_metadata_name": wrapped.get_metadata().get("name"),
    }


def _run_click_enumeration_smoke() -> dict:
    """Fire click reactions across a small building-block pool."""
    from molmetal.molmetal_lam.sbdd_env.click_building_blocks import (
        CLICK_BUILDING_BLOCKS,
        enumerate_click_products,
        is_click_block,
    )

    pool = [smi for smi, _ in CLICK_BUILDING_BLOCKS]
    n_click_blocks = sum(1 for s in pool if is_click_block(s))

    t0 = time.time()
    products = enumerate_click_products(pool, max_products=24)
    dt = time.time() - t0

    return {
        "test": "enumerate_click_products",
        "pool_size": len(pool),
        "pool_with_click_handle": n_click_blocks,
        "n_products_returned": len(products),
        "elapsed_seconds": round(dt, 4),
        "first_5_products": products[:5],
    }


def _run_passthrough_smoke() -> dict:
    """Backward-compat: append_blocks=False must be a transparent pass."""
    from molmetal.molmetal_lam.sbdd_env.flowr_adapter import (
        FLOWRReferenceAdapter,
    )
    from molmetal.molmetal_lam.sbdd_env.click_building_blocks import (
        wrap_with_click_blocks,
    )
    adapter = FLOWRReferenceAdapter()
    adapter.setup(device="cpu")
    raw_output = adapter.generate(_tiny_pocket(), GenerationConfig(n_samples=2))
    wrapped = wrap_with_click_blocks(adapter, append_blocks=False)
    wrapped_output = wrapped.generate(_tiny_pocket(), GenerationConfig(n_samples=2))
    # Compare heavy-atom skeletons (RDKit canonical, Hs removed) so the
    # check is robust to different RDKit embed conventions.
    from rdkit import Chem as _Chem
    def _heavy_smi(m):
        c = _Chem.RemoveHs(_Chem.MolFromSmiles(m.smiles))
        return _Chem.MolToSmiles(c) if c is not None else m.smiles
    raw_smiles = [_heavy_smi(m) for m in raw_output]
    wrapped_smiles = [_heavy_smi(m) for m in wrapped_output]
    return {
        "test": "append_blocks_False_passthrough",
        "raw_n": len(raw_output),
        "wrapped_n": len(wrapped_output),
        "raw_smiles": raw_smiles,
        "wrapped_smiles": wrapped_smiles,
        "value_match": raw_smiles == wrapped_smiles,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {
        "workflow": "Wf-222",
        "task": "Wire click reactions into SBDD pipeline (flowr/softmol building blocks)",
        "py_version": sys.version.split()[0],
        "started_unix": started,
        "runs": {
            "flowr": _run_flowr_smoke(),
            "softmol": _run_softmol_smoke(),
            "click_enumeration": _run_click_enumeration_smoke(),
            "passthrough": _run_passthrough_smoke(),
        },
        "elapsed_seconds": round(time.time() - started, 4),
    }

    out_path = REPORT_DIR / "smoke_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2))

    # Console summary
    print(json.dumps(report, indent=2))
    print(f"\n[wf222-smoke] wrote {out_path}")


if __name__ == "__main__":
    main()