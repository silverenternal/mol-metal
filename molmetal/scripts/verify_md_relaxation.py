#!/usr/bin/env python3
"""Smoke verification for the MD relaxation CPU stub.

This script exercises the public surface of
:mod:`molmetal.molmetal_lam.lam_chem.md_relaxation_stub` (per the
TODO-30 P2.5 brief):

  1. :class:`SimpleForceField` — energy + gradient on a tiny mol.
  2. :class:`MDRelaxation`     — pre/post energy diff on methane.
  3. :class:`GeometryValidator` — bad-contact detection on a poisoned
                                   coordinate set.
  4. :func:`register_md_relaxation` — registration hook returns >= 1.

The script is **CPU-only** — no CUDA, no torch.  It runs in < 2 s on
any laptop.

Usage::

    python molmetal/scripts/verify_md_relaxation.py --smoke
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Make ``molmetal`` importable when run as a script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.molmetal_lam.lam_chem.md_relaxation_stub import (  # noqa: E402
    GeometryValidator,
    MDRelaxation,
    SimpleForceField,
    register_md_relaxation,
)


def _build_methane_coords() -> tuple:
    """Tetrahedral-ish methane coords — 4 H at unit-sphere, C at origin."""
    rng = np.random.default_rng(0)
    h = rng.normal(size=(4, 3))
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    coords = np.zeros((5, 3), dtype=np.float64)
    coords[1:] = h
    types = ["C", "H", "H", "H", "H"]
    return coords, types


def _smoke_force_field() -> dict:
    """Run energy + gradient on methane."""
    ff = SimpleForceField()
    coords, types = _build_methane_coords()
    e = ff.energy(coords, types)
    g = ff.gradient(coords, types)
    return {
        "energy_kcal_mol": e,
        "grad_norm": float(np.linalg.norm(g)),
    }


def _smoke_relax() -> dict:
    """Run relaxation on methane and report pre/post energy."""
    ff = SimpleForceField()
    relax = MDRelaxation(ff, n_steps=20, step_size=0.005, temperature=100, seed=0)
    coords, types = _build_methane_coords()
    e0 = ff.energy(coords, types)
    relaxed = relax.relax(coords, types)
    e1 = ff.energy(relaxed, types)
    delta = e1 - e0
    return {
        "e_pre_kcal_mol": e0,
        "e_post_kcal_mol": e1,
        "delta_kcal_mol": delta,
        "moved": bool(np.linalg.norm(relaxed - coords) > 1e-6),
    }


def _smoke_validator() -> dict:
    """Run validator on a bad-contact poisoned pair."""
    val = GeometryValidator()
    bad = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    violations = val.report_bad_contacts(bad, ["H", "H"])
    return {
        "n_violations": len(violations),
        "first_reason": violations[0][3] if violations else None,
        "is_reasonable": val.is_reasonable_geometry(bad, ["H", "H"]),
    }


def _smoke_register() -> dict:
    """Run register_md_relaxation hook."""
    n = register_md_relaxation()
    return {"registered_count": n}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke verification for MD relaxation stub (TODO-30 P2.5).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke harness only (default behaviour).",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    print("[1/4] SimpleForceField")
    ff = _smoke_force_field()
    print(
        f"      energy={ff['energy_kcal_mol']:.3f} kcal/mol, "
        f"grad_norm={ff['grad_norm']:.3f}"
    )
    print("[2/4] MDRelaxation")
    rx = _smoke_relax()
    print(
        f"      pre={rx['e_pre_kcal_mol']:.3f}, "
        f"post={rx['e_post_kcal_mol']:.3f}, "
        f"delta={rx['delta_kcal_mol']:+.3f} kcal/mol, "
        f"moved={rx['moved']}"
    )
    print("[3/4] GeometryValidator")
    vt = _smoke_validator()
    print(
        f"      n_violations={vt['n_violations']}, "
        f"reason={vt['first_reason']!r}, "
        f"reasonable={vt['is_reasonable']}"
    )
    print("[4/4] register_md_relaxation")
    rg = _smoke_register()
    print(f"      registered_count={rg['registered_count']}")
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.2f}s")

    ok_ff = math.isfinite(ff["energy_kcal_mol"]) and ff["energy_kcal_mol"] > 0
    ok_rx = rx["delta_kcal_mol"] <= 1e-6 and rx["moved"]
    ok_vt = vt["n_violations"] >= 1 and vt["first_reason"] == "close_contact"
    ok_rg = rg["registered_count"] >= 1
    if ok_ff and ok_rx and ok_vt and ok_rg:
        print("MD_RELAXATION SMOKE PASS")
        return 0
    print("MD_RELAXATION SMOKE FAIL")
    if not ok_ff:
        print(f"  ff energy not finite or non-positive: {ff['energy_kcal_mol']}")
    if not ok_rx:
        print(f"  relax did not lower energy: delta={rx['delta_kcal_mol']}")
    if not ok_vt:
        print(f"  validator did not detect bad contact: {vt}")
    if not ok_rg:
        print(f"  registered_count < 1: {rg['registered_count']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
