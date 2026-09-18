"""WF-R15-CFM-CPU-Verify Phase 1 — 500-step CPU smoke with all 4 fixes stacked.

Runs a single LipmanFlowMatchingAdapter with:
- YuelBond decoder module (available, see phase2_yuelbond_wire.md)
- joint_train=True (default flipped in WF-CFM-Frontier-Phase2 Fix #1)
- n_train=32 default (WF-CFM-Rescue Phase 3)
- midpoint ODE solver (WF-CFM-Frontier-Phase2 Fix #2)
- rectified flow x_0 (DOC: NOT YET WIRED — x_0 = randn still)

Reports decode_ratio at checkpoints step=100, 200, 500.

CPU-friendly: ~60-90 s on RX 7800 XT (CPU-only path).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Pocket
from molmetal.ports import GenerationConfig
from molmetal.scripts.r10_cfg_real_crossdocked import SizedGenerationConfig


def build_pocket(seed: int = 0, n_atoms: int = 12) -> Pocket:
    """Build a minimal dummy pocket context."""
    g = torch.Generator().manual_seed(seed)
    pcoords = torch.randn(n_atoms, 3, generator=g) * 2.0
    return Pocket(
        pdb_id=f"r15_500step_{seed}",
        coords=pcoords,
        atom_types=torch.randint(1, 18, (n_atoms,), generator=g),
        residue_ids=torch.zeros(n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms, dtype=torch.long),
        mask=torch.ones(n_atoms, dtype=torch.bool),
        center=pcoords.mean(0),
        radius=6.0,
    )


def run_smoke(n_steps: int, n_samples: int, seed: int = 0) -> tuple:
    """Run a single decode smoke and return (n_decoded, mols, elapsed)."""
    torch.manual_seed(seed)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2)
    adapter.setup(device="cpu")
    pocket = build_pocket(seed=seed)
    config = SizedGenerationConfig(n_samples=n_samples, n_steps=n_steps, seed=seed)
    t0 = time.monotonic()
    mols = adapter.generate(pocket, config)
    elapsed = time.monotonic() - t0
    n_decoded = sum(1 for m in mols if m.bonds is not None and m.bonds.shape[-1] > 0)
    return n_decoded, mols, elapsed


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[WF-R15-Phase1] CPU 500-step smoke with all 4 fixes stacked")
    print(f"[WF-R15-Phase1] hidden_dim=64, n_layers=2, joint_train=True")
    print(f"[WF-R15-Phase1] n_train=32 default, midpoint solver, BondAwareDecoder")
    print(f"[WF-R15-Phase1] NOTE: x_0=randn (rectified flow deferred to retrain)")
    print()
    results = {}
    for n_steps in (100, 200, 500):
        n_dec, mols, elapsed = run_smoke(n_steps=n_steps, n_samples=8, seed=0)
        ratio = n_dec / 8
        results[f"step_{n_steps}"] = {
            "n_decoded": n_dec,
            "n_total": 8,
            "decode_ratio": ratio,
            "elapsed_s": elapsed,
        }
        print(f"[step={n_steps:>3}] decode_ratio = {n_dec}/8 = {ratio:.3f}  ({elapsed:.1f}s)")
    out = {
        "phase": "WF-R15-Phase1",
        "timestamp": "2026-09-16",
        "device": "cpu",
        "hidden_dim": 64,
        "n_layers": 2,
        "method": "midpoint",
        "bond_head": "learned",
        "joint_train": True,
        "n_train_default": 32,
        "yuelbond_wired": False,  # NOTE: YuelBond is available module, NOT yet wired in production
        "rectified_flow_x0": False,  # NOTE: x_0=randn, rectified flow deferred
        "checkpoints": results,
        "notes": (
            "Phase 1 of WF-R15-CFM-CPU-Verify: 500-step CPU smoke with 4 fixes "
            "stacked (joint_train=True + n_train=32 + midpoint solver + BondAwareDecoder). "
            "YuelBond module is available but NOT yet wired into _generate_impl; "
            "see phase2_yuelbond_wire.md for the parallel-path architecture. "
            "Rectified flow (x_0=0) NOT yet wired — deferred to next retrain per "
            "WF-CFM-Rescue Phase 4 documentation."
        ),
    }
    out_path = out_dir / "phase1_500step_smoke.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()