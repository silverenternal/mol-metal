"""WF-CFM-Rescue Phase 5 — actual 200-step 8-sample decode smoke runner.

Runs a single LipmanFlowMatchingAdapter with the Phase 1+2+3+4 fixes
applied and reports decode_ratio.  CPU-only; takes ~30 s.

Honest expectation (per wf_cfm_frontier_research/final.md): on a fresh
random init, decode_ratio = 0/8.  The smoke is here to confirm the
*path* runs end-to-end with the fixes (no NaN, no crash) and to provide
a baseline against future GPU retrains.

Usage:
    uv run python molmetal/reports/wf_cfm_rescue/run_200step_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Pocket
from molmetal.ports import GenerationConfig
from molmetal.scripts.r10_cfg_real_crossdocked import SizedGenerationConfig, run_decode_smoke


def main():
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2)
    adapter.setup(device="cpu")
    # Minimal 8-atom pocket context (the smallest the harness can sample)
    g = torch.Generator().manual_seed(0)
    pcoords = torch.randn(12, 3, generator=g) * 2.0
    pocket = Pocket(
        pdb_id="rescue_200step",
        coords=pcoords,
        atom_types=torch.randint(1, 18, (12,), generator=g),
        residue_ids=torch.zeros(12, dtype=torch.long),
        chain_ids=torch.zeros(12, dtype=torch.long),
        mask=torch.ones(12, dtype=torch.bool),
        center=pcoords.mean(0),
        radius=6.0,
    )
    n_decoded, mols = run_decode_smoke(adapter, pocket, n_samples=8, n_steps=200)
    decode_ratio = n_decoded / 8
    print(f"decode_ratio: {n_decoded}/8 = {decode_ratio:.3f}")
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase5_200step_smoke.json").write_text(json.dumps({
        "decode_ratio_n": n_decoded,
        "decode_ratio_total": 8,
        "decode_ratio": decode_ratio,
        "hidden_dim": 64,
        "n_layers": 2,
        "n_steps": 200,
        "n_samples": 8,
        "method": "midpoint",
        "bond_head": "learned",
        "joint_train": True,
        "n_train_default": 32,
        "notes": (
            "200-step 8-sample decode smoke on freshly-init h=64 adapter. "
            "Honest expectation: 0/8 (per wf_cfm_frontier_research/final.md). "
            "This smoke confirms the path runs end-to-end with no NaN/crash."
        ),
    }, indent=2) + "\n")
    print(f"Wrote {out_dir}/phase5_200step_smoke.json")


if __name__ == "__main__":
    main()
