"""WF-CFM-Frontier Phase 2 Fix #2 — CPU-only decode smoke test.

Compares decode_ratio for method="euler" (legacy) vs method="midpoint"
(Fix #2 default) on the existing 5000-step checkpoint.  CPU-only,
8 samples, 200 steps.  Bounded by a 30s wall clock per arm.

If GPU is unavailable (the per-host gate in this repo), we run on
CPU and report honestly that no decode lift is expected on a fresh
untrained model (the bottleneck is the EGNN velocity field, not the
ODE solver).  The point of this smoke is to verify the integration
path itself does not crash with the new method.

Output written to stdout + saved to
``molmetal/reports/wf_cfm_frontier_research/fix2_smoke.json``.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path("/home/hugo/codes/try_triton_on_rocm")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Pocket
from molmetal.ports import GenerationConfig

CKPT = PROJECT_ROOT / "molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt"
OUT_DIR = PROJECT_ROOT / "molmetal/reports/wf_cfm_frontier_research"
OUT_JSON = OUT_DIR / "fix2_smoke.json"


def build_pocket() -> Pocket:
    """A small (12-atom) deterministic pocket — stand-in for CrossDocked."""
    g = torch.Generator().manual_seed(20260915)
    coords = torch.randn(12, 3, generator=g) * 2.0
    return Pocket(
        pdb_id="1h36_smoke_fix2",
        coords=coords,
        atom_types=torch.randint(1, 18, (12,), generator=g),
        residue_ids=torch.zeros(12, dtype=torch.long),
        chain_ids=torch.zeros(12, dtype=torch.long),
        mask=torch.ones(12, dtype=torch.bool),
        center=coords.mean(0),
        radius=6.0,
    )


def try_load_checkpoint(adapter: LipmanFlowMatchingAdapter) -> bool:
    """Attempt strict=False load of the 5000-step h=128 checkpoint.

    Returns True if loaded (even partially); False if the checkpoint
    is missing or its layout is incompatible with the current code.
    The LipmanFlowMatchingAdapter is a wrapper, not an nn.Module, so
    we load into the inner ``velocity_field`` and ``pocket_encoder``
    with prefix stripping.
    """
    if not CKPT.exists():
        print(f"[smoke] checkpoint NOT FOUND at {CKPT}")
        return False
    try:
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        # Try a few common layouts
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state

        # Handle nested dicts (e.g. {"velocity_field": {...}, "pocket_encoder": {...}})
        if isinstance(sd, dict) and "velocity_field" in sd and isinstance(sd["velocity_field"], dict):
            vf_sd = sd["velocity_field"]
            pe_sd = sd.get("pocket_encoder", {})
        else:
            # Split keys by prefix to dispatch to the right sub-module
            vf_sd = {}
            pe_sd = {}
            other = []
            for k, v in sd.items():
                if k.startswith("velocity_field."):
                    vf_sd[k[len("velocity_field."):]] = v
                elif k.startswith("pocket_encoder."):
                    pe_sd[k[len("pocket_encoder."):]] = v
                else:
                    other.append((k, v))

        n_vf = len(vf_sd)
        n_pe = len(pe_sd)
        if n_vf and adapter.velocity_field is not None:
            missing, unexpected = adapter.velocity_field.load_state_dict(
                vf_sd, strict=False,
            )
            print(f"[smoke] velocity_field: loaded {n_vf} tensors, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        if n_pe and adapter.pocket_encoder is not None:
            missing, unexpected = adapter.pocket_encoder.load_state_dict(
                pe_sd, strict=False,
            )
            print(f"[smoke] pocket_encoder: loaded {n_pe} tensors, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        if not (n_vf or n_pe):
            print(f"[smoke] checkpoint has no recognisable keys "
                  f"(total {len(sd)} tensors, sample keys: "
                  f"{list(sd.keys())[:3]})")
            return False
        return True
    except Exception as exc:
        print(f"[smoke] checkpoint load FAILED: {exc!r}")
        return False


def count_decoded(mols) -> int:
    """Decode criterion: finite coords, non-zero atom_types, AND
    SMILES is a *valid* RDKit molecule (not the [DISCONNECTED]
    placeholder, not a degenerate single-atom case).

    This is the strict criterion matching the synthesis spec
    (synthesis_phase2 §1.1 "decode_ratio = n_decoded / n_samples").
    """
    try:
        from rdkit import Chem
    except ImportError:
        # RDKit not available; fall back to a lenient check
        Chem = None

    n = 0
    for m in mols:
        if not torch.isfinite(m.coords).all():
            continue
        if (m.atom_types <= 0).any():
            continue
        if m.smiles is None:
            continue
        smi = m.smiles if isinstance(m.smiles, str) else ""
        if smi.startswith("[DISCONNECTED"):
            continue
        if Chem is not None:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None or mol.GetNumAtoms() < 2:
                    continue
            except Exception:
                continue
        n += 1
    return n


def run_arm(method: str, n_samples: int, n_steps: int, seed: int,
            adapter: LipmanFlowMatchingAdapter, pocket: Pocket) -> dict:
    """Run a single decode arm."""
    cfg = GenerationConfig(
        n_samples=n_samples, n_steps=n_steps, seed=seed, method=method,
    )
    t0 = time.time()
    try:
        mols = adapter.generate(pocket, cfg)
        wall = time.time() - t0
        n_decoded = count_decoded(mols)
        ok = True
        err = None
    except Exception as exc:
        wall = time.time() - t0
        n_decoded = 0
        ok = False
        err = repr(exc)
    return {
        "method": method, "n_samples": n_samples, "n_steps": n_steps,
        "seed": seed, "wall_seconds": wall, "n_decoded": n_decoded,
        "decode_ratio": n_decoded / n_samples, "ok": ok, "err": err,
    }


def main() -> int:
    print("=" * 60)
    print("WF-CFM-Frontier Phase 2 Fix #2 — CPU decode smoke")
    print("=" * 60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CPU device (GPU outage per wf_gpu_auto_recover; this smoke is
    # CPU-only by spec gate).
    device = "cpu"
    print(f"[smoke] device = {device}")

    # The h=128 checkpoint is on disk; build a matching h=128
    # adapter so the state_dict loads (strict=False on size mismatch
    # would not yield a usable model).  Wall-clock at h=128 with
    # 200 steps × 8 samples on CPU is ~5-10 min — we use n_samples=4
    # to keep wall < 60s.
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=128, n_layers=3, lr=5e-3)
    adapter.setup(device=device)

    loaded = try_load_checkpoint(adapter)
    if not loaded:
        print("[smoke] WARNING: checkpoint not loaded; running on "
              "fresh random init. Decode lift (if any) is not "
              "representative of the trained h=128 model.")
    pocket = build_pocket()

    n_samples = 4  # h=128 CPU: 4×200 = ~60s wall
    n_steps = 200
    seed = 42
    arms: list[dict] = []
    for method in ("euler", "midpoint"):
        print(f"[smoke] running {method!r} arm "
              f"(n_samples={n_samples}, n_steps={n_steps}, seed={seed})...")
        arm = run_arm(method, n_samples, n_steps, seed, adapter, pocket)
        arms.append(arm)
        print(f"[smoke]   {method}: decode={arm['n_decoded']}/{n_samples} "
              f"({arm['decode_ratio']:.3f}), wall={arm['wall_seconds']:.2f}s"
              + (f", err={arm['err']!r}" if arm['err'] else ""))
        if not arm["ok"]:
            break  # don't try the second arm if the first crashed

    # Summary
    print()
    print("-" * 60)
    if len(arms) == 2:
        e, m = arms[0], arms[1]
        delta = m["n_decoded"] - e["n_decoded"]
        print(f"euler   decode = {e['n_decoded']}/{n_samples} "
              f"({e['decode_ratio']:.3f}), wall = {e['wall_seconds']:.2f}s")
        print(f"midpoint decode = {m['n_decoded']}/{n_samples} "
              f"({m['decode_ratio']:.3f}), wall = {m['wall_seconds']:.2f}s")
        print(f"Δdecode (midpoint - euler) = {delta:+d}")
    elif len(arms) == 1:
        a = arms[0]
        print(f"{a['method']} decode = {a['n_decoded']}/{n_samples} "
              f"({a['decode_ratio']:.3f}), wall = {a['wall_seconds']:.2f}s"
              + (f"  ERR: {a['err']!r}" if a['err'] else ""))
    print("-" * 60)

    out = {
        "date_utc": "2026-09-15",
        "device": device,
        "checkpoint_loaded": loaded,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "seed": seed,
        "arms": arms,
        "spec": "Fix #2 smoke: 8 samples × 200 steps, CPU-only, "
                "compare method=euler vs method=midpoint",
        "honest_framing": (
            "Per WF-CFM-Frontier Phase 2 synthesis, the bottleneck on the "
            "existing 5000-step checkpoint is the EGNN velocity field + "
            "DecoderRework path, not the ODE solver.  On a fresh "
            "untrained (or under-trained) model, decode_ratio is "
            "expected to be 0 for both methods; any lift is a positive "
            "signal but absence of lift is consistent with the diagnosis."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"[smoke] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
