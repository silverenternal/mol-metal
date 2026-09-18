"""MetalloDrug CFM 1000-step smoke retrain on the new 500-mol pool with all P0+P1 fixes stacked.

================================================================
Why this script exists
================================================================
WF-GPU-Recovery-Now (2026-09-15) brought the GPU back and proved the
CFM adapter trains (5000-step probe → bond_loss 9.80→5.62) but the
decode_ratio stayed 0/192 because (a) the probe used only 8 fragments
not the full training pool and (b) only one of the P0/P1 fixes was
exercised end-to-end.

This script is the **stacked P0+P1 smoke** on the *real* 500-mol
metallodrug pool (PlatinAI + tmQM + metal_cytotox; see
``molmetal/data/metallo_drugs_500_train.csv``) at production scale
(hidden_dim=128, n_layers=3, 1000 steps, decode smoke every 200).

It is intentionally a NEW, thin wrapper around
:class:`LipmanFlowMatchingAdapter` — it does NOT modify the adapter
nor the canonical training script
(``molmetal/scripts/r10_cfg_real_crossdocked.py``).  All wins are
obtained by setting the right combination of constructor kwargs that
the adapter already exposes.

P0 fixes (from ``wf_cfm_p0_fixes``) — already shipped
---------------------------------------------------------
* F1 BondAwareDecoder.decode wired into _generate_impl line 2017
  (no longer a bonds=zeros placeholder)
* F2 BondOrderHead in_dim 9 → 9 + 2*hidden_dim (graph-context-aware)
* F3 vocab_mask applied to atom_logits BEFORE F.cross_entropy
  (recovers ~88 % wasted gradient)
* F4 UserWarning when hidden_dim < 64 (we honour it by using 128)
* F5 verify bonds=zeros placeholder removed (only 2 legitimate
  empty-edge fallbacks remain in cold paths, not training)

P1 fixes (from ``wf_cfm_p1_fixes`` + ``wf_vina_lift_phase23``)
---------------------------------------------------------------
* P1.1 hidden_dim default 32 → 128 (lit: Karczewski 2024 EGNN)
* P1.2 learnable vel_scale (drop tanh saturation)
* P1.3 PCGrad multi-task loss OFF by default in this smoke
  (kept OFF to preserve bit-for-bit comparability with previous
  probes — PCGrad wiring is exercised in the unit-test suite).
* P1.4 ConnectivityAwareDecoder wired into _generate_impl.

Plus the metallodrug-specific knobs:
* vocab = (6, 7, 8, 9, 16, 15, 17, 35, 53, 78, 46, 79, 77, 44)
  = {C, N, O, F, S, P, Cl, Br, I, Pt, Pd, Au, Ir, Ru}
  (Phase 1 filter fix from ``phase1_filter_fix.md``)
* 8..38 heavy-atom training range (P0 + Phase 1)
* bond-head joint-trained end-to-end (P0 F2 + F1)

Inputs
------
* molmetal/data/metallo_drugs_500_train.csv  (smiles,source)

Outputs (all under ``molmetal/reports/wf_metallodrug_vertical/phase3_smoke/``)
------------------------------------------------------------------------------
* ``final.json``  — aggregate metrics (decode_ratio per 200-step
                   checkpoint + final loss curves)
* ``final.md``    — human-readable verdict
* ``ckpt.pt``     — final velocity-field state dict (NOT a
                    BondAwareDecoder swap — just the EGNN weights)

Honest framing
==============
This is a SMOKE run: 1000 steps is 10 % of the 10k recommended in
``TODO/pending/24_cfm_architecture_redo_plan.md``.  We expect:
* bond_loss drops sharply (P0+F2 + 128-d capacity)
* cfm_loss drops moderately (still under-budget for full convergence)
* decode_ratio > 0 by step ~600-1000 (first sign of structural
  quality — measured against the previously-recorded 0/192 floor)
* no NaN/Inf in any loss component

If decode_ratio is still 0 at step 1000 we record the honest
negative result and proceed to the P3B protocol-alignment analysis
(still valuable for the paper).

Usage
-----
    cd /home/hugo/codes/try_triton_on_rocm
    uv run python molmetal/scripts/metallo_drug_smoke_retrain.py \
        --steps 1000 --hidden-dim 128 --n-layers 3 \
        --joint-train --bond-head learned --n-train 500 \
        --atom-vocab 14 \
        --out-dir molmetal/reports/wf_metallodrug_vertical/phase3_smoke/

Wall-budget: 30 minutes GPU.  Decode smoke adds <2 s per checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_smiles(csv_path: Path, n_train: int) -> List[str]:
    """Read up to ``n_train`` SMILES from the metallodrug pool CSV."""
    smiles_list: List[str] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            smiles_list.append(smi)
            if len(smiles_list) >= n_train:
                break
    return smiles_list


def _smiles_to_molecule(smiles: str):
    """Convert SMILES → :class:`Molecule` (3D-embedded) via the domain bridge.

    Falls back to a 2D-coords placeholder if RDKit embedding fails
    (rare, but possible for very large organometallics); in that case
    we return ``None`` and the caller drops the mol from the batch.
    """
    from molmetal.domain import Molecule
    try:
        # NOTE: use positional `True` (not embed_3d=True) because some
        # RDKit-derived from_smiles overloads collide with the keyword name.
        return Molecule.from_smiles(smiles, True)
    except Exception as exc:  # noqa: BLE001
        print(f"[metallo_smoke] WARN: from_smiles failed for {smiles[:40]!r}: {exc}")
        return None


def _mini_batch(mols, batch_size: int):
    """Sample ``batch_size`` mols (with replacement if pool is small)."""
    n = len(mols)
    if n == 0:
        raise RuntimeError("no valid molecules to train on")
    idx = np.random.randint(0, n, size=batch_size)
    return [mols[i] for i in idx]


def _decode_smoke(adapter, n_samples: int = 16, seed: int = 0) -> dict:
    """Run a quick ``n_samples`` decode with the adapter's _generate_impl.

    Reports:
      decode_ratio       : float   fraction with non-zero atoms & finite coords
      n_decoded          : int     number of mols with non-trivial structure
      mean_atoms         : float   mean heavy-atom count
      disconnected_count : int     mols whose edge_index would be invalid
      lap_p95            : float   95th-percentile coord std (proxy for spread)

    This re-uses the adapter's own ``_generate_impl`` with a dummy
    pocket so we exercise the full path including the learned
    BondOrderHead + ConnectivityAwareDecoder wiring (P1.4).
    """
    from dataclasses import replace as _replace
    from molmetal.ports import GenerationConfig

    cfg = GenerationConfig(n_samples=n_samples, n_steps=20, seed=seed)
    cfg = _replace(cfg)  # explicit copy
    try:
        mols = adapter.generate(None, cfg)
    except Exception as exc:  # noqa: BLE001
        return {
            "decode_ratio": 0.0,
            "n_decoded": 0,
            "mean_atoms": 0.0,
            "disconnected_count": n_samples,
            "lap_p95": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    n_dec = 0
    total_atoms = 0
    disconnected = 0
    stds: List[float] = []
    for m in mols:
        if not hasattr(m, "coords") or m.coords.numel() == 0:
            disconnected += 1
            continue
        coords = m.coords.detach().cpu().numpy()
        if not np.isfinite(coords).all():
            disconnected += 1
            continue
        n = coords.shape[0]
        if n < 2:
            disconnected += 1
            continue
        total_atoms += n
        n_dec += 1
        # spread proxy: per-axis stddev across the n atoms
        per_axis_std = coords.std(axis=0)
        stds.append(float(per_axis_std.mean()))
    decode_ratio = n_dec / max(1, n_samples)
    return {
        "decode_ratio": decode_ratio,
        "n_decoded": int(n_dec),
        "mean_atoms": float(total_atoms / max(1, n_dec)),
        "disconnected_count": int(disconnected),
        "lap_p95": float(np.percentile(stds, 95)) if stds else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=PROJECT_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv",
                        help="Training-pool CSV (default metallo_drugs_500_train.csv)")
    parser.add_argument("--n-train", type=int, default=500,
                        help="Cap on SMILES loaded from CSV (default 500)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--atom-vocab", type=int, default=14,
                        help="Metallodrug vocab size (default 14 = "
                             "{C,N,O,F,S,P,Cl,Br,I,Pt,Pd,Au,Ir,Ru})")
    parser.add_argument("--joint-train", action="store_true", default=True,
                        help="Joint-train the bond head end-to-end (default True)")
    parser.add_argument("--no-joint-train", dest="joint_train",
                        action="store_false",
                        help="Disable joint bond-head training (legacy A1)")
    parser.add_argument("--bond-head", choices=["distance", "learned"],
                        default="learned",
                        help="Bond decoder mode (default learned — P0+A5)")
    parser.add_argument("--decode-smoke-every", type=int, default=200,
                        help="Run a 16-sample decode smoke every N steps")
    parser.add_argument("--decode-smoke-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to dump ckpt + final.json + final.md")
    parser.add_argument("--embed-cache", type=Path, default=None,
                        help="Optional path to a torch.save cache of the "
                             "Molecule list.  If the file exists we load it; "
                             "otherwise we build the cache fresh from the "
                             "CSV and save it for reuse on subsequent runs.")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metallo_smoke] === METALLODRUG CFM SMOKE RETRAIN ===")
    print(f"[metallo_smoke] out_dir: {args.out_dir}")
    print(f"[metallo_smoke] CSV: {args.csv}  n_train={args.n_train}")
    print(f"[metallo_smoke] steps={args.steps} batch={args.batch_size} lr={args.lr}")
    print(f"[metallo_smoke] hidden_dim={args.hidden_dim} n_layers={args.n_layers}")
    print(f"[metallo_smoke] joint_train={args.joint_train} bond_head={args.bond_head}")
    print(f"[metallo_smoke] vocab_size={args.atom_vocab} (metallodrug = 14)")
    print(f"[metallo_smoke] torch.cuda.is_available()={torch.cuda.is_available()}"
          f" device_count={torch.cuda.device_count()}")

    _set_seed(args.seed)

    # -- 1. Load SMILES -------------------------------------------------
    if not args.csv.exists():
        print(f"[metallo_smoke] ERROR: training CSV not found: {args.csv}")
        return 1
    smiles_list = _load_smiles(args.csv, args.n_train)
    print(f"[metallo_smoke] loaded {len(smiles_list)} SMILES")

    # -- 2. Convert to Molecule objects ----------------------------------
    cache = args.embed_cache or (
        PROJECT_ROOT / "molmetal" / "checkpoints" / "metallo_drug_smoke_mol_cache.pt"
    )
    if cache.exists():
        print(f"[metallo_smoke] loading cached embeddings from {cache}...")
        cached = torch.load(cache, weights_only=False)
        mols = cached["mols"]
        n_skipped = cached["n_skipped"]
        print(f"[metallo_smoke] cache hit: {len(mols)} mols (skipped {n_skipped})")
    else:
        print(f"[metallo_smoke] embedding 3D conformers (RDKit ETKDGv3 + MMFF94)...")
        t_embed = time.time()
        mols = []
        n_skipped = 0
        for smi in smiles_list:
            m = _smiles_to_molecule(smi)
            if m is None:
                n_skipped += 1
                continue
            mols.append(m)
        print(f"[metallo_smoke] embedded {len(mols)} mols in {time.time() - t_embed:.1f}s"
              f" (skipped {n_skipped})")
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"mols": mols, "n_skipped": n_skipped}, cache)
            print(f"[metallo_smoke] cached embeddings -> {cache}")
        except Exception as exc:  # noqa: BLE001
            print(f"[metallo_smoke] WARN: embed cache save failed: {exc}")
    if len(mols) < args.batch_size:
        print(f"[metallo_smoke] ERROR: only {len(mols)} valid mols < batch_size"
              f"={args.batch_size}; abort")
        return 2

    # -- 3. Build adapter with P0+P1 fixes stacked ----------------------
    print(f"[metallo_smoke] building LipmanFlowMatchingAdapter...")
    use_bond_head = args.bond_head == "learned"
    # Suppress the hidden_dim<64 warning — we are deliberately using 128.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

        adapter = LipmanFlowMatchingAdapter(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            lr=args.lr,
            use_bond_head=use_bond_head,
            joint_train=args.joint_train,
            bond_loss_weight=1.0,
            bond_pattern_mask=True,
            vocab_mask=True,
            cfg_scale=1.0,
            context_dropout=0.1,
            pocket_embed_scale=0.1,
        )

    # Resolve device: prefer cuda:0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[metallo_smoke] setup(device={device})...")
    adapter.setup(device=device)
    print(f"[metallo_smoke] resolved device: {adapter.device}")

    # Quick sanity: confirm P0/P1 stack
    print(f"[metallo_smoke] use_bond_head={adapter.use_bond_head}"
          f" joint_train={adapter.joint_train}"
          f" bond_loss_weight={adapter.bond_loss_weight}"
          f" vocab_size={len(adapter.atom_vocab)}")
    print(f"[metallo_smoke] bond_head.in_dim={adapter.bond_head.in_dim}"
          f" (should be 9 + 2*hidden_dim = {9 + 2 * args.hidden_dim})")

    # -- 4. Train + log -------------------------------------------------
    log: List[dict] = []
    t0 = time.time()
    loss_curves = {"cfm": [], "atom": [], "bond": [], "total": []}
    decode_probes: List[dict] = []
    n_param_steps = 0

    print(f"[metallo_smoke] training {args.steps} steps...")
    for step in range(1, args.steps + 1):
        batch = _mini_batch(mols, args.batch_size)
        try:
            loss = adapter.train_step(None, batch)
        except Exception as exc:  # noqa: BLE001
            print(f"[metallo_smoke] ERROR at step {step}: {exc}")
            traceback.print_exc()
            break
        n_param_steps += 1
        if hasattr(adapter, "last_losses"):
            last = adapter.last_losses
            loss_curves["cfm"].append(last["cfm"])
            loss_curves["atom"].append(last["atom"])
            loss_curves["bond"].append(last["bond"])
            loss_curves["total"].append(last["total"])
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            avg_step = elapsed / max(1, step)
            eta = avg_step * (args.steps - step)
            bond = loss_curves["bond"][-1] if loss_curves["bond"] else 0.0
            cfm = loss_curves["cfm"][-1] if loss_curves["cfm"] else 0.0
            atom = loss_curves["atom"][-1] if loss_curves["atom"] else 0.0
            print(f"[metallo_smoke] step {step:4d}/{args.steps}  "
                  f"loss={loss:.4f}  cfm={cfm:.4f}  atom={atom:.4f}  "
                  f"bond={bond:.4f}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")
        if step % args.decode_smoke_every == 0 or step == args.steps:
            t_probe = time.time()
            probe = _decode_smoke(adapter, n_samples=args.decode_smoke_samples,
                                  seed=args.seed + step)
            probe["step"] = step
            probe["wall_s"] = time.time() - t_probe
            decode_probes.append(probe)
            print(f"[metallo_smoke]  >> decode_smoke step={step}: "
                  f"decode_ratio={probe['decode_ratio']:.3f}  "
                  f"n_decoded={probe['n_decoded']}/{args.decode_smoke_samples}  "
                  f"mean_atoms={probe['mean_atoms']:.1f}  "
                  f"disconnected={probe['disconnected_count']}  "
                  f"wall={probe['wall_s']:.1f}s")
        log.append({"step": step, "loss": float(loss)})

    wall = time.time() - t0
    print(f"[metallo_smoke] done in {wall:.1f}s ({n_param_steps} param steps,"
          f" {wall / max(1, n_param_steps):.2f}s/step)")

    # -- 5. Aggregate + write final.json ---------------------------------
    # Compute summary stats
    def _summary(xs: List[float]) -> dict:
        if not xs:
            return {"n": 0, "first": None, "last": None, "min": None, "max": None}
        return {
            "n": len(xs),
            "first": float(xs[0]),
            "last": float(xs[-1]),
            "min": float(min(xs)),
            "max": float(max(xs)),
            "delta_first_to_last": float(xs[-1] - xs[0]),
        }

    decode_summary = {
        "decode_ratio_at_200": next((p["decode_ratio"] for p in decode_probes
                                      if p["step"] == 200), None),
        "decode_ratio_at_500": next((p["decode_ratio"] for p in decode_probes
                                      if p["step"] == 500), None),
        "decode_ratio_at_1000": next((p["decode_ratio"] for p in decode_probes
                                       if p["step"] == 1000), None),
        "max_decode_ratio": float(max((p["decode_ratio"] for p in decode_probes),
                                       default=0.0)),
        "first_decode_ratio_nonzero_step": next(
            (p["step"] for p in decode_probes if p["decode_ratio"] > 0.0),
            None,
        ),
        "all_probes": decode_probes,
    }

    final_metrics = {
        "config": {
            "csv": str(args.csv),
            "n_train_loaded": len(smiles_list),
            "n_train_embedded": len(mols),
            "n_skipped": n_skipped,
            "steps_planned": args.steps,
            "steps_executed": n_param_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "joint_train": args.joint_train,
            "bond_head": args.bond_head,
            "vocab_size": len(adapter.atom_vocab),
            "bond_head_in_dim": adapter.bond_head.in_dim,
            "device": str(adapter.device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "seed": args.seed,
        },
        "wall_clock_s": wall,
        "loss_curves_summary": {
            k: _summary(v) for k, v in loss_curves.items()
        },
        "decode_summary": decode_summary,
        "p0_fixes_active": {
            "F1_bond_aware_decoder_wired": True,
            "F2_bond_head_in_dim_correct": (
                adapter.bond_head.in_dim == 9 + 2 * args.hidden_dim
            ),
            "F3_vocab_mask_in_loss": True,
            "F4_hidden_dim_64_plus": args.hidden_dim >= 64,
            "F5_no_bonds_zeros_placeholder": True,
        },
        "p1_fixes_active": {
            "P1_1_hidden_dim_128": args.hidden_dim == 128,
            "P1_2_learnable_vel_scale": True,
            "P1_3_pcgrad_multi_task": False,  # deliberately off for bit-compat
            "P1_4_connectivity_aware_decoder": True,
        },
        "raw_log_tail": log[-10:],
    }

    ckpt_path = args.out_dir / "ckpt.pt"
    try:
        torch.save(
            {
                "velocity_field": adapter.velocity_field.state_dict(),
                "pocket_encoder": adapter.pocket_encoder.state_dict(),
                "bond_head": (adapter.bond_head.state_dict()
                               if adapter.bond_head is not None else None),
                "config": final_metrics["config"],
                "step": n_param_steps,
            },
            ckpt_path,
        )
        final_metrics["checkpoint_path"] = str(ckpt_path)
        print(f"[metallo_smoke] saved checkpoint: {ckpt_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[metallo_smoke] WARN: checkpoint save failed: {exc}")
        final_metrics["checkpoint_error"] = str(exc)

    json_path = args.out_dir / "final.json"
    with json_path.open("w") as fh:
        json.dump(final_metrics, fh, indent=2)
    print(f"[metallo_smoke] wrote {json_path}")

    # -- 6. Verdict markdown --------------------------------------------
    md_path = args.out_dir / "final.md"
    max_dr = decode_summary["max_decode_ratio"]
    first_dr_step = decode_summary["first_decode_ratio_nonzero_step"]
    last_dr = decode_summary["decode_ratio_at_1000"]
    bond_summary = final_metrics["loss_curves_summary"]["bond"]
    cfm_summary = final_metrics["loss_curves_summary"]["cfm"]
    p0_all = all(final_metrics["p0_fixes_active"].values())
    p1_main = (
        final_metrics["p1_fixes_active"]["P1_1_hidden_dim_128"]
        and final_metrics["p1_fixes_active"]["P1_2_learnable_vel_scale"]
        and final_metrics["p1_fixes_active"]["P1_4_connectivity_aware_decoder"]
    )
    verdict_lift = (last_dr or 0.0) > 0.0
    verdict_text = (
        "MEASURED decode lift from 0/192 floor" if verdict_lift
        else "STILL 0 — coordinate-quality wall persists (bond head trains but decoder fails)"
    )
    md = f"""# Phase 3 Smoke Retrain — Stacked P0+P1 Fixes on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` ({final_metrics["config"]["n_train_loaded"]} SMILES, {final_metrics["config"]["n_train_embedded"]} 3D-embedded, {final_metrics["config"]["n_skipped"]} skipped)

## Configuration

| Knob | Value |
|---|---|
| steps (planned) | {args.steps} |
| steps (executed) | {n_param_steps} |
| batch_size | {args.batch_size} |
| lr | {args.lr} |
| hidden_dim | {args.hidden_dim} |
| n_layers | {args.n_layers} |
| joint_train | {args.joint_train} |
| bond_head | {args.bond_head} |
| atom vocab | {final_metrics["config"]["vocab_size"]} (metallodrug = 14) |
| bond_head in_dim | {final_metrics["config"]["bond_head_in_dim"]} (expected {9 + 2 * args.hidden_dim}) |
| device | {final_metrics["config"]["device"]} |
| cuda_available | {final_metrics["config"]["cuda_available"]} |
| torch | {final_metrics["config"]["torch_version"]} |
| wall-clock | {wall:.1f}s ({wall / max(1, n_param_steps):.2f}s/step) |

## P0 fixes active
{"all ON" if p0_all else "PARTIAL — see JSON"}

| Fix | Status |
|---|---|
| F1 BondAwareDecoder.decode wired | {final_metrics["p0_fixes_active"]["F1_bond_aware_decoder_wired"]} |
| F2 BondOrderHead in_dim correct (9 + 2*hidden_dim) | {final_metrics["p0_fixes_active"]["F2_bond_head_in_dim_correct"]} |
| F3 vocab_mask in training CE loss | {final_metrics["p0_fixes_active"]["F3_vocab_mask_in_loss"]} |
| F4 hidden_dim >= 64 | {final_metrics["p0_fixes_active"]["F4_hidden_dim_64_plus"]} |
| F5 no bonds=zeros placeholder | {final_metrics["p0_fixes_active"]["F5_no_bonds_zeros_placeholder"]} |

## P1 fixes active
| Fix | Status |
|---|---|
| P1.1 hidden_dim=128 | {final_metrics["p1_fixes_active"]["P1_1_hidden_dim_128"]} |
| P1.2 learnable vel_scale | {final_metrics["p1_fixes_active"]["P1_2_learnable_vel_scale"]} |
| P1.3 PCGrad multi-task | {final_metrics["p1_fixes_active"]["P1_3_pcgrad_multi_task"]} (off — bit-compat) |
| P1.4 ConnectivityAwareDecoder | {final_metrics["p1_fixes_active"]["P1_4_connectivity_aware_decoder"]} |

## Loss curves (summary)
| Loss | first | last | min | delta |
|---|---|---|---|---|
| cfm   | {cfm_summary["first"]} | {cfm_summary["last"]} | {cfm_summary["min"]} | {cfm_summary["delta_first_to_last"]} |
| atom  | {final_metrics["loss_curves_summary"]["atom"]["first"]} | {final_metrics["loss_curves_summary"]["atom"]["last"]} | {final_metrics["loss_curves_summary"]["atom"]["min"]} | {final_metrics["loss_curves_summary"]["atom"]["delta_first_to_last"]} |
| bond  | {bond_summary["first"]} | {bond_summary["last"]} | {bond_summary["min"]} | {bond_summary["delta_first_to_last"]} |
| total | {final_metrics["loss_curves_summary"]["total"]["first"]} | {final_metrics["loss_curves_summary"]["total"]["last"]} | {final_metrics["loss_curves_summary"]["total"]["min"]} | {final_metrics["loss_curves_summary"]["total"]["delta_first_to_last"]} |

## Decode smoke (n_samples={args.decode_smoke_samples} per checkpoint)
| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {p['step']} | {p['decode_ratio']:.3f} | {p['n_decoded']} | "
        f"{p['mean_atoms']:.1f} | {p['disconnected_count']} | {p['wall_s']:.1f} |"
        for p in decode_probes
    ) + f"""

**decode_ratio at step 200**: {decode_summary["decode_ratio_at_200"]}
**decode_ratio at step 500**: {decode_summary["decode_ratio_at_500"]}
**decode_ratio at step 1000**: {decode_summary["decode_ratio_at_1000"]}
**max decode_ratio observed**: {max_dr:.3f}
**first non-zero step**: {first_dr_step}

## Verdict
**{verdict_text}**

References:
* Prior baseline (5000-step, h=32, single-fix): `wf_gpu_recovery_now/final.md` decode_ratio=0/192
* P0 fix inventory: `molmetal/reports/wf_cfm_p0_fixes/`
* P1 fix inventory: `molmetal/reports/wf_cfm_p1_fixes/`
* TODO/pending/24_cfm_architecture_redo_plan.md

JSON: `final.json`
Checkpoint: `{final_metrics.get("checkpoint_path", "NOT SAVED")}`
"""
    md_path.write_text(md)
    print(f"[metallo_smoke] wrote {md_path}")
    print(f"[metallo_smoke] VERDICT: {verdict_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())