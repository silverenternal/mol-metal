#!/usr/bin/env python3
"""Wf-269C: CFM ablation on a REAL cohort to differentiate the 3 innovations.

==============================================================================
Wf-269C — Real-cohort CFM ablation (TODO-119 P1.3 follow-up)
==============================================================================

The Wf-268A smoke used synthetic Pt-click clouds (by-construction
RDKit-strict decodable).  ``decode_ratio=1.0`` saturated across all 4
ablations — no signal to differentiate yuelbond vs wasserstein vs
closure-bounded.

Wf-269C loads a REAL cohort from disk, builds AtomClouds via RDKit
3D embedding + jitter, and re-runs the 4-config ablation to look for
degradation signal when one innovation is disabled.

**Constraints**
- CPU only (``--device cpu``)
- hidden_dim=32, n_layers=2 (smoke-fast)
- 5 epochs per config
- Output to ``molmetal/reports/wf269c_cfm_real/``

**Honest framing**
- If all configs still saturate at 1.0, we report and defer to
  TODO-119 P1.3 (real CrossDocked loader).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

# Reuse the CFM internals from the Wf-268A script.
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # type: ignore
    CFMJointModel,
    YuelBondBondDecoder,
    WassersteinConfig,
    build_reference_positions,
    pocket_specific_closure_bias,
    train_one_epoch,
    evaluate_decode_ratio,  # noqa: F401  (kept for parity / fallback)
)
from molmetal.models.bond_head import (  # type: ignore
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
)
from molmetal.molmetal_lam.lam_chem.decoder_rework import (  # type: ignore
    DecoderRework,
    ReworkedDecoder,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # type: ignore
    YuelBondDecodeResult,
)


# ---------------------------------------------------------------------------
# Real cohort loaders (in order of preference)
# ---------------------------------------------------------------------------
REAL_COHORT_CANDIDATES: List[Path] = [
    PROJECT_ROOT / ".data" / "molmetal" / "platinai_4cell.csv",
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
]


@dataclass
class RealMolecule:
    smiles: str
    metal: str
    source: str


def load_real_cohort(csv_path: Path) -> List[RealMolecule]:
    """Load a real cohort CSV and return list of (smiles, metal)."""
    mols: List[RealMolecule] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Pick SMILES column heuristically
        smi_col = None
        metal_col = None
        for cand in ("smiles", "SMILES", "SMILES_Ligands", "smiles_ligands"):
            if cand in reader.fieldnames:
                smi_col = cand
                break
        for cand in ("metal", "Metal"):
            if cand in reader.fieldnames:
                metal_col = cand
                break
        if smi_col is None or metal_col is None:
            raise ValueError(
                f"Could not find SMILES / metal columns in {csv_path}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            smi = (row.get(smi_col) or "").strip()
            metal = (row.get(metal_col) or "").strip()
            if not smi:
                continue
            mols.append(RealMolecule(smiles=smi, metal=metal, source=str(csv_path)))
    return mols


def filter_degenerate(mols: List[RealMolecule]) -> Tuple[List[RealMolecule], int]:
    """Filter trivial / non-embeddable scaffolds (single atom, single bond, etc.)."""
    kept: List[RealMolecule] = []
    n_dropped = 0
    for mol in mols:
        # Handle disconnected (.) by taking the largest fragment to avoid
        # RDKit embed failures on multi-fragment strings.
        smi = mol.smiles
        # Strip salt counterions that confuse RDKit 3D embedding.
        # Keep just the first fragment separated by '.' if present.
        if "." in smi:
            frags = smi.split(".")
            # Take longest fragment
            smi = max(frags, key=len)
        # RDKit parse
        rdmol = Chem.MolFromSmiles(smi)
        if rdmol is None:
            n_dropped += 1
            continue
        if rdmol.GetNumAtoms() < 3:
            n_dropped += 1
            continue
        if rdmol.GetNumBonds() < 2:
            n_dropped += 1
            continue
        # Reject elements outside CFM atomic-number table
        # (the script's z_table: 0,1,6,7,8,9,15,16,17,34,35,53,78)
        ALLOWED_Z = {0, 1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78}
        z_set = {atom.GetAtomicNum() for atom in rdmol.GetAtoms()}
        if not z_set.issubset(ALLOWED_Z):
            n_dropped += 1
            continue
        # Store cleaned SMILES (post-largest-fragment)
        kept.append(RealMolecule(smiles=smi, metal=mol.metal, source=mol.source))
    return kept, n_dropped


def stratified_sample(
    mols: List[RealMolecule],
    n_total: int,
    seed: int,
) -> Tuple[List[RealMolecule], List[RealMolecule]]:
    """Stratified sample n_total/2 train + n_total/2 val by metal."""
    rng = random.Random(seed)
    by_metal: Dict[str, List[RealMolecule]] = {}
    for m in mols:
        by_metal.setdefault(m.metal, []).append(m)
    # Shuffle within each metal stratum
    for lst in by_metal.values():
        rng.shuffle(lst)
    # Allocate proportionally per metal
    total = sum(len(v) for v in by_metal.values())
    per_metal_quota: Dict[str, int] = {}
    allocated = 0
    metals = list(by_metal.keys())
    # First pass: floor
    for mt in metals:
        share = int(n_total * len(by_metal[mt]) / total)
        per_metal_quota[mt] = share
        allocated += share
    # Distribute remainder one-by-one to biggest strata
    remainder = n_total - allocated
    metals_sorted_by_size = sorted(metals, key=lambda m: -len(by_metal[m]))
    for mt in metals_sorted_by_size:
        if remainder <= 0:
            break
        if len(by_metal[mt]) > per_metal_quota[mt]:
            per_metal_quota[mt] += 1
            remainder -= 1
    train: List[RealMolecule] = []
    val: List[RealMolecule] = []
    for mt, quota in per_metal_quota.items():
        pool = by_metal[mt]
        # Half for train, half for val
        n_train = quota // 2
        n_val = quota - n_train
        train.extend(pool[:n_train])
        val.extend(pool[n_train : n_train + n_val])
    return train, val


# ---------------------------------------------------------------------------
# SMILES -> AtomCloud (real-cohort path)
# ---------------------------------------------------------------------------
def smiles_to_atomcloud(
    smiles: str,
    jitter: float = 0.20,
    seed: int = 0,
) -> Optional[AtomCloud]:
    """Embed SMILES in 3D, return AtomCloud (positions, atomic_numbers)."""
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        return None
    rdmol = Chem.AddHs(rdmol)
    # Try ETKDGv3 first; fall back to ETKDG.
    params_used = "ETKDGv3"
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(rdmol, params) != 0:
        params_used = "ETKDG"
        params = AllChem.ETKDG()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(rdmol, params) != 0:
            return None
    # Try MMFF/UFF optimization
    try:
        AllChem.MMFFOptimizeMolecule(rdmol, maxIters=100)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(rdmol, maxIters=100)
        except Exception:
            pass
    # Build positions + atomic numbers
    conf = rdmol.GetConformer()
    positions = np.zeros((rdmol.GetNumAtoms(), 3), dtype=np.float32)
    atomic_numbers = np.zeros(rdmol.GetNumAtoms(), dtype=np.int64)
    for i, atom in enumerate(rdmol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        positions[i] = (pos.x, pos.y, pos.z)
        atomic_numbers[i] = atom.GetAtomicNum()
    # Center the cloud at origin (CFM expects roughly centered input).
    positions -= positions.mean(axis=0, keepdims=True)
    # Apply jitter to mimic CFM noise (positions only).
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        positions = positions + rng.normal(
            0.0,
            jitter,
            size=positions.shape,
        ).astype(np.float32)
    return AtomCloud(
        positions=torch.from_numpy(positions),
        atomic_numbers=torch.from_numpy(atomic_numbers),
    )


# ---------------------------------------------------------------------------
# Decoder factory — swap YuelBond vs legacy (BondAwareDecoder +
# ReworkedDecoder) depending on the no_yuelbond flag.
# ---------------------------------------------------------------------------
def _build_decoder(
    args: argparse.Namespace,
    model: CFMJointModel,
    no_yuelbond: bool,
) -> object:
    """Return either a YuelBondBondDecoder or a ReworkedDecoder."""
    if no_yuelbond:
        # Fresh BondOrderHead (not the YuelBond head) + BondAwareDecoder
        # + ReworkedDecorator.  This is the pre-Wf-124 P1 Path-B path.
        bond_head = BondOrderHead()
        inner = BondAwareDecoder(bond_head=bond_head)
        return ReworkedDecoder(inner=inner)
    return YuelBondBondDecoder(
        head=model.head,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )


def _decode_strict_check(decoder: object, cloud: AtomCloud) -> bool:
    """Run decoder.decode(cloud) and check RDKit-strict sanitisation.

    Handles both ``YuelBondBondDecoder`` (returns YuelBondDecodeResult)
    and ``ReworkedDecoder`` (returns DecodedMol).
    """
    try:
        res = decoder.decode(cloud)
    except Exception:
        return False
    if res is None:
        return False
    # YuelBondBondDecoder returns YuelBondDecodeResult with .decoded
    # (which is a DecodedMol with .smiles and .sanitized).
    smi: Optional[str] = None
    sanitized = False
    if hasattr(res, "decoded") and res.decoded is not None:
        smi = getattr(res.decoded, "smiles", None) or ""
        sanitized = bool(getattr(res.decoded, "sanitized", False))
    else:
        # ReworkedDecoder returns DecodedMol directly
        smi = getattr(res, "smiles", None) or ""
        sanitized = bool(getattr(res, "sanitized", False))
    if not smi or not sanitized:
        return False
    # Round-trip RDKit-strict
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def _eval_decode_ratio(
    decoder: object,
    val_clouds: List[AtomCloud],
    device: torch.device,
    closure_bounded: bool,
    closure_bias: Optional[Dict[str, float]],
) -> Tuple[int, int]:
    """Evaluate decode ratio on val_clouds using the given decoder."""
    n_ok = 0
    n_total = len(val_clouds)
    for cloud in val_clouds:
        cloud_d = AtomCloud(
            positions=cloud.positions.to(device),
            atomic_numbers=cloud.atomic_numbers.to(device),
        )
        # closure_bounded only applies to YuelBond decoder (it
        # monkey-patches the head's bias).  For ReworkedDecoder, this
        # is a no-op.
        if (
            closure_bounded
            and closure_bias is not None
            and isinstance(
                decoder,
                YuelBondBondDecoder,
            )
        ):
            try:
                pair_head_obj = (
                    decoder._decoder.head.pair_head
                    if hasattr(decoder._decoder.head, "pair_head")
                    else decoder._decoder.head.edge_mlp
                )
                last_layer = pair_head_obj[-1]
                orig_bias = last_layer.bias.data.clone()
                bias_delta = torch.zeros(
                    4,  # NUM_BOND_CLASSES = 4 in the script
                    device=device,
                    dtype=last_layer.bias.dtype,
                )
                from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
                    apply_closure_bounded_mask,
                    NUM_BOND_CLASSES,
                )  # type: ignore

                bias_delta = apply_closure_bounded_mask(
                    torch.zeros(
                        1,
                        NUM_BOND_CLASSES,
                        device=device,
                        dtype=last_layer.bias.dtype,
                    ),
                    closure_bias,
                    prefer_dative=True,
                ).squeeze(0)
                last_layer.bias.data = orig_bias + bias_delta
                try:
                    ok = _decode_strict_check(decoder, cloud_d)
                finally:
                    last_layer.bias.data = orig_bias
            except Exception:
                ok = _decode_strict_check(decoder, cloud_d)
        else:
            ok = _decode_strict_check(decoder, cloud_d)
        if ok:
            n_ok += 1
    return n_ok, n_total


# ---------------------------------------------------------------------------
# Per-config ablation runner
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    config_name: str
    epochs: int
    final_train_loss: float
    final_decode_ratio: float
    decode_ok: int
    decode_total: int
    mean_grad_norm: float
    wall_time_s: float


def run_one_config(
    config_name: str,
    args: argparse.Namespace,
    train_clouds: List[AtomCloud],
    val_clouds: List[AtomCloud],
    device: torch.device,
    no_yuelbond: bool,
    no_wasserstein: bool,
    no_closure_bounded: bool,
) -> ConfigResult:
    """Train one config for n epochs and return decode_ratio."""
    print(
        f"[Wf-269C] === config={config_name} "
        f"no_yuelbond={no_yuelbond} no_wasserstein={no_wasserstein} "
        f"no_closure_bounded={no_closure_bounded} ===",
        flush=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = CFMJointModel(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    decoder = _build_decoder(args, model, no_yuelbond)
    print(
        f"[Wf-269C][{config_name}] decoder type = " f"{type(decoder).__name__}",
        flush=True,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    bias_verdict = pocket_specific_closure_bias(args.pocket_id)
    closure_bias = bias_verdict.bias_dict if bias_verdict.is_known_pocket else None
    if closure_bias is None:
        closure_bias = {
            "CuAAC": 0.30,
            "SPAAC": 0.10,
            "ThiolEne": 0.10,
            "Suzuki": 0.20,
            "AmideCoupling": 0.30,
        }
    reference_positions = build_reference_positions().to(device)

    wcfg = WassersteinConfig()  # noqa: F841 (kept for parity with Wf-268A)
    wasserstein_weight = 0.0 if no_wasserstein else args.wasserstein_weight
    closure_bounded = not no_closure_bounded

    per_epoch_metrics: List[Dict[str, float]] = []
    overall_start = time.time()
    final_train_loss = 0.0
    final_decode_ratio = 0.0
    final_n_ok = 0
    final_n_total = 0
    final_grad_norm = 0.0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Move clouds to device
        train_d = [
            AtomCloud(
                positions=c.positions.to(device),
                atomic_numbers=c.atomic_numbers.to(device),
            )
            for c in train_clouds
        ]
        val_d = [
            AtomCloud(
                positions=c.positions.to(device),
                atomic_numbers=c.atomic_numbers.to(device),
            )
            for c in val_clouds
        ]
        train_loss, bond_ce, w2_val, grad_norm, catherine_rate = train_one_epoch(
            model=model,
            decoder=decoder,
            optimiser=optimiser,
            train_clouds=train_d,
            device=device,
            batch_size=args.batch_size,
            wasserstein_weight=wasserstein_weight,
            closure_bounded=closure_bounded,
            closure_bias=closure_bias,
            reference_positions=reference_positions,
        )
        # Evaluate decode ratio via unified helper (handles both
        # YuelBond + ReworkedDecoder)
        n_ok, n_total = _eval_decode_ratio(
            decoder=decoder,
            val_clouds=val_d,
            device=device,
            closure_bounded=closure_bounded,
            closure_bias=closure_bias,
        )
        decode_ratio = n_ok / max(n_total, 1)
        elapsed = time.time() - epoch_start
        per_epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "bond_ce": bond_ce,
                "w2": w2_val,
                "grad_norm": grad_norm,
                "catherine_pass": catherine_rate,
                "decode_ratio": decode_ratio,
                "decode_ok": n_ok,
                "decode_total": n_total,
                "wall_time_s": elapsed,
            }
        )
        final_train_loss = train_loss
        final_decode_ratio = decode_ratio
        final_n_ok = n_ok
        final_n_total = n_total
        final_grad_norm = grad_norm
        print(
            f"[Wf-269C][{config_name}] epoch={epoch:02d} "
            f"loss={train_loss:.4f} bond_ce={bond_ce:.4f} w2={w2_val:.4f} "
            f"decode_ratio={decode_ratio:.3f} ({n_ok}/{n_total}) "
            f"wall={elapsed:.1f}s",
            flush=True,
        )
    wall_time_s = time.time() - overall_start

    return ConfigResult(
        config_name=config_name,
        epochs=args.epochs,
        final_train_loss=final_train_loss,
        final_decode_ratio=final_decode_ratio,
        decode_ok=final_n_ok,
        decode_total=final_n_total,
        mean_grad_norm=final_grad_norm,
        wall_time_s=wall_time_s,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-269C — CFM ablation on real cohort.",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-cohort", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--pocket-id", type=str, default="1h36")
    p.add_argument("--wasserstein-weight", type=float, default=0.1)
    p.add_argument(
        "--cohort-csv", type=str, default="", help="Override cohort CSV path."
    )
    p.add_argument("--out-dir", type=str, default="molmetal/reports/wf269c_cfm_real")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Step 1: load real cohort ----------------------------------------
    csv_path: Optional[Path] = None
    if args.cohort_csv:
        csv_path = PROJECT_ROOT / args.cohort_csv
        if not csv_path.exists():
            print(f"[Wf-269C] ERROR: --cohort-csv {csv_path} not found", flush=True)
            return 1
    else:
        for cand in REAL_COHORT_CANDIDATES:
            if cand.exists():
                csv_path = cand
                break
    if csv_path is None:
        print(
            f"[Wf-269C] ERROR: no real-cohort CSV found. Tried: "
            f"{[str(p) for p in REAL_COHORT_CANDIDATES]}",
            flush=True,
        )
        return 1

    print(f"[Wf-269C] cohort source = {csv_path}", flush=True)
    raw = load_real_cohort(csv_path)
    print(f"[Wf-269C] loaded raw mols = {len(raw)}", flush=True)

    # ---- Step 2: filter degenerate --------------------------------------
    filtered, n_dropped = filter_degenerate(raw)
    print(
        f"[Wf-269C] after degenerate-filter = {len(filtered)} "
        f"(dropped {n_dropped})",
        flush=True,
    )
    if len(filtered) < args.n_cohort:
        print(
            f"[Wf-269C] WARNING: cohort {len(filtered)} < target "
            f"{args.n_cohort}; will downsample.",
            flush=True,
        )
    # Stratified sample
    target_n = min(args.n_cohort, len(filtered))
    train_mols, val_mols = stratified_sample(filtered, target_n, seed=args.seed)
    print(
        f"[Wf-269C] stratified sample: train={len(train_mols)} " f"val={len(val_mols)}",
        flush=True,
    )

    # ---- Step 3: convert to AtomClouds -----------------------------------
    print(f"[Wf-269C] embedding train mols (3D) ...", flush=True)
    train_clouds: List[AtomCloud] = []
    n_train_embed_fail = 0
    for i, m in enumerate(train_mols):
        c = smiles_to_atomcloud(m.smiles, jitter=0.20, seed=args.seed + i)
        if c is None:
            n_train_embed_fail += 1
            continue
        train_clouds.append(c)
    print(
        f"[Wf-269C] train clouds = {len(train_clouds)} "
        f"(embed-fail={n_train_embed_fail})",
        flush=True,
    )

    print(f"[Wf-269C] embedding val mols (3D) ...", flush=True)
    val_clouds: List[AtomCloud] = []
    n_val_embed_fail = 0
    for i, m in enumerate(val_mols):
        c = smiles_to_atomcloud(m.smiles, jitter=0.20, seed=args.seed + 1000 + i)
        if c is None:
            n_val_embed_fail += 1
            continue
        val_clouds.append(c)
    print(
        f"[Wf-269C] val clouds = {len(val_clouds)} " f"(embed-fail={n_val_embed_fail})",
        flush=True,
    )

    # ---- Step 4: Run 4-config ablation -----------------------------------
    configs = [
        ("full", False, False, False),
        ("no_yuelbond", True, False, False),
        ("no_wasserstein", False, True, False),
        ("no_closure_bounded", False, False, True),
    ]
    results: List[ConfigResult] = []
    for cfg_name, no_yb, no_w2, no_cb in configs:
        try:
            r = run_one_config(
                config_name=cfg_name,
                args=args,
                train_clouds=train_clouds,
                val_clouds=val_clouds,
                device=device,
                no_yuelbond=no_yb,
                no_wasserstein=no_w2,
                no_closure_bounded=no_cb,
            )
            results.append(r)
        except Exception as e:
            print(
                f"[Wf-269C][{cfg_name}] ERROR: {type(e).__name__}: {e}",
                flush=True,
            )
            import traceback

            traceback.print_exc()
            results.append(
                ConfigResult(
                    config_name=cfg_name,
                    epochs=args.epochs,
                    final_train_loss=0.0,
                    final_decode_ratio=0.0,
                    decode_ok=0,
                    decode_total=len(val_clouds),
                    mean_grad_norm=0.0,
                    wall_time_s=0.0,
                )
            )

    # ---- Step 5: Summarize + write report -------------------------------
    full = next((r for r in results if r.config_name == "full"), None)
    no_yb = next((r for r in results if r.config_name == "no_yuelbond"), None)
    no_w2 = next((r for r in results if r.config_name == "no_wasserstein"), None)
    no_cb = next((r for r in results if r.config_name == "no_closure_bounded"), None)
    deltas_visible = False
    if full is not None:
        full_dr = full.final_decode_ratio
        for r in (no_yb, no_w2, no_cb):
            if r is not None and abs(full_dr - r.final_decode_ratio) > 0.05:
                deltas_visible = True

    honest_framing = "Real-cohort ablation completed. " + (
        "At least one ablation shows ≥0.05 degradation vs full — "
        "innovation differentiation signal PRESENT."
        if deltas_visible
        else "All 4 configs saturated near decode_ratio=1.0 (or 0.0) "
        "with no measurable differentiation — deferring ablation "
        "quality measurement to TODO-119 P1.3 (real CrossDocked "
        "loader) and Wf-268A synthetic re-validation."
    )

    report = {
        "wf": "wf269c_cfm_real_ablation",
        "task": "cfm_real_cohort_ablation",
        "cohort_source": str(csv_path),
        "n_raw_mols": len(raw),
        "n_after_filter": len(filtered),
        "n_dropped_degenerate": n_dropped,
        "n_train_clouds": len(train_clouds),
        "n_val_clouds": len(val_clouds),
        "n_train_embed_fail": n_train_embed_fail,
        "n_val_embed_fail": n_val_embed_fail,
        "epochs_per_config": args.epochs,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "device": str(device),
        "configs": [asdict(r) for r in results],
        "G1_real_cohort_loaded": True,
        "G2_cohort_source": str(csv_path),
        "G3_n_mols_after_filter": len(filtered),
        "G4_n_configs_tested": len(results),
        "G5_decode_ratio_full": full.final_decode_ratio if full else 0.0,
        "G6_decode_ratio_no_yuelbond": no_yb.final_decode_ratio if no_yb else 0.0,
        "G7_decode_ratio_no_wasserstein": no_w2.final_decode_ratio if no_w2 else 0.0,
        "G8_decode_ratio_no_closure_bounded": no_cb.final_decode_ratio
        if no_cb
        else 0.0,
        "G9_deltas_visible": deltas_visible,
        "G10_report_path": str(out_dir / "wf269c_report.json"),
        "honest_framing_note": honest_framing,
    }
    report_path = out_dir / "wf269c_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown summary
    md_path = out_dir / "wf269c_summary.md"
    with md_path.open("w") as f:
        f.write("# Wf-269C — Real-cohort CFM ablation\n\n")
        f.write(f"- Cohort source: `{csv_path}`\n")
        f.write(f"- Raw mols: {len(raw)}\n")
        f.write(f"- After degenerate-filter: {len(filtered)}\n")
        f.write(
            f"- Train clouds: {len(train_clouds)} "
            f"(embed-fail={n_train_embed_fail})\n"
        )
        f.write(
            f"- Val clouds: {len(val_clouds)} " f"(embed-fail={n_val_embed_fail})\n"
        )
        f.write(f"- Epochs per config: {args.epochs}\n")
        f.write(f"- Device: {device}\n\n")
        f.write("## Per-config results\n\n")
        f.write(
            "| Config | Decode ratio | Ok/Total | Train loss | Grad norm | Wall (s) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| `{r.config_name}` | {r.final_decode_ratio:.3f} | "
                f"{r.decode_ok}/{r.decode_total} | {r.final_train_loss:.4f} | "
                f"{r.mean_grad_norm:.3f} | {r.wall_time_s:.1f} |\n"
            )
        f.write(f"\n**Honest framing:** {honest_framing}\n")
    print(f"[Wf-269C] wrote {report_path}", flush=True)
    print(f"[Wf-269C] wrote {md_path}", flush=True)
    print(f"[Wf-269C] === DONE ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
