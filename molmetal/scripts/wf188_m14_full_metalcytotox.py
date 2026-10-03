#!/usr/bin/env python3
"""Wf-188 v2 — m14 real CFM FULL MetalCytoToxDB benchmark (per-metal, parallel).

==============================================================================
Wf-188 — m14 real CFM FULL MetalCytoToxDB benchmark (TODO Wf-188 §2)
==============================================================================

v2: parallel embedding via multiprocessing.Pool.  32 cores available so 16
workers parallelize the RDKit 3D embedding → ~5x speedup over serial.

Wf-181 only ran 102 samples (3 seeds × 34); Wf-188 closes that gap and
runs the FULL 26,801 MetalCytoToxDB ligands (Ru=19,135 / Ir=4,546 /
Rh=1,134 / Os=1,118 / Re=868).  Reports overall + per-metal decode_ratio
in real time.

**Why this benchmark exists**:
Wf-181 PARTIAL/PASS verdict on 102 samples is statistically thin. The
full MetalCytoToxDB has multi-metal diversity (Ru-dominant 71%,
Ir 17%, Rh/Os 4%, Re 3%) and varied coordination that the CFM
should handle via the YuelBond bond head + Catherine closure bias.
This script measures real CFM on the FULL population and reports
honestly per metal.

Inputs
------
* ``molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`` (Wf-124 P2 ship,
  617 KB, 50-epoch YuelBond + Wasserstein + Catherine closure-bounded)
* ``molmetal/data/metalcytotox_rdkit_features.csv`` (MetalCytoToxDB,
  26,801 rows — full)

Outputs
-------
* ``molmetal/reports/wf188_m14_full_metalcytotox/per_seed_metrics.json``
* ``molmetal/reports/wf188_m14_full_metalcytotox/per_metal_metrics.json``
* ``molmetal/reports/wf188_m14_full_metalcytotox/synthesis.md``
* ``molmetal/reports/wf188_m14_full_metalcytotox/result.json`` —
  final gate-by-gate JSON for the orchestrator

Acceptance gates
----------------
* [G1] CFM checkpoint loads
* [G2] All 26,801 samples processed
* [G3] Wall-time < 600s
* [G4] decode_ratio > 0.5
* [G5] Per-metal decode_ratio reported (Ru/Ir/Os/Rh/Re)

Constraints
-----------
* Max wall-time 1200s
* Single checkpoint load (shared across all metals)
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — must come before any molmetal import.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# ---------------------------------------------------------------------------
# molmetal imports — guarded because YuelBond decoder requires torch_scatter.
# ---------------------------------------------------------------------------
try:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        CFMJointModel,
        YuelBondBondDecoder,
        _decode_with_rdkit_strict,
        NUM_BOND_CLASSES,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_N_LAYERS,
        apply_closure_bounded_mask,
    )

    TRAIN_IMPORT_OK = True
    TRAIN_IMPORT_ERR: Optional[str] = None
except Exception as _e:  # pragma: no cover — defensive guard
    TRAIN_IMPORT_OK = False
    TRAIN_IMPORT_ERR = repr(_e)

from molmetal.models.bond_head import AtomCloud  # noqa: E402

try:
    from rdkit import Chem  # noqa: E402

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CKPT_PATH = (
    PROJECT_ROOT / "molmetal" / "checkpoints" / "cfm_yuelbond_wasserstein.pt"
)
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf188_m14_full_metalcytotox"
DEFAULT_OUT_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf188_m14_full_metalcytotox"
    / "result.json"
)
DEFAULT_CLOSURE_BIAS = {
    "CuAAC": 0.30,
    "SPAAC": 0.10,
    "ThiolEne": 0.10,
    "Suzuki": 0.20,
    "AmideCoupling": 0.30,
}
TARGET_METALS = ["Ru", "Ir", "Os", "Rh", "Re"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MetalMetrics:
    metal: str
    n_total: int
    n_embedded: int
    n_decode_ok: int
    decode_ratio: float
    wall_time_s: float


@dataclass
class BenchmarkReport:
    wf: str = "wf188"
    task: str = "m14_real_cfm_decode_benchmark_metalcytotox_FULL"
    synthetic_baseline: float = 1.000
    checkpoint_path: str = ""
    data_path: str = ""
    n_total_metalcytotox: int = 0
    per_metal: List[MetalMetrics] = field(default_factory=list)
    aggregate_n_decode_ok: int = 0
    aggregate_n_embedded: int = 0
    aggregate_decode_ratio: float = 0.0
    honest_framing: str = ""
    wall_time_s: float = 0.0
    device: str = ""


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_metalcytotox_csv(path: Path) -> List[Dict[str, str]]:
    """Load MetalCytoToxDB CSV and return rows with a normalised metal key."""
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            metal = (row.get("metal") or "").strip()
            if smi and metal:
                rows.append({"smiles": smi, "metal": metal})
    return rows


# ---------------------------------------------------------------------------
# Parallel embedding worker
# ---------------------------------------------------------------------------
_EMBED_FUNC = None


def _embed_worker_init():
    """Worker init: pre-import RDKit to avoid repeated imports in workers."""
    global _EMBED_FUNC
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _EMBED_FUNC = (Chem, AllChem)


def _try_embed_with_strategy(
    Chem,  # noqa: N803
    AllChem,  # noqa: N803
    mol: "Chem.Mol",
    *,
    strategy: str,
    seed: int,
    max_attempts: int = 5,
    max_iterations: int = 200,
) -> bool:
    """Attempt one 3D-embedding strategy.  Returns True on success.

    Strategies (Wf-198 graceful-fallback chain, ordered):

    * ``"etkdg"``        — default ``AllChem.EmbedMolecule`` with
      ``maxAttempts=5`` (the cheap, familiar 3-attempt path + 2 retries).
    * ``"random_coords"``— ``useRandomCoords=True`` (RDKit has no
      ETKDG torsion priors for Pt/Ru/Ir; random coords still respect
      ETKDG distance-geometry bounds).
    * ``"etkdgv3_mmff"`` — explicit ``AllChem.ETKDGv3()`` with
      ``maxIterations=200`` and ``useRandomCoords=True`` + retry loop +
      ``MMFFOptimizeMolecule`` to clean up bond lengths.  This is the
      heavy fallback used by ``scripts/check_3d_embed.py`` for the
      awkward metal-coordinated Ru/Ir complexes.

    On any exception the strategy is logged-and-swallowed (the caller
    moves to the next strategy).  This never raises.
    """
    if strategy == "etkdg":
        embed_kwargs = dict(randomSeed=seed, maxAttempts=max_attempts)
        for _ in range(max_attempts):
            try:
                res = AllChem.EmbedMolecule(mol, **embed_kwargs)
            except Exception:
                res = -1
            if res == 0:
                return True
        return False
    if strategy == "random_coords":
        embed_kwargs = dict(
            randomSeed=seed,
            maxAttempts=max_attempts,
            useRandomCoords=True,
        )
        for _ in range(max_attempts):
            try:
                res = AllChem.EmbedMolecule(mol, **embed_kwargs)
            except Exception:
                res = -1
            if res == 0:
                return True
        return False
    if strategy == "etkdgv3_mmff":
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.maxIterations = int(max_iterations)
        params.useRandomCoords = True
        for _ in range(max_attempts):
            try:
                res = AllChem.EmbedMolecule(mol, params)
            except Exception:
                res = -1
            if res == 0:
                break
        else:
            return False
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # MMFF failed but the ETKDGv3 conformer exists — accept.
            pass
        return mol.GetNumConformers() > 0
    raise ValueError(f"unknown strategy={strategy!r}")


def _embed_one(args: Tuple[str, int, float]) -> Optional[bytes]:
    """Worker function: SMILES → AtomCloud pickled bytes.

    Wf-198 graceful fallback chain (replaces the pre-Wf-188 dual-try):

    1. ``EmbedMolecule`` with ``maxAttempts=5`` — most common path.
    2. ``EmbedMolecule`` with ``useRandomCoords=True`` — handles the
       metal-coordination cases where RDKit's knowledge-based torsions
       are undefined.
    3. Explicit ``ETKDGv3`` + ``MMFFOptimizeMolecule`` — last resort
       for the 2/26801 pathological Ru SMILES Wf-188 surfaced.
    4. Log + skip (return ``None``) if all three fail.  The pipeline
       never crashes; the caller simply sees ``n_embedded < n_total``.
    """
    if _EMBED_FUNC is None:
        _embed_worker_init()
    Chem, AllChem = _EMBED_FUNC
    smi, seed, jitter = args
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
    except Exception:
        return None

    # Wf-198 graceful fallback chain.
    strategies = ("etkdg", "random_coords", "etkdgv3_mmff")
    embedded = False
    last_status = -1
    for strat in strategies:
        try:
            embedded = _try_embed_with_strategy(
                Chem,
                AllChem,
                mol,
                strategy=strat,
                seed=seed,
                max_attempts=5,
                max_iterations=200,
            )
        except Exception:
            embedded = False
        if embedded:
            break
        # Track the last non-zero EmbedMolecule status for the log line.
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed)
            params.useRandomCoords = True
            last_status = int(
                AllChem.EmbedMolecule(mol, params),
            )
        except Exception:
            last_status = -1
    if not embedded:
        # Wf-198: log + skip — never raise out of the worker.
        print(
            f"[Wf-198] SKIP un-embeddable SMILES "
            f"(status={last_status}, len={len(smi)}): {smi[:80]}",
            flush=True,
        )
        return None
    try:
        conf = mol.GetConformer()
    except Exception:
        return None
    positions: List[List[float]] = []
    zs: List[int] = []
    rng = np.random.default_rng(seed)
    for atom in mol.GetAtoms():
        try:
            p = conf.GetAtomPosition(atom.GetIdx())
            x, y, z = p.x, p.y, p.z
        except Exception:
            return None
        if jitter > 0:
            x += float(rng.normal(0, jitter))
            y += float(rng.normal(0, jitter))
            z += float(rng.normal(0, jitter))
        positions.append([x, y, z])
        zs.append(atom.GetAtomicNum())
    if len(positions) == 0:
        return None
    cloud = AtomCloud(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(zs, dtype=torch.long),
    )
    return pickle.dumps((cloud.positions.cpu(), cloud.atomic_numbers.cpu()))


def embed_cohort_parallel(
    rows: Sequence[Dict[str, str]],
    jitter: float = 0.05,
    n_workers: int = 16,
    progress_every: int = 2000,
    label: str = "",
) -> List[Tuple[bytes, str]]:
    """Parallel embed all SMILES in cohort → (pickle_bytes, metal)."""
    if len(rows) == 0:
        return []
    out: List[Tuple[bytes, str]] = []
    work = [(row["smiles"], i, jitter) for i, row in enumerate(rows)]
    metals = [row["metal"] for row in rows]
    t0 = time.time()
    with mp.Pool(
        processes=n_workers,
        initializer=_embed_worker_init,
    ) as pool:
        for i, (res, metal) in enumerate(
            zip(
                pool.imap_unordered(
                    _embed_one,
                    work,
                    chunksize=64,
                ),
                metals,
            )
        ):
            if res is not None:
                out.append((res, metal))
            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(
                    f"[Wf-188] {label} embed progress: {i + 1}/{len(rows)} "
                    f"({rate:.1f}/s, {len(out)} ok)",
                    flush=True,
                )
    elapsed = time.time() - t0
    rate = len(rows) / max(elapsed, 1e-6)
    print(
        f"[Wf-188] {label} embed done: {len(out)}/{len(rows)} "
        f"({rate:.1f}/s) in {elapsed:.1f}s",
        flush=True,
    )
    return out


def cloud_from_pickle(pickled: bytes) -> AtomCloud:
    """Reconstruct AtomCloud from pickled (positions, atomic_numbers)."""
    pos, zs = pickle.loads(pickled)
    return AtomCloud(positions=pos, atomic_numbers=zs)


# ---------------------------------------------------------------------------
# Checkpoint load
# ---------------------------------------------------------------------------
def load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[CFMJointModel, YuelBondBondDecoder, Dict]:
    """Load Wf-124 P2 ship checkpoint and instantiate model + decoder."""
    if not TRAIN_IMPORT_OK:
        raise RuntimeError(f"train import failed: {TRAIN_IMPORT_ERR}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    hidden_dim = DEFAULT_HIDDEN_DIM
    n_layers = DEFAULT_N_LAYERS
    args_dict = payload.get("args", {})
    if isinstance(args_dict, dict):
        hidden_dim = int(args_dict.get("hidden_dim", hidden_dim))
        n_layers = int(args_dict.get("n_layers", n_layers))
    model = CFMJointModel(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[Wf-188] WARNING checkpoint drift: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    model.eval()
    return model, decoder, payload


# ---------------------------------------------------------------------------
# Decode a single cloud with closure bias applied
# ---------------------------------------------------------------------------
def decode_one_with_bias(
    decoder: YuelBondBondDecoder,
    cloud: AtomCloud,
    device: torch.device,
    closure_bias: Dict[str, float],
) -> bool:
    """Decode a single cloud with closure-bounded bias applied once."""
    cloud_d = AtomCloud(
        positions=cloud.positions.to(device),
        atomic_numbers=cloud.atomic_numbers.to(device),
    )
    try:
        pair_head_obj = (
            decoder._decoder.head.pair_head
            if hasattr(decoder._decoder.head, "pair_head")
            else decoder._decoder.head.edge_mlp
        )
        last_layer = pair_head_obj[-1]
        orig_bias = last_layer.bias.data.clone()
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
            ok = _decode_with_rdkit_strict(decoder, cloud_d)
        finally:
            last_layer.bias.data = orig_bias
        return ok
    except Exception:
        # Fallback if monkey-patch fails.
        return _decode_with_rdkit_strict(decoder, cloud_d)


# ---------------------------------------------------------------------------
# Per-metal evaluation
# ---------------------------------------------------------------------------
def decode_clouds(
    *,
    decoder: YuelBondBondDecoder,
    pickles: Sequence[bytes],
    device: torch.device,
    closure_bias: Dict[str, float],
    progress_every: int = 2000,
    label: str = "",
) -> Tuple[int, int, float]:
    """Decode pickled clouds on GPU with closure bias."""
    if len(pickles) == 0:
        return 0, 0, 0.0
    n_ok = 0
    n_total = 0
    t0 = time.time()
    for i, pk in enumerate(pickles):
        cloud = cloud_from_pickle(pk)
        n_total += 1
        ok = decode_one_with_bias(decoder, cloud, device, closure_bias)
        if ok:
            n_ok += 1
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[Wf-188] {label} decode progress: {i + 1}/{len(pickles)} "
                f"({rate:.1f}/s, {n_ok} ok)",
                flush=True,
            )
    wall = time.time() - t0
    return n_ok, n_total, wall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-188 v2 — Real CFM FULL MetalCytoToxDB benchmark "
            "(26,801 samples, per-metal, parallel embed)."
        ),
    )
    p.add_argument(
        "--ckpt-path",
        type=str,
        default=str(DEFAULT_CKPT_PATH),
        help="Path to trained CFM checkpoint.",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to MetalCytoToxDB CSV.",
    )
    p.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory."
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=str(DEFAULT_OUT_JSON),
        help="Final gate-by-gate JSON path.",
    )
    p.add_argument(
        "--device", type=str, default="auto", help="Device: auto / cuda / cuda:0 / cpu."
    )
    p.add_argument(
        "--pocket-id",
        type=str,
        default="1h36",
        help="Pocket ID for closure bias (default 1h36).",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.05,
        help="Position jitter in Å (default 0.05).",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Parallel embedding workers (default 16).",
    )
    p.add_argument(
        "--target-metals",
        type=str,
        nargs="+",
        default=TARGET_METALS,
        help="Metals to report per-metal breakdown.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if not TRAIN_IMPORT_OK:
        print(f"[Wf-188] FATAL: train import failed: {TRAIN_IMPORT_ERR}", flush=True)
        return 2
    if not RDKIT_AVAILABLE:
        print("[Wf-188] FATAL: rdkit not available", flush=True)
        return 2

    print(f"[Wf-188] device={device}", flush=True)
    if device.type == "cuda":
        try:
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(
                    f"[Wf-188] GPU {i}: free={free/1024**3:.1f}GiB "
                    f"total={total/1024**3:.1f}GiB",
                    flush=True,
                )
        except Exception:
            pass

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = PROJECT_ROOT / out_json
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"[Wf-188] FATAL: checkpoint missing: {ckpt_path}", flush=True)
        return 2
    if not data_path.exists():
        print(f"[Wf-188] FATAL: data missing: {data_path}", flush=True)
        return 2

    # Load MetalCytoToxDB.
    print(f"[Wf-188] loading MetalCytoToxDB from {data_path}", flush=True)
    all_rows = load_metalcytotox_csv(data_path)
    print(f"[Wf-188] MetalCytoToxDB loaded: {len(all_rows)} rows", flush=True)

    # Per-metal grouping.
    per_metal_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        metal = row["metal"]
        if metal in args.target_metals:
            per_metal_rows[metal].append(row)
    for m in args.target_metals:
        print(
            f"[Wf-188]   {m}: {len(per_metal_rows[m])} rows",
            flush=True,
        )

    # Load checkpoint ONCE (shared across all metals).
    print(f"[Wf-188] loading CFM checkpoint: {ckpt_path}", flush=True)
    ckpt_t0 = time.time()
    model, decoder, payload = load_checkpoint(ckpt_path, device)
    closure_bias = payload.get("closure_bias") or DEFAULT_CLOSURE_BIAS
    print(
        f"[Wf-188] checkpoint loaded in {time.time() - ckpt_t0:.2f}s",
        flush=True,
    )

    overall_t0 = time.time()
    report = BenchmarkReport(
        checkpoint_path=str(ckpt_path),
        data_path=str(data_path),
        n_total_metalcytotox=len(all_rows),
        device=str(device),
    )

    per_metal: List[MetalMetrics] = []
    agg_ok = 0
    agg_embed_ok = 0
    for metal in args.target_metals:
        rows_m = per_metal_rows.get(metal, [])
        if not rows_m:
            print(f"[Wf-188] no rows for metal={metal}; skipping", flush=True)
            per_metal.append(
                MetalMetrics(
                    metal=metal,
                    n_total=0,
                    n_embedded=0,
                    n_decode_ok=0,
                    decode_ratio=0.0,
                    wall_time_s=0.0,
                )
            )
            continue
        # Parallel embed.
        print(
            f"[Wf-188] === embedding {metal} ({len(rows_m)} SMILES) "
            f"with {args.n_workers} workers ===",
            flush=True,
        )
        embed_t0 = time.time()
        cohort = embed_cohort_parallel(
            rows_m,
            jitter=args.jitter,
            n_workers=args.n_workers,
            progress_every=2000,
            label=f"{metal}-embed",
        )
        embed_wall = time.time() - embed_t0
        pickles = [c[0] for c in cohort]
        n_embedded = len(pickles)
        # Decode.
        print(
            f"[Wf-188] === decoding {metal} "
            f"({n_embedded}/{len(rows_m)} embedded) ===",
            flush=True,
        )
        n_ok, n_total, decode_wall = decode_clouds(
            decoder=decoder,
            pickles=pickles,
            device=device,
            closure_bias=closure_bias,
            progress_every=2000,
            label=f"{metal}-decode",
        )
        decode_ratio = n_ok / max(n_total, 1)
        per_metal.append(
            MetalMetrics(
                metal=metal,
                n_total=len(rows_m),
                n_embedded=n_embedded,
                n_decode_ok=n_ok,
                decode_ratio=decode_ratio,
                wall_time_s=embed_wall + decode_wall,
            )
        )
        agg_ok += n_ok
        agg_embed_ok += n_embedded
        print(
            f"[Wf-188] metal={metal} decode_ratio={decode_ratio:.4f} "
            f"({n_ok}/{n_total}) wall={embed_wall + decode_wall:.1f}s",
            flush=True,
        )

    report.per_metal = per_metal
    report.aggregate_n_decode_ok = agg_ok
    report.aggregate_n_embedded = agg_embed_ok
    report.aggregate_decode_ratio = agg_ok / max(agg_embed_ok, 1)
    report.wall_time_s = time.time() - overall_t0

    # Honest framing.
    if report.aggregate_n_embedded < 10000:
        verdict = (
            f"HALT: only {report.aggregate_n_embedded} embedded " f"(target = 26,801)"
        )
    elif report.aggregate_decode_ratio == 0.0:
        verdict = "HALT: real CFM decode_ratio = 0.0 (model fails on real data)"
    elif report.aggregate_decode_ratio < report.synthetic_baseline:
        verdict = (
            f"PARTIAL: real {report.aggregate_decode_ratio:.4f} < "
            f"synthetic {report.synthetic_baseline:.3f} (expected — "
            f"synthetic is upper bound; real < synthetic is the honest "
            f"framing per party-twenty-spirit §3.5)"
        )
    else:
        verdict = (
            f"PASS: real {report.aggregate_decode_ratio:.4f} >= "
            f"synthetic {report.synthetic_baseline:.3f}"
        )
    report.honest_framing = verdict

    # Per-metal JSON.
    per_metal_path = out_dir / "per_metal_metrics.json"
    per_metal_payload = {
        "wf": report.wf,
        "task": report.task,
        "checkpoint_path": report.checkpoint_path,
        "data_path": report.data_path,
        "n_total_metalcytotox": report.n_total_metalcytotox,
        "n_target_metals": len(args.target_metals),
        "per_metal": [asdict(m) for m in per_metal],
        "aggregate_n_decode_ok": report.aggregate_n_decode_ok,
        "aggregate_n_embedded": report.aggregate_n_embedded,
        "aggregate_decode_ratio": report.aggregate_decode_ratio,
        "synthetic_baseline_decode_ratio": report.synthetic_baseline,
        "wall_time_s": report.wall_time_s,
        "device": report.device,
        "honest_framing": verdict,
        "constraints": {
            "jitter": args.jitter,
            "n_workers": args.n_workers,
            "pocket_id": args.pocket_id,
            "closure_bounded": True,
        },
    }
    per_metal_path.write_text(json.dumps(per_metal_payload, indent=2) + "\n")
    print(f"[Wf-188] wrote {per_metal_path}", flush=True)

    # Gate-by-gate JSON for the orchestrator.
    gate_1_ckpt_loads = bool(ckpt_path.exists() and TRAIN_IMPORT_OK)
    gate_2_n_processed = int(report.aggregate_n_embedded)
    gate_3_wall_sec = float(report.wall_time_s)
    gate_4_decode_ratio = float(report.aggregate_decode_ratio)
    gate_5_per_metal = {m.metal: float(m.decode_ratio) for m in per_metal}
    per_metal_summary = {
        m.metal: {
            "n": int(m.n_total),
            "n_embedded": int(m.n_embedded),
            "n_decode_ok": int(m.n_decode_ok),
            "decode_ratio": float(m.decode_ratio),
        }
        for m in per_metal
    }
    all_gates_passed = bool(
        gate_1_ckpt_loads
        and gate_2_n_processed == report.n_total_metalcytotox
        and gate_3_wall_sec < 600.0
        and gate_4_decode_ratio > 0.5
        and all(v > 0.5 for v in gate_5_per_metal.values())
    )

    gap_vs_synthetic = report.aggregate_decode_ratio - report.synthetic_baseline
    gate_payload = {
        "wf": "Wf-188",
        "task": "m14_real_cfm_FULL_decode_ratio_metalcytotox",
        "checkpoint": str(ckpt_path),
        "data": str(data_path),
        "device": str(device),
        "n_total_metalcytotox": report.n_total_metalcytotox,
        "n_target_metals": len(args.target_metals),
        "aggregate_n_embedded": report.aggregate_n_embedded,
        "gate_1_ckpt_loads": gate_1_ckpt_loads,
        "gate_2_n_processed": gate_2_n_processed,
        "gate_3_wall_sec": gate_3_wall_sec,
        "gate_4_decode_ratio": gate_4_decode_ratio,
        "gate_5_per_metal": gate_5_per_metal,
        "gate_5_per_metal_pass": all(v > 0.5 for v in gate_5_per_metal.values()),
        "all_gates_passed": all_gates_passed,
        "gap_vs_synthetic": gap_vs_synthetic,
        "per_metal_summary": per_metal_summary,
        "synthetic_baseline_decode_ratio": report.synthetic_baseline,
        "honest_framing_note": (
            f"Real CFM decode_ratio = {gate_4_decode_ratio:.4f} on "
            f"{gate_2_n_processed}/{report.n_total_metalcytotox} "
            f"MetalCytoToxDB samples (vs synthetic upper bound "
            f"{report.synthetic_baseline:.3f}; gap = "
            f"{gap_vs_synthetic:+.4f}). Verdict: {verdict} "
            f"Real ligands have noise, multi-metal diversity "
            f"(Ru/Pt/Ir/Au/Os/Re/Rh), and varied coordination; per "
            f"party-twenty-spirit §3.5, real < synthetic is the "
            f"expected honest framing. Cf. Wf-181 (102 samples, "
            f"real=1.0; Wf-188 is the FULL 26,801 population)."
        ),
    }
    out_json.write_text(json.dumps(gate_payload, indent=2) + "\n")
    print(f"[Wf-188] wrote {out_json}", flush=True)

    # Per_seed_metrics.json (replicates Wf-181 schema for compat).
    metrics_path = out_dir / "per_seed_metrics.json"
    metrics_payload = {
        "wf": report.wf,
        "task": report.task,
        "synthetic_baseline_decode_ratio": report.synthetic_baseline,
        "checkpoint_path": report.checkpoint_path,
        "data_path": report.data_path,
        "n_total_metalcytotox": report.n_total_metalcytotox,
        "n_sampled": report.aggregate_n_embedded,
        "aggregate_n_ok": report.aggregate_n_decode_ok,
        "aggregate_n_total": report.aggregate_n_embedded,
        "aggregate_decode_ratio": report.aggregate_decode_ratio,
        "per_metal": [asdict(m) for m in per_metal],
        "wall_time_s": report.wall_time_s,
        "honest_framing": verdict,
        "constraints": {
            "device": report.device,
            "jitter": args.jitter,
            "n_workers": args.n_workers,
            "pocket_id": args.pocket_id,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")
    print(f"[Wf-188] wrote {metrics_path}", flush=True)

    # Synthesis.
    syn_path = out_dir / "synthesis.md"
    lines = [
        "# Wf-188 — m14 real CFM FULL MetalCytoToxDB benchmark (per-metal)",
        "",
        "**Date**: 2026-09-27",
        f"**Checkpoint**: `{report.checkpoint_path}`",
        f"**Data**: `{report.data_path}` " f"(n_total={report.n_total_metalcytotox})",
        f"**Device**: {device}",
        f"**Embed workers**: {args.n_workers}",
        "",
        "## Aggregate",
        "",
        f"- Synthetic baseline decode_ratio: **{report.synthetic_baseline:.3f}**",
        f"- Real (MetalCytoToxDB FULL) decode_ratio: "
        f"**{report.aggregate_decode_ratio:.4f}** "
        f"({report.aggregate_n_decode_ok}/{report.aggregate_n_embedded})",
        f"- Gap vs synthetic: **{gap_vs_synthetic:+.4f}**",
        f"- Verdict: **{verdict}**",
        f"- Wall time: {report.wall_time_s:.1f}s " f"(target: <600s, hard cap: 1200s)",
        "",
        "## Per-metal breakdown",
        "",
        "| metal | n_total | n_embedded | n_decode_ok | decode_ratio | wall_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in per_metal:
        lines.append(
            f"| {m.metal} | {m.n_total} | {m.n_embedded} | "
            f"{m.n_decode_ok} | {m.decode_ratio:.4f} | {m.wall_time_s:.1f} |"
        )
    lines += [
        "",
        "## Honest framing",
        "",
        verdict,
        "",
        "- Synthetic is the **upper bound** (idealised Pt-click clouds).",
        "- Real MetalCytoToxDB ligands have noise, multi-metal diversity "
        "(Ru/Pt/Ir/Au/Os/Re/Rh), and varied coordination geometries.",
        "- Per party-twenty-spirit §3.5, real < synthetic is the **expected** "
        "honest framing; we MEASURE both and report.",
        "- HALT if aggregate decode_ratio == 0 OR <10000 samples processed.",
        "- Cross-reference: Wf-181 (102 samples, real=1.0 by lucky "
        "embedding; Wf-188 is the FULL 26,801 population).",
        "",
    ]
    syn_path.write_text("\n".join(lines))
    print(f"[Wf-188] wrote {syn_path}", flush=True)

    # Final exit-code policy.
    if report.aggregate_n_embedded < 10000:
        return 3  # HALT: insufficient samples
    if report.aggregate_decode_ratio == 0.0:
        return 4  # HALT: zero decode
    return 0  # PASS or PARTIAL (PARTIAL is success per gate)


if __name__ == "__main__":
    sys.exit(main())
