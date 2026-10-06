#!/usr/bin/env python3
"""Wf-148 — Real CFM decode benchmark on NCI-60 9-cell + DrugOOD native subset.

==============================================================================
Wf-148 — m14 real CFM decode benchmark (TODO-136 §2.2 + TODO-141 §2)
==============================================================================

Loads the Wf-124 P2 ship ``cfm_yuelbond_wasserstein.pt`` checkpoint and runs
inference on the **NCI-60 9-cell + DrugOOD native** subset.  Computes the
RDKit-strict ``decode_ratio`` on REAL (not synthetic) atom clouds.

**Why this benchmark exists**:
The 1.000 decode_ratio reported in :file:`decode_ratio.json` is from
**synthetic Pt-click clouds** (the canonical CFM smoke input).  The
**real CFM-trained decoder** has not been benchmarked on real
ligand coordinates.  This script closes that gap.

Inputs
------
* ``molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`` (Wf-124 P2 ship,
  617 KB)
* ``molmetal/data/nci60_precious_9cell.csv`` (NCI-60 9-cell, ~7800 rows)
* ``.data/molmetal/drugood/ic50/data/train.csv`` (DrugOOD native, 1.58M rows;
  subsampled)

Outputs
-------
* ``metrics/by_metric/cfm_decode_real.json`` — final real-CFM decode_ratio
  for paper §4.5.7 (Wf-148)
* ``molmetal/reports/wf148_m14_real_cfm.log`` — run log
* ``molmetal/reports/wf139_p5_cfm_decode_real/per_seed_metrics.json``
  — per-seed decode_ratio / n_ok / n_total / wall-time
* ``molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md`` — close-out
  summary

Usage
-----
::

    cd /home/hugo/codes/mol-metal
    nohup .venv/bin/python -u molmetal/scripts/bench_cfm_decode_real.py \\
        --n-rows 2000 \\
        --data-config nci60+drugood \\
        --device cuda:0 \\
        --out-json metrics/by_metric/cfm_decode_real.json \\
        > molmetal/reports/wf148_m14_real_cfm.log 2>&1 &

Honest framing
--------------
* Real CFM decode_ratio is **expected to be lower** than the synthetic
  1.000 baseline (the synthetic clouds are idealised Pt-click motifs;
  real ligands have noise, non-Pt atoms, and varied coordination).
* We DO NOT claim real >= synthetic; we MEASURE both and report
  honestly.
* If real decode_ratio is 0, that is a HALT signal — the model is
  failing on real data.

Constraints
-----------
* GPU-0 Blackwell 96GB (single GPU; sequential by design)
* CPU fallback supported via ``--device cpu`` (10× slower but valid)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
        evaluate_decode_ratio,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_N_LAYERS,
    )

    TRAIN_IMPORT_OK = True
    TRAIN_IMPORT_ERR: Optional[str] = None
except Exception as _e:  # pragma: no cover — defensive guard
    TRAIN_IMPORT_OK = False
    TRAIN_IMPORT_ERR = repr(_e)

from molmetal.models.bond_head import AtomCloud  # noqa: E402

try:
    from rdkit import Chem  # noqa: E402
    from rdkit.Chem import AllChem  # noqa: E402

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config defaults — match Wf-124 P2 ship
# ---------------------------------------------------------------------------
DEFAULT_CKPT_PATH = (
    PROJECT_ROOT / "molmetal" / "checkpoints" / "cfm_yuelbond_wasserstein.pt"
)
DEFAULT_DATA_PATH = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_NCI60_PATH = PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
DEFAULT_DRUGOOD_PATH = (
    PROJECT_ROOT / ".data" / "molmetal" / "drugood" / "ic50" / "data" / "train.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf139_p5_cfm_decode_real"
DEFAULT_OUT_JSON = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_decode_real.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "molmetal" / "reports" / "wf148_m14_real_cfm.log"
DEFAULT_CLOSURE_BIAS = {
    "CuAAC": 0.30,
    "SPAAC": 0.10,
    "ThiolEne": 0.10,
    "Suzuki": 0.20,
    "AmideCoupling": 0.30,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SeedMetrics:
    seed: int
    n_samples: int
    n_ok: int
    decode_ratio: float
    wall_time_s: float
    device: str
    closure_bounded: bool
    pocket_id: str


@dataclass
class BenchmarkReport:
    wf: str = "wf139_phase_5"
    task: str = "cfm_decode_real_benchmark"
    synthetic_baseline: float = 1.000
    checkpoint_path: str = ""
    data_path: str = ""
    n_total_platinai: int = 0
    n_pt_subset: int = 0
    n_sampled: int = 0
    per_seed: List[SeedMetrics] = field(default_factory=list)
    aggregate_decode_ratio: float = 0.0
    aggregate_n_ok: int = 0
    aggregate_n_total: int = 0
    honest_framing: str = ""
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_platinai_csv(path: Path) -> List[Dict[str, str]]:
    """Load PlatinAI 5K CSV into a list of dicts."""
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def filter_pt_subset(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter rows where the metal column is 'Pt' or contains 'Pt'."""
    out: List[Dict[str, str]] = []
    for row in rows:
        metal = (row.get("metal") or "").strip()
        if metal == "Pt" or "Pt" in metal:
            out.append(row)
    return out


def load_nci60_csv(path: Path) -> List[Dict[str, str]]:
    """Load NCI-60 9-cell CSV with columns smiles,cell_line,pIC50,active.

    All rows are kept (no metal filter); cell_line is dropped (we only
    need SMILES for the CFM decode benchmark).
    """
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                rows.append({"smiles": smi})
    return rows


def load_drugood_csv(
    path: Path,
    n_rows: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Load DrugOOD native IC50 train CSV; subsample n_rows rows.

    Columns: smiles,y,assay_id,scaffold_split_id,size_split_id.
    Only the SMILES column is retained.
    """
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                rows.append({"smiles": smi})
    if n_rows >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n_rows, replace=False)
    return [rows[int(i)] for i in idx]


def build_combined_subset(
    nci60_rows: List[Dict[str, str]],
    drugood_rows: List[Dict[str, str]],
    n_total: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Combine NCI-60 and DrugOOD rows; subsample n_total if larger.

    If both sources together are smaller than n_total, we return all
    rows.  Otherwise we uniformly subsample to n_total.
    """
    combined = list(nci60_rows) + list(drugood_rows)
    if n_total >= len(combined) or n_total <= 0:
        return combined
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combined), size=n_total, replace=False)
    return [combined[int(i)] for i in idx]


def smiles_to_atomcloud(
    smiles: str, jitter: float = 0.05, seed: int = 0, embed_timeout_s: float = 5.0
) -> Optional[AtomCloud]:
    """Convert SMILES → 3D-embedded atom cloud.

    Returns None if RDKit cannot parse or embed the molecule, or if
    embedding exceeds ``embed_timeout_s`` seconds.

    Parameters
    ----------
    smiles : str
        RDKit-parseable SMILES (with metal notation ``[Pt+2]`` etc.).
    jitter : float
        Gaussian position jitter (Å) added after embedding, to simulate
        CFM-style noisy coordinates.  Set to 0 for clean conformers.
    seed : int
        RNG seed for jitter.
    embed_timeout_s : float
        Per-molecule embedding timeout in seconds.  Uses signal.alarm
        on Unix; falls back to no-timeout on platforms without it.
    """
    if not RDKIT_AVAILABLE:
        return None
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    # Embed with bounded attempts; skip MMFF optimise to avoid hanging on
    # large metal complexes.  The decoder benchmark does not require
    # force-field quality — only 3D coordinates with correct topology.
    embed_kwargs = dict(randomSeed=seed, maxAttempts=10)
    # Wall-clock guard via SIGALRM (Unix only).
    import platform

    if platform.system() != "Windows":
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("embedding timeout")

        old_handler = signal.signal(
            signal.SIGALRM,
            _timeout_handler,
        )
        signal.setitimer(signal.ITIMER_REAL, embed_timeout_s)
        try:
            try:
                res = AllChem.EmbedMolecule(mol, **embed_kwargs)
                if res != 0:
                    res = AllChem.EmbedMolecule(
                        mol,
                        useRandomCoords=True,
                        **embed_kwargs,
                    )
                    if res != 0:
                        return None
            except (TimeoutError, Exception):
                return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        try:
            res = AllChem.EmbedMolecule(mol, **embed_kwargs)
            if res != 0:
                res = AllChem.EmbedMolecule(
                    mol,
                    useRandomCoords=True,
                    **embed_kwargs,
                )
                if res != 0:
                    return None
        except Exception:
            return None
    conf = mol.GetConformer()
    positions: List[List[float]] = []
    zs: List[int] = []
    rng = np.random.default_rng(seed)
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        x, y, z = p.x, p.y, p.z
        if jitter > 0:
            x += float(rng.normal(0, jitter))
            y += float(rng.normal(0, jitter))
            z += float(rng.normal(0, jitter))
        positions.append([x, y, z])
        zs.append(atom.GetAtomicNum())
    if len(positions) == 0:
        return None
    return AtomCloud(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(zs, dtype=torch.long),
    )


def build_real_clouds(
    rows: Sequence[Dict[str, str]],
    n_samples: int,
    seed: int = 0,
    jitter: float = 0.05,
) -> List[AtomCloud]:
    """Sample ``n_samples`` clouds from ``rows`` after SMILES → AtomCloud."""
    if n_samples >= len(rows):
        sample_rows = list(rows)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=n_samples, replace=False)
        sample_rows = [rows[int(i)] for i in idx]
    clouds: List[AtomCloud] = []
    for i, row in enumerate(sample_rows):
        smi = row.get("smiles") or ""
        cloud = smiles_to_atomcloud(smi, jitter=jitter, seed=seed + i)
        if cloud is not None:
            clouds.append(cloud)
    return clouds


# ---------------------------------------------------------------------------
# Checkpoint load
# ---------------------------------------------------------------------------
def load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[CFMJointModel, YuelBondBondDecoder, Dict]:
    """Load Wf-124 P2 ship checkpoint and instantiate model + decoder."""
    if not TRAIN_IMPORT_OK:
        raise RuntimeError(
            f"train_cfm_yuelbond_wasserstein import failed: {TRAIN_IMPORT_ERR}"
        )
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    # Infer hidden_dim / n_layers from state shapes.
    hidden_dim = DEFAULT_HIDDEN_DIM
    n_layers = DEFAULT_N_LAYERS
    # We rely on the checkpoint's "args" if present (preferred path).
    args_dict = payload.get("args", {})
    if isinstance(args_dict, dict):
        hidden_dim = int(args_dict.get("hidden_dim", hidden_dim))
        n_layers = int(args_dict.get("n_layers", n_layers))
    model = CFMJointModel(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)
    # Load with strict=False to be robust against minor head-shape drift.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[Wf-139 P5] WARNING checkpoint drift: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    # 2026-10-05 (Round-3 D2) — the connectivity prior defaults to ON, so this
    # bench's numbers are NOT comparable with a pre-D2 run of the same
    # checkpoint. Printed because the decoder is built here with defaults and
    # nothing downstream would otherwise record that the decode changed.
    print(
        f"[Wf-139 P5] connectivity_prior="
        f"{'on' if decoder.connectivity_prior else 'off'} (D2 postprocess)",
        flush=True,
    )
    model.eval()
    return model, decoder, payload


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------
def evaluate_seed(
    *,
    ckpt_path: Path,
    rows: Sequence[Dict[str, str]],
    n_samples: int,
    seed: int,
    device: torch.device,
    pocket_id: str,
    closure_bounded: bool = True,
    jitter: float = 0.05,
    source_tag: str = "",
) -> SeedMetrics:
    """Run a single seed of the benchmark.

    ``rows`` is the already-filtered cohort (e.g. NCI-60 9-cell + DrugOOD
    native subsample).  We do **not** re-load PlatinAI here.
    """
    clouds = build_real_clouds(
        rows,
        n_samples=n_samples,
        seed=seed,
        jitter=jitter,
    )
    if len(clouds) == 0:
        return SeedMetrics(
            seed=seed,
            n_samples=0,
            n_ok=0,
            decode_ratio=0.0,
            wall_time_s=0.0,
            device=str(device),
            closure_bounded=closure_bounded,
            pocket_id=pocket_id,
        )
    clouds_d = [
        AtomCloud(
            positions=c.positions.to(device),
            atomic_numbers=c.atomic_numbers.to(device),
        )
        for c in clouds
    ]
    model, decoder, payload = load_checkpoint(ckpt_path, device)
    closure_bias = payload.get("closure_bias") or DEFAULT_CLOSURE_BIAS
    t0 = time.time()
    n_ok, n_total = evaluate_decode_ratio(
        model=model,
        decoder=decoder,
        val_clouds=clouds_d,
        device=device,
        closure_bounded=closure_bounded,
        closure_bias=closure_bias,
    )
    wall = time.time() - t0
    return SeedMetrics(
        seed=seed,
        n_samples=n_total,
        n_ok=n_ok,
        decode_ratio=n_ok / max(n_total, 1),
        wall_time_s=wall,
        device=str(device),
        closure_bounded=closure_bounded,
        pocket_id=pocket_id,
    )


def evaluate_seed_platinai(
    *,
    ckpt_path: Path,
    data_path: Path,
    n_samples: int,
    seed: int,
    device: torch.device,
    pocket_id: str,
    closure_bounded: bool = True,
    jitter: float = 0.05,
) -> SeedMetrics:
    """Legacy PlatinAI-path seed evaluator (kept for Wf-139 compat)."""
    rows = load_platinai_csv(data_path)
    pt_rows = filter_pt_subset(rows)
    return evaluate_seed(
        ckpt_path=ckpt_path,
        rows=pt_rows,
        n_samples=n_samples,
        seed=seed,
        device=device,
        pocket_id=pocket_id,
        closure_bounded=closure_bounded,
        jitter=jitter,
        source_tag="platinai_5k_pt",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-148 — Real CFM decode benchmark on NCI-60 9-cell + "
            "DrugOOD native subset."
        ),
    )
    p.add_argument(
        "--ckpt-path",
        type=str,
        default=str(DEFAULT_CKPT_PATH),
        help="Path to trained CFM checkpoint (default Wf-124 P2).",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to PlatinAI 5K CSV (legacy Wf-139 path).",
    )
    p.add_argument(
        "--data-config",
        type=str,
        default="platinai_5k_pt",
        choices=["platinai_5k_pt", "nci60", "drugood", "nci60+drugood"],
        help=(
            "Data source. nci60+drugood concatenates the NCI-60 "
            "9-cell CSV with a DrugOOD IC50 train subsample."
        ),
    )
    p.add_argument(
        "--nci60-path",
        type=str,
        default=str(DEFAULT_NCI60_PATH),
        help="Path to NCI-60 9-cell CSV.",
    )
    p.add_argument(
        "--drugood-path",
        type=str,
        default=str(DEFAULT_DRUGOOD_PATH),
        help="Path to DrugOOD IC50 train CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for per_seed_metrics.json + synthesis.md.",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=str(DEFAULT_OUT_JSON),
        help="Path to write final decode_ratio JSON for paper §4.5.7.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of Pt-subset samples per seed (default 2000).",
    )
    p.add_argument(
        "--n-rows",
        type=int,
        default=2000,
        help="Alias for --n-samples; total samples (split per seed).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2).",
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
        "--smoke", action="store_true", help="Smoke run: n_samples=20, single seed."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_samples = 20
        args.seeds = [0]

    # --n-rows is the canonical total; if --n-samples was explicitly set
    # by the user, fall back to that for legacy compat.  Otherwise default
    # --n-rows wins.
    total_rows = args.n_rows if args.n_rows and args.n_rows > 0 else args.n_samples
    if total_rows != args.n_samples:
        # Split total_rows across seeds evenly; each seed sees
        # ceil(total_rows / len(seeds)) samples so that aggregate >= total_rows.
        per_seed_n = max(1, (total_rows + len(args.seeds) - 1) // len(args.seeds))
        args.n_samples = per_seed_n

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if not TRAIN_IMPORT_OK:
        print(f"[Wf-139 P5] FATAL: train import failed: {TRAIN_IMPORT_ERR}", flush=True)
        return 2
    if not RDKIT_AVAILABLE:
        print("[Wf-139 P5] FATAL: rdkit not available", flush=True)
        return 2

    print(f"[Wf-139 P5] device={device}", flush=True)
    if device.type == "cuda":
        try:
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(
                    f"[Wf-139 P5] GPU {i}: free={free/1024**3:.1f}GiB "
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
        print(f"[Wf-139 P5] FATAL: checkpoint missing: {ckpt_path}", flush=True)
        return 2

    # Resolve data sources per --data-config.
    nci60_path = Path(args.nci60_path)
    if not nci60_path.is_absolute():
        nci60_path = PROJECT_ROOT / nci60_path
    drugood_path = Path(args.drugood_path)
    if not drugood_path.is_absolute():
        drugood_path = PROJECT_ROOT / drugood_path

    nci60_rows: List[Dict[str, str]] = []
    drugood_rows: List[Dict[str, str]] = []
    if args.data_config in ("nci60", "nci60+drugood"):
        if not nci60_path.exists():
            print(f"[Wf-139 P5] FATAL: NCI-60 data missing: {nci60_path}", flush=True)
            return 2
        print(f"[Wf-139 P5] loading NCI-60 9-cell from {nci60_path}", flush=True)
        nci60_rows = load_nci60_csv(nci60_path)
        print(f"[Wf-139 P5] NCI-60 loaded: {len(nci60_rows)} rows", flush=True)
    if args.data_config in ("drugood", "nci60+drugood"):
        if not drugood_path.exists():
            print(
                f"[Wf-139 P5] FATAL: DrugOOD data missing: {drugood_path}", flush=True
            )
            return 2
        # Split total_rows / 2 to drugood when combined.
        drugood_n = (
            total_rows // 2 if args.data_config == "nci60+drugood" else total_rows
        )
        print(
            f"[Wf-139 P5] loading DrugOOD native from {drugood_path} "
            f"(n_rows={drugood_n})",
            flush=True,
        )
        drugood_rows = load_drugood_csv(
            drugood_path,
            n_rows=drugood_n,
            seed=0,
        )
        print(f"[Wf-139 P5] DrugOOD loaded: {len(drugood_rows)} rows", flush=True)

    if args.data_config in ("nci60", "drugood", "nci60+drugood"):
        combined = build_combined_subset(
            nci60_rows=nci60_rows,
            drugood_rows=drugood_rows,
            n_total=total_rows,
            seed=0,
        )
        eval_rows: List[Dict[str, str]] = list(combined)
        report_data_path = (
            f"nci60:{nci60_path} + drugood:{drugood_path} "
            f"({len(nci60_rows)}+{len(drugood_rows)} rows -> "
            f"{len(combined)} subsampled)"
        )
    else:
        if not data_path.exists():
            print(f"[Wf-139 P5] FATAL: data missing: {data_path}", flush=True)
            return 2
        all_rows = load_platinai_csv(data_path)
        eval_rows = filter_pt_subset(all_rows)
        report_data_path = str(data_path)

    print(
        f"[Wf-139 P5] data_config={args.data_config} "
        f"eval_rows={len(eval_rows)} per_seed_n={args.n_samples} "
        f"seeds={args.seeds}",
        flush=True,
    )

    overall_t0 = time.time()
    report = BenchmarkReport(
        checkpoint_path=str(ckpt_path),
        data_path=report_data_path,
    )

    report.n_total_platinai = (
        len(load_platinai_csv(data_path)) if data_path.exists() else 0
    )
    report.n_pt_subset = (
        len(filter_pt_subset(load_platinai_csv(data_path))) if data_path.exists() else 0
    )

    per_seed: List[SeedMetrics] = []
    agg_ok = 0
    agg_n = 0
    for seed in args.seeds:
        print(f"[Wf-139 P5] === seed={seed} n_samples={args.n_samples} ===", flush=True)
        sm = evaluate_seed(
            ckpt_path=ckpt_path,
            rows=eval_rows,
            n_samples=args.n_samples,
            seed=seed,
            device=device,
            pocket_id=args.pocket_id,
            jitter=args.jitter,
            source_tag=args.data_config,
        )
        per_seed.append(sm)
        agg_ok += sm.n_ok
        agg_n += sm.n_samples
        print(
            f"[Wf-139 P5] seed={seed} decode_ratio={sm.decode_ratio:.4f} "
            f"({sm.n_ok}/{sm.n_samples}) wall={sm.wall_time_s:.1f}s",
            flush=True,
        )

    report.per_seed = per_seed
    report.aggregate_n_ok = agg_ok
    report.aggregate_n_total = agg_n
    report.aggregate_decode_ratio = agg_ok / max(agg_n, 1)
    report.n_sampled = agg_n
    report.wall_time_s = time.time() - overall_t0

    # Honest framing verdict.
    if report.aggregate_n_total < 100:
        verdict = "HALT: <100 samples processed (gate FAIL)"
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

    # Write per_seed_metrics.json.
    metrics_path = out_dir / "per_seed_metrics.json"
    payload = {
        "wf": report.wf,
        "task": report.task,
        "synthetic_baseline_decode_ratio": report.synthetic_baseline,
        "checkpoint_path": report.checkpoint_path,
        "data_path": report.data_path,
        "data_config": args.data_config,
        "n_total_platinai": report.n_total_platinai,
        "n_pt_subset": report.n_pt_subset,
        "n_sampled": report.n_sampled,
        "aggregate_n_ok": report.aggregate_n_ok,
        "aggregate_n_total": report.aggregate_n_total,
        "aggregate_decode_ratio": report.aggregate_decode_ratio,
        "per_seed": [asdict(m) for m in per_seed],
        "wall_time_s": report.wall_time_s,
        "honest_framing": verdict,
        "constraints": {
            "device": str(device),
            "jitter": args.jitter,
            "pocket_id": args.pocket_id,
            "smoke": args.smoke,
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Wf-139 P5] wrote {metrics_path}", flush=True)

    # Write canonical real-CFM decode_ratio JSON for paper §4.5.7.
    gap_vs_synthetic = report.aggregate_decode_ratio - report.synthetic_baseline
    real_metric_payload = {
        "metric": "cfm_decode_real",
        "description": (
            "Real (not synthetic) CFM decode_ratio on NCI-60 9-cell + "
            "DrugOOD native SMILES, RDKit-strict."
        ),
        "unit": "ratio (0-1)",
        "scope": "Real CFM (cfm_yuelbond_wasserstein.pt) on real ligands",
        "synthetic_baseline_decode_ratio": report.synthetic_baseline,
        "checkpoint_path": report.checkpoint_path,
        "data_config": args.data_config,
        "nci60_path": str(nci60_path) if nci60_rows else None,
        "drugood_path": str(drugood_path) if drugood_rows else None,
        "n_nci60_loaded": len(nci60_rows),
        "n_drugood_loaded": len(drugood_rows),
        "n_total_platinai": report.n_total_platinai,
        "n_pt_subset": report.n_pt_subset,
        "n_sampled": report.n_sampled,
        "aggregate_n_ok": report.aggregate_n_ok,
        "aggregate_n_total": report.aggregate_n_total,
        "value": report.aggregate_decode_ratio,
        "gap_vs_synthetic": gap_vs_synthetic,
        "wall_time_s": report.wall_time_s,
        "device": str(device),
        "pocket_id": args.pocket_id,
        "jitter": args.jitter,
        "seeds": list(args.seeds),
        "date_utc": "2026-09-27",
        "status": (
            "MEASURED_HALT"
            if report.aggregate_n_total < 100
            else "MEASURED_ZERO"
            if report.aggregate_decode_ratio == 0.0
            else "MEASURED_PASS"
            if report.aggregate_decode_ratio >= 0.8
            else "MEASURED_PARTIAL"
            if report.aggregate_decode_ratio >= 0.5
            else "MEASURED_LOW"
        ),
        "per_seed": [asdict(m) for m in per_seed],
        "honest_framing": verdict,
        "notes": (
            "Synthetic baseline = 1.000 (Wf-124 ship on Pt-click clouds). "
            "Real CFM may be lower; this measurement is the honest gap."
        ),
    }
    out_json.write_text(json.dumps(real_metric_payload, indent=2) + "\n")
    print(f"[Wf-148] wrote {out_json}", flush=True)

    # Write synthesis.md.
    syn_path = out_dir / "synthesis.md"
    lines = [
        "# Wf-148 — Real CFM decode benchmark synthesis",
        "",
        "**Date**: 2026-09-27",
        f"**Checkpoint**: `{report.checkpoint_path}`",
        f"**Data config**: `{args.data_config}`",
        f"**Data**: `{report.data_path}`",
        f"**Device**: {device}",
        "",
        "## Aggregate",
        "",
        f"- Synthetic baseline decode_ratio: **{report.synthetic_baseline:.3f}**",
        f"- Real (NCI-60 9-cell + DrugOOD native) decode_ratio: "
        f"**{report.aggregate_decode_ratio:.4f}** "
        f"({report.aggregate_n_ok}/{report.aggregate_n_total})",
        f"- Gap vs synthetic: **{gap_vs_synthetic:+.4f}**",
        f"- Verdict: **{verdict}**",
        f"- Wall time: {report.wall_time_s:.1f}s",
        "",
        "## Per-seed",
        "",
        "| seed | n | n_ok | decode_ratio | wall_s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for m in per_seed:
        lines.append(
            f"| {m.seed} | {m.n_samples} | {m.n_ok} | "
            f"{m.decode_ratio:.4f} | {m.wall_time_s:.1f} |"
        )
    lines += [
        "",
        "## Honest framing",
        "",
        verdict,
        "",
        "- Synthetic is the **upper bound** (idealised Pt-click clouds).",
        "- Real ligands have noise, non-Pt atoms, and varied coordination.",
        "- Per party-twenty-spirit §3.5, real < synthetic is the **expected** "
        "honest framing; we MEASURE both and report.",
        "- HALT if decode_ratio == 0 OR <100 samples processed.",
        "",
    ]
    syn_path.write_text("\n".join(lines))
    print(f"[Wf-139 P5] wrote {syn_path}", flush=True)

    # Final exit-code policy.
    if report.aggregate_n_total < 100:
        return 3  # HALT: insufficient samples
    if report.aggregate_decode_ratio == 0.0:
        return 4  # HALT: zero decode
    return 0  # PASS or PARTIAL (PARTIAL is success per gate)


if __name__ == "__main__":
    sys.exit(main())
