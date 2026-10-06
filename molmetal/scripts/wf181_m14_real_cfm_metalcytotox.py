#!/usr/bin/env python3
"""Wf-181 — m14 real CFM decode_ratio benchmark on MetalCytoToxDB subset.

==============================================================================
Wf-181 — m14 real CFM decode ratio benchmark (TODO-181 §2)
==============================================================================

Loads the Wf-124 P2 ship ``cfm_yuelbond_wasserstein.pt`` checkpoint and runs
inference on a REAL MetalCytoToxDB subset (~100 samples).  Computes the
RDKit-strict ``decode_ratio`` on REAL (not synthetic) atom clouds.

**Why this benchmark exists**:
The synthetic Pt-click-only 1.000 decode_ratio is the upper bound.  Real
CFM-trained decoder on REAL MetalCytoToxDB ligands has not been measured
at the m14 level.  This script closes that gap and reports honestly:
real CFM is expected to be lower than synthetic.

Inputs
------
* ``molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`` (Wf-124 P2 ship,
  617 KB, 50-epoch YuelBond + Wasserstein + Catherine closure-bounded)
* ``molmetal/data/metalcytotox_rdkit_features.csv`` (MetalCytoToxDB,
  26801 rows; subsample to ~100)

Outputs
-------
* ``molmetal/reports/wf181_m14_real_cfm_metalcytotox/per_seed_metrics.json``
* ``molmetal/reports/wf181_m14_real_cfm_metalcytotox/synthesis.md``
* ``molmetal/reports/wf181_m14_real_cfm_metalcytotox/result.json`` —
  final gate-by-gate JSON for the orchestrator

Acceptance gates
----------------
* [G1] CFM checkpoint loads
* [G2] decode_ratio computed
* [G3] Total wall-time < 600s
* [G4] decode_ratio > 0.5 (above random; may be < 1.0 since real CFM
        is harder than synthetic)

Constraints
-----------
* Max wall-time 1800s
* 3-seed × ~33 samples ≈ 100 total samples (per seed: ceil(100/3))
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import signal
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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CKPT_PATH = (
    PROJECT_ROOT / "molmetal" / "checkpoints" / "cfm_yuelbond_wasserstein.pt"
)
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf181_m14_real_cfm_metalcytotox"
)
DEFAULT_OUT_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf181_m14_real_cfm_metalcytotox"
    / "result.json"
)
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
    pocket_id: str


@dataclass
class BenchmarkReport:
    wf: str = "wf181"
    task: str = "m14_real_cfm_decode_benchmark_metalcytotox"
    synthetic_baseline: float = 1.000
    checkpoint_path: str = ""
    data_path: str = ""
    n_total_metalcytotox: int = 0
    n_sampled: int = 0
    per_seed: List[SeedMetrics] = field(default_factory=list)
    aggregate_decode_ratio: float = 0.0
    aggregate_n_ok: int = 0
    aggregate_n_total: int = 0
    honest_framing: str = ""
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_metalcytotox_csv(path: Path) -> List[Dict[str, str]]:
    """Load MetalCytoToxDB CSV (smiles, metal, activity, ...)."""
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                rows.append(row)
    return rows


def smiles_to_atomcloud(
    smiles: str, jitter: float = 0.05, seed: int = 0, embed_timeout_s: float = 5.0
) -> Optional[AtomCloud]:
    """SMILES → 3D-embedded atom cloud with bounded wall-clock."""
    if not RDKIT_AVAILABLE:
        return None
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    embed_kwargs = dict(randomSeed=seed, maxAttempts=10)
    if platform.system() != "Windows":

        def _timeout_handler(signum, frame):
            raise TimeoutError("embedding timeout")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
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
    """Sample n_samples SMILES → AtomCloud from rows."""
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
        raise RuntimeError(f"train import failed: {TRAIN_IMPORT_ERR}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    hidden_dim = DEFAULT_HIDDEN_DIM
    n_layers = DEFAULT_N_LAYERS
    args_dict = payload.get("args", {})
    if isinstance(args_dict, dict):
        hidden_dim = int(args_dict.get("hidden_dim", hidden_dim))
        n_layers = int(args_dict.get("n_layers", n_layers))
    model = CFMJointModel(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[Wf-181] WARNING checkpoint drift: "
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
        f"[Wf-181] connectivity_prior="
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
) -> SeedMetrics:
    """Run a single seed of the benchmark on the pre-filtered cohort."""
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
        pocket_id=pocket_id,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-181 — Real CFM decode_ratio benchmark on " "MetalCytoToxDB subset."
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
        "--n-samples",
        type=int,
        default=100,
        help="Total samples across seeds (default 100).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds (default 0 1 2).",
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total_rows = args.n_samples
    per_seed_n = max(1, (total_rows + len(args.seeds) - 1) // len(args.seeds))
    args.n_samples = per_seed_n

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if not TRAIN_IMPORT_OK:
        print(f"[Wf-181] FATAL: train import failed: {TRAIN_IMPORT_ERR}", flush=True)
        return 2
    if not RDKIT_AVAILABLE:
        print("[Wf-181] FATAL: rdkit not available", flush=True)
        return 2

    print(f"[Wf-181] device={device}", flush=True)
    if device.type == "cuda":
        try:
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(
                    f"[Wf-181] GPU {i}: free={free/1024**3:.1f}GiB "
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
        print(f"[Wf-181] FATAL: checkpoint missing: {ckpt_path}", flush=True)
        return 2
    if not data_path.exists():
        print(f"[Wf-181] FATAL: data missing: {data_path}", flush=True)
        return 2

    # Load MetalCytoToxDB.
    print(f"[Wf-181] loading MetalCytoToxDB from {data_path}", flush=True)
    rows = load_metalcytotox_csv(data_path)
    print(f"[Wf-181] MetalCytoToxDB loaded: {len(rows)} rows", flush=True)

    # Aggregate (no per-metal filter: use ALL MetalCytoToxDB ligands
    # so that the benchmark captures the full chemistry spread —
    # Ru/Pt/Ir/Au/Os/Re/Rh across diverse coordination).
    eval_rows = list(rows)

    overall_t0 = time.time()
    report = BenchmarkReport(
        checkpoint_path=str(ckpt_path),
        data_path=str(data_path),
        n_total_metalcytotox=len(rows),
    )

    per_seed: List[SeedMetrics] = []
    agg_ok = 0
    agg_n = 0
    for seed in args.seeds:
        print(
            f"[Wf-181] === seed={seed} n_samples={per_seed_n} ===",
            flush=True,
        )
        sm = evaluate_seed(
            ckpt_path=ckpt_path,
            rows=eval_rows,
            n_samples=per_seed_n,
            seed=seed,
            device=device,
            pocket_id=args.pocket_id,
            jitter=args.jitter,
        )
        per_seed.append(sm)
        agg_ok += sm.n_ok
        agg_n += sm.n_samples
        print(
            f"[Wf-181] seed={seed} decode_ratio={sm.decode_ratio:.4f} "
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
        "n_total_metalcytotox": report.n_total_metalcytotox,
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
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Wf-181] wrote {metrics_path}", flush=True)

    # Gate-by-gate JSON for the orchestrator.
    gap_vs_synthetic = report.aggregate_decode_ratio - report.synthetic_baseline
    gate_1_ckpt_loads = bool(ckpt_path.exists() and TRAIN_IMPORT_OK)
    gate_2_decode_ratio = float(report.aggregate_decode_ratio)
    gate_3_wall_sec = float(report.wall_time_s)
    gate_4_above_threshold = bool(gate_2_decode_ratio > 0.5)
    all_gates_passed = bool(
        gate_1_ckpt_loads
        and gate_2_decode_ratio is not None
        and gate_3_wall_sec < 600.0
        and gate_4_above_threshold
    )
    gate_payload = {
        "wf": "Wf-181",
        "task": "m14_real_cfm_decode_ratio_metalcytotox",
        "checkpoint": str(ckpt_path),
        "data": str(data_path),
        "device": str(device),
        "n_total_metalcytotox": report.n_total_metalcytotox,
        "n_sampled": report.n_sampled,
        "synthetic_baseline_decode_ratio": report.synthetic_baseline,
        "gate_1_ckpt_loads": gate_1_ckpt_loads,
        "gate_2_decode_ratio": gate_2_decode_ratio,
        "gate_3_wall_sec": gate_3_wall_sec,
        "gate_4_above_threshold": gate_4_above_threshold,
        "all_gates_passed": all_gates_passed,
        "gap_vs_synthetic": gap_vs_synthetic,
        "per_seed": [asdict(m) for m in per_seed],
        "honest_framing_note": (
            f"Real CFM decode_ratio = {gate_2_decode_ratio:.4f} on "
            f"{report.aggregate_n_total} MetalCytoToxDB samples (vs "
            f"synthetic upper bound {report.synthetic_baseline:.3f}; "
            f"gap = {gap_vs_synthetic:+.4f}). Verdict: {verdict} "
            f"Real ligands have noise, non-Pt atoms, and varied "
            f"coordination; per party-twenty-spirit §3.5, real < "
            f"synthetic is the expected honest framing — we MEASURE "
            f"both and report. Cf. Wf-139 (PlatinAI 5K Pt subset, "
            f"real=0.9745) and Wf-148 (NCI-60+DrugOOD, real=0.9980)."
        ),
    }
    out_json.write_text(json.dumps(gate_payload, indent=2) + "\n")
    print(f"[Wf-181] wrote {out_json}", flush=True)

    # Write synthesis.md.
    syn_path = out_dir / "synthesis.md"
    lines = [
        "# Wf-181 — m14 real CFM decode benchmark (MetalCytoToxDB)",
        "",
        "**Date**: 2026-09-27",
        f"**Checkpoint**: `{report.checkpoint_path}`",
        f"**Data**: `{report.data_path}` (n_total={report.n_total_metalcytotox})",
        f"**Device**: {device}",
        "",
        "## Aggregate",
        "",
        f"- Synthetic baseline decode_ratio: **{report.synthetic_baseline:.3f}**",
        f"- Real (MetalCytoToxDB) decode_ratio: "
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
        "- Real MetalCytoToxDB ligands have noise, multi-metal diversity "
        "(Ru/Pt/Ir/Au/Os/Re/Rh), and varied coordination geometries.",
        "- Per party-twenty-spirit §3.5, real < synthetic is the **expected** "
        "honest framing; we MEASURE both and report.",
        "- HALT if decode_ratio == 0 OR <100 samples processed.",
        "- Cross-reference: Wf-139 P5 (PlatinAI 5K Pt subset, real=0.9745, "
        "274/274); Wf-148 (NCI-60 9-cell + DrugOOD native, real=0.9980, "
        "5979/5991).",
        "",
    ]
    syn_path.write_text("\n".join(lines))
    print(f"[Wf-181] wrote {syn_path}", flush=True)

    # Final exit-code policy.
    if report.aggregate_n_total < 100:
        return 3  # HALT: insufficient samples
    if report.aggregate_decode_ratio == 0.0:
        return 4  # HALT: zero decode
    return 0  # PASS or PARTIAL (PARTIAL is success per gate)


if __name__ == "__main__":
    sys.exit(main())
