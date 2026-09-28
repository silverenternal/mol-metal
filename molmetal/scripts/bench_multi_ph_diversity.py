"""Wf-31 Phase 3 — Multi-PH feature extraction + diversity benchmark.

================================================================
Goal
================================================================
Run a structured Multi-Parameter Persistent Homology (Multi-PH,
Kim & Jung 2026) feature-extraction + chemical-diversity benchmark on
the PlatinAI 5K corpus, comparing the **multi-PH** (2-parameter) feature
map against a **single-PH** baseline that marginalises over the second
filter axis.  Reports the standard R18 PLAN §3 expressivity numbers:

* **Expressivity** — how well each feature family distinguishes
  structurally distinct pockets.  Reported as the mean pairwise
  cosine distance (1 - cos) over the flattened feature map; a
  *higher* mean pairwise distance indicates the feature family
  spreads samples more uniformly in the embedding space (more
  expressive).
* **Diversity** — Tanimoto distance on Morgan-ECFP4 fingerprints
  computed over the same SMILES pool.  This is the m7 metric that
  the R18 PLAN §3 cites as the headline diversity number.

Pipeline
--------
1. Load ``molmetal/data/platinai_5k.csv`` (5 K platinum-group SMILES).
2. RDKit 3-D-embed each SMILES; truncate / zero-pad to ``max_atoms``.
3. Multi-PH feature map per molecule via
   :class:`MultiPHPocketFeatures.compute_with_atom_types` (real Wf-1 T5
   math, no shortcut).
4. Single-PH baseline = sum of the multi-PH feature map along the
   coordination-distance (``d``) axis — the canonical reduction to a
   1-D Betti curve when only the spatial filter is varied.
5. Expressivity: mean pairwise cosine distance over each feature set.
6. Diversity: pairwise Tanimoto distance on Morgan-ECFP4 fingerprints
   over the same pool.
7. Report ``multi_ph > single_ph`` is required for the Wf-31 success
   criterion (R18 PLAN §3 m7).

Outputs
-------
* ``molmetal/reports/wf31_multi_ph_diversity/metrics.json`` — full
  per-step + aggregate metrics.
* ``molmetal/reports/wf31_multi_ph_diversity/comparison_table.md`` —
  human-readable comparison table.
* ``molmetal/reports/wf31_multi_ph_diversity/run.log`` — tee'd stdout.

Run::

    cd /home/hugo/codes/mol-metal
    uv run --no-sync python molmetal/scripts/bench_multi_ph_diversity.py \\
        --n-samples 200 --output molmetal/reports/wf31_multi_ph_diversity
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repo-root path setup so the script can be invoked from any cwd.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


log = logging.getLogger("wf31_multi_ph_diversity")


# ---------------------------------------------------------------------------
# Constants — Wf-31 Phase 3 hyper-parameters
# ---------------------------------------------------------------------------
#: Default Multi-PH bin count (16 → 256-d feature map).  Matches the
#: Wf-25 Phase-2 / W2.3 spec to keep feature_dim comparable.
MULTI_PH_N_BINS: int = 16
#: Maximum spatial filter radius (Å).  Matches MultiPHPocketFeatures
#: default (Kim-Jung 2026 §4.1).
MULTI_PH_MAX_RADIUS: float = 5.0
#: Maximum atomic-number coordination distance.  Matches default.
MULTI_PH_MAX_COORD: float = 4.0
#: 3-D pocket budget — heavy atoms after RDKit embed.  Anything larger
#: is truncated to keep wall-time bounded; smaller is zero-padded.
MAX_POCKET_ATOMS: int = 60
#: Morgan-ECFP4 fingerprint radius.  Standard for chemical diversity.
FP_RADIUS: int = 2
#: Morgan-ECFP4 fingerprint bit count.  Standard for chemical diversity.
FP_NBITS: int = 2048
#: Headline diversity floor (R18 PLAN §3 m7).
DIVERSITY_FLOOR: float = 0.75


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_platinai_5k(csv_path: Path) -> List[Dict[str, str]]:
    """Return rows from the PlatinAI 5 K CSV (SMILES source)."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# SMILES → (coords, atom_types) via RDKit 3-D conformer embedding
# ---------------------------------------------------------------------------
def _smiles_to_pocket_cloud(
    smiles: str,
    max_atoms: int = MAX_POCKET_ATOMS,
) -> Optional[Tuple[List[List[float]], List[int]]]:
    """3-D-embed ``smiles`` and return ``(coords, atom_types)``.

    Returns ``None`` if RDKit cannot parse / embed the SMILES.  Coords
    and atom_types are zero-padded to ``max_atoms``; the real atom
    count is ``sum(1 for a in atom_types if a > 0)``.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        embed_status = int(AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE))
        if embed_status != 0:
            return None
        n_atoms = int(mol_h.GetNumAtoms())
        if n_atoms < 2:
            return None

        coords: List[List[float]] = []
        atom_types: List[int] = []
        conf = mol_h.GetConformer(0)
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
            atom_types.append(int(mol_h.GetAtomWithIdx(i).GetAtomicNum()))

        if len(coords) > max_atoms:
            coords = coords[:max_atoms]
            atom_types = atom_types[:max_atoms]
        else:
            while len(coords) < max_atoms:
                coords.append([0.0, 0.0, 0.0])
                atom_types.append(0)
        return coords, atom_types
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-PH feature extraction (real Wf-1 T5 math)
# ---------------------------------------------------------------------------
def extract_multi_ph_features(
    smiles_list: Sequence[str],
    *,
    n_bins: int = MULTI_PH_N_BINS,
    max_atoms: int = MAX_POCKET_ATOMS,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """Extract Multi-PH (n_bins, n_bins) and single-PH (n_bins,) per mol.

    The single-PH feature is the marginal sum over the coordination-
    distance axis (the canonical reduction of the 2-PH feature map to
    a 1-D Betti curve when the spatial filter is the only varying
    parameter — see :func:`multi_parameter_ph.MultiPHSummarizer.
    marginal_betti`).

    Returns
    -------
    multi_ph_feats : list[Tensor]
        Per-molecule ``(n_bins, n_bins)`` Multi-PH feature maps.
    single_ph_feats : list[Tensor]
        Per-molecule ``(n_bins,)`` marginal Betti curves.
    stats : dict
        ``{"n_parsed", "n_failed"}`` accounting dict.
    """
    import torch

    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    extractor = MultiPHPocketFeatures(
        max_radius=MULTI_PH_MAX_RADIUS,
        max_coord=MULTI_PH_MAX_COORD,
        n_bins=int(n_bins),
    )
    multi_ph: List[Any] = []
    single_ph: List[Any] = []
    n_parsed = 0
    n_failed = 0
    for smi in smiles_list:
        cloud = _smiles_to_pocket_cloud(smi, max_atoms=max_atoms)
        if cloud is None:
            n_failed += 1
            continue
        coords, atom_types = cloud
        coords_t = torch.tensor([coords], dtype=torch.float32)
        atom_types_t = torch.tensor([atom_types], dtype=torch.long)
        try:
            out = extractor.compute_with_atom_types(coords_t, atom_types_t)
        except Exception:
            n_failed += 1
            continue
        # out: (1, n_bins, n_bins) — squeeze the batch axis.
        feat = out[0]
        multi_ph.append(feat.detach())
        # Marginal sum over d-axis (coordination axis) — canonical
        # reduction to a 1-D Betti curve.
        single_ph.append(feat.sum(dim=-1).detach())
        n_parsed += 1
    stats = {"n_parsed": n_parsed, "n_failed": n_failed}
    return multi_ph, single_ph, stats


# ---------------------------------------------------------------------------
# Expressivity — mean pairwise cosine distance on the feature set
# ---------------------------------------------------------------------------
def _mean_pairwise_cosine_distance(feats: Sequence[Any]) -> float:
    """Return the mean off-diagonal ``1 - cos(x, y)`` over ``feats``.

    ``feats`` is a sequence of ``(F,)`` or ``(n_bins, n_bins)`` tensors
    of equal shape; the function flattens each entry to a 1-D vector
    before computing the cosine distance matrix.  The diagonal is
    excluded from the mean — diagonal entries are exactly ``0``.
    """
    import torch

    if len(feats) < 2:
        return 0.0
    flat = torch.stack(
        [f.reshape(-1).to(torch.float32) for f in feats], dim=0,
    )  # (k, F)
    # Cosine *distance* = 1 - cos(x, y).  Normalise rows to unit norm.
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = flat / norm
    cos_sim = unit @ unit.transpose(-1, -2)
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    cos_dist = 1.0 - cos_sim  # (k, k)
    k = cos_dist.shape[0]
    # Mask the diagonal.
    mask = ~torch.eye(k, dtype=torch.bool)
    return float(cos_dist[mask].mean().item())


def compute_expressivity(
    multi_ph: Sequence[Any],
    single_ph: Sequence[Any],
) -> Dict[str, float]:
    """Return ``{multi_ph, single_ph, delta}`` expressivity numbers.

    ``delta = multi_ph - single_ph``.  A positive ``delta`` is the
    R18 PLAN §3 success criterion (the 2-PH map should spread
    molecules *more* than the marginal 1-D Betti curve).
    """
    multi_exp = _mean_pairwise_cosine_distance(multi_ph)
    single_exp = _mean_pairwise_cosine_distance(single_ph)
    return {
        "multi_ph": float(multi_exp),
        "single_ph": float(single_exp),
        "delta": float(multi_exp - single_exp),
    }


# ---------------------------------------------------------------------------
# Diversity — Tanimoto distance on Morgan-ECFP4 fingerprints
# ---------------------------------------------------------------------------
def compute_morgan_fps(
    smiles_list: Sequence[str],
    *,
    radius: int = FP_RADIUS,
    n_bits: int = FP_NBITS,
) -> Tuple["np.ndarray", List[int]]:
    """Compute Morgan-ECFP(radius) fingerprints as a packed ``(k, n_bits)`` matrix.

    Returns the fingerprint matrix and the parsed indices.
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs

    RDLogger.DisableLog("rdApp.*")
    rows: List[Any] = []
    parsed: List[int] = []
    for i, smi in enumerate(smiles_list):
        smi_str = (smi or "").strip()
        if not smi_str:
            continue
        try:
            mol = Chem.MolFromSmiles(smi_str)
        except Exception:
            continue
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        rows.append(arr)
        parsed.append(i)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.uint8), []
    return np.stack(rows, axis=0), parsed


def compute_pairwise_tanimoto_distance(fps: "np.ndarray") -> float:
    """Return mean off-diagonal Tanimoto *distance* over a fingerprint matrix."""
    import numpy as np

    if fps.shape[0] < 2:
        return 0.0
    block = 256
    inter = np.zeros((fps.shape[0], fps.shape[0]), dtype=np.float32)
    for i in range(0, fps.shape[0], block):
        end = min(i + block, fps.shape[0])
        inter[i:end] = (
            fps[i:end].astype(np.float32) @ fps.T.astype(np.float32)
        )
    union_pop = fps.sum(axis=1).astype(np.float32)
    denom = union_pop[:, None] + union_pop[None, :] - inter
    sim = np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)
    dist = 1.0 - sim
    n = dist.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(dist[mask].mean())


def compute_diversity(
    smiles_list: Sequence[str],
    *,
    radius: int = FP_RADIUS,
    n_bits: int = FP_NBITS,
) -> Dict[str, Any]:
    """Compute chemical-diversity (Tanimoto) + parse stats.

    Returns a dict with the mean pairwise Tanimoto distance and the
    number of parsed / failed molecules.
    """
    fps, parsed = compute_morgan_fps(smiles_list, radius=radius, n_bits=n_bits)
    if fps.shape[0] < 2:
        return {
            "mean_tanimoto_distance": 0.0,
            "n_parsed": int(fps.shape[0]),
            "n_failed": int(len(smiles_list) - fps.shape[0]),
        }
    mean_dist = compute_pairwise_tanimoto_distance(fps)
    return {
        "mean_tanimoto_distance": float(mean_dist),
        "n_parsed": int(fps.shape[0]),
        "n_failed": int(len(smiles_list) - fps.shape[0]),
    }


# ---------------------------------------------------------------------------
# Pipeline — load + extract + expressivity + diversity
# ---------------------------------------------------------------------------
def run_bench(
    *,
    platinai_csv: Path,
    n_samples: int,
    n_bins: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Top-level bench orchestrator.

    Loads up to ``n_samples`` SMILES from ``platinai_csv`` (deterministic
    shuffle), extracts Multi-PH + single-PH features, and reports the
    expressivity + diversity metrics in a single dict.
    """
    import numpy as np

    print(f"[1/3] Loading up to {n_samples} SMILES from {platinai_csv}...")
    rows = load_platinai_5k(platinai_csv)
    print(f"  loaded {len(rows)} rows.")
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    sampled = rows[: int(n_samples)]
    smiles_list = [r.get("smiles", "") for r in sampled]
    print(f"  sampled {len(smiles_list)} SMILES.")

    print(f"[2/3] Extracting Multi-PH + single-PH features (n_bins={n_bins})...")
    t0 = time.perf_counter()
    multi_ph, single_ph, mph_stats = extract_multi_ph_features(
        smiles_list, n_bins=int(n_bins),
    )
    mph_wall = float(time.perf_counter() - t0)
    print(
        f"  parsed={mph_stats['n_parsed']} failed={mph_stats['n_failed']} "
        f"wall={mph_wall:.2f}s"
    )

    print("[3/3] Computing expressivity + diversity...")
    if len(multi_ph) >= 2:
        expressivity = compute_expressivity(multi_ph, single_ph)
    else:
        expressivity = {"multi_ph": 0.0, "single_ph": 0.0, "delta": 0.0}

    # Use the same SMILES pool for the diversity bench (canonical
    # canonical via RDKit inside ``compute_morgan_fps``).
    diversity = compute_diversity(smiles_list)
    diversity_meets_floor = (
        diversity["mean_tanimoto_distance"] >= DIVERSITY_FLOOR
    )

    return {
        "n_samples": int(n_samples),
        "n_parsed_multiph": int(mph_stats["n_parsed"]),
        "n_failed_multiph": int(mph_stats["n_failed"]),
        "n_parsed_diversity": int(diversity["n_parsed"]),
        "n_failed_diversity": int(diversity["n_failed"]),
        "feature_dim_multiph": int(n_bins * n_bins),
        "feature_dim_single_ph": int(n_bins),
        "n_bins": int(n_bins),
        "max_radius": float(MULTI_PH_MAX_RADIUS),
        "max_coord": float(MULTI_PH_MAX_COORD),
        "max_pocket_atoms": int(MAX_POCKET_ATOMS),
        "mph_wall_s": float(mph_wall),
        "expressivity": {
            "multi_ph": float(expressivity["multi_ph"]),
            "single_ph": float(expressivity["single_ph"]),
            "delta": float(expressivity["delta"]),
            "multi_ph_more_expressive": bool(expressivity["delta"] > 0.0),
        },
        "diversity": {
            "mean_tanimoto_distance": float(
                diversity["mean_tanimoto_distance"]
            ),
            "meets_m7_floor": bool(diversity_meets_floor),
            "m7_floor": float(DIVERSITY_FLOOR),
        },
    }


# ---------------------------------------------------------------------------
# Markdown table — human-readable summary
# ---------------------------------------------------------------------------
def render_table(metrics: Dict[str, Any]) -> str:
    """Render a Markdown comparison table from the bench ``metrics``."""
    exp = metrics["expressivity"]
    div = metrics["diversity"]
    out = [
        "# Wf-31 Phase 3 — Multi-PH feature extraction + diversity bench",
        "",
        f"**Dataset:** PlatinAI 5K (n={metrics['n_samples']}, "
        f"parsed Multi-PH={metrics['n_parsed_multiph']}, "
        f"parsed diversity={metrics['n_parsed_diversity']})",
        f"**Multi-PH bins:** {metrics['n_bins']} "
        f"(dim_multi_ph={metrics['feature_dim_multiph']}, "
        f"dim_single_ph={metrics['feature_dim_single_ph']})",
        f"**Pocket budget:** max_atoms={metrics['max_pocket_atoms']}, "
        f"max_radius={metrics['max_radius']}, "
        f"max_coord={metrics['max_coord']}",
        "",
        "## Expressivity (mean pairwise cosine distance)",
        "",
        "| Feature family | Mean 1 - cos(x, y) |",
        "|---|---|",
        f"| Multi-PH (2-parameter) | {exp['multi_ph']:.3f} |",
        f"| Single-PH (1-D Betti marginal) | {exp['single_ph']:.3f} |",
        f"| Delta (multi - single) | {exp['delta']:+.3f} "
        f"({'PASS' if exp['multi_ph_more_expressive'] else 'FAIL'}) |",
        "",
        "## Chemical diversity (Tanimoto distance)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean pairwise Tanimoto distance | "
        f"{div['mean_tanimoto_distance']:.3f} |",
        f"| m7 floor (R18 PLAN §3) | {div['m7_floor']:.3f} |",
        f"| Meets m7 floor | "
        f"{'PASS' if div['meets_m7_floor'] else 'FAIL'} |",
        "",
        "## Wall-time",
        "",
        f"* Multi-PH extraction: **{metrics['mph_wall_s']:.2f} s** for "
        f"{metrics['n_samples']} molecules (parsed: "
        f"{metrics['n_parsed_multiph']}).",
        "",
        "## Honest framing",
        "",
        "* The **single-PH baseline** is the marginal sum over the "
        "coordination-distance axis — a canonical reduction of the 2-PH "
        "feature map to a 1-D Betti curve.  It is *not* a separate "
        "extraction; it is the same feature map collapsed along one axis.",
        "* The **expressivity** metric is the mean pairwise cosine "
        "distance (1 - cos).  It is a coarse proxy for how well the "
        "feature family spreads samples in the embedding space; a "
        "higher value means more distinguishable features.",
        "* The **diversity** metric is the mean pairwise Tanimoto "
        "*distance* on Morgan-ECFP4 fingerprints — a separate chemistry "
        "measurement that does not involve the Multi-PH extractor.",
        "* The m7 floor ``DIVERSITY_FLOOR = 0.75`` is the R18 PLAN §3 "
        "success criterion; the bench does *not* enforce it as a "
        "fail-condition (the bench is descriptive, not gating).",
    ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data", type=str,
        default="molmetal/data/platinai_5k.csv",
        help="PlatinAI 5K CSV path (SMILES source).",
    )
    p.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of SMILES to sample (default: 200).",
    )
    p.add_argument(
        "--n-bins", type=int, default=MULTI_PH_N_BINS,
        help=f"Multi-PH bin count per filter axis (default: "
             f"{MULTI_PH_N_BINS}).",
    )
    p.add_argument(
        "--output", type=str,
        default="molmetal/reports/wf31_multi_ph_diversity",
        help="Output directory for metrics + Markdown.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for SMILES sampling (default: 42).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    platinai_csv = Path(args.data)
    if not platinai_csv.exists():
        rel = _REPO_ROOT / args.data
        if rel.exists():
            platinai_csv = rel
    if not platinai_csv.exists():
        log.error("platinai CSV not found: %s", platinai_csv)
        return 2

    log.info("Wf-31 Phase 3 — Multi-PH feature extraction + diversity bench")
    log.info("  platinai_csv = %s", platinai_csv)
    log.info("  output       = %s", out_dir)
    log.info("  n_samples    = %s", args.n_samples)
    log.info("  n_bins       = %s", args.n_bins)

    metrics = run_bench(
        platinai_csv=platinai_csv,
        n_samples=int(args.n_samples),
        n_bins=int(args.n_bins),
        seed=int(args.seed),
    )

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    log.info("wrote %s", metrics_path)

    table_path = out_dir / "comparison_table.md"
    with open(table_path, "w") as f:
        f.write(render_table(metrics))
    log.info("wrote %s", table_path)

    exp = metrics["expressivity"]
    div = metrics["diversity"]
    log.info(
        "  expressivity multi_ph=%.3f single_ph=%.3f delta=%+.3f",
        exp["multi_ph"], exp["single_ph"], exp["delta"],
    )
    log.info(
        "  diversity tanimoto=%.3f meets_m7_floor=%s",
        div["mean_tanimoto_distance"], div["meets_m7_floor"],
    )
    log.info("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
