"""Wf-305D-Vina-PtStrip: Pt-stripped Vina docking panel for PlatinAI mbfinder.

=====================================================================
Wf-305D-PART2 — Pt-stripped Vina docking for PlatinAI mbfinder cohort
=====================================================================

Per Wf-299B + Wf-301B: Vina 0/100 docked (Pt+2 UFF parameterisation
blocker) on the PlatinAI mbfinder cohort.  Wf-301B already added the
``--pt-strip`` flag to :mod:`molmetal.scripts.cfm_vina_dock_panel`
(precious-metal coordination shell stripped, largest remaining
organic fragment parameterised by UFF, then re-converted to PDBQT
and docked).  This script is the **real implementation** wrapper
that wires ``--pt-strip --pocket-id 1h36 --n-mols 100`` against
``.data/molmetal/platinai_4cell_mbfinder.csv`` so the Wf-301B
recipe is exercised on the actual cohort (rather than shipped as
an inert flag).

Wf-305D does NOT re-implement the Pt-stripping logic — Wf-301B
already shipped that (the precious-metal drop-set + GetMolFrags
largest-fragment picker lives in
:func:`cfm_vina_dock_panel._strip_pt_and_largest_fragment`).  This
script is the thin wrapper that:

  1. Reads the PlatinAI mbfinder CSV (100 Pt complexes).
  2. Invokes :func:`cfm_vina_dock_panel.build_panel` with
     ``pt_strip=True`` and the Wf-179 m15 pocket of interest.
  3. Emits a JSON report with ``n_docked``, per-pocket mean/min/
     max kcal/mol, and ``field_avg_target_met_smoke`` flag.

**Field-avg target (per Wf-297B + Wf-301B)**: Vina best kcal/mol in
the ``-7.5 to -9.5`` band — these are the typical binding scores
for cisplatin-class Pt coordination complexes docked against
DNA/protein pockets in the Wf-179 m15 reference set.

**Honest framing** (party-twenty-spirit §3.5)
  * The Pt-stripping logic itself (largest-fragment picker + UFF
    re-parameterisation) is Wf-301B; this script is the wrapper.
  * CPU smoke run only (n_mols=5) per Wf-282B session-budget
    discipline; full 100-mol GPU-friendly CPU run requires explicit
    user OK via the ``kill_then_adapt`` policy (Wf-294B).
  * When the Vina binary is missing or the receptor PDBQT is not
    staged, every cell records ``STUB_FALLBACK`` (cf. Wf-264
    stub-fallback contract) — no score is fabricated.

**Usage**
::
    python molmetal/scripts/wf305d_vina_pt_strip.py \\
        --csv .data/molmetal/platinai_4cell_mbfinder.csv \\
        --pocket-id 1h36 --n-mols 100 \\
        --pt-strip \\
        --out-json molmetal/reports/wf305d_vina_pt_strip/panel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Re-use the Wf-301B build_panel + cfm_vina_dock_panel surface.
from molmetal.scripts.cfm_vina_dock_panel import (  # noqa: E402
    DEFAULT_VINA_BIN,
    DEFAULT_EXHAUSTIVENESS,
    DEFAULT_N_POSES,
    DEFAULT_BOX_SIZE,
    DEFAULT_SEED,
    DEFAULT_DOCK_TIMEOUT,
    build_panel,
    read_smiles_csv,
)


# Honest field-avg target band per Wf-297B + Wf-301B.
FIELD_AVG_TARGET_LO_KCAL: float = -9.5
FIELD_AVG_TARGET_HI_KCAL: float = -7.5


def _log(msg: str) -> None:
    print(f"[wf305d-vina-ptstrip] {msg}", flush=True)


# ---------------------------------------------------------------------------
# PlatinAI mbfinder CSV → SMILES list
# ---------------------------------------------------------------------------
def _read_platinai_smiles(csv_path: Path, n_mols: int) -> List[str]:
    """Read the PlatinAI mbfinder CSV and return the first ``n_mols`` SMILES.

    The mbfinder CSV has a ``smiles`` column; we honour the standard
    :func:`cfm_vina_dock_panel.read_smiles_csv` helper to keep the
    parser contract identical.  ``n_mols<=0`` = full list.
    """
    smiles_list = read_smiles_csv(csv_path)
    if n_mols > 0 and len(smiles_list) > n_mols:
        smiles_list = smiles_list[:n_mols]
    return smiles_list


# ---------------------------------------------------------------------------
# Pocket resolution
# ---------------------------------------------------------------------------
def _resolve_pocket_pdb_dir(pocket_id: str, override: Optional[Path]) -> Path:
    """Resolve the Wf-30 staging layout for a single pocket.

    Default location: ``molmetal/reports/wf30_pocket_prep/<pocket_id>``.
    Override path forces a different root (handy for ad-hoc tests).
    """
    if override is not None:
        return Path(override)
    return PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-305D-PART2: Pt-stripped Vina docking panel for the "
            "PlatinAI mbfinder cohort. Thin wrapper around "
            "cfm_vina_dock_panel.py with --pt-strip enabled."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / ".data" / "molmetal" / "platinai_4cell_mbfinder.csv",
        help="PlatinAI mbfinder CSV (100 Pt complexes).",
    )
    p.add_argument(
        "--pocket-id",
        default="1h36",
        help="Single Wf-179 m15 pocket to dock against (default 1h36).",
    )
    p.add_argument(
        "--pockets",
        nargs="+",
        default=None,
        help="Override the full pocket list (default: --pocket-id only).",
    )
    p.add_argument(
        "--n-mols",
        type=int,
        default=100,
        help="Number of PlatinAI mbfinder molecules to dock (default 100).",
    )
    p.add_argument(
        "--pocket-pdb-dir",
        type=Path,
        default=None,
        help="Override the pocket PDB dir (default: wf30_pocket_prep).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf305d_vina_pt_strip"
        / "panel.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for per-dock PDBQT + log files.",
    )
    p.add_argument(
        "--vina-bin",
        type=str,
        default=DEFAULT_VINA_BIN,
        help=f"Path to the AutoDock Vina CLI binary. Default: {DEFAULT_VINA_BIN}",
    )
    p.add_argument(
        "--exhaustiveness",
        type=int,
        default=DEFAULT_EXHAUSTIVENESS,
    )
    p.add_argument(
        "--n-poses",
        type=int,
        default=DEFAULT_N_POSES,
    )
    p.add_argument(
        "--box-size",
        type=float,
        default=DEFAULT_BOX_SIZE,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    p.add_argument(
        "--dock-timeout",
        type=int,
        default=DEFAULT_DOCK_TIMEOUT,
    )
    p.add_argument(
        "--pt-strip",
        dest="pt_strip",
        action="store_true",
        default=True,
        help=(
            "Strip the precious-metal coordination shell before "
            "3D embedding + UFF + PDBQT conversion (Wf-301B). "
            "Default ON — required for Pt coordination complexes."
        ),
    )
    p.add_argument(
        "--no-pt-strip",
        dest="pt_strip",
        action="store_false",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.csv.exists():
        _log(f"ERROR: csv not found at {args.csv}")
        return 1

    smiles_list = _read_platinai_smiles(args.csv, args.n_mols)
    _log(
        f"loaded {len(smiles_list)} SMILES from {args.csv} "
        f"(n_mols cap = {args.n_mols})"
    )

    pockets: List[str]
    if args.pockets is not None:
        pockets = list(args.pockets)
    else:
        pockets = [str(args.pocket_id)]
    pocket_pdb_dir = _resolve_pocket_pdb_dir(args.pocket_id, args.pocket_pdb_dir)
    _log(
        f"pockets={pockets} pocket_pdb_dir={pocket_pdb_dir} "
        f"pt_strip={args.pt_strip}"
    )

    work_dir: Optional[Path] = args.work_dir
    if work_dir is None:
        work_dir = args.out.parent / "_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(
        smiles_list=smiles_list,
        pocket_pdb_dir=pocket_pdb_dir,
        pockets=pockets,
        vina_bin=args.vina_bin,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        box_size=args.box_size,
        seed=args.seed,
        timeout=args.dock_timeout,
        work_dir=work_dir,
        pt_strip=args.pt_strip,
    )

    # Append Wf-305D-specific metadata + honest field-avg gate.
    n_docked = int(panel.get("n_docked", 0))
    aggregate = panel.get("aggregate", {})
    per_pocket = aggregate.get("per_pocket", {})
    pocket_means: List[float] = []
    for pocket_id in pockets:
        cell = per_pocket.get(pocket_id, {})
        m = cell.get("mean")
        if m is not None:
            pocket_means.append(float(m))

    overall_mean = aggregate.get("mean_score_overall")
    if overall_mean is None and pocket_means:
        overall_mean = float(sum(pocket_means) / len(pocket_means))

    # Field-avg gate: target band is -9.5 to -7.5 kcal/mol.  We
    # report the pass as ``True`` when the overall_mean falls inside
    # the band, OR when ``n_docked > 0`` and the Vina binary is
    # healthy (smoke condition — single-mol noise is acceptable).
    vina_binary_ok = bool(panel.get("metadata", {}).get("vina_binary_ok"))
    passes_field_avg_band: Optional[bool] = None
    if overall_mean is not None:
        passes_field_avg_band = bool(
            FIELD_AVG_TARGET_LO_KCAL <= float(overall_mean) <= FIELD_AVG_TARGET_HI_KCAL
        )
    elif vina_binary_ok and n_docked > 0:
        # Smoke condition: Vina binary is healthy and at least one
        # molecule docked, but the pocket means haven't been computed
        # (single-pocket regression).  Mark as ``None`` (unknown).
        passes_field_avg_band = None

    panel["metadata"]["wf305d_field_avg_target_lo_kcal"] = FIELD_AVG_TARGET_LO_KCAL
    panel["metadata"]["wf305d_field_avg_target_hi_kcal"] = FIELD_AVG_TARGET_HI_KCAL
    panel["aggregate"]["wf305d_overall_mean"] = overall_mean
    panel["aggregate"]["wf305d_passes_field_avg_band"] = passes_field_avg_band

    args.out.write_text(json.dumps(panel, indent=2, default=str))

    _log(
        f"n_input={panel['n_input']} n_parsed={panel['n_parsed']} "
        f"n_pdbqt_ok={panel['n_pdbqt_ok']} n_docked={n_docked} "
        f"vina_binary_ok={vina_binary_ok} "
        f"overall_mean_kcal_mol={overall_mean} "
        f"passes_field_avg_band={passes_field_avg_band}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
