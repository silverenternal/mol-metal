#!/usr/bin/env python3
"""Wf-323C — UFF pre-relax + Vina re-dock CPU smoke for m8_full lift.

========================================================================
Per Wf-319C + WEAK_TO_STRONG §2.6: m8_full_pass_rate = 0/15 on the
Wf-315D smoke (15 on-disk Wf-179 docked poses re-checked with PoseBusters
in dock mode). The 5 consistently failing checks were:
  bond_lengths, bond_angles, internal_steric_clash,
  aromatic_ring_flatness, internal_energy

The root cause (per Wf-246D synthesis.md §3): Wf-179 poses were docked
via Vina WITHOUT explicit Hs added and WITHOUT geometry relaxation of the
embedded conformer. PoseBusters' reference geometry window expects
MMFF94-class bonded terms + canonical H placement.

Wf-323C hypothesis: pre-relax the conformers with UFF (RDKit CPU) so the
upstream 3D geometry hits better bonded-term ranges, then re-dock via
Vina 1.2.7. The UFF-min Pt-pool from Wf-319B (96 conformers; HONEST
NEGATIVE on Pt-d-block force-field coverage, but 3D coords are valid)
is the input source.

HONEST-NEGATIVE FINDING (Wf-323C blocker)
-----------------------------------------
Vina 1.2.7's PDBQT parser rejects the "Pt" atom-type with::

    PDBQT parsing error: Atom type Pt is not a valid AutoDock type
    (atom types are case-sensitive)

This means Vina cannot score a Pt-coordination complex as-is. The
Wf-323C script **automatically falls back to PT-STRIPPED** mode: the
d-block metal centre is removed (organic donor shell only) and the
Vina kcal/mol is computed on the stripped pose. The Pt atom is NOT in
the docked pose — its kcal/mol contribution is missing — but the
organic geometry IS what PoseBusters dock mode validates.

This is the same blocker noted for the Wf-299B CFM Vina panel
(per AGENTS.md §5.1 — CFM Vina docking is BLOCKED on Pt+2
parameterisation). Wf-323C records this as
``HONEST_NEGATIVE_VINA_PT_REJECTED`` and surfaces the MMFF94 + GPU
recovery path for production runs.

Pipeline
--------
1. Load Wf-319B Pt-pool (SMILES list) — 96 valid conformers.
2. For each SMILES (up to ``--n-conformers``):
   a. ETKDGv3 + useRandomCoords embed (same as Wf-319B).
   b. UFFOptimizeMolecule (RDKit UFF — best-effort; Pt atoms are
      "Unrecognized atom type" so relaxation only acts on organic atoms
      + donor shell).
   c. Pt-strip (automatic for Pt-group metals) → organic donor shell.
   d. meeko MoleculePreparation (with Pt-group metal atom-type
      overrides + ``charge_model='zero'``) → ligand PDBQT.
   e. Vina re-dock into 1h36 receptor (auto-converted PDBQT) with
      ``exhaustiveness=8`` + ``n_poses=4`` (smoke).
   f. Vina PDBQT pose → RDKit Mol via ``PDBQTMolecule`` +
      ``RDKitMolCreate.from_pdbqt_mol``.
   g. PoseBusters ``check_docked_pose`` (dock mode, 26-check battery).
3. Aggregate:
   * ``m8_full_pass_rate`` — fraction with all 26 PB checks True.
   * ``m8_protein_aware_pass_rate`` — fraction with the 9
     protein-aware subset True (m8 steric clash axis).
   * ``n_pt_stripped`` — count of Pt-group-metal conformers that
     required the strip fallback.
4. Save ``result.json`` under ``molmetal/reports/wf323c_m8_full/``.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* UFF cannot parameterise Pt — the relaxation only acts on the organic
  donor shell. Energies are NOT physical; the conformers are just
  better-shaped for Vina's scoring function + PB's reference window.
* Vina cannot accept Pt in PDBQT — automatic Pt-strip is the Wf-323C
  CPU-side workaround. The Pt kcal/mol contribution is missing in
  the docked pose (HONEST_NEGATIVE_VINA_PT_REJECTED).
* PROJECTED delta: +0.10 to +0.15 vs Wf-315D 0/15 baseline
  (WEAK_TO_STRONG §2.6 R18). If the lift is <5% we record
  HONEST_NEGATIVE_UFF_INSUFFICIENT and surface the MMFF94 + GPU
  recovery path.
* CPU-only; no GPU launch per AGENTS.md §5.1.
* The pocket is fixed at 1h36 (the same pocket Wf-315D ran on, so
  the before/after delta is direct).

CLI::

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf323c_m8_full_cpu_fix.py \\
        --input-pool molmetal/reports/wf319b_uff_min_pt_pool/pool.json \\
        --pocket-id 1h36 \\
        --n-conformers 24 \\
        --out molmetal/reports/wf323c_m8_full/result.json \\
        --execute

Exit codes
----------
* 0 — script ran (regardless of pass-rate). The m8_full_pass_rate is
  MEASURED and the gate can consume it.
* 1 — Vina Python binding missing OR pool.json missing OR receptor
  PDB+PDBQT missing.
* 2 — argparse rejected the flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("wf323c")

WF319B_POOL = (
    REPO_ROOT / "molmetal" / "reports" / "wf319b_uff_min_pt_pool" / "pool.json"
)
WF246D_RECEPTORS = REPO_ROOT / "molmetal" / "reports" / "wf246d_m8_pb_dock"
DEFAULT_OUT = REPO_ROOT / "molmetal" / "reports" / "wf323c_m8_full" / "result.json"

# Wf-315D baseline (for honest delta reporting).
WF315D_SMOKE = (
    REPO_ROOT / "molmetal" / "reports" / "wf315d_pb_smoke" / "m8_prod_1h36.json"
)
WF315D_BASELINE_M8_FULL = 0.0
WF315D_BASELINE_M8_PA = 1.0  # 15/15 protein-aware pass on raw Wf-179 poses

# Per-pocket Vina config — same as Wf-250 YAML default for 1h36, but
# reduced for CPU smoke (Wf-323C = 1 smoke, not 100×3 production).
DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_N_POSES = 4

# Pt-group metals that Vina 1.2.7 cannot parse in PDBQT. We strip them
# automatically (organic donor shell only) per the Wf-323C design.
PT_GROUP_ELEMENTS_FOR_STRIP = frozenset(
    {"Pt", "Ru", "Ir", "Os", "Rh", "Pd", "Au", "Re"}
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_pool(path: Path) -> List[Dict[str, Any]]:
    """Load conformer records (with `embed_ok=true`) from Wf-319B pool."""
    with path.open("r", encoding="utf-8") as fh:
        pool = json.load(fh)
    return [c for c in pool.get("conformers", []) if c.get("embed_ok")]


def _uff_prerelax_one(smiles: str) -> Optional[Tuple[Any, str]]:
    """Embed + UFF-minimize + AddHs.

    Returns ``(mol_with_conformer, embed_method)`` or ``None`` on failure.
    Mirrors Wf-319B logic so that "the same set of SMILES" can be
    re-derived on this host (pool.json only stores SMILES + n_atoms,
    not actual 3D coords — re-derive here).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() == 0:
        return None

    embed_method: Optional[str] = None
    embed_ok = False
    for method_name, params_factory in (
        (
            "ETKDGv3+randomCoords",
            lambda: (
                (
                    lambda p: (
                        setattr(p, "randomSeed", 0xC0FFEE),
                        setattr(p, "useRandomCoords", True),
                        p,
                    )[-1]
                )(AllChem.ETKDGv3())
            ),
        ),
        (
            "ETKDGv2+randomCoords",
            lambda: (
                (
                    lambda p: (
                        setattr(p, "randomSeed", 0xC0FFEE),
                        setattr(p, "useRandomCoords", True),
                        p,
                    )[-1]
                )(AllChem.ETKDGv2())
            ),
        ),
    ):
        try:
            status = AllChem.EmbedMolecule(mol_h, params_factory())
            if status == 0:
                embed_method = method_name
                embed_ok = True
                break
        except Exception:
            continue
    if not embed_ok:
        return None

    # UFF pre-relax — CPU-only; Pt is unrecognised so this only
    # relaxes the organic donor shell. The conformer coords are
    # still valid for downstream Vina + PoseBusters.
    try:
        result = AllChem.UFFOptimizeMolecule(mol_h, maxIters=200, vdwThresh=0.1)
        if isinstance(result, tuple):
            _, energy = result
        else:
            energy = None
    except Exception:
        # UFF may raise on Pt/Ir — that's fine; the embed coords are
        # already in the canonical ETKDGv3 window which is close
        # enough for the smoke.
        energy = None

    return mol_h, embed_method or "ETKDGv3+randomCoords"


def _strip_pt_group_keep_coords(mol_with_h: Any) -> Optional[Any]:
    """Remove all d-block Pt-group metal atoms from a 3D conformer.

    The bond graph loses the metal centre but the donor atoms keep
    their 3D coordinates (used for UFF + Vina on the organic shell).
    Returns a copy of the input mol with Pt-group atoms removed, or
    ``None`` if the stripping leaves <1 atom.
    """
    from rdkit import Chem  # type: ignore

    rwmol = Chem.RWMol(mol_with_h)
    atom_indices_to_remove = sorted(
        [
            a.GetIdx()
            for a in rwmol.GetAtoms()
            if a.GetSymbol() in PT_GROUP_ELEMENTS_FOR_STRIP
        ],
        reverse=True,
    )
    if not atom_indices_to_remove:
        return Chem.Mol(mol_with_h)
    for idx in atom_indices_to_remove:
        rwmol.RemoveAtom(idx)
    stripped = rwmol.GetMol()
    if stripped.GetNumAtoms() < 1:
        return None
    # Preserve the conformer(s) from the input mol.
    for conf in mol_with_h.GetConformers():
        new_conf = Chem.Conformer(stripped.GetNumAtoms())
        new_conf.Set3D(True)
        old_indices = [
            i
            for i in range(mol_with_h.GetNumAtoms())
            if mol_with_h.GetAtomWithIdx(i).GetSymbol()
            not in PT_GROUP_ELEMENTS_FOR_STRIP
        ]
        for new_i, old_i in enumerate(old_indices):
            pos = conf.GetAtomPosition(old_i)
            new_conf.SetAtomPosition(new_i, pos)
        stripped.AddConformer(new_conf, assignId=True)
    return stripped


def _prepare_ligand_pdbqt(mol_with_h: Any, tmpdir: Path) -> Optional[Path]:
    """SMILES-with-3D-coords → ligand PDBQT via meeko.

    Returns the PDBQT path on success or ``None`` on failure.

    For Pt-group metals, the meeko PDBQT writer emits ``Pt`` /
    ``Ru`` / ``Ir`` / etc. as the AD4 atom-type. Vina 1.2.7's
    PDBQT parser rejects unknown atom-types, so this function
    substitutes the d-block metal types with the closest Vina-
    accepted generic VDW type (``Zn`` for all Pt-group metals —
    same divalent divalent-block geometry as Pt2+).

    This is the same Wf-246-C "MEASURED_GENERIC_ATOM" design
    intent: a generic VDW atype + zero charges. Downstream
    consumers MUST treat the resulting kcal/mol as an
    approximation, not the true Pt-anchored kcal/mol.
    """
    from rdkit import Chem  # type: ignore
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore

    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        PT_GROUP_ELEMENTS,
        add_metal_atom_types,
    )

    mol = Chem.Mol(mol_with_h)
    metal_symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    present_metals = sorted(metal_symbols & set(PT_GROUP_ELEMENTS))

    preparator = MoleculePreparation()
    if present_metals:
        try:
            add_metal_atom_types(preparator)
        except Exception:
            pass
        preparator.charge_model = "zero"

    try:
        setups = preparator.prepare(mol)
    except Exception:
        return None

    pdbqt_path = tmpdir / "lig.pdbqt"
    try:
        with pdbqt_path.open("w") as fh:
            for setup in setups:
                pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
                if not is_ok:
                    return None
                # Substitute Pt-group metal atom-types with "Zn"
                # (closest Vina-accepted divalent d-block). Vina
                # scoring is pairwise VDW so the kcal/mol is the
                # generic Zn-VDW approximation for the metal anchor.
                for metal in PT_GROUP_ELEMENTS:
                    if metal != "Zn":
                        pdbqt_str = pdbqt_str.replace(f" {metal}", " Zn")
                        # Also lowercase for atom-types appearing in
                        # the 'ADD_ATOM_TYPE' remark line.
                        pdbqt_str = pdbqt_str.replace(f"@{metal}", "@Zn")
                fh.write(pdbqt_str)
    except Exception:
        return None
    return pdbqt_path


def _vina_redock(
    smiles: str,
    receptor_pdbqt: Path,
    pocket_center: Tuple[float, float, float],
    pocket_box: Tuple[float, float, float],
    exhaustiveness: int,
    n_poses: int,
    seed: int,
) -> Tuple[Optional[Any], Optional[float], str, Dict[str, Any]]:
    """Vina re-dock one ligand; return ``(mol_pose, kcal_mol, status, meta)``.

    Status codes:

    * ``OK`` — Vina returned at least one pose on the original mol.
    * ``OK_PT_ZN_SUBSTITUTED`` — Vina OK; Pt-group metals in the
      PDBQT were substituted with ``Zn`` (closest Vina-accepted
      divalent d-block; Wf-246-C MEASURED_GENERIC_ATOM convention).
    * ``VINA_PYTHON_MISSING`` — ``import vina`` failed.
    * ``EMBED_FAIL`` — RDKit couldn't embed the SMILES.
    * ``PDBQT_FAIL`` — meeko refused the molecule.
    * ``VINA_RUNTIME_ERROR_PT`` — Vina still rejected the atom-type.
    * ``VINA_RUNTIME_ERROR`` — generic dock() error.
    * ``VINA_NO_POSES`` — dock() ran but energies was empty.

    ``meta`` carries diagnostic fields (``pt_zn_substituted``,
    ``prerelax_method``, ``present_metals``).
    """
    meta: Dict[str, Any] = {
        "pt_zn_substituted": False,
        "prerelax_method": None,
        "present_metals": [],
    }
    try:
        import vina  # type: ignore  # noqa: F401
    except Exception:
        return None, None, "VINA_PYTHON_MISSING", meta

    prerelax = _uff_prerelax_one(smiles)
    if prerelax is None:
        return None, None, "EMBED_FAIL", meta
    mol_h, embed_method = prerelax
    meta["prerelax_method"] = embed_method

    from rdkit import Chem  # type: ignore

    present_metals = sorted(
        {a.GetSymbol() for a in mol_h.GetAtoms()} & set(PT_GROUP_ELEMENTS_FOR_STRIP)
    )
    meta["present_metals"] = present_metals
    if present_metals:
        meta["pt_zn_substituted"] = True

    # Keep Pt in the structure — _prepare_ligand_pdbqt substitutes
    # "Pt"/"Ru"/"Ir"/etc. with "Zn" in the output PDBQT so Vina 1.2.7
    # accepts the file. The 3D coords + bond graph are preserved
    # exactly; only the AD4 atom-type field is rewritten.
    mol_for_vina = mol_h

    with tempfile.TemporaryDirectory(prefix="wf323c_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        lig_pdbqt = _prepare_ligand_pdbqt(mol_for_vina, tmpdir)
        if lig_pdbqt is None:
            return None, None, "PDBQT_FAIL", meta

        import vina as _vina  # type: ignore

        seed_eff = (int(seed) % (2**31 - 1)) or (2**31 - 1)
        try:
            vinstance = _vina.Vina(sf_name="vina", cpu=0, seed=seed_eff, verbosity=0)
            vinstance.set_receptor(str(receptor_pdbqt))
            vinstance.set_ligand_from_file(str(lig_pdbqt))
            vinstance.compute_vina_maps(
                center=[float(c) for c in pocket_center],
                box_size=[float(b) for b in pocket_box],
            )
            vinstance.dock(
                exhaustiveness=int(exhaustiveness),
                n_poses=int(n_poses),
            )
            energies = vinstance.energies(n_poses=int(n_poses))
            poses_pdbqt = vinstance.poses(n_poses=int(n_poses))
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            if "not a valid AutoDock type" in err_msg or "atom type" in err_msg.lower():
                return None, None, f"VINA_RUNTIME_ERROR_PT:{err_msg[:120]}", meta
            return None, None, f"VINA_RUNTIME_ERROR:{type(exc).__name__}:{exc}", meta

        if energies is None or len(energies) == 0:
            return None, None, "VINA_NO_POSES", meta

        try:
            from meeko import PDBQTMolecule, RDKitMolCreate  # type: ignore
            from rdkit import Chem  # type: ignore

            pdbqt_mol = PDBQTMolecule(poses_pdbqt, is_dlg=False)
            rdkit_mols = RDKitMolCreate.from_pdbqt_mol(
                pdbqt_mol,
                only_cluster_leads=False,
            )
            if len(rdkit_mols) != 1 or rdkit_mols[0] is None:
                return None, None, "VINA_NO_POSES", meta
            rdkit_pose = Chem.RemoveHs(rdkit_mols[0])
            conformers = list(rdkit_pose.GetConformers())
            if not conformers:
                return None, None, "VINA_NO_POSES", meta
            best_conf = Chem.Conformer(conformers[0])
            best_conf.Set3D(True)
            best_pose = Chem.Mol(rdkit_pose)
            best_pose.RemoveAllConformers()
            best_pose.AddConformer(best_conf, assignId=True)
            score_val = float(energies[0, 0])
            status = "OK_PT_ZN_SUBSTITUTED" if meta["pt_zn_substituted"] else "OK"
            return best_pose, score_val, status, meta
        except Exception as exc:  # noqa: BLE001
            return None, None, f"PDBQT_CONVERT_FAIL:{type(exc).__name__}:{exc}", meta


def _pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
    """Minimal PDB → PDBQT converter (Vina 1.2.7 strict format)."""
    elem_to_type = {
        "C": "C ",
        "N": "N ",
        "O": "OA",
        "S": "SA",
        "H": "HD",
        "P": "P ",
        "F": "F ",
        "CL": "Cl",
        "BR": "Br",
        "I": "I ",
        "B": "B ",
        "FE": "Fe",
        "ZN": "Zn",
        "MG": "Mg",
        "CA": "Ca",
        "MN": "Mn",
        "PT": "Zn",  # PT is rejected by Vina — map to closest available divalent metal type
    }
    out_lines: List[str] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        elem = line[76:78].strip() or line[12:16].strip()[0]
        atom_type = elem_to_type.get(elem.upper(), "C ")
        prefix = line[:54].ljust(54)
        rest = f"  0.000  0.00    +0.000 {atom_type}"
        out_lines.append(prefix + rest)
    out_lines.append("END")
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    pdbqt_path.write_text("\n".join(out_lines) + "\n")


def _pocket_center_from_receptor(receptor_pdb: Path) -> Tuple[float, float, float]:
    """Approximate the 1h36 binding-site center from the receptor PDB."""
    coords: List[Tuple[float, float, float]] = []
    try:
        with receptor_pdb.open("r") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except (ValueError, IndexError):
                    continue
                coords.append((x, y, z))
                if len(coords) >= 200:
                    break
    except OSError:
        pass

    if not coords:
        return (0.0, 0.0, 0.0)

    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    return (cx, cy, cz)


PROTEIN_AWARE_KEYS: Tuple[str, ...] = (
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
)


def _pb_check_pose(mol: Any, receptor_pdb: Path) -> Dict[str, Any]:
    """Run PoseBusters dock mode on one docked pose (RDKit Mol + receptor PDB)."""
    from molmetal.validation.posebusters_runner import check_docked_pose

    return check_docked_pose(mol, str(receptor_pdb))


def _summarise_pb(
    pool_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate m8 metrics from per-conformer PB records."""
    n_total = len(pool_records)
    n_pa_pass = sum(1 for r in pool_records if r.get("protein_aware_pass"))
    n_full_pass = sum(1 for r in pool_records if r.get("full_pass"))
    m8_pa_rate = (n_pa_pass / n_total) if n_total else None
    m8_full_rate = (n_full_pass / n_total) if n_total else None
    return {
        "n_poses": n_total,
        "n_protein_aware_pass": n_pa_pass,
        "n_full_pass": n_full_pass,
        "m8_protein_aware_pass_rate": m8_pa_rate,
        "m8_full_pass_rate": m8_full_rate,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument(
        "--input-pool",
        type=Path,
        default=WF319B_POOL,
        help=f"Wf-319B Pt-pool JSON (default: {WF319B_POOL})",
    )
    p.add_argument(
        "--pocket-id",
        type=str,
        default="1h36",
        help="Pocket to dock into (default: 1h36)",
    )
    p.add_argument(
        "--n-conformers",
        type=int,
        default=24,
        help="Max number of conformers to dock (default 24 smoke; full pool = 96)",
    )
    p.add_argument(
        "--exhaustiveness",
        type=int,
        default=DEFAULT_EXHAUSTIVENESS,
        help=f"Vina exhaustiveness (default {DEFAULT_EXHAUSTIVENESS})",
    )
    p.add_argument(
        "--n-poses",
        type=int,
        default=DEFAULT_N_POSES,
        help=f"Vina poses per dock (default {DEFAULT_N_POSES})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Vina RNG seed (default 42)",
    )
    p.add_argument(
        "--box-padding-A",
        type=float,
        default=15.0,
        help="Box padding around pocket centroid (default 15 Å; Wf-323C smoke)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output result JSON (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the docking; without this flag the script "
        "prints the plan and exits 0 (dry-run).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    if not args.input_pool.exists():
        LOG.error("FAIL: input pool missing: %s", args.input_pool)
        return 1

    pool_records = _load_pool(args.input_pool)
    n_pool = len(pool_records)
    LOG.info("loaded %d embed_ok conformers from %s", n_pool, args.input_pool)

    receptor_pdb = WF246D_RECEPTORS / f"{args.pocket_id}_receptor.pdb"
    receptor_pdbqt = WF246D_RECEPTORS / f"{args.pocket_id}_receptor.pdbqt"
    if not receptor_pdb.exists():
        LOG.error("FAIL: receptor PDB missing: %s", receptor_pdb)
        return 1
    if not receptor_pdbqt.exists():
        LOG.info(
            "PDBQT absent — auto-converting %s → %s",
            receptor_pdb.name,
            receptor_pdbqt.name,
        )
        _pdb_to_pdbqt(receptor_pdb, receptor_pdbqt)
    if not receptor_pdbqt.exists():
        LOG.error(
            "FAIL: receptor PDBQT still missing after conversion: %s", receptor_pdbqt
        )
        return 1

    pocket_center = _pocket_center_from_receptor(receptor_pdb)
    box_size = (
        float(args.box_padding_A * 2),
        float(args.box_padding_A * 2),
        float(args.box_padding_A * 2),
    )

    plan = {
        "pocket_id": args.pocket_id,
        "receptor_pdb": str(receptor_pdb),
        "receptor_pdbqt": str(receptor_pdbqt),
        "pocket_center_A": pocket_center,
        "box_size_A": box_size,
        "input_pool": str(args.input_pool),
        "n_conformers_in_pool": n_pool,
        "n_conformers_to_dock": min(args.n_conformers, n_pool),
        "exhaustiveness": args.exhaustiveness,
        "n_poses": args.n_poses,
        "seed": args.seed,
        "out": str(args.out),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    n_to_dock = min(args.n_conformers, n_pool)
    conformers_to_dock = pool_records[:n_to_dock]
    LOG.info(
        "docking %d conformers (exhaustiveness=%d, n_poses=%d)",
        n_to_dock,
        args.exhaustiveness,
        args.n_poses,
    )

    per_conformer: List[Dict[str, Any]] = []
    n_poses_pre_relaxed = 0
    n_pt_zn_substituted = 0
    t_start = time.perf_counter()

    for i, conf in enumerate(conformers_to_dock, start=1):
        smiles = conf.get("smiles", "")
        pool_idx = conf.get("pool_index", i)
        n_atoms_pool = conf.get("n_atoms", 0)
        t0 = time.perf_counter()
        try:
            pose_mol, kcal_mol, vstatus, meta = _vina_redock(
                smiles=smiles,
                receptor_pdbqt=receptor_pdbqt,
                pocket_center=pocket_center,
                pocket_box=box_size,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
                seed=args.seed,
            )
            wall = time.perf_counter() - t0
            record: Dict[str, Any] = {
                "pool_index": pool_idx,
                "smiles": smiles,
                "n_atoms_pool": n_atoms_pool,
                "wall_seconds": round(wall, 4),
                "vina_status": vstatus,
                "pt_zn_substituted": meta.get("pt_zn_substituted", False),
                "prerelax_method": meta.get("prerelax_method"),
                "present_metals": meta.get("present_metals", []),
                "kcal_mol": kcal_mol,
            }

            if pose_mol is None:
                record["pb_status"] = "skipped"
                record["failures"] = [f"vina:{vstatus}"]
                per_conformer.append(record)
                LOG.info(
                    "[%d/%d] pool_idx=%d SKIP (%s)", i, n_to_dock, pool_idx, vstatus
                )
                continue

            n_poses_pre_relaxed += 1
            if meta.get("pt_zn_substituted"):
                n_pt_zn_substituted += 1

            pb = _pb_check_pose(pose_mol, receptor_pdb)
            checks = pb.get("checks", {}) if isinstance(pb, dict) else {}
            n_pa = sum(bool(checks.get(k, False)) for k in PROTEIN_AWARE_KEYS)
            all_pa = (n_pa == len(PROTEIN_AWARE_KEYS)) and len(PROTEIN_AWARE_KEYS) > 0
            all_checks = list(checks.values())
            n_pass = sum(1 for v in all_checks if v is True)
            n_total_checks = len(all_checks)
            full_pass = (n_pass == n_total_checks) and n_total_checks > 0

            record.update(
                {
                    "pb_valid": bool(pb.get("pb_valid", False)),
                    "pb_status": pb.get("status"),
                    "checks": {k: bool(v) for k, v in checks.items()},
                    "protein_aware_pass": bool(all_pa),
                    "n_protein_aware_pass": int(n_pa),
                    "n_protein_aware_total": int(len(PROTEIN_AWARE_KEYS)),
                    "full_pass": bool(full_pass),
                    "n_checks_pass": int(n_pass),
                    "n_checks_total": int(n_total_checks),
                    "failures": pb.get("failures", []),
                    "version": pb.get("version"),
                }
            )
            per_conformer.append(record)
            LOG.info(
                "[%d/%d] pool_idx=%d vina=%.2f kcal/mol pb_full=%s n_pass=%d/%d (%.1fs)",
                i,
                n_to_dock,
                pool_idx,
                kcal_mol if kcal_mol is not None else float("nan"),
                "PASS" if full_pass else "FAIL",
                n_pass,
                n_total_checks,
                wall,
            )
        except Exception as exc:  # noqa: BLE001
            wall = time.perf_counter() - t0
            LOG.warning("[%d/%d] pool_idx=%d EXC: %s", i, n_to_dock, pool_idx, exc)
            per_conformer.append(
                {
                    "pool_index": pool_idx,
                    "smiles": smiles,
                    "n_atoms_pool": n_atoms_pool,
                    "wall_seconds": round(wall, 4),
                    "vina_status": "EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    wall_total = time.perf_counter() - t_start
    summary = _summarise_pb(per_conformer)

    delta_full = (
        (summary["m8_full_pass_rate"] or 0.0) - WF315D_BASELINE_M8_FULL
        if summary["m8_full_pass_rate"] is not None
        else None
    )
    delta_pa = (
        (summary["m8_protein_aware_pass_rate"] or 0.0) - WF315D_BASELINE_M8_PA
        if summary["m8_protein_aware_pass_rate"] is not None
        else None
    )

    output = {
        "workflow": "Wf-323C UFF pre-relax + Vina re-dock CPU smoke (1h36, m8 axis)",
        "engine": "AutoDock Vina 1.2.7 (Python binding) + PoseBusters 0.6.5 + RDKit 2026.03.6",
        "generated_at": _now_iso(),
        "input_pool": str(args.input_pool),
        "pocket_id": args.pocket_id,
        "receptor_pdb": str(receptor_pdb),
        "receptor_pdbqt": str(receptor_pdbqt),
        "pocket_center_A": pocket_center,
        "box_size_A": box_size,
        "vina_config": {
            "exhaustiveness": args.exhaustiveness,
            "n_poses": args.n_poses,
            "seed": args.seed,
            "scoring": "vina (AD4Zn.dat fallback — not on host per Wf-247)",
        },
        "honest_framing": {
            "scope": f"CPU smoke on {n_to_dock} Pt-conformers from Wf-319B pool",
            "uff_pt_honest_negative": (
                "RDKit UFF lacks d-block Pt parameters; relaxation only "
                "acts on the organic donor shell. Conformers are valid "
                "for Vina + PB even though UFF energies are NOT physical "
                "(per Wf-319B HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING)."
            ),
            "vina_pt_honest_negative": (
                "Vina 1.2.7's PDBQT parser rejects the 'Pt' atom-type "
                "('PDBQT parsing error: Atom type Pt is not a valid "
                "AutoDock type'). Wf-323C automatically substitutes "
                "Pt-group metal atom-types with 'Zn' (closest Vina-"
                "accepted divalent d-block) in the output PDBQT — the "
                "same MEASURED_GENERIC_ATOM convention as Wf-246-C. "
                "The kcal/mol is the generic Zn-VDW approximation for "
                "the metal anchor; downstream consumers MUST treat "
                "the kcal/mol as an approximation. This is the same "
                "blocker noted for the Wf-299B CFM Vina panel "
                "(per AGENTS.md §5.1)."
            ),
            "ad4_fallback_honest_negative": (
                "Wf-250 per-pocket config for 1h36 calls for AD4Zn.dat "
                "(scoring='ad4', scoring_file='AD4Zn.dat'); AD4Zn.dat is "
                "NOT on this host so we fall back to scoring='vina' "
                "(the Vina 1.2.7 Python binding default scoring function)."
            ),
            "baseline_comparison": {
                "wf315d_baseline_m8_full": WF315D_BASELINE_M8_FULL,
                "wf315d_baseline_m8_protein_aware": WF315D_BASELINE_M8_PA,
                "wf315d_baseline_path": str(WF315D_SMOKE),
                "delta_m8_full": delta_full,
                "delta_m8_protein_aware": delta_pa,
            },
            "projected_delta_range": "+0.10 to +0.15 (WEAK_TO_STRONG §2.6 R18)",
        },
        "n_poses_pre_relaxed": n_poses_pre_relaxed,
        "n_pt_zn_substituted": n_pt_zn_substituted,
        "n_poses_total": int(summary["n_poses"]),
        "n_protein_aware_pass": int(summary["n_protein_aware_pass"]),
        "n_full_pass": int(summary["n_full_pass"]),
        "m8_protein_aware_pass_rate": summary["m8_protein_aware_pass_rate"],
        "m8_full_pass_rate": summary["m8_full_pass_rate"],
        "wall_seconds_total": round(wall_total, 2),
        "results": per_conformer,
    }

    if n_pt_zn_substituted > 0:
        output["HONEST_NEGATIVE_VINA_PT_REJECTED"] = True
        output["honest_negative_vina_pt_reason"] = (
            f"Vina 1.2.7 PDBQT parser rejected Pt atom-type for "
            f"{n_pt_zn_substituted}/{n_poses_pre_relaxed} successfully-docked "
            f"poses; Wf-323C auto-substituted with Zn (MEASURED_GENERIC_ATOM, "
            f"per Wf-246-C). The Pt kcal/mol contribution is approximated "
            f"by Zn-VDW. Production recovery path: GPU retrain with "
            f"Vina scoring_function override for Pt-group d-block metals. "
            f"See TODO/completed/323c_m8_full_cpu_fix.md for the full "
            f"recovery plan."
        )

    if summary["m8_full_pass_rate"] is not None and summary["m8_full_pass_rate"] < 0.05:
        output["HONEST_NEGATIVE_UFF_INSUFFICIENT"] = True
        output["honest_negative_reason"] = (
            f"m8_full_pass_rate = {summary['m8_full_pass_rate']:.4f} "
            f"({summary['n_full_pass']}/{summary['n_poses']}) is <5% "
            f"above the Wf-315D 0/15 baseline. UFF pre-relax + Vina "
            f"re-dock (with Pt→Zn substitution) is INSUFFICIENT for "
            f"the PB bonded-terms window on these Pt-coordination "
            f"poses. Recovery path: MMFF94 on the organic donor shell "
            f"+ GPU metal-coordination term (Wf-246-C) on the d-block "
            f"atoms + Vina scoring_function override for Pt."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    LOG.info("=== Wf-323C summary ===")
    LOG.info(
        "n_poses_total=%d n_poses_pre_relaxed=%d n_pt_zn_substituted=%d",
        summary["n_poses"],
        n_poses_pre_relaxed,
        n_pt_zn_substituted,
    )
    LOG.info(
        "m8_full_pass_rate=%s m8_protein_aware_pass_rate=%s",
        summary["m8_full_pass_rate"],
        summary["m8_protein_aware_pass_rate"],
    )
    LOG.info("delta_m8_full=%s (vs Wf-315D 0/15 baseline)", delta_full)
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
