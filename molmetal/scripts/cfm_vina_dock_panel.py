#!/usr/bin/env python3
"""Wf-264-Vina — Vina docking panel for CFM generated molecules.

======================================================================
Wf-264-Vina — AutoDock Vina 5-pocket panel for CFM-generated (or any)
SMILES list.
======================================================================

For each SMILES in the input CSV, the panel:

  1. Parses the SMILES via RDKit.
  2. Generates a 3D conformer via RDKit (ETKDGv3 + MMFF94).
  3. Converts the conformer to PDBQT via RDKit + Meeko.
  4. Runs AutoDock Vina (project wrapper ``molmetal/bin/vina`` by
     default) against each of the five canonical Wf-179 m15 pockets:
     **1h36, 3eml, 3ny8, 4aaw, 4rlu**.
  5. Captures the **best** docking score (kcal/mol) per pocket.

If the Vina binary is missing, every cell records
``status="STUB_FALLBACK"`` and ``best_kcal_mol=None`` — no score is
ever fabricated.

Inputs
------
* ``--smiles-csv PATH``  CSV with at minimum a ``smiles`` column.
* ``--pocket-pdb-dir PATH``  Directory containing one subdirectory per
  pocket (each with ``receptor.pdbqt``). Default layout matches
  ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``.

Outputs
-------
* ``--out-json PATH``  JSON with shape::

    {
      "panel_version": "wf264-vina/v1",
      "metadata": {...},
      "n_input": int,
      "n_parsed": int,
      "n_pdbqt_ok": int,
      "n_docked": int,
      "per_molecule": [
        {
          "smiles": str,
          "parse_ok": bool,
          "pdbqt_ok": bool,
          "per_pocket": {
            "1h36": {"best_kcal_mol": float|null,
                     "n_poses": int,
                     "wall_time_s": float,
                     "status": str, "note": str},
            ...
          },
          "best_pocket": str|null,
          "best_kcal_mol": float|null,
          "mean_kcal_mol": float|null
        },
        ...
      ],
      "aggregate": {
        "per_pocket": {
          "1h36": {"n": int, "n_real_vina": int,
                   "n_stub_fallback": int,
                   "mean": float|null, "min": float|null,
                   "max": float|null},
          ...
        },
        "best_pocket_overall": str|null,
        "best_score_overall": float|null,
        "mean_score_overall": float|null
      }
    }

Usage
-----
::

    .venv/bin/python molmetal/scripts/cfm_vina_dock_panel.py \\
        --smiles-csv /tmp/cfm_generated.csv \\
        --pocket-pdb-dir molmetal/reports/wf30_pocket_prep \\
        --out-json molmetal/reports/wf264_vina/panel.json \\
        --n-candidates 5

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* kcal/mol scores are real Vina 1.2.7 output — no stub floor, no hash
  placeholder.
* When the Vina binary is missing or the receptor PDBQT is not
  staged, every (molecule, pocket) cell records
  ``status="STUB_FALLBACK"`` and ``best_kcal_mol=None``.
* Per-pocket ``status="STUB_FALLBACK"`` is the diagnostic — it tells
  downstream consumers the score should not be trusted.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PANEL_VERSION = "wf264-vina/v1"

# ---------------------------------------------------------------------------
# Constants — 5 canonical Wf-179 m15 pockets
# ---------------------------------------------------------------------------
#: Canonical 5-pocket Wf-179 m15 cohort (per Wf-179 + Wf-30 staging).
DEFAULT_POCKETS: Tuple[str, ...] = (
    "1h36", "3eml", "3ny8", "4aaw", "4rlu",
)

#: Default Vina CLI binary location (project wrapper).
DEFAULT_VINA_BIN: str = str(
    Path(__file__).resolve().parent.parent / "bin" / "vina"
)

#: Wf-338-V2 — Engine selector constants. Wf-264 sub-panel + recovery:
#: when the canonical AutoDock Vina binary is missing or the panel
#: MEASURED 0/100 docked ok on 1h36 (meeko 0.8.0 metal-atom limit per
#: Wf-264 §2), the panel can transparently fall back to either
#: **Smina** (Scoring functions for **M**olecular **i**nteraction
#: **a**nalysis, Forli lab; same AutoDock Vina 1.2.x CLI surface but
#: with extended scoring functions including a custom MetalIon
#: term that handles Pt/Ru/Ir/Au coordination shells) or
#: **QuickVina 2** (Handoko et al. 2018, ~2x faster than Vina 1.1.2
#: but same scoring function — no metal-specific term, but useful
#: for higher-throughput scans when the covalent mechanism is not
#: the bottleneck).  All three engines share the same CLI flag set
#: (``--score_only`` / ``--local_only`` / ``--out`` / ``--log`` /
#: ``--seed`` / ``--exhaustiveness`` / ``--num_modes``) so the
#: dispatch logic in :func:`_invoke_vina_engine` is uniform.
DEFAULT_ENGINE: str = "vina"
SUPPORTED_ENGINES: Tuple[str, ...] = ("vina", "smina", "quickvina2")
#: Default Smina CLI binary (well-known name on PATH).
DEFAULT_SMINA_BIN: str = "smina"
#: Default QuickVina 2 CLI binary (well-known name on PATH).
DEFAULT_QUICKVINA2_BIN: str = "qvina2"

#: Default exhaustiveness (AutoDock Vina CLI flag).
DEFAULT_EXHAUSTIVENESS: int = 8

#: Default n_poses (AutoDock Vina CLI flag --num_modes).
DEFAULT_N_POSES: int = 5

#: Default box size (Å) — cubed around pocket centre.
DEFAULT_BOX_SIZE: float = 20.0

#: Default random seed (AutoDock Vina CLI flag --seed).
DEFAULT_SEED: int = 0

#: Default per-pocket docking timeout (seconds).
DEFAULT_DOCK_TIMEOUT: int = 120

LOG = logging.getLogger("wf264_vina")


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STATUS_REAL_VINA = "REAL_VINA"
STATUS_STUB_FALLBACK = "STUB_FALLBACK"
STATUS_PARSE_FAIL = "PARSE_FAIL"
STATUS_PDBQT_FAIL = "PDBQT_FAIL"
STATUS_DOCK_FAIL = "DOCK_FAIL"
STATUS_NO_RECEPTOR = "NO_RECEPTOR"
STATUS_TIMEOUT = "TIMEOUT"
# Wf-336-X4: meeko alternative path — ligand prep succeeded but
# docking engine (Vina CLI or in-process) was unavailable; the
# generated PDBQT is preserved on disk for downstream consumption.
STATUS_MEEKO_PREP_ONLY = "MEEKO_PREP_ONLY"


# ---------------------------------------------------------------------------
# Wf-336-X4: meeko availability probe (alternative integration)
# ---------------------------------------------------------------------------
# Per Wf-329-W7 §3.4 + Wf-323G §2.5: meeko 0.8.0 is the canonical
# Python-only fallback for SMILES → PDBQT ligand preparation when
# Vina CLI is unreachable. The integration is already wired into
# ``_smiles_to_pdbqt_string`` (MoleculePreparation + PDBQTWriterLegacy);
# this probe surfaces the runtime availability so the panel JSON can
# emit an honest ``meeko_available: true/false`` field without each
# downstream consumer re-importing meeko.
# ---------------------------------------------------------------------------
def _probe_meeko_alternative() -> Dict[str, Any]:
    """Probe the meeko Python wrapper availability.

    Returns a dict with keys:
      ``available`` (bool) — meeko importable in current interpreter
      ``version`` (str) — meeko __version__ when available, else ""
      ``moleculeprep_class`` (str) — class name when available
      ``pdbqt_writer`` (str) — writer class name when available
      ``import_error`` (str) — repr of the import error when not avail

    The probe is intentionally cheap (single import + attribute access)
    so it can be called from ``build_panel`` without measurable
    overhead.

    Wf-336-X4: this probe was added as the meeko alternative
    integration; the actual SMILES → PDBQT prep is in
    ``_smiles_to_pdbqt_string`` (which lazy-imports meeko via the same
    try/except pattern).
    """
    out: Dict[str, Any] = {
        "available": False,
        "version": "",
        "moleculeprep_class": "",
        "pdbqt_writer": "",
        "import_error": "",
    }
    try:
        import meeko  # type: ignore  # noqa: WPS433
        out["available"] = True
        out["version"] = str(getattr(meeko, "__version__", "unknown"))
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore  # noqa: WPS433
            out["moleculeprep_class"] = str(MoleculePreparation.__name__)
            out["pdbqt_writer"] = str(PDBQTWriterLegacy.__name__)
        except Exception as exc:  # noqa: BLE001
            # meeko present but specific classes missing — partial OK.
            out["import_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001
        out["import_error"] = repr(exc)
    return out


# Module-level cache: probes are expensive only when run many times;
# cache once at import so downstream consumers see a stable snapshot.
_MEEKO_PROBE: Optional[Dict[str, Any]] = None


def get_meeko_probe(force_refresh: bool = False) -> Dict[str, Any]:
    """Return the cached meeko probe; refresh when ``force_refresh``."""
    global _MEEKO_PROBE
    if _MEEKO_PROBE is None or force_refresh:
        _MEEKO_PROBE = _probe_meeko_alternative()
    return _MEEKO_PROBE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PocketCell:
    """One (molecule, pocket) docking cell."""
    pocket: str
    best_kcal_mol: Optional[float] = None
    n_poses: int = 0
    wall_time_s: float = 0.0
    status: str = STATUS_STUB_FALLBACK
    note: str = ""
    # Wf-329-W3: F2(a) MetalLigandExchange covalent scoring fields.
    f2a_kcal_mol: Optional[float] = None
    f2a_n_bonds: int = 0
    f2a_modifier_total: float = 0.0
    f2a_matched_patterns: str = ""
    f2a_recovery_status: str = "NO_SCORE"
    # Wf-333-V3 P2: per-bond F2(a) selector.
    f2a_per_bond_kcal_mol: Optional[float] = None
    f2a_per_bond_n_bonds: int = 0
    f2a_per_bond_modifier_total: float = 0.0
    f2a_per_bond_recovery_status: str = "NO_SCORE"
    # Wf-333-V3 P3: distance sub-score wire-in (Wf-246-C metal-coord).
    metal_coord_score: Optional[float] = None  # composite in [0, 1]
    metal_coord_distance_score: Optional[float] = None
    metal_coord_hsab_score: Optional[float] = None
    metal_coord_geometry_score: Optional[float] = None
    metal_coord_adjustment: Optional[float] = None  # kcal/mol
    metal_coord_donor_distances: str = ""  # JSON {cls: Å}
    # Wf-333-V3 combined V3 lift (Vina + per-bond F2(a) + metal-coord).
    v3_combined_kcal_mol: Optional[float] = None
    v3_combined_lift_kcal_mol: Optional[float] = None
    v3_combined_recovery_status: str = "NO_SCORE"
    # Wf-333-V2: Wf-246-C metal-coordination composite fields.
    # ``metal_coord_kcal_mol`` is the F2(a)-adjusted kcal/mol
    # + Wf-246-C metal-coordination adjustment (combined additive).
    # When Wf-246-C is disabled or the SMILES is organic, this is
    # the same as ``f2a_kcal_mol`` (or ``best_kcal_mol`` when F2(a)
    # is also disabled).
    metal_coord_kcal_mol: Optional[float] = None
    metal_coord_adjustment: float = 0.0
    metal_coord_total: Optional[float] = None
    metal_coord_geometry_score: Optional[float] = None
    metal_coord_hsab_score: Optional[float] = None
    metal_coord_distance_score: Optional[float] = None
    metal_coord_recovery_status: str = "NO_SCORE"


@dataclass
class MoleculeRecord:
    """Per-molecule aggregation across all pockets."""
    smiles: str
    parse_ok: bool = False
    pdbqt_ok: bool = False
    per_pocket: Dict[str, PocketCell] = field(default_factory=dict)

    @property
    def best_pocket(self) -> Optional[str]:
        scored = [
            (p, c.best_kcal_mol) for p, c in self.per_pocket.items()
            if c.best_kcal_mol is not None
        ]
        if not scored:
            return None
        # Lower kcal/mol = better binding.
        return min(scored, key=lambda t: t[1])[0]

    @property
    def best_kcal_mol(self) -> Optional[float]:
        scored = [
            c.best_kcal_mol for c in self.per_pocket.values()
            if c.best_kcal_mol is not None
        ]
        return min(scored) if scored else None

    @property
    def mean_kcal_mol(self) -> Optional[float]:
        scored = [
            c.best_kcal_mol for c in self.per_pocket.values()
            if c.best_kcal_mol is not None
        ]
        if not scored:
            return None
        return float(sum(scored) / len(scored))

    # ------------------------------------------------------------------
    # Wf-329-W3 F2(a) MetalLigandExchange aggregate properties.
    # ------------------------------------------------------------------
    @property
    def f2a_best_kcal_mol(self) -> Optional[float]:
        """Best F2(a)-adjusted kcal/mol across pockets (None when
        F2(a) was disabled or no pocket produced a score).
        """
        scored = [
            c.f2a_kcal_mol for c in self.per_pocket.values()
            if c.f2a_kcal_mol is not None
        ]
        return min(scored) if scored else None

    @property
    def f2a_mean_kcal_mol(self) -> Optional[float]:
        scored = [
            c.f2a_kcal_mol for c in self.per_pocket.values()
            if c.f2a_kcal_mol is not None
        ]
        if not scored:
            return None
        return float(sum(scored) / len(scored))

    @property
    def f2a_n_bonds_total(self) -> int:
        """Sum of F2(a) covalent bonds detected across pockets (1st
        pocket's count is used — F2(a) detection is SMILES-based and
        is constant across pockets).
        """
        for c in self.per_pocket.values():
            if c.f2a_n_bonds > 0:
                return int(c.f2a_n_bonds)
        return 0

    # ------------------------------------------------------------------
    # Wf-333-V2 Wf-246-C metal-coordination aggregate properties.
    # ------------------------------------------------------------------
    @property
    def metal_coord_best_kcal_mol(self) -> Optional[float]:
        """Best Wf-246-C-adjusted kcal/mol across pockets (None when
        Wf-246-C was disabled or no pocket produced a score).
        """
        scored = [
            c.metal_coord_kcal_mol for c in self.per_pocket.values()
            if c.metal_coord_kcal_mol is not None
        ]
        return min(scored) if scored else None

    @property
    def metal_coord_mean_kcal_mol(self) -> Optional[float]:
        scored = [
            c.metal_coord_kcal_mol for c in self.per_pocket.values()
            if c.metal_coord_kcal_mol is not None
        ]
        if not scored:
            return None
        return float(sum(scored) / len(scored))

    @property
    def metal_coord_total_score(self) -> Optional[float]:
        """Wf-246-C composite total (constant across pockets when
        the term fires — it is SMILES-based). Returns ``None`` when
        the SMILES is organic or Wf-246-C was disabled.
        """
        for c in self.per_pocket.values():
            if c.metal_coord_total is not None:
                return float(c.metal_coord_total)
        return None

    # ------------------------------------------------------------------
    # Wf-333-V3 V3 combined lift aggregate properties.
    # ------------------------------------------------------------------
    @property
    def v3_combined_best_kcal_mol(self) -> Optional[float]:
        """Best V3-combined kcal/mol across pockets (None when V3
        did not fire for any pocket)."""
        scored = [
            c.v3_combined_kcal_mol for c in self.per_pocket.values()
            if c.v3_combined_kcal_mol is not None
        ]
        return min(scored) if scored else None

    @property
    def v3_combined_mean_kcal_mol(self) -> Optional[float]:
        scored = [
            c.v3_combined_kcal_mol for c in self.per_pocket.values()
            if c.v3_combined_kcal_mol is not None
        ]
        if not scored:
            return None
        return float(sum(scored) / len(scored))

    @property
    def v3_combined_lift_total(self) -> Optional[float]:
        """Total V3 combined lift = v3_combined_kcal_mol - best_kcal_mol
        (using the best pocket's lift)."""
        best_v3 = self.v3_combined_best_kcal_mol
        if best_v3 is None or self.best_kcal_mol is None:
            return None
        return float(best_v3) - float(self.best_kcal_mol)


# ---------------------------------------------------------------------------
# Helpers — CLI / IO
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def read_smiles_csv(path: Path) -> List[str]:
    """Read a CSV with a ``smiles`` column.

    Accepts both quoted and unquoted SMILES, strips whitespace, drops
    empty rows.  Raises ``ValueError`` if no ``smiles`` column is found.
    """
    smiles_list: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header: Optional[List[str]] = None
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if header is None:
                lowered = [c.strip().lower() for c in row]
                if "smiles" not in lowered:
                    raise ValueError(
                        f"CSV at {path} has no 'smiles' column "
                        f"(header={row!r})"
                    )
                header = row
                col_idx = lowered.index("smiles")
                continue
            if col_idx >= len(row):
                continue
            s = row[col_idx].strip()
            if s:
                smiles_list.append(s)
    return smiles_list


def _resolve_pocket_receptor(
    pocket_pdb_dir: Path, pocket_id: str
) -> Optional[Path]:
    """Locate ``receptor.pdbqt`` under ``<pocket_pdb_dir>/<pocket_id>/``.

    Returns ``None`` if the receptor file is not staged.  Accepts both
    ``receptor.pdbqt`` (Wf-30 standard) and ``<pocket_id>.pdbqt``
    (legacy fallback) at the top level of the per-pocket dir.
    """
    pocket_dir = pocket_pdb_dir / pocket_id
    if not pocket_dir.is_dir():
        return None
    for candidate in ("receptor.pdbqt", f"{pocket_id}.pdbqt"):
        p = pocket_dir / candidate
        if p.is_file():
            return p
    return None


def _pocket_center(receptor: Path) -> Tuple[float, float, float]:
    """Compute the geometric centre of receptor ATOM/HETATM lines.

    Falls back to (0, 0, 0) if the receptor has no parseable atoms.
    """
    coords: List[Tuple[float, float, float]] = []
    try:
        with receptor.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    coords.append((
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ))
                except (ValueError, IndexError):
                    continue
    except OSError:
        return (0.0, 0.0, 0.0)
    if not coords:
        return (0.0, 0.0, 0.0)
    n = len(coords)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Helpers — Pt-stripping (Wf-301B)
# ---------------------------------------------------------------------------
PT_ATOMIC_NUM: int = 78

#: Atom symbols that should NEVER be parameterised by UFF/MMFF (the
#: force-field Meeko depends on).  When a precious-metal coordination
#: complex still has these atoms after Pt-stripping (e.g. exotic Pt
#: + PPh3 / S / Se / I shells), we additionally drop them so UFF has
#: a chance to converge.
UFF_UNSUPPORTED_ATOMIC_NUMS: frozenset[int] = frozenset({
    78,  # Pt
    46,  # Pd
    77,  # Ir
    44,  # Ru
    79,  # Au
    45,  # Rh
})


def _strip_pt_and_largest_fragment(
    mol: "Chem.Mol",
) -> Tuple[Optional["Chem.Mol"], str]:
    """Return the largest organic fragment after stripping Pt + coordination shell.

    Strategy
    --------
    1. Find every Pt atom (or any precious-metal atom in
       :data:`UFF_UNSUPPORTED_ATOMIC_NUMS`).
    2. Mark those atoms + every atom directly bonded to them for
       removal — this drops the entire ``[Pt](lig1)(lig2)(lig3)(lig4)``
       coordination shell from cisplatin-/oxaliplatin-like complexes.
    3. Build a copy of the molecule with those atoms removed.
    4. Use :func:`rdkit.Chem.GetMolFrags` (sanitized) to split the
       surviving graph into connected fragments — counterions
       (``.Cl``, ``.[H]``, etc.) and any leftover scaffold fall apart
       cleanly.
    5. Pick the **largest** fragment (heavy-atom count) — this is the
       organic pharmacophore we want Vina to dock.  This is the
       "largest organic fragment" the Wf-301B spec asks for.

    Returns
    -------
    (largest_mol, note)
        ``largest_mol`` is ``None`` if the stripping produced no
        fragment with at least one heavy atom — caller should map
        that to ``PDBQT_FAIL``.  ``note`` is a human-readable
        diagnostic appended to the cell.
    """
    try:
        from rdkit import Chem  # type: ignore  # noqa: WPS433
    except Exception:  # pragma: no cover
        return None, "rdkit import failed"

    drop_indices: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in UFF_UNSUPPORTED_ATOMIC_NUMS:
            drop_indices.add(atom.GetIdx())
            for neighbour in atom.GetNeighbors():
                drop_indices.add(neighbour.GetIdx())

    if not drop_indices:
        # No precious metal present — fall back to "largest fragment
        # only" so counterions / salts are still dropped.
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None, "no fragments"
        largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return largest, "no_pt_strip_largest_only"

    rwmol = Chem.RWMol(mol)
    # RemoveAtom requires descending index order to keep remapping valid.
    for idx in sorted(drop_indices, reverse=True):
        if idx < rwmol.GetNumAtoms():
            rwmol.RemoveAtom(idx)
    stripped = rwmol.GetMol()
    # Wf-301B: Pt-strip can leave dangling aromaticity flags when the
    # coordination shell carried an aromatic system (e.g. N-ligated
    # pyridine / imine).  RDKit's full sanitize() then fails on
    # kekulization.  Fall back to a soft clean-up path:
    #   1. clear all atom aromatic flags
    #   2. clear all bond aromatic flags + bond directions
    #   3. SanitizeMol with ``sanitizeOps=CHEM.SANITIZE_ALL ^
    #      CHEM.SANITIZE_KEKULIZE`` so 3D embedding still works.
    # Only if that fails do we abandon the molecule.
    sanitized_ok = False
    sanitize_exc: Optional[Exception] = None
    try:
        Chem.SanitizeMol(stripped)
        sanitized_ok = True
    except Exception as exc:  # noqa: BLE001
        sanitize_exc = exc
    if not sanitized_ok:
        try:
            for atom in stripped.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in stripped.GetBonds():
                bond.SetIsAromatic(False)
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    bond.SetBondType(Chem.BondType.SINGLE)
            Chem.SanitizeMol(
                stripped,
                sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE,
            )
            sanitized_ok = True
        except Exception as exc:  # noqa: BLE001
            return None, (
                f"sanitize after Pt-strip failed: {sanitize_exc!r}; "
                f"fallback also failed: {exc!r}"
            )

    frags = Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=True)
    if not frags:
        return None, "no fragments after Pt-strip"
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if largest.GetNumHeavyAtoms() == 0:
        return None, "largest fragment has 0 heavy atoms after Pt-strip"
    return largest, (
        f"pt_strip dropped {len(drop_indices)} atoms "
        f"(Pt-coord shell + Pt), kept largest organic fragment"
    )


# ---------------------------------------------------------------------------
# Helpers — F2(a) MetalLigandExchange covalent scoring (Wf-329-W3)
# ---------------------------------------------------------------------------
# Wf-329-W3 ships the F2(a) MetalLigandExchange SMARTS rule + a
# covalent scoring modifier.  When ``--use-f2a-metal-ligand`` is
# set (default True) the Vina panel:
#   1. Runs Vina as usual.
#   2. After Vina returns ``best_kcal_mol``, the F2(a) scorer
#      detects metal-N7 covalent-capable centres in the SMILES
#      (Pt, Pd, Ru, Au; bare + leaving-group flavours).
#   3. Applies the covalent scoring modifier (default -1.5
#      kcal/mol per detected bond; PROJECTED conservative midpoint
#      per Wf-329-W3 task brief).
#   4. Records both the Vina kcal/mol (``best_kcal_mol``) and the
#      F2(a)-adjusted kcal/mol (``f2a_kcal_mol``) on each cell so
#      downstream consumers can audit the lift.
#
# When ``--use-f2a-metal-ligand`` is False the F2(a) scorer is
# skipped and the cell is identical to the pre-F2(a) Wf-264-Vina
# baseline (backward-compat test seam).
#
# Wf-333-V3 P2: ``--use-per-bond-f2a`` (default True) replaces the
# per-centre count with per-metal-donor-arm-bond counting.  For
# cisplatin this lifts the F2(a) modifier from -1.5 (per-centre)
# to -3.0 (per-bond: 2 Pt-N) — an additional -1.5 kcal/mol
# PROJECTED lift per Wf-331-R1 §6.3.
#
# Wf-333-V3 P3: ``--use-distance-score`` (default True) wires the
# Wf-246-C distance sub-score on top.  We compute the closest
# pocket-donor-atom distance to the metal centre (3-D coords from
# the RDKit ETKDGv3 conformer) and pass via ``metal_ligand_distances``
# to ``MetalCoordinationTerm.score()``.  PROJECTED lift: -0.5 to
# -1.0 kcal/mol per Wf-331-R1 §6.4.
# ---------------------------------------------------------------------------
_F2A_SCORER: Optional["F2AMetalLigandScorer"] = None
_F2A_IMPORT_ERROR: Optional[str] = None


def _get_f2a_scorer(
    modifier_kcal_mol: Optional[float] = None,
    use_per_bond: bool = False,
) -> Tuple[Optional["F2AMetalLigandScorer"], Optional[str]]:
    """Lazy-load and cache the F2(a) MetalLigandScorer.

    Returns ``(scorer, error_msg)``. ``scorer`` is ``None`` when
    the F2(a) module is not importable; ``error_msg`` carries the
    import error for the audit log.

    Wf-333-V3 P2: ``use_per_bond=True`` activates the per-metal-
    donor-arm-bond counting selector.  Cached singleton is keyed by
    the per-bond flag so per-centre and per-bond scorers can coexist.
    """
    global _F2A_SCORER, _F2A_IMPORT_ERROR
    cache_key = (modifier_kcal_mol, use_per_bond)
    if (
        _F2A_SCORER is not None
        and getattr(_F2A_SCORER, "use_per_bond", False) == use_per_bond
        and modifier_kcal_mol is None
    ):
        return _F2A_SCORER, None
    try:
        from molmetal.scripts.wf329w3_f2a_metal_ligand import (  # type: ignore
            F2AMetalLigandScorer,
        )
        if modifier_kcal_mol is not None:
            return (
                F2AMetalLigandScorer(
                    modifier_kcal_mol=modifier_kcal_mol,
                    use_per_bond=use_per_bond,
                ),
                None,
            )
        _F2A_SCORER = F2AMetalLigandScorer(use_per_bond=use_per_bond)
        _F2A_IMPORT_ERROR = None
        return _F2A_SCORER, None
    except Exception as exc:  # noqa: BLE001
        _F2A_IMPORT_ERROR = repr(exc)
        return None, _F2A_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Helpers — Distance sub-score wire-in (Wf-333-V3 P3 + Wf-246-C)
# ---------------------------------------------------------------------------
# We compute the closest distance from the metal centre to each
# pocket donor class (N/O/S/P).  The metal centre 3-D position is
# taken from the RDKit ETKDGv3 conformer of the original SMILES
# (pre-Pt-strip); pocket donor atoms are read from the receptor
# .pdbqt.  When the metal centre is not in the SMILES (organic
# ligand) or the conformer embed fails, we return ``None``.
# ---------------------------------------------------------------------------
_METAL_COORD_TERM: Optional["MetalCoordinationTerm"] = None
_METAL_COORD_IMPORT_ERROR: Optional[str] = None


def _get_metal_coord_term() -> Tuple[Optional["MetalCoordinationTerm"], Optional[str]]:
    """Lazy-load and cache the Wf-246-C MetalCoordinationTerm."""
    global _METAL_COORD_TERM, _METAL_COORD_IMPORT_ERROR
    if _METAL_COORD_TERM is not None:
        return _METAL_COORD_TERM, None
    try:
        from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (  # type: ignore
            MetalCoordinationTerm,
        )
        _METAL_COORD_TERM = MetalCoordinationTerm()
        _METAL_COORD_IMPORT_ERROR = None
        return _METAL_COORD_TERM, None
    except Exception as exc:  # noqa: BLE001
        _METAL_COORD_IMPORT_ERROR = repr(exc)
        return None, _METAL_COORD_IMPORT_ERROR


def _read_pocket_donor_coords(receptor: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    """Read pocket atom coords grouped by donor class (N/O/S/P).

    Returns ``{"N": [(x,y,z), ...], "O": [...], "S": [...], "P": [...]}``.
    Hydrogen atoms and non-donor heavy atoms are skipped.
    """
    coords: Dict[str, List[Tuple[float, float, float]]] = {
        "N": [], "O": [], "S": [], "P": [],
    }
    try:
        with receptor.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                # PDB atom-name column 77-78 (1-based); element-symbol
                # cols 77-78 in PDBQT.  Use a robust parse: strip whitespace,
                # pick the element symbol from the atom-name column.
                # PDBQT atom-type field (cols 77-78 in fixed-width) is
                # the element for receptor atoms.
                if len(line) < 78:
                    continue
                elem_field = line[76:78].strip()
                # Element may have a charge suffix; strip digits/+/-.
                elem = "".join(c for c in elem_field if c.isalpha())[:2]
                if elem not in ("N", "O", "S", "P"):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                coords[elem].append((x, y, z))
    except OSError:
        pass
    return coords


def _closest_donor_distances(
    metal_pos: Tuple[float, float, float],
    donor_coords: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, float]:
    """Return ``{cls: closest_distance_angstrom}`` per donor class.

    Classes with zero donor atoms are omitted from the result.
    """
    import math as _math
    out: Dict[str, float] = {}
    for cls, atoms in donor_coords.items():
        if not atoms:
            continue
        best = float("inf")
        for (x, y, z) in atoms:
            dx = x - metal_pos[0]
            dy = y - metal_pos[1]
            dz = z - metal_pos[2]
            d = _math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < best:
                best = d
        if best != float("inf"):
            out[cls] = float(best)
    return out


def _metal_centre_coords_from_smiles(smiles: str) -> Optional[Tuple[float, float, float]]:
    """Return the 3-D position of the first metal centre (Pt/Pd/Ir/Ru/Au/Os/Rh/Re)
    from an RDKit ETKDGv3 conformer, or ``None`` when the SMILES has no
    metal centre or the embed fails.

    The position is the metal atom's (x, y, z) from the conformer after
    MMFF94 optimisation.  Hydrogen atoms are NOT added by us — RDKit
    ``Chem.MolFromSmiles`` already returns the heavy-atom graph; we
    embed a 3-D conformer via ETKDGv3 and read the metal's
    ``GetConformer().GetAtomPosition(idx)``.

    Note: per Wf-319B, RDKit UFF lacks d-block Pt parameters; this is
    FINE for distance scoring because we ONLY need the metal centre
    position (the metal position is independent of force-field params).
    The ligand-internal atoms may not be minimised properly, but the
    metal centre coordinates are still valid (we don't minimise them).
    """
    try:
        from rdkit import Chem  # type: ignore  # noqa: WPS433
        from rdkit.Chem import AllChem  # type: ignore  # noqa: WPS433
    except Exception:
        return None

    METAL_ANUMS = {44, 45, 46, 75, 76, 77, 78, 79}
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None

    metal_idx = None
    for atom in mol.GetAtoms():
        if int(atom.GetAtomicNum()) in METAL_ANUMS:
            metal_idx = int(atom.GetIdx())
            break
    if metal_idx is None:
        return None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    code = AllChem.EmbedMolecule(mol_h, params)
    if int(code) != 0:
        params_v2 = AllChem.ETKDGv2()
        params_v2.randomSeed = 0
        params_v2.useRandomCoords = True
        code = AllChem.EmbedMolecule(mol_h, params_v2)
        if int(code) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    try:
        conf = mol_h.GetConformer()
        pt = conf.GetAtomPosition(metal_idx)
        return (float(pt.x), float(pt.y), float(pt.z))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers — Wf-246-C metal-coordination composite (Wf-333-V2)
# ---------------------------------------------------------------------------
# Wf-246-C ships the MetalCoordinationTerm (coordination geometry +
# HSAB donor matching + M-ligand distance).  Per Wf-331-R1 §4.2,
# the sign convention was INVERTED — Wf-333-V2 P0 fixed it (a
# perfect composite yields -1.0 kcal/mol BONUS; zero yields +1.0
# kcal/mol PENALTY).  When ``--use-wf246c-metal-coord`` is set
# (default True), the Vina panel:
#   1. After Vina returns ``best_kcal_mol`` (and after F2(a) when
#      enabled), the Wf-246-C scorer runs against the SMILES +
#      pocket donor fingerprint (when a receptor is staged) to
#      compute the composite (coordination geometry + HSAB +
#      distance weighted sum).
#   2. The composite is converted to kcal/mol via
#      ``MetalCoordinationTerm.kcal_adjustment(score)`` which has
#      the Wf-333-V2 sign-corrected formula
#      ``(1 - 2 * composite) * kcal_per_unit``.
#   3. The adjustment is added to the F2(a)-adjusted kcal/mol
#      (or to the Vina kcal/mol when F2(a) is disabled), giving
#      ``metal_coord_kcal_mol`` = combined adjusted kcal/mol.
#   4. Both the per-cell composite (geometry / HSAB / distance /
#      total) and the kcal/mol adjustment are recorded for audit.
#
# When ``--use-wf246c-metal-coord`` is False the Wf-246-C term is
# skipped and the cell is identical to the pre-Wf-333-V2 Wf-329-W3
# F2(a)-only baseline (backward-compat test seam).
#
# For organic-only ligands (no metal centre), the Wf-246-C composite
# is None and the kcal/mol adjustment is exactly 0.0 (backward-compat
# with all earlier Wf-30 / Wf-121 / Wf-264 sweeps).
# ---------------------------------------------------------------------------
_WF246C_TERM: Optional["MetalCoordinationTerm"] = None
_WF246C_IMPORT_ERROR: Optional[str] = None


def _get_wf246c_term(
    kcal_per_unit: Optional[float] = None,
) -> Tuple[Optional["MetalCoordinationTerm"], Optional[str]]:
    """Lazy-load and cache the Wf-246-C :class:`MetalCoordinationTerm`.

    Returns ``(term, error_msg)``. ``term`` is ``None`` when the
    metal_coord_term module is not importable; ``error_msg`` carries
    the import error for the audit log.
    """
    global _WF246C_TERM, _WF246C_IMPORT_ERROR
    if _WF246C_TERM is not None and kcal_per_unit is None:
        return _WF246C_TERM, None
    try:
        from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (  # type: ignore  # noqa: WPS433
            MetalCoordinationTerm,
        )
        if kcal_per_unit is not None:
            return MetalCoordinationTerm(kcal_per_unit=float(kcal_per_unit)), None
        _WF246C_TERM = MetalCoordinationTerm()
        _WF246C_IMPORT_ERROR = None
        return _WF246C_TERM, None
    except Exception as exc:  # noqa: BLE001
        _WF246C_IMPORT_ERROR = repr(exc)
        return None, _WF246C_IMPORT_ERROR


def _compute_wf246c_adjustment_for_smiles(
    smiles: str,
    pocket_donor_counts: Optional[Dict[str, int]] = None,
    kcal_per_unit: float = 1.0,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Compute the Wf-246-C metal-coordination kcal/mol adjustment.

    Returns ``(adjustment_kcal_mol, diagnostics_dict)``. When the
    SMILES has no metal centre (organic ligand) or the term fails
    to fire, ``adjustment_kcal_mol`` is ``None`` (no change to the
    Vina score) and ``diagnostics_dict`` is also ``None``.

    ``pocket_donor_counts`` is the optional pocket-side HSAB donor
    fingerprint ``{cls: count}`` (N/O/S/P). When ``None``, the
    composite falls back to ligand-internal donor counting
    (HSAB-only mode).
    """
    term, _err = _get_wf246c_term(kcal_per_unit=kcal_per_unit)
    if term is None:
        return None, None
    try:
        score = term.score(smiles, pocket_donor_counts=pocket_donor_counts)
    except Exception as exc:  # noqa: BLE001
        return None, {"error": repr(exc)}
    if score.total is None:
        return None, None
    adjustment = term.kcal_adjustment(score)
    diagnostics = {
        "metal_coord_total": float(score.total),
        "metal_coord_kcal_adjustment": float(adjustment),
        "geometry_score": score.geometry_score,
        "hsab_score": score.hsab_score,
        "distance_score": score.distance_score,
        "status": "MEASURED_METAL_COORD_TERM",
    }
    return float(adjustment), diagnostics


def _smiles_to_pdbqt_string(
    smiles: str, seed: int = 0, pt_strip: bool = False
) -> Tuple[Optional[str], str, str]:
    """Convert a SMILES string to a Meeko-emitted PDBQT string.

    Returns ``(pdbqt_string, status, note)``.  ``pdbqt_string`` is
    ``None`` on failure (PARSE_FAIL or PDBQT_FAIL).  Imports RDKit +
    Meeko lazily so the module-level import path is cheap.

    When ``pt_strip`` is True, the precious-metal coordination shell
    (``[Pt](lig1)(lig2)...`` and bonded atoms) is dropped and the
    largest remaining organic fragment is used for 3D conformer
    generation.  This is the Wf-301B fix — RDKit UFF cannot
    parameterise Pt+2 so Meeko write_string returns no OK setup.
    """
    try:
        from rdkit import Chem  # type: ignore  # noqa: WPS433
        from rdkit.Chem import AllChem  # type: ignore  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - rdkit mandatory
        return None, STATUS_DOCK_FAIL, f"rdkit import failed: {exc!r}"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, STATUS_PARSE_FAIL, "RDKit parse failed"

    pt_strip_note: str = ""
    if pt_strip:
        stripped, pt_strip_note = _strip_pt_and_largest_fragment(mol)
        if stripped is None or stripped.GetNumHeavyAtoms() == 0:
            return None, STATUS_PDBQT_FAIL, (
                f"pt-strip produced no dockable fragment: {pt_strip_note}"
            )
        mol = stripped

    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        embed_code = AllChem.EmbedMolecule(mol_h, params)
        if int(embed_code) != 0:
            # Fallback: try ETKDGv2 with random coords.
            params_v2 = AllChem.ETKDGv2()
            params_v2.randomSeed = int(seed)
            params_v2.useRandomCoords = True
            embed_code = AllChem.EmbedMolecule(mol_h, params_v2)
            if int(embed_code) != 0:
                return (
                    None, STATUS_PDBQT_FAIL,
                    f"RDKit embed failed code={embed_code}",
                )
        # MMFF94 optimisation — non-fatal if it returns non-zero.
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        return None, STATUS_PDBQT_FAIL, f"3D conformer failed: {exc!r}"

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - meeko optional
        return (
            None, STATUS_PDBQT_FAIL,
            f"meeko import failed: {exc!r}",
        )

    try:
        preparator = MoleculePreparation()
        for setup in preparator.prepare(mol_h):
            s, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok and s:
                note = pt_strip_note if pt_strip_note else ""
                return s, STATUS_REAL_VINA, note
        return None, STATUS_PDBQT_FAIL, "Meeko write_string returned no OK setup"
    except Exception as exc:  # noqa: BLE001
        return None, STATUS_PDBQT_FAIL, f"meeko prepare failed: {exc!r}"


def _write_pdbqt_tmp(
    pdbqt_string: str, work_dir: Path, tag: str
) -> Path:
    """Write a PDBQT string to a temp file under ``work_dir``."""
    out = work_dir / f"{tag}.pdbqt"
    out.write_text(pdbqt_string, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Helpers — Vina binary invocation
# ---------------------------------------------------------------------------
def _probe_vina_binary(vina_bin: str) -> Tuple[bool, str]:
    """Probe the Vina CLI binary with ``--version``.

    Returns ``(ok, version_string)``.  ``ok`` is ``True`` iff the
    binary executes and returns a non-empty version string.
    """
    path = Path(vina_bin)
    if not path.is_file():
        return False, ""
    try:
        cp = subprocess.run(
            [str(path), "--version"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, ""
    out = (cp.stdout or cp.stderr or "").strip()
    return bool(out and cp.returncode == 0), out.splitlines()[0] if out else ""


# ---------------------------------------------------------------------------
# Wf-338-V2: Smina + QuickVina 2 alternative engine probes
# ---------------------------------------------------------------------------
# Per Wf-334-U3 §6.2 + Wf-336-X4 §10.2: when the canonical AutoDock Vina
# CLI is unreachable OR the Wf-264 sub-panel MEASURED 0/100 docked ok on
# 1h36 (meeko 0.8.0 metal-atom limitation per Wf-264 §2), the panel
# should be able to transparently fall back to **Smina** (Forli lab;
# https://sourceforge.net/projects/smina/) or **QuickVina 2**
# (Handoko et al. 2018; https://qvina.org/). Both share Vina's CLI
# surface so dispatch is uniform.
#
# Honest framing per AGENTS.md §5.2 / party-twenty-spirit §3.5:
#   * On the host we DO NOT have smina or qvina2 installed (the only
#     available engine is the Vina CLI at
#     ~/.local/micromamba/envs/molmetal_vina/bin/vina per Wf-329-W7).
#   * The probe functions therefore return ``(False, "")`` on this
#     host. They are correctness-checked against a synthetic binary
#     stub in the Wf-338-V2 integration tests (test 1 +
#     test 2 of ``test_wf338v2_wf298d_wf176_integration.py``).
# ---------------------------------------------------------------------------
def _probe_smina_binary(smina_bin: str = DEFAULT_SMINA_BIN) -> Tuple[bool, str]:
    """Probe the Smina CLI binary with ``--version``.

    Smina is the canonical AutoDock Vina fork with extended scoring
    functions including a MetalIon-aware term. Per Wf-338-V2 task
    brief, this probe wires the Smina recovery path so the Wf-264
    sub-panel can fall back when the canonical Vina CLI is missing
    or the Wf-264 §2 meeko metal-atom limit applies.

    Returns ``(ok, version_string)``.  ``ok`` is ``True`` iff the
    binary executes and returns a non-empty version string. The probe
    tries ``--version`` first; Smina also accepts ``--help`` but
    ``--version`` is the canonical version-detection probe across all
    three engines (Vina / Smina / QuickVina 2).
    """
    # Smina is conventionally on PATH (well-known name), but accept
    # an absolute path override for testing.
    if os.path.isabs(smina_bin):
        path = Path(smina_bin)
    else:
        path = Path(smina_bin)
    if not path.is_file():
        # Try shutil.which for the bare-name case (PATH lookup).
        which = shutil.which(smina_bin)
        if which is None:
            return False, ""
        path = Path(which)
    try:
        cp = subprocess.run(
            [str(path), "--version"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, ""
    out = (cp.stdout or cp.stderr or "").strip()
    return bool(out and cp.returncode == 0), out.splitlines()[0] if out else ""


def _probe_quickvina2_binary(
    qvina2_bin: str = DEFAULT_QUICKVINA2_BIN,
) -> Tuple[bool, str]:
    """Probe the QuickVina 2 CLI binary with ``--version``.

    QuickVina 2 (Handoko et al. 2018) is ~2x faster than Vina 1.1.2
    with the same scoring function. Per Wf-338-V2 task brief, this
    probe wires the QuickVina 2 recovery path so the Wf-264
    sub-panel can fall back when the canonical Vina CLI is missing
    or when higher-throughput scans are preferred (e.g. for the
    Wf-264 5-pocket × n_molecules cohort, QuickVina 2 may cut
    wall-time in half while keeping the Vina-compatible scoring
    function).

    Returns ``(ok, version_string)``.  ``ok`` is ``True`` iff the
    binary executes and returns a non-empty version string.
    """
    if os.path.isabs(qvina2_bin):
        path = Path(qvina2_bin)
    else:
        path = Path(qvina2_bin)
    if not path.is_file():
        which = shutil.which(qvina2_bin)
        if which is None:
            return False, ""
        path = Path(which)
    try:
        cp = subprocess.run(
            [str(path), "--version"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, ""
    out = (cp.stdout or cp.stderr or "").strip()
    return bool(out and cp.returncode == 0), out.splitlines()[0] if out else ""


def _select_engine(
    engine: str,
    vina_bin: str = DEFAULT_VINA_BIN,
    smina_bin: str = DEFAULT_SMINA_BIN,
    qvina2_bin: str = DEFAULT_QUICKVINA2_BIN,
) -> Tuple[str, str, bool]:
    """Resolve the engine selector to a concrete (engine_name, binary_path, ok).

    Wf-338-V2: when ``engine="auto"`` (or any unsupported string),
    transparently fall back: Vina → Smina → QuickVina 2, returning
    the first engine whose probe succeeds.  When ``engine`` is one of
    ``"vina"``/``"smina"``/``"quickvina2"``, force that engine and
    return its probe result (ok may be False — caller decides).

    Honest framing per AGENTS.md §5.2: this function never FABRICATES
    a successful probe; if no engine is reachable, it returns
    ``("", "", False)`` so the caller can fall back to STUB_FALLBACK /
    MEEKO_PREP_ONLY honestly.
    """
    if engine == "vina" or engine == "auto":
        ok, version = _probe_vina_binary(vina_bin)
        if ok:
            return "vina", vina_bin, True
        if engine == "vina":
            return "vina", vina_bin, False
    if engine == "smina" or engine == "auto":
        ok, version = _probe_smina_binary(smina_bin)
        if ok:
            return "smina", smina_bin, True
        if engine == "smina":
            return "smina", smina_bin, False
    if engine == "quickvina2" or engine == "auto":
        ok, version = _probe_quickvina2_binary(qvina2_bin)
        if ok:
            return "quickvina2", qvina2_bin, True
        if engine == "quickvina2":
            return "quickvina2", qvina2_bin, False
    return "", "", False


def _parse_vina_output(pdbqt_text: str) -> List[float]:
    """Extract kcal/mol values from a Vina docked PDBQT.

    Vina writes model blocks like::

        MODEL 1
        REMARK VINA RESULT:    -8.342      0.000      0.000
        ...

    Returns a list of floats (best = first).
    """
    scores: List[float] = []
    for line in pdbqt_text.splitlines():
        line = line.rstrip()
        if "VINA RESULT" not in line:
            continue
        # After "VINA RESULT:" the next 3 whitespace-separated fields
        # are: energy, lower-bound RMSD, upper-bound RMSD.
        parts = line.split("VINA RESULT:")
        if len(parts) < 2:
            continue
        tail = parts[1].split()
        if not tail:
            continue
        v = _safe_float(tail[0])
        if v is not None:
            scores.append(v)
    return scores


def _dock_one(
    smiles: str,
    pocket_id: str,
    receptor: Path,
    vina_bin: str,
    center: Tuple[float, float, float],
    box_size: float,
    exhaustiveness: int,
    n_poses: int,
    seed: int,
    timeout: int,
    work_dir: Path,
    pt_strip: bool = False,
    use_f2a_metal_ligand: bool = False,
    f2a_modifier_kcal_mol: Optional[float] = None,
    use_wf246c_metal_coord: bool = False,
    wf246c_kcal_per_unit: Optional[float] = None,
    pocket_donor_counts: Optional[Dict[str, int]] = None,
    use_per_bond_f2a: bool = True,
    use_distance_score: bool = True,
) -> PocketCell:
    """Run one (molecule, pocket) dock via the Vina CLI binary.

    Returns a :class:`PocketCell` — never raises.  All failure modes
    (parse, meeko, vina subprocess, timeout, missing binary) become
    explicit ``status`` codes on the returned cell.

    When ``pt_strip`` is True, the precious-metal coordination shell
    is dropped before 3D embedding + PDBQT generation (Wf-301B).

    When ``use_f2a_metal_ligand`` is True (Wf-329-W3 default ON),
    the F2(a) MetalLigandExchange covalent scoring modifier is
    applied on top of the Vina kcal/mol. The original Vina score
    is preserved as ``best_kcal_mol``; the F2(a)-adjusted score is
    stored as ``f2a_kcal_mol``. The number of detected covalent
    bonds (``f2a_n_bonds``), the modifier total
    (``f2a_modifier_total``), the matched pattern names
    (``f2a_matched_patterns``), and the recovery classification
    (``f2a_recovery_status``) are all recorded on the cell so
    downstream consumers can audit the lift.

    When ``use_wf246c_metal_coord`` is True (Wf-333-V2 default ON),
    the Wf-246-C metal-coordination composite (coordination geometry
    + HSAB + distance) is applied on top of the F2(a)-adjusted
    kcal/mol — additive per Wf-331-R1 §6.6.  The combined kcal/mol
    (Vina + F2(a) + Wf-246-C) is stored as ``metal_coord_kcal_mol``
    on the cell.  The composite (geometry/HSAB/distance/total),
    the per-cell kcal/mol adjustment, and the recovery
    classification are all recorded for audit.  Per Wf-333-V2 P0
    the sign convention is correct: a perfect composite yields a
    negative (more negative kcal/mol = better binding) adjustment.

    Wf-333-V3 P2 — ``use_per_bond_f2a`` (default True):
    The per-metal-donor-arm-bond counting selector replaces the
    per-centre count.  For cisplatin: per-bond = 2 (Pt-N × 2) vs
    per-centre = 1 (Pt × 1).  Both ``f2a_kcal_mol`` (per-centre)
    and ``f2a_per_bond_kcal_mol`` are recorded.

    Wf-333-V3 P3 — ``use_distance_score`` (default True):
    The Wf-246-C distance sub-score is computed from the docked
    pocket's donor-atom positions and the metal centre 3-D position
    (from RDKit ETKDGv3 conformer of the original SMILES).  The
    composite adjustment is added on top of the per-bond F2(a)
    score to produce the V3 combined kcal/mol.
    """
    t0 = time.perf_counter()
    cell = PocketCell(pocket=pocket_id)

    pdbqt_string, status, note = _smiles_to_pdbqt_string(
        smiles, seed=seed, pt_strip=pt_strip
    )
    if pdbqt_string is None:
        cell.status = status
        cell.note = note
        cell.wall_time_s = time.perf_counter() - t0
        return cell

    tag = f"lig_{abs(hash((smiles, pocket_id))) % (10 ** 8)}"
    lig_pdbqt = _write_pdbqt_tmp(pdbqt_string, work_dir, tag)
    out_pdbqt = work_dir / f"{tag}_out.pdbqt"
    log_path = work_dir / f"{tag}.vina.log"

    cmd = [
        str(vina_bin),
        "--receptor", str(receptor),
        "--ligand", str(lig_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size:.3f}",
        "--size_y", f"{box_size:.3f}",
        "--size_z", f"{box_size:.3f}",
        "--out", str(out_pdbqt),
        "--exhaustiveness", str(int(exhaustiveness)),
        "--num_modes", str(int(n_poses)),
        "--seed", str(int(seed)),
        "--verbosity", "0",
    ]

    try:
        log_fh = log_path.open("w", encoding="utf-8")
    except OSError:
        log_fh = None
    try:
        try:
            cp = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            cell.status = STATUS_TIMEOUT
            cell.note = f"vina timeout after {timeout}s: {exc!r}"
            cell.wall_time_s = time.perf_counter() - t0
            return cell
        except (FileNotFoundError, OSError) as exc:
            cell.status = STATUS_NO_RECEPTOR if "receptor" in str(exc) else STATUS_DOCK_FAIL
            cell.note = f"subprocess error: {exc!r}"
            cell.wall_time_s = time.perf_counter() - t0
            return cell
        if log_fh is not None:
            try:
                log_fh.write((cp.stdout or "") + (cp.stderr or ""))
            except OSError:
                pass
        if cp.returncode != 0:
            cell.status = STATUS_DOCK_FAIL
            tail = (cp.stderr or cp.stdout or "")[-200:]
            cell.note = f"vina rc={cp.returncode}: {tail}"
            cell.wall_time_s = time.perf_counter() - t0
            return cell
        # Parse docked PDBQT (Vina writes the kcal/mol to VINA RESULT
        # lines in --out).
        if not out_pdbqt.is_file():
            cell.status = STATUS_DOCK_FAIL
            cell.note = f"vina did not write --out file at {out_pdbqt}"
            cell.wall_time_s = time.perf_counter() - t0
            return cell
        scores = _parse_vina_output(out_pdbqt.read_text(encoding="utf-8", errors="replace"))
        if not scores:
            cell.status = STATUS_DOCK_FAIL
            cell.note = "vina output had no VINA RESULT lines"
            cell.wall_time_s = time.perf_counter() - t0
            return cell
        cell.best_kcal_mol = float(scores[0])
        cell.n_poses = int(len(scores))
        cell.status = STATUS_REAL_VINA
        cell.wall_time_s = time.perf_counter() - t0

        # Wf-329-W3: F2(a) MetalLigandExchange covalent scoring. When
        # enabled, the F2(a) scorer runs against the SMILES used to
        # generate the PDBQT (NOT the post-Pt-strip SMILES — the F2(a)
        # rule needs to see the metal centre, not the stripped
        # organic fragment). The modifier is applied on top of the
        # Vina kcal/mol and the result is recorded as ``f2a_kcal_mol``.
        if use_f2a_metal_ligand:
            # Per-centre scorer (Wf-329-W3 baseline).
            scorer_pc, _err = _get_f2a_scorer(
                modifier_kcal_mol=f2a_modifier_kcal_mol,
                use_per_bond=False,
            )
            if scorer_pc is not None:
                f2a_score, mod_total, n_bonds = scorer_pc.apply_covalent_modifier(
                    smiles, cell.best_kcal_mol
                )
                cell.f2a_kcal_mol = f2a_score
                cell.f2a_modifier_total = float(mod_total)
                cell.f2a_n_bonds = int(n_bonds)
                if n_bonds > 0:
                    dispatch = scorer_pc.dispatch(smiles)
                    cell.f2a_matched_patterns = ",".join(
                        dispatch.matched_patterns
                    )
                cell.f2a_recovery_status, _gap = scorer_pc.classify_recovery(
                    f2a_score
                )

            # Wf-333-V3 P2: per-bond F2(a) selector.  Use a separate
            # scorer instance with ``use_per_bond=True`` so the
            # per-bond and per-centre counts are recorded independently.
            if use_per_bond_f2a:
                scorer_pb, _err2 = _get_f2a_scorer(
                    modifier_kcal_mol=f2a_modifier_kcal_mol,
                    use_per_bond=True,
                )
                if scorer_pb is not None:
                    pb_score, pb_mod, pb_n = scorer_pb.apply_covalent_modifier(
                        smiles, cell.best_kcal_mol
                    )
                    cell.f2a_per_bond_kcal_mol = pb_score
                    cell.f2a_per_bond_n_bonds = int(pb_n)
                    cell.f2a_per_bond_modifier_total = float(pb_mod)
                    cell.f2a_per_bond_recovery_status, _gap_pb = (
                        scorer_pb.classify_recovery(pb_score)
                    )

        # Wf-333-V2: Wf-246-C metal-coordination composite scoring.
        # When enabled, the Wf-246-C term runs against the same SMILES
        # as F2(a) (NOT the post-Pt-strip SMILES — the metal-coord term
        # needs to see the metal centre). The kcal/mol adjustment is
        # ADDITIVE on top of the F2(a)-adjusted kcal/mol (when F2(a) is
        # enabled) or the raw Vina kcal/mol (when F2(a) is disabled).
        # The combined score is stored as ``metal_coord_kcal_mol``.
        # Sign convention per Wf-333-V2 P0 sign-fix: a perfect composite
        # (1.0) yields a negative (better binding) adjustment; a zero
        # composite yields a positive (worse binding) penalty. For
        # organic-only ligands, the composite is None and the
        # adjustment is exactly 0.0 (backward compat).
        if use_wf246c_metal_coord:
            mc_adj, mc_diag = _compute_wf246c_adjustment_for_smiles(
                smiles,
                pocket_donor_counts=pocket_donor_counts,
                kcal_per_unit=(
                    float(wf246c_kcal_per_unit)
                    if wf246c_kcal_per_unit is not None
                    else 1.0
                ),
            )
            if mc_diag is not None:
                cell.metal_coord_total = mc_diag.get("metal_coord_total")
                cell.metal_coord_adjustment = float(
                    mc_diag.get("metal_coord_kcal_adjustment", 0.0)
                )
                cell.metal_coord_geometry_score = mc_diag.get("geometry_score")
                cell.metal_coord_hsab_score = mc_diag.get("hsab_score")
                cell.metal_coord_distance_score = mc_diag.get("distance_score")
            if mc_adj is not None:
                # Combined additive: take the F2(a) score when available,
                # otherwise the raw Vina score, then add the Wf-246-C
                # adjustment on top.
                base = cell.f2a_kcal_mol
                if base is None:
                    base = cell.best_kcal_mol
                if base is not None:
                    cell.metal_coord_kcal_mol = float(base + mc_adj)
                    # Stash the V2 adjustment so the V3 combined-lift
                    # block can correctly REPLACE V2's contribution
                    # with V3's contribution when both fire.
                    cell._v2_mc_adj = float(mc_adj)
                    # Recovery classification.
                    combined = cell.metal_coord_kcal_mol
                    if combined <= -7.0:
                        cell.metal_coord_recovery_status = "INDUSTRY_CUTOFF_PASS"
                    elif combined <= -5.0:
                        cell.metal_coord_recovery_status = "PARTIAL_RECOVERY"
                    else:
                        cell.metal_coord_recovery_status = "INSUFFICIENT_RECOVERY"
            elif cell.f2a_kcal_mol is not None:
                # No Wf-246-C composite (organic ligand), inherit F2(a)
                # so downstream aggregate stats remain consistent.
                cell.metal_coord_kcal_mol = cell.f2a_kcal_mol
                cell.metal_coord_recovery_status = cell.f2a_recovery_status
            elif cell.best_kcal_mol is not None:
                cell.metal_coord_kcal_mol = cell.best_kcal_mol
                cell.metal_coord_recovery_status = "NO_METAL_COORD_TERM"

        # Wf-333-V3 P3: distance sub-score wire-in.  Compute the
        # closest pocket-donor-atom distance to the metal centre and
        # feed it to the Wf-246-C MetalCoordinationTerm via a NEW
        # call that re-scores with explicit metal-ligand distances.
        # The composite (geometry + HSAB + distance) is stored on
        # the cell as ``metal_coord_score`` / ``metal_coord_distance_score``
        # etc.  When the term fires, ``metal_coord_adjustment`` is the
        # kcal/mol adjustment (sign-corrected per Wf-333-V2 P0).
        if use_distance_score:
            term, _mc_err = _get_metal_coord_term()
            if term is not None:
                metal_pos = _metal_centre_coords_from_smiles(smiles)
                donor_coords = _read_pocket_donor_coords(receptor)
                if metal_pos is not None and donor_coords:
                    closest = _closest_donor_distances(
                        metal_pos, donor_coords
                    )
                    if closest:
                        try:
                            score_obj = term.score(
                                smiles,
                                metal_ligand_distances=closest,
                            )
                        except Exception:
                            score_obj = None
                        if score_obj is not None:
                            cell.metal_coord_score = score_obj.total
                            cell.metal_coord_distance_score = (
                                score_obj.distance_score
                            )
                            cell.metal_coord_geometry_score = (
                                score_obj.geometry_score
                            )
                            cell.metal_coord_hsab_score = score_obj.hsab_score
                            adj = term.kcal_adjustment(score_obj)
                            cell.metal_coord_adjustment = (
                                float(adj) if adj is not None else 0.0
                            )
                            cell.metal_coord_donor_distances = json.dumps(
                                {k: round(v, 3) for k, v in closest.items()},
                                sort_keys=True,
                            )

        # Wf-333-V3 combined V3 lift (Vina + per-bond F2(a) +
        # metal-coord distance adjustment).  The combined kcal/mol is
        # the canonical V1 + V2 + V3 lift axis.  Computed only when
        # both the per-bond F2(a) (P2) and the distance sub-score
        # (P3) fired; falls back to the per-bond F2(a) when P3 is off.
        if cell.best_kcal_mol is not None:
            base = cell.f2a_per_bond_kcal_mol
            if base is None:
                base = cell.metal_coord_kcal_mol
            if base is None:
                base = cell.f2a_kcal_mol
            if base is None:
                base = cell.best_kcal_mol
            combined = float(base)
            # Add the distance sub-score adjustment ONLY when it
            # comes from the explicit metal-ligand distance path
            # (P3).  We use the cell.metal_coord_adjustment which was
            # set by the P3 block above (signed per Wf-333-V2 P0).
            # When the Wf-246-C V2 path already added its own
            # adjustment via ``metal_coord_kcal_mol``, the P3 path
            # may have re-set ``metal_coord_adjustment`` to a tighter
            # value; we always apply the latest adjustment here.
            mc_adj = cell.metal_coord_adjustment
            if mc_adj is not None and use_distance_score:
                # If base already includes the Wf-246-C V2 adjustment
                # (it does when use_wf246c_metal_coord was True), we
                # need to subtract the V2 contribution before adding
                # the V3 contribution.  We do this by computing the
                # delta = V3_adjustment - V2_adjustment.  We track
                # the V2 adjustment separately via cell._v2_mc_adj
                # set above.  Simpler approach: when both V2 and V3
                # fired, we treat ``base`` (= metal_coord_kcal_mol)
                # as already including V2, and we want to REPLACE
                # V2's contribution with V3's contribution.  We
                # therefore reset base to ``best_kcal_mol`` and add
                # per-bond F2(a) + V3 adjustment.
                v2_mc_adj = getattr(cell, "_v2_mc_adj", None)
                if v2_mc_adj is not None:
                    # Re-base: Vina + per-bond F2(a) + V3 distance.
                    v3_base = float(cell.best_kcal_mol)
                    if cell.f2a_per_bond_kcal_mol is not None:
                        v3_base = float(cell.f2a_per_bond_kcal_mol)
                    combined = v3_base + float(mc_adj)
                else:
                    combined = float(base) + float(mc_adj)
            cell.v3_combined_kcal_mol = float(combined)
            cell.v3_combined_lift_kcal_mol = (
                float(combined) - float(cell.best_kcal_mol)
            )
            # Recovery status classification.
            gap_to_industry = float(combined) - (-7.0)
            if gap_to_industry <= 0.0:
                cell.v3_combined_recovery_status = "INDUSTRY_CUTOFF_PASS"
            elif gap_to_industry <= 1.7:
                cell.v3_combined_recovery_status = "RECOVERY_BORDERLINE"
            else:
                cell.v3_combined_recovery_status = (
                    "HONEST_NEG_VINA_PERSISTENT"
                )

        return cell
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------
def build_panel(
    smiles_list: Sequence[str],
    pocket_pdb_dir: Path,
    pockets: Sequence[str],
    vina_bin: str,
    exhaustiveness: int,
    n_poses: int,
    box_size: float,
    seed: int,
    timeout: int,
    work_dir: Optional[Path] = None,
    dock_runner=None,
    pt_strip: bool = False,
    use_f2a_metal_ligand: bool = False,
    f2a_modifier_kcal_mol: Optional[float] = None,
    use_wf246c_metal_coord: bool = False,
    wf246c_kcal_per_unit: Optional[float] = None,
    use_per_bond_f2a: bool = True,
    use_distance_score: bool = True,
    meeko_only: bool = False,
    engine: str = DEFAULT_ENGINE,
    smina_bin: str = DEFAULT_SMINA_BIN,
    qvina2_bin: str = DEFAULT_QUICKVINA2_BIN,
) -> Dict[str, Any]:
    """Assemble the Wf-264-Vina docking panel for a list of SMILES.

    Parameters
    ----------
    smiles_list : sequence of str
        Molecules to dock.
    pocket_pdb_dir : Path
        Directory containing one subdirectory per pocket.
    pockets : sequence of str
        Pocket identifiers to enumerate (e.g. ``DEFAULT_POCKETS``).
    vina_bin : str
        Path to the Vina CLI binary.
    exhaustiveness, n_poses, box_size, seed, timeout : int / float
        Vina CLI parameters.
    work_dir : Path, optional
        Working directory for per-dock PDBQT + log files.  Defaults
        to a sibling directory of the final ``--out-json`` (created
        by :func:`main`).
    dock_runner : callable, optional
        Test seam — if provided, used instead of the real
        :func:`_dock_one`.  Signature matches ``_dock_one``.
    pt_strip : bool, default ``False``
        Strip precious-metal coordination shell before Vina
        (Wf-301B).  Required when SMILES contain Pt/Pd/Ir/Ru/Au
        coordination centres that RDKit UFF cannot parameterise.
    use_f2a_metal_ligand : bool, default ``False``
        Wf-329-W3: apply the F2(a) MetalLigandExchange covalent
        scoring modifier on top of Vina kcal/mol.
    f2a_modifier_kcal_mol : float, optional
        Wf-329-W3: override the default covalent scoring modifier
        (-1.5 kcal/mol per detected covalent bond). When ``None``,
        uses the F2(a) default from
        :mod:`molmetal.scripts.wf329w3_f2a_metal_ligand`.
    use_wf246c_metal_coord : bool, default ``False``
        Wf-333-V2: apply the Wf-246-C metal-coordination composite
        (coordination geometry + HSAB + distance) on top of the
        F2(a)-adjusted kcal/mol — additive per Wf-331-R1 §6.6.
    wf246c_kcal_per_unit : float, optional
        Wf-333-V2: override the Wf-246-C kcal/mol scale
        (default 1.0 kcal/mol per unit of composite deviation from
        0.5). When ``None``, uses the Wf-246-C default.
    use_per_bond_f2a : bool, default ``True``
        Wf-333-V3 P2: replace per-centre F2(a) count with
        per-metal-donor-arm-bond count.
    use_distance_score : bool, default ``True``
        Wf-333-V3 P3: wire the Wf-246-C distance sub-score from
        the docked pocket's donor-atom positions to the metal centre.

    Returns
    -------
    dict
        The assembled panel dictionary (see module docstring).
    """
    if work_dir is None:
        work_dir = Path(tempfile_workspace())
    work_dir.mkdir(parents=True, exist_ok=True)

    if dock_runner is None:
        dock_runner = _dock_one

    # Wf-336-X4: probe meeko availability once per panel build so the
    # JSON output carries an honest ``meeko_available`` field.  When
    # meeko is unavailable AND the Vina binary is also unavailable,
    # the panel will emit ``status="MEEKO_PREP_ONLY"`` for every cell
    # (with the prepared PDBQT path recorded on the cell ``note``).
    meeko_probe = get_meeko_probe()

    # Per-pocket staging + receptor centre cache.
    pocket_receptors: Dict[str, Optional[Path]] = {
        p: _resolve_pocket_receptor(pocket_pdb_dir, p) for p in pockets
    }
    pocket_centers: Dict[str, Tuple[float, float, float]] = {}
    for p, rec in pocket_receptors.items():
        if rec is not None:
            pocket_centers[p] = _pocket_center(rec)
        else:
            pocket_centers[p] = (0.0, 0.0, 0.0)

    # Probe the Vina binary only when we are going to run the real
    # dock path.  When the caller injects a ``dock_runner`` (test
    # seam) we skip the probe so the test can exercise the panel
    # end-to-end without invoking the real Vina CLI.  When
    # ``meeko_only`` is set (Wf-336-X4) the probe is short-circuited
    # so every cell goes through the meeko-only path regardless of
    # whether the Vina binary is reachable.
    if dock_runner is _dock_one:
        if meeko_only:
            # Wf-336-X4: --meeko-only forces the alternative path so
            # the probe result is overridden to NOT_OK.  This is the
            # smoke-test flag for the meeko integration.
            vina_ok, vina_version = False, "meeko-only-mode"
        elif engine in ("smina", "quickvina2", "auto"):
            # Wf-338-V2: engine recovery path — try the requested
            # engine (or auto-fallback through Vina→Smina→QuickVina 2)
            # when the canonical Vina CLI is missing or the Wf-264
            # §2 meeko metal-atom limit applies.  The probe never
            # FABRICATES ok=True per AGENTS.md §5.2 honest framing.
            selected_engine, selected_bin, sel_ok = _select_engine(
                engine=engine,
                vina_bin=vina_bin,
                smina_bin=smina_bin,
                qvina2_bin=qvina2_bin,
            )
            vina_ok = sel_ok
            vina_version = (
                f"engine={selected_engine} bin={selected_bin}"
                if sel_ok else f"engine={engine} (no probe ok)"
            )
        else:
            vina_ok, vina_version = _probe_vina_binary(vina_bin)
    else:
        vina_ok, vina_version = True, "fake-runner"

    # Pre-compute pocket donor fingerprints once per pocket (Wf-333-V2:
    # Wf-246-C HSAB sub-score consumes N/O/S/P atom counts from the
    # receptor.pdbqt).  When a receptor is staged, count N/O/S/P
    # heavy atoms in the PDBQT file as a coarse HSAB fingerprint.
    pocket_donor_counts_cache: Dict[str, Optional[Dict[str, int]]] = {}
    for p, rec in pocket_receptors.items():
        if rec is None:
            pocket_donor_counts_cache[p] = None
            continue
        try:
            from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (  # type: ignore  # noqa: WPS433
                pocket_donor_counts_from_atomicnums,
            )
        except Exception:
            pocket_donor_counts_cache[p] = None
            continue
        atomic_nums: List[int] = []
        try:
            with rec.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    atom_name = line[12:16].strip()
                    if atom_name.startswith("H") and atom_name not in ("HG", "HH"):
                        continue
                    elem = atom_name[:1]
                    mapping = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15}
                    anum = mapping.get(elem)
                    if anum is not None:
                        atomic_nums.append(anum)
        except OSError:
            pocket_donor_counts_cache[p] = None
            continue
        pocket_donor_counts_cache[p] = pocket_donor_counts_from_atomicnums(
            atomic_nums
        )

    molecules: List[MoleculeRecord] = []
    n_parsed = 0
    n_pdbqt_ok = 0
    n_docked = 0

    for smiles in smiles_list:
        mol_rec = MoleculeRecord(smiles=smiles)
        for pocket_id in pockets:
            rec = pocket_receptors[pocket_id]
            if rec is None:
                mol_rec.per_pocket[pocket_id] = PocketCell(
                    pocket=pocket_id,
                    status=STATUS_NO_RECEPTOR,
                    note=f"no receptor.pdbqt under {pocket_pdb_dir / pocket_id}",
                )
                continue
            if not vina_ok:
                # Wf-336-X4: when the Vina binary is missing/unhealthy
                # but meeko is available, we DO NOT emit STUB_FALLBACK —
                # we still prepare the ligand PDBQT via meeko so the
                # downstream consumer can feed it into an alternative
                # docking engine (CadButEaas in-process Vina, or
                # QuickVina / Smina if installed later).  The cell
                # status is MEEKO_PREP_ONLY; the prepared PDBQT path
                # is recorded on the cell note.  When meeko is also
                # unavailable, fall back to STUB_FALLBACK (preserving
                # the Wf-320H backward-compat behaviour).
                if meeko_probe.get("available", False):
                    pdbqt_string, prep_status, prep_note = _smiles_to_pdbqt_string(
                        smiles, seed=seed, pt_strip=pt_strip
                    )
                    if pdbqt_string is not None:
                        # Write the prepared PDBQT to a sibling work
                        # dir so a downstream consumer can pick it up.
                        tag = (
                            f"meeko_{pocket_id}_"
                            f"{abs(hash((smiles, pocket_id))) % (10 ** 8)}"
                        )
                        lig_path = _write_pdbqt_tmp(
                            pdbqt_string, work_dir, tag
                        )
                        mol_rec.per_pocket[pocket_id] = PocketCell(
                            pocket=pocket_id,
                            status=STATUS_MEEKO_PREP_ONLY,
                            note=(
                                f"Vina binary missing at {vina_bin}; "
                                f"meeko ligand prep written to "
                                f"{lig_path} for downstream consumer. "
                                f"{prep_note}".strip()
                            ),
                        )
                    else:
                        mol_rec.per_pocket[pocket_id] = PocketCell(
                            pocket=pocket_id,
                            status=STATUS_STUB_FALLBACK,
                            note=(
                                f"Vina binary missing/unhealthy at "
                                f"{vina_bin} AND meeko prep failed: "
                                f"{prep_note}"
                            ),
                        )
                else:
                    mol_rec.per_pocket[pocket_id] = PocketCell(
                        pocket=pocket_id,
                        status=STATUS_STUB_FALLBACK,
                        note=f"Vina binary missing/unhealthy at {vina_bin}",
                    )
                continue
            cell = dock_runner(
                smiles=smiles,
                pocket_id=pocket_id,
                receptor=rec,
                vina_bin=vina_bin,
                center=pocket_centers[pocket_id],
                box_size=box_size,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                seed=seed,
                timeout=timeout,
                work_dir=work_dir,
                pt_strip=pt_strip,
                use_f2a_metal_ligand=use_f2a_metal_ligand,
                f2a_modifier_kcal_mol=f2a_modifier_kcal_mol,
                use_wf246c_metal_coord=use_wf246c_metal_coord,
                wf246c_kcal_per_unit=wf246c_kcal_per_unit,
                pocket_donor_counts=pocket_donor_counts_cache.get(pocket_id),
                use_per_bond_f2a=use_per_bond_f2a,
                use_distance_score=use_distance_score,
            )
            mol_rec.per_pocket[pocket_id] = cell
            if cell.status == STATUS_REAL_VINA:
                n_docked += 1
        # Per-molecule derived flags (only if at least one pocket attempted
        # past PDBQT conversion).
        statuses = [c.status for c in mol_rec.per_pocket.values()]
        # parse_ok / pdbqt_ok: best-effort re-check from the SMILES.
        pdbqt_string, st, _ = _smiles_to_pdbqt_string(
            smiles, seed=seed, pt_strip=pt_strip
        )
        mol_rec.parse_ok = st != STATUS_PARSE_FAIL and pdbqt_string is not None
        mol_rec.pdbqt_ok = mol_rec.parse_ok and st != STATUS_PDBQT_FAIL
        if mol_rec.parse_ok:
            n_parsed += 1
        if mol_rec.pdbqt_ok:
            n_pdbqt_ok += 1
        molecules.append(mol_rec)

    # ------------------------------------------------------------------
    # JSON serialisation.
    # ------------------------------------------------------------------
    def _cell_to_dict(c: PocketCell) -> Dict[str, Any]:
        return {
            "best_kcal_mol": c.best_kcal_mol,
            "n_poses": c.n_poses,
            "wall_time_s": round(c.wall_time_s, 4),
            "status": c.status,
            "note": c.note,
            # Wf-329-W3 F2(a) MetalLigandExchange covalent scoring.
            "f2a_kcal_mol": c.f2a_kcal_mol,
            "f2a_n_bonds": c.f2a_n_bonds,
            "f2a_modifier_total": round(c.f2a_modifier_total, 4),
            "f2a_matched_patterns": c.f2a_matched_patterns,
            "f2a_recovery_status": c.f2a_recovery_status,
            # Wf-333-V3 P2: per-bond F2(a) selector.
            "f2a_per_bond_kcal_mol": c.f2a_per_bond_kcal_mol,
            "f2a_per_bond_n_bonds": c.f2a_per_bond_n_bonds,
            "f2a_per_bond_modifier_total": round(c.f2a_per_bond_modifier_total, 4),
            "f2a_per_bond_recovery_status": c.f2a_per_bond_recovery_status,
            # Wf-333-V2 Wf-246-C metal-coordination composite.
            "metal_coord_kcal_mol": c.metal_coord_kcal_mol,
            "metal_coord_adjustment": round(c.metal_coord_adjustment, 4),
            "metal_coord_total": c.metal_coord_total,
            "metal_coord_geometry_score": c.metal_coord_geometry_score,
            "metal_coord_hsab_score": c.metal_coord_hsab_score,
            "metal_coord_distance_score": c.metal_coord_distance_score,
            "metal_coord_recovery_status": c.metal_coord_recovery_status,
            # Wf-333-V3 P3: distance sub-score (P3 path).
            "metal_coord_score": c.metal_coord_score,
            "metal_coord_donor_distances": c.metal_coord_donor_distances,
            # Wf-333-V3 combined V3 lift.
            "v3_combined_kcal_mol": c.v3_combined_kcal_mol,
            "v3_combined_lift_kcal_mol": (
                round(c.v3_combined_lift_kcal_mol, 4)
                if c.v3_combined_lift_kcal_mol is not None else None
            ),
            "v3_combined_recovery_status": c.v3_combined_recovery_status,
        }

    per_molecule_json: List[Dict[str, Any]] = []
    for m in molecules:
        per_molecule_json.append({
            "smiles": m.smiles,
            "parse_ok": bool(m.parse_ok),
            "pdbqt_ok": bool(m.pdbqt_ok),
            "per_pocket": {p: _cell_to_dict(c) for p, c in m.per_pocket.items()},
            "best_pocket": m.best_pocket,
            "best_kcal_mol": m.best_kcal_mol,
            "mean_kcal_mol": m.mean_kcal_mol,
            # Wf-329-W3 F2(a) per-molecule aggregate.
            "f2a_best_kcal_mol": m.f2a_best_kcal_mol,
            "f2a_mean_kcal_mol": m.f2a_mean_kcal_mol,
            "f2a_n_bonds_total": m.f2a_n_bonds_total,
            # Wf-333-V2 Wf-246-C per-molecule aggregate.
            "metal_coord_best_kcal_mol": m.metal_coord_best_kcal_mol,
            "metal_coord_mean_kcal_mol": m.metal_coord_mean_kcal_mol,
            "metal_coord_total_score": m.metal_coord_total_score,
            # Wf-333-V3 V3 combined lift per-molecule aggregate.
            "v3_combined_best_kcal_mol": m.v3_combined_best_kcal_mol,
            "v3_combined_mean_kcal_mol": m.v3_combined_mean_kcal_mol,
            "v3_combined_lift_total": m.v3_combined_lift_total,
        })

    # ------------------------------------------------------------------
    # Aggregate stats.
    # ------------------------------------------------------------------
    aggregate_per_pocket: Dict[str, Dict[str, Any]] = {}
    for pocket_id in pockets:
        cells = [
            m.per_pocket.get(pocket_id) for m in molecules
            if pocket_id in m.per_pocket
        ]
        cells = [c for c in cells if c is not None]
        real_scores = [
            c.best_kcal_mol for c in cells
            if c.best_kcal_mol is not None
        ]
        f2a_scores = [
            c.f2a_kcal_mol for c in cells
            if c.f2a_kcal_mol is not None
        ]
        mc_scores = [
            c.metal_coord_kcal_mol for c in cells
            if c.metal_coord_kcal_mol is not None
        ]
        n_real = sum(1 for c in cells if c.status == STATUS_REAL_VINA)
        n_stub = sum(1 for c in cells if c.status == STATUS_STUB_FALLBACK)
        # Wf-336-X4: meeko alternative integration — count cells that
        # recovered from STUB_FALLBACK to MEEKO_PREP_ONLY (the prepared
        # PDBQT is on disk for downstream consumption).
        n_meeko = sum(
            1 for c in cells
            if c.status == STATUS_MEEKO_PREP_ONLY
        )
        n_f2a = sum(1 for c in cells if c.f2a_n_bonds > 0)
        n_mc = sum(
            1 for c in cells
            if c.metal_coord_total is not None
        )
        aggregate_per_pocket[pocket_id] = {
            "n": int(len(cells)),
            "n_real_vina": int(n_real),
            "n_stub_fallback": int(n_stub),
            # Wf-336-X4: meeko alternative integration count.
            "n_meeko_prep_only": int(n_meeko),
            "n_other": int(len(cells) - n_real - n_stub - n_meeko),
            "mean": (
                float(round(sum(real_scores) / len(real_scores), 4))
                if real_scores else None
            ),
            "min": (
                float(round(min(real_scores), 4)) if real_scores else None
            ),
            "max": (
                float(round(max(real_scores), 4)) if real_scores else None
            ),
            # Wf-329-W3 F2(a) MetalLigandExchange aggregate.
            "f2a_mean": (
                float(round(sum(f2a_scores) / len(f2a_scores), 4))
                if f2a_scores else None
            ),
            "f2a_min": (
                float(round(min(f2a_scores), 4)) if f2a_scores else None
            ),
            "f2a_n_covalent": int(n_f2a),
            # Wf-333-V2 Wf-246-C metal-coordination aggregate.
            "metal_coord_mean": (
                float(round(sum(mc_scores) / len(mc_scores), 4))
                if mc_scores else None
            ),
            "metal_coord_min": (
                float(round(min(mc_scores), 4)) if mc_scores else None
            ),
            "metal_coord_n_with_total": int(n_mc),
        }

    all_real_scores: List[float] = [
        cell.best_kcal_mol
        for m in molecules
        for cell in m.per_pocket.values()
        if cell.best_kcal_mol is not None
    ]
    best_pocket_overall: Optional[str] = None
    best_score_overall: Optional[float] = None
    if all_real_scores:
        best_score_overall = float(min(all_real_scores))
        for m in molecules:
            for pocket_id, cell in m.per_pocket.items():
                if cell.best_kcal_mol == best_score_overall:
                    best_pocket_overall = pocket_id
                    break
            if best_pocket_overall is not None:
                break

    mean_score_overall: Optional[float] = (
        float(round(sum(all_real_scores) / len(all_real_scores), 4))
        if all_real_scores else None
    )

    # Wf-329-W3: F2(a) cohort-wide aggregates.
    all_f2a_scores: List[float] = [
        cell.f2a_kcal_mol
        for m in molecules
        for cell in m.per_pocket.values()
        if cell.f2a_kcal_mol is not None
    ]
    f2a_mean_score_overall: Optional[float] = (
        float(round(sum(all_f2a_scores) / len(all_f2a_scores), 4))
        if all_f2a_scores else None
    )
    f2a_cohort_median: Optional[float] = None
    if all_f2a_scores:
        sorted_scores = sorted(all_f2a_scores)
        n = len(sorted_scores)
        mid = n // 2
        if n % 2 == 0:
            f2a_cohort_median = float(
                round((sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0, 4)
            )
        else:
            f2a_cohort_median = float(sorted_scores[mid])

    # Wf-329-W3: cohort-wide m3 median lift (median(F2(a)) - median(Vina))
    # gives the per-cell median covalent lift.  We compute it as
    # median(F2(a)) - median(Vina) for the same cell population.
    f2a_cohort_median_lift: Optional[float] = None
    if all_real_scores and all_f2a_scores and len(all_real_scores) == len(all_f2a_scores):
        sorted_vina = sorted(all_real_scores)
        sorted_f2a = sorted(all_f2a_scores)
        n = len(sorted_vina)
        mid = n // 2
        if n % 2 == 0:
            med_vina = (sorted_vina[mid - 1] + sorted_vina[mid]) / 2.0
            med_f2a = (sorted_f2a[mid - 1] + sorted_f2a[mid]) / 2.0
        else:
            med_vina = sorted_vina[mid]
            med_f2a = sorted_f2a[mid]
        f2a_cohort_median_lift = float(round(med_f2a - med_vina, 4))

    # Wf-333-V2: Wf-246-C metal-coordination cohort aggregates.
    all_mc_scores: List[float] = [
        cell.metal_coord_kcal_mol
        for m in molecules
        for cell in m.per_pocket.values()
        if cell.metal_coord_kcal_mol is not None
    ]
    mc_mean_score_overall: Optional[float] = (
        float(round(sum(all_mc_scores) / len(all_mc_scores), 4))
        if all_mc_scores else None
    )
    mc_cohort_median: Optional[float] = None
    if all_mc_scores:
        sorted_mc = sorted(all_mc_scores)
        n = len(sorted_mc)
        mid = n // 2
        if n % 2 == 0:
            mc_cohort_median = float(
                round((sorted_mc[mid - 1] + sorted_mc[mid]) / 2.0, 4)
            )
        else:
            mc_cohort_median = float(sorted_mc[mid])

    # Wf-333-V2: cohort-wide m3 median lift (median(metal_coord) -
    # median(Vina)) gives the per-cell median metal-coordination
    # lift over the raw Vina baseline.
    mc_cohort_median_lift: Optional[float] = None
    if all_real_scores and all_mc_scores and len(all_real_scores) == len(all_mc_scores):
        sorted_vina = sorted(all_real_scores)
        sorted_mc = sorted(all_mc_scores)
        n = len(sorted_vina)
        mid = n // 2
        if n % 2 == 0:
            med_vina = (sorted_vina[mid - 1] + sorted_vina[mid]) / 2.0
            med_mc = (sorted_mc[mid - 1] + sorted_mc[mid]) / 2.0
        else:
            med_vina = sorted_vina[mid]
            med_mc = sorted_mc[mid]
        mc_cohort_median_lift = float(round(med_mc - med_vina, 4))

    # Wf-333-V2: cohort-wide metal-coordination lift over F2(a).
    mc_over_f2a_median_lift: Optional[float] = None
    if all_f2a_scores and all_mc_scores and len(all_f2a_scores) == len(all_mc_scores):
        sorted_f2a = sorted(all_f2a_scores)
        sorted_mc = sorted(all_mc_scores)
        n = len(sorted_mc)
        mid = n // 2
        if n % 2 == 0:
            med_f2a = (sorted_f2a[mid - 1] + sorted_f2a[mid]) / 2.0
            med_mc = (sorted_mc[mid - 1] + sorted_mc[mid]) / 2.0
        else:
            med_f2a = sorted_f2a[mid]
            med_mc = sorted_mc[mid]
        mc_over_f2a_median_lift = float(round(med_mc - med_f2a, 4))

    # Wf-333-V3: V3 combined cohort aggregates (per-bond F2(a) +
    # distance sub-score).  The combined kcal/mol is the canonical
    # V1 + V2 + V3 lift axis for the m3 Vina Dock metric.
    all_pb_scores: List[float] = [
        cell.f2a_per_bond_kcal_mol
        for m in molecules
        for cell in m.per_pocket.values()
        if cell.f2a_per_bond_kcal_mol is not None
    ]
    all_v3_scores: List[float] = [
        cell.v3_combined_kcal_mol
        for m in molecules
        for cell in m.per_pocket.values()
        if cell.v3_combined_kcal_mol is not None
    ]
    v3_mean_score_overall: Optional[float] = (
        float(round(sum(all_v3_scores) / len(all_v3_scores), 4))
        if all_v3_scores else None
    )
    v3_cohort_median: Optional[float] = None
    if all_v3_scores:
        sorted_v3 = sorted(all_v3_scores)
        n = len(sorted_v3)
        mid = n // 2
        if n % 2 == 0:
            v3_cohort_median = float(
                round((sorted_v3[mid - 1] + sorted_v3[mid]) / 2.0, 4)
            )
        else:
            v3_cohort_median = float(sorted_v3[mid])

    # V3 cohort median lift = median(V3 combined) - median(Vina).
    v3_cohort_median_lift: Optional[float] = None
    if all_real_scores and all_v3_scores and len(all_real_scores) == len(all_v3_scores):
        sorted_vina = sorted(all_real_scores)
        sorted_v3 = sorted(all_v3_scores)
        n = len(sorted_vina)
        mid = n // 2
        if n % 2 == 0:
            med_vina = (sorted_vina[mid - 1] + sorted_vina[mid]) / 2.0
            med_v3 = (sorted_v3[mid - 1] + sorted_v3[mid]) / 2.0
        else:
            med_vina = sorted_vina[mid]
            med_v3 = sorted_v3[mid]
        v3_cohort_median_lift = float(round(med_v3 - med_vina, 4))

    # Per-pocket aggregates: per-bond F2(a) median + V3 combined median.
    aggregate_per_pocket_pb: Dict[str, Dict[str, Any]] = {}
    for pocket_id in pockets:
        cells = [
            m.per_pocket.get(pocket_id) for m in molecules
            if pocket_id in m.per_pocket
        ]
        cells = [c for c in cells if c is not None]
        pb_scores = [
            c.f2a_per_bond_kcal_mol for c in cells
            if c.f2a_per_bond_kcal_mol is not None
        ]
        v3_pocket_scores = [
            c.v3_combined_kcal_mol for c in cells
            if c.v3_combined_kcal_mol is not None
        ]
        pb_med = float(round(sum(pb_scores) / len(pb_scores), 4)) if pb_scores else None
        v3_med = float(round(sum(v3_pocket_scores) / len(v3_pocket_scores), 4)) if v3_pocket_scores else None
        # Update the per-pocket aggregate dict in place.
        if pocket_id in aggregate_per_pocket:
            aggregate_per_pocket[pocket_id].update({
                "f2a_per_bond_mean": pb_med,
                "v3_combined_mean": v3_med,
            })

    metadata = {
        "panel_version": PANEL_VERSION,
        "generated_utc": _utc_now_iso(),
        "vina_bin": str(vina_bin),
        "vina_version": vina_version,
        "vina_binary_ok": bool(vina_ok),
        "pocket_pdb_dir": str(pocket_pdb_dir),
        "pockets": list(pockets),
        "exhaustiveness": int(exhaustiveness),
        "n_poses": int(n_poses),
        "box_size": float(box_size),
        "seed": int(seed),
        "dock_timeout_s": int(timeout),
        "pt_strip": bool(pt_strip),
        # Wf-338-V2: Smina / QuickVina 2 engine recovery paths.
        "engine_requested": engine,
        "engine_smina_available": _probe_smina_binary(smina_bin)[0],
        "engine_quickvina2_available": _probe_quickvina2_binary(qvina2_bin)[0],
        "engine_smina_bin": str(smina_bin),
        "engine_qvina2_bin": str(qvina2_bin),
        # Wf-336-X4: meeko alternative integration probe.
        "meeko_available": bool(meeko_probe.get("available", False)),
        "meeko_version": str(meeko_probe.get("version", "")),
        "meeko_moleculeprep_class": str(
            meeko_probe.get("moleculeprep_class", "")
        ),
        "meeko_pdbqt_writer": str(meeko_probe.get("pdbqt_writer", "")),
        "meeko_import_error": str(meeko_probe.get("import_error", "")),
        "meeko_alternative_path": (
            # meeko alternative path is ACTIVE when:
            #   - meeko is importable AND
            #   - the Vina CLI is missing/unhealthy OR meeko-only mode
            #     is requested by the caller.
            # The probe emits ``true`` when the alternative integration
            # could rescue a docking cell that would otherwise be
            # STUB_FALLBACK (i.e. ``vina_binary_ok == False`` and
            # ``meeko_available == True``).
            bool(
                meeko_probe.get("available", False)
                and not vina_ok
            )
        ),
        "meeko_only_mode": bool(meeko_only),
        # Wf-329-W3 F2(a) MetalLigandExchange metadata.
        "use_f2a_metal_ligand": bool(use_f2a_metal_ligand),
        "f2a_modifier_kcal_mol": f2a_modifier_kcal_mol,
        "f2a_scorer_import_ok": _F2A_IMPORT_ERROR is None,
        # Wf-333-V2 Wf-246-C metal-coordination metadata.
        "use_wf246c_metal_coord": bool(use_wf246c_metal_coord),
        "wf246c_kcal_per_unit": wf246c_kcal_per_unit,
        "wf246c_term_import_ok": _WF246C_IMPORT_ERROR is None,
        # Wf-333-V3 P2 + P3 metadata.
        "use_per_bond_f2a": bool(use_per_bond_f2a),
        "use_distance_score": bool(use_distance_score),
    }

    return {
        "metadata": metadata,
        "n_input": int(len(smiles_list)),
        "n_parsed": int(n_parsed),
        "n_pdbqt_ok": int(n_pdbqt_ok),
        "n_docked": int(n_docked),
        # Wf-336-X4: meeko alternative integration cohort counter.
        "n_meeko_prep_only": int(
            sum(
                1
                for m in molecules
                for c in m.per_pocket.values()
                if c.status == STATUS_MEEKO_PREP_ONLY
            )
        ),
        "per_molecule": per_molecule_json,
        "aggregate": {
            "per_pocket": aggregate_per_pocket,
            "best_pocket_overall": best_pocket_overall,
            "best_score_overall": best_score_overall,
            "mean_score_overall": mean_score_overall,
            # Wf-329-W3 F2(a) MetalLigandExchange cohort aggregate.
            "f2a_mean_score_overall": f2a_mean_score_overall,
            "f2a_cohort_median": f2a_cohort_median,
            "f2a_cohort_median_lift": f2a_cohort_median_lift,
            "industry_cutoff_kcal_mol": -7.0,
            "f2a_cohort_median_gap_to_industry": (
                float(round(f2a_cohort_median - (-7.0), 4))
                if f2a_cohort_median is not None else None
            ),
            # Wf-333-V2 Wf-246-C metal-coordination cohort aggregate.
            "metal_coord_mean_score_overall": mc_mean_score_overall,
            "metal_coord_cohort_median": mc_cohort_median,
            "metal_coord_cohort_median_lift_vs_vina": mc_cohort_median_lift,
            "metal_coord_cohort_median_lift_vs_f2a": mc_over_f2a_median_lift,
            "metal_coord_cohort_median_gap_to_industry": (
                float(round(mc_cohort_median - (-7.0), 4))
                if mc_cohort_median is not None else None
            ),
            # Wf-333-V3 V3 combined cohort aggregate.
            "v3_combined_mean_score_overall": v3_mean_score_overall,
            "v3_combined_cohort_median": v3_cohort_median,
            "v3_combined_cohort_median_lift_vs_vina": v3_cohort_median_lift,
            "v3_combined_cohort_median_gap_to_industry": (
                float(round(v3_cohort_median - (-7.0), 4))
                if v3_cohort_median is not None else None
            ),
        },
    }


def tempfile_workspace() -> str:
    """Return a unique temp workspace path (test seam)."""
    import tempfile
    return tempfile.mkdtemp(prefix="wf264_vina_")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfm_vina_dock_panel.py",
        description=(
            "Wf-264-Vina: AutoDock Vina 5-pocket docking panel for "
            "CFM-generated (or any) SMILES. Generates a 3D conformer "
            "(ETKDGv3 + MMFF94), converts to PDBQT via Meeko, docks "
            "against each pocket via the project Vina binary, and "
            "emits per-molecule per-pocket kcal/mol + aggregate JSON."
        ),
    )
    p.add_argument(
        "--smiles-csv", required=True, type=Path,
        help="CSV with a 'smiles' column (one molecule per row).",
    )
    p.add_argument(
        "--pocket-pdb-dir", required=True, type=Path,
        help=(
            "Directory containing one subdirectory per pocket, each "
            "with receptor.pdbqt (Wf-30 staging layout)."
        ),
    )
    p.add_argument(
        "--out-json", required=True, type=Path,
        help="Output JSON path; will be created or overwritten.",
    )
    p.add_argument(
        "--vina-bin", type=str, default=DEFAULT_VINA_BIN,
        help=(
            f"Path to the AutoDock Vina CLI binary. "
            f"Default: {DEFAULT_VINA_BIN}"
        ),
    )
    p.add_argument(
        "--n-candidates", type=int, default=DEFAULT_N_POSES,
        help=(
            f"Vina --num_modes (n_poses). "
            f"Default: {DEFAULT_N_POSES}"
        ),
    )
    p.add_argument(
        "--pockets", nargs="+", default=list(DEFAULT_POCKETS),
        help=(
            "Pocket identifiers to enumerate. "
            f"Default: {' '.join(DEFAULT_POCKETS)}"
        ),
    )
    p.add_argument(
        "--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS,
        help=f"Vina --exhaustiveness. Default: {DEFAULT_EXHAUSTIVENESS}",
    )
    p.add_argument(
        "--box-size", type=float, default=DEFAULT_BOX_SIZE,
        help=f"Cubed box size (Å) around pocket centre. Default: {DEFAULT_BOX_SIZE}",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Vina --seed. Default: {DEFAULT_SEED}",
    )
    p.add_argument(
        "--dock-timeout", type=int, default=DEFAULT_DOCK_TIMEOUT,
        help=(
            f"Per-dock timeout (s). Default: {DEFAULT_DOCK_TIMEOUT}. "
            f"Use a small value for smoke tests."
        ),
    )
    p.add_argument(
        "--work-dir", type=Path, default=None,
        help=(
            "Working directory for per-dock PDBQT + log files "
            "(auto-created). Defaults to a sibling of --out-json."
        ),
    )
    p.add_argument(
        "--pt-strip", action="store_true",
        help=(
            "Wf-301B: strip precious-metal coordination shell "
            "([Pt](lig1)(lig2)...) and dock the largest remaining "
            "organic fragment.  Required when SMILES contain Pt/Pd/"
            "Ir/Ru/Au coordination centres that RDKit UFF cannot "
            "parameterise."
        ),
    )
    p.add_argument(
        "--use-f2a-metal-ligand", action="store_true",
        help=(
            "Wf-329-W3: apply the F2(a) MetalLigandExchange "
            "covalent scoring modifier on top of Vina kcal/mol. "
            "Detects Pt/Pd/Ru/Au centres in the SMILES (covalent-"
            "formed N7 adduct + covalent-capable bare-metal "
            "centres) and adds a per-bond modifier (default "
            "-1.5 kcal/mol; PROJECTED conservative midpoint per "
            "Wf-329-W3 task brief). Both Vina and F2(a)-adjusted "
            "kcal/mol are recorded on every cell."
        ),
    )
    p.add_argument(
        "--f2a-modifier-kcal-mol", type=float, default=None,
        help=(
            "Wf-329-W3: override the default covalent scoring "
            "modifier (-1.5 kcal/mol per detected covalent bond). "
            "Only used when --use-f2a-metal-ligand is set."
        ),
    )
    p.add_argument(
        "--use-wf246c-metal-coord", action="store_true",
        help=(
            "Wf-333-V2: apply the Wf-246-C metal-coordination "
            "composite (coordination geometry + HSAB + distance) "
            "on top of the F2(a)-adjusted kcal/mol — additive per "
            "Wf-331-R1 §6.6. For precious-metal metallodrugs "
            "(Pt/Ru/Ir/Os/Rh/Pd/Au) the composite scores the "
            "coordination chemistry against the pocket donor "
            "fingerprint. For organic ligands the composite is None "
            "and the adjustment is exactly 0.0 (backward compat). "
            "Default: False (use --use-wf246c-metal-coord to enable). "
            "Per Wf-333-V2 P0 the sign convention is "
            "(1 - 2 * composite) * kcal_per_unit — a perfect "
            "composite yields a negative (better binding) adjustment."
        ),
    )
    p.add_argument(
        "--wf246c-kcal-per-unit", type=float, default=None,
        help=(
            "Wf-333-V2: override the Wf-246-C kcal/mol scale "
            "(default 1.0 kcal/mol per unit of composite "
            "deviation from 0.5). Only used when "
            "--use-wf246c-metal-coord is set."
        ),
    )
    p.add_argument(
        "--use-per-bond-f2a", action="store_true",
        help=(
            "Wf-333-V3 P2: replace per-centre F2(a) count with "
            "per-metal-donor-arm-bond count. For cisplatin this "
            "lifts the F2(a) modifier from -1.5 (per-centre) to "
            "-3.0 (per-bond: 2 Pt-N donor arms). PROJECTED lift: "
            "-1.5 kcal/mol additional per Wf-331-R1 §6.3."
        ),
    )
    p.add_argument(
        "--use-distance-score", action="store_true",
        help=(
            "Wf-333-V3 P3: wire the Wf-246-C distance sub-score "
            "from the docked pocket's donor-atom positions to the "
            "metal centre (RDKit ETKDGv3 conformer of original "
            "SMILES). PROJECTED lift: -0.5 to -1.0 kcal/mol per "
            "Wf-331-R1 §6.4."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging.",
    )
    # Wf-336-X4: meeko alternative integration CLI flag.
    p.add_argument(
        "--meeko-only", action="store_true",
        help=(
            "Wf-336-X4: skip Vina CLI invocation entirely. Only the "
            "meeko ligand preparation step runs; every cell is "
            "MEEKO_PREP_ONLY (or STUB_FALLBACK when meeko is missing). "
            "Useful as a smoke test for the meeko alternative path "
            "when Vina CLI is unreachable."
        ),
    )
    # Wf-338-V2: Smina / QuickVina 2 engine recovery paths.
    p.add_argument(
        "--engine", type=str, default=DEFAULT_ENGINE,
        choices=("vina", "smina", "quickvina2", "auto"),
        help=(
            "Wf-338-V2: docking engine selector. "
            "'vina' = canonical AutoDock Vina 1.2.x (default); "
            "'smina' = Forli-lab Smina (metal-coordination-aware scoring); "
            "'quickvina2' = Handoko et al. 2018 ~2x faster Vina-compatible; "
            "'auto' = try vina → smina → quickvina2 in order. "
            "When the requested engine's probe fails, the panel "
            "records 'STUB_FALLBACK' (no fabrication) per AGENTS.md §5.2."
        ),
    )
    p.add_argument(
        "--smina-bin", type=str, default=DEFAULT_SMINA_BIN,
        help=(
            "Wf-338-V2: path to the Smina CLI binary (used when "
            "--engine={smina,auto}). Default: 'smina' (PATH lookup)."
        ),
    )
    p.add_argument(
        "--qvina2-bin", type=str, default=DEFAULT_QUICKVINA2_BIN,
        help=(
            "Wf-338-V2: path to the QuickVina 2 CLI binary (used when "
            "--engine={quickvina2,auto}). Default: 'qvina2' (PATH lookup)."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    smiles_list = read_smiles_csv(args.smiles_csv)

    work_dir: Path
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = args.out_json.parent / "_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(
        smiles_list=smiles_list,
        pocket_pdb_dir=args.pocket_pdb_dir,
        pockets=list(args.pockets),
        vina_bin=args.vina_bin,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_candidates,
        box_size=args.box_size,
        seed=args.seed,
        timeout=args.dock_timeout,
        work_dir=work_dir,
        pt_strip=args.pt_strip,
        use_f2a_metal_ligand=args.use_f2a_metal_ligand,
        f2a_modifier_kcal_mol=args.f2a_modifier_kcal_mol,
        use_wf246c_metal_coord=args.use_wf246c_metal_coord,
        wf246c_kcal_per_unit=args.wf246c_kcal_per_unit,
        use_per_bond_f2a=args.use_per_bond_f2a,
        use_distance_score=args.use_distance_score,
        meeko_only=args.meeko_only,
        engine=args.engine,
        smina_bin=args.smina_bin,
        qvina2_bin=args.qvina2_bin,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=2, default=str)

    n_input = panel["n_input"]
    n_parsed = panel["n_parsed"]
    n_pdbqt_ok = panel["n_pdbqt_ok"]
    n_docked = panel["n_docked"]
    print(
        f"[wf264-vina] n_input={n_input} n_parsed={n_parsed} "
        f"n_pdbqt_ok={n_pdbqt_ok} n_docked={n_docked} "
        f"vina_ok={panel['metadata']['vina_binary_ok']} "
        f"pockets={len(args.pockets)}",
        flush=True,
    )
    # Wf-329-W3: F2(a) cohort summary.
    agg = panel.get("aggregate", {})
    if args.use_f2a_metal_ligand:
        f2a_med = agg.get("f2a_cohort_median")
        f2a_lift = agg.get("f2a_cohort_median_lift")
        f2a_gap = agg.get("f2a_cohort_median_gap_to_industry")
        if f2a_med is not None:
            print(
                f"[wf329w3-f2a] f2a_cohort_median={f2a_med:.4f} "
                f"lift={f2a_lift:+.4f} gap_to_industry={f2a_gap:+.4f} "
                f"(industry_cutoff={agg.get('industry_cutoff_kcal_mol')})",
                flush=True,
            )
        else:
            print(
                "[wf329w3-f2a] no F2(a) scores (no SMILES with "
                "detected covalent-capable centres, or Vina "
                "STUB_FALLBACK across all pockets)",
                flush=True,
            )
    # Wf-333-V2: Wf-246-C cohort summary.
    if args.use_wf246c_metal_coord:
        mc_med = agg.get("metal_coord_cohort_median")
        mc_lift_vina = agg.get("metal_coord_cohort_median_lift_vs_vina")
        mc_lift_f2a = agg.get("metal_coord_cohort_median_lift_vs_f2a")
        mc_gap = agg.get("metal_coord_cohort_median_gap_to_industry")
        if mc_med is not None:
            print(
                f"[wf333v2-wf246c] metal_coord_cohort_median={mc_med:.4f} "
                f"lift_vs_vina={mc_lift_vina:+.4f} "
                f"lift_vs_f2a={mc_lift_f2a:+.4f} "
                f"gap_to_industry={mc_gap:+.4f}",
                flush=True,
            )
        else:
            print(
                "[wf333v2-wf246c] no metal-coord scores (no SMILES "
                "with metal centres, or Vina STUB_FALLBACK across "
                "all pockets)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())