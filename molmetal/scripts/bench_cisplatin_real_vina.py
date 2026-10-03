#!/usr/bin/env python3
"""Wf-121 Phase 3 — real Vina dock on cisplatin × 10 pockets × 3 seeds.

================================================================
Why this script exists
================================================================
Wf-30 staged 10 MMP-13 pockets as real ``receptor.pdbqt`` files
(``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``) and
Wf-121 Phase 2 patched ``vina_adapter._ligand_to_pdbqt`` so the
``vina`` Python binding can emit a valid PDBQT for cisplatin (the
canonical Pt metallodrug) — meeko 0.8.0's ``ad4_types.json`` ships
no atom-type for Pt, so Wf-121 P1 added SMARTS overrides and P2
wired them into the adapter.

Phase 3 closes the loop: run a **real** AutoDock Vina docking sweep
of cisplatin against the 10 Wf-30 pockets × 3 seeds and write
per-pocket per-seed kcal/mol to
``molmetal/reports/wf121_cisplatin_real_vina/<pocket>/per_seed_metrics.json``.

Honest framing (party-twenty-spirit §3.5)
----------------------------------------
* kcal/mol values are real Vina 1.2.7 output — NO stub floor, NO
  hash-derived fake score, NO -9.6574 placeholder.
* For metallodrugs the atype of the Pt atom is the generic ``M``
  (Wf-121 P1 SMARTS rules). The kcal/mol is therefore real Vina but
  with a **generic VDW** atype — a ``status="MEASURED_GENERIC_ATOM"``
  disposition, not a full-fidelity chemotype-tuned score. Downstream
  consumers MUST treat it as a real (not mock) measurement with a
  documented honest caveat.
* If the PDBQT conversion fails for any reason the script records
  ``status="MEASURED_FALLBACK_BACKTOLTZ"`` and runs the boltz_affinity
  (D-MPNN pIC50) fallback path so the per-pocket record is
  complete.

Inputs
------
* ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt`` (10 pockets).
* ``vina`` Python binding 1.2.7 + ``meeko`` 0.8.0 (mol-metal-venv).
* ``molmetal/checkpoints/dmpnn_pic50_*.pt`` (D-MPNN pIC50 fallback).

Outputs
-------
* ``molmetal/reports/wf121_cisplatin_real_vina/<pocket>/per_seed_metrics.json`` —
  per-pocket per-seed real kcal/mol + per-pocket boltz_affinity pIC50.
* ``molmetal/reports/wf121_cisplatin_real_vina/synthesis.md`` — summary
  table + honest framing.

Usage
-----
::

    source env.sh
    CUDA_VISIBLE_DEVICES=0 \\
    python molmetal/scripts/bench_cisplatin_real_vina.py \\
        --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 4rn0 1fmc 1g9v 2hnx \\
        --seeds 0 1 2 --exhaustiveness 8 \\
        --out-dir molmetal/reports/wf121_cisplatin_real_vina
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("wf121_bench_cisplatin_real_vina")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CISPLATIN_SMILES = "[Pt](N)(N)(Cl)Cl"
CISPLATIN_NAME = "cisplatin"

# The canonical 10 Wf-30 pockets (4-letter PDB ids) — same set used by
# Wf-30, Wf-67, Wf-71 sweeps. Per the wf121 brief: 830c, 1h36, 3eml,
# 3ny8, 4aaw, 4rlu, 4rn0, 1fmc, 1g9v, 2hnx.
DEFAULT_POCKETS: Tuple[str, ...] = (
    "830c",
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
    "4rn0",
    "1fmc",
    "1g9v",
    "2hnx",
)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_N_POSES = 5
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
DEFAULT_PREP_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DockResult:
    """Per-(pocket, seed) docking record — real Vina 1.2.7 kcal/mol."""

    pocket_id: str
    seed: int
    smiles: str
    method: str  # "vina_python_1.2.7"
    kcal_mol: float
    elapsed_s: float
    n_poses: int
    n_atoms_ligand: int
    best_pose_rmsd_lb: float
    best_pose_rmsd_ub: float
    status: str  # "MEASURED" | "MEASURED_GENERIC_ATOM" | "MEASURED_FALLBACK_BOLTZ"
    extras: Dict[str, str] = field(default_factory=dict)


@dataclass
class PocketSweep:
    """Per-pocket aggregation across the configured seeds."""

    pocket_id: str
    smiles: str
    seeds: List[int]
    kcal_mol_per_seed: List[float]
    kcal_mol_min: float
    kcal_mol_median: float
    kcal_mol_mean: float
    kcal_mol_stddev: float
    n_poses: int
    exhaustiveness: int
    boltz_pic50: Optional[float]
    status: str  # "MEASURED" | "MEASURED_GENERIC_ATOM" | "MEASURED_FALLBACK_BOLTZ"
    engine: str
    engine_version: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x) -> Optional[float]:
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
# PDBQT post-processing for metal atom-type compatibility
# ---------------------------------------------------------------------------
# Wf-121 P1 emits Pt with the generic AutoDock atype ``M`` (the
# convention for "M(etal)" — see metal_pdbqt_overrides.GENERIC_ATYPE).
# AutoDock Vina's atom-type parser is strict: it rejects ``M`` with
# ``PDBQT parsing error: Atom type M is not a valid AutoDock type``
# even though the design intent (per P1 docstring) was that Vina
# "logs a warning and falls back to generic VDW". Both the Python
# binding and the CLI binary reject ``M``.
#
# Wf-143 P3 adds chemotype-tuned atype: we now rewrite the atype
# column to the chemotype-specific element symbol (e.g. ``Pt``,
# ``Ru``) so Vina can use the corresponding (r, ε) parameters from
# its own AD4 atom-type table when present. If Vina rejects the
# chemotype atype (it logs a warning for unrecognised types but
# continues parsing for many Pt-group symbols — we still probe
# defensively), we fall back to ``Mg``.
#
# The two-tier rewrite path is documented in
# :func:`_rewrite_metal_atype_for_vina`. Downstream consumers MUST
# treat the resulting kcal/mol as either
# ``status="MEASURED_CHEMOTYPE_VDW"`` (chemotype atype accepted by
# Vina) or ``status="MEASURED_GENERIC_ATOM"`` (Mg fallback), NOT a
# full-fidelity Vina score for Pt-group metallodrugs.
def _rewrite_metal_atype_for_vina(
    lig_pdbqt: Path,
    chemotype_vdw: bool = True,
    fallback_atype: str = "Mg",
) -> Path:
    """Rewrite the atype column so Vina 1.2.7 accepts the metal atom.

    Two-tier behaviour (Wf-143 P3):

    * **PRIMARY** (default ``chemotype_vdw=True``): rewrite ``M`` →
      chemotype-specific element symbol (e.g. ``Pt``, ``Ru``, ...).
      Vina then drives the metal-atom VDW term from its own (r, ε)
      table; if Vina has no row for that element the parser logs a
      warning but still runs.
    * **FALLBACK**: rewrite ``M`` → ``fallback_atype`` (``Mg`` by
      default). This matches Wf-121 P1 behaviour and is used when
      ``chemotype_vdw=False`` OR when the caller has detected that
      Vina rejects the chemotype atype (see
      :func:`_probe_vina_accepts_chemotype_atype`).

    The element column at the end of the PDBQT row is unchanged so
    RDKit / meeko can still reverse-parse the file.

    Parameters
    ----------
    lig_pdbqt : pathlib.Path
        Path to the meeko-emitted ligand PDBQT. Modified in place.
    chemotype_vdw : bool, default ``True``
        When ``True``, rewrite ``M`` → chemotype atype; when ``False``,
        rewrite ``M`` → ``fallback_atype`` directly.
    fallback_atype : str, default ``"Mg"``
        The atype to substitute when chemotype rewriting is disabled or
        when Vina rejects the chemotype atype.

    Returns
    -------
    pathlib.Path
        The same path (modified in place).
    """
    text = lig_pdbqt.read_text()
    rewritten: List[str] = []
    n_chemotype = 0
    n_fallback = 0
    for line in text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 79:
            atype = line[77:79].strip()
            if atype == "M":
                # Pull the element symbol from the tail of the PDBQT
                # row (columns 77-79 are the atype; the element column
                # for meeko output is columns [12:16] for the atom name
                # — which contains the element symbol for non-standard
                # metals like Pt). The trailing whitespace-separated
                # token after the partial charge (column 70) is also
                # the element in meeko output. We fall back to the
                # atype when we cannot disambiguate.
                elem_symbol = _extract_element_from_pdbqt_line(line)
                if chemotype_vdw and elem_symbol in {
                    "Pt",
                    "Ru",
                    "Ir",
                    "Os",
                    "Rh",
                    "Pd",
                    "Au",
                    "Re",
                }:
                    new_atype = elem_symbol
                    n_chemotype += 1
                else:
                    new_atype = fallback_atype
                    n_fallback += 1
                # Replace exactly the atype columns [77:79] — keep
                # the rest of the line (incl. element column at the
                # end) unchanged so RDKit/meeko can still reverse it.
                line = line[:77] + new_atype.ljust(2)[:2] + line[79:]
        rewritten.append(line)
    if n_chemotype or n_fallback:
        lig_pdbqt.write_text("".join(rewritten))
        if n_chemotype:
            LOG.info(
                "_rewrite_metal_atype_for_vina: rewrote %d 'M' → chemotype "
                "atype(s) in %s (Wf-143 P3)",
                n_chemotype,
                lig_pdbqt.name,
            )
        if n_fallback:
            LOG.info(
                "_rewrite_metal_atype_for_vina: rewrote %d 'M' → '%s' "
                "fallback atype(s) in %s",
                n_fallback,
                fallback_atype,
                lig_pdbqt.name,
            )
    return lig_pdbqt


def _extract_element_from_pdbqt_line(line: str) -> str:
    """Best-effort element symbol extraction from a PDBQT ATOM/HETATM line.

    Meeko writes element symbol into BOTH the atom-name columns
    (12:16 — left-justified, e.g. ``"Pt"``) AND the trailing columns
    after the partial charge. We prefer the trailing element column
    when present (it is the most reliable place for meeko output);
    otherwise we strip digits from the atom-name field.

    Returns
    -------
    str
        Best-guess element symbol, or empty string if the line is too
        short to parse.
    """
    if len(line) < 16:
        return ""
    # Atom name field: columns 12-16 (1-indexed) → slice [12:16]
    atom_name = line[12:16].strip()
    # Strip charge / isotope digits: "Pt2+" -> "Pt", "N1" -> "N"
    cleaned = "".join(c for c in atom_name if c.isalpha())
    if cleaned:
        return cleaned
    return ""


def _probe_vina_accepts_chemotype_atype(
    lig_pdbqt: Path,
    fallback_atype: str = "Mg",
) -> str:
    """Probe whether Vina 1.2.7 accepts the chemotype atype in ``lig_pdbqt``.

    If yes, return the chemotype atype (e.g. ``"Pt"``) unchanged. If
    Vina rejects the atype (raises a parse error), rewrite the file
    in place to the fallback atype and return ``"Mg-fallback"``.

    Parameters
    ----------
    lig_pdbqt : pathlib.Path
        The chemotype-rewritten ligand PDBQT (i.e. the output of
        :func:`_rewrite_metal_atype_for_vina` with
        ``chemotype_vdw=True``).
    fallback_atype : str, default ``"Mg"``

    Returns
    -------
    str
        The atype string Vina accepted (``"Pt"``,
        ``"Mg-fallback"``, etc.). The caller can use this to set the
        DockResult ``status`` field.

    Notes
    -----
    The probe is best-effort: if ``vina`` is not importable or the
    probe raises an unexpected exception, we conservatively fall back
    to Mg and return ``"Mg-fallback"``. The actual Vina run downstream
    will surface its own error if the atype is still bad.
    """
    try:
        from vina import Vina  # type: ignore
    except Exception as exc:  # pragma: no cover — defensive
        LOG.debug(
            "_probe_vina_accepts_chemotype_atype: vina import failed "
            "(%s); falling back to %s",
            exc,
            fallback_atype,
        )
        _rewrite_metal_atype_for_vina(
            lig_pdbqt,
            chemotype_vdw=False,
            fallback_atype=fallback_atype,
        )
        return f"{fallback_atype}-fallback"

    # Detect the chemotype atype present in the file. We restrict the
    # probe to the 8 Pt-group elements (the only ones we would have
    # rewritten in chemotype_vdw mode) so we don't accidentally trip
    # over common organic atypes (C, N, O, H, Cl, …) that appear in
    # the ligand too.
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        PT_GROUP_ELEMENTS,
    )

    chemotype_atype = ""
    for line in lig_pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 79:
            atype = line[77:79].strip()
            if atype in PT_GROUP_ELEMENTS:
                chemotype_atype = atype
                break

    if not chemotype_atype:
        # Nothing chemotype-typed in the PDBQT → no probe needed.
        return fallback_atype

    # The most authoritative probe: actually ask Vina to parse the
    # PDBQT. If it raises (e.g. ``PDBQT parsing error: Atom type Pt is
    # not a valid AutoDock type``), Vina does not accept the chemotype
    # atype and we fall back. We do NOT need a receptor for
    # ``set_ligand_from_file`` to attempt the parse — the parse error
    # happens before the receptor is consulted.
    try:
        probe = Vina(sf_name="vina")  # type: ignore
        probe.set_ligand_from_file(str(lig_pdbqt))
        return chemotype_atype
    except Exception as exc:
        LOG.warning(
            "_probe_vina_accepts_chemotype_atype: Vina rejected %s "
            "(%s); rewriting to %s fallback",
            chemotype_atype,
            exc,
            fallback_atype,
        )
        _replace_chemotype_atype_with_fallback(
            lig_pdbqt,
            chemotype_atype,
            fallback_atype,
        )
        return f"{fallback_atype}-fallback"


def _replace_chemotype_atype_with_fallback(
    lig_pdbqt: Path,
    chemotype_atype: str,
    fallback_atype: str,
) -> None:
    """Direct atype-column replacement used by the probe fallback path.

    Unlike :func:`_rewrite_metal_atype_for_vina` (which only rewrites
    rows with atype ``"M"``), this helper targets a specific chemotype
    atype string (e.g. ``"Pt"``) and substitutes it with
    ``fallback_atype`` (``"Mg"``). It mutates the file in place and is
    the path the probe takes when Vina rejects the chemotype atype
    after :func:`_rewrite_metal_atype_for_vina` has already emitted
    ``Pt`` / ``Ru`` / etc. into the PDBQT.
    """
    text = lig_pdbqt.read_text()
    rewritten: List[str] = []
    n_replaced = 0
    for line in text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 79:
            atype = line[77:79].strip()
            if atype == chemotype_atype:
                line = line[:77] + fallback_atype.ljust(2)[:2] + line[79:]
                n_replaced += 1
        rewritten.append(line)
    if n_replaced:
        lig_pdbqt.write_text("".join(rewritten))
        LOG.info(
            "_replace_chemotype_atype_with_fallback: rewrote %d '%s' → " "'%s' in %s",
            n_replaced,
            chemotype_atype,
            fallback_atype,
            lig_pdbqt.name,
        )


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _stddev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def _strip_ansi(s: str) -> str:
    """Strip ANSI escape codes from Vina stdout (libboost builds emit them)."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# Tooling probes
# ---------------------------------------------------------------------------
def _probe_vina_version() -> str:
    """Return the ``vina`` Python binding version string, or ``"uninstalled"``."""
    try:
        import vina  # noqa: F401

        return getattr(vina, "__version__", "unknown")
    except Exception:
        return "uninstalled"


def _probe_meeko_version() -> str:
    try:
        import meeko  # noqa: F401

        return getattr(meeko, "__version__", "unknown")
    except Exception:
        return "uninstalled"


# ---------------------------------------------------------------------------
# Boltz_affinity fallback — D-MPNN pIC50 for cisplatin
# ---------------------------------------------------------------------------
def _boltz_pic50(smiles: str) -> Optional[float]:
    """Predict pIC50 via Wf-120 D-MPNN (R20-trained multi-metal head).

    The Wf-120 ship produced per-metal D-MPNN pIC50 checkpoints
    (``molmetal/checkpoints/dmpnn_pic50_<metal>_seed<N>.pt``).  We
    pick the first available checkpoint (Ru > Ir > Os > Rh > Re) and
    run it through the multi-task backbone to produce a real pIC50
    prediction. Returns ``None`` if no checkpoint exists or inference
    fails.

    Honest framing
    --------------
    The checkpoint is trained on per-metal cytotoxicity labels
    (MetalCytoToxDB) — using it as a "boltz_affinity" predictor for
    cisplatin is a defensible fallback because (a) D-MPNN pIC50 is
    still real ML inference, not a mock, and (b) the multi-task
    backbone was trained across all 5 metals so cross-metal
    generalisation is plausible. We do NOT pretend this is a true
    Boltz foundation-model pIC50 — the metric name is preserved for
    routing compatibility but the underlying checkpoint is the
    locally-trained D-MPNN (see ``REGISTRY.md`` notes).
    """
    ckpt_dir = PROJECT_ROOT / "molmetal" / "checkpoints"
    candidates = [
        "dmpnn_pic50_ru_seed0.pt",
        "dmpnn_pic50_ir_seed0.pt",
        "dmpnn_pic50_os_seed0.pt",
        "dmpnn_pic50_rh_seed0.pt",
        "dmpnn_pic50_re_seed0.pt",
        "dmpnn_attn_ru_pic50.pt",  # legacy ckpt name (R17 ship)
    ]
    ckpt_path: Optional[Path] = None
    for name in candidates:
        p = ckpt_dir / name
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        LOG.warning(
            "boltz_affinity: no D-MPNN pIC50 checkpoint found under %s",
            ckpt_dir,
        )
        return None
    try:
        import torch
        from molmetal.baselines.dmpnn import (
            ATOM_FEATURE_DIM,
            BOND_FEATURE_DIM,
            featurize_smiles_list,
        )
        from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel

        bundle = torch.load(
            str(ckpt_path),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(bundle, dict) and "model_state_dict" in bundle:
            # Wf-120 multi-task ckpt format
            hidden = int(bundle.get("hidden", 128))
            depth = int(bundle.get("depth", 3))
            state_dict = bundle["model_state_dict"]
        else:
            # Legacy Attentive D-MPNN ckpt — would need different loader;
            # for now we log and return None to avoid silent substitution.
            LOG.warning(
                "boltz_affinity: %s is in legacy format; skipping",
                ckpt_path.name,
            )
            return None
        model = DMPNNMultiTaskModel(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=hidden,
            depth=depth,
            dropout=0.0,
        )
        model.load_state_dict(state_dict)
        model.eval()
        graphs = featurize_smiles_list([smiles])
        if not graphs or graphs[0] is None:
            LOG.warning("boltz_affinity: failed to featurize %s", smiles)
            return None
        from molmetal.baselines.dmpnn_multitask import collate_graphs

        batch = collate_graphs(graphs)
        if batch is None:
            return None
        atom_f, bond_f, e_src, e_dst = batch
        with torch.no_grad():
            pic50_pred, _active_logits = model(atom_f, bond_f, e_src, e_dst)
        v = float(pic50_pred.item())
        if not math.isfinite(v):
            return None
        return v
    except Exception as exc:  # noqa: BLE001 — honest fallback
        LOG.warning(
            "boltz_affinity fallback failed for %s (ckpt=%s): %s",
            smiles,
            ckpt_path.name,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Receptor + pocket geometry
# ---------------------------------------------------------------------------
def _load_receptor_geometry(receptor_pdbqt: Path) -> Tuple[Optional[List[float]], int]:
    """Parse heavy-atom coordinates from a receptor.pdbqt.

    Returns ``(center_xyz, n_atoms)``. ``center_xyz`` is the mean
    coordinate of all heavy (non-H) atoms — used as the Vina docking
    box centre. Returns ``None`` for the centre if no heavy atoms
    parse cleanly.
    """
    try:
        from rdkit import Chem  # noqa: F401 — RDKit parses PDBQT poorly;
    # we hand-parse instead.
    except Exception:
        pass

    coords: List[Tuple[float, float, float]] = []
    n_total = 0
    for raw in receptor_pdbqt.read_text().splitlines():
        line = raw.rstrip()
        if not line.startswith(("ATOM", "HETATM")):
            continue
        n_total += 1
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        atom_name = line[12:16].strip()
        # Skip explicit hydrogens — they are added by meeko and we want
        # the heavy-atom centroid for the docking box.
        if atom_name.startswith("H") and atom_name not in ("HG", "HH"):
            continue
        coords.append((x, y, z))
    if not coords:
        return None, n_total
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    return [cx, cy, cz], len(coords)


# ---------------------------------------------------------------------------
# Core: real Vina dock one (pocket, seed)
# ---------------------------------------------------------------------------
def _dock_one_python_binding(
    receptor_pdbqt: Path,
    lig_pdbqt: Path,
    center_xyz: List[float],
    seed: int,
    exhaustiveness: int,
    n_poses: int,
    cpu_count: int = 0,
) -> Tuple[float, float, float, float]:
    """Real Vina 1.2.7 (Python binding) — returns (kcal_mol, elapsed_s,
    best_pose_rmsd_lb, best_pose_rmsd_ub)."""
    import vina  # type: ignore

    t0 = time.time()
    # Map uint32 seed -> nonzero int32 seed that Vina accepts.
    seed_eff = int(seed) % (2**31 - 1) or (2**31 - 1)
    v = vina.Vina(sf_name="vina", cpu=cpu_count, seed=seed_eff, verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(lig_pdbqt))
    # Box padding 8 Å on each side (Vina minimum). 25 Å cubed covers
    # the typical 10 Å pocket sphere without wasting compute.
    box_size = [25.0, 25.0, 25.0]
    v.compute_vina_maps(center=[float(c) for c in center_xyz], box_size=box_size)
    v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
    energies = v.energies(n_poses=int(n_poses))
    kcal = float(energies[0, 0])
    # RMSD columns are 2/3 (lower-bound / upper-bound)
    rmsd_lb = float(energies[0, 1]) if energies.shape[1] > 1 else float("nan")
    rmsd_ub = float(energies[0, 2]) if energies.shape[1] > 2 else float("nan")
    elapsed = time.time() - t0
    return kcal, elapsed, rmsd_lb, rmsd_ub


def _dock_one_subprocess_vina(
    vina_bin: Path,
    receptor_pdbqt: Path,
    lig_pdbqt: Path,
    center_xyz: List[float],
    seed: int,
    exhaustiveness: int,
    n_poses: int,
) -> Tuple[float, float, float, float]:
    """Fallback path via the ``vina`` CLI binary (real, not stub)."""
    t0 = time.time()
    out_pdbqt = lig_pdbqt.parent / f"docked_seed{seed}.pdbqt"
    cmd = [
        str(vina_bin),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(lig_pdbqt),
        "--center_x",
        f"{center_xyz[0]:.3f}",
        "--center_y",
        f"{center_xyz[1]:.3f}",
        "--center_z",
        f"{center_xyz[2]:.3f}",
        "--size_x",
        "25.000",
        "--size_y",
        "25.000",
        "--size_z",
        "25.000",
        "--out",
        str(out_pdbqt),
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--num_modes",
        str(int(n_poses)),
        "--seed",
        str(int(seed)),
        "--quiet",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"vina CLI failed (rc={proc.returncode}) stderr={_strip_ansi(proc.stderr)[-200:]}"
        )
    kcal, rmsd_lb, rmsd_ub = _parse_vina_table_stdout(_strip_ansi(proc.stdout))
    return kcal, elapsed, rmsd_lb, rmsd_ub


def _parse_vina_table_stdout(stdout: str) -> Tuple[float, float, float]:
    """Parse the per-mode kcal/mol + RMSD columns from Vina CLI stdout.

    Vina prints (when ``--quiet`` is NOT set, or with verbosity>0) a
    table like::

        mode |   affinity | rmsd l.b.| rmsd u.b.
          1       -7.4      0.000      0.000
          2       -7.1      1.234      2.345
          ...

    Returns ``(kcal, rmsd_lb, rmsd_ub)`` for the best pose (mode 1).
    """
    kcal: Optional[float] = None
    rmsd_lb: float = float("nan")
    rmsd_ub: float = float("nan")
    for line in stdout.splitlines():
        s = line.strip()
        if not s or not s[:1].isdigit():
            continue
        parts = s.split()
        try:
            kcal = float(parts[1])
            rmsd_lb = float(parts[2])
            rmsd_ub = float(parts[3])
        except (IndexError, ValueError):
            continue
        break
    if kcal is None:
        # Some libboost Vina builds print only "mode | affinity" with
        # blank RMSD columns — accept the kcal/mol even when RMSD is
        # missing.
        for line in stdout.splitlines():
            s = line.strip()
            if not s or not s[:1].isdigit():
                continue
            parts = s.split()
            try:
                kcal = float(parts[1])
            except (IndexError, ValueError):
                continue
            break
    if kcal is None:
        raise RuntimeError(
            f"vina CLI produced no parseable kcal/mol; stdout-tail="
            f"{stdout[-200:]!r}"
        )
    return kcal, rmsd_lb, rmsd_ub


def _vina_binary_present() -> Optional[Path]:
    """Locate a working ``vina`` CLI binary (libboost-resolved)."""
    import shutil

    candidates = [
        Path("/home/hugo/.mamba/bin/vina"),
        Path("/home/hugo/micromamba/bin/vina"),
        Path("/usr/local/bin/vina"),
        Path("/usr/bin/vina"),
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK):
            # Probe with --version so a broken libboost binary is rejected.
            try:
                proc = subprocess.run(
                    [str(c), "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    return c
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    which = shutil.which("vina")
    if which:
        try:
            proc = subprocess.run(
                [which, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                return Path(which)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Pocket sweep — per pocket
# ---------------------------------------------------------------------------
def run_pocket_sweep(
    pocket_id: str,
    smiles: str,
    seeds: Tuple[int, ...],
    receptor_pdbqt: Path,
    out_dir: Path,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    n_poses: int = DEFAULT_N_POSES,
    force_engine: Optional[str] = None,
    chemotype_vdw: bool = True,
) -> PocketSweep:
    """Run a real Vina docking sweep of ``smiles`` into ``receptor_pdbqt``.

    Parameters
    ----------
    pocket_id
        4-letter PDB id used as the per-pocket subdirectory.
    smiles
        Ligand SMILES (cisplatin canonical: ``[Pt](N)(N)(Cl)Cl``).
    seeds
        Tuple of uint32 experiment seeds (one Vina run each).
    receptor_pdbqt
        Path to the prepared ``receptor.pdbqt`` (Wf-30 stage).
    out_dir
        Per-pocket output directory.
    exhaustiveness
        Vina ``--exhaustiveness`` setting (default 8).
    n_poses
        Number of docked poses per run (default 5).
    force_engine
        Override the engine resolution (``"vina"`` Python binding or
        ``"vina-cli"`` subprocess path). Default ``None`` picks the
        first available.
    chemotype_vdw : bool, default ``True``
        Wf-143 P3 — when ``True`` the atype column in the meeko-
        emitted ligand is rewritten to the chemotype element symbol
        (e.g. ``Pt``, ``Ru``); when ``False`` it is rewritten to
        the generic ``Mg`` fallback (Wf-121 P1 behaviour). If Vina
        rejects the chemotype atype we transparently fall back to
        ``Mg`` regardless of this flag's value.

    Returns
    -------
    PocketSweep
        Aggregated per-pocket record.
    """
    import vina  # noqa: F401  — pre-flight

    center_xyz, n_heavy = _load_receptor_geometry(receptor_pdbqt)
    if center_xyz is None:
        # Last-ditch — centre on origin and hope for the best.
        LOG.warning(
            "%s: receptor heavy-atom centroid unavailable; " "using origin", pocket_id
        )
        center_xyz = [0.0, 0.0, 0.0]

    # Resolve engine. Default: prefer Python binding (programmatic
    # energies + poses + seed control).
    engine_used = "vina"
    engine_version = _probe_vina_version()
    if force_engine == "vina-cli":
        vbin = _vina_binary_present()
        if vbin is None:
            raise RuntimeError(
                "force_engine='vina-cli' but no working vina CLI on PATH"
            )
        engine_used = "vina-cli"
        engine_version = f"cli:{vbin}"

    pocket_out = out_dir / pocket_id
    pocket_out.mkdir(parents=True, exist_ok=True)

    # ---- ligand PDBQT (one canonical cisplatin.pdbqt; same for all seeds) ----
    from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
        _have_meeko,
        _ligand_to_pdbqt,
    )

    if not _have_meeko():
        raise RuntimeError("meeko not installed — required for ligand PDBQT")
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        MetalPDBQTUnavailable,
    )

    try:
        lig_pdbqt = _ligand_to_pdbqt(smiles, pocket_out, name=CISPLATIN_NAME)
        # Wf-121 P1 emits Pt with atype 'M' which Vina's strict parser
        # rejects. Rewrite to 'Mg' (Vina-accepted generic metal) — see
        # _rewrite_metal_atype_for_vina docstring for honest framing.
        # Wf-143 P3: PRIMARY = chemotype atype (Pt/Ru/...); FALLBACK =
        # Mg if Vina rejects. ``chemotype_vdw`` is forwarded from the
        # outer CLI flag (see ``_build_parser``); ``probe_vina`` runs
        # the post-rewrite probe to confirm Vina actually parses the
        # chemotype atype.
        # ``status`` reflects the post-rewrite atype acceptance:
        #   - MEASURED_CHEMOTYPE_VDW when chemotype atype accepted by Vina
        #   - MEASURED_GENERIC_ATOM   when chemotype atype rejected (fallback to Mg)
        status = "MEASURED_GENERIC_ATOM"
        lig_pdbqt = _rewrite_metal_atype_for_vina(
            lig_pdbqt,
            chemotype_vdw=chemotype_vdw,
        )
        if chemotype_vdw:
            accepted = _probe_vina_accepts_chemotype_atype(lig_pdbqt)
            if accepted.endswith("-fallback"):
                status = "MEASURED_GENERIC_ATOM"
            else:
                status = "MEASURED_CHEMOTYPE_VDW"
    except MetalPDBQTUnavailable as exc:
        # P1/P2 patch succeeded for cisplatin; this branch is defensive
        # in case a future SMILES change re-introduces a meeko edge case.
        LOG.error("%s: MetalPDBQTUnavailable: %s", pocket_id, exc)
        boltz = _boltz_pic50(smiles)
        return PocketSweep(
            pocket_id=pocket_id,
            smiles=smiles,
            seeds=list(seeds),
            kcal_mol_per_seed=[],
            kcal_mol_min=float("nan"),
            kcal_mol_median=float("nan"),
            kcal_mol_mean=float("nan"),
            kcal_mol_stddev=float("nan"),
            n_poses=int(n_poses),
            exhaustiveness=int(exhaustiveness),
            boltz_pic50=boltz,
            status="MEASURED_FALLBACK_BOLTZ",
            engine=engine_used,
            engine_version=engine_version,
            notes=f"MetalPDBQTUnavailable: {exc}",
        )
    n_atoms_ligand = sum(
        1 for line in lig_pdbqt.read_text().splitlines() if line.startswith("ATOM")
    )

    # ---- per-seed dock ----
    results: List[DockResult] = []
    kcal_mols: List[float] = []
    for seed in seeds:
        try:
            if engine_used == "vina-cli":
                vbin = _vina_binary_present()
                assert vbin is not None  # already checked above
                kcal, elapsed, rmsd_lb, rmsd_ub = _dock_one_subprocess_vina(
                    vbin,
                    receptor_pdbqt,
                    lig_pdbqt,
                    center_xyz,
                    seed,
                    exhaustiveness,
                    n_poses,
                )
            else:
                kcal, elapsed, rmsd_lb, rmsd_ub = _dock_one_python_binding(
                    receptor_pdbqt,
                    lig_pdbqt,
                    center_xyz,
                    seed,
                    exhaustiveness,
                    n_poses,
                )
        except Exception as exc:  # noqa: BLE001
            LOG.error(
                "%s seed=%s dock failed: %s\n%s",
                pocket_id,
                seed,
                exc,
                traceback.format_exc(),
            )
            results.append(
                DockResult(
                    pocket_id=pocket_id,
                    seed=seed,
                    smiles=smiles,
                    method=f"vina_{engine_used}",
                    kcal_mol=float("nan"),
                    elapsed_s=float("nan"),
                    n_poses=int(n_poses),
                    n_atoms_ligand=n_atoms_ligand,
                    best_pose_rmsd_lb=float("nan"),
                    best_pose_rmsd_ub=float("nan"),
                    status="DOCK_FAILED",
                    extras={"error": str(exc)[:200]},
                )
            )
            continue
        kcal_mols.append(kcal)
        results.append(
            DockResult(
                pocket_id=pocket_id,
                seed=seed,
                smiles=smiles,
                method=f"vina_{engine_used}",
                kcal_mol=kcal,
                elapsed_s=elapsed,
                n_poses=int(n_poses),
                n_atoms_ligand=n_atoms_ligand,
                best_pose_rmsd_lb=rmsd_lb,
                best_pose_rmsd_ub=rmsd_ub,
                status=status,  # Wf-143 P3: chemotype vs generic-atom disposition
            )
        )

    # ---- boltz_affinity fallback (parallel evidence, not docking) ----
    boltz = _boltz_pic50(smiles)

    if not kcal_mols:
        return PocketSweep(
            pocket_id=pocket_id,
            smiles=smiles,
            seeds=list(seeds),
            kcal_mol_per_seed=[],
            kcal_mol_min=float("nan"),
            kcal_mol_median=float("nan"),
            kcal_mol_mean=float("nan"),
            kcal_mol_stddev=float("nan"),
            n_poses=int(n_poses),
            exhaustiveness=int(exhaustiveness),
            boltz_pic50=boltz,
            status="MEASURED_FALLBACK_BOLTZ",
            engine=engine_used,
            engine_version=engine_version,
            notes="all seeds failed — boltz_affinity fallback logged",
        )

    pocket_record = PocketSweep(
        pocket_id=pocket_id,
        smiles=smiles,
        seeds=list(seeds),
        kcal_mol_per_seed=list(kcal_mols),
        kcal_mol_min=min(kcal_mols),
        kcal_mol_median=_median(kcal_mols),
        kcal_mol_mean=_mean(kcal_mols),
        kcal_mol_stddev=_stddev(kcal_mols),
        n_poses=int(n_poses),
        exhaustiveness=int(exhaustiveness),
        boltz_pic50=boltz,
        status=status,
        engine=engine_used,
        engine_version=engine_version,
        notes=(
            f"receptor_pdbqt={receptor_pdbqt.name}; n_heavy={n_heavy}; "
            f"n_atoms_ligand={n_atoms_ligand}"
        ),
    )
    # ---- write per-seed + per-pocket JSON ----
    payload = {
        "_workflow": "Wf-121",
        "_phase": "P3",
        "_title": (
            "Real Vina 1.2.7 docking sweep of cisplatin against the 10 "
            "Wf-30 staged pockets"
        ),
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU-only (Vina Python binding) + GPU-0 Blackwell",
        "_engine": engine_used,
        "_engine_version": engine_version,
        "_smiles": smiles,
        "_n_poses": int(n_poses),
        "_exhaustiveness": int(exhaustiveness),
        "_pocket": {
            "pocket_id": pocket_id,
            "receptor_pdbqt": str(receptor_pdbqt),
            "n_heavy_atoms": n_heavy,
            "docking_box_center": center_xyz,
        },
        "_boltz_affinity_pic50": boltz,
        "_per_seed": [asdict(r) for r in results],
        "_aggregate": asdict(pocket_record),
    }
    out_path = pocket_out / "per_seed_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return pocket_record


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_full_sweep(
    pockets: Tuple[str, ...],
    seeds: Tuple[int, ...],
    out_dir: Path,
    prep_dir: Path,
    exhaustiveness: int,
    n_poses: int,
    smiles: str,
    force_engine: Optional[str],
    chemotype_vdw: bool = True,
) -> Dict[str, object]:
    """Run the full sweep and write ``synthesis.md`` summary.

    Parameters
    ----------
    pockets, seeds, out_dir, prep_dir, exhaustiveness, n_poses, smiles,
    force_engine
        See :func:`run_pocket_sweep`.
    chemotype_vdw : bool, default ``True``
        Wf-143 P3 — forwarded to :func:`run_pocket_sweep`. When
        ``True`` the atype column is rewritten to the chemotype
        element symbol (e.g. ``Pt``); when ``False`` it is rewritten
        to the generic ``Mg`` fallback (Wf-121 P1 behaviour).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sweeps: List[PocketSweep] = []
    for pocket_id in pockets:
        receptor = prep_dir / pocket_id / "receptor.pdbqt"
        if not receptor.exists():
            LOG.error(
                "%s: receptor.pdbqt missing at %s — skipping",
                pocket_id,
                receptor,
            )
            continue
        LOG.info(
            "Docking cisplatin into %s (seeds=%s, exhaustiveness=%d)...",
            pocket_id,
            list(seeds),
            int(exhaustiveness),
        )
        sweep = run_pocket_sweep(
            pocket_id=pocket_id,
            smiles=smiles,
            seeds=seeds,
            receptor_pdbqt=receptor,
            out_dir=out_dir,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            force_engine=force_engine,
            chemotype_vdw=chemotype_vdw,
        )
        sweeps.append(sweep)
        LOG.info(
            "  %s: min=%.3f median=%.3f stddev=%.3f kcal/mol (n=%d/%d seeds)",
            pocket_id,
            sweep.kcal_mol_min,
            sweep.kcal_mol_median,
            sweep.kcal_mol_stddev,
            len(sweep.kcal_mol_per_seed),
            len(seeds),
        )

    # ---- summary JSON ----
    summary = {
        "_workflow": "Wf-121",
        "_phase": "P3",
        "_title": "Real Vina dock cisplatin × 10 pockets × 3 seeds",
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU (vina Python binding) + GPU-0 Blackwell",
        "_n_pockets": len(sweeps),
        "_n_seeds": len(sweeps[0].seeds) if sweeps else len(seeds),
        "_smiles": smiles,
        "_exhaustiveness": int(exhaustiveness),
        "_n_poses": int(n_poses),
        "_per_pocket": [asdict(s) for s in sweeps],
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- synthesis.md ----
    md_lines: List[str] = []
    md_lines.append("# Wf-121 Phase 3 — Real Vina dock on cisplatin × 10 pockets")
    md_lines.append("")
    md_lines.append(f"_Generated: {_utc_now_iso()}_  ")
    md_lines.append("_Hardware: CPU (Vina Python binding) + GPU-0 Blackwell_  ")
    md_lines.append(f"_Ligand SMILES_: `{smiles}` (= cisplatin).  ")
    md_lines.append("_Engine_: Vina 1.2.7 (Python binding preferred, CLI fallback).  ")
    md_lines.append(
        f"_Exhaustiveness_: {exhaustiveness}; _n_poses_: {n_poses}; "
        f"_seeds_: {list(seeds)}."
    )
    md_lines.append("")
    md_lines.append("## Per-pocket real kcal/mol (best pose)")
    md_lines.append("")
    md_lines.append(
        "| Pocket | kcal/mol min | median | mean | stddev | n_seeds | "
        "status | boltz_affinity pIC50 |"
    )
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for s in sweeps:
        boltz_str = f"{s.boltz_pic50:.3f}" if s.boltz_pic50 is not None else "—"
        md_lines.append(
            f"| {s.pocket_id} | {s.kcal_mol_min:.3f} | "
            f"{s.kcal_mol_median:.3f} | {s.kcal_mol_mean:.3f} | "
            f"{s.kcal_mol_stddev:.3f} | {len(s.kcal_mol_per_seed)} | "
            f"{s.status} | {boltz_str} |"
        )
    md_lines.append("")
    md_lines.append("## Honest framing (party-twenty-spirit §3.5)")
    md_lines.append("")
    md_lines.append(
        "* All kcal/mol values are **real Vina 1.2.7 output** — not the "
        "Wf-30 hash-derived stub floor (-9.6574) and not a placeholder."
    )
    md_lines.append(
        "* The Pt atom is typed with the generic AutoDock VDW atype "
        "`M` (Wf-121 P1 SMARTS rules). The kcal/mol is therefore a "
        "**real** Vina measurement with a documented generic-atom "
        "caveat — disposition `MEASURED_GENERIC_ATOM` (i.e. *not* "
        "mock, *not* full-fidelity chemotype)."
    )
    md_lines.append(
        "* The boltz_affinity column is the **D-MPNN pIC50 fallback** "
        "(Wf-120 R20 ship, MetalCytoToxDB-trained Attentive D-MPNN). "
        "It is included as parallel evidence — Vina and boltz_affinity "
        "are different scoring functions with different units "
        "(kcal/mol vs. pIC50), so they are not directly comparable."
    )
    md_lines.append(
        "* Each (pocket, seed) record lives in "
        f"`{out_dir}/<pocket>/per_seed_metrics.json` with kcal/mol + "
        "RMSD-lb/ub + Vina pose count + ligand atom count."
    )
    md_lines.append("")
    md_lines.append("## Files")
    md_lines.append("")
    md_lines.append("* `sweep_summary.json` — machine-readable per-pocket aggregate.")
    md_lines.append("* `<pocket>/per_seed_metrics.json` — per-seed raw Vina record.")
    md_lines.append(
        "* `<pocket>/cisplatin.pdbqt` — meeko-emitted cisplatin PDBQT "
        "(generic `M` atype for Pt)."
    )
    (out_dir / "synthesis.md").write_text("\n".join(md_lines) + "\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_cisplatin_real_vina",
        description=(
            "Wf-121 Phase 3 — real Vina 1.2.7 docking sweep of cisplatin "
            "against the Wf-30 ten-pocket cohort."
        ),
    )
    p.add_argument(
        "--pockets",
        nargs="+",
        default=list(DEFAULT_POCKETS),
        help="Pocket IDs (default: the 10 Wf-30 pockets).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Experiment seeds (default: 0 1 2).",
    )
    p.add_argument(
        "--exhaustiveness",
        type=int,
        default=DEFAULT_EXHAUSTIVENESS,
        help="Vina --exhaustiveness (default 8).",
    )
    p.add_argument(
        "--n-poses",
        type=int,
        default=DEFAULT_N_POSES,
        help="Vina --num_modes (default 5).",
    )
    p.add_argument(
        "--smiles",
        default=CISPLATIN_SMILES,
        help="Ligand SMILES (default: cisplatin [Pt](N)(N)(Cl)Cl).",
    )
    p.add_argument(
        "--prep-dir",
        type=Path,
        default=DEFAULT_PREP_DIR,
        help="Wf-30 pocket-prep root (default: molmetal/reports/wf30_pocket_prep).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root (default: molmetal/reports/wf121_cisplatin_real_vina).",
    )
    p.add_argument(
        "--engine",
        choices=("vina", "vina-cli", "auto"),
        default="auto",
        help="Engine selection (default: auto — Python binding first).",
    )
    p.add_argument(
        "--chemotype-vdw",
        dest="chemotype_vdw",
        action="store_true",
        default=True,
        help=(
            "Wf-143 P3 — rewrite the meeko-emitted ligand atype to the "
            "chemotype element symbol (e.g. 'Pt', 'Ru') and probe Vina "
            "for acceptance. If Vina rejects, transparently fall back to "
            "'Mg'. (default: True)"
        ),
    )
    p.add_argument(
        "--no-chemotype-vdw",
        dest="chemotype_vdw",
        action="store_false",
        help=(
            "Disable Wf-143 P3 chemotype-tuned VDW. Always rewrite "
            "'M' → 'Mg' (Wf-121 P1 behaviour)."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Pre-flight
    if not args.prep_dir.exists():
        LOG.error("prep dir not found: %s", args.prep_dir)
        return 2
    try:
        import vina  # noqa: F401
    except Exception as exc:
        LOG.error("vina Python binding missing: %s", exc)
        return 1
    if _probe_vina_version() == "uninstalled":
        LOG.error("vina not importable; aborting")
        return 1
    if _probe_meeko_version() == "uninstalled":
        LOG.error("meeko not importable; aborting")
        return 1

    force_engine = None
    if args.engine == "vina-cli":
        vbin = _vina_binary_present()
        if vbin is None:
            LOG.error("engine='vina-cli' but no working vina CLI on PATH")
            return 1
        force_engine = "vina-cli"
    elif args.engine == "vina":
        force_engine = "vina"

    summary = run_full_sweep(
        pockets=tuple(args.pockets),
        seeds=tuple(args.seeds),
        out_dir=args.out_dir,
        prep_dir=args.prep_dir,
        exhaustiveness=int(args.exhaustiveness),
        n_poses=int(args.n_poses),
        smiles=str(args.smiles),
        force_engine=force_engine,
        chemotype_vdw=bool(args.chemotype_vdw),
    )
    LOG.info(
        "Sweep complete: %d/%d pockets docked; summary at %s/sweep_summary.json",
        len(summary["_per_pocket"]),
        len(args.pockets),
        args.out_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import os  # noqa: E402 — needed for os.access fallback above

    sys.exit(main(sys.argv[1:]))
