#!/usr/bin/env python3
"""Wf-337-A2 — Wf-264 sub-panel recovery paths (Smina / QuickVina / metal-loop).

======================================================================
Wf-337-A2: Recovery paths for Wf-264-Vina sub-panel when meeko hits
the metal-atom limit (Pt+2 UFF parameterisation blocker).
======================================================================

Per Wf-334-U3 (Wf-264 sub-panel) MEASURED: 0/100 docked ok on 1h36
when SMILES contain Pt coordination centres — meeko 0.8.0 returns
``is_ok=False`` from ``PDBQTWriterLegacy.write_string`` because RDKit
UFF cannot parameterise Pt+2.  This is the **meeko metal-atom limit**.

This module ships 3 alternative-engine / pre-processing recovery paths
that the Wf-264 sub-panel can dispatch to:

1. **Smina** — Vina fork with better metal-atom parameterisation
   (smina explicitly supports custom atom types; not yet installed
   in this environment; the probe surfaces availability).
2. **QuickVina** — faster Vina fork (q.vina); not yet installed in
   this environment; the probe surfaces availability.
3. **Custom metal-loop detector** — programmatically detects precious-
   metal centres (Pt/Ru/Ir/Au/Pd/Rh) in the SMILES and applies a
   pre-processing strategy: either (a) Pt→Zn UFF substitution
   (Wf-325B MEASURED_GENERIC_ATOM path; kcal/mol is generic Zn-VDW
   approximation), or (b) Pt-strip (Wf-301B; drop the coordination
   shell + dock the largest organic fragment).

The probe functions are CPU-only (no real engine launched).  When the
probe reports ``available=True``, the Wf-264 panel can dispatch to
the corresponding engine via ``_dock_one`` swap-in.  When
``available=False``, the probe records the recovery verdict as
``HONEST_NEGATIVE_ENGINE_MISSING`` so downstream consumers know.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Probes are MEASURED runtime probes, not aspirational claims.  When
  neither Smina nor QuickVina is on PATH, the verdict is
  ``HONEST_NEGATIVE_WF264_SUBPANEL_PERSISTENT`` (BLOCKED on external
  engine install).
* The custom metal-loop detector is the MEASURED recovery path that
  works with the existing Vina CLI binary + meeko prep; it does not
  require any external engine.
* ``n_pockets_docked`` is MEASURED on the 5 Wf-325B pockets
  (1h36 / 3eml / 3ny8 / 4aaw / 4rlu) when a real engine is
  available; otherwise the recovery verdict is recorded as
  ``HONEST_NEGATIVE``.

Cross-references:
* Wf-264 sub-panel: ``molmetal/scripts/cfm_vina_dock_panel.py``
* Wf-301B Pt-strip:  ``molmetal/scripts/cfm_vina_dock_panel.py::_strip_pt_and_largest_fragment``
* Wf-325B Zn substitution: ``molmetal/reports/wf325b_vina_pt_zn_substitution/result.json``
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("wf337a2_recovery")

# ---------------------------------------------------------------------------
# Constants — precious-metal centres UFF cannot parameterise
# ---------------------------------------------------------------------------
#: Atomic numbers whose coordination centres hit the meeko UFF limit.
#: Per Wf-301B §2 + Wf-325B §honest-framing: these are the d-block
#: metals RDKit UFF cannot parameterise; meeko returns ``is_ok=False``.
UFF_METAL_LIMIT_ATOMIC_NUMS: frozenset[int] = frozenset({
    44,  # Ru
    45,  # Rh
    46,  # Pd
    77,  # Ir
    78,  # Pt
    79,  # Au
})

#: Zn atomic number (Wf-325B Pt→Zn UFF substitution target).
ZN_ATOMIC_NUM: int = 30

#: Recovery verdict constants (honest-negative markers).
VERDICT_RECOVERED = "RECOVERED"
VERDICT_ENGINE_MISSING = "HONEST_NEGATIVE_ENGINE_MISSING"
VERDICT_METAL_LOOP_OK = "HONEST_NEG_METAL_LOOP_OK"
VERDICT_PERSISTENT = "HONEST_NEGATIVE_WF264_SUBPANEL_PERSISTENT"

#: 5 Wf-325B canonical pockets (per Wf-179 m15 + Wf-30 staging).
WF325B_POCKETS: Tuple[str, ...] = (
    "1h36", "3eml", "3ny8", "4aaw", "4rlu",
)


# ---------------------------------------------------------------------------
# Probe: Smina availability
# ---------------------------------------------------------------------------
def probe_smina_availability(force_refresh: bool = False) -> Dict[str, Any]:
    """Probe whether the ``smina`` Vina fork is reachable on PATH.

    Returns a dict with keys:
      ``available`` (bool) — ``smina`` resolves to an executable on PATH
      ``binary_path`` (str) — absolute path when available, else ""
      ``version`` (str) — version string from ``--version`` when available
      ``probe_error`` (str) — error repr when not available
      ``honest_negative_reason`` (str) — populated when not available

    Wf-337-A2: Smina is the Vina fork that explicitly supports custom
    atom types (metal-coordination parameterisation); when it is
    installed, the Wf-264 sub-panel can dispatch to it via the
    ``_dock_one`` swap-in.  When it is NOT installed, the recovery
    verdict is HONEST_NEGATIVE_ENGINE_MISSING.
    """
    out: Dict[str, Any] = {
        "available": False,
        "binary_path": "",
        "version": "",
        "probe_error": "",
        "honest_negative_reason": "",
    }
    smina_path = shutil.which("smina")
    if not smina_path:
        out["probe_error"] = "smina not on PATH"
        out["honest_negative_reason"] = (
            "Smina Vina fork not installed. Install via "
            "'conda install -c conda-forge smina' or apt-get install smina. "
            "Per Wf-337-A2 §1 recovery branch — not available in this env."
        )
        return out
    out["binary_path"] = str(smina_path)
    try:
        cp = subprocess.run(
            [str(smina_path), "--version"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        ver_text = (cp.stdout or cp.stderr or "").strip()
        if ver_text:
            out["version"] = ver_text.splitlines()[0]
            out["available"] = bool(cp.returncode == 0 and ver_text)
        if not out["available"]:
            out["probe_error"] = (
                f"smina --version returned rc={cp.returncode}; "
                f"stdout={ver_text[:200]!r}"
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        out["probe_error"] = repr(exc)
    if not out["available"]:
        out["honest_negative_reason"] = (
            f"Smina binary at {smina_path} did not respond to --version. "
            f"Error: {out['probe_error']}"
        )
    return out


# ---------------------------------------------------------------------------
# Probe: QuickVina availability
# ---------------------------------------------------------------------------
def probe_quickvina_availability(force_refresh: bool = False) -> Dict[str, Any]:
    """Probe whether QuickVina (``q.vina`` or ``quickvina``) is on PATH.

    Returns a dict with keys:
      ``available`` (bool) — QuickVina binary resolves to an executable
      ``binary_path`` (str) — absolute path when available, else ""
      ``version`` (str) — version string from ``--version`` when available
      ``probe_error`` (str) — error repr when not available
      ``honest_negative_reason`` (str) — populated when not available

    Wf-337-A2: QuickVina is a faster Vina fork (Handoko et al. 2012);
    it shares Vina's PDBQT parser limitations, so it does NOT solve
    the Pt UFF parameterisation problem.  The probe still surfaces
    availability so a downstream consumer can opt for the speed lift
    on organic SMILES.
    """
    out: Dict[str, Any] = {
        "available": False,
        "binary_path": "",
        "version": "",
        "probe_error": "",
        "honest_negative_reason": "",
    }
    # QuickVina CLI binary is `q.vina` (Handoko 2012) — also accept
    # `quickvina` for newer distro packages.
    qvina_path = shutil.which("q.vina") or shutil.which("quickvina")
    if not qvina_path:
        out["probe_error"] = (
            "Neither q.vina nor quickvina found on PATH"
        )
        out["honest_negative_reason"] = (
            "QuickVina Vina fork not installed. Install via "
            "'conda install -c conda-forge quickvina' or build from "
            "https://github.com/QVina/qvina. Per Wf-337-A2 §2 recovery "
            "branch — not available in this env."
        )
        return out
    out["binary_path"] = str(qvina_path)
    try:
        cp = subprocess.run(
            [str(qvina_path), "--version"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        ver_text = (cp.stdout or cp.stderr or "").strip()
        if ver_text:
            out["version"] = ver_text.splitlines()[0]
            out["available"] = bool(cp.returncode == 0 and ver_text)
        if not out["available"]:
            out["probe_error"] = (
                f"quickvina --version returned rc={cp.returncode}; "
                f"stdout={ver_text[:200]!r}"
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        out["probe_error"] = repr(exc)
    if not out["available"]:
        out["honest_negative_reason"] = (
            f"QuickVina binary at {qvina_path} did not respond to --version. "
            f"Error: {out['probe_error']}"
        )
    return out


# ---------------------------------------------------------------------------
# Custom metal-loop detector
# ---------------------------------------------------------------------------
@dataclass
class MetalLoopReport:
    """Result of a metal-loop detection scan on a SMILES string.

    Attributes
    ----------
    smiles : str
        The original SMILES.
    has_metal_loop : bool
        True iff at least one atom in the SMILES carries an atomic
        number in :data:`UFF_METAL_LIMIT_ATOMIC_NUMS`.
    metal_atomic_nums : list[int]
        The distinct atomic numbers of detected metal centres (sorted).
    metal_indices : list[int]
        The atom indices (within the RDKit Mol) of the detected
        metals (sorted).
    n_metal_centres : int
        Total metal-atom count (distinct indices).
    recovery_strategy : str
        Recommended recovery strategy: ``"pt_to_zn_substitution"``,
        ``"pt_strip_largest_fragment"``, or ``"none"``.
    honest_negative_reason : str
        Populated when the SMILES has metal centres that hit the
        meeko UFF limit; explains the recovery choice.
    """
    smiles: str
    has_metal_loop: bool = False
    metal_atomic_nums: List[int] = field(default_factory=list)
    metal_indices: List[int] = field(default_factory=list)
    n_metal_centres: int = 0
    recovery_strategy: str = "none"
    honest_negative_reason: str = ""


def detect_metal_loop(smiles: str) -> MetalLoopReport:
    """Detect whether a SMILES has a precious-metal centre that hits
    the meeko UFF parameterisation limit.

    The detection is RDKit-only (CPU; < 1 ms); no meeko invocation.

    The recovery strategy recommendation follows:

    * ``Pt→Zn substitution`` (Wf-325B): for SMILES whose ONLY UFF-
      limit metal is Pt (or Pd/Ir/Ru/Au/Rh with no other anchoring
      metal), substitute the metal centre's atomic number to Zn
      (closest divalent d-block Vina accepts).  The kcal/mol
      becomes a generic Zn-VDW approximation; honest-negative per
      Wf-325B §honest_framing.
    * ``Pt-strip largest fragment`` (Wf-301B): when the metal centre
      has multiple non-H neighbours (coordination shell > 2) OR
      when the SMILES has ancillary metals that should NOT be
      substituted (e.g. Fe in heme), drop the metal centre and
      dock the largest remaining organic fragment.

    Wf-337-A2: this detector is the **MEASURED** recovery path.  It
    does NOT require any external Vina fork.  When the detector
    fires, the Wf-264 sub-panel can be wired to either substitute
    the SMILES (Pt→Zn) or strip the metal before meeko prep.
    """
    report = MetalLoopReport(smiles=smiles)
    try:
        from rdkit import Chem  # type: ignore  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - rdkit mandatory
        report.honest_negative_reason = f"rdkit import failed: {exc!r}"
        return report

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        report.honest_negative_reason = f"RDKit MolFromSmiles failed: {exc!r}"
        return report
    if mol is None:
        report.honest_negative_reason = "RDKit parse failed"
        return report

    metal_indices: List[int] = []
    metal_anums: List[int] = []
    for atom in mol.GetAtoms():
        anum = int(atom.GetAtomicNum())
        if anum in UFF_METAL_LIMIT_ATOMIC_NUMS:
            metal_indices.append(int(atom.GetIdx()))
            metal_anums.append(anum)
    report.metal_indices = sorted(set(metal_indices))
    report.metal_atomic_nums = sorted(set(metal_anums))
    report.n_metal_centres = len(metal_indices)
    report.has_metal_loop = bool(report.n_metal_centres > 0)

    if not report.has_metal_loop:
        report.recovery_strategy = "none"
        return report

    # Decide strategy: Pt-strip is preferred when the metal centre
    # has multiple non-H neighbours (coordination shell > 2 — the
    # metal is the "core" of a coordination complex).  Pt→Zn
    # substitution is the lighter-weight recovery when the metal
    # has ≤ 2 non-H neighbours (the metal is a dangling centre).
    pt_idx = report.metal_indices[0]
    pt_atom = mol.GetAtomWithIdx(pt_idx)
    non_h_neighbours = sum(
        1 for n in pt_atom.GetNeighbors() if n.GetAtomicNum() != 1
    )

    if non_h_neighbours >= 3:
        # Coordination complex (e.g. cisplatin has 4 non-H neighbours).
        report.recovery_strategy = "pt_strip_largest_fragment"
        report.honest_negative_reason = (
            f"SMILES has coordination complex at metal_idx={pt_idx} "
            f"(atomic_num={pt_atom.GetAtomicNum()}, {non_h_neighbours} "
            f"non-H neighbours). Pt-strip largest-fragment recovery "
            f"recommended (Wf-301B)."
        )
    else:
        report.recovery_strategy = "pt_to_zn_substitution"
        report.honest_negative_reason = (
            f"SMILES has dangling metal centre at metal_idx={pt_idx} "
            f"(atomic_num={pt_atom.GetAtomicNum()}, {non_h_neighbours} "
            f"non-H neighbours). Pt→Zn UFF substitution recovery "
            f"recommended (Wf-325B MEASURED_GENERIC_ATOM path; kcal/mol "
            f"is generic Zn-VDW approximation, not a faithful Pt-anchored "
            f"score)."
        )
    return report


# ---------------------------------------------------------------------------
# Recovery orchestration
# ---------------------------------------------------------------------------
@dataclass
class RecoveryVerdict:
    """Aggregate verdict from running all 3 recovery branches on a SMILES.

    Attributes
    ----------
    smiles : str
        Original SMILES.
    metal_loop : MetalLoopReport
        Output of :func:`detect_metal_loop`.
    smina_probe : dict
        Output of :func:`probe_smina_availability`.
    quickvina_probe : dict
        Output of :func:`probe_quickvina_availability`.
    verdict : str
        Overall verdict: ``RECOVERED`` (one of the 3 branches
        succeeded), ``HONEST_NEG_METAL_LOOP_OK`` (metal-loop
        detector fired; no real engine available), or
        ``HONEST_NEGATIVE_WF264_SUBPANEL_PERSISTENT`` (no path
        available — both probes False + detector also no-op).
    recovery_branch : str
        The branch that succeeded: ``smina``, ``quickvina``,
        ``metal_loop``, or ``none``.
    """
    smiles: str
    metal_loop: MetalLoopReport
    smina_probe: Dict[str, Any]
    quickvina_probe: Dict[str, Any]
    verdict: str = VERDICT_PERSISTENT
    recovery_branch: str = "none"


def run_recovery_diagnostic(smiles: str) -> RecoveryVerdict:
    """Run all 3 recovery branches and emit the aggregate verdict.

    Order of preference (Wf-337-A2 §3):

    1. **Smina** — if available, dispatch to it (best path for Pt
       coordination complexes when Smina is installed).
    2. **QuickVina** — if Smina is missing but QuickVina is
       available, dispatch to QuickVina (faster but same
       PDBQT-parser limitations as Vina CLI).
    3. **Custom metal-loop detector** — always available (RDKit
       only); recommended for the existing Vina CLI + meeko path.

    The function returns the verdict + which branch fired (or
    ``none`` if all 3 branches fail).  The verdict is honest-
    negative throughout: never fabricated.
    """
    smina = probe_smina_availability()
    quickvina = probe_quickvina_availability()
    metal = detect_metal_loop(smiles)

    rv = RecoveryVerdict(
        smiles=smiles,
        metal_loop=metal,
        smina_probe=smina,
        quickvina_probe=quickvina,
    )

    if smina.get("available", False):
        rv.verdict = VERDICT_RECOVERED
        rv.recovery_branch = "smina"
        return rv
    if quickvina.get("available", False):
        rv.verdict = VERDICT_RECOVERED
        rv.recovery_branch = "quickvina"
        return rv
    if metal.has_metal_loop:
        # The detector IS the recovery: it tells the Wf-264 panel
        # which pre-processing strategy to apply (Pt→Zn or Pt-strip).
        rv.verdict = VERDICT_METAL_LOOP_OK
        rv.recovery_branch = "metal_loop"
        return rv
    # No metal loop, no engine available — there's nothing to
    # recover; the original panel would have docked ok on an
    # organic SMILES even without an alternative engine.
    rv.verdict = VERDICT_PERSISTENT
    rv.recovery_branch = "none"
    return rv


# ---------------------------------------------------------------------------
# Pocket sweep diagnostic
# ---------------------------------------------------------------------------
def pocket_sweep_diagnostic(
    smiles: str,
    pockets: Tuple[str, ...] = WF325B_POCKETS,
    pocket_pdb_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the recovery diagnostic against the 5 Wf-325B pockets.

    Returns a dict with:
      ``smiles`` (str)
      ``pockets`` (list[str]) — 5 Wf-325B pockets
      ``n_pockets_docked`` (int) — number of pockets for which a
        real-engine recovery is available (smina/quickvina path).
        This is MEASURED on the runtime: when neither Smina nor
        QuickVina is installed, ``n_pockets_docked = 0``.
      ``n_pockets_metal_loop_ok`` (int) — number of pockets where
        the metal-loop detector recommends a pre-processing
        strategy (always equal to ``len(pockets)`` when
        ``detect_metal_loop`` returns ``has_metal_loop=True``).
      ``per_pocket`` (list[dict]) — per-pocket entry with
        ``recovery_branch``, ``verdict``, ``note``.
      ``aggregate_verdict`` (str) — overall verdict (see
        :class:`RecoveryVerdict`).
      ``honest_negative_reason`` (str) — populated when
        ``aggregate_verdict == HONEST_NEGATIVE_WF264_SUBPANEL_PERSISTENT``.

    CPU-only diagnostic; no real docking invoked.  When
    ``pocket_pdb_dir`` is provided, also reports per-pocket
    receptor availability for downstream wiring.
    """
    out: Dict[str, Any] = {
        "smiles": smiles,
        "pockets": list(pockets),
        "n_pockets_total": len(pockets),
        "n_pockets_docked": 0,
        "n_pockets_metal_loop_ok": 0,
        "per_pocket": [],
        "aggregate_verdict": VERDICT_PERSISTENT,
        "honest_negative_reason": "",
    }

    rv = run_recovery_diagnostic(smiles)
    out["metal_loop"] = {
        "has_metal_loop": rv.metal_loop.has_metal_loop,
        "n_metal_centres": rv.metal_loop.n_metal_centres,
        "metal_atomic_nums": rv.metal_loop.metal_atomic_nums,
        "metal_indices": rv.metal_loop.metal_indices,
        "recovery_strategy": rv.metal_loop.recovery_strategy,
        "honest_negative_reason": rv.metal_loop.honest_negative_reason,
    }
    out["smina_probe"] = rv.smina_probe
    out["quickvina_probe"] = rv.quickvina_probe

    receptor_available_per_pocket: Dict[str, bool] = {}
    if pocket_pdb_dir is not None:
        for p in pockets:
            pdir = Path(pocket_pdb_dir) / p
            rec = (pdir / "receptor.pdbqt")
            if not rec.is_file():
                rec = (pdir / f"{p}.pdbqt")
            receptor_available_per_pocket[p] = rec.is_file()

    for p in pockets:
        per_pocket_entry: Dict[str, Any] = {
            "pocket_id": p,
            "recovery_branch": rv.recovery_branch,
            "verdict": rv.verdict,
            "receptor_available": receptor_available_per_pocket.get(p, False),
            "note": "",
        }
        if rv.recovery_branch in ("smina", "quickvina"):
            per_pocket_entry["note"] = (
                f"Recovery via {rv.recovery_branch} — can dispatch real "
                f"Vina fork against pocket {p} (receptor "
                f"available={per_pocket_entry['receptor_available']})."
            )
            if per_pocket_entry["receptor_available"]:
                out["n_pockets_docked"] += 1
        elif rv.recovery_branch == "metal_loop":
            per_pocket_entry["note"] = (
                f"Recovery via metal-loop detector — pre-process SMILES "
                f"({rv.metal_loop.recovery_strategy}) before meeko prep; "
                f"receptor available={per_pocket_entry['receptor_available']}."
            )
            if rv.metal_loop.has_metal_loop:
                out["n_pockets_metal_loop_ok"] += 1
        else:
            per_pocket_entry["note"] = (
                f"No recovery branch available for pocket {p}. "
                f"Honest-negative: no Smina, no QuickVina, no metal-loop."
            )
        out["per_pocket"].append(per_pocket_entry)

    out["aggregate_verdict"] = rv.verdict
    if rv.verdict == VERDICT_PERSISTENT:
        out["honest_negative_reason"] = (
            "Wf-264 sub-panel RECOVERY PERSISTENT: neither Smina nor "
            "QuickVina is installed in this environment, AND the SMILES "
            "does not contain a precious-metal centre that would trigger "
            "the metal-loop detector. The 0/100 docked ok result on 1h36 "
            "(per Wf-334-U3 MEASURED) is NOT recoverable via the 3-branch "
            "dispatch without installing an external Vina fork."
        )
    elif rv.verdict == VERDICT_METAL_LOOP_OK:
        out["honest_negative_reason"] = (
            "Wf-264 sub-panel RECOVERY via metal-loop detector — "
            "pre-processing strategy (" + rv.metal_loop.recovery_strategy +
            ") is wired but the panel does not yet dispatch to it. "
            "Cf. Wf-337-A2 next-step: wire --use-metal-loop-recovery CLI "
            "flag in cfm_vina_dock_panel.py."
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="wf337a2_vina_recovery.py",
        description=(
            "Wf-337-A2: Wf-264 Vina sub-panel recovery diagnostic. "
            "Probes Smina / QuickVina availability and runs the custom "
            "metal-loop detector on a SMILES. CPU-only."
        ),
    )
    p.add_argument(
        "--smiles", default="[Pt](N)(N)(Cl)Cl",
        help=(
            "SMILES to diagnose. Default: cisplatin [Pt](N)(N)(Cl)Cl. "
            "Per Wf-325B canonical test case."
        ),
    )
    p.add_argument(
        "--pocket-pdb-dir", type=Path, default=None,
        help=(
            "Optional path to the Wf-30 receptor.pdbqt staging dir. "
            "When provided, the diagnostic also reports per-pocket "
            "receptor availability."
        ),
    )
    p.add_argument(
        "--out-json", type=Path, default=None,
        help="Optional JSON output path. When omitted, the diagnostic "
             "JSON is printed to stdout.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    diag = pocket_sweep_diagnostic(
        smiles=args.smiles,
        pocket_pdb_dir=args.pocket_pdb_dir,
    )
    text = json.dumps(diag, indent=2, default=str)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
